"""B0.4: Pairwise-Difference Time-Uniform EB-CS Planner (advice/009.md §7-§12).

Core random variable (009 §7): instead of estimating Q_a and Q_b separately and
comparing U_a - L_b, define directly

    Z_t^{a,b} = G_a(W_t) - G_b(W_t),   W_t ~ P(W | x) i.i.d.,
    E[ Z_t^{a,b} ] = Q_a^{pi_b} - Q_b^{pi_b} =: Delta_{a,b},

and build a time-uniform confidence sequence for Delta_{a,b}.  Because the
nested-evidence world couples the two returns (measured coupling efficiency
kappa = Var(G_a)+Var(G_b) / Var(G_a-G_b) ~ 13, 008 §9), the pair difference has
far smaller variance than the sum of individual variances, so the CS closes
much faster than arm-wise Hoeffding — this is the sample-efficiency claim.

Confidence machinery: Waudby-Smith-Ramdas-style betting martingale (Ville's
inequality), time-uniform and optional-stopping valid.  No invented closed form:
the CS is { mu : K_n(mu) < 1/alpha } with the wealth process

    K_n(mu) = prod_{t=1..n} (1 + lambda_t(mu) (X_t - mu)),
    lambda_t(mu) = clip( 4 (mu_hat_{t-1} - mu),  +/- 0.9 / max(mu, 1-mu) ),

for X_t in [0,1] (linear normalization of Z_t to its pairwise range), mu_hat
the running mean.  For the true mean mu*, E[1 + lambda_t(mu*)(X_t - mu*) | F_{t-1}]
= 1 (lambda_t is F_{t-1}-measurable), so K_n(mu*) is a non-negative martingale
with K_0 = 1 and Ville's inequality gives P( exists n: K_n(mu*) >= 1/alpha ) <=
alpha.  The CS is evaluated on a fine grid of mu and expanded by one grid step
(discretization safety); the G0 gate verifies anytime coverage empirically.

Predictable candidate-challenger sampling (009 §10): the pair (a_t, b_t) is
chosen from F_{t-1} BEFORE W_t is drawn; per-pair alpha_ab with
sum_{a<b} alpha_ab <= delta gives simultaneous validity.  Each world costs 2
rollouts (G_{candidate} + G_{challenger}); 1 rollout when the challenger is
STOP (whose value R_stop(x) is exact).  This replaces the B0.3c full-pairing
(|A| rollouts per world) and is the scaling path for larger N (009 §13).

Pairwise range (009 §9): G_a in [l_a, u_a], G_b in [l_b, u_b]  =>
  Z^{a,b} in [ l_a - u_b, u_a - l_b ],  D_ab = D_a + D_b,
with the B0.3c budget-aware action diameters D_a = min{ c_max_rem(x; a),
h - c_a } + R_max (009 §4).

Certificate (009 §8): for the candidate a_hat,
    U_{a_hat, b} <= eps  for all b in A^+(x) minus {a_hat}
  =>  Q_{a_hat} <= min_b Q_b + eps   (prob >= 1 - delta).
STOP is exact, so the STOP competitor needs only the pair (a_hat, STOP) with
Z = G_{a_hat}(W) - R_stop(x).
"""
from __future__ import annotations

import math

import numpy as np

from .rbl_cr import CRRBL, LatentWorld, z_code_b, z_decode_b


class PairCS:
    """Time-uniform betting confidence sequence for the mean of a bounded
    pair difference Z in [lo, hi], via the WSR betting martingale with
    variance-adaptive (predictable plug-in) bets.

    lambda_t(mu) = clip( kappa_t * (mu_hat_{t-1} - mu),  +/- 0.9 / max(mu,1-mu) )
    with kappa_t = 4 / (sigma^2_hat_{t-1} + eps)  (predictable running sample
    variance of X).  Validity: K_n(mu*) is a non-negative martingale for the
    true mean mu* (Ville's inequality); the kappa scaling only accelerates the
    wealth growth (variance-adaptive rate ~ sigma^2 log(1/alpha)/n), never
    affects validity (any F_{t-1}-measurable lambda within the cap is valid).
    """

    def __init__(self, lo, hi, alpha, grid=501):
        if not (hi > lo):
            raise ValueError(f"degenerate pair range [{lo}, {hi}]")
        self.lo = float(lo)
        self.hi = float(hi)
        self.alpha = float(alpha)
        self.mu_grid = np.linspace(0.0, 1.0, grid)
        self.step = self.mu_grid[1] - self.mu_grid[0]
        self.wealth = np.ones(grid)
        self.mu_hat = 0.0
        self.m2 = 0.0                    # Welford sum of squares
        self.n = 0

    def update(self, z):
        """Feed one paired observation z in [lo, hi]."""
        x = (float(z) - self.lo) / (self.hi - self.lo)
        # predictable lambda from F_{t-1} (mu_hat / variance BEFORE this obs)
        cap = 0.9 / np.maximum(self.mu_grid, 1.0 - self.mu_grid)
        var_hat = self.m2 / max(self.n - 1, 1)
        kappa = 4.0 / (var_hat + 1e-6)
        lam = np.clip(kappa * (self.mu_hat - self.mu_grid), -cap, cap)
        self.wealth *= (1.0 + lam * (x - self.mu_grid))
        np.clip(self.wealth, 0.0, 1e300, out=self.wealth)   # avoid inf artifacts
        # update running mean / variance
        self.n += 1
        delta = x - self.mu_hat
        self.mu_hat += delta / self.n
        self.m2 += delta * (x - self.mu_hat)

    def bounds(self):
        """(L, U) on the mean of Z; unsampled -> the full range."""
        if self.n == 0:
            return self.lo, self.hi
        inside = self.wealth < (1.0 / self.alpha)
        if not inside.any():
            return self.lo, self.hi
        lo_m = max(0.0, float(self.mu_grid[inside].min()) - self.step)
        hi_m = min(1.0, float(self.mu_grid[inside].max()) + self.step)
        return self.lo + (self.hi - self.lo) * lo_m, \
            self.lo + (self.hi - self.lo) * hi_m


class CRRBLEB:
    """B0.4 pairwise-difference certified planner (wraps the CRRBL rollout
    machinery: LatentWorld, _rollout, budget-aware diameters)."""

    def __init__(self, quants, mu_M, mu_F, b_h, base, pi=(0.5, 0.5),
                 delta_c=1.0, levels=(1, 2, 4, 8), seed=0, top_k_uavs=None):
        self.cr = CRRBL(quants, mu_M, mu_F, b_h, base, pi=pi,
                        delta_c=delta_c, levels=levels, seed=seed,
                        top_k_uavs=top_k_uavs)
        self.b_h = self.cr.b_h
        self.R_max = self.cr.R_max

    # ------------------------------------------------------------ per-action
    def _arms(self, x_int, h):
        """Feasible reporting actions + STOP (None) with exact cost endpoints
        [l_a, u_a] under the hard budget (009 §4)."""
        actions = self.cr.feasible_actions(x_int, h)
        R0 = self.cr.pl.r_stop(x_int)
        arms = {}
        for a in actions:
            i, r2 = a
            r_old, _ = z_decode_b(self.cr._z_digit(x_int, i))
            c_a = self.b_h + (r2 - r_old)
            dia = self.cr.bound_a(x_int, h, c_a, action=a)
            arms[a] = (c_a, c_a + dia)               # l_a, u_a
        arms[None] = (R0, R0)                        # STOP: exact
        return arms, R0

    def plan(self, x_int, h, eps, delta, max_worlds=4000, n_min=0, pen=0.0):
        """Pairwise-difference certified planning at (x, h).

        n_min: round-robin minimum samples per action before the loosest-pair
        challenger rule takes over (stabilizes early means).
        pen: caution penalty pen/sqrt(n) added to point estimates of
        under-sampled actions (keeps noisy challengers from hijacking the
        candidate selection).

        Returns (action | None, info); None == STOP.  info["certified"] True
        only when U_{a_hat,b} <= eps for every b in A^+(x) minus {a_hat}
        (STOP: pair (a_hat, STOP) with Z = G_{a_hat} - R_stop).  On budget
        exhaustion the empirical best is returned with certified=False.
        """
        cr = self.cr
        actions_feas, R0 = self._arms(x_int, h)
        actions = [a for a in actions_feas if a is not None]
        if not actions:
            return None, {"certified": True, "n_worlds": 0, "n_rollouts": 0,
                          "q_best": R0, "q_stop": R0, "best": None,
                          "cert_world": 0}
        # per-pair alpha for simultaneous validity (all possible pairs incl. STOP)
        n_arms = len(actions) + 1
        P = n_arms * (n_arms - 1) // 2
        alpha_ab = delta / P

        # pairwise ranges
        def _key(a):
            return (-1, -1) if a is None else tuple(a)

        def pair_key(a, b):
            ka, kb = _key(a), _key(b)
            return (ka, kb) if ka <= kb else (kb, ka)

        range_ab = {}
        for ia in range(len(actions)):
            for ib in range(ia + 1, len(actions)):
                a, b = actions[ia], actions[ib]
                l_a, u_a = actions_feas[a]
                l_b, u_b = actions_feas[b]
                range_ab[pair_key(a, b)] = (l_a - u_b, u_a - l_b)
        for a in actions:
            l_a, u_a = actions_feas[a]
            range_ab[pair_key(a, None)] = (R0 - u_a, R0 - l_a)

        cs = {}                       # pair_key -> PairCS
        mu_hat = {}                   # action -> running mean of G_a
        n_obs = {}                    # action -> count

        def point_est(a):
            """optimistic point estimate: running mean, else lower endpoint;
            under-sampled actions carry a caution penalty pen/sqrt(n)."""
            if a is None:
                return R0
            if a not in mu_hat:
                return actions_feas[a][0]
            return mu_hat[a] + pen / math.sqrt(max(n_obs[a], 1))

        def _is_first(a, k):
            return _key(a) == k[0]

        def upper(a, b):
            """upper CS bound on Delta_{a,b} = Q_a - Q_b (oriented!).  The pair
            CS stores the canonical G_{k0} - G_{k1}; when a is the SECOND
            element, Delta_{a,b} = -(stored variable), so its upper bound is
            the NEGATED lower bound of the stored CS."""
            k = pair_key(a, b)
            if k not in cs:
                if _is_first(a, k):
                    return range_ab[k][1]
                return -range_ab[k][0]
            L, U = cs[k].bounds()
            if _is_first(a, k):
                return U
            return -L

        def lower(a, b):
            k = pair_key(a, b)
            if k not in cs:
                if _is_first(a, k):
                    return range_ab[k][0]
                return -range_ab[k][1]
            L, U = cs[k].bounds()
            if _is_first(a, k):
                return L
            return -U

        def n_pair(a, b):
            k = pair_key(a, b)
            return cs[k].n if k in cs else 0

        def feed(a, b, z):
            """Feed z = G_a - G_b (roles = current cand/chall) into the pair
            CS, oriented to the CANONICAL key order: the CS must always see
            G_{key_first} - G_{key_second}, so flip the sign when a is the
            second element of the key (candidate change would otherwise mix
            opposite-sign samples in one CS — corrupting the bounds)."""
            k = pair_key(a, b)
            if tuple(a) != k[0]:
                z = -z
            if k not in cs:
                lo, hi = range_ab[k]
                cs[k] = PairCS(lo, hi, alpha_ab)
            cs[k].update(z)

        n_worlds = 0
        n_rollouts = 0
        last_cand = None
        for _ in range(max_worlds):
            # ---- predictable selection from F_{t-1} (009 §10):
            # candidate = smallest point estimate, challenger = loosest pair
            # upper bound vs the candidate (any predictable rule is valid;
            # the certificate uses the pairwise CS).
            best_val = min([R0] + [point_est(a) for a in actions])
            if best_val == R0:
                cand = None
            else:
                cand = min((a for a in actions if point_est(a) == best_val),
                           key=lambda a: str(a))
            if cand is None:                     # STOP is the candidate
                chall = max(actions, key=lambda b: (upper(None, b), -n_pair(None, b)))
            else:
                others = actions + [None]
                others.remove(cand)
                if n_min > 0 and any(n_obs.get(b, 0) < n_min for b in others
                                     if b is not None):
                    # round-robin: sample the least-sampled reporting action
                    chall = min((b for b in others if b is not None),
                                key=lambda b: (n_obs.get(b, 0), str(b)))
                else:
                    chall = max(others, key=lambda b: (upper(cand, b), -n_pair(cand, b)))
            # ---- draw ONE world and evaluate the pair (2 rollouts/world)
            W = LatentWorld(cr, x_int)
            n_worlds += 1
            if cand is None:
                # STOP candidate: STOP's value is exact; sample the challenger
                # reporting action vs STOP (1 rollout).
                g_chall = cr._rollout(x_int, h, chall, W)
                n_rollouts += 1
                mu_hat[chall] = (mu_hat.get(chall, 0.0) * n_obs.get(chall, 0) + g_chall) \
                    / (n_obs.get(chall, 0) + 1)
                n_obs[chall] = n_obs.get(chall, 0) + 1
                feed(chall, None, g_chall - R0)
            else:
                g_cand = cr._rollout(x_int, h, cand, W)
                n_rollouts += 1
                mu_hat[cand] = (mu_hat.get(cand, 0.0) * n_obs.get(cand, 0) + g_cand) \
                    / (n_obs.get(cand, 0) + 1)
                n_obs[cand] = n_obs.get(cand, 0) + 1
                feed(cand, None, g_cand - R0)      # (cand, STOP) pair is free
                if chall is not None:
                    g_chall = cr._rollout(x_int, h, chall, W)
                    n_rollouts += 1
                    mu_hat[chall] = (mu_hat.get(chall, 0.0) * n_obs.get(chall, 0) + g_chall) \
                        / (n_obs.get(chall, 0) + 1)
                    n_obs[chall] = n_obs.get(chall, 0) + 1
                    feed(cand, chall, g_cand - g_chall)
                    # evidence-based switch (009 §13 structure): the candidate
                    # changes only when the pairwise CS proves the challenger
                    # beats it (L_{cand,chall} > 0), never on raw-mean flips.
                    if lower(cand, chall) > 0.0:
                        cand = chall
            last_cand = cand
            # ---- certificate (009 §8)
            if cand is None:
                if all(upper(None, b) <= eps for b in actions):
                    return None, {"certified": True, "n_worlds": n_worlds,
                                  "n_rollouts": n_rollouts, "q_best": R0,
                                  "q_stop": R0, "best": None,
                                  "cert_world": n_worlds}
            else:
                others = actions + [None]
                others.remove(cand)
                if all(upper(cand, b) <= eps for b in others):
                    return cand, {"certified": True, "n_worlds": n_worlds,
                                  "n_rollouts": n_rollouts,
                                  "q_best": point_est(cand), "q_stop": R0,
                                  "best": cand, "cert_world": n_worlds}
        # budget exhausted: empirical best among OBSERVED actions (or STOP);
        # never-sampled actions must not win via the optimistic initialization.
        obs_vals = {a: mu_hat[a] for a in actions if a in mu_hat}
        best_val = min([R0] + list(obs_vals.values()))
        if best_val == R0:
            best = None
        else:
            best = min((a for a in obs_vals if obs_vals[a] == best_val),
                       key=lambda a: str(a))
        return best, {"certified": False, "n_worlds": n_worlds,
                      "n_rollouts": n_rollouts,
                      "q_best": (R0 if best is None else point_est(best)),
                      "q_stop": R0, "best": best, "cert_world": None}
