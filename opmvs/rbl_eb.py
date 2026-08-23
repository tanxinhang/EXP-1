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

from .fusion import log_sigmoid, log_one_minus_sigmoid
from .sparse import BASE_B, z_code_b
from .rbl_cr import CRRBL, LatentWorld, z_decode_b


class PairCS:
    """Time-uniform confidence sequence for the mean of a bounded pair
    difference Z in [lo, hi], normalized X = (Z - lo)/(hi - lo) in [0,1].

    mode="eb" (FORMAL certificate path, B0.4r / advice/010 §2): predictable
    plug-in empirical-Bernstein CS via the Maurer-Pontil (2009, Thm 4) per-n
    bound + the union bound (peeling) over n with delta-spending:
        r_n = sqrt( 2 V_hat_n t_n / (n-1) ) + 7 t_n / (3 (n-1)),
        t_n = log( 2 / delta_n ),  delta_n = 6 alpha / (pi^2 n^2),
        V_hat_n = sample variance of X_1..X_n,   mu_hat_n = sample mean.
    (011 §B0.4s: the closed form sqrt(2V log(2/delta)/n) + 7 log(2/delta)/
    (3(n-1)) is Maurer-Pontil Theorem 4; Theorem 6 is the uniform extension
    over function classes.  The code uses (n-1) in the sqrt denominator —
    MORE conservative than the theorem's (n) — so this is a citation-number
    correction, not a validity change.)
    P( exists n >= 1 : |mu_hat_n - mu| >= r_n ) <= sum_n delta_n = alpha,
    since the per-n MP bound holds for the iid sample and the union bound
    makes it time-uniform (optional-stopping valid).  This is a CONTINUOUS
    interval (no grid inversion) and is the theorem-backed certificate.

    mode="betting" (EXPERIMENTAL ablation, not the formal certificate): the
    WSR-style variance-adaptive betting martingale (Ville's inequality) with
    the wealth evaluated on a finite grid and expanded by one step — an
    empirical tightening that does NOT carry a strict continuum guarantee
    (010 §2 keeps it as the tighter ablation).

    Hard invariant (010 §1 R2): every fed z must lie in [lo, hi] (the
    canonical sample orientation must match the canonical support range).
    """

    def __init__(self, lo, hi, alpha, mode="eb", grid=501):
        if not (hi > lo):
            raise ValueError(f"degenerate pair range [{lo}, {hi}]")
        self.lo = float(lo)
        self.hi = float(hi)
        self.alpha = float(alpha)
        self.mode = mode
        self.mu_hat = 0.0
        self.m2 = 0.0                    # Welford sum of squares
        self.n = 0
        if mode == "betting":
            self.mu_grid = np.linspace(0.0, 1.0, grid)
            self.step = self.mu_grid[1] - self.mu_grid[0]
            self.wealth = np.ones(grid)
        elif mode != "eb":
            raise ValueError(f"unknown PairCS mode {mode}")

    def update(self, z):
        """Feed one paired observation z; MUST lie in [lo, hi] (R2 hard
        invariant — catches canonical sample/support orientation bugs)."""
        x = (float(z) - self.lo) / (self.hi - self.lo)
        assert -1e-9 <= x <= 1.0 + 1e-9, \
            f"z={z} outside pair range [{self.lo}, {self.hi}] (orientation bug)"
        if self.mode == "eb":
            # predictable plug-in EB: accumulate the sample variance about the
            # PRE-update mean (Welford) — the per-n MP bound uses the final
            # sample variance, which Welford tracks incrementally.
            self.n += 1
            delta = x - self.mu_hat
            self.mu_hat += delta / self.n
            self.m2 += delta * (x - self.mu_hat)
        else:                             # betting (experimental)
            cap = 0.9 / np.maximum(self.mu_grid, 1.0 - self.mu_grid)
            var_hat = self.m2 / max(self.n - 1, 1)
            kappa = 4.0 / (var_hat + 1e-6)
            lam = np.clip(kappa * (self.mu_hat - self.mu_grid), -cap, cap)
            self.wealth *= (1.0 + lam * (x - self.mu_grid))
            np.clip(self.wealth, 0.0, 1e300, out=self.wealth)
            self.n += 1
            delta = x - self.mu_hat
            self.mu_hat += delta / self.n
            self.m2 += delta * (x - self.mu_hat)

    def bounds(self):
        """(L, U) on the mean of Z; unsampled -> the full range."""
        if self.n == 0:
            return self.lo, self.hi
        if self.mode == "eb":
            n = self.n
            V = self.m2 / max(n - 1, 1)
            dn = 6.0 * self.alpha / (math.pi * math.pi * n * n)
            t = math.log(2.0 / dn)
            r = math.sqrt(2.0 * V * t / max(n - 1, 1)) + 7.0 * t / (3.0 * max(n - 1, 1))
            lo_m = max(0.0, self.mu_hat - r)
            hi_m = min(1.0, self.mu_hat + r)
            return self.lo + (self.hi - self.lo) * lo_m, \
                self.lo + (self.hi - self.lo) * hi_m
        inside = self.wealth < (1.0 / self.alpha)
        if not inside.any():
            return self.lo, self.hi
        lo_m = max(0.0, float(self.mu_grid[inside].min()) - self.step)
        hi_m = min(1.0, float(self.mu_grid[inside].max()) + self.step)
        return self.lo + (self.hi - self.lo) * lo_m, \
            self.lo + (self.hi - self.lo) * hi_m


class PairHoeffding:
    """Pairwise HOEFFDING CS on the pair difference (ablation cell H1 of
    010 §4 — isolates the CS contribution from the pair-statistic one):
        r_ab(n) = (u - l) * sqrt( log(2 / delta_ab(n)) / (2 n) ),
        delta_ab(n) = 6 alpha / (pi^2 n^2)   (sum_n = alpha, time-uniform).
    Uses the FULL range (u - l) — no variance adaptation."""

    def __init__(self, lo, hi, alpha):
        self.lo = float(lo)
        self.hi = float(hi)
        self.alpha = float(alpha)
        self.n = 0
        self.s = 0.0

    def update(self, z):
        assert self.lo - 1e-9 <= z <= self.hi + 1e-9, \
            f"z={z} outside pair range [{self.lo}, {self.hi}]"
        self.n += 1
        self.s += float(z)

    def bounds(self):
        if self.n == 0:
            return self.lo, self.hi
        mu = self.s / self.n
        dn = 6.0 * self.alpha / (math.pi * math.pi * self.n * self.n)
        r = (self.hi - self.lo) * math.sqrt(math.log(2.0 / dn) / (2.0 * self.n))
        return mu - r, mu + r


class CRRBLEB:
    """B0.4 pairwise-difference certified planner (wraps the CRRBL rollout
    machinery: LatentWorld, _rollout, budget-aware diameters).

    cs_mode: "eb" (default, FORMAL: predictable plug-in empirical-Bernstein CS
    via Maurer-Pontil + peeling, continuous), "betting" (experimental tighter
    grid CS), "hoeffding" (pairwise Hoeffding — ablation cell H1 of 010 §4).
    shared: True (default) uses ONE latent world per iteration for both
    actions (nested CRN coupling); False draws independent worlds per action
    (ablation cell E0 — isolates the coupling benefit).
    """

    def __init__(self, quants, mu_M, mu_F, b_h, base, pi=(0.5, 0.5),
                 delta_c=1.0, levels=(1, 2, 4, 8), seed=0, top_k_uavs=None,
                 cs_mode="eb", shared=True):
        self.cr = CRRBL(quants, mu_M, mu_F, b_h, base, pi=pi,
                        delta_c=delta_c, levels=levels, seed=seed,
                        top_k_uavs=top_k_uavs)
        self.b_h = self.cr.b_h
        self.R_max = self.cr.R_max
        self.cs_mode = cs_mode
        self.shared = bool(shared)

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
                k = pair_key(a, b)                 # CANONICALIZE FIRST (R0)
                c0, c1 = k[0], k[1]                # G_{c0} - G_{c1}
                l_c0, u_c0 = actions_feas[c0]
                l_c1, u_c1 = actions_feas[c1]
                range_ab[k] = (l_c0 - u_c1, u_c0 - l_c1)   # support follows key
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
            if _key(a) != k[0]:
                z = -z
            if k not in cs:
                lo, hi = range_ab[k]
                if self.cs_mode == "hoeffding":
                    cs[k] = PairHoeffding(lo, hi, alpha_ab)
                else:
                    cs[k] = PairCS(lo, hi, alpha_ab, mode=self.cs_mode)
            cs[k].update(z)                # R2: asserts z in [lo, hi]

        n_worlds = 0
        n_rollouts = 0
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
            # ---- draw the world(s) and evaluate the pair (2 rollouts/world;
            # shared=False uses independent worlds per action — ablation E0)
            if cand is None:
                # STOP candidate: STOP's value is exact; sample the challenger
                # reporting action vs STOP (1 rollout).
                W = LatentWorld(cr, x_int)
                n_worlds += 1
                g_chall = cr._rollout(x_int, h, chall, W)
                n_rollouts += 1
                mu_hat[chall] = (mu_hat.get(chall, 0.0) * n_obs.get(chall, 0) + g_chall) \
                    / (n_obs.get(chall, 0) + 1)
                n_obs[chall] = n_obs.get(chall, 0) + 1
                feed(chall, None, g_chall - R0)
            else:
                if self.shared:
                    W = LatentWorld(cr, x_int)
                    n_worlds += 1
                    g_cand = cr._rollout(x_int, h, cand, W)
                    g_chall = (cr._rollout(x_int, h, chall, W)
                               if chall is not None else None)
                else:
                    W1 = LatentWorld(cr, x_int)
                    n_worlds += 1
                    g_cand = cr._rollout(x_int, h, cand, W1)
                    if chall is not None:
                        W2 = LatentWorld(cr, x_int)
                        n_worlds += 1
                        g_chall = cr._rollout(x_int, h, chall, W2)
                    else:
                        g_chall = None
                n_rollouts += 1
                mu_hat[cand] = (mu_hat.get(cand, 0.0) * n_obs.get(cand, 0) + g_cand) \
                    / (n_obs.get(cand, 0) + 1)
                n_obs[cand] = n_obs.get(cand, 0) + 1
                feed(cand, None, g_cand - R0)      # (cand, STOP) pair is free
                if chall is not None:
                    n_rollouts += 1
                    mu_hat[chall] = (mu_hat.get(chall, 0.0) * n_obs.get(chall, 0) + g_chall) \
                        / (n_obs.get(chall, 0) + 1)
                    n_obs[chall] = n_obs.get(chall, 0) + 1
                    feed(cand, chall, g_cand - g_chall)
                    # evidence-based switch (009 §13 structure): the candidate
                    # may also be switched by the pairwise CS when it proves
                    # the challenger beats it (L_{cand,chall} > 0).  NOTE: the
                    # top-of-loop raw-mean selection can still override this
                    # (010 §7) — a persistent incumbent comes with B0.4a.
                    if lower(cand, chall) > 0.0:
                        cand = chall
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


class VoIBase:
    """One-step conditional-VoI base policy (011 §7):

        Q_a^(1) = c_a + E[ R_stop(X') | x, a ],   Q_stop^(1) = R_stop(x),
        a_VoI(x, h) = argmin_{a in A_h^+(x)} Q_a^(1).

    Equivalently VoI_a(x) = R_stop(x) - E[R_stop(X')|x,a] - c_a and acquisition
    happens only when VoI_a > 0.  Objective-consistent: it uses the current
    posterior, the header/setup cost b_h + (r'-r), and the message resolution
    automatically; one layer of conditional expectation, no learning.

    act(planner, x_int, om, h=None): first arg is the SparsePlanner (same
    convention as SNRDirectBase, so it drops into CRRBL._rollout and
    base_policy_value unchanged).  h given -> only budget-feasible actions are
    considered (CPI execution); h=None -> all legal actions (rollout base
    path; the rollout's budget check enforces feasibility, and the exact
    base-policy recursion uses the same rule, keeping MC and exact consistent).
    """

    def __init__(self, b_h):
        self.b_h = float(b_h)

    def q1(self, pl, x_int, a):
        """Exact one-step lookahead Q_a^(1); None (STOP) -> R_stop(x)."""
        if a is None:
            return pl.r_stop(x_int)
        i, r2 = a
        zi = (x_int // pl.powers[i]) % BASE_B
        r_old, _ = z_decode_b(zi)
        c = self.b_h + (r2 - r_old)
        om = pl.omega(x_int)
        lp = float(log_sigmoid(om))
        lq = float(log_one_minus_sigmoid(om))
        cells = next(cells for (r2b, _ct, _qb, cells) in pl._tpl[i][zi] if r2b == r2)
        E = 0.0
        for (m2, lp0c, lp1c) in cells:
            a_ = lp + lp1c
            b_ = lq + lp0c
            m_ = a_ if a_ >= b_ else b_
            w = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
            z2 = z_code_b(r2, m2)
            cx = x_int + (z2 - zi) * pl.powers[i]
            E += w * pl.r_stop(cx)
        return c + E

    def act(self, pl, x_int, om, h=None):
        q_h = int(np.floor(h / pl.delta_c)) if h is not None else 10 ** 9
        R0 = pl.r_stop(x_int)
        best_q = R0
        best = None
        rem = int(x_int)
        zs = []
        for _ in range(pl.N):
            zs.append(rem % BASE_B)
            rem //= BASE_B
        for i in range(pl.N):
            zi = zs[i]
            for (r2, c_true, qb, _cells) in pl._tpl[i][zi]:
                if qb > q_h:
                    continue
                Q1 = self.q1(pl, x_int, (i, r2))
                if Q1 < best_q:
                    best_q = Q1
                    best = (i, r2)
        return best


class CPI:
    """B0.4a Certified Policy Improvement (011 §3-§6, §9).

    Base by default; override only with certified evidence of improvement:
        a_inc^(0) = a_b = pi_b(x, h),
        a_inc -> c  only when  U_{c, a_inc} < 0   (Q_c^{pi_b} < Q_{a_inc}^{pi_b}
        certified by the pairwise CS; STOP's value R_stop(x) is exact).
    The executed action is the final incumbent.  On the confidence event the
    incumbent chain is strictly decreasing in Q^{pi_b}, so the one-step
    deviation argument for the finite acyclic evidence DAG gives
        V^{pi_CPI}(x) <= V^{pi_b}(x),  hence  J^{pi_CPI} <= J^{pi_b}
    with prob >= 1 - delta_episode (011 §4).

    Episode-level delta (011 §5): decision t spends delta_t =
    6 delta_episode / (pi^2 t^2); within a decision every possible pair
    (all C(|A|+1,2) candidate-vs-incumbent pairs) is allocated
    alpha = delta_t / P, so the union bound over pairs gives
    P(any pair CS fails in decision t) <= delta_t, and
    sum_t delta_t = delta_episode  =>  P(all executed overrides valid) >=
    1 - delta_episode.

    Sampling: 2 rollouts/world (the incumbent anchor + the most dangerous
    challenger; 1 rollout when one of them is STOP).
    """

    def __init__(self, quants, mu_M, mu_F, b_h, base, pi=(0.5, 0.5),
                 delta_c=1.0, levels=(1, 2, 4, 8), seed=0, top_k_uavs=None,
                 cs_mode="eb"):
        self.cr = CRRBL(quants, mu_M, mu_F, b_h, base, pi=pi,
                        delta_c=delta_c, levels=levels, seed=seed,
                        top_k_uavs=top_k_uavs)
        self.b_h = self.cr.b_h
        self.R_max = self.cr.R_max
        self.base = base
        self.cs_mode = cs_mode

    def _arms(self, x_int, h):
        return CRRBLEB._arms(self, x_int, h)

    def decide(self, x_int, h, delta_t, max_worlds=2000, seed=None):
        """One acquisition decision at (x, h) spending confidence budget
        delta_t.  Returns (a_exec, info):
            a_exec = a_b unless a challenger is CERTIFIED better
                     (U_{c,a_inc} < 0), in which case the incumbent moves.
            info: override (a_exec != a_b), a_b, n_worlds, n_rollouts,
                  n_switches.
        """
        cr = self.cr
        om = cr.pl.omega(x_int)
        a_b = self.base.act(cr.pl, x_int, om, h=h)
        actions_feas, R0 = self._arms(x_int, h)
        actions = [a for a in actions_feas if a is not None]
        if not actions:
            return a_b, {"override": False, "a_b": a_b, "n_worlds": 0,
                         "n_rollouts": 0, "n_switches": 0}
        n_arms = len(actions) + 1
        P = n_arms * (n_arms - 1) // 2
        alpha_pair = delta_t / P

        def _key(a):
            return (-1, -1) if a is None else tuple(a)

        def pair_key(a, b):
            ka, kb = _key(a), _key(b)
            return (ka, kb) if ka <= kb else (kb, ka)

        range_ab = {}
        for ia in range(len(actions)):
            for ib in range(ia + 1, len(actions)):
                a, b = actions[ia], actions[ib]
                k = pair_key(a, b)
                c0, c1 = k[0], k[1]
                l_c0, u_c0 = actions_feas[c0]
                l_c1, u_c1 = actions_feas[c1]
                range_ab[k] = (l_c0 - u_c1, u_c0 - l_c1)
        for a in actions:
            l_a, u_a = actions_feas[a]
            range_ab[pair_key(a, None)] = (R0 - u_a, R0 - l_a)

        cs = {}
        mu_hat = {}
        n_obs = {}

        def _is_first(a, k):
            return _key(a) == k[0]

        def upper(a, b):
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
            k = pair_key(a, b)
            if _key(a) != k[0]:
                z = -z
            if k not in cs:
                lo, hi = range_ab[k]
                cs[k] = PairCS(lo, hi, alpha_pair, mode=self.cs_mode)
                cs[k].tag = k
            cs[k].update(z)

        a_inc = a_b
        n_switches = 0
        n_worlds = 0
        n_rollouts = 0
        arms_all = actions + [None]
        alive = [c for c in arms_all if c != a_inc]      # challenger set

        def point_est(c):
            if c is None:
                return R0
            return mu_hat[c] if c in mu_hat else actions_feas[c][0]

        def sample_pair(chall, W):
            """one paired sample (chall, a_inc); STOP handled exactly."""
            nonlocal n_worlds, n_rollouts
            n_worlds += 1
            if a_inc is None:
                g_chall = cr._rollout(x_int, h, chall, W)
                n_rollouts += 1
                mu_hat[chall] = (mu_hat.get(chall, 0.0) * n_obs.get(chall, 0) + g_chall) \
                    / (n_obs.get(chall, 0) + 1)
                n_obs[chall] = n_obs.get(chall, 0) + 1
                feed(chall, None, g_chall - R0)
            else:
                g_inc = cr._rollout(x_int, h, a_inc, W)
                n_rollouts += 1
                mu_hat[a_inc] = (mu_hat.get(a_inc, 0.0) * n_obs.get(a_inc, 0) + g_inc) \
                    / (n_obs.get(a_inc, 0) + 1)
                n_obs[a_inc] = n_obs.get(a_inc, 0) + 1
                if chall is None:
                    feed(None, a_inc, R0 - g_inc)        # STOP value EXACT
                else:
                    g_chall = cr._rollout(x_int, h, chall, W)
                    n_rollouts += 1
                    mu_hat[chall] = (mu_hat.get(chall, 0.0) * n_obs.get(chall, 0) + g_chall) \
                        / (n_obs.get(chall, 0) + 1)
                    n_obs[chall] = n_obs.get(chall, 0) + 1
                    feed(chall, a_inc, g_chall - g_inc)

        # phase 1: round-robin initialization (n_init per challenger) so the
        # point estimates are informative before focusing.
        n_init = min(40, max(1, max_worlds // (4 * max(len(alive), 1))))
        for c in list(alive):
            for _ in range(n_init):
                if n_worlds >= max_worlds:
                    break
                W = LatentWorld(cr, x_int)
                sample_pair(c, W)
                if upper(c, a_inc) < 0.0:
                    a_inc = c
                    n_switches += 1
                    alive = [c2 for c2 in arms_all if c2 != a_inc]
                    break
                elif lower(c, a_inc) > 0.0:
                    alive.remove(c)
                    break
        # phase 2: focus on the best-looking challenger until resolved / budget
        while alive and n_worlds < max_worlds:
            chall = min(alive, key=lambda c: (point_est(c), str(c)))
            W = LatentWorld(cr, x_int)
            sample_pair(chall, W)
            if upper(chall, a_inc) < 0.0:
                a_inc = chall
                n_switches += 1
                alive = [c for c in arms_all if c != a_inc]
            elif lower(chall, a_inc) > 0.0:
                alive.remove(chall)
        return a_inc, {"override": (a_inc != a_b), "a_b": a_b,
                       "n_worlds": n_worlds, "n_rollouts": n_rollouts,
                       "n_switches": n_switches}
