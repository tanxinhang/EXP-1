"""CR-RBL: Confidence-Certified Rollout Resource-Bounded Lookahead.

B0.3a (advice/007.md §1-§7, §10-§12): credibility patch on top of the B0.3
first version.  Four P0 repairs:

  P0-A  true cross-action CRN: one latent world
            W_m = ( H_m, M_1^(8), ..., M_N^(8) ) | x
        is sampled per Monte-Carlo iteration and EVERY candidate action is
        evaluated on the SAME world (paired returns G_a(W_m), a in A).  The
        world includes the latent hypothesis H_m: sample H from the posterior
        at x, then draw each UAV's level-8 cell H-conditionally — the old
        per-UAV marginal product breaks the H-induced cross-UAV correlation
        and biases rollouts (verified ~ +12 bits on the N=4 exact oracle).
        The old implementation re-sampled a latent per rollout_return call,
        so W_a^(m) != W_b^(m) and the claimed "nested-evidence CRN" never
        actually coupled the actions (007.md §1).

  P0-B  the certificate's competitor set is A^+(x) = A(x) ∪ {STOP}, where
        STOP has the EXACT value R_stop(x) = min{ C01 p, C10 (1-p) }:
            reporting a_hat certified iff   U_a_hat <= min{ R_stop,
                                                    min_{b != a_hat} L_b } + eps,
            STOP         certified iff   R_stop <= min_b L_b + eps.
        (007.md §2.)

  P0-C/D  exact-oracle helpers: exact_qa_pi_b gives Q_a^{pi_b} =
        c_a + E[J^{pi_b}(x')]  (the value the rollouts actually estimate),
        and base_policy_value gives J^{pi_b} — used by the G0/G1/G3 gates
        instead of the Q_a^star / base-policy-value-as-STOP mis-calibration
        of the first version (007.md §3).

  P0-E  hard-budget semantics: plan(x, h) treats h as the REMAINING
        communication budget; the receding loop must pass h_t = H - C_t and
        guarantee C_T <= H pathwise (007.md §4; enforced by the conservative
        lattice q_a = ceil(c_a / delta_c) <= floor(h / delta_c)).

B0.3b (007.md §11): the module exposes the primitives needed by regression
invariants T15-T20 (LatentWorld nested projection, paired rollout_returns,
anytime delta-spending, certificate condition with STOP).

B0.3c (advice/008.md §3-§4): the Hoeffding diameter is now ACTION-SPECIFIC and
budget-aware,  D_a(x,h) = min{ c_max_rem(x), h } + R_max - c_a  (tight range
0 <= G_a - c_a <= min{ c_max_rem(x), h - c_a } + R_max), instead of the loose
B(x) = c_max_rem(x) + R_max — this is pure baseline fairness (the certificate
gets tighter without any new statistics).  MarginalWorld is kept as the B0.3-era
marginal-product world for the bias-vs-variance ablation (008.md §3).

The certificate remains relative to the base policy pi_b:
    P( Q_hat_a^{pi_b} <= min_{a in A^+(x)} Q_a^{pi_b} + eps ) >= 1 - delta,
NOT relative to V* (CR-RBL+ / Bellman sandwich is a later stage, 007.md §8-§9;
per 008.md §13, Bellman sandwich moves AFTER the B0.6 matched-QoS gate).
"""
from __future__ import annotations

import math

import numpy as np

from .fusion import log_sigmoid, log_one_minus_sigmoid
from .sparse import BASE_B, SparsePlanner, z_code_b, z_decode_b


def max_stop_risk(C01, C10):
    """R_max = max_p min(C01 p, C10 (1-p)) = C01*C10/(C01+C10)."""
    return C01 * C10 / (C01 + C10)


class SNRDirectBase:
    """Base policy pi_b: |Omega| >= eta_b -> STOP; else refine the strongest
    (by sensing SNR) UAV not at r_max with a direct full-precision packet."""

    def __init__(self, quants, gamma_db, b_h, eta_b, levels=(1, 2, 4, 8)):
        self.quants = quants
        self.order = list(np.argsort(-np.asarray(gamma_db, float)))
        self.b_h = float(b_h)
        self.eta_b = float(eta_b)
        self.levels = levels
        self.r_max = max(levels)

    def act(self, planner, x_int, om, h=None):
        """pi_b(x, om[, h]): STOP if |Omega| >= eta_b, else refine the
        strongest (by sensing SNR) UAV not at r_max.  h is accepted for the
        unified budget-aware base interface (012 §5) but ignored — the
        rollout's budget check enforces feasibility."""
        if abs(om) >= self.eta_b:
            return None
        rem = int(x_int)
        zs = []
        for _ in range(len(self.quants)):
            zs.append(rem % BASE_B)
            rem //= BASE_B
        for i in self.order:                       # digit at the CORRECT index
            r, _ = z_decode_b(zs[i])
            if r < self.r_max:
                return (i, self.r_max)
        return None


class LatentWorld:
    """One latent evidence realization W = ( H_m, M_1^(8), ..., M_N^(8) ) | x.

    The latent HYPOTHESIS H_m is sampled first from the posterior at x, then
    every UAV's level-8 cell is sampled from its H-conditional law given its
    current cell (UAVs are conditionally independent given H).  This reproduces
    the TRUE joint law p(W|x) — the UAVs are correlated through H, so drawing
    each cell from its marginal would break the joint coupling (007.md §1
    writes W_m := ( H_m, M_{1,m}^{(8)}, ..., M_{N,m}^{(8)} ) | x).

    Every message M_i^(r) = proj_r( M_i^(8) ) = cells[i] >> (r_max - r) is a
    deterministic projection, so all candidate actions evaluated on this world
    share the same latent evidence (true paired CRN).
    """

    __slots__ = ("cr", "x_int", "om", "h", "cells")

    def __init__(self, cr, x_int):
        self.cr = cr
        self.x_int = int(x_int)
        self.om = cr.pl.omega(self.x_int)
        p = 1.0 / (1.0 + np.exp(-self.om))
        self.h = int(cr.rng.random() < p)          # latent hypothesis
        self.cells = {}
        for i in range(cr.N):
            self.cells[i] = cr._sample_latent_h(i, cr._z_digit(self.x_int, i),
                                                self.h)

    def msg(self, i, r2):
        """Level-r2 message of UAV i in this world (nested projection)."""
        return self.cells[i] >> (self.cr.r_max - r2)


class MarginalWorld:
    """B0.3-era (pre-B0.3a) world: each UAV's level-8 cell drawn independently
    from its per-UAV MARGINAL given x (the product law prod_i P(M_i^(8)|x)).

    This is NOT a valid joint law (UAVs are correlated through H), so rollouts
    on it are biased — it is kept ONLY as the ablation cell "marginal-product x
    independent" of advice/008.md §3 (separating bias correction from paired
    variance reduction)."""

    __slots__ = ("cr", "x_int", "om", "cells")

    def __init__(self, cr, x_int):
        self.cr = cr
        self.x_int = int(x_int)
        self.om = cr.pl.omega(self.x_int)
        self.cells = {}
        for i in range(cr.N):
            self.cells[i] = cr._sample_latent(i, cr._z_digit(self.x_int, i),
                                              self.om)

    def msg(self, i, r2):
        """Level-r2 message of UAV i in this world (nested projection)."""
        return self.cells[i] >> (self.cr.r_max - r2)


class CRRBL:
    """Confidence-certified rollout RBL (B0.3c: budget-aware Hoeffding range)."""

    def __init__(self, quants, mu_M, mu_F, b_h, base, pi=(0.5, 0.5),
                 delta_c=1.0, levels=(1, 2, 4, 8), seed=0, top_k_uavs=None):
        self.quants = quants
        self.N = len(quants)
        self.mu_M = float(mu_M)
        self.mu_F = float(mu_F)
        self.b_h = float(b_h)
        self.base = base
        self.C01 = mu_M / pi[1]
        self.C10 = mu_F / pi[0]
        self.R_max = max_stop_risk(self.C01, self.C10)
        self.levels = levels
        self.r_max = max(levels)
        self.delta_c = float(delta_c)
        self.pl = SparsePlanner(quants, mu_M, mu_F, b_h=b_h, cross_level=True,
                                levels=levels, delta_c=delta_c)
        self.powers = self.pl.powers
        self.rng = np.random.default_rng(seed)
        self._delta = 0.05
        self._uavs = list(range(self.N)) if top_k_uavs is None else list(top_k_uavs)

    # ------------------------------------------------------------- bounds
    def c_max_rem(self, x_int):
        rem = int(x_int)
        total = 0.0
        for i in range(self.N):
            z = rem % BASE_B
            rem //= BASE_B
            r, _ = z_decode_b(z)
            k = sum(1 for r2 in self.levels if r2 > r)     # max txns remaining
            total += k * self.b_h + (self.r_max - r)
        return total

    def c_max_rem_after(self, x_int, i, r2):
        """max future communication cost after refining UAV i to level r2
        (cell-independent: c_max_rem depends only on the levels)."""
        rem = int(x_int)
        total = 0.0
        for j in range(self.N):
            z = rem % BASE_B
            rem //= BASE_B
            r, _ = z_decode_b(z)
            if j == i:
                r = r2
            k = sum(1 for r3 in self.levels if r3 > r)
            total += k * self.b_h + (self.r_max - r)
        return total

    def bound(self, x_int):
        """Loose (budget-free) Hoeffding diameter: G_a <= c_max_rem(x) + R_max
        (kept for backward compatibility / reference)."""
        return self.c_max_rem(x_int) + self.R_max

    def bound_a(self, x_int, h, c_a, action=None):
        """Action-specific Hoeffding diameter under the HARD budget h (008 §4,
        proof wording corrected per 009 §4).

        The total radio cost of any rollout from x satisfies
            C_T <= min{ c_max_rem(x), h },
        G_a = C_T + R_T with G_a >= c_a and R_T <= R_max, hence
            G_a in [ c_a,  min{ c_max_rem(x), h } + R_max ]
        and the deterministic diameter is
            D_a(x, h) = min{ c_max_rem(x), h } + R_max - c_a
        (NOT the intermediate statement 0 <= G_a - c_a <=
        min{c_max_rem(x), h-c_a} + R_max, which is not generally equivalent).

        Free tightening (009 §4): when the first action (i, r2) is known, the
        continuation after it is bounded by min{ c_max_rem(x; a), h - c_a } +
        R_max, giving D_a = min{ c_max_rem(x; a), h - c_a } + R_max  <=  the
        general form (c_max_rem(x; a) <= c_max_rem(x) - c_a).
        """
        if action is not None:
            i, r2 = action
            return min(self.c_max_rem_after(x_int, i, r2),
                       float(h) - float(c_a)) + self.R_max
        return min(self.c_max_rem(x_int), float(h)) + self.R_max - float(c_a)

    # ------------------------------------------------------------ sampling
    def _z_digit(self, x_int, i):
        return int(x_int // self.powers[i]) % BASE_B

    def _sample_latent(self, i, zi, om):
        """Sample the latent level-8 CELL INDEX for UAV i from its MARGINAL
        given the current cell and posterior odds om (per-UAV law; kept for
        reference).  The paired world uses _sample_latent_h (H-conditional)."""
        q = self.quants[i]
        r, m = z_decode_b(zi)
        if r >= self.r_max:
            return int(m)                           # already at finest level
        desc = q.desc_cells(r, m, self.r_max)
        p = 1.0 / (1.0 + np.exp(-om))
        lp = float(np.log(p))
        lq = float(np.log1p(-p))
        lp1r = 0.0 if r == 0 else q.logP1[r][m]
        lp0r = 0.0 if r == 0 else q.logP0[r][m]
        a_ = lp + (q.logP1[self.r_max][desc] - lp1r)
        b_ = lq + (q.logP0[self.r_max][desc] - lp0r)
        m_ = np.maximum(a_, b_)
        w = np.exp(m_ + np.log1p(np.exp(-np.abs(a_ - b_))))
        w /= w.sum()
        return int(self.rng.choice(desc, p=w))

    def _sample_latent_h(self, i, zi, h):
        """Sample the latent level-8 cell of UAV i from p( . | H=h, current
        cell ).  UAVs drawn with a common H are jointly consistent (the true
        law p(W|x) = sum_h p(h|x) prod_i p(W_i | h, x))."""
        q = self.quants[i]
        r, m = z_decode_b(zi)
        if r >= self.r_max:
            return int(m)                           # already at finest level
        desc = q.desc_cells(r, m, self.r_max)
        lpr = (0.0 if r == 0 else q.logP1[r][m]) if h else (0.0 if r == 0 else q.logP0[r][m])
        lp = (q.logP1[self.r_max][desc] - lpr) if h else (q.logP0[self.r_max][desc] - lpr)
        w = np.exp(lp - lp.max())
        w /= w.sum()
        return int(self.rng.choice(desc, p=w))

    def _apply(self, x_int, om, i, r2, latents):
        z = latents[i]                                     # latent level-8 cell
        m2 = z >> (self.r_max - r2)                        # nested projection
        zi = self._z_digit(x_int, i)
        z2 = z_code_b(r2, m2)
        x2 = x_int + (z2 - zi) * self.powers[i]
        om2 = om + self.pl._llr_i[i][z2] - self.pl._llr_i[i][zi]
        return x2, om2

    # -------------------------------------------------------- paired rollouts
    def _rollout(self, x_int, h, action, world):
        """One rollout on a GIVEN world: take `action` (or None = skip), then
        follow pi_b under the remaining budget h.  Returns G = radio cost +
        R_stop(X_T).  All latents come from `world` (paired CRN)."""
        latents = world.cells
        om = self.pl.omega(x_int)
        h_rem = float(h)
        cost = 0.0
        x2, om2 = x_int, om
        if action is not None:
            i, r2 = action
            r_old, _ = z_decode_b(self._z_digit(x_int, i))
            c = self.b_h + (r2 - r_old)
            x2, om2 = self._apply(x_int, om, i, r2, latents)
            cost += c
            h_rem -= c
        # follow the base policy (respecting the remaining budget).  B0.4a-r
        # (012 §5): the base policy is defined on the augmented state (x, h) —
        # base.act receives the REMAINING budget so pi_b(x, h_rem) is the same
        # object the CPI anchor and the exact oracle use.
        for _ in range(4 * self.N + 2):
            a = self.base.act(self.pl, x2, om2, h=h_rem)
            if a is None:
                break
            i, r2 = a
            r_old, _ = z_decode_b(self._z_digit(x2, i))
            c = self.b_h + (r2 - r_old)
            if c > h_rem:
                break                                   # budget exhausted
            x2, om2 = self._apply(x2, om2, i, r2, latents)
            cost += c
            h_rem -= c
        p = 1.0 / (1.0 + np.exp(-om2))
        cost += min(self.C01 * p, self.C10 * (1.0 - p))
        return cost

    def rollout_return(self, x_int, h, action):
        """Single-action rollout on a freshly sampled world (kept for backward
        compatibility; the paired path is rollout_returns/plan)."""
        world = LatentWorld(self, x_int)
        return self._rollout(x_int, h, action, world)

    def rollout_returns(self, x_int, h, actions, world=None):
        """Evaluate ALL `actions` on ONE shared latent world (true paired CRN).
        Returns {action: G_a(W)}."""
        if world is None:
            world = LatentWorld(self, x_int)
        return {a: self._rollout(x_int, h, a, world) for a in actions}

    # ------------------------------------------------------------ planner
    def feasible_actions(self, x_int, h):
        q = int(np.floor(h / self.delta_c))
        acts = []
        rem = int(x_int)
        zs = []
        for _ in range(self.N):
            zs.append(rem % BASE_B)
            rem //= BASE_B
        for i in self._uavs:
            for (r2, _c_true, qb, _cells) in self.pl._tpl[i][zs[i]]:
                if qb <= q:
                    acts.append((i, r2))
        return acts

    def _delta_n(self, n):
        """Anytime delta spending per action (sum_n = delta / |A|)."""
        nA = max(1, len(self._actions))
        return 6.0 * self._delta / (math.pi * math.pi * nA * n * n)

    def plan(self, x_int, h, eps, delta, max_samples=4000):
        """CR-RBL planning at (x, h) with the B0.3a paired-CRN loop.

        Semantics:
          * h is the REMAINING communication budget (hard budget; the planner
            only ever returns actions whose true cost fits: q_a <= floor(h/dc)).
          * One latent world W_m is sampled per iteration; every feasible
            action is evaluated on W_m (full paired CRN), so after n worlds
            every action has n samples:  Qhat_a = (1/n) sum_m G_a(W_m).
          * Anytime Hoeffding radius with delta-spending
                delta_{a,n} = 6 delta / (pi^2 |A| n^2),
                r_{a,n} = D_a(x,h) sqrt(log(2/delta_{a,n}) / (2n)),
            where D_a(x,h) = min{c_max_rem(x), h} + R_max - c_a is the
            action-specific diameter under the hard budget (B0.3c, 008 §4),
            and the certificate's competitor set includes STOP (exact value
            R_stop(x), zero width).

        Returns (action | None, info):
          action None == STOP.  info["certified"] True only when the
          certificate fired:
            reporting a_hat:  U_a_hat <= min{ R_stop, min_{b != a_hat} L_b } + eps
            STOP:             R_stop    <= min_b L_b + eps
          If max_samples worlds are exhausted without certification, the
          empirical best of {STOP} ∪ A is returned with certified=False.
        """
        actions = self.feasible_actions(x_int, h)
        self._actions = actions
        self._delta = delta
        R0 = self.pl.r_stop(x_int)                 # STOP pseudo-action (exact)
        if not actions:
            return None, {"certified": True, "samples": 0, "n_worlds": 0,
                          "n_rollouts": 0, "q_best": R0, "q_stop": R0,
                          "width_best": 0.0, "best": None}
        nA = len(actions)
        # B0.3c (008 §4) / B0.4 prelude (009 §4): action-specific Hoeffding
        # diameter under the hard budget, tightened with the post-action max
        # future cost: D_a = min{ c_max_rem(x; a), h - c_a } + R_max.
        c_of = {}
        for a in actions:
            i, r2 = a
            r_old, _ = z_decode_b(self._z_digit(x_int, i))
            c_of[a] = self.b_h + (r2 - r_old)
        dia = {a: self.bound_a(x_int, h, c_of[a], action=a) for a in actions}
        qhat = {a: 0.0 for a in actions}
        L = {a: 0.0 for a in actions}
        U = {a: 0.0 for a in actions}

        def best_sel():
            """Empirical best over {STOP} ∪ A: returns (best, q_best)."""
            qmin = min([R0] + [qhat[a] for a in actions])
            if qmin == R0:
                return None, R0
            cands = sorted((a for a in actions if qhat[a] == qmin))
            return cands[0], qmin

        n_worlds = 0
        n_rollouts = 0
        for _ in range(max_samples):
            world = LatentWorld(self, x_int)       # P0-A: one world -> all A
            n_worlds += 1
            n = n_worlds
            for a in actions:
                rad = dia[a] * math.sqrt(math.log(2.0 / self._delta_n(n)) / (2.0 * n))
                g = self._rollout(x_int, h, a, world)
                n_rollouts += 1
                qhat[a] = qhat[a] + (g - qhat[a]) / n
                L[a] = qhat[a] - rad
                U[a] = qhat[a] + rad
            best, q_best = best_sel()
            if best is None:                       # P0-B: STOP certificate
                min_comp = min([L[a] for a in actions])
                if R0 <= min_comp + eps:
                    return None, {"certified": True, "samples": n,
                                  "n_worlds": n, "n_rollouts": n_rollouts,
                                  "q_best": R0, "q_stop": R0, "width_best": 0.0,
                                  "best": None,
                                  "cert_cond": (R0, min_comp, eps)}
            else:
                min_comp = min([R0] + [L[a] for a in actions if a != best])
                if U[best] <= min_comp + eps:
                    return best, {"certified": True, "samples": n,
                                  "n_worlds": n, "n_rollouts": n_rollouts,
                                  "q_best": q_best, "q_stop": R0,
                                  "width_best": U[best] - q_best, "best": best,
                                  "cert_cond": (U[best], min_comp, eps)}
        # budget exhausted without certification: empirical best (uncertified)
        best, q_best = best_sel()
        return best, {"certified": False, "samples": n_worlds, "n_worlds": n_worlds,
                      "n_rollouts": n_rollouts, "q_best": q_best, "q_stop": R0,
                      "width_best": 0.0, "best": best}


# ------------------------------------------------------ exact-oracle helpers
def base_policy_value(crr, x_int, h, memo=None):
    """Exact cost-to-go J^{pi_b}(x, h) of the rollout base policy, memoized
    (budget-aware: the base policy is defined on (x, h) — base.act receives
    the remaining budget h (012 §5), so pi_b(x, h) is the SAME object the
    CPI anchor and the MC rollouts use)."""
    if memo is None:
        memo = {}
    key = (int(x_int), int(np.floor(h / crr.delta_c)))
    if key in memo:
        return memo[key]
    om = crr.pl.omega(x_int)
    a = crr.base.act(crr.pl, x_int, om, h=h)
    if a is None:
        p = 1.0 / (1.0 + np.exp(-om))
        val = min(crr.C01 * p, crr.C10 * (1.0 - p))
    else:
        i, r2 = a
        r_old, _ = z_decode_b(crr._z_digit(x_int, i))
        c = crr.b_h + (r2 - r_old)
        if c > h:
            p = 1.0 / (1.0 + np.exp(-om))
            val = min(crr.C01 * p, crr.C10 * (1.0 - p))
        else:
            zi = crr._z_digit(x_int, i)
            cells = next(cells for (r2b, _ct, _qb, cells) in crr.pl._tpl[i][zi]
                         if r2b == r2)
            lp = float(log_sigmoid(om))
            lq = float(log_one_minus_sigmoid(om))
            E = 0.0
            for (m2, lp0c, lp1c) in cells:
                z2 = z_code_b(r2, m2)
                cx = x_int + (z2 - zi) * crr.powers[i]
                om_c = om + crr.pl._llr_i[i][z2] - crr.pl._llr_i[i][zi]
                a_ = lp + lp1c
                b_ = lq + lp0c
                m_ = a_ if a_ >= b_ else b_
                w = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
                E += w * base_policy_value(crr, cx, h - c, memo)
            val = c + E
    memo[key] = val
    return val


def exact_qa_pi_b(crr, x_int, h):
    """Exact Q_a^{pi_b}(x, h) = c_a + E[J^{pi_b}(x', h - c_a)] for every
    feasible reporting action.  STOP's exact value is crr.pl.r_stop(x)
    (caller adds it); this is the oracle the MC rollouts estimate."""
    om = crr.pl.omega(x_int)
    Qs = {}
    for (i, r2) in crr.feasible_actions(x_int, h):
        r_old, _ = z_decode_b(crr._z_digit(x_int, i))
        c = crr.b_h + (r2 - r_old)
        zi = crr._z_digit(x_int, i)
        cells = next(cells for (r2b, _ct, _qb, cells) in crr.pl._tpl[i][zi]
                     if r2b == r2)
        lp = float(log_sigmoid(om))
        lq = float(log_one_minus_sigmoid(om))
        E = 0.0
        for (m2, lp0c, lp1c) in cells:
            z2 = z_code_b(r2, m2)
            cx = x_int + (z2 - zi) * crr.powers[i]
            om_c = om + crr.pl._llr_i[i][z2] - crr.pl._llr_i[i][zi]
            a_ = lp + lp1c
            b_ = lq + lp0c
            m_ = a_ if a_ >= b_ else b_
            w = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
            E += w * base_policy_value(crr, cx, h - c)
        Qs[(i, r2)] = c + E
    return Qs
