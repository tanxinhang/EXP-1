"""CR-RBL: Confidence-Certified Rollout Resource-Bounded Lookahead
(advice/006.md §9-§15).

For a state x with budget h, each feasible action a in A_h(x) is evaluated by
Monte-Carlo rollout with a fixed base policy pi_b:

    Q_a^{pi_b}(x) = E[ G_a ],   G_a = c_a + future reporting cost + R_stop(X_T).

Rollout return is strictly bounded (006.md §10):

    0 <= G <= B(x) = C_max^rem(x) + R_max,
    C_max^rem(x) = sum_i [ k_i(x) * b_setup + (r_max - r_i) ],
    R_max = C01 * C10 / (C01 + C10).

Certification: anytime Hoeffding radius with delta-spending (006.md §11):
    delta_{a,n} = 6 delta / (pi^2 |A| n^2),   r_{a,n} = B(x) sqrt(log(2/d_{a,n})/2n).
LUCB-style challenger sampling (006.md §14).  Action certificate (§13):

    U_hat_a <= min_{b != hat_a} L_b + eps  =>  P( Q_hat_a <= min_a Q_a + eps ) >= 1 - delta.

Nested-evidence CRN (006.md §15): a latent level-8 cell M_i^(8) is sampled
once per UAV per rollout; every message M_i^(r) = M_i^(8) >> (8 - r), so all
candidate actions share the same latent realization (paired returns).

First-version certificate is relative to the base policy pi_b (NOT V*):
    P( Q_hat_a^{pi_b} <= min_a Q_a^{pi_b} + eps ) >= 1 - delta.
"""
from __future__ import annotations

import math

import numpy as np

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

    def act(self, planner, x_int, om):
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


class CRRBL:
    """Confidence-certified rollout RBL."""

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

    def bound(self, x_int):
        return self.c_max_rem(x_int) + self.R_max

    # ------------------------------------------------------------ sampling
    def _z_digit(self, x_int, i):
        return int(x_int // self.powers[i]) % BASE_B

    def _sample_latent(self, i, zi, om):
        """Sample the latent level-8 CELL INDEX for UAV i consistent with its
        current cell, under the Bayesian predictive at posterior odds om.
        Returns a cell index in 0..(2^r_max - 1).  Vectorized."""
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

    def _apply(self, x_int, om, i, r2, latents):
        z = latents[i]                                     # latent level-8 cell
        m2 = z >> (self.r_max - r2)                        # nested projection
        zi = self._z_digit(x_int, i)
        z2 = z_code_b(r2, m2)
        x2 = x_int + (z2 - zi) * self.powers[i]
        om2 = om + self.pl._llr_i[i][z2] - self.pl._llr_i[i][zi]
        return x2, om2

    def rollout_return(self, x_int, h, action):
        """One rollout: take `action` (or STOP), then follow pi_b under the
        remaining budget h.  Returns G = radio cost + R_stop(X_T)."""
        latents = {}
        om = self.pl.omega(x_int)
        h_rem = float(h)
        cost = 0.0
        if action is not None:
            i, r2 = action
            if i not in latents:
                latents[i] = self._sample_latent(i, self._z_digit(x_int, i), om)
            r_old, _ = z_decode_b(self._z_digit(x_int, i))
            c = self.b_h + (r2 - r_old)
            x_int, om = self._apply(x_int, om, i, r2, latents)
            cost += c
            h_rem -= c
        # follow the base policy (respecting the remaining budget)
        for _ in range(4 * self.N + 2):
            a = self.base.act(self.pl, x_int, om)
            if a is None:
                break
            i, r2 = a
            r_old, _ = z_decode_b(self._z_digit(x_int, i))
            c = self.b_h + (r2 - r_old)
            if c > h_rem:
                break                                   # budget exhausted
            if i not in latents:
                latents[i] = self._sample_latent(i, self._z_digit(x_int, i), om)
            x_int, om = self._apply(x_int, om, i, r2, latents)
            cost += c
            h_rem -= c
        p = 1.0 / (1.0 + np.exp(-om))
        cost += min(self.C01 * p, self.C10 * (1.0 - p))
        return cost

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

    def _delta_n(self, a, n):
        nA = len(self._actions)
        return 6.0 * self._delta / (math.pi * math.pi * nA * n * n)

    def plan(self, x_int, h, eps, delta, max_samples=4000):
        """CR-RBL planning at (x, h): returns (action|None, info) with the
        anytime Hoeffding certificate (relative to the base policy)."""
        actions = self.feasible_actions(x_int, h)
        self._actions = actions
        self._delta = delta
        Bx = self.bound(x_int)
        R0 = self.pl.r_stop(x_int)                 # STOP pseudo-action (exact)
        if not actions:
            return None, {"certified": True, "samples": 0, "q_best": R0, "q_stop": R0}
        qhat = {}
        cnt = {}
        width = {}

        def L(a):
            return qhat[a] - width[a]

        def U(a):
            return qhat[a] + width[a]

        # initial sample for every action
        for a in actions:
            cnt[a] = 1
            qhat[a] = self.rollout_return(x_int, h, a)
            width[a] = Bx * math.sqrt(math.log(2.0 / self._delta_n(a, 1)) / 2.0)

        for _ in range(max_samples):
            best = min(actions, key=lambda a: qhat[a])
            if len(actions) > 1:
                chall = min((a for a in actions if a != best), key=L)
                a_s = best if width[best] >= width[chall] else chall
            else:
                a_s = best
            cnt[a_s] += 1
            g = self.rollout_return(x_int, h, a_s)
            qhat[a_s] = qhat[a_s] + (g - qhat[a_s]) / cnt[a_s]
            width[a_s] = Bx * math.sqrt(math.log(2.0 / self._delta_n(a_s, cnt[a_s]))
                                        / (2.0 * cnt[a_s]))
            # certificate: best action vs all others (+ the STOP option)
            if U(best) <= min([L(b) for b in actions if b != best], default=-np.inf) + eps:
                if R0 <= min([L(b) for b in actions], default=np.inf):
                    return None, {"certified": True, "samples": sum(cnt.values()),
                                  "q_best": R0, "q_stop": R0}
                return best, {"certified": True, "samples": sum(cnt.values()),
                              "q_best": qhat[best], "q_stop": R0,
                              "width_best": width[best]}
        # budget exhausted without certification: pick the empirical best
        best = min(actions, key=lambda a: qhat[a])
        if R0 <= qhat[best] - width[best]:
            return None, {"certified": False, "samples": sum(cnt.values()),
                          "q_best": R0, "q_stop": R0}
        return best, {"certified": False, "samples": sum(cnt.values()),
                      "q_best": qhat[best], "q_stop": R0, "width_best": width[best]}
