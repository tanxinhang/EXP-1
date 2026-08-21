"""Resource-bounded lookahead (RBL) solver — R2 of adcice/002.md.

    V_h(x) = min{ R_stop(x),  min_{a : c_a <= h} [ c_a + E V_{h - c_a}(x') ] }

The horizon h is measured in *future communication resource* (payload bits),
NOT in the number of actions.  This removes the action-size confounder that
made action-count depth (O-PEF-1/2/3) inconsistent across action families
(002.md §5: 'action-count depth is the wrong horizon coordinate').

Hard certification: with H = 4N = 16 (max remaining payload bits at the root)
the budget never binds, so V_16(x) == V*(x) pointwise to machine precision.

Evaluation of a budget-H policy uses exact forward probability propagation
over (state, remaining budget) — no Monte Carlo (002.md §8).
"""
from __future__ import annotations

import numpy as np

from .state import R_LEVELS, action_code, action_decode


class ResourceBoundedLookahead:
    def __init__(self, ss, mu_M, mu_F, pi=(0.5, 0.5), H_max=None):
        self.ss = ss
        self.mu_M = float(mu_M)
        self.mu_F = float(mu_F)
        self.pi = pi
        self.C01 = mu_M / pi[1]
        self.C10 = mu_F / pi[0]
        self.H_max = int(H_max) if H_max else 4 * ss.N     # max remaining bits
        p = ss.p
        self.Rstop = np.minimum(self.C01 * p, self.C10 * (1.0 - p))
        self.logp = ss.logp
        self.logq = ss.logq

    # ----------------------------------------------------------------- solve
    def solve(self):
        """Compute V_h and pi_h for h = 0..H_max.  Returns (V, policies)."""
        ss = self.ss
        H = self.H_max
        V = np.empty((H + 1, ss.n_states))
        pol = np.zeros((H + 1, ss.n_states), dtype=np.int16)
        V[0] = self.Rstop                                   # no action fits
        pol[0] = 0
        logp = self.logp
        logq = self.logq
        Rstop = self.Rstop
        for h in range(1, H + 1):
            Vh = V[h]
            polh = pol[h]
            Vh[:] = Rstop
            for L in range(ss.max_level, -1, -1):
                for idx in ss.states_by_level[L]:
                    best = float(Rstop[idx])
                    best_a = 0
                    lp = float(logp[idx])
                    lq = float(logq[idx])
                    zrow = ss.zcodes[idx]
                    for i in range(ss.N):
                        zi = int(zrow[i])
                        for (r2, c_a, children) in ss.actions[i][zi]:
                            if c_a > h:
                                continue
                            Vhc = V[h - c_a]
                            E = 0.0
                            for (delta, l1, l0) in children:
                                a_ = lp + l1
                                b_ = lq + l0
                                m_ = a_ if a_ >= b_ else b_
                                logw = m_ + np.log1p(np.exp(-abs(a_ - b_)))
                                E += np.exp(logw) * Vhc[idx + delta]
                            Q = c_a + E
                            if Q < best:
                                best = Q
                                best_a = action_code(i, r2)
                    Vh[idx] = best
                    polh[idx] = best_a
        self.V = V
        self.policies = pol
        return V, pol

    def verify_full_budget(self, dp_V):
        """max_x |V_H(x) - V*(x)| — hard certification for H = 4N."""
        dev = np.abs(self.V[self.H_max] - dp_V).max()
        # V_h must be non-increasing in h (more budget -> no worse)
        mono = max(float((self.V[h] - self.V[h - 1]).max()) for h in range(1, self.H_max + 1))
        return {"max_dev": float(dev), "monotonicity_dev": mono,
                "passed": dev < 1e-8 and mono < 1e-8}


# -------------------------------------------------- exact evaluation (idx, h)
def exact_evaluate_rbl(ss, policies, H, pi=(0.5, 0.5)):
    """Exact forward propagation of a budget-H RBL policy family over the
    (state, remaining-budget) product space.

    policies: (H+1, n_states) int16 — policy at (idx, h) = policies[h, idx].

    Returns the STOP distribution (omega, m0, m1) plus E[B|H0], E[B|H1], E[B].
    """
    n = ss.n_states
    m0 = np.zeros((H + 1, n))
    m1 = np.zeros((H + 1, n))
    m0[H, 0] = pi[0]
    m1[H, 0] = pi[1]
    cost0 = 0.0
    cost1 = 0.0
    stop_omega = []
    stop_m0 = []
    stop_m1 = []
    for h in range(H, -1, -1):
        m0h = m0[h]
        m1h = m1[h]
        for L in range(ss.max_level, -1, -1):
            for idx in ss.states_by_level[L]:
                a0 = m0h[idx]
                a1 = m1h[idx]
                if a0 == 0.0 and a1 == 0.0:
                    continue
                act = int(policies[h, idx])
                if act == 0:
                    stop_omega.append(float(ss.omega[idx]))
                    stop_m0.append(a0)
                    stop_m1.append(a1)
                    continue
                i, r2 = action_decode(act)
                zi = int(ss.zcodes[idx, i])
                children = None
                for (r2b, c_a, ch) in ss.actions[i][zi]:
                    if r2b == r2:
                        children = ch
                        break
                if children is None:
                    raise RuntimeError(f"illegal action ({i},{r2}) at state {idx}, h={h}")
                cost0 += a0 * c_a
                cost1 += a1 * c_a
                hc = h - c_a
                m0hc = m0[hc]
                m1hc = m1[hc]
                for (delta, l1, l0) in children:
                    jdx = idx + delta
                    m0hc[jdx] += a0 * float(np.exp(l0))
                    m1hc[jdx] += a1 * float(np.exp(l1))
    return {
        "omega": np.array(stop_omega),
        "m0": np.array(stop_m0),
        "m1": np.array(stop_m1),
        "eb0": cost0 / pi[0],
        "eb1": cost1 / pi[1],
        "eb": cost0 + cost1,
    }
