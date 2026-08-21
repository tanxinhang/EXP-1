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


class OnlinePlanner:
    """Sparse memoized recursive planner (R2.1-G3 of adcice/003.md).

    Solves the same recursion as ResourceBoundedLookahead but *lazily*:
    only the (x, h) pairs reachable from the requested state are expanded —
    no (H+1) x n_states table is ever built.  Deployed receding: re-call
    solve(current_state, H) at every newly received message.
    """

    def __init__(self, ss, mu_M, mu_F, pi=(0.5, 0.5)):
        self.ss = ss
        self.mu_M = float(mu_M)
        self.mu_F = float(mu_F)
        self.pi = pi
        self.C01 = mu_M / pi[1]
        self.C10 = mu_F / pi[0]
        p = ss.p
        self.Rstop = np.minimum(self.C01 * p, self.C10 * (1.0 - p))
        self.logp = ss.logp
        self.logq = ss.logq
        self.memo = {}
        self.n_expansions = 0

    def solve(self, idx, h):
        """Return (value, action) for state idx with remaining budget h."""
        key = (int(idx), int(h))
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        self.n_expansions += 1
        ss = self.ss
        best = float(self.Rstop[idx])
        best_a = 0
        if h > 0:
            lp = float(self.logp[idx])
            lq = float(self.logq[idx])
            zrow = ss.zcodes[idx]
            for i in range(ss.N):
                zi = int(zrow[i])
                for (r2, c_a, children) in ss.actions[i][zi]:
                    if c_a > h:
                        continue
                    E = 0.0
                    for (delta, l1, l0) in children:
                        a_ = lp + l1
                        b_ = lq + l0
                        m_ = a_ if a_ >= b_ else b_
                        logw = m_ + np.log1p(np.exp(-abs(a_ - b_)))
                        w = float(np.exp(logw))
                        val, _ = self.solve(idx + delta, h - c_a)
                        E += w * val
                    Q = c_a + E
                    if Q < best:
                        best = Q
                        best_a = action_code(i, r2)
        self.memo[key] = (best, best_a)
        return best, best_a

    def reset(self):
        self.memo = {}
        self.n_expansions = 0


def online_equivalence_audit(ss, V_table, pol_table, H, mu_M, mu_F, idxs):
    """Compare the online sparse planner's (value, first action) against the
    eager table RBL for every state in `idxs` (R2.1-G3).  A shared memo is
    used across the test calls; its final size reports the reachable cone."""
    planner = OnlinePlanner(ss, mu_M, mu_F)
    max_val = 0.0
    n_mismatch = 0
    for idx in idxs:
        val, act = planner.solve(int(idx), H)
        max_val = max(max_val, abs(val - V_table[H, idx]))
        if act != int(pol_table[H, idx]):
            n_mismatch += 1
    return {
        "max_val_dev": float(max_val),
        "n_action_mismatch": int(n_mismatch),
        "n_states": int(len(idxs)),
        "memo_size": len(planner.memo),
        "n_expansions": planner.n_expansions,
        "full_table_size": (H + 1) * ss.n_states,
    }



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
