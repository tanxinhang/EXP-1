"""Exact acyclic backward DAG-DP (SystemModel §18, §21-§22, §35).

Because MVS-A has no packet loss and refinement is irreversible, the state
transition graph is a DAG ordered by the number of unrevealed bits
(level = sum_i (r_max - r_i)).  We evaluate

    V*(x) = min{ R_stop(x), min_{a in A(x)} [ c_a + E[V*(x') | x, a] ] }

by memoized backward recursion over the DAG (terminal states at level 0),
NOT by value iteration.  A Bellman-residual audit
(max_x |V(x) - T V(x)|) is run to verify double-precision correctness (§35).

Terminal risks (Lagrangian, §17):
    C_01 = mu_M / pi_1 ,  C_10 = mu_F / pi_0
    R_0(x) = C_01 p ,  R_1(x) = C_10 (1-p) ,  R_stop(x) = min{R_0, R_1}.
"""
from __future__ import annotations

import numpy as np

from .state import R_LEVELS, action_code


class SolverBase:
    """Shared machinery: terminal risks + one Bellman/TV pass."""

    def __init__(self, ss):
        self.ss = ss

    # ------------------------------------------------------- terminal risks
    def risks(self, mu_M, mu_F, pi=(0.5, 0.5)):
        ss = self.ss
        C01 = float(mu_M) / pi[1]
        C10 = float(mu_F) / pi[0]
        p = ss.p
        q = ss.q
        Rstop = np.minimum(C01 * p, C10 * q)
        return {
            "C01": C01, "C10": C10,
            "logp": ss.logp, "logq": ss.logq,
            "Rstop": Rstop,
        }

    # --------------------------------------------------------- one DP pass
    def _pass(self, out, policy, cont, r):
        """Fill `out`/`policy` with min{R_stop, min_a [c_a + E cont(x')]}.

        cont : array of continuation values indexed by child state index.
        Children always lie at a strictly lower level, so `cont` entries for
        them are already final when `out` aliases `cont` (DP case).
        """
        ss = self.ss
        Rstop = r["Rstop"]
        logp = r["logp"]
        logq = r["logq"]
        for L in range(ss.max_level + 1):
            for idx in ss.states_by_level[L]:
                best = float(Rstop[idx])
                best_a = 0
                lp = float(logp[idx])
                lq = float(logq[idx])
                zrow = ss.zcodes[idx]
                for i in range(ss.N):
                    zi = int(zrow[i])
                    for (r2, c_a, children) in ss.actions[i][zi]:
                        E = 0.0
                        for (delta, l1, l0) in children:
                            a_ = lp + l1
                            b_ = lq + l0
                            m_ = a_ if a_ >= b_ else b_
                            logw = m_ + np.log1p(np.exp(-abs(a_ - b_)))
                            E += np.exp(logw) * cont[idx + delta]
                        Q = c_a + E
                        if Q < best:
                            best = Q
                            best_a = action_code(i, r2)
                out[idx] = best
                policy[idx] = best_a
        return out, policy


class ExactDP(SolverBase):
    """Exact DAG-DP optimal policy (MVS-A oracle, N=4)."""

    def solve(self, mu_M, mu_F, pi=(0.5, 0.5)):
        r = self.risks(mu_M, mu_F, pi)
        V = np.empty(self.ss.n_states)
        policy = np.zeros(self.ss.n_states, dtype=np.int16)
        self._pass(V, policy, cont=V, r=r)      # cont aliases V: in-place
        self.V = V
        self.policy = policy
        self.risks_ = r
        return V, policy

    def bellman_residual(self):
        """max_x |V(x) - T V(x)| and mean residual (Section 35 / Gate G1)."""
        V = self.V
        r = self.risks_
        TV = np.empty(self.ss.n_states)
        pol = np.zeros(self.ss.n_states, dtype=np.int16)
        self._pass(TV, pol, cont=V, r=r)
        res = np.abs(V - TV)
        return float(res.max()), float(res.mean())
