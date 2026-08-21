"""O-PEF solvers: depth-1 (O-PEF-1) and depth-2 exact (O-PEF-2E).

O-PEF-1 (§23, low-complexity ablation):
    Q_a^(1)(x) = c_a + E[ R_stop(x') | x, a ]
    stop iff R_stop(x) <= min_a Q_a^(1)(x).

O-PEF-2E (§24, §27, main algorithm, exact discrete expectation):
    Q_a^(2)(x) = c_a + E_{x'|x,a}[ min{ R_stop(x'), min_b [ c_b + E[R_stop(x'')|x',b] ] } ]
               = c_a + E[ V1(x') | x, a ]          (V1 = O-PEF-1 value)
    stop iff R_stop(x) <= min_a Q_a^(2)(x).

Both policies are greedy lookahead approximations of the exact DP; for MVS-A
the depth-2 expectation is evaluated exactly over all children (no sampling,
no candidate pruning — §27 says O-PEF-2E is for exact evaluation).
"""
from __future__ import annotations

import numpy as np

from .dp import SolverBase


class OPEF1(SolverBase):
    """Depth-1 lookahead policy (value V1, policy pi1)."""

    def solve(self, mu_M, mu_F, pi=(0.5, 0.5)):
        r = self.risks(mu_M, mu_F, pi)
        V1 = np.empty(self.ss.n_states)
        policy = np.zeros(self.ss.n_states, dtype=np.int16)
        self._pass(V1, policy, cont=r["Rstop"], r=r)   # one-step lookahead
        self.V1 = V1
        self.policy = policy
        self.risks_ = r
        return V1, policy


class OPEF2(SolverBase):
    """Depth-2 exact lookahead policy (needs V1 from an OPEF1 solve)."""

    def solve(self, mu_M, mu_F, v1, pi=(0.5, 0.5)):
        r = self.risks(mu_M, mu_F, pi)
        V2 = np.empty(self.ss.n_states)
        policy = np.zeros(self.ss.n_states, dtype=np.int16)
        self._pass(V2, policy, cont=v1, r=r)           # depth-2 lookahead
        self.V2 = V2
        self.policy = policy
        self.risks_ = r
        return V2, policy


class OPEF3(SolverBase):
    """Depth-3 exact lookahead policy (diagnostic solver improvement, §36/§66:
    deeper truncation reduces the depth-2 gap toward the exact DP)."""

    def solve(self, mu_M, mu_F, v2, pi=(0.5, 0.5)):
        r = self.risks(mu_M, mu_F, pi)
        V3 = np.empty(self.ss.n_states)
        policy = np.zeros(self.ss.n_states, dtype=np.int16)
        self._pass(V3, policy, cont=v2, r=r)           # depth-3 lookahead
        self.V3 = V3
        self.policy = policy
        self.risks_ = r
        return V3, policy
