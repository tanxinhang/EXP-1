"""Evidence state space (SystemModel §10, §20).

Per-UAV evidence state:  z_i = empty (r=0)  or  (r_i, m_i) with r_i in {1,2,4}.

    z_i = 0        : no message yet (r=0)
    z_i = 1..2     : 1-bit message, cell m = z_i - 1
    z_i = 3..6     : 2-bit message, cell m = z_i - 3
    z_i = 7..22    : 4-bit message, cell m = z_i - 7

System state:  z = (z_1, ..., z_N)  — a sufficient Markov state (§10).
The posterior odds Omega = sum_i ell_i(z_i) (+ prior log odds) is derived
from z and cached to avoid recomputation.

State count per UAV = 1 + 2 + 4 + 16 = 23, hence MVS-A N=4 has 23^4 states
(§20), which the exact DAG-DP can solve exhaustively.
"""
from __future__ import annotations

import numpy as np

from .fusion import log_sigmoid, log_one_minus_sigmoid

R_LEVELS = (1, 2, 4)          # message levels (MVS-A)
RMAX = 4                      # deepest level
BASE = 1 + 2 + 4 + 16         # 23 per-UAV evidence states


def z_code(r, m):
    if r == 0:
        return 0
    if r == 1:
        return 1 + int(m)
    if r == 2:
        return 3 + int(m)
    if r == 4:
        return 7 + int(m)
    raise ValueError(f"unsupported level r={r}")


def z_code_vec(r, m):
    """Vectorized z_code for array cell indices m."""
    m = np.asarray(m, dtype=np.int64)
    if r == 1:
        return 1 + m
    if r == 2:
        return 3 + m
    if r == 4:
        return 7 + m
    raise ValueError(f"unsupported level r={r}")


def z_decode(z):
    z = int(z)
    if z == 0:
        return 0, -1
    if 1 <= z <= 2:
        return 1, z - 1
    if 3 <= z <= 6:
        return 2, z - 3
    if 7 <= z <= 22:
        return 4, z - 7
    raise ValueError(f"invalid z={z}")


def action_code(i, r2):
    """Global action code (0 = STOP)."""
    return 1 + int(i) * len(R_LEVELS) + R_LEVELS.index(int(r2))


def action_decode(code):
    """Decode action code -> (i, r2) or None for STOP."""
    code = int(code)
    if code == 0:
        return None
    code -= 1
    i = code // len(R_LEVELS)
    r2 = R_LEVELS[code % len(R_LEVELS)]
    return int(i), int(r2)


def r_next(r):
    """The next message level after r in R_LEVELS (adjacent refinement)."""
    for r2 in R_LEVELS:
        if r2 > r:
            return r2
    return None


class StateSpace:
    def __init__(self, model, quantizers, prior_log_odds=0.0, cross_level=True):
        self.model = model
        self.N = model.N
        self.quants = list(quantizers)
        self.prior_log_odds = float(prior_log_odds)
        self.cross_level = bool(cross_level)
        self.powers = BASE ** np.arange(self.N)          # (N,)
        self.n_states = int(BASE ** self.N)

        # per-UAV lookup tables over z codes
        self.r_of = np.zeros((self.N, BASE), dtype=np.int8)
        self.m_of = np.full((self.N, BASE), -1, dtype=np.int16)
        self.llr_tab = np.zeros((self.N, BASE), dtype=float)
        for i in range(self.N):
            for z in range(BASE):
                r, m = z_decode(z)
                self.r_of[i, z] = r
                self.m_of[i, z] = m
                if r > 0:
                    self.llr_tab[i, z] = self.quants[i].llr[r][m]

        # per-UAV, per-z legal refinement actions
        #   actions[i][z] = [ (r2, c_a, [(delta, logP1c, logP0c), ...]), ... ]
        # with delta = (z2 - z) * BASE^i  (child state index offset)
        #
        # cross_level=True : all r -> r' > r (0->1, 0->2, 0->4, 1->2, 1->4, 2->4)
        # cross_level=False: adjacent-only (0->1, 1->2, 2->4)  [MVS-A-R1 main]
        # Note (§5 of the audit): with b_h=0 / perfect channel the cross-level
        # action is weakly dominated by adjacent refinement in the *exact* DP,
        # but it structurally biases finite-depth lookahead — hence MVS-A-R1
        # freezes the adjacent-only family.
        self.actions = [[[] for _ in range(BASE)] for _ in range(self.N)]
        for i in range(self.N):
            q = self.quants[i]
            for z in range(BASE):
                r, m = z_decode(z)
                r2s = R_LEVELS if self.cross_level else (r_next(r),)
                for r2 in r2s:
                    if r2 is None or r2 <= r:
                        continue
                    c_a = r2 - r                            # payload bits, §16.1
                    children = []
                    for m2 in q.desc_cells(r, m, r2):
                        lp1c = q.logP1[r2][m2] - (0.0 if r == 0 else q.logP1[r][m])
                        lp0c = q.logP0[r2][m2] - (0.0 if r == 0 else q.logP0[r][m])
                        delta = int((z_code(r2, m2) - z) * self.powers[i])
                        children.append((delta, float(lp1c), float(lp0c)))
                    self.actions[i][z].append((r2, c_a, children))

        self._build_states()

    # ------------------------------------------------------------ state table
    def _build_states(self):
        n = self.n_states
        idx = np.arange(n, dtype=np.int64)
        digits = np.empty((n, self.N), dtype=np.int16)
        rem = idx.copy()
        for j in range(self.N):
            digits[:, j] = rem % BASE
            rem //= BASE
        self.zcodes = digits                              # (n, N)
        omega = np.full(n, self.prior_log_odds)
        lev = np.zeros(n, dtype=np.int8)
        for j in range(self.N):
            omega += self.llr_tab[j, digits[:, j]]
            lev += (RMAX - self.r_of[j, digits[:, j]])    # unrevealed bits
        self.omega = omega                                # posterior log odds
        self.level = lev
        self.max_level = int(lev.max())
        self.states_by_level = [np.flatnonzero(lev == L) for L in range(self.max_level + 1)]
        # cached posterior pieces
        self.logp = log_sigmoid(omega)
        self.logq = log_one_minus_sigmoid(omega)
        self.p = np.exp(self.logp)
        self.q = np.exp(self.logq)

    # ------------------------------------------------------------ encodings
    def encode(self, z):
        """(n, N) z codes -> (n,) state indices."""
        return (z.astype(np.int64) * self.powers).sum(axis=1)

    def decode(self, idx):
        """State indices -> (n, N) z codes."""
        idx = np.asarray(idx, dtype=np.int64)
        z = np.empty((len(idx), self.N), dtype=np.int16)
        rem = idx.copy()
        for j in range(self.N):
            z[:, j] = rem % BASE
            rem //= BASE
        return z

    def action_children(self, i, z_i):
        return self.actions[i][int(z_i)]

    def summary(self):
        return {
            "N": self.N,
            "base": BASE,
            "n_states": self.n_states,
            "max_level": self.max_level,
            "states_per_level": [int(len(x)) for x in self.states_by_level],
        }
