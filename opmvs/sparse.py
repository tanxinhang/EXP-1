"""MVS-B0 sparse state backend (adcice/004.md §2-§3).

State:  x = (z_1, ..., z_N),  z_i = per-UAV evidence code
        (0 = no message; else the cell code at its precision level;
         levels R = {1,2,4,8} for MVS-B, so z_i in 0..278, BASE = 279).
Posterior odds computed on the fly:
        Omega(x) = prior_log_odds + sum_i ell_i(z_i).
Legal actions, conditional PMFs and child states are generated on demand —
**no global state table is ever built** (N=8 -> 279^8 states would be
infeasible; this backend is what MVS-B actually runs on).

Internally the state is a single integer  x_int = sum_i z_i * BASE^i
(mixed-radix), giving cheap child arithmetic and fast memo keys.

Solver: resource-bounded receding-horizon planner (R2.1 recursion):

        V(x, h) = min{ R_stop(x), min_{a: c_a <= h} [ c_a + E V(x', h - c_a) ] },
cost  c_a = b_h + (r' - r)   (header + payload, radio bits; B0: p_succ = 1).
"""
from __future__ import annotations

import numpy as np

from .fusion import log_sigmoid, log_one_minus_sigmoid

R_LEVELS_B = (1, 2, 4, 8)
Z_OFFSETS = {0: 0, 1: 1, 2: 3, 4: 7, 8: 23}
BASE_B = 279


def z_code_b(r, m):
    if r == 0:
        return 0
    return Z_OFFSETS[int(r)] + int(m)


def z_decode_b(z):
    z = int(z)
    if z == 0:
        return 0, -1
    if 1 <= z <= 2:
        return 1, z - 1
    if 3 <= z <= 6:
        return 2, z - 3
    if 7 <= z <= 22:
        return 4, z - 7
    if 23 <= z <= 278:
        return 8, z - 23
    raise ValueError(f"invalid z={z}")


# module-level template cache: per-(i, z_i) action templates depend only on
# (quantizers, b_h, cross_level, levels) — NOT on (mu_M, mu_F) — so planner
# construction is cheap even when a fresh planner is created per MC step.
_TPL_CACHE = {}


def _build_templates(quantizers, b_h, cross_level, levels):
    N = len(quantizers)
    r_max = max(levels)
    tpl = [[None] * BASE_B for _ in range(N)]
    for i in range(N):
        q = quantizers[i]
        for z in range(BASE_B):
            r, m = z_decode_b(z)
            if r >= r_max:
                tpl[i][z] = []
                continue
            r2s = levels if cross_level else \
                (next((r2 for r2 in levels if r2 > r), None),)
            out = []
            for r2 in r2s:
                if r2 is None or r2 <= r:
                    continue
                c_a = b_h + (r2 - r)
                cells = []
                for m2 in q.desc_cells(r, m, r2):
                    lp0c = q.logP0[r2][m2] - (0.0 if r == 0 else q.logP0[r][m])
                    lp1c = q.logP1[r2][m2] - (0.0 if r == 0 else q.logP1[r][m])
                    cells.append((int(m2), float(lp0c), float(lp1c)))
                out.append((r2, c_a, cells))
            tpl[i][z] = out
    return tpl


def _get_templates(quantizers, b_h, cross_level, levels):
    key = (tuple(id(q) for q in quantizers), float(b_h), bool(cross_level), tuple(levels))
    if key not in _TPL_CACHE:
        _TPL_CACHE[key] = _build_templates(quantizers, float(b_h), bool(cross_level),
                                           tuple(levels))
    return _TPL_CACHE[key]


class SparsePlanner:
    """Memoized resource-bounded receding-horizon planner over int states."""

    def __init__(self, quantizers, mu_M, mu_F, pi=(0.5, 0.5), b_h=0.0,
                 cross_level=True, prior_log_odds=0.0, levels=R_LEVELS_B):
        self.quants = list(quantizers)
        self.N = len(self.quants)
        self.mu_M = float(mu_M)
        self.mu_F = float(mu_F)
        self.pi = pi
        self.C01 = mu_M / pi[1]
        self.C10 = mu_F / pi[0]
        self.b_h = float(b_h)
        self.cross_level = bool(cross_level)
        self.prior_log_odds = float(prior_log_odds)
        self.levels = tuple(levels)
        self.r_max = max(levels)
        self.powers = [BASE_B ** i for i in range(self.N)]   # Python ints (279^8 > int64)
        self.memo = {}
        self.n_expansions = 0
        self._tpl = _get_templates(self.quants, self.b_h, self.cross_level, self.levels)

    # ----------------------------------------------------------- state utils
    def encode(self, z_tuple):
        return int(sum(int(z) * self.powers[i] for i, z in enumerate(z_tuple)))

    def decode(self, x_int):
        rem = int(x_int)
        out = []
        for j in range(self.N):
            out.append(rem % BASE_B)
            rem //= BASE_B
        return tuple(out)

    def omega(self, x_int):
        om = self.prior_log_odds
        rem = int(x_int)
        for i in range(self.N):
            z = rem % BASE_B
            rem //= BASE_B
            if z:
                r, m = z_decode_b(z)
                om += self.quants[i].llr[r][m]
        return om

    def posterior(self, x_int):
        om = self.omega(x_int)
        logp = float(log_sigmoid(om))
        logq = float(log_one_minus_sigmoid(om))
        p = float(np.exp(logp))
        return om, p, logp, logq

    def r_stop(self, x_int):
        om, p, _, _ = self.posterior(x_int)
        return min(self.C01 * p, self.C10 * (1.0 - p))

    # ------------------------------------------------------------ solver
    def solve(self, x_int, h):
        """Return (value, action) for int state x_int with budget h;
        action = (i, r2) or None for STOP.  Memoized on (x_int, h)."""
        key = (int(x_int), int(h))
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        self.n_expansions += 1
        om, p, logp, logq = self.posterior(x_int)
        best = min(self.C01 * p, self.C10 * (1.0 - p))
        best_a = None
        if h > 0:
            rem = int(x_int)
            zs = []
            for _ in range(self.N):
                zs.append(rem % BASE_B)
                rem //= BASE_B
            for i in range(self.N):
                zi = zs[i]
                pw = self.powers[i]
                for (r2, c_a, cells) in self._tpl[i][zi]:
                    if c_a > h:
                        continue
                    E = 0.0
                    for (m2, lp0c, lp1c) in cells:
                        z2 = z_code_b(r2, m2)
                        cx = x_int + (z2 - zi) * pw
                        a_ = logp + lp1c
                        b_ = logq + lp0c
                        m_ = a_ if a_ >= b_ else b_
                        logw = m_ + np.log1p(np.exp(-abs(a_ - b_)))
                        w = float(np.exp(logw))
                        val, _ = self.solve(cx, h - c_a)
                        E += w * val
                    Q = c_a + E
                    if Q < best:
                        best = Q
                        best_a = (i, r2)
        self.memo[key] = (best, best_a)
        return best, best_a

    def reset(self):
        self.memo = {}
        self.n_expansions = 0


# ------------------------------------------------------------- equivalence
def equivalence_with_old_backend(ss_old, rbl_old, mu_M, mu_F, H_values,
                                 idxs=None, b_h=0.0):
    """B0-G0: compare the sparse planner with the eager table RBL on N=4.

    ss_old: the old StateSpace (N=4, levels {1,2,4});
    rbl_old: ResourceBoundedLookahead solved on ss_old (V, policies);
    For every state idx, the sparse planner (int state = mixed-radix of the
    old z-codes) must return the same (value, first action) as
    (rbl_old.V[H, idx], rbl_old.policies[H, idx]).
    """
    from .rbl import ResourceBoundedLookahead
    if rbl_old is None:
        rbl_old = ResourceBoundedLookahead(ss_old, mu_M, mu_F)
        rbl_old.solve()
    planner = SparsePlanner(ss_old.quants, mu_M, mu_F, b_h=b_h,
                            cross_level=True, levels=(1, 2, 4))
    if idxs is None:
        idxs = np.arange(ss_old.n_states)
    results = {}
    for H in H_values:
        max_val = 0.0
        n_mis = 0
        n_near_tie = 0
        for idx in idxs:
            zs = tuple(int(v) for v in ss_old.zcodes[idx])
            x_int = planner.encode(zs)
            val, act = planner.solve(x_int, H)
            dev = abs(val - rbl_old.V[H, idx])
            max_val = max(max_val, dev)
            tgt = int(rbl_old.policies[H, idx])
            if act is None:
                got = 0
            else:
                i, r2 = act
                got = 1 + i * len((1, 2, 4)) + (1, 2, 4).index(r2)
            if got != tgt:
                n_mis += 1
                if dev < 1e-6:
                    n_near_tie += 1
        results[H] = {"max_val_dev": float(max_val), "n_action_mismatch": int(n_mis),
                      "n_near_tie": int(n_near_tie),
                      "n_states": int(len(idxs)), "memo_size": len(planner.memo),
                      "n_expansions": planner.n_expansions}
    return results
