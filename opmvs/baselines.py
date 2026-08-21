"""Baseline algorithms (SystemModel §49-§53).

B0  Raw Full Fusion          — continuous LLR, ideal upper reference (§49)
B1  Max-Bit All-Neighbor     — every UAV sends 4-bit message (achievable
                               full-report reference, 4N bits) (§49)
B2  Random-(K)               — K random UAVs report full 4-bit (§50)
B3  Sensing-SNR Top-(K)      — K best UAVs by gamma^s report full 4-bit (§50)
B4  Cost-Aware Top-(K)       — score = D_i / c_i_full; in MVS-A costs are
                               uniform, so B4 == B3 (documented, §50)
B5  Censoring                — report 4-bit iff |L_i| > tau_c (§51)
B6  OTS-F                    — ordered full reporting by |L_i| (oracle order,
                               flagged OTS-Oracle-Order, §51) with posterior
                               early stopping
B8  P-OTS                    — ordered + fixed progressive 1->2->4 (§52)
B9  Global Fixed Progressive — all UAVs synchronously 1->2->4 (§53)
B11 Static Cost-Aware Prog.  — fixed SNR-based order, progressive 1->2->4,
                               no realized-evidence adaptation (§53)

Every baseline returns (Omega_at_stop, cost) on shared (H, L) episodes;
P_FA = 0.05 is enforced later by thresholding Omega (see mc.evaluate).
"""
from __future__ import annotations

import numpy as np

from .state import R_LEVELS, action_code, r_next


def _llr_matrix(ss, L, level):
    """(n, N) message-LLR of each UAV at `level` for LLR samples L."""
    out = np.empty((len(L), ss.N))
    for i in range(ss.N):
        m = ss.quants[i].cell_index(level, L[:, i])
        out[:, i] = ss.quants[i].llr[level][m]
    return out


def _stop_at_crossing(cum, scost, eta_s):
    """First step where |cum| >= eta_s; cost = cumulative step cost there."""
    n, S = cum.shape
    mask = (cum >= eta_s) | (cum <= -eta_s)
    has = mask.any(axis=1)
    t = np.where(has, mask.argmax(axis=1), S - 1)
    lam = np.where(has, cum[np.arange(n), t], cum[:, -1])
    cost = np.where(has, scost[t], scost[-1])
    return lam, cost


def _ladder(ss, L, order, eta_s):
    """Progressive ladder: for each UAV in `order` (n,N): refine 1->2->4,
    stop when |Omega| >= eta_s.  Returns (Omega, cost)."""
    n = len(L)
    N = ss.N
    llr_lev = {r: _llr_matrix(ss, L, r) for r in (1, 2, 4)}
    u_idx = np.repeat(order, 3, axis=1)                 # (n, 3N)
    step_level = np.tile([1, 2, 4], N)
    step_cost = np.tile([1, 1, 2], N).astype(float)
    ar = np.arange(n)
    inc = np.empty((n, 3 * N))
    for s in range(3 * N):
        u = u_idx[:, s]
        r = step_level[s]
        if s % 3 == 0:
            inc[:, s] = llr_lev[r][ar, u]
        else:
            inc[:, s] = llr_lev[r][ar, u] - llr_lev[step_level[s - 1]][ar, u]
    cum = np.cumsum(inc, axis=1)
    return _stop_at_crossing(cum, np.cumsum(step_cost), eta_s)


# ------------------------------------------------------------- baselines
def baseline_raw(ss, H, L):
    """B0: continuous full fusion (no communication)."""
    lam = ss.prior_log_odds + L.sum(axis=1)
    return lam, np.zeros(len(L))


def baseline_all_neighbor(ss, H, L, level=4):
    """B1: all UAVs send max-bit messages."""
    lam = ss.prior_log_odds + _llr_matrix(ss, L, level).sum(axis=1)
    cost = np.full(len(L), level * ss.N)
    return lam, cost


def baseline_random_k(ss, H, L, K, rng):
    """B2: K random UAVs report full 4-bit."""
    n = len(L)
    N = ss.N
    order = np.argsort(rng.random((n, N)), axis=1)[:, :K]
    mask = np.zeros((n, N), dtype=bool)
    for i in range(N):
        mask[:, i] = np.any(order == i, axis=1)
    lam = ss.prior_log_odds + (_llr_matrix(ss, L, 4) * mask).sum(axis=1)
    cost = np.full(n, 4 * K)
    return lam, cost


def baseline_snr_topk(ss, H, L, K):
    """B3: K UAVs with best sensing SNR report full 4-bit (B4 == B3 here)."""
    n = len(L)
    order = np.argsort(-ss.model.gamma_db)                # static, fixed
    sel = order[:K]
    lam = ss.prior_log_odds + _llr_matrix(ss, L, 4)[:, sel].sum(axis=1)
    cost = np.full(n, 4 * K)
    return lam, cost


def baseline_censoring(ss, H, L, tau):
    """B5: report 4-bit iff |L_i| > tau (local censoring)."""
    llr4 = _llr_matrix(ss, L, 4)
    mask = np.abs(L) > tau
    lam = ss.prior_log_odds + (llr4 * mask).sum(axis=1)
    cost = 4.0 * mask.sum(axis=1)
    return lam, cost


def _oracle_order(ss, L):
    """Order UAVs by |L_i| descending (OTS-Oracle-Order, §51)."""
    return np.argsort(-np.abs(L), axis=1)


def baseline_ots_f(ss, H, L, eta_s):
    """B6: ordered full reporting (oracle order) with early stopping."""
    n = len(L)
    N = ss.N
    order = _oracle_order(ss, L)
    llr4 = _llr_matrix(ss, L, 4)
    sorted_llr = np.take_along_axis(llr4, order, axis=1)
    cum = np.cumsum(sorted_llr, axis=1)
    lam, cost = _stop_at_crossing(cum, 4.0 * np.arange(1, N + 1), eta_s)
    return ss.prior_log_odds + lam, cost


def baseline_pots(ss, H, L, eta_s):
    """B8: P-OTS — ordered + fixed progressive 1->2->4, early stopping."""
    order = _oracle_order(ss, L)
    lam, cost = _ladder(ss, L, order, eta_s)
    return ss.prior_log_odds + lam, cost


def baseline_global_fixed(ss, H, L, eta_s):
    """B9: all UAVs synchronously refine 1->2->4, stop by |Omega|.

    Nested consistency makes the accumulated statistic telescope:
    l1 + (l2-l1) + (l4-l2) = l4, i.e. the round statistics are [l1, l2, l4].
    """
    n = len(L)
    N = ss.N
    llr_lev = {r: _llr_matrix(ss, L, r) for r in (1, 2, 4)}
    l1 = llr_lev[1].sum(axis=1)
    l2 = llr_lev[2].sum(axis=1)
    l4 = llr_lev[4].sum(axis=1)
    cum = np.column_stack([l1, l2, l4])
    scost = np.array([N, 2 * N, 4 * N], dtype=float)
    lam, cost = _stop_at_crossing(cum, scost, eta_s)
    return ss.prior_log_odds + lam, cost


def baseline_static_progressive(ss, H, L, eta_s):
    """B11: static cost-aware progressive — fixed SNR order, no adaptation."""
    order = np.tile(np.argsort(-ss.model.gamma_db), (len(L), 1))
    lam, cost = _ladder(ss, L, order, eta_s)
    return ss.prior_log_odds + lam, cost


# ------------------------------------------------------------ sweep registry
def baseline_sweeps(ss):
    """Parameter settings for each baseline: (param_name, values, callable)."""
    return {
        "B2_RandomK":      ("K", [1, 2, 3, 4], baseline_random_k),
        "B3_SNRTopK":      ("K", [1, 2, 3, 4], baseline_snr_topk),
        "B5_Censoring":    ("tau", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], baseline_censoring),
        "B6_OTSF":         ("eta_s", [0.5, 1.0, 2.0, 3.0, 4.0, 6.0], baseline_ots_f),
        "B8_POTS":         ("eta_s", [0.5, 1.0, 2.0, 3.0, 4.0, 6.0], baseline_pots),
        "B9_GlobalFixed":  ("eta_s", [0.5, 1.0, 2.0, 3.0, 4.0, 6.0], baseline_global_fixed),
        "B11_StaticProg":  ("eta_s", [0.5, 1.0, 2.0, 3.0, 4.0, 6.0], baseline_static_progressive),
    }


# -------------------------------------------- exact table-policy baselines
# Fair realizable baselines are deterministic functions of the evidence state,
# so they admit exact table policies over the finite state space (audit §8).
# Builders return a policy array: state idx -> action code (0 = STOP).

def build_global_fixed_policy(ss, eta_s):
    """B9 exact table policy: synchronous rounds 1->2->4, implemented as
    sequential single-UAV refinements with no mid-round stop checks; STOP when
    |Omega| >= eta_s (checked only at uniform-level states)."""
    N = ss.N
    policy = np.zeros(ss.n_states, dtype=np.int16)
    r_of = ss.r_of
    zcodes = ss.zcodes
    for idx in range(ss.n_states):
        rs = r_of[np.arange(N), zcodes[idx]]
        om = ss.omega[idx]
        if np.all(rs == rs[0]):
            r = int(rs[0])
            if r >= R_LEVELS[-1] or abs(om) >= eta_s:
                policy[idx] = 0                       # STOP
            else:
                policy[idx] = action_code(0, r_next(r))
        else:
            rmin = int(rs.min())
            r2 = r_next(rmin)
            for i in range(N):
                if rs[i] == rmin:
                    policy[idx] = action_code(i, r2)
                    break
    return policy


def build_static_progressive_policy(ss, eta_s):
    """B11 exact table policy: fixed SNR order, ladder 1->2->4 per UAV,
    |Omega| >= eta_s early stopping.  No realized-evidence adaptation."""
    N = ss.N
    order = list(np.argsort(-ss.model.gamma_db))
    policy = np.zeros(ss.n_states, dtype=np.int16)
    r_of = ss.r_of
    zcodes = ss.zcodes
    for idx in range(ss.n_states):
        rs = r_of[np.arange(N), zcodes[idx]]
        if abs(ss.omega[idx]) >= eta_s:
            policy[idx] = 0
            continue
        for i in order:
            if rs[i] < R_LEVELS[-1]:
                policy[idx] = action_code(i, r_next(int(rs[i])))
                break
    return policy


def build_seeded_pots_policy(ss, eta_s):
    """1-bit-seeded P-OTS exact table policy (audit §6, fair baseline):
    stage 1: every UAV pays 1 bit (0->1, no stop checks);
    stage 2: owner orders UAVs by the *paid* realized |ell^1_i| and runs the
    ladder 1->2->4 with |Omega| >= eta_s early stopping."""
    N = ss.N
    policy = np.zeros(ss.n_states, dtype=np.int16)
    r_of = ss.r_of
    zcodes = ss.zcodes
    llr_tab = ss.llr_tab
    for idx in range(ss.n_states):
        rs = r_of[np.arange(N), zcodes[idx]]
        om = ss.omega[idx]
        if (rs == 0).any():                           # seeding stage
            for i in range(N):
                if rs[i] == 0:
                    policy[idx] = action_code(i, 1)
                    break
            continue
        l1 = np.abs([llr_tab[i, int(zcodes[idx, i])] for i in range(N)])
        order = np.argsort(-l1, kind="stable")
        if abs(om) >= eta_s:
            policy[idx] = 0
            continue
        for i in order:
            if rs[i] < R_LEVELS[-1]:
                policy[idx] = action_code(i, r_next(int(rs[i])))
                break
    return policy
