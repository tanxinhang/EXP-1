"""Exact forward probability propagation for table policies (MVS-A-R1).

The MVS-A state space is finite and the transition graph is a DAG, so a table
policy (state -> action|STOP) can be evaluated *exactly* by propagating
probability mass from the root state, separately under H0 and H1.  This gives
machine-precision P_D, P_FA and E[B] without Monte Carlo — much stronger
certification than MC (§8 of adcice/001.md).

Primary evaluation is objective-consistent (P0 fix of the audit): the policy's
own terminal Bayes decision is used — declare H1 iff Omega > log(mu_F/mu_M),
equivalently R_1(x) < R_0(x).  The STOP-state log-odds distribution is exposed
so that the terminal-statistic ROC diagnostic (NP threshold on Omega at stop)
is available as a *secondary* diagnostic, never as the primary metric.

Also provides the independent certification gates:
  G1a — transition normalization, posterior martingale, information
        monotonicity (independent of the DP solve);
  G1b — J^{pi_DP}(x0) == V*(x0) via forward propagation vs backward solve.
"""
from __future__ import annotations

import numpy as np

from .state import BASE, action_decode


# ------------------------------------------------------- stop distribution
def exact_stop_distribution(ss, policy, pi=(0.5, 0.5)):
    """Exact forward propagation of a table policy from the root state.

    Returns the STOP-state distribution:
      omega : (K,) log posterior odds Omega at each STOP state
      m0    : (K,) P(stop at that state, H0)
      m1    : (K,) P(stop at that state, H1)
      eb0, eb1, eb : E[B|H0], E[B|H1], E[B]
    """
    ss_ = ss
    n = ss_.n_states
    m0 = np.zeros(n)
    m1 = np.zeros(n)
    m0[0] = pi[0]
    m1[0] = pi[1]
    cost0 = 0.0
    cost1 = 0.0
    stop_idx = []
    for L in range(ss_.max_level, -1, -1):
        for idx in ss_.states_by_level[L]:
            a0 = m0[idx]
            a1 = m1[idx]
            if a0 == 0.0 and a1 == 0.0:
                continue
            act = int(policy[idx])
            if act == 0:                                   # STOP
                stop_idx.append(idx)
                continue
            i, r2 = action_decode(act)
            zi = int(ss_.zcodes[idx, i])
            children = None
            for (r2b, c_a, ch) in ss_.actions[i][zi]:
                if r2b == r2:
                    children = ch
                    break
            if children is None:
                raise RuntimeError(f"illegal action ({i},{r2}) at state {idx}")
            cost0 += a0 * c_a
            cost1 += a1 * c_a
            for (delta, l1, l0) in children:
                jdx = idx + delta
                m0[jdx] += a0 * float(np.exp(l0))
                m1[jdx] += a1 * float(np.exp(l1))
            m0[idx] = 0.0
            m1[idx] = 0.0
    s0 = m0[stop_idx]
    s1 = m1[stop_idx]
    return {
        "omega": ss_.omega[stop_idx],
        "m0": s0,
        "m1": s1,
        "eb0": cost0 / pi[0],
        "eb1": cost1 / pi[1],
        "eb": cost0 + cost1,
    }


def exact_np_roc(omega, m0, m1, pfa_target):
    """Exact randomized Neyman-Pearson threshold on a discrete STOP-Ω
    distribution: returns (eta, p_rand, pfa, pd) with pfa == pfa_target.

    Normalization is per-hypothesis (R1.1-B fix of adcice/002.md): P_FA uses
    n0 = sum(m0), P_D uses n1 = sum(m1).  Identical to the old code for
    pi_0 = pi_1, correct for arbitrary priors.
    """
    n0 = float(m0.sum())
    n1 = float(m1.sum())
    u = np.unique(omega)
    p0_ge = np.array([m0[omega >= t].sum() / n0 for t in u])
    p0_gt = np.array([m0[omega > t].sum() / n0 for t in u])
    ok = p0_ge >= pfa_target
    j = int(np.flatnonzero(ok)[-1]) if ok.any() else 0
    eta = float(u[j])
    denom = p0_ge[j] - p0_gt[j]
    p_rand = 0.0 if denom <= 0 else min(1.0, max(0.0, (pfa_target - p0_gt[j]) / denom))
    pfa = p0_gt[j] + p_rand * denom
    pd = float((m1[omega > eta]).sum()) / n1 + p_rand * float((m1[omega == eta]).sum()) / n1
    return eta, p_rand, pfa, pd


def exact_evaluate(ss, policy, mu_M, mu_F, pi=(0.5, 0.5)):
    """Objective-consistent evaluation: natural terminal decision
    (declare H1 iff Omega > log(mu_F/mu_M)) plus the Lagrangian J."""
    sd = exact_stop_distribution(ss, policy, pi)
    eta_nat = float(np.log(mu_F / mu_M))
    pd = float((sd["m1"][sd["omega"] > eta_nat]).sum()) / pi[1]
    pfa = float((sd["m0"][sd["omega"] > eta_nat]).sum()) / pi[0]
    pm = 1.0 - pd
    j = sd["eb"] + mu_M * pm + mu_F * pfa
    return {
        "pd": pd, "pfa": pfa, "pm": pm,
        "eb": sd["eb"], "eb0": sd["eb0"], "eb1": sd["eb1"],
        "j": j, "eta_dec": eta_nat,
        "omega": sd["omega"], "m0": sd["m0"], "m1": sd["m1"],
    }


def exact_evaluate_threshold(ss, policy, eta_dec, pi=(0.5, 0.5)):
    """Diagnostic: evaluate a table policy with an external terminal threshold
    on the log-odds Ω at STOP (declare H1 iff Ω >= eta_dec).  This is the
    'terminal-statistic ROC diagnostic' of the audit — NOT the primary metric.
    """
    sd = exact_stop_distribution(ss, policy, pi)
    pd = float((sd["m1"][sd["omega"] >= eta_dec]).sum()) / pi[1]
    pfa = float((sd["m0"][sd["omega"] >= eta_dec]).sum()) / pi[0]
    return {"pd": pd, "pfa": pfa, "eb": sd["eb"], "eb0": sd["eb0"], "eb1": sd["eb1"]}


# --------------------------------------------------------------- G1a gates
def g1a_invariants(ss, mu_M=1.0, mu_F=1.0, pi=(0.5, 0.5)):
    """Independent transition-table invariants (audit §7):

      (1) normalization   sum_{m'} P(m'|m, H_h) == 1 for every transition;
      (2) posterior martingale  sum_{x'} P(x'|x,a) p(x') == p(x);
      (3) information monotonicity  E[R_stop(x')|x,a] <= R_stop(x).
    """
    C01 = float(mu_M) / pi[1]
    C10 = float(mu_F) / pi[0]
    max_norm = 0.0
    for i in range(ss.N):
        for z in range(BASE):
            for (_r2, _c, children) in ss.actions[i][z]:
                s1 = sum(float(np.exp(c[1])) for c in children)
                s0 = sum(float(np.exp(c[2])) for c in children)
                max_norm = max(max_norm, abs(s1 - 1.0), abs(s0 - 1.0))

    max_mar = 0.0
    max_mono = 0.0
    for i in range(ss.N):
        for z in range(BASE):
            rows = np.flatnonzero(ss.zcodes[:, i] == z)
            if rows.size == 0:
                continue
            om = ss.omega[rows]
            p = ss.p[rows]
            lp = ss.logp[rows]
            lq = ss.logq[rows]
            R = np.minimum(C01 * p, C10 * (1.0 - p))
            for (_r2, _c, children) in ss.actions[i][z]:
                if not children:
                    continue
                dl = np.array([c[1] - c[2] for c in children])        # child LLR inc
                l1 = np.array([c[1] for c in children])
                l0 = np.array([c[2] for c in children])
                a_ = lp[:, None] + l1[None, :]
                b_ = lq[:, None] + l0[None, :]
                mx = np.maximum(a_, b_)
                logw = mx + np.log1p(np.exp(-np.abs(a_ - b_)))
                w = np.exp(logw)
                p_child = 1.0 / (1.0 + np.exp(-(om[:, None] + dl[None, :])))
                mar = np.abs((w * p_child).sum(axis=1) - p).max()
                max_mar = max(max_mar, float(mar))
                Rc = np.minimum(C01 * p_child, C10 * (1.0 - p_child))
                mono = ((w * Rc).sum(axis=1) - R).max()
                max_mono = max(max_mono, float(mono))
    return {
        "norm_dev": max_norm,
        "martingale_dev": max_mar,
        "monotonicity_dev": max_mono,
        "passed": max_norm < 1e-9 and max_mar < 1e-9 and max_mono < 1e-9,
    }


# --------------------------------------------------------------- G1b gate
def g1b_check(ss, dp_solver, mu_M, mu_F, pi=(0.5, 0.5)):
    """Independent forward-propagation certification:

        J^{pi_DP}(x0) == V*(x0)   (to double precision)
    """
    res = exact_evaluate(ss, dp_solver.policy, mu_M, mu_F, pi)
    v0 = float(dp_solver.V[0])
    dev = abs(res["j"] - v0)
    return {
        "j_policy": res["j"],
        "v_star": v0,
        "abs_dev": dev,
        "rel_dev": dev / max(abs(v0), 1e-12),
        "passed": dev < 1e-8,
    }


# ------------------------------------------------------------- references
def exact_all_neighbor_roc(ss, pfa_target):
    """Exact ROC of the all-neighbor 4-bit statistic (P_D,max reference, §49).

    Omega = sum_i ell_i^4(m_i) over the 16^N terminal states; exact
    randomized Neyman-Pearson threshold on the discrete H0 distribution.
    """
    N = ss.N
    cells = [np.arange(16) for _ in range(N)]
    grid = np.array(np.meshgrid(*cells, indexing="ij")).reshape(N, -1)   # (N, 16^N)
    n = grid.shape[1]
    omega = np.zeros(n)
    logp1 = np.zeros(n)
    logp0 = np.zeros(n)
    for i in range(N):
        q = ss.quants[i]
        m = grid[i]
        omega += q.llr[4][m]
        logp1 += q.logP1[4][m]
        logp0 += q.logP0[4][m]
    m1 = np.exp(logp1 - logp1.max())
    m1 /= m1.sum()
    m0 = np.exp(logp0 - logp0.max())
    m0 /= m0.sum()
    eta, p_rand, pfa, pd = exact_np_roc(omega, m0, m1, pfa_target)
    return {"pd_max": pd, "pfa": pfa, "eta": eta}


def exact_pd_max(ss, pfa_target):
    return exact_all_neighbor_roc(ss, pfa_target)["pd_max"]
