"""Gate checks G0/G1/G2 (SystemModel §34-§36, §66).

G0 Statistical sanity: raw ROC, message PMF normalization, nested
   consistency, per-bit P_D and quantizer loss Delta_Q, log-domain stress.
G1 Exact DP: acyclic backward DP with Bellman residual at double precision.
G2 Solver quality: cost gap of O-PEF-1 / O-PEF-2E relative to the DP oracle.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .fusion import log_sigmoid, log_one_minus_sigmoid, logsumexp2, softplus


# ------------------------------------------------------------------- G0
def g0_raw_roc(model, n=200000, seed=0, pfa_target=0.05, tol=5e-3):
    """G0.1: analytical raw P_D vs Monte Carlo."""
    from .mc import sample_episodes, evaluate
    H, L = sample_episodes(model, n, seed)
    lam = model.prior_log_odds + L.sum(axis=1)
    m = evaluate(lam, np.zeros(n), H, pfa_target)
    ana = model.raw_fusion_pd(pfa_target)
    return {
        "pd_analytical": ana,
        "pd_mc": m["pd"],
        "pfa_mc": m["pfa"],
        "delta": abs(m["pd"] - ana),
        "passed": abs(m["pd"] - ana) < tol,
    }


def g0_pmf_normalization(model, quants, rtol=1e-10):
    """G0.2: sum_m P(M_i^(r)=m | H_h) == 1."""
    ok = True
    details = []
    for i, q in enumerate(quants):
        for r in q.levels:
            for h in (0, 1):
                lp = q.logP1[r] if h else q.logP0[r]
                s = float(np.exp(lp).sum())
                ok &= abs(s - 1.0) <= rtol
                details.append((i, r, h, s))
    return {"passed": ok, "max_dev": max(abs(d[3] - 1.0) for d in details)}


def g0_nested_consistency(quants, rtol=1e-9):
    """G0.3: P(m|H_h) == sum_{m' in children} P(m'|H_h)."""
    ok = True
    max_dev = 0.0
    for i, q in enumerate(quants):
        for r in q.levels:
            if r >= q.r_max:
                continue
            for m in range(2 ** r):
                for r2 in q.levels:
                    if r2 <= r:
                        continue
                    for h in (0, 1):
                        lp = q.logP1[r] if h else q.logP0[r]
                        lp2 = q.logP1[r2] if h else q.logP0[r2]
                        ch = q.desc_cells(r, m, r2)
                        dev = abs(float(np.exp(lp2[ch]).sum()) - np.exp(lp[m]))
                        max_dev = max(max_dev, dev)
                        ok &= dev <= rtol
    return {"passed": ok, "max_dev": max_dev}


def g0_per_bit_reference(model, ss, n=200000, seed=0, pfa_target=0.05):
    """G0.4: all-node P_D at 1/2/4 bit and quantizer losses Delta_Q."""
    from .mc import sample_episodes, evaluate
    from .baselines import _llr_matrix
    H, L = sample_episodes(model, n, seed)
    raw = model.raw_fusion_pd(pfa_target)
    rows = {}
    for r in (1, 2, 4):
        lam = ss.prior_log_odds + _llr_matrix(ss, L, r).sum(axis=1)
        m = evaluate(lam, np.full(n, r * ss.N), H, pfa_target)
        rows[r] = {"pd": m["pd"], "pfa": m["pfa"], "delta_q": raw - m["pd"]}
    return {"raw_pd": raw, "rows": rows,
            "passed": rows[4]["delta_q"] > 0 and rows[4]["delta_q"] < 0.15}


def g0_log_domain_stress():
    """G0.5: Omega in [-100, 100] — no NaN / overflow / illegal probability."""
    om = np.linspace(-100.0, 100.0, 20001)
    lp = log_sigmoid(om)
    lq = log_one_minus_sigmoid(om)
    p = np.exp(lp)
    q = np.exp(lq)
    ok_nan = np.all(np.isfinite(lp)) and np.all(np.isfinite(lq))
    ok_range = np.all((p >= 0.0) & (p <= 1.0)) and np.all((q >= 0.0) & (q <= 1.0))
    ok_sum = np.all(np.abs(p + q - 1.0) < 1e-12)
    # extreme logsumexp arguments
    a = np.array([1000.0, -1000.0, 0.0, -40.0])
    b = np.array([-1000.0, 1000.0, 0.0, -45.0])
    lse = logsumexp2(a, b)
    ok_lse = np.all(np.isfinite(lse))
    expected = np.logaddexp(a, b)                      # reference (stable)
    ok_lse_close = np.allclose(lse, expected, rtol=1e-12)
    # softplus extreme
    sp = softplus(np.array([-1000.0, 1000.0]))
    ok_sp = np.allclose(sp, [0.0, 1000.0], atol=1e-12)
    return {"passed": ok_nan and ok_range and ok_sum and ok_lse and ok_lse_close and ok_sp}


# ------------------------------------------------------------------- G1
def g1_bellman_residual(dp_solver):
    """G1: max_x |V(x) - T V(x)| for the exact DAG-DP solve."""
    max_res, mean_res = dp_solver.bellman_residual()
    return {"max_residual": max_res, "mean_residual": mean_res,
            "passed": max_res < 1e-8}


# ------------------------------------------------------------------- G2
def g2_solver_gap(eb_dp, eb_opef1, eb_opef2):
    """G2: (C_OPEF - C_DP) / C_DP at a matched operating point."""
    gap1 = (eb_opef1 - eb_dp) / eb_dp if eb_dp > 0 else float("inf")
    gap2 = (eb_opef2 - eb_dp) / eb_dp if eb_dp > 0 else float("inf")
    return {
        "gap_opef1": gap1,
        "gap_opef2": gap2,
        "passed_opef2_10pct": gap2 <= 0.10,
        "passed_opef2_20pct": gap2 <= 0.20,
    }
