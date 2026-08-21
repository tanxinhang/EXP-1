"""Certified constrained Markov decision process (CMDP) via column generation
(R2.1-G0 of adcice/003.md).

Restricted master problem (RMP) over the currently-generated deterministic
policies k = 1..K:

    min_w  sum_k w_k B_k
    s.t.   sum_k w_k P_FA,k <= alpha,   sum_k w_k P_M,k <= beta,
           sum_k w_k = 1,   w_k >= 0.

Duals (scipy HiGHS convention): lambda_F >= 0, lambda_M >= 0, nu (free).
Strong duality:  nu = -obj - lambda_F*alpha - lambda_M*beta.

Pricing problem (exact, via ExactDP — the Lagrangian we already solve):

    min_pi [ B_pi + lambda_M P_M,pi + lambda_F P_FA,pi ]
        =  min_pi J(pi)  at  (mu_M, mu_F) = (lambda_M, lambda_F).

Reduced cost of a candidate policy:  r(pi) = J(pi) + nu  (scipy convention).
If min r < -eps, add the pricing policy's column and re-solve; otherwise the
RMP optimum is the *global* CMDP optimum — certified by exact pricing over
ALL deterministic policies (003.md §2).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from . import eval_exact as ee
from .dp import ExactDP


def master_lp(columns, alpha, beta):
    """Solve the restricted master LP and its dual.

    columns: list of dicts with keys B, pfa, pm.
    Returns (w, obj, lam_F, lam_M, nu).

    The duals are obtained by solving the *dual LP* directly (variables are
    the dual LP's primal variables — no solver marginal sign ambiguity):

        max -lam_F*alpha - lam_M*beta - nu
        s.t.  B_k + lam_F*PFA_k + lam_M*PM_k + nu >= 0  (all k)
              lam_F, lam_M >= 0, nu free
    """
    B = np.array([c["B"] for c in columns])
    PFA = np.array([c["pfa"] for c in columns])
    PM = np.array([c["pm"] for c in columns])
    # ---------------- primal ----------------
    A_ub = np.vstack([PFA, PM])
    b_ub = np.array([alpha, beta])
    A_eq = np.ones((1, len(columns)))
    b_eq = np.array([1.0])
    res = linprog(B, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * len(columns), method="highs")
    if not res.success:
        raise RuntimeError(f"master LP failed: {res.message}")
    obj = float(res.fun)
    # ---------------- dual ----------------
    # variables y = [lam_F, lam_M, nu];  minimize alpha*lam_F + beta*lam_M + nu
    # s.t.  -PFA_k*lam_F - PM_k*lam_M - nu <= B_k,  lam_F,lam_M >= 0
    c_d = np.array([alpha, beta, 1.0])
    A_d = np.column_stack([-PFA, -PM, -np.ones(len(columns))])
    b_d = B
    bounds_d = [(0, None), (0, None), (None, None)]
    res_d = linprog(c_d, A_ub=A_d, b_ub=b_d, bounds=bounds_d, method="highs")
    if not res_d.success:
        raise RuntimeError(f"dual LP failed: {res_d.message}")
    lam_F = float(res_d.x[0])
    lam_M = float(res_d.x[1])
    nu = float(res_d.x[2])
    # strong duality: obj == -(alpha*lam_F + beta*lam_M + nu)
    dual_obj = -(alpha * lam_F + beta * lam_M + nu)
    assert abs(obj - dual_obj) < 1e-6 * max(1.0, abs(obj)), \
        f"strong duality violated: primal={obj} dual={dual_obj}"
    return {
        "w": res.x,
        "obj": obj,
        "lam_F": lam_F,
        "lam_M": lam_M,
        "nu": nu,
    }


def column_generation(ss, initial_columns, alpha, beta, eps=1e-8, max_iter=60,
                      verbose=True):
    """Column generation with exact ExactDP pricing.

    Returns dict with the certified global CMDP optimum.
    """
    columns = [dict(c) for c in initial_columns]
    log = []
    B_hist = []
    for it in range(max_iter):
        m = master_lp(columns, alpha, beta)
        # pricing: ExactDP at (mu_M, mu_F) = (lam_M, lam_F)
        d = ExactDP(ss)
        V, pol = d.solve(m["lam_M"], m["lam_F"])
        ev = ee.exact_evaluate(ss, pol, m["lam_M"], m["lam_F"])
        j_pricing = ev["j"]
        r_min = j_pricing + m["nu"]                   # reduced cost
        log.append({"it": it, "obj": m["obj"], "lam_F": m["lam_F"], "lam_M": m["lam_M"],
                    "nu": m["nu"], "j_pricing": j_pricing, "r_min": r_min,
                    "n_cols": len(columns)})
        B_hist.append(m["obj"])
        if verbose:
            print(f"  [CG {it}] obj={m['obj']:.6f} n_cols={len(columns)} "
                  f"lam=({m['lam_F']:.4f},{m['lam_M']:.4f}) nu={m['nu']:.4f} "
                  f"r_min={r_min:+.3e}")
        # terminate when the reduced cost is above the tolerance OR the
        # pricing policy's column is already present (float-noise floor of
        # the ExactDP chain ~1e-7 makes r plateau at ~-9e-8; a repeated
        # column proves no new improving column exists).
        dup = any(abs(ev["eb"] - c["B"]) < 1e-9 and abs(ev["pfa"] - c["pfa"]) < 1e-9
                  and abs(ev["pm"] - c["pm"]) < 1e-9 for c in columns)
        if r_min >= -eps or dup:
            break
        # add the pricing policy as a new column
        columns.append({"B": ev["eb"], "pfa": ev["pfa"], "pm": ev["pm"],
                        "policy": pol, "lam": (m["lam_M"], m["lam_F"])})
    else:
        raise RuntimeError("column generation did not converge")
    # final certified solution
    m_final = master_lp(columns, alpha, beta)
    active = [(k, float(m_final["w"][k])) for k in range(len(columns))
              if m_final["w"][k] > 1e-9]
    return {
        "b_cmdp": m_final["obj"],
        "w": m_final["w"],
        "active": active,
        "n_columns": len(columns),
        "n_iterations": len(log),
        "r_final": r_min,
        "log": log,
        "columns": columns,
    }
