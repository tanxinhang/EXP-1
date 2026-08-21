"""Vectorized Monte Carlo episode simulation and metric evaluation (§33, §57).

Episodes are simulated in parallel as numpy arrays: for a table policy
(state index -> action code) we advance every not-yet-stopped episode by one
action per iteration, applying the correct quantizer cell / message-LLR
replacement update (Section 9 invariant: replace, never add).

All methods (adaptive policies and baselines) are evaluated on the *same*
(H, L) episode samples per seed, so method-to-method comparisons share the
same randomness.  The decision statistic recorded is the log posterior odds
Omega at stopping time; for every method we then enforce P_FA = 0.05 by
thresholding Omega (Neyman-Pearson style), and report P_D, P_FA, E[B].
"""
from __future__ import annotations

import numpy as np

from .state import R_LEVELS, z_code_vec, action_code, action_decode

MAX_STEPS = 24


# ---------------------------------------------------------------- table MC
def simulate_table_policy(ss, policy, H, L, max_steps=MAX_STEPS):
    """Simulate a table policy (state idx -> action code; 0 = STOP).

    Returns (lam, cost, z_final, n_steps):
      lam    : (n,) log posterior odds Omega at stop
      cost   : (n,) total payload bits consumed
      z_final: (n, N) final evidence z codes per UAV
      n_steps: (n,) number of actions taken
    """
    n = len(H)
    N = ss.N
    z = np.zeros((n, N), dtype=np.int16)
    lam = np.full(n, ss.prior_log_odds)
    cost = np.zeros(n)
    n_steps = np.zeros(n, dtype=np.int32)
    done = np.zeros(n, dtype=bool)
    quants = ss.quants
    r_of = ss.r_of
    llr_tab = ss.llr_tab

    for _ in range(max_steps):
        active = ~done
        if not active.any():
            break
        idx = ss.encode(z[active])
        a = policy[idx].astype(np.int16)
        stop = a == 0
        for i in range(N):
            for r2 in R_LEVELS:
                code = action_code(i, r2)
                m_act = a == code
                if not m_act.any():
                    continue
                sub = np.flatnonzero(active)[m_act]
                m2 = quants[i].cell_index(r2, L[sub, i])
                znew = z_code_vec(r2, m2)
                r_old = r_of[i, z[sub, i]]
                lam[sub] += quants[i].llr[r2][m2] - llr_tab[i, z[sub, i]]
                cost[sub] += r2 - r_old.astype(np.float64)
                z[sub, i] = znew
        n_steps[active] += (~stop).astype(np.int32)   # count actions, not stop-checks
        done[active] = stop

    return lam, cost, z, n_steps


# ---------------------------------------------------------------- sampling
def sample_episodes(model, n, seed):
    """Sample (H, L) once per seed; reused by every method for fair MC."""
    rng = np.random.default_rng(seed)
    H = model.sample_hypotheses(n, rng)
    L = model.sample_llr(H, rng)
    return H, L


# --------------------------------------------------------------- evaluation
def _randomized_threshold(lam0, lam1, pfa_target):
    """Randomized Neyman-Pearson thresholding for (possibly discrete)
    decision statistics: finds eta with P0(lam>eta) <= pfa <= P0(lam>=eta)
    and randomizes ties at eta so that P_FA == pfa_target exactly.

    Returns (eta, p_rand, pfa, pd).
    """
    n0 = len(lam0)
    if n0 == 0:
        return 0.0, 0.0, float("nan"), float("nan")
    s0 = np.sort(lam0)
    u = np.unique(s0)
    ge = n0 - np.searchsorted(s0, u, side="left")      # count >= u[j]
    gt = n0 - np.searchsorted(s0, u, side="right")     # count >  u[j]
    p_ge = ge / n0
    p_gt = gt / n0
    ok = p_ge >= pfa_target
    if not ok.any():
        j = 0
    else:
        j = int(np.flatnonzero(ok)[-1])                # last eta with P0(>=) >= target
    eta = float(u[j])
    p0_ge = float(p_ge[j])
    p0_gt = float(p_gt[j])
    denom = p0_ge - p0_gt
    p_rand = 0.0 if denom <= 0 else min(1.0, max(0.0, (pfa_target - p0_gt) / denom))
    pfa = p0_gt + p_rand * denom
    if len(lam1):
        pd = float(np.mean(lam1 > eta)) + p_rand * float(np.mean(lam1 == eta))
    else:
        pd = float("nan")
    return eta, p_rand, pfa, pd


def evaluate(lam, cost, H, pfa_target=0.05, n_steps=None):
    """Randomized-NP evaluation of a policy: P_FA == pfa_target exactly,
    P_D at that threshold, E[B], E[N_query].  (SystemModel §57-§58.)

    n_steps (optional): per-episode action counts -> E[N_query].
    """
    H = np.asarray(H) == 1
    lam0 = lam[~H]
    lam1 = lam[H]
    eta, p_rand, pfa, pd = _randomized_threshold(lam0, lam1, pfa_target)
    out = {
        "pd": pd,
        "pfa": pfa,
        "eb": float(cost.mean()),
        "eta": eta,
        "e_nq": float(np.mean(n_steps)) if n_steps is not None else float("nan"),
    }
    return out


def summarize_runs(runs, fields=("pd", "pfa", "eb")):
    """Aggregate per-seed metric dicts -> {field: (mean, std)}."""
    out = {}
    for f in fields:
        vals = np.array([r[f] for r in runs], dtype=float)
        out[f] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return out


def precision_audit(z_final, ss):
    """P(r_i* = r) per UAV from final z codes; plus P(r* = rmax | r* > 0)."""
    N = ss.N
    n = z_final.shape[0]
    r_final = np.empty_like(z_final)
    for j in range(N):
        r_final[:, j] = ss.r_of[j, z_final[:, j]]
    rows = []
    for j in range(N):
        counts = {r: float(np.mean(r_final[:, j] == r)) for r in (0, 1, 2, 4)}
        nz = r_final[:, j] > 0
        p_full_given_reported = float(np.mean(r_final[nz, j] == 4)) if nz.any() else float("nan")
        rows.append({"uav": j, "p0": counts[0], "p1": counts[1], "p2": counts[2],
                     "p4": counts[4], "p4_given_reported": p_full_given_reported})
    return rows
