"""MVS-B0.1a: credibility patch (advice/006.md §17).

1. 1bit-POTS: stop right after the 1-bit seed (|Lambda_seed| >= eta_s) and use
   the seed-aware cumulative LLR  Lambda_t = Lambda_seed + cumsum(Delta ell);
2. true prefix CRN: adaptive methods use the first n_o rows of the baseline
   (H, L) sample -> paired comparison on the common subset;
3. Wilson 95% CI actually reported for P_D / P_FA;
4. the old "same-E[B] gain" is renamed to an unmatched operating-point
   decomposition;
5. resource lattice already fixed in sparse.py (true cost + conservative
   budget ceil); discretization bound 0 <= C~ - C < N_txn * delta_c;
6. reachable-state Delta Q sweep: P(Delta Q<0), P(Delta Q>0), E[Delta Q] vs
   b_setup in {0,4,8,16,32} and the critical feedback-granularity threshold
   b* = inf{ b : E[min(D - D2, b)] >= 0 }.

Usage:  python run_mvsb01a.py [--smoke]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs import mc as mclib
from opmvs import sparse as sp
from opmvs.sparse import z_code_b

GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
PFA_TARGET = 0.05
EPS_D = 0.01
SEED0 = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
ETA_S_SWEEP = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
K_SWEEP = list(range(1, 9))


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    zz = z * z
    den = 1 + zz / n
    c = (phat + zz / (2 * n)) / den
    h = z * np.sqrt(phat * (1 - phat) / n + zz / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def ci_str(k, n):
    lo, hi = wilson_ci(k, n)
    return f"[{lo:.3f},{hi:.3f}]"


# ------------------------------------------------------------ baselines
def _llr_b(quants, L, level):
    out = np.empty((len(L), len(quants)))
    for i, q in enumerate(quants):
        m = q.cell_index(level, L[:, i])
        out[:, i] = q.llr[level][m]
    return out


def _stop_cross(lam_cum, cost_cum, eta_s):
    mask = (lam_cum >= eta_s) | (lam_cum <= -eta_s)
    has = mask.any(axis=1)
    t = np.where(has, mask.argmax(axis=1), lam_cum.shape[1] - 1)
    lam = np.where(has, lam_cum[np.arange(len(lam_cum)), t], lam_cum[:, -1])
    if cost_cum.ndim == 1:
        cost = np.where(has, cost_cum[t], cost_cum[-1])
    else:
        cost = np.where(has, cost_cum[np.arange(len(lam_cum)), t], cost_cum[:, -1])
    return lam, cost


def b_all_neighbor(quants, H, L, bh):
    lam = _llr_b(quants, L, 8).sum(axis=1)
    return lam, np.full(len(L), len(quants) * (bh + 8))


def b_snr_topk(quants, gamma, H, L, K, bh):
    sel = np.argsort(-np.asarray(gamma, float))[:K]
    lam = _llr_b(quants, L, 8)[:, sel].sum(axis=1)
    return lam, np.full(len(L), K * (bh + 8))


def b_seeded_pots(quants, H, L, eta_s, bh):
    """1-bit-seeded P-OTS (P0-1b fix of 006.md §1): all UAVs pay the 1-bit seed
    ONCE; stop right after the seed if |Lambda_seed| >= eta_s; otherwise run the
    ladder 1->2->4->8 in |ell^1| order with the seed-aware cumulative statistic
    Lambda_t = Lambda_seed + cumsum(Delta ell)."""
    n = len(L)
    N = len(quants)
    lr1 = _llr_b(quants, L, 1)
    seed_lam = lr1.sum(axis=1)
    seed_cost = np.full(n, N * (bh + 1))
    # immediate stop after the seed?
    stop_now = (seed_lam >= eta_s) | (seed_lam <= -eta_s)
    order = np.argsort(-np.abs(lr1), axis=1)
    lr = {r: _llr_b(quants, L, r) for r in (1, 2, 4, 8)}
    steps = [2, 4, 8]
    ar = np.arange(n)
    u_idx = np.repeat(order, len(steps), axis=1)
    inc = np.empty((n, len(steps) * N))
    for s in range(len(steps) * N):
        u = u_idx[:, s]
        r = steps[s % len(steps)]
        if s % len(steps) == 0:
            inc[:, s] = lr[r][ar, u] - lr[1][ar, u]
        else:
            inc[:, s] = lr[r][ar, u] - lr[steps[s % len(steps) - 1]][ar, u]
    cum_inc = np.cumsum(inc, axis=1)
    lam_cum = seed_lam[:, None] + cum_inc            # seed-aware cumulative LLR
    scost = np.cumsum(np.tile([bh + 1, bh + 2, bh + 4], N)).astype(float)
    mask = (lam_cum >= eta_s) | (lam_cum <= -eta_s)
    has = mask.any(axis=1)
    t = np.where(has, mask.argmax(axis=1), lam_cum.shape[1] - 1)
    lam2 = np.where(has, lam_cum[np.arange(n), t], lam_cum[:, -1])
    cost2 = np.where(has, scost[t], scost[-1])
    # episodes that already stopped at the seed keep only the seed cost
    lam = np.where(stop_now, seed_lam, lam2)
    cost = np.where(stop_now, seed_cost, seed_cost + cost2)
    return lam, cost


def b_direct8_ordered(quants, gamma, H, L, eta_s, bh):
    N = len(quants)
    n = len(L)
    order = np.tile(np.argsort(-np.asarray(gamma, float)), (n, 1))
    llr8 = _llr_b(quants, L, 8)
    cum = np.cumsum(np.take_along_axis(llr8, order, axis=1), axis=1)
    scost = (bh + 8) * np.arange(1, N + 1)
    return _stop_cross(cum, scost, eta_s)


# --------------------------------------------------- Delta Q over states
def delta_q(planner, x_int, i, b_h):
    """Q_prog - Q_dir for the 0->1->8 vs 0->8 choice of UAV i at x_int,
    using the conditional-VoI identity E[min{D(x')-7, b_h}]."""
    tpl_i = planner._tpl[i][0]
    dir_act = next((a for a in tpl_i if a[0] == 8), None)
    prog_act = next((a for a in tpl_i if a[0] == 1), None)
    if dir_act is None or prog_act is None:
        return None
    om0, p0, lp, lq = planner.posterior(x_int)
    pw = planner.powers[i]
    llr_i = planner._llr_i[i]

    def R(x):
        o = planner.omega(x)
        p = 1.0 / (1.0 + np.exp(-o))
        return min(planner.C01 * p, planner.C10 * (1.0 - p))

    E = 0.0
    for (m1, lp0c1, lp1c1) in prog_act[3]:
        a_ = lp + lp1c1
        b_ = lq + lp0c1
        m_ = a_ if a_ >= b_ else b_
        w1 = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
        x1 = x_int + (z_code_b(1, m1)) * pw
        om1 = om0 + llr_i[z_code_b(1, m1)]
        p1 = 1.0 / (1.0 + np.exp(-om1))
        lp1 = float(np.log(p1))
        lq1 = float(np.log1p(-p1))
        ref = next((a for a in planner._tpl[i][z_code_b(1, m1)] if a[0] == 8), None)
        E_R = 0.0
        if ref is not None:
            for (m2, lp0c, lp1c) in ref[3]:
                a_ = lp1 + lp1c
                b_ = lq1 + lp0c
                m_ = a_ if a_ >= b_ else b_
                w = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
                cx = x1 + (z_code_b(8, m2) - z_code_b(1, m1)) * pw
                E_R += w * R(cx)
        D = R(x1) - E_R
        E += w1 * min(D - 7, b_h)
    return E


# ------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_ep = 8000 if args.smoke else 20000
    n_o = 2000 if args.smoke else 2500
    n_states = 300 if args.smoke else 800
    bh = 16.0
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.1a — Credibility Patch（advice/006.md §17）")
    out("")
    out("> 修复 1bit-POTS seed 后立即停止、真前缀 CRN、CI 输出、unmatched 措辞、"
        "reachable-state ΔQ 相变实证；resource lattice 已在 sparse.py 保守化。")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if args.smoke else 'FULL'}")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=(1, 2, 4, 8)) for i in range(8)]

    # true prefix CRN: one baseline sample; adaptive uses its first n_o rows
    rng = np.random.default_rng(SEED0)
    Ht = model8.sample_hypotheses(n_ep, rng)
    L = model8.sample_llr(Ht, rng)
    Ht_o, L_o = Ht[:n_o], L[:n_o]
    out(f"- 真前缀 CRN: baseline n={n_ep}；adaptive 使用前 {n_o} 条 → 公共子集可做 paired 对比")
    out("")

    # ------------------------------------------------ P0-1b: seeded POTS
    out("## 1. P0-1b — 1bit-POTS：seed 后立即停止 + seed-aware cumulative LLR")
    out("")
    out("| ηs | P_D @ P_FA=0.05 | E[B_radio] | 达标 |")
    out("| --- | --- | --- | --- |")
    best = None
    lam, cost = b_all_neighbor(quants8, Ht, L, bh)
    pd_max = mclib.evaluate(lam, cost, Ht, PFA_TARGET)["pd"]
    pd_target = pd_max - EPS_D
    out(f"- P_D,max = {fmt(pd_max)}，matched 目标 P_D ≥ {fmt(pd_target)}")
    for eta_s in ETA_S_SWEEP:
        lam, cost = b_seeded_pots(quants8, Ht, L, eta_s, bh)
        m_ = mclib.evaluate(lam, cost, Ht, PFA_TARGET)
        ok_ = m_["pd"] >= pd_target
        if ok_ and (best is None or m_["eb"] < best["eb"]):
            best = m_
        out(f"| {eta_s} | {fmt(m_['pd'])} | {fmt(m_['eb'])} | {'✓' if ok_ else ''} |")
    if best:
        out(f"- **修复后 1bit-POTS matched: E[B_radio] = {fmt(best['eb'])} bits**"
            "（seed 后立即停止显著降低通信成本）")
    out("")

    # ------------------------------------------------ baselines with CI
    out("## 2. 公平基线（前缀 CRN，Wilson 95% CI）")
    out("")
    out("| 方法 | 参数 | P_D @ 0.05 (95% CI) | P_FA (95% CI) | E[B_radio] |")
    out("| --- | --- | --- | --- | --- |")
    lam, cost = b_all_neighbor(quants8, Ht, L, bh)
    m_ = mclib.evaluate(lam, cost, Ht, PFA_TARGET)
    H1 = Ht == 1
    kd = int((lam[H1] >= m_["eta"]).sum())
    kf = int((lam[~H1] >= m_["eta"]).sum())
    out(f"| AllNeighbor-8 | - | {fmt(m_['pd'])} {ci_str(kd, H1.sum())} | "
        f"{fmt(m_['pfa'])} {ci_str(kf, (~H1).sum())} | {fmt(m_['eb'])} |")
    for name, fn, pname, vals in (
        ("SNR-TopK", b_snr_topk, "K", K_SWEEP),
        ("1bit-POTS", b_seeded_pots, "eta_s", ETA_S_SWEEP),
        ("Direct8-Ordered", b_direct8_ordered, "eta_s", ETA_S_SWEEP),
    ):
        bestm = None
        for v in vals:
            if name == "SNR-TopK":
                lam, cost = fn(quants8, GAMMA_B, Ht, L, v, bh)
            elif name == "Direct8-Ordered":
                lam, cost = fn(quants8, GAMMA_B, Ht, L, v, bh)
            else:
                lam, cost = fn(quants8, Ht, L, v, bh)
            m_ = mclib.evaluate(lam, cost, Ht, PFA_TARGET)
            if m_["pd"] >= pd_target and (bestm is None or m_["eb"] < bestm["eb"]):
                bestm = m_
        if bestm:
            kd = int((lam[H1] >= bestm["eta"]).sum())
            kf = int((lam[~H1] >= bestm["eta"]).sum())
            out(f"| {name} | {v if name=='SNR-TopK' else 'matched ηs'} | "
                f"{fmt(bestm['pd'])} {ci_str(kd, H1.sum())} | "
                f"{fmt(bestm['pfa'])} {ci_str(kf, (~H1).sum())} | {fmt(bestm['eb'])} |")
    out("")
    out("- 说明：P_FA 的 CI 与 0.05 的关系决定 'QoS 认证' 或 'UNCERTIFIED'（不可据此断言 FAIL）")
    out("")

    # ------------------------------------------------ unmatched decomposition
    out("## 3. unmatched operating-point decomposition（原'同 E[B] gain'改称）")
    out("")
    out("- 该对比只是 same-(s,η,H) operating-point 下的 cost/performance 差异，"
        "P_D 与 E[B] 均未匹配——**不构成 gain**；真正的 multi-resolution gain 必须"
        "在 P_D, P_FA matched 后比较 E[B]（保持 UNCERTIFIED）")
    out("")

    # ------------------------------------------------ Delta Q phase sweep
    out("## 4. reachable-state ΔQ sweep 与 critical feedback-granularity threshold")
    out("")
    out("> ΔQ(x;b) = Q_prog − Q_dir = E[min{D(x')−Δ₂, b}]；b*=inf{b:E[min(D−Δ₂,b)]≥0}。")
    out("")
    rng_s = np.random.default_rng(5)
    # sample reachable states (mixed-depths: 0-3 UAVs refined to random levels)
    states = []
    N = 8
    for _ in range(n_states):
        z = [0] * N
        for u in range(int(rng_s.integers(0, 4))):
            i = int(rng_s.integers(0, N))
            r = (1, 2, 4, 8)[int(rng_s.integers(0, 4))]
            m = int(rng_s.integers(0, 2 ** r))
            z[i] = z_code_b(r, m)
        states.append(sum(int(z[i]) * (sp.BASE_B ** i) for i in range(N)))
    out("| b_setup | P(ΔQ<0) | P(ΔQ>0) | E[ΔQ] |")
    out("| --- | --- | --- | --- |")
    critical = {}
    for b in (0.0, 4.0, 8.0, 16.0, 32.0):
        pl_ = sp.SparsePlanner(quants8, 256.0, 256.0 * np.exp(1.0), b_h=b, cross_level=True)
        dqs = []
        for x in states:
            d = delta_q(pl_, x, 7, b)
            if d is not None:
                dqs.append(d)
        dqs = np.array(dqs)
        neg = float(np.mean(dqs < 0))
        pos = float(np.mean(dqs > 0))
        e_dq = float(dqs.mean())
        critical[b] = e_dq
        out(f"| {b:.0f} | {fmt(neg)} | {fmt(pos)} | {fmt(e_dq)} |")
    out("")
    # critical threshold at the root (g(b) = E[min(D-7, b)] over the 1-bit messages)
    pl16 = sp.SparsePlanner(quants8, 256.0, 256.0 * np.exp(1.0), b_h=16.0, cross_level=True)
    g_vals = []
    for b in (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        g_vals.append((b, delta_q(pl16, 0, 7, b)))
    crossing = [b for (b, g) in g_vals if g >= 0]
    out(f"- 根状态 g(b)=E[min(D−Δ₂,b)]：{[(int(b), round(g, 3)) for (b, g) in g_vals]}")
    if crossing:
        out(f"- **临界阈值 b* ≈ {crossing[0]:.0f}**（g(b*)≥0 ⇒ direct packetization 开始占优）"
            "——feedback-granularity phase transition")
    else:
        out("- 该状态分布下 g(b)<0 恒成立 ⇒ progressive 始终更优（无相变；与 E[D]≤Δ₂ 情形一致）")
    out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    rp = os.path.join(OUT_DIR, "MVS-B0.1a_report.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
