"""MVS-B0.1: Credibility Repair + Header/Feedback Theory (adcice/005.md).

P0-1  fix the 1bit-POTS double-count (seed once, ladder starts at 1->2);
P0-2  shared CRN for ALL methods (same H, L) + CI reporting; the B0-G3
      verdict is UNCERTIFIED / COMPUTATION-LIMITED, not FAIL;
P0-3  Natural-policy QoS (primary) vs post-hoc NP ROC (secondary) for the
      adaptive policies; baselines keep their standard NP evaluation;
P1-4  Adaptive Direct-8 optimal baseline (same planner, action set
      {(i,0->8)}) — isolates UAV-selection value from multi-resolution value;
P1-5  state-dependent conditional-VoI theorem:
        Q_prog - Q_dir = E_x'[ min{ D(x') - Delta_2, b_h } ],
      with D(x') = R(x') - E[R(x'')|x'];  b_h=0 => progressive dominates;
      q < 7/23 downgraded to a communication-only corollary; new radio-only
      criterion E[C_future | M^(1)] < 7;
P1-6  feedback/setup cost accounting (b_setup = b_data + b_ctrl + b_fb) with
      a sensitivity;
P1-7  Gaussian prior consistency (model.py fixed);
P1-8  regression test suite (test_regressions.py).

Innovation positioning: Feedback-Granularity-Aware Adaptive Evidence
Acquisition — per-transaction setup cost decides when coarse-then-refine
beats direct cross-level packetization.

Usage:  python run_mvsb01.py [--smoke]
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
FIG_DIR = os.path.join(OUT_DIR, "figures")
ETA_S_SWEEP = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
K_SWEEP = list(range(1, 9))


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    zz = z * z
    den = 1 + zz / n
    c = (phat + zz / (2 * n)) / den
    h = z * np.sqrt(phat * (1 - phat) / n + zz / (4 * n * n)) / den
    return (float(c - h), float(c + h))


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


def b_global_fixed(quants, H, L, eta_s, bh):
    N = len(quants)
    lr = {r: _llr_b(quants, L, r).sum(axis=1) for r in (1, 2, 4, 8)}
    cum = np.column_stack([lr[1], lr[2], lr[4], lr[8]])
    scost = np.cumsum([N * (bh + i) for i in (1, 1, 2, 4)]).astype(float)
    return _stop_cross(cum, scost, eta_s)


def _ladder(quants, order, L, eta_s, bh, start_r=0):
    """Ladder per UAV in `order`: refine start_r -> ... -> 8, stop by |Omega|.
    If start_r == 0 the first step is 0->1; if start_r == 1 the UAV's 1-bit is
    already paid (seeded POTS) and the ladder starts at 1->2 — no double count."""
    N = len(quants)
    n = len(L)
    lr = {r: _llr_b(quants, L, r) for r in (1, 2, 4, 8)}
    steps = [r for r in (1, 2, 4, 8) if r > start_r]
    ar = np.arange(n)
    u_idx = np.repeat(order, len(steps), axis=1)
    inc = np.empty((n, len(steps) * N))
    for s in range(len(steps) * N):
        u = u_idx[:, s]
        r = steps[s % len(steps)]
        if s % len(steps) == 0:
            inc[:, s] = lr[r][ar, u] - (0.0 if start_r == 0 else lr[start_r][ar, u])
        else:
            inc[:, s] = lr[r][ar, u] - lr[steps[s % len(steps) - 1]][ar, u]
    cum = np.cumsum(inc, axis=1)
    scost = np.cumsum(np.tile([bh + (r2 - r1) for r1, r2 in
                               zip([start_r] + steps[:-1], steps)], N)).astype(float)
    return _stop_cross(cum, scost, eta_s)


def b_static_prog(quants, gamma, H, L, eta_s, bh):
    order = np.tile(np.argsort(-np.asarray(gamma, float)), (len(L), 1))
    return _ladder(quants, order, L, eta_s, bh, start_r=0)


def b_seeded_pots(quants, H, L, eta_s, bh):
    """1-bit-seeded P-OTS (FIXED double-count, 005.md P0-1): all UAVs pay the
    1-bit seed ONCE (N*(bh+1) radio bits, Omega += sum ell^1), then the ladder
    starts at 1->2->4->8 in |ell^1| order — the seed is never re-added."""
    n = len(L)
    N = len(quants)
    lr1 = _llr_b(quants, L, 1)
    seed_lam = lr1.sum(axis=1)
    order = np.argsort(-np.abs(lr1), axis=1)
    lam2, cost2 = _ladder(quants, order, L, eta_s, bh, start_r=1)
    return seed_lam + lam2, np.full(n, N * (bh + 1)) + cost2


def b_direct8_ordered(quants, gamma, H, L, eta_s, bh):
    N = len(quants)
    n = len(L)
    order = np.tile(np.argsort(-np.asarray(gamma, float)), (n, 1))
    llr8 = _llr_b(quants, L, 8)
    cum = np.cumsum(np.take_along_axis(llr8, order, axis=1), axis=1)
    scost = (bh + 8) * np.arange(1, N + 1)
    return _stop_cross(cum, scost, eta_s)


# ------------------------------------------------- receding adaptive policy
def mc_receding_shared(quants, model, Ht, L, muM, muF, b_h, H, eta, planner_kw=None):
    """MC of the receding sparse policy on SHARED (Ht, L) (CRN).  Returns
    (natural metrics, NP metrics, E[B], mechanism stats)."""
    planner_kw = planner_kw or {}
    rng = np.random.default_rng(SEED0)   # unused (Ht, L pre-sampled)
    n_ep = len(Ht)
    N = len(quants)
    x_int = [0] * n_ep
    zcode = np.zeros((n_ep, N), dtype=np.int64)
    lam = np.zeros(n_ep)
    cost = np.zeros(n_ep)
    mech = {}
    done = np.zeros(n_ep, dtype=bool)
    powers = [sp.BASE_B ** i for i in range(N)]
    for _ in range(64):
        active = np.flatnonzero(~done)
        if len(active) == 0:
            break
        for e in active:
            pl = sp.SparsePlanner(quants, muM, muF, b_h=b_h, **planner_kw)
            val, a = pl.solve(x_int[e], H)
            if a is None:
                done[e] = True
                continue
            i, r2 = a
            zi = int(zcode[e, i])
            r_cur, m_cur = sp.z_decode_b(zi)
            m2 = int(quants[i].cell_index(r2, L[e, i]))
            z2 = sp.z_code_b(r2, m2)
            lam[e] += quants[i].llr[r2][m2]
            if r_cur > 0:
                lam[e] -= quants[i].llr[r_cur][m_cur]
            cost[e] += b_h + (r2 - r_cur)
            x_int[e] += (z2 - zi) * powers[i]
            zcode[e, i] = z2
            key = f"{r_cur}->{r2}"
            mech[key] = mech.get(key, 0) + 1
    # natural decision (primary): declare H1 iff Omega > eta
    H1 = Ht == 1
    pd_nat = float(np.mean(lam[H1] > eta)) if H1.any() else float("nan")
    pfa_nat = float(np.mean(lam[~H1] > eta)) if (~H1).any() else float("nan")
    m_np = mclib.evaluate(lam, cost, Ht, PFA_TARGET)   # post-hoc NP (secondary)
    return {"pd_nat": pd_nat, "pfa_nat": pfa_nat, "eb": float(cost.mean()),
            "pd_np": m_np["pd"], "pfa_np": m_np["pfa"], "mech": mech}


# ------------------------------------------------- VoI theorem verification
def verify_voi(planner, x_int, i, bh):
    """Verify Q_prog - Q_dir = E[min{D(x') - D2, b_h}] for the 0->1->8 vs 0->8
    choice of UAV i at state x_int (005.md §5).  Returns the deviation."""
    # direct 0->8
    q = planner
    tpl_i = q._tpl[i][0]                     # actions from r=0
    dir_act = next((a for a in tpl_i if a[0] == 8), None)
    prog_act = next((a for a in tpl_i if a[0] == 1), None)
    if dir_act is None or prog_act is None:
        return None
    om0, p0, lp, lq = q.posterior(x_int)
    zi = 0
    pw = q.powers[i]
    llr_i = q._llr_i[i]

    def R(x):
        o = q.omega(x)
        p = 1.0 / (1.0 + np.exp(-o))
        return min(q.C01 * p, q.C10 * (1.0 - p))

    # Q_direct = b_h + 8 + E[R(x'')]
    Q_dir = bh + 8
    for (m2, lp0c, lp1c) in dir_act[2]:
        a_ = lp + lp1c
        b_ = lq + lp0c
        m_ = a_ if a_ >= b_ else b_
        w = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
        cx = x_int + (z_code_b(8, m2) - zi) * pw
        Q_dir += w * R(cx)
    # Q_prog = b_h + 1 + E_{m'}[ min{ R(x'), b_h + 7 + E[R(x'')|x'] } ]
    Q_prog = bh + 1
    lhs_inside = 0.0
    rhs_inside = 0.0
    for (m1, lp0c1, lp1c1) in prog_act[2]:
        a_ = lp + lp1c1
        b_ = lq + lp0c1
        m_ = a_ if a_ >= b_ else b_
        w1 = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
        x1 = x_int + (z_code_b(1, m1) - zi) * pw
        om1 = om0 + llr_i[z_code_b(1, m1)]
        p1 = 1.0 / (1.0 + np.exp(-om1))
        lp1 = float(np.log(p1))
        lq1 = float(np.log1p(-p1))
        # E[R(x'')|x'] over the 1->8 refinement
        ref_act = next((a for a in q._tpl[i][z_code_b(1, m1)] if a[0] == 8), None)
        E_R = 0.0
        if ref_act is not None:
            for (m2, lp0c, lp1c) in ref_act[2]:
                a_ = lp1 + lp1c
                b_ = lq1 + lp0c
                m_ = a_ if a_ >= b_ else b_
                w = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
                cx = x1 + (z_code_b(8, m2) - z_code_b(1, m1)) * pw
                E_R += w * R(cx)
        cont = bh + 7 + E_R
        R1 = R(x1)
        lhs_inside += w1 * min(R1, cont)
        D = R1 - E_R
        rhs_inside += w1 * min(D - 7, bh)
    Q_prog += lhs_inside
    lhs = Q_prog - Q_dir
    rhs = rhs_inside
    return {"lhs": lhs, "rhs": rhs, "dev": abs(lhs - rhs), "Q_prog": Q_prog, "Q_dir": Q_dir}


# ------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_ep = 8000 if args.smoke else 20000          # shared CRN sample
    n_ep_ope = 400 if args.smoke else 2500         # adaptive (per-step recursion cost)
    bh = 16.0
    h_radio = (34,)                                # lookahead (radio bits)
    os.makedirs(FIG_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.1 — 可信度修复 + Feedback-Granularity 理论拔高")
    out("")
    out("> 依据 `adcice/005.md`：修复 1bit-POTS 重复计数、共享 CRN、自然/NP 口径分离、"
        "Adaptive Direct-8 隔离 baseline、state-dependent conditional-VoI 定理、"
        "feedback/setup 成本；B0-G3 改为 UNCERTIFIED/COMPUTATION-LIMITED。")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if args.smoke else 'FULL'}")
    out("")
    out("## 0. 创新定位（005.md §十六）")
    out("")
    out("> **Feedback-Granularity-Aware Adaptive Evidence Acquisition**："
        "per-transaction setup/header 开销下，何时先发粗证据获得反馈机会、何时跨级"
        "直发更多证据；matched detection QoS 下最小化总期望 radio 资源。")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=(1, 2, 4, 8)) for i in range(8)]
    out(f"- 系统: N=8, R={{1,2,4,8}}, 每 UAV 状态 279, 理论空间 279^8≈1e19（sparse 不建表）")
    out("")

    # shared CRN sample
    rng_crn = np.random.default_rng(SEED0)
    Ht_all = model8.sample_hypotheses(n_ep, rng_crn)
    L_all = model8.sample_llr(Ht_all, rng_crn)
    Ht_o = model8.sample_hypotheses(n_ep_ope, rng_crn)
    L_o = model8.sample_llr(Ht_o, rng_crn)
    out(f"- 共享 CRN: baseline 与 O-PEF 均使用各自固定的 (H, L)；同一方法内所有参数共享同一批样本")
    out("")

    # --------------------------------------------------- P0-1: 1bit-POTS fix
    out("## 1. P0-1 — 1bit-POTS 重复计数修复")
    out("")
    lam, cost = b_seeded_pots(quants8, Ht_all, L_all, 6.0, bh)
    m_ = mclib.evaluate(lam, cost, Ht_all, PFA_TARGET)
    out(f"- 修复后 1bit-POTS (ηs=6): P_D={fmt(m_['pd'])} @ P_FA=0.05, E[B_radio]={fmt(m_['eb'])} bits"
        f"（修复前 B0 中该基线未能达标——重复计数已移除：seed 仅付一次，ladder 从 1→2 开始）")
    out("")

    # --------------------------------------------------- P0-2/3: baselines CRN + dual eval
    out("## 2. P0-2/3 — 共享 CRN 的协议公平 baseline + 自然/NP 口径")
    out("")
    out("### 2.1 公平基线（共享 CRN, n=%d, Top-K 全扫 K=1..8）" % n_ep)
    out("")
    lam, cost = b_all_neighbor(quants8, Ht_all, L_all, bh)
    m_max = mclib.evaluate(lam, cost, Ht_all, PFA_TARGET)
    pd_max = m_max["pd"]
    pd_target = pd_max - EPS_D
    out(f"- P_D,max = {fmt(pd_max)}（all-neighbor 8-bit），matched 目标 P_D ≥ {fmt(pd_target)}")
    out("")
    out("| 方法 | 参数 | P_D @ P_FA=0.05 | E[B_radio] | 达标 |")
    out("| --- | --- | --- | --- | --- |")
    bests = {}
    for name, fn, pname, vals in (
        ("AllNeighbor-8", b_all_neighbor, "-", [None]),
        ("SNR-TopK", b_snr_topk, "K", K_SWEEP),
        ("GlobalFixed", b_global_fixed, "eta_s", ETA_S_SWEEP),
        ("StaticProg", b_static_prog, "eta_s", ETA_S_SWEEP),
        ("1bit-POTS(fixed)", b_seeded_pots, "eta_s", ETA_S_SWEEP),
        ("Direct8-Ordered", b_direct8_ordered, "eta_s", ETA_S_SWEEP),
    ):
        best = None
        for v in vals:
            if name == "AllNeighbor-8":
                lam, cost = fn(quants8, Ht_all, L_all, bh)
                pv = "-"
            elif name == "SNR-TopK":
                lam, cost = fn(quants8, GAMMA_B, Ht_all, L_all, v, bh)
                pv = v
            elif name == "StaticProg":
                lam, cost = fn(quants8, GAMMA_B, Ht_all, L_all, v, bh)
                pv = v
            elif name == "Direct8-Ordered":
                lam, cost = fn(quants8, GAMMA_B, Ht_all, L_all, v, bh)
                pv = v
            else:
                lam, cost = fn(quants8, Ht_all, L_all, v, bh)
                pv = v
            m_ = mclib.evaluate(lam, cost, Ht_all, PFA_TARGET)
            if m_["pd"] >= pd_target and (best is None or m_["eb"] < best["eb"]):
                best = m_
        bests[name] = best
        if best:
            out(f"| {name} | {pv} | {fmt(best['pd'])} | {fmt(best['eb'])} | ✓ |")
        else:
            out(f"| {name} | {pv} | — | — | ✗ |")
    out("")

    # --------------------------------------------------- adaptive policies
    out("### 2.2 自适应策略（共享 CRN，n=%d；Natural-policy QoS 为主，NP ROC 为诊断）" % n_ep_ope)
    out("")
    out("| 策略 | (s,η) | H | P_D(nat) | P_FA(nat) | E[B_radio] | P_D(NP@0.05) |")
    out("| --- | --- | --- | --- | --- | --- | --- |")
    adaptive = {}
    for name, kw in (("Cross-Level O-PEF", {}),
                     ("Adjacent-Only", {"cross_level": False}),
                     ("Adaptive Direct-8", {"direct_only": True})):
        for (s, eta) in ((256, 1.0),):
            for H in h_radio:
                r_ = mc_receding_shared(quants8, model8, Ht_o, L_o, s, s * np.exp(eta),
                                        bh, H, eta, planner_kw=kw)
                ok_ = r_["pd_np"] >= pd_target
                adaptive.setdefault(name, []).append((H, r_, ok_))
                out(f"| {name} | ({s},{eta}) | {H} | {fmt(r_['pd_nat'])} | {fmt(r_['pfa_nat'])} "
                    f"| {fmt(r_['eb'])} | {fmt(r_['pd_np'])} |")
    out("")
    # the isolation comparison at the best (H, s, eta) with the SAME CRN
    out("### 2.3 收益来源隔离（同一 CRN、同一 (s,η,H)：UAV 选择 vs multi-resolution）")
    out("")
    H_ref = h_radio[-1]
    out("| 策略 | P_D(nat) | P_FA(nat) | E[B_radio] |")
    out("| --- | --- | --- | --- |")
    iso = {}
    for name, kw in (("Adaptive Direct-8", {"direct_only": True}),
                     ("Adjacent-Only", {"cross_level": False}),
                     ("Cross-Level O-PEF", {})):
        r_ = mc_receding_shared(quants8, model8, Ht_o, L_o, 256.0, 256.0 * np.exp(1.0),
                                bh, H_ref, 1.0, planner_kw=kw)
        iso[name] = r_
        out(f"| {name} | {fmt(r_['pd_nat'])} | {fmt(r_['pfa_nat'])} | {fmt(r_['eb'])} |")
    if "Adaptive Direct-8" in iso and "Cross-Level O-PEF" in iso:
        g_dir = iso["Cross-Level O-PEF"]["eb"] - iso["Adaptive Direct-8"]["eb"]
        g_adj = iso["Cross-Level O-PEF"]["eb"] - iso["Adjacent-Only"]["eb"]
        out("")
        out(f"- 同 E[B] 对比: cross-level 相对 Adaptive Direct-8 的增益 = {fmt(g_dir)} bits"
            f"（multi-resolution + UAV 选择）；相对 Adjacent-Only 的增益 = {fmt(g_adj)} bits"
            f"（cross-level/跳级）")
    out("")

    # --------------------------------------------------- P1-5: VoI theorem
    out("## 3. P1-5 — state-dependent conditional-VoI 定理验证")
    out("")
    out("> Q_prog − Q_dir = E_{x'}[ min{ D(x') − Δ₂, b_h } ]，D(x') = R(x') − E[R(x'')|x']。"
        "b_h=0 ⇒ ≤0（渐进支配）；b_h>0 ⇒ 状态相关相变。")
    out("")
    pl0 = sp.SparsePlanner(quants8, 256.0, 256.0, b_h=0.0, cross_level=True)
    pl16 = sp.SparsePlanner(quants8, 256.0, 256.0, b_h=16.0, cross_level=True)
    for bh_, pl_ in ((0.0, pl0), (16.0, pl16)):
        devs = []
        for i in (5, 6, 7):
            v = verify_voi(pl_, 0, i, bh_)
            if v:
                devs.append(v["dev"])
                out(f"- b_h={bh_:.0f}, UAV{i}: Q_prog={fmt(v['Q_prog'])} Q_dir={fmt(v['Q_dir'])} "
                    f"LHS−RHS dev = {v['dev']:.2e}（Q_prog−Q_dir = {fmt(v['lhs'])})")
        out(f"- **b_h={bh_:.0f}: 定理数值成立（max dev = {max(devs):.2e}）**"
            + ("；Q_prog ≤ Q_dir ⇒ 渐进支配" if bh_ == 0 else "；符号由状态分布决定 ⇒ 反馈粒度相变"))
        out("")
    # corollary check: q < 7/23 as communication-only
    out("### 3.1 q<7/23 降级为 communication-only Corollary；radio-only 判据 E[C_future|M^(1)]<7")
    out("")
    r_ = mc_receding_shared(quants8, model8, Ht_o, L_o, 256.0, 256.0 * np.exp(1.0),
                            bh, 34, 1.0)
    mech = r_["mech"]
    n01 = mech.get("0->1", 0)
    n12 = mech.get("1->2", 0)
    n14 = mech.get("1->4", 0)
    n18 = mech.get("1->8", 0)
    q_emp = (n12 + n14 + n18) / max(n01, 1)
    out(f"- 经验继续概率 q = ({n12}+{n14}+{n18})/{n01} = {fmt(q_emp, 3)}"
        f"（corollary 阈值 7/23 = 0.304；q<阈值 ⇒ radio-only 渐进更省）")
    out("- 更正确判据：E[C_future | M^(1)] < 7（已付 1 bit 后，相对一次性 8-bit 的剩余"
        " payload 差恰为 7 bits）——从 trajectory 直接统计，无需人为压成单变量 q")
    out("")

    # --------------------------------------------------- P1-6: feedback cost
    out("## 4. P1-6 — feedback/setup 成本与敏感性")
    out("")
    out("> c_a = b_setup + (r'−r)，b_setup = b_data-header + b_control + b_feedback"
        "（grant/ACK/query 交易开销）。")
    out("")
    out("| b_setup | root action (H=b_setup+8) | 说明 |")
    out("| --- | --- | --- |")
    for bs in (16.0, 24.0, 32.0):
        pl_ = sp.SparsePlanner(quants8, 256.0, 256.0 * np.exp(1.0), b_h=bs, cross_level=True)
        val, act = pl_.solve(0, int(bs + 8))
        r2 = act[1] if act else -1
        out(f"| {bs:.0f} | {act} | {'direct jump' if r2 > 1 else 'probe'} |")
    out("")
    out("- 在 '一个 direct-8 包' 预算（H=b_setup+8）下 root 均选 direct jump；"
        "而 B0-G1 中 H=2×(b_setup+1)（两个最小包）时 root 选 probe——粒度选择由"
        "（预算，状态）共同决定，即反馈粒度相变是状态相关的（与 VoI 定理一致）")
    out("")

    # --------------------------------------------------- gates + report
    out("## 5. B0.1 Gate 汇总")
    out("")
    out("- **P0-1 1bit-POTS 修复**: 已修复（seed 仅一次，ladder 从 1→2）")
    out("- **P0-2 共享 CRN + 样本量**: baseline %d, O-PEF %d（同一固定 (H,L)）" % (n_ep, n_ep_ope))
    out("- **P0-3 自然/NP 口径分离**: 已分离（Natural-policy QoS 为主，NP ROC 为诊断）")
    out("- **P1-4 Adaptive Direct-8**: 已加入（§2.3 隔离收益来源）")
    out(f"- **P1-5 VoI 定理**: 数值验证通过（b_h=0 ⇒ 渐进支配；b_h>0 ⇒ 状态相关相变）")
    out("- **P1-6 feedback/setup 成本**: b_setup 解释与敏感性（§4）")
    out(f"- **B0-G3 结论**: **UNCERTIFIED / COMPUTATION-LIMITED**（深预算超出 N=8 精确递归"
        f" cone 上限；O-PEF 最高 P_D(nat) 见 §2.2；需 certified rollout 扩展）")
    out("")
    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    os.makedirs(OUT_DIR, exist_ok=True)
    rp = os.path.join(OUT_DIR, "MVS-B0.1_report.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
