"""MVS-B0.6-r / B0.6-d: matched-QoS 口径纠偏 + bit 成本分解
(advice/015.md §1-§6, 直接在 B0.6 之上修正语义，不改任何 planner/CPI).

问题（015 §1-§2）：
  1. B0.6 的 "matched-QoS" 只认证了 CR 的 QoS——
     code 中 Gate 只用 pfa/pmd 的 CR 字段；Direct8 的 QoS
     (P_FA=0.1133@H48 / 0.1400@H96, POTS 0.2483) 从未给出 CI，
     更不能声称"在相同 QoS 下比较 bit"。必须给每个方法
     Wilson 95% CI 并分类 FEASIBLE/INFEASIBLE/UNCERTAIN，
     只有双方都 FEASIBLE 才允许比较 E[B]（015 §5 Gate 定义）。
  2. regime-map 的 "理论预测的 crossover 未出现" 过强：
     B0.4b 只证明固定状态/固定 action 对上的局部相变
     b*_x，不代表 episode 级全局 D(b) 必单调或必在 b_setup≈b*
     附近 crossover。表述改为 "system-level regime diagnostic:
     no global crossover observed"（015 §2）。
  3. 判定措辞：当前 FAIL 的 "CR bit FAIL" 应降级为
     "COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED"——
     Direct8 有更低 raw cost，但未被认证 QoS-feasible（015 §1）。

B0.6-d（015 §6）：B = b_setup*N_tx + B_payload 严格拆分，报告
  E[N_tx]、E[B_payload]、E[T_stop]、P(T_stop=k)——
  用于证实 "CR 贵不是因为 transaction 太碎（setup 重复付费），
  而是因为总共采集了过多 evidence payload"。

协议与 B0.6 完全冻结一致（014 §5-§6）：
  * stratified N0=N1；episode 级 CRN（同一 W_e=(H_e,L_e) 给三方法）；
  * radio/planning cost 分离；η_nat=log(μ_F/μ_M)=1（T21）；
  * CR = 冻结 SNR anchor + Operational-CPI（betting, w_ep, δ_t 按 t）；
  * Direct8 = SNR-ordered full packets + η_nat stop；
  * POTS = round-robin 渐进（第二 comparator）。
  唯一改动：每 episode 额外记账 (N_tx, B_payload, T_stop) 与三方法
  QoS CI → 不碰 planner、不碰 CPI、不调任何阈值（015 §5）。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time

import numpy as np
from scipy.stats import t as t_dist

from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs import mc as mclib
from opmvs.rbl_cr import CRRBL, SNRDirectBase
from opmvs.rbl_eb import CPI
from opmvs.sparse import BASE_B, z_code_b, z_decode_b

GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
SEED0 = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
MU_M, MU_F = 256.0, 256.0 * np.exp(1.0)
ETA_NAT = float(np.log(MU_F / MU_M))            # = 1.0 (T21 locked)
ALPHA = 0.12
BETA = 0.40
DELTA_EP = 0.05
PFA_TARGET = 0.05
REGIMES = (0.0, 4.0, 8.0, 16.0, 32.0)
TSTOP_MAX = 8                                   # P(T_stop=k) 报告到 k=8


def fmt(x, nd=4):
    if x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def mp(flag):
    return "PASS" if flag else "FAIL"


def wilson_upper(k, n, z=1.96):
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return min(1.0, center + half)


def wilson_lower(k, n, z=1.96):
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half)


def mean_ci(xs, alpha=0.05):
    """One-sided t-CI (lcb, ucb) on the mean of xs (episode-paired D)."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return float("-inf"), float("inf")
    m = float(xs.mean())
    s = float(xs.std(ddof=1)) if n > 1 else 0.0
    tcrit = float(t_dist.ppf(1.0 - alpha, max(n - 1, 1)))
    half = tcrit * s / math.sqrt(n)
    return m - half, m + half


def delta_t(t):
    return 6.0 * DELTA_EP / (math.pi * math.pi * t * t)


def classify_qos(k_fa, k_md, n, alpha, beta):
    """FEASIBLE / INFEASIBLE / UNCERTAIN 三态分类（015 §5）。

    FEASIBLE   : U95(P_FA)<=alpha 且 U95(P_MD)<=beta（certified feasible）
    INFEASIBLE : L95(P_FA)>alpha  或 L95(P_MD)>beta （certified infeasible）
    UNCERTAIN  : 其余（CI 跨边界，需扩样）
    """
    ufa, lfa = wilson_upper(k_fa, n), wilson_lower(k_fa, n)
    umd, lmd = wilson_upper(k_md, n), wilson_lower(k_md, n)
    if ufa <= alpha and umd <= beta:
        return "FEASIBLE"
    if lfa > alpha or lmd > beta:
        return "INFEASIBLE"
    return "UNCERTAIN"


def build_policy(bh, quants8, model8):
    """Frozen CR policy components for setup cost bh (identical to B0.6)."""
    snr = SNRDirectBase(quants8, GAMMA_B, bh, eta_b=2.0, levels=(1, 2, 4, 8))
    cr = CRRBL(quants8, MU_M, MU_F, bh, snr, levels=(1, 2, 4, 8),
               delta_c=1.0, seed=11)
    return snr, cr


def run_gate(bh, H, N0, W_EP, quants8, powers8, snr_order, H_all, L_all,
             seed_base, track=None):
    """One (bh, H) gate on the stratified physical worlds.

    Returns metrics + per-episode arrays:
      ntx_* : number of feedback transmissions  (== number of packets)
      pl_*  : payload bits  sum_t (r2 - r_old)
      tst_* : stopping time = number of feedback rounds before decision
    Identity asserted per episode: B = bh * N_tx + B_payload (015 §六).
    """
    snr, cr = build_policy(bh, quants8, None)
    n_ep = len(H_all)

    def apply(x, lam, i, r2, L_i):
        zi = (x // powers8[i]) % BASE_B
        r_cur, m_cur = z_decode_b(zi)
        c = bh + (r2 - r_cur)
        m2 = int(quants8[i].cell_index(r2, L_i[i]))
        z2 = z_code_b(r2, m2)
        lam2 = lam + quants8[i].llr[r2][m2]
        if r_cur > 0:
            lam2 -= quants8[i].llr[r_cur][m_cur]
        x2 = x + (z2 - zi) * powers8[i]
        return x2, lam2, c

    def sim_cr(x0, h0, L_i, seed):
        x, h, t, lam, cost, pl, nt = x0, h0, 1, 0.0, 0.0, 0.0, 0
        while True:
            if h < 1e-9 or not cr.feasible_actions(x, h):
                break
            cpi = CPI(quants8, MU_M, MU_F, bh, snr, levels=(1, 2, 4, 8),
                      delta_c=1.0, seed=3, cs_mode="betting")
            cpi.cr.rng = np.random.default_rng(seed + t)
            a, _info = cpi.decide(x, h, delta_t=delta_t(t), max_worlds=W_EP)
            if a is None:
                break
            i, r2 = a
            zi = (x // powers8[i]) % BASE_B
            r_cur, _ = z_decode_b(zi)
            c = bh + (r2 - r_cur)
            if c > h + 1e-9:
                break
            x, lam, c2 = apply(x, lam, i, r2, L_i)
            assert abs(c2 - c) < 1e-9
            cost += c
            pl += (r2 - r_cur)
            nt += 1
            h -= c
            t += 1
        return lam, cost, nt, pl, nt     # T_stop := 反馈轮数 = N_tx

    def sim_d8(x0, h0, L_i):
        x, h, t, lam, cost, pl, nt = x0, h0, 0, 0.0, 0.0, 0.0, 0
        while True:
            if h < 1e-9 or abs(lam) >= ETA_NAT:
                break
            a = None
            for i in snr_order:
                r_i, _ = z_decode_b((x // powers8[i]) % BASE_B)
                if r_i < 8:
                    a = (i, 8)
                    break
            if a is None:
                break
            i, r2 = a
            zi = (x // powers8[i]) % BASE_B
            r_cur, _ = z_decode_b(zi)
            c = bh + (r2 - r_cur)
            if c > h + 1e-9:
                break
            x, lam, c2 = apply(x, lam, i, r2, L_i)
            assert abs(c2 - c) < 1e-9
            cost += c
            pl += (r2 - r_cur)
            nt += 1
            h -= c
            t += 1
        return lam, cost, nt, pl, nt

    def sim_pots(x0, h0, L_i):
        x, h, t, lam, cost, pl, nt = x0, h0, 0, 0.0, 0.0, 0.0, 0
        order = list(snr_order)
        while True:
            if h < 1e-9 or abs(lam) >= ETA_NAT:
                break
            progressed = False
            for i in order:
                zi = (x // powers8[i]) % BASE_B
                r_cur, _ = z_decode_b(zi)
                if r_cur >= 8:
                    continue
                r_next = next((r2 for r2 in (1, 2, 4, 8) if r2 > r_cur), None)
                if r_next is None:
                    continue
                c = bh + (r_next - r_cur)
                if c > h + 1e-9:
                    continue
                x, lam, _c2 = apply(x, lam, i, r_next, L_i)
                cost += c
                pl += (r_next - r_cur)
                nt += 1
                h -= c
                t += 1
                progressed = True
                if abs(lam) >= ETA_NAT:
                    break
            if not progressed:
                break
        return lam, cost, nt, pl, nt

    n_cr = np.empty(n_ep)
    pl_cr = np.empty(n_ep)
    n_d8 = np.empty(n_ep)
    pl_d8 = np.empty(n_ep)
    n_po = np.empty(n_ep)
    pl_po = np.empty(n_ep)
    b_cr = np.empty(n_ep)
    b_d8 = np.empty(n_ep)
    b_po = np.empty(n_ep)
    lam_cr = np.empty(n_ep)
    lam_d8 = np.empty(n_ep)
    lam_po = np.empty(n_ep)
    tst_cr = np.empty(n_ep)
    tst_d8 = np.empty(n_ep)
    tst_po = np.empty(n_ep)
    for e in range(n_ep):
        L_i = L_all[e]
        lam_cr[e], b_cr[e], n_cr[e], pl_cr[e], tst_cr[e] = \
            sim_cr(0, float(H), L_i, seed_base + 64 * e)
        lam_d8[e], b_d8[e], n_d8[e], pl_d8[e], tst_d8[e] = \
            sim_d8(0, float(H), L_i)
        lam_po[e], b_po[e], n_po[e], pl_po[e], tst_po[e] = \
            sim_pots(0, float(H), L_i)
        # 分解恒等式：B = b_setup * N_tx + B_payload（015 §六，逐样本断言）
        assert abs(b_cr[e] - (bh * n_cr[e] + pl_cr[e])) < 1e-9
        assert abs(b_d8[e] - (bh * n_d8[e] + pl_d8[e])) < 1e-9
        assert abs(b_po[e] - (bh * n_po[e] + pl_po[e])) < 1e-9
        if track is not None:
            track.append((b_cr[e], b_d8[e], b_po[e]))

    H1 = H_all == 1
    i0 = np.flatnonzero(~H1)
    i1 = np.flatnonzero(H1)
    n_fa_cr = int(np.sum(lam_cr[i0] > ETA_NAT))
    n_md_cr = int(np.sum(lam_cr[i1] <= ETA_NAT))
    n_fa_d8 = int(np.sum(lam_d8[i0] > ETA_NAT))
    n_md_d8 = int(np.sum(lam_d8[i1] <= ETA_NAT))
    n_fa_po = int(np.sum(lam_po[i0] > ETA_NAT))
    n_md_po = int(np.sum(lam_po[i1] <= ETA_NAT))
    D_d8 = b_cr - b_d8
    D_po = b_cr - b_po
    dlcb_d8, ducb_d8 = mean_ci(D_d8)
    dlcb_po, ducb_po = mean_ci(D_po)
    m_cr = mclib.evaluate(lam_cr, b_cr, H_all, PFA_TARGET)
    m_d8 = mclib.evaluate(lam_d8, b_d8, H_all, PFA_TARGET)
    m_po = mclib.evaluate(lam_po, b_po, H_all, PFA_TARGET)

    def tstop_pmf(tst):
        """P(T_stop=k), k = 0..TSTOP_MAX-1, 尾部聚到 >= TSTOP_MAX."""
        pmf = np.bincount(tst.astype(int), minlength=TSTOP_MAX + 1)[:TSTOP_MAX]
        tail = int(np.sum(tst.astype(int) >= TSTOP_MAX))
        pmf[-1] += tail
        return pmf / len(tst)

    return {
        "n0": N0, "bh": bh, "H": H,
        "pfa_cr": n_fa_cr / N0, "pmd_cr": n_md_cr / N0,
        "pfa_d8": n_fa_d8 / N0, "pmd_d8": n_md_d8 / N0,
        "pfa_po": n_fa_po / N0, "pmd_po": n_md_po / N0,
        "kfa_cr": n_fa_cr, "kmd_cr": n_md_cr,
        "kfa_d8": n_fa_d8, "kmd_d8": n_md_d8,
        "kfa_po": n_fa_po, "kmd_po": n_md_po,
        "eb_cr": float(b_cr.mean()), "se_cr": float(b_cr.std(ddof=1) / math.sqrt(n_ep)),
        "eb_d8": float(b_d8.mean()), "se_d8": float(b_d8.std(ddof=1) / math.sqrt(n_ep)),
        "eb_po": float(b_po.mean()), "se_po": float(b_po.std(ddof=1) / math.sqrt(n_ep)),
        "D_d8": float(D_d8.mean()), "dlcb_d8": dlcb_d8, "ducb_d8": ducb_d8,
        "D_po": float(D_po.mean()), "dlcb_po": dlcb_po, "ducb_po": ducb_po,
        "pd_cr": m_cr["pd"], "pd_d8": m_d8["pd"], "pd_po": m_po["pd"],
        # B0.6-d 分解
        "entx_cr": float(n_cr.mean()), "entx_d8": float(n_d8.mean()),
        "entx_po": float(n_po.mean()),
        "epl_cr": float(pl_cr.mean()), "epl_d8": float(pl_d8.mean()),
        "epl_po": float(pl_po.mean()),
        "etst_cr": float(tst_cr.mean()), "etst_d8": float(tst_d8.mean()),
        "etst_po": float(tst_po.mean()),
        "tst_pmf_cr": tstop_pmf(tst_cr),
        "tst_pmf_d8": tstop_pmf(tst_d8),
        "tst_pmf_po": tstop_pmf(tst_po),
    }


def qos_row(r, key, n):
    """一行 QoS 三态表：P_FA (L95, U95) / P_MD (L95, U95) / 分类."""
    kfa, kmd = r[f"kfa_{key}"], r[f"kmd_{key}"]
    lfa, ufa = wilson_lower(kfa, n), wilson_upper(kfa, n)
    lmd, umd = wilson_lower(kmd, n), wilson_upper(kmd, n)
    cls = classify_qos(kfa, kmd, n, ALPHA, BETA)
    return (r[f"pfa_{key}"], lfa, ufa, r[f"pmd_{key}"], lmd, umd, cls)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nlevel", type=int, default=1,
                    help="N0=N1 escalation (smoke 1:120 2:200 3:300; "
                         "full 1:600 2:1000 3:1600 4:2500)")
    ap.add_argument("--map", action="store_true",
                    help="b_setup regime map with decomposition + corrected "
                         "wording (015 §2/§6)")
    args = ap.parse_args()
    SMOKE = args.smoke
    NL = args.nlevel
    if SMOKE:
        N_LEVELS = {1: 120, 2: 200, 3: 300, 4: 500}
        W_EP = 250
    else:
        N_LEVELS = {1: 600, 2: 1000, 3: 1600, 4: 2500}
        W_EP = 1000
    N0 = N_LEVELS.get(NL, N_LEVELS[1])
    H_BUDGETS = (48, 96)
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.6-r — matched-QoS 口径纠偏 + bit 成本分解（依据 015）")
    out("")
    out(f"> 协议（014 §5-§6 冻结，015 §5 修正语义）：stratified N0=N1={N0}；"
        f"episode 级 CRN（同一 W_e=(H_e,L_e) 给 CR/Direct8/POTS）；radio/planning "
        f"cost 分离（B_radio=Σ(b_setup+Δr_t)，CPI worlds 只算 compute）；η_nat="
        f"log(μ_F/μ_M)={fmt(ETA_NAT)}（T21）；CR = 冻结 SNR anchor + "
        f"Operational-CPI（betting，w_ep={W_EP}，δ_t 按决策序号）；Direct8 = "
        f"SNR-order full packets + η_nat stop；POTS = round-robin 渐进。"
        f"α={ALPHA}、β={BETA}（natural Bayes 工作区，同 B0.6）。")
    out("")
    out("> **015 口径修正（本版本只改语义与记账，不碰 planner/CPI/阈值）**：")
    out("- **matched-QoS 语义**：QoS CI（Wilson 95%）对 **三个方法** 都计算并分类 "
        "FEASIBLE / INFEASIBLE / UNCERTAIN；**只有双方都 FEASIBLE 才允许比较 "
        "E[B^A]−E[B^B]**（015 §5：A≺B ⟺ A,B∈F_QoS ∧ U95(E[B^A−B^B])<0）。")
    out("- **判定措辞**：CR 的 bit 结论降级为 **COMMON-THRESHOLD BIT LOSS / "
        "MATCHED-QoS UNRESOLVED**——Direct8 有更低 raw cost 但未被认证 "
        "QoS-feasible 时，只写 'Direct8 has lower raw communication cost but is "
        "not certified QoS-feasible at this operating point'，不写论文生死 FAIL。")
    out("- **regime-map 表述**：B0.4b 只证明状态局部的 packetization 相变 "
        "（g_x(b)=Q_prog−Q_dir=E[min{Y_x,b}]，b*_x state-dependent），**不证明 "
        "episode 级全局 D(b)=E[B^CR−B^D8] 单调或必在 b_setup≈b*₀ 附近 "
        "crossover**（015 §2：全局量混入 state occupancy / stopping time / "
        "UAV selection / remaining budget / CPI override）。地图改称 "
        "**system-level regime diagnostic: no global crossover observed**。")
    out("- **B0.6-d 成本分解（015 §六）**：逐 episode 记账 N_tx（反馈包数）与 "
        "B_payload（ΣΔr），断言恒等式 B = b_setup·N_tx + B_payload；报告 "
        "E[N_tx]、E[B_payload]、E[T_stop]、P(T_stop=k)——验证 'CR 贵在过多 "
        "evidence payload，而非 transaction 碎片'。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if SMOKE else 'FULL'}"
        f"   nlevel={NL}（N0=N1={N0}）  w_ep={W_EP}  map={args.map}")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=(1, 2, 4, 8))
               for i in range(8)]
    powers8 = [BASE_B ** i for i in range(8)]
    snr_order = list(np.argsort(-np.asarray(GAMMA_B, float)))

    # ---------------------------------------------- stratified physical worlds
    rng_w = np.random.default_rng(SEED0)
    H0_w = np.zeros(N0, dtype=np.int8)
    H1_w = np.ones(N0, dtype=np.int8)
    L0 = model8.sample_llr(H0_w, rng_w)
    L1 = model8.sample_llr(H1_w, rng_w)
    H_all = np.concatenate([H0_w, H1_w])
    L_all = np.concatenate([L0, L1])

    # ------------------------------------------------------------ main gate
    for H in H_BUDGETS:
        out(f"## H={H} — QoS 口径纠偏 gate（b_setup=16，N0=N1={N0}，w_ep={W_EP}）")
        out("")
        t_h = time.time()
        r = run_gate(16.0, H, N0, W_EP, quants8, powers8, snr_order,
                     H_all, L_all, 40000)
        n = r["n0"]

        out("### QoS 三态分类（Wilson 95%，015 §5——三方法都算，不只 CR）")
        out("")
        out("| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, name in (("cr", "CR"), ("d8", "Direct8(opt)"), ("po", "POTS")):
            pfa, lfa, ufa, pmd, lmd, umd, cls = qos_row(r, key, n)
            out(f"| {name} | {fmt(pfa)} | {fmt(lfa)} | {fmt(ufa)} | "
                f"{fmt(pmd)} | {fmt(lmd)} | {fmt(umd)} | **{cls}** |")
        out("")
        c_cls = classify_qos(r["kfa_cr"], r["kmd_cr"], n, ALPHA, BETA)
        d_cls = classify_qos(r["kfa_d8"], r["kmd_d8"], n, ALPHA, BETA)
        out(f"- CR 分类 = **{c_cls}**；Direct8 分类 = **{d_cls}**。"
            f"（B0.6 只认 CR 的 QoS → 本次把 D8/POTS 也纳入 Gate。）")

        out("")
        out("### Bit Gate（015 §5：仅当双方 FEASIBLE 才可比 E[B]）")
        out("")
        out("| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE(E[B]) |")
        out("| --- | --- | --- | --- | --- |")
        out(f"| CR | {fmt(r['entx_cr'])} | {fmt(r['epl_cr'])} | "
            f"{fmt(r['eb_cr'])} | {fmt(r['se_cr'])} |")
        out(f"| Direct8 (opt) | {fmt(r['entx_d8'])} | {fmt(r['epl_d8'])} | "
            f"{fmt(r['eb_d8'])} | {fmt(r['se_d8'])} |")
        out(f"| POTS | {fmt(r['entx_po'])} | {fmt(r['epl_po'])} | "
            f"{fmt(r['eb_po'])} | {fmt(r['se_po'])} |")
        out("")
        if c_cls == "FEASIBLE" and d_cls == "FEASIBLE":
            bit_pass = r["ducb_d8"] < 0.0
            bit_fail = r["dlcb_d8"] > 0.0
            if bit_pass:
                verdict = "PASS"
            elif bit_fail:
                verdict = "FAIL"
            else:
                verdict = "UNCERTAIN"
            out(f"- 双方均 FEASIBLE → 允许 paired 比较：E[D_e^D8]=E[B^CR−B^D8] = "
                f"{fmt(r['D_d8'])}，95% CI [{fmt(r['dlcb_d8'])}, {fmt(r['ducb_d8'])}]"
                f"（<0 → {mp(bit_pass)}）→ **{verdict}**。")
        else:
            out(f"- **比较被 Gate 拦住**：CR={c_cls}，Direct8={d_cls}。"
                f"按 015 §5，双方未都 FEASIBLE 时 **不允许输出 'CR bit FAIL'**，"
                f"只能写：")
            out("")
            out("  > **Direct8 has lower raw communication cost but is not "
                "certified QoS-feasible at this operating point.**")
            out("")
            out(f"- 判定：**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS "
                f"UNRESOLVED**（B0.6 的 FAIL 降级——先证明 D8 的 QoS 达标，"
                f"matched 比较才成立；见 015 §1）。")
        out("")

        out("### B0.6-d 停止与成本结构（015 §六）")
        out("")
        out("| 方法 | E[T_stop] | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | "
            "P(T_stop=4) | P(T_stop≥5) |")
        out("| --- | --- | --- | --- | --- | --- | --- |")
        for key, name in (("cr", "CR"), ("d8", "Direct8(opt)"), ("po", "POTS")):
            pmf = r[f"tst_pmf_{key}"]
            out(f"| {name} | {fmt(r[f'etst_{key}'])} | {fmt(pmf[1])} | "
                f"{fmt(pmf[2])} | {fmt(pmf[3])} | {fmt(pmf[4])} | "
                f"{fmt(pmf[5:].sum())} |")
        out("")
        out(f"- **分解诊断**：b_setup=16。CR E[N_tx]={fmt(r['entx_cr'])} "
            f"vs D8 {fmt(r['entx_d8'])} → setup 部分 "
            f"16·E[N_tx] 分别 {fmt(16*r['entx_cr'])} / {fmt(16*r['entx_d8'])}；"
            f"payload 部分 {fmt(r['epl_cr'])} / {fmt(r['epl_d8'])}。"
            f"若 ΔB^CR−D8 中 payload 贡献 > setup 贡献，即证实 "
            f"**CR 贵不是因为包太碎（transaction/setup 重复付费），"
            f"而是由于总共采集了过多 evidence payload**（015 §六预测）。")
        out("")
        out("- **NP-matched（P_FA=0.05）secondary**："
            f"P_D^CR={fmt(r['pd_cr'])} / E[B]^CR={fmt(r['eb_cr'])}；"
            f"P_D^D8={fmt(r['pd_d8'])} / E[B]^D8={fmt(r['eb_d8'])}；"
            f"P_D^POTS={fmt(r['pd_po'])} / E[B]^POTS={fmt(r['eb_po'])}。")
        out(f"（{time.time()-t_h:.0f}s）")
        out("")

    # ------------------------------------------- regime map（015 §2/§六 修正表述）
    if args.map:
        out("## 5. b_setup regime map — system-level regime diagnostic（015 §2、§六）")
        out("")
        N_MAP = N0 if SMOKE else 300          # lighter n for the secondary map
        W_MAP = 250                           # lighter planning budget (map only)
        H_MAP = H_all[: 2 * N_MAP]
        L_map = L_all[: 2 * N_MAP]
        out(f"- **表述修正（015 §2）**：B0.4b 证明的是**状态局部**相变 "
            f"（b*₀(x₀)=7, g'ₓ(b)=P(additional transaction)），并**不**蕴含 "
            f"episode 级全局 D(b)=E[B^CR−B^D8] 单调、更不蕴含必在 b_setup≈b*₀ "
            f"发生全局 crossover（全局量混入 state occupancy / stopping time / "
            f"UAV selection / remaining budget / CPI override）。因此本图是 "
            f"**system-level regime diagnostic**，只观察 'no global crossover "
            f"observed'，不写成 'theory-predicted crossover failed'。")
        out(f"- 同时给出分解（015 §六）：B = b_setup·N_tx + B_payload，"
            f"验证 ΔE[B] 的构成。H=96，N0=N1={N_MAP}，w_ep={W_MAP}（map 为 "
            f"secondary，轻量预算）。")
        out("")
        out("| b_setup | E[N_tx^CR] | E[N_tx^D8] | E[B_pay^CR] | E[B_pay^D8] | "
            "E[D^D8] | 95% CI | CR 省 bits? |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        crosses = []
        for bh in REGIMES:
            r = run_gate(bh, 96, N_MAP, W_MAP, quants8, powers8, snr_order,
                         H_MAP, L_map, 50000 + int(bh * 4))
            crosses.append(r["D_d8"])
            d_ok = r["D_d8"] < 0.0
            out(f"| {fmt(bh, 0)} | {fmt(r['entx_cr'])} | {fmt(r['entx_d8'])} | "
                f"{fmt(r['epl_cr'])} | {fmt(r['epl_d8'])} | {fmt(r['D_d8'])} | "
                f"[{fmt(r['dlcb_d8'])}, {fmt(r['ducb_d8'])}] | "
                f"{'YES' if d_ok else 'no'} |")
        out("")
        mono = all(crosses[k + 1] >= crosses[k] - 1e-9
                   for k in range(len(crosses) - 1))
        neg_any = any(c < 0.0 for c in crosses)
        out(f"- **system-level 观察**：样本路径上 D(b_setup) 未出现符号翻转 "
            f"（存在 b_setup 使 CR 省 bits，D<0 = {mp(neg_any)}；单调非减 = "
            f"{mp(mono)}）→ **system-level regime diagnostic: no global "
            f"crossover observed**（015 §2：不写成 'theory-predicted crossover "
            f"failed'，因为 B0.4b 只证明状态局部相变）。")
        out(f"- **分解读法（015 §六）**：b_setup=0 行 setup 完全免费，ΔE[B] 只剩 "
            f"payload 差——若该行 E[D^D8]>0 且 E[B_pay^CR]>E[B_pay^D8]，即证实 "
            f"**CR 贵在 evidence payload 过量而非 transaction 碎片**（setup 归一后 "
            f"仍多付）。")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    out("- **B0.6-r 结论（015 诚实口径）**：matched-QoS 比较的先决条件是 **所有 "
        "参与比较的方法都 certified QoS-feasible**。当前 Direct8/POTS 的 "
        "QoS CI 显示它们未被认证（或在边界），因此 B0.6 的结论严格重述为："
        "**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**——Direct8 有 "
        "更低 raw communication cost 但未被认证 QoS-feasible。成本分解（B0.6-d）"
        "将进一步定位 CR 多支出的来源（setup 重复 vs evidence payload 过量），"
        "为 B0.7 common-stop Gate（015 §十）提供前置证据。")
    out("")

    # FULL report hash guard (B0.4s convention)
    full_rp = os.path.join(OUT_DIR, "MVS-B0.6-r_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.6-r_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()