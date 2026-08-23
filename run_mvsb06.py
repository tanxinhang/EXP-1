"""MVS-B0.6: matched-QoS certified communication — CR vs optimized Direct8
(POTS second comparator).  THE paper-critical gate (advice/014 §4-§6).

Question (014 §4):  under STRICT matched QoS, does multi-granularity feedback
(CR) statistically beat optimized Direct8 in radio bits?

Protocol (014 §5-§6, frozen):
  * STRATIFIED evaluation N0 = N1 (H0/H1 episodes sampled separately).
  * Episode-level CRN: the SAME physical world W_e = (H_e, L_e) drives CR,
    Direct8 and POTS; each algorithm's internal planner RNG is independent.
  * Radio/planning cost separation: B_radio = sum_t (b_setup + Δr_t).
  * Natural decision threshold eta_nat = log(mu_F/mu_M) = 1.0 (T21).
  * CR = FROZEN policy: SNR anchor + Operational-CPI (betting) at w_ep,
    delta_t = 6 delta_episode/(pi^2 t^2).
  * Direct8 = OPTIMIZED: SNR-ordered full packets + eta_nat stop (the
    direct_only planner is exact but intractable at H=96: 4 full packets ->
    256^4 cells).  POTS = round-robin progressive (second comparator).
  * Episode-paired D_e^{D8} = B_e^{CR} - B_e^{D8}.
  * Gates: U95(P_FA^CR) <= alpha, U95(P_MD^CR) <= beta, U95(E[D^{D8}]) < 0.
    alpha = 0.12, beta = 0.40 (natural Bayes operating region).
  * Outcome: PASS / FAIL / UNCERTAIN (escalate N0=N1 with --nlevel).
  * Secondary: NP-matched (P_FA=0.05) P_D and E[B].
  * Pre-declared secondary analysis (014 §7): --map runs the b_setup in
    {0,4,8,16,32} regime map verifying the theory-predicted crossover.
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
    """One-sided t-CI (lcb, ucb) on the mean of xs."""
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


def build_policy(bh, quants8, model8):
    """Frozen CR policy components for setup cost bh."""
    snr = SNRDirectBase(quants8, GAMMA_B, bh, eta_b=2.0, levels=(1, 2, 4, 8))
    cr = CRRBL(quants8, MU_M, MU_F, bh, snr, levels=(1, 2, 4, 8),
               delta_c=1.0, seed=11)
    return snr, cr


def run_gate(bh, H, N0, W_EP, quants8, powers8, snr_order, H_all, L_all,
             seed_base, track=None):
    """One (bh, H) matched-QoS gate on the stratified physical worlds.
    Returns a metrics dict.  track: optional list to append per-episode
    (b_cr, b_d8, b_po, lam_cr, lam_d8, lam_po)."""
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
        x, h, t, lam, cost = x0, h0, 1, 0.0, 0.0
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
            h -= c
            t += 1
        return lam, cost, t - 1

    def sim_d8(x0, h0, L_i):
        x, h, t, lam, cost = x0, h0, 0, 0.0, 0.0
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
            h -= c
            t += 1
        return lam, cost, t

    def sim_pots(x0, h0, L_i):
        x, h, t, lam, cost = x0, h0, 0, 0.0, 0.0
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
                h -= c
                t += 1
                progressed = True
                if abs(lam) >= ETA_NAT:
                    break
            if not progressed:
                break
        return lam, cost, t

    lam_cr = np.empty(n_ep)
    lam_d8 = np.empty(n_ep)
    lam_po = np.empty(n_ep)
    b_cr = np.empty(n_ep)
    b_d8 = np.empty(n_ep)
    b_po = np.empty(n_ep)
    for e in range(n_ep):
        L_i = L_all[e]
        lam_cr[e], b_cr[e], _ = sim_cr(0, float(H), L_i, seed_base + 64 * e)
        lam_d8[e], b_d8[e], _ = sim_d8(0, float(H), L_i)
        lam_po[e], b_po[e], _ = sim_pots(0, float(H), L_i)
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
    return {
        "n0": N0, "bh": bh, "H": H,
        "pfa_cr": n_fa_cr / N0, "pmd_cr": n_md_cr / N0,
        "pfa_d8": n_fa_d8 / N0, "pmd_d8": n_md_d8 / N0,
        "pfa_po": n_fa_po / N0, "pmd_po": n_md_po / N0,
        "eb_cr": float(b_cr.mean()), "se_cr": float(b_cr.std(ddof=1) / math.sqrt(n_ep)),
        "eb_d8": float(b_d8.mean()), "se_d8": float(b_d8.std(ddof=1) / math.sqrt(n_ep)),
        "eb_po": float(b_po.mean()), "se_po": float(b_po.std(ddof=1) / math.sqrt(n_ep)),
        "D_d8": float(D_d8.mean()), "dlcb_d8": dlcb_d8, "ducb_d8": ducb_d8,
        "D_po": float(D_po.mean()), "dlcb_po": dlcb_po, "ducb_po": ducb_po,
        "pd_cr": m_cr["pd"], "pd_d8": m_d8["pd"], "pd_po": m_po["pd"],
        "pfa_ucb": wilson_upper(n_fa_cr, N0), "pfa_lcb": wilson_lower(n_fa_cr, N0),
        "pmd_ucb": wilson_upper(n_md_cr, N0), "pmd_lcb": wilson_lower(n_md_cr, N0),
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nlevel", type=int, default=1,
                    help="N0=N1 escalation level (smoke: 1:120 2:200 3:300; "
                         "full: 1:600 2:1000 3:1600 4:2500)")
    ap.add_argument("--map", action="store_true",
                    help="pre-declared secondary b_setup regime map "
                         "(014 §7): b_setup in {0,4,8,16,32}")
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

    out("# O-PEF MVS-B0.6 — matched-QoS：CR vs optimized Direct8（论文生死 Gate，014 §4-§6）")
    out("")
    out(f"> 协议（014 §5-§6 冻结）：**stratified N0=N1={N0}**（H0/H1 独立采样）；episode 级 "
        f"**CRN**——同一 physical world W_e=(H_e,L_e) 给 CR/Direct8/POTS，planner RNG 独立；"
        f"**radio cost 与 planning cost 分离**（B_radio=Σ(b_setup+Δr_t)，CPI worlds 只算 "
        f"compute）；判决阈值 η_nat=log(μ_F/μ_M)={fmt(ETA_NAT)}（T21）；CR = 冻结的 "
        f"SNR anchor + Operational-CPI（betting，w_ep={W_EP}，δ_t 按决策序号）；Direct8 = "
        f"**optimized**（SNR-order full packets + η_nat stop；direct_only planner 在 "
        f"H=96 不可行——4 个全包 → 256⁴ cells）；POTS = round-robin 渐进（第二 "
        f"comparator）。Gate：U95(P_FA^CR)≤{ALPHA}、U95(P_MD^CR)≤{BETA}、"
        f"U95(E[D_e^D8])<0（D_e^D8=B_e^CR−B_e^D8，episode-paired）。")
    out(f"- α={ALPHA}, β={BETA} 的选取：natural Bayes 判决的工作区（G5 实测 "
        f"P_FA^nat≈0.09、P_MD^nat≈0.22-0.37 @ η=1）——Gate 检验 QoS-viability 而非重调阈值。")
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
        out(f"## H={H} — matched-QoS gate（b_setup=16，N0=N1={N0}，w_ep={W_EP}）")
        out("")
        t_h = time.time()
        r = run_gate(16.0, H, N0, W_EP, quants8, powers8, snr_order,
                     H_all, L_all, 40000)
        out("| 方法 | P_FA^nat | P_MD^nat | E[B] | SE(E[B]) |")
        out("| --- | --- | --- | --- | --- |")
        out(f"| CR | {fmt(r['pfa_cr'])} | {fmt(r['pmd_cr'])} | "
            f"{fmt(r['eb_cr'])} | {fmt(r['se_cr'])} |")
        out(f"| Direct8 (opt) | {fmt(r['pfa_d8'])} | {fmt(r['pmd_d8'])} | "
            f"{fmt(r['eb_d8'])} | {fmt(r['se_d8'])} |")
        out(f"| POTS | {fmt(r['pfa_po'])} | {fmt(r['pmd_po'])} | "
            f"{fmt(r['eb_po'])} | {fmt(r['se_po'])} |")
        out("")
        out(f"- **QoS（CR，η_nat）**：P_FA^CR={fmt(r['pfa_cr'])}，"
            f"U95={fmt(r['pfa_ucb'])}（≤{ALPHA} → {mp(r['pfa_ucb'] <= ALPHA)}）；"
            f"P_MD^CR={fmt(r['pmd_cr'])}，U95={fmt(r['pmd_ucb'])}"
            f"（≤{BETA} → {mp(r['pmd_ucb'] <= BETA)}）。")
        out(f"- **Bit（episode-paired）**：E[D_e^D8] = {fmt(r['D_d8'])} bits"
            f"（CR−Direct8），95% CI [{fmt(r['dlcb_d8'])}, {fmt(r['ducb_d8'])}]"
            f"（<0 → {mp(r['ducb_d8'] < 0.0)}）；E[D_e^POTS] = {fmt(r['D_po'])}"
            f"（[{fmt(r['dlcb_po'])}, {fmt(r['ducb_po'])}]）。")
        qos_pass = (r["pfa_ucb"] <= ALPHA) and (r["pmd_ucb"] <= BETA)
        bit_pass = r["ducb_d8"] < 0.0
        qos_fail = (r["pfa_lcb"] > ALPHA) or (r["pmd_lcb"] > BETA)
        bit_fail = r["dlcb_d8"] > 0.0
        if qos_pass and bit_pass:
            verdict = "PASS"
        elif qos_fail or bit_fail:
            verdict = "FAIL"
        else:
            verdict = "UNCERTAIN"
        out(f"- **H={H} 判定：{verdict}**（QoS={mp(qos_pass)}，bit={mp(bit_pass)}"
            f"{'；CI 跨边界 → 需扩样（--nlevel 增加）' if verdict == 'UNCERTAIN' else ''}"
            f"{'；CI 已证明不满足 → 诚实 FAIL' if verdict == 'FAIL' else ''}）")
        out("")
        out("- **NP-matched（P_FA=0.05）secondary**："
            f"P_D^CR={fmt(r['pd_cr'])} / E[B]^CR={fmt(r['eb_cr'])}；"
            f"P_D^D8={fmt(r['pd_d8'])} / E[B]^D8={fmt(r['eb_d8'])}；"
            f"P_D^POTS={fmt(r['pd_po'])} / E[B]^POTS={fmt(r['eb_po'])}。")
        out(f"（{time.time()-t_h:.0f}s）")
        out("")

    # ------------------------------------------- pre-declared regime map (014 §7)
    if args.map:
        out("## 5. b_setup regime map（secondary analysis，014 §7 预先声明）")
        out("")
        N_MAP = N0 if SMOKE else 300          # lighter n for the secondary map
        W_MAP = 250                           # lighter planning budget (map only)
        H_MAP = H_all[: 2 * N_MAP]
        L_map = L_all[: 2 * N_MAP]
        out(f"- 理论预测（B0.4b）：root b⋆(x₀)=7；b_setup 小 ⇒ setup 便宜 ⇒ progressive/"
            f"CR 赢；b_setup 大 ⇒ direct 赢；crossover 应在 b_setup≈b⋆ 附近。"
            f"验证 E[D_e^D8]=E[B^CR−B^D8] 随 b_setup 的走向。H=96，N0=N1={N_MAP}，"
            f"w_ep={W_MAP}（map 为 secondary，用轻量预算）。")
        out("")
        out("| b_setup | E[B^CR] | E[B^D8] | E[B^POTS] | E[D^D8] | 95% CI | CR 省 bits? |")
        out("| --- | --- | --- | --- | --- | --- | --- |")
        crosses = []
        for bh in REGIMES:
            r = run_gate(bh, 96, N_MAP, W_MAP, quants8, powers8, snr_order,
                         H_MAP, L_map, 50000 + int(bh * 4))
            d = r["D_d8"]
            d_ok = d < 0.0
            crosses.append(d)
            out(f"| {fmt(bh, 0)} | {fmt(r['eb_cr'])} | {fmt(r['eb_d8'])} | "
                f"{fmt(r['eb_po'])} | {fmt(d)} | [{fmt(r['dlcb_d8'])}, "
                f"{fmt(r['ducb_d8'])}] | {'YES' if d_ok else 'no'} |")
        out("")
        mono = all(crosses[k + 1] >= crosses[k] - 1e-9
                   for k in range(len(crosses) - 1))
        neg_any = any(c < 0.0 for c in crosses)
        out(f"- **crossover 验证**：D(b_setup) 单调不减 = {mp(mono)}；"
            f"存在 b_setup 使 CR 省 bits（D<0）= {mp(neg_any)}"
            f"（b_setup=16/32 时 CR 贵 = 与主 Gate 一致；小 b_setup 是否翻转为 CR 省 "
            f"bits 即理论 crossover）。")
        out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    out("- **B0.6 结论（014 §4/§7 诚实口径）**：若 E[B]^CR ≥ E[B]^Direct8——"
        "phase transition 与 state-dependent adaptive packetization 理论成立，"
        "但在当前 b_setup=16 regime 下 optimized direct packetization 已接近或达到"
        "最优通信工作区间；CR 的保守 certified acquisition（base by default）会"
        "过度投资证据（bits），QoS 反而不输。不再改算法‘调赢’；regime map 是预先"
        "声明的 secondary analysis。")
    out("")

    # FULL report hash guard (B0.4s convention)
    full_rp = os.path.join(OUT_DIR, "MVS-B0.6_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.6_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
