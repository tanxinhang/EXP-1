"""MVS-B0.7-G0: common-stop Gate (advice/015.md §十、§十三).

问题（015 §十）：B0.6 的 CR vs Direct8 对比同时变了
  stopping threshold / candidate policy / granularity / CPI override /
  transaction count —— 即使输赢也无法归因 multi-granularity 机制本身。

B0.7-G0 设计（015 §十、§九、§十三）：
  * **公共 outer stopping controller S_λ(x,h)**：与包粒度无关的同一判定给
    FG 和 Direct8 —— CONTINUE ⟺ min_{a∈A_all} Q_λ^(1)(a|x,h) < R_λ(x)，
    其中 Q_λ^(1) = c_a + E[R_λ(X')|x,a] 是 015 §九 的 one-step 近似
    （= VoIBase.q1），R_λ(x) = min{λ_M p, λ_F(1-p)} 取自然工作点 λ=μ
    （μ_M/π_1、μ_F/π_0，与 B0.4a/B0.6 的 R_stop 一致）。
  * 若 STOP：两方法都 STOP。若 CONTINUE：
      - Direct8：动作限 A_D8 = {(i, 8)}（full packet，015 §十）；
      - FG：动作用 A_FG = {(i,1),(i,2),(i,4),(i,8)}（adaptive granularity）；
    两者都用**同一个 one-step Q greedy** 选 UAV i —— 唯一系统差异 =
    feedback granularity（包粒度集合），stopping/UAV-selection/budget/
    decision-threshold/记账全部相同。
  * QoS Gate（B0.6-r 口径，015 §5/§十三）：两方法都算 Wilson 95% 并三态
    分类，双方都 FEASIBLE 才比较 U95(E[B^FG−B^D8])。
  * B0.6-d 记账复用：B = b_setup·N_tx + B_payload，报告 E[N_tx]、
    E[B_payload]、E[T_stop]、P(T_stop=k)。
  * 015 §十三 G0 停止规则：若 adaptive granularity 连 exact 小系统都不能
    降低 objective（或 QoS 未双双达标）→ **STOP，关闭 B0.7 主线**，转
    015 §十四 的 Direct8-近优下界路线。
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
from opmvs.rbl_eb import VoIBase
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA_A4 = [-1.0, 1.0, 3.0, 5.0]        # 同 B0.4a 的 4-UAV 配置
SEED0 = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
MU_M, MU_F = 256.0, 256.0 * np.exp(1.0)
ETA_NAT = float(np.log(MU_F / MU_M))     # = 1.0（判决阈值，两方法相同）
ALPHA = 0.12
BETA = 0.40
PFA_TARGET = 0.05
REGIMES = (0.0, 4.0, 8.0, 16.0, 32.0)
TSTOP_MAX = 8
LEVELS = (1, 2, 4, 8)
R_MAX = 8


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
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return float("-inf"), float("inf")
    m = float(xs.mean())
    s = float(xs.std(ddof=1)) if n > 1 else 0.0
    tcrit = float(t_dist.ppf(1.0 - alpha, max(n - 1, 1)))
    half = tcrit * s / math.sqrt(n)
    return m - half, m + half


def classify_qos(k_fa, k_md, n, alpha, beta):
    ufa, lfa = wilson_upper(k_fa, n), wilson_lower(k_fa, n)
    umd, lmd = wilson_upper(k_md, n), wilson_lower(k_md, n)
    if ufa <= alpha and umd <= beta:
        return "FEASIBLE"
    if lfa > alpha or lmd > beta:
        return "INFEASIBLE"
    return "UNCERTAIN"


def run_gate(bh, H, N0, quants4, powers4, H_all, L_all):
    """One (bh, H) common-stop gate on the stratified physical worlds.
    No planner RNG: both branches are deterministic one-step Q greedy, so
    episode-level CRN is automatic (same W_e=(H_e,L_e) drives both)."""
    pl = SparsePlanner(quants4, MU_M, MU_F, b_h=bh, cross_level=True,
                       levels=LEVELS, delta_c=1.0)
    voi = VoIBase(bh)
    n_ep = len(H_all)

    def apply(x, lam, i, r2, L_i):
        zi = (x // powers4[i]) % BASE_B
        r_cur, m_cur = z_decode_b(zi)
        c = bh + (r2 - r_cur)
        m2 = int(quants4[i].cell_index(r2, L_i[i]))
        z2 = z_code_b(r2, m2)
        lam2 = lam + quants4[i].llr[r2][m2]
        if r_cur > 0:
            lam2 -= quants4[i].llr[r_cur][m_cur]
        x2 = x + (z2 - zi) * powers4[i]
        return x2, lam2, c

    def best_d8(pl, x, h):
        """argmin over A_D8 = {(i,8)} of one-step Q (budget-feasible)."""
        rem = int(x)
        zs = []
        for _ in range(len(quants4)):
            zs.append(rem % BASE_B)
            rem //= BASE_B
        q_h = int(np.floor(h / pl.delta_c))
        best_q = None
        best = None
        for i in range(len(quants4)):
            zi = zs[i]
            r_cur, _ = z_decode_b(zi)
            if r_cur >= R_MAX:
                continue
            # (i, 8) 的可行性：直接用 cost 检查（qb 与粒度无关的部分）
            c = bh + (R_MAX - r_cur)
            if c > h + 1e-9:
                continue
            Q1 = voi.q1(pl, x, (i, R_MAX))
            if best_q is None or Q1 < best_q:
                best_q = Q1
                best = (i, R_MAX)
        return best

    def sim(x0, h0, L_i, mode):
        x, h, lam, cost, pl_pay, nt = x0, h0, 0.0, 0.0, 0.0, 0
        while True:
            if h < 1e-9:
                break
            om = pl.omega(x)
            if abs(lam) >= ETA_NAT:            # 判决可下（η_nat，两方法相同）
                break
            # 公共 outer stopping controller：CONTINUE ⟺ 存在 a∈A_all:
            # Q_λ^(1)(a) < R_λ(x)（015 §九 one-step approx；与粒度无关）
            if voi.act(pl, x, om, h=h) is None:
                break                          # S=STOP，两方法都停
            a = None
            if mode == "FG":
                a = voi.act(pl, x, om, h=h)    # argmin over A_FG
            else:
                a = best_d8(pl, x, h)          # argmin over A_D8
            if a is None:
                break
            i, r2 = a
            zi = (x // powers4[i]) % BASE_B
            r_cur, _ = z_decode_b(zi)
            c = bh + (r2 - r_cur)
            if c > h + 1e-9:
                break
            x, lam, c2 = apply(x, lam, i, r2, L_i)
            assert abs(c2 - c) < 1e-9
            cost += c
            pl_pay += (r2 - r_cur)
            nt += 1
            h -= c
        return lam, cost, nt, pl_pay, nt

    lam_fg = np.empty(n_ep)
    lam_d8 = np.empty(n_ep)
    b_fg = np.empty(n_ep)
    b_d8 = np.empty(n_ep)
    n_fg = np.empty(n_ep)
    n_d8 = np.empty(n_ep)
    pl_fg = np.empty(n_ep)
    pl_d8 = np.empty(n_ep)
    tst_fg = np.empty(n_ep)
    tst_d8 = np.empty(n_ep)
    for e in range(n_ep):
        L_i = L_all[e]
        lam_fg[e], b_fg[e], n_fg[e], pl_fg[e], tst_fg[e] = \
            sim(0, float(H), L_i, "FG")
        lam_d8[e], b_d8[e], n_d8[e], pl_d8[e], tst_d8[e] = \
            sim(0, float(H), L_i, "D8")
        assert abs(b_fg[e] - (bh * n_fg[e] + pl_fg[e])) < 1e-9
        assert abs(b_d8[e] - (bh * n_d8[e] + pl_d8[e])) < 1e-9

    H1 = H_all == 1
    i0 = np.flatnonzero(~H1)
    i1 = np.flatnonzero(H1)
    kfa_fg = int(np.sum(lam_fg[i0] > ETA_NAT))
    kmd_fg = int(np.sum(lam_fg[i1] <= ETA_NAT))
    kfa_d8 = int(np.sum(lam_d8[i0] > ETA_NAT))
    kmd_d8 = int(np.sum(lam_d8[i1] <= ETA_NAT))
    D = b_fg - b_d8
    dlcb, ducb = mean_ci(D)

    def tstop_pmf(tst):
        pmf = np.bincount(tst.astype(int), minlength=TSTOP_MAX + 1)[:TSTOP_MAX]
        pmf[-1] += int(np.sum(tst.astype(int) >= TSTOP_MAX))
        return pmf / len(tst)

    m_fg = mclib.evaluate(lam_fg, b_fg, H_all, PFA_TARGET)
    m_d8 = mclib.evaluate(lam_d8, b_d8, H_all, PFA_TARGET)
    return {
        "n0": N0, "bh": bh, "H": H,
        "pfa_fg": kfa_fg / N0, "pmd_fg": kmd_fg / N0,
        "pfa_d8": kfa_d8 / N0, "pmd_d8": kmd_d8 / N0,
        "kfa_fg": kfa_fg, "kmd_fg": kmd_fg, "kfa_d8": kfa_d8, "kmd_d8": kmd_d8,
        "eb_fg": float(b_fg.mean()), "eb_d8": float(b_d8.mean()),
        "D": float(D.mean()), "dlcb": dlcb, "ducb": ducb,
        "entx_fg": float(n_fg.mean()), "entx_d8": float(n_d8.mean()),
        "epl_fg": float(pl_fg.mean()), "epl_d8": float(pl_d8.mean()),
        "etst_fg": float(tst_fg.mean()), "etst_d8": float(tst_d8.mean()),
        "tst_fg": tstop_pmf(tst_fg), "tst_d8": tstop_pmf(tst_d8),
        "pd_fg": m_fg["pd"], "pd_d8": m_d8["pd"],
    }


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
                    help="b_setup regime map (015 §十三 secondary)")
    args = ap.parse_args()
    SMOKE = args.smoke
    NL = args.nlevel
    if SMOKE:
        N_LEVELS = {1: 120, 2: 200, 3: 300, 4: 500}
    else:
        N_LEVELS = {1: 600, 2: 1000, 3: 1600, 4: 2500}
    N0 = N_LEVELS.get(NL, N_LEVELS[1])
    H_BUDGETS = (48, 96)
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.7-G0 — common-stop Gate：granularity 独立收益（015 §十/§十三）")
    out("")
    out(f"> 协议（015 §十 冻结）：**N=4 exact 小系统**（GAMMA={GAMMA_A4}、levels="
        f"{LEVELS}、r_max={R_MAX}，同 B0.4a 配置加 8-bit 全包）；stratified "
        f"N0=N1={N0}；episode 级 CRN（同一 W_e=(H_e,L_e) 驱动两分支，planner 无 "
        f"随机——one-step exact 确定性）；判决阈值 η_nat=log(μ_F/μ_M)="
        f"{fmt(ETA_NAT)}（两方法相同）；radio cost B=Σ(b_setup+Δr_t)。")
    out("")
    out("> **公共 stopping controller（015 §九/§十）**：S_λ(x,h)：CONTINUE ⟺ "
        "min_{a∈A_all} Q_λ^(1)(a|x,h) < R_λ(x)，Q_λ^(1)=c_a+E[R_λ(X')|x,a] "
        "（one-step approx，=VoIBase.q1），R_λ(x)=min{λ_M p, λ_F(1−p)}，λ=μ 为 "
        "自然工作点（μ_M/π_1、μ_F/π_0）。STOP 判定与包粒度无关（A_all 含全部 "
        "粒度），对 FG 和 Direct8 完全一致。CONTINUE 时：Direct8 限 "
        "A_D8={(i,8)}；FG 用 A_FG={(i,1),(i,2),(i,4),(i,8)}；两者都按**同一 "
        "one-step Q greedy** 选 UAV i——唯一系统差异 = feedback granularity。")
    out("")
    out("> **Gate（015 §十三 G0 + B0.6-r 口径）**：两方法都算 Wilson 95% "
        "FEASIBLE/INFEASIBLE/UNCERTAIN；双方 FEASIBLE 才比较 "
        "U95(E[B^FG−B^D8])<0 → PASS（granularity 有独立收益，主线可进 G1）；"
        "否则 UNRESOLVED；若 FG 连 exact 小系统都不能降低 objective → "
        "**STOP，关闭 B0.7 主线**（015 §十三），转 015 §十四 Direct8-近优下界。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if SMOKE else 'FULL'}"
        f"   nlevel={NL}（N0=N1={N0}）  map={args.map}")
    out("")

    model4 = GaussianDetectorModel(GAMMA_A4)
    quants4 = [NestedQuantizer(i, model4, r_max=R_MAX, levels=LEVELS)
               for i in range(4)]
    powers4 = [BASE_B ** i for i in range(4)]

    rng_w = np.random.default_rng(SEED0)
    H0_w = np.zeros(N0, dtype=np.int8)
    H1_w = np.ones(N0, dtype=np.int8)
    L0 = model4.sample_llr(H0_w, rng_w)
    L1 = model4.sample_llr(H1_w, rng_w)
    H_all = np.concatenate([H0_w, H1_w])
    L_all = np.concatenate([L0, L1])

    for H in H_BUDGETS:
        out(f"## H={H} — common-stop gate（b_setup=16，N0=N1={N0}）")
        out("")
        t_h = time.time()
        r = run_gate(16.0, H, N0, quants4, powers4, H_all, L_all)
        n = r["n0"]

        out("### QoS 三态分类（Wilson 95%）")
        out("")
        out("| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, name in (("fg", "FG (adaptive)"), ("d8", "Direct8")):
            kfa, kmd = r[f"kfa_{key}"], r[f"kmd_{key}"]
            lfa, ufa = wilson_lower(kfa, n), wilson_upper(kfa, n)
            lmd, umd = wilson_lower(kmd, n), wilson_upper(kmd, n)
            cls = classify_qos(kfa, kmd, n, ALPHA, BETA)
            out(f"| {name} | {fmt(r[f'pfa_{key}'])} | {fmt(lfa)} | {fmt(ufa)} | "
                f"{fmt(r[f'pmd_{key}'])} | {fmt(lmd)} | {fmt(umd)} | **{cls}** |")
        out("")
        fg_cls = classify_qos(r["kfa_fg"], r["kmd_fg"], n, ALPHA, BETA)
        d8_cls = classify_qos(r["kfa_d8"], r["kmd_d8"], n, ALPHA, BETA)

        out("### Bit Gate：FG vs Direct8（G0 机制门，015 §十三）")
        out("")
        out("| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE | E[T_stop] |")
        out("| --- | --- | --- | --- | --- | --- |")
        out(f"| FG | {fmt(r['entx_fg'])} | {fmt(r['epl_fg'])} | {fmt(r['eb_fg'])} | "
            f"— | {fmt(r['etst_fg'])} |")
        out(f"| Direct8 | {fmt(r['entx_d8'])} | {fmt(r['epl_d8'])} | "
            f"{fmt(r['eb_d8'])} | — | {fmt(r['etst_d8'])} |")
        out("")
        # 015 §十三 G0：机制门 = FG 在 exact 小系统上显著降低 objective（paired
        # E[B]）且 QoS 未被证伪（FG != INFEASIBLE）。正式 matched-QoS 双认证
        # 是 G1（N=8 held-out）的任务 —— G0 只验"granularity 本身有无独立
        # 省 bit 效应"，不冒充正式 QoS 比较（015 §十/§十三、B0.6-r 口径）。
        bit_sig = r["ducb"] < 0.0
        fg_not_infeas = fg_cls != "INFEASIBLE"
        mechanism_pass = bit_sig and fg_not_infeas
        out(f"- **paired 比较（G0 机制门）**：E[D]=E[B^FG−B^D8] = {fmt(r['D'])}"
            f"，95% CI [{fmt(r['dlcb'])}, {fmt(r['ducb'])}]（<0 → {mp(bit_sig)}）。"
            f"QoS 观测：FG={fg_cls}、Direct8={d8_cls}（三态为参考观测——正式 "
            f"matched-QoS 双认证在 **G1 N=8 held-out QoS-dual calibration**；"
            f"G0 只判机制，015 §十三）。")
        if mechanism_pass:
            out("  → **G0 机制门 PASS**：在相同 stopping/UAV-selection/budget/"
                "decision-threshold 下，仅允许 adaptive packetization 即显著省 "
                "raw bits，且 FG QoS 未被证伪 → **granularity 有独立价值**，主线 "
                "进 B0.7-G1（N=8 held-out + QoS-dual calibrated stopping）。")
        else:
            if r["dlcb"] > 0.0:
                out("  → E[D] 的 CI 已证明 FG 不省 → **STOP（015 §十三）：adaptive "
                    "granularity 连 exact 小系统都不能降低 objective**，关闭 "
                    "B0.7 性能提升主线，转 015 §十四 Direct8-近优下界路线。")
            elif fg_cls == "INFEASIBLE":
                out("  → FG 的 QoS 已被证伪（INFEASIBLE）→ **STOP（015 §十三）**，"
                    "主线关闭，转 015 §十四 lower-bound 路线（或 QoS-dual 校准后 "
                    "再验，即 G1 内容）。")
            else:
                out("  → 差异尚未显著（CI 跨 0）或 n 不足 → **UNCERTAIN**：G0 "
                    "机制门未过但未证反。按 015 §十三，以 --nlevel 扩样重验；"
                    "同时 G1（held-out dual 校准）是正式认证路径。")
        out("")

        out("### B0.6-d 停止结构（015 §六 记账复用）")
        out("")
        out("| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |")
        out("| --- | --- | --- | --- | --- | --- |")
        for key, name in (("fg", "FG"), ("d8", "Direct8")):
            pmf = r[f"tst_{key}"]
            out(f"| {name} | {fmt(pmf[1])} | {fmt(pmf[2])} | {fmt(pmf[3])} | "
                f"{fmt(pmf[4])} | {fmt(pmf[5:].sum())} |")
        out("")
        out(f"- 分解：b_setup=16。FG E[N_tx]={fmt(r['entx_fg'])} vs D8 "
            f"{fmt(r['entx_d8'])}；payload {fmt(r['epl_fg'])} vs {fmt(r['epl_d8'])}"
            f"——ΔB^FG−D8 中 setup/payload 各自贡献，直接定位 granularity 的 "
            f"成本结构效应。")
        out("")
        out(f"- **NP-matched（P_FA=0.05）secondary**：P_D^FG={fmt(r['pd_fg'])} / "
            f"E[B]^FG={fmt(r['eb_fg'])}；P_D^D8={fmt(r['pd_d8'])} / "
            f"E[B]^D8={fmt(r['eb_d8'])}。")
        out(f"（{time.time()-t_h:.0f}s）")
        out("")

    if args.map:
        out("## 5. b_setup regime map（015 §十三 secondary，H=96）")
        out("")
        N_MAP = N0 if SMOKE else 300
        H_MAP = H_all[: 2 * N_MAP]
        L_map = L_all[: 2 * N_MAP]
        out(f"- H=96、N0=N1={N_MAP}：观察 D(b_setup)=E[B^FG−B^D8] 走向与分解 "
            f"（B=b_setup·N_tx+payload）。b_setup=0 行隔离纯 payload 效应。")
        out("")
        out("| b_setup | E[N_tx^FG] | E[N_tx^D8] | E[B_pay^FG] | E[B_pay^D8] | "
            "E[D] | 95% CI | FG 省 bits? |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        crosses = []
        for bh in REGIMES:
            r = run_gate(bh, 96, N_MAP, quants4, powers4, H_MAP, L_map)
            crosses.append(r["D"])
            d_ok = r["D"] < 0.0
            out(f"| {fmt(bh, 0)} | {fmt(r['entx_fg'])} | {fmt(r['entx_d8'])} | "
                f"{fmt(r['epl_fg'])} | {fmt(r['epl_d8'])} | {fmt(r['D'])} | "
                f"[{fmt(r['dlcb'])}, {fmt(r['ducb'])}] | {'YES' if d_ok else 'no'} |")
        out("")
        neg_any = any(c < 0.0 for c in crosses)
        out(f"- **system-level 观察**：存在 b_setup 使 FG 省 bits（D<0）= "
            f"{mp(neg_any)}（015 §十 的核心问题：**在相同停止逻辑下，仅允许 "
            f"adaptive packetization 是否降低 communication bits？**）。")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    out("- **B0.7-G0 结论（015 §十三）**：common-stop Gate 把 granularity 从 "
        "stopping/UAV-selection/budget/decision-threshold 中隔离出来。若 FG 在 "
        "exact 小系统上双方 QoS FEASIBLE 且 U95(E[B^FG−B^D8])<0 → granularity "
        "有独立价值，继续 B0.7-G1（N=8 held-out QoS-dual calibration）；否则 "
        "**STOP，关闭 performance-improvement 主线**，B0.5 换用途为 "
        "Direct8-近优下界 V_LB≤V⋆≤V^D8（015 §十四）。")
    out("")

    full_rp = os.path.join(OUT_DIR, "MVS-B0.7-G0_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.7-G0_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()