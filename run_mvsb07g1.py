"""MVS-B0.7-G1: N=8 held-out QoS-dual calibrated common-stop Gate
(advice/015.md §七-§九、§十三 B0.7-G1).

在 G0 之上做正式 matched-QoS 认证：
  * **双参数（QoS-dual prices）只在 calibration seeds 上确定**，test seeds
    完全 fresh（015 §十三 G1）——从结构上排除 "看到 D8 赢后再改阈值" 的
    post-hoc tuning 嫌疑（015 §七）。
  * 校准框架（015 §七-§九）：
      min_π E_π[B]  s.t.  P_FA^π<=α, P_MD^π<=β
      Lagrangian: L = E[B] + λ_F(P_FA-α) + λ_M(P_MD-β)
      terminal dual risk:  R_λ(x) = min{ λ_M p, λ_F(1-p) }
      one-step:            Q_λ^(1)(a|x,h) = c_a + E[R_λ(X')|x,a]
      STOP <=> R_λ(x) <= min_a Q_λ^(1)(a|x,h)   （015 §九；无 |Ω|>=κ 对称停止）
      判决: Ω > log(λ_F/λ_M) -> H1。
  * 参数化：λ_M 固定标度 LAM_M=512（=G0 自然点标度），扫
    eta_dec = log(λ_F/λ_M) ∈ GRID_ETA；校准集上选 "两方法都 FEASIBLE
    (Wilson U95)" 且 E[B^FG]+E[B^D8] 最小的 eta_star；冻结。
  * 两方法共用**同一** S_λ 与同一判决（common-stop，015 §十），唯一
    系统差异 = feedback granularity（FG: A={(i,1),(i,2),(i,4),(i,8)}；
    D8: A={(i,8)}）。注意 A_FG = A_all，因此 S 判定与 FG 动作选择合并为
    单次 A_all 遍历（快速实现）。
  * N=8（015 §十三 G1），GAMMA_B 同 B0.6，levels=(1,2,4,8)，b_setup=16。
  * test Gate（015 §十三 G1）：两方法均 U95(P_FA)<=α=0.12 且
    U95(P_MD)<=β=0.40（FEASIBLE）才比较 U95(E[B^FG-B^D8])<0 -> PASS。
  * 记账复用 B0.6-d：B=b_setup·N_tx+payload、E[N_tx]、E[T_stop]、
    P(T_stop=k)。恒等式逐样本断言。
  * 速度：one-step 枚举改用 Python math 标量（np 标量调用开销 ~10-20x
    math），A_FG=A_all 合一次遍历；校准 N_CAL=300（015 只要求参数在
    calibration 上确定，未限定样本量），test N_TEST=600 正式 Gate。
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
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
SEED0 = 2026
SEED_CAL = SEED0 + 100
SEED_TEST = SEED0 + 200
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
LAM_M = 512.0
ALPHA = 0.12
BETA = 0.40
BH = 16.0
PFA_TARGET = 0.05
LEVELS = (1, 2, 4, 8)
R_MAX = 8
GRID_ETA = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
TSTOP_MAX = 8
N_UAV = 8


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


def r_lam(om, lam_f):
    """R_λ(x) = min{λ_M p, λ_F(1-p)}; λ_M 固定标度，λ_F=λ_M·e^{eta}."""
    p = 1.0 / (1.0 + math.exp(-om))
    return min(LAM_M * p, lam_f * (1.0 - p))


def q1_fast(pl, x, om, i, r2, lam_f):
    """Q_λ^(1)(a|x,h) = c_a + E[R_λ(X')|x,a]，Python-math 标量实现（数
    值与 VoIBase.q1 的 np 版等价；每 cell 的混合权重 = P(H1)p(cell|H1)
    + P(H0)p(cell|H0)，用 logsumexp 形式计算）。"""
    zi = (x // pl.powers[i]) % BASE_B
    r_cur, _ = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    cells = next(cells for (r2b, _ct, _qb, cells) in pl._tpl[i][zi]
                 if r2b == r2)
    lp = -math.log1p(math.exp(-om))       # log σ(om)
    lq = -math.log1p(math.exp(om))        # log(1-σ(om))
    E = 0.0
    for (m2, lp0c, lp1c) in cells:
        a_ = lp + lp1c
        b_ = lq + lp0c
        m_ = a_ if a_ >= b_ else b_
        w = math.exp(m_ + math.log1p(math.exp(-abs(a_ - b_))))
        z2 = z_code_b(r2, m2)
        om_c = om + pl._llr_i[i][z2] - pl._llr_i[i][zi]
        E += w * r_lam(om_c, lam_f)
    return c + E


def sim_branch(pl, eta, H, L_i, mode, quants8, powers8):
    """One episode under the common S_λ (A_FG=A_all 合并遍历).
    mode: "FG"（A_FG 选择 = A_all argmin）或 "D8"（A_D8={(i,8)} argmin）。
    返回 (lam, cost, n_tx, payload, n_tx)。"""
    lam_f = LAM_M * math.exp(eta)
    x, h, lam, cost, pay, nt = 0, float(H), 0.0, 0.0, 0.0, 0
    while True:
        if h < 1e-9:
            break
        om = pl.omega(x)
        R = r_lam(om, lam_f)
        # A_all = A_FG 遍历：公共 S 判定 + FG 动作合一
        rem = int(x)
        zs = []
        for _ in range(N_UAV):
            zs.append(rem % BASE_B)
            rem //= BASE_B
        best_q = None
        best_a = None
        for i in range(N_UAV):
            zi = zs[i]
            r_cur, _ = z_decode_b(zi)
            for (r2, c_true, _qb, _cells) in pl._tpl[i][zi]:
                if c_true > h:
                    continue
                q = q1_fast(pl, x, om, i, r2, lam_f)
                if best_q is None or q < best_q:
                    best_q = q
                    best_a = (i, r2)
        if best_q is None or best_q >= R:
            break                              # 公共 S：STOP
        if mode == "D8":
            # D8 分支：只能在 A_D8={(i,8)} 选 argmin
            best_q2 = None
            best_a2 = None
            for i in range(N_UAV):
                r_cur, _ = z_decode_b(zs[i])
                if r_cur >= R_MAX:
                    continue
                c = BH + (R_MAX - r_cur)
                if c > h:
                    continue
                q = q1_fast(pl, x, om, i, R_MAX, lam_f)
                if best_q2 is None or q < best_q2:
                    best_q2 = q
                    best_a2 = (i, R_MAX)
            if best_a2 is None:
                break
            best_a = best_a2
        i, r2 = best_a
        zi = zs[i]
        r_cur, m_cur = z_decode_b(zi)
        c = BH + (r2 - r_cur)
        if c > h + 1e-9:
            break
        m2 = int(quants8[i].cell_index(r2, L_i[i]))
        lam2 = lam + quants8[i].llr[r2][m2]
        if r_cur > 0:
            lam2 -= quants8[i].llr[r_cur][m_cur]
        z2 = z_code_b(r2, m2)
        x2 = x + (z2 - zi) * powers8[i]
        cost += c
        pay += (r2 - r_cur)
        nt += 1
        h -= c
        lam, x = lam2, x2
    return lam, cost, nt, pay, nt


def run_pass(eta, H, H_all, L_all, quants8, powers8):
    """両分支在一个 stratified episode 集上运行（同一 W_e=(H_e,L_e)，CRN）。"""
    pl = SparsePlanner(quants8, LAM_M, LAM_M * math.exp(eta), b_h=BH,
                       cross_level=True, levels=LEVELS, delta_c=1.0)
    n_ep = len(H_all)
    n0 = n_ep // 2
    lam_fg = np.empty(n_ep); b_fg = np.empty(n_ep)
    n_fg = np.empty(n_ep); pl_fg = np.empty(n_ep); tst_fg = np.empty(n_ep)
    lam_d8 = np.empty(n_ep); b_d8 = np.empty(n_ep)
    n_d8 = np.empty(n_ep); pl_d8 = np.empty(n_ep); tst_d8 = np.empty(n_ep)
    for e in range(n_ep):
        L_i = L_all[e]
        lam_fg[e], b_fg[e], n_fg[e], pl_fg[e], tst_fg[e] = sim_branch(
            pl, eta, H, L_i, "FG", quants8, powers8)
        lam_d8[e], b_d8[e], n_d8[e], pl_d8[e], tst_d8[e] = sim_branch(
            pl, eta, H, L_i, "D8", quants8, powers8)
        assert abs(b_fg[e] - (BH * n_fg[e] + pl_fg[e])) < 1e-9
        assert abs(b_d8[e] - (BH * n_d8[e] + pl_d8[e])) < 1e-9
    H1 = H_all == 1
    i0 = np.flatnonzero(~H1)
    i1 = np.flatnonzero(H1)
    kfa_fg = int(np.sum(lam_fg[i0] > eta))
    kmd_fg = int(np.sum(lam_fg[i1] <= eta))
    kfa_d8 = int(np.sum(lam_d8[i0] > eta))
    kmd_d8 = int(np.sum(lam_d8[i1] <= eta))
    D = b_fg - b_d8
    dlcb, ducb = mean_ci(D)

    def tstop_pmf(tst):
        pmf = np.bincount(tst.astype(int), minlength=TSTOP_MAX + 1)[:TSTOP_MAX]
        pmf[-1] += int(np.sum(tst.astype(int) >= TSTOP_MAX))
        return pmf / len(tst)

    m_fg = mclib.evaluate(lam_fg, b_fg, H_all, PFA_TARGET)
    m_d8 = mclib.evaluate(lam_d8, b_d8, H_all, PFA_TARGET)
    return {
        "n0": n0, "eta": eta, "H": H,
        "pfa_fg": kfa_fg / n0, "pmd_fg": kmd_fg / n0,
        "pfa_d8": kfa_d8 / n0, "pmd_d8": kmd_d8 / n0,
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
    args = ap.parse_args()
    SMOKE = args.smoke
    NL = args.nlevel
    if SMOKE:
        N_LEVELS = {1: 120, 2: 200, 3: 300, 4: 500}
    else:
        N_LEVELS = {1: 600, 2: 1000, 3: 1600, 4: 2500}
    N_TEST = N_LEVELS.get(NL, N_LEVELS[1])
    N_CAL = N_TEST // 2          # 校准轻量（015 只要求参数在 cal 上确定）
    H_BUDGETS = (48, 96)
    CAL_H = 96
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.7-G1 — N=8 held-out QoS-dual calibrated common-stop "
        "Gate（015 §七-§九/§十三）")
    out("")
    out(f"> 协议（015 §十三 G1 冻结）：**N=8**（GAMMA={GAMMA_B}、levels={LEVELS}、"
        f"r_max={R_MAX}）；b_setup={BH}；stratified N0=N1={N_TEST}（calibration "
        f"N_CAL={N_CAL} @ H={CAL_H}，test N_TEST={N_TEST} @ H∈{H_BUDGETS}）；"
        f"episode 级 CRN（同一 W_e=(H_e,L_e) 给 FG/D8；planner 确定性）。")
    out(f"> **QoS-dual 校准（015 §七-§九）**：λ_M={LAM_M}（标度固定），扫 "
        f"η_dec=log(λ_F/λ_M) ∈ {GRID_ETA}；终止风险 R_λ(x)=min{{λ_M p, "
        f"λ_F(1-p)}}，单步继续值 Q_λ^(1)(a|x,h)=c_a+E[R_λ(X')|x,a]；**STOP "
        f"⟺ R_λ(x) ≤ min_a Q_λ^(1)**（无 |Ω|≥κ 对称停止，015 §三）；判决 "
        f"Ω>η_dec→H1。两方法共用同一 S_λ/判决；只选 calibration 上双方 "
        f"FEASIBLE 且 E[B^FG]+E[B^D8] 最小的 η_star 冻结（**双参数不在 test 上"
        f"触碰**——015 §七 的 anti-post-hoc 结构）。")
    out("")
    out(f"> **Gate（015 §十三 G1）**：test 上两方法均 U95(P_FA)≤{ALPHA} 且 "
        f"U95(P_MD)≤{BETA}（FEASIBLE）才比较 U95(E[B^FG−B^D8])<0 → PASS；"
        f"任一 INFEASIBLE → 该行不具 matched 地位；双方 UNCERTAIN → UNRESOLVED "
        f"（--nlevel 扩样）。")
    out(f"> 记账：B=b_setup·N_tx+payload（逐样本断言）；E[N_tx]、E[B_payload]、"
        f"E[T_stop]、P(T_stop=k)。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: "
        f"{'SMOKE' if SMOKE else 'FULL'}   nlevel={NL}（N_TEST={N_TEST}，"
        f"N_CAL={N_CAL}）")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=LEVELS)
               for i in range(8)]
    powers8 = [BASE_B ** i for i in range(8)]

    def sample(Nn, seed):
        rng = np.random.default_rng(seed)
        H0 = np.zeros(Nn, dtype=np.int8)
        H1 = np.ones(Nn, dtype=np.int8)
        L0 = model8.sample_llr(H0, rng)
        L1 = model8.sample_llr(H1, rng)
        return np.concatenate([H0, H1]), np.concatenate([L0, L1])

    H_cal, L_cal = sample(N_CAL, SEED_CAL)
    H_t48, L_t48 = sample(N_TEST, SEED_TEST * 1000 + 1)
    H_t96, L_t96 = sample(N_TEST, SEED_TEST * 1000 + 2)

    # ------------------------------------------------ calibration：选 η_star
    out("## 1. Calibration（calibration seeds，H=96）— 求 η_star")
    out("")
    out("| η_dec | 方法 | P_FA | U95(P_FA) | P_MD | U95(P_MD) | 分类 | E[B] |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- |")
    cal_costs = {}
    for eta in GRID_ETA:
        r = run_pass(eta, CAL_H, H_cal, L_cal, quants8, powers8)
        for key, name in (("fg", "FG"), ("d8", "D8")):
            kfa, kmd = r[f"kfa_{key}"], r[f"kmd_{key}"]
            ufa, lfa = wilson_upper(kfa, N_CAL), wilson_lower(kfa, N_CAL)
            umd, lmd = wilson_upper(kmd, N_CAL), wilson_lower(kmd, N_CAL)
            cls = classify_qos(kfa, kmd, N_CAL, ALPHA, BETA)
            out(f"| {fmt(eta, 1)} | {name} | {fmt(r[f'pfa_{key}'])} | "
                f"{fmt(ufa)} | {fmt(r[f'pmd_{key}'])} | {fmt(umd)} | "
                f"{cls} | {fmt(r[f'eb_{key}'])} |")
        out("")
        cls_fg = classify_qos(r["kfa_fg"], r["kmd_fg"], N_CAL, ALPHA, BETA)
        cls_d8 = classify_qos(r["kfa_d8"], r["kmd_d8"], N_CAL, ALPHA, BETA)
        if cls_fg == "FEASIBLE" and cls_d8 == "FEASIBLE":
            cal_costs[eta] = r["eb_fg"] + r["eb_d8"]
    if cal_costs:
        eta_star = min(cal_costs, key=cal_costs.get)
        out(f"- **η_star = {fmt(eta_star, 1)}**（校准集上双方 FEASIBLE 且 "
            f"E[B^FG]+E[B^D8]={fmt(cal_costs[eta_star])} 最小；其余达标 η："
            f"{', '.join(fmt(e, 1) + '(' + fmt(v) + ')' for e, v in sorted(cal_costs.items()))}）。"
            f"冻结，test 上**不再触碰**。")
    else:
        eta_star = None
        out(f"- **校准 UNRESOLVED**：GRID_ETA 中无 η_dec 使两方法在 calibration "
            f"上同时 FEASIBLE。诚实报告（015 §十三：无法认证 matched 比较 → "
            f"转 015 §十四 lower-bound 路线）。")
    out("")

    # ---------------------------------------------------------------- test
    if eta_star is None:
        out("## 2. Test — 跳过（无冻结 η_star）")
        out("")
        out(f"- 判定：**UNRESOLVED（校准失败）**——B0.7-G1 Gate 未建立，按 015 "
            f"§十三 转 lower-bound 路线。")
        out("")
    else:
        for H in H_BUDGETS:
            Ht, Lt = (H_t48, L_t48) if H == 48 else (H_t96, L_t96)
            out(f"## 2. Test（fresh seeds，H={H}，η_star={fmt(eta_star, 1)} 冻结）")
            out("")
            t_h = time.time()
            r = run_pass(eta_star, H, Ht, Lt, quants8, powers8)
            n = r["n0"]
            out("### QoS 三态分类（Wilson 95%，test）")
            out("")
            out("| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |")
            out("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for key, name in (("fg", "FG (adaptive)"), ("d8", "Direct8")):
                kfa, kmd = r[f"kfa_{key}"], r[f"kmd_{key}"]
                lfa, ufa = wilson_lower(kfa, n), wilson_upper(kfa, n)
                lmd, umd = wilson_lower(kmd, n), wilson_upper(kmd, n)
                cls = classify_qos(kfa, kmd, n, ALPHA, BETA)
                out(f"| {name} | {fmt(r[f'pfa_{key}'])} | {fmt(lfa)} | {fmt(ufa)} "
                    f"| {fmt(r[f'pmd_{key}'])} | {fmt(lmd)} | {fmt(umd)} | "
                    f"**{cls}** |")
            out("")
            fg_cls = classify_qos(r["kfa_fg"], r["kmd_fg"], n, ALPHA, BETA)
            d8_cls = classify_qos(r["kfa_d8"], r["kmd_d8"], n, ALPHA, BETA)
            out("### Bit Gate（015 §十三 G1：双方 FEASIBLE 才比较）")
            out("")
            out("| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | "
                "E[T_stop] |")
            out("| --- | --- | --- | --- | --- |")
            out(f"| FG | {fmt(r['entx_fg'])} | {fmt(r['epl_fg'])} | "
                f"{fmt(r['eb_fg'])} | {fmt(r['etst_fg'])} |")
            out(f"| Direct8 | {fmt(r['entx_d8'])} | {fmt(r['epl_d8'])} | "
                f"{fmt(r['eb_d8'])} | {fmt(r['etst_d8'])} |")
            out("")
            if fg_cls == "FEASIBLE" and d8_cls == "FEASIBLE":
                bit_pass = r["ducb"] < 0.0
                bit_fail = r["dlcb"] > 0.0
                verdict = "PASS" if bit_pass else ("FAIL" if bit_fail else "UNCERTAIN")
                out(f"- 双方 FEASIBLE → paired 比较：E[D]=E[B^FG−B^D8] = "
                    f"{fmt(r['D'])}，95% CI [{fmt(r['dlcb'])}, {fmt(r['ducb'])}]"
                    f"（<0 → {mp(bit_pass)}）→ **{verdict}**。")
                if bit_pass:
                    out("  → **matched-QoS 下 granularity 有独立收益**：N=8 "
                        "held-out、λ 只由 calibration 定，FG 显著省 bits → 主线"
                        "可进 B0.7-G2（frozen CPI override / fresh Gate）或直接"
                        "作为论文主线证据。")
                else:
                    out("  → FG 未在 matched-QoS 下显著省 bits → 按 015 §十三 "
                        "关闭 performance-improvement 主线，转 015 §十四 "
                        "lower-bound 路线。")
            else:
                out(f"- **Gate 拦截**：FG={fg_cls}、Direct8={d8_cls}——未双方 "
                    f"FEASIBLE，matched 比较不成立（B0.6-r 口径）。")
                out(f"  → 判定 **UNRESOLVED**（--nlevel 扩样；或 QoS 未达标时转 "
                    f"015 §十四 lower-bound 路线/G2 fresh Gate）。")
            out("")
            out("### 停止结构（B0.6-d 记账，test）")
            out("")
            out("| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) "
                "| P(T_stop≥5) |")
            out("| --- | --- | --- | --- | --- | --- |")
            for key, name in (("fg", "FG"), ("d8", "Direct8")):
                pmf = r[f"tst_{key}"]
                out(f"| {name} | {fmt(pmf[1])} | {fmt(pmf[2])} | {fmt(pmf[3])} "
                    f"| {fmt(pmf[4])} | {fmt(pmf[5:].sum())} |")
            out("")
            setup_d = BH * (r["entx_fg"] - r["entx_d8"])
            pay_d = r["epl_fg"] - r["epl_d8"]
            out(f"- 分解：E[D]={fmt(r['D'])}，其中 setup 部分 {BH}·(E[N_tx^FG]−"
                f"E[N_tx^D8])={fmt(setup_d)}，payload 部分 {fmt(pay_d)}。")
            out("")
            out(f"- **NP-matched（P_FA=0.05）secondary**：P_D^FG={fmt(r['pd_fg'])} "
                f"/ E[B]^FG={fmt(r['eb_fg'])}；P_D^D8={fmt(r['pd_d8'])} / "
                f"E[B]^D8={fmt(r['eb_d8'])}。")
            out(f"（{time.time() - t_h:.0f}s）")
            out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    out("- **B0.7-G1 结论（015 §十三）**：held-out 协议下 matched-QoS 双认证 "
        "+ paired bit 比较。若 test 双方 FEASIBLE 且 U95(E[B^FG−B^D8])<0 → "
        "**granularity 在正式 QoS 口径下有独立收益**（论文主线证据）；否则 "
        "**STOP / UNRESOLVED**，B0.5 换用途为 Direct8-近优下界 "
        "（V_LB≤V⋆≤V^D8，015 §十四）。")
    out("")

    full_rp = os.path.join(OUT_DIR, "MVS-B0.7-G1_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.7-G1_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()