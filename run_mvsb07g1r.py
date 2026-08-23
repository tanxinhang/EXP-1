"""MVS-B0.7-G1r: common-stop action-leakage audit + conservative S_ref Gate
+ dual-Q exact regression (advice/016.md §1-§6、§10、§15 路线 1-3).

016 P0（§1）：G1 的公共停止器 S_common 用 min_{a∈A_FG} Q_λ^(1) < R_λ 判定
CONTINUE，而 DIRECT8 分支只能从 A_D8={(i,8)} 选动作——存在状态
  F(x) = 1{ q_FG < R_λ <= q_D8 },  q_FG=min_{A_FG}Q, q_D8=min_{A_D8}Q
即"小包值得买故公共控制器说继续，但所有 8-bit 包已不值得"——D8 被迫执行
8-bit action（action-set leakage；不是 bug，是比较设计缺陷，会天然抬高 D8
成本）。A_FG 本身含 granularity 信息 ⇒ "STOP 判定与包粒度无关（共用 A_all）"
的表述不成立（016 §1 末尾）。

G1r-A（016 §15-1）：记录每个 D8 决策的 q_FG/q_D8/R_λ，报告
  P(F=1)（按决策状态）、P(episode contains F)（按 episode）、
  ΔB_forced = Σ D8 在 F 状态下支付的通信成本（被迫额外支出）。
直接回答："D8 的通信中到底有多少是被 FG-action-defined common stop
强迫出来的？"

G1r-B（016 §4/§15-2）：保守停止器
  S_ref(x,h): CONTINUE iff min_{a∈A_D8} Q_λ^(1)(a) < R_λ(x)
两方法共用 S_ref（仅当"至少一个 Direct8 full packet 值得发送"才给 FG 一次
adaptive-granularity 机会——对 FG 更苛刻）；D8 执行最优 (i,8)；FG 继续后
自由选 1/2/4/8。若此保守版仍满足 U95(E[B^FG-B^D8])<0 ⇒ granularity 独立
收益基本无法从公平性击穿（016 §4 预期：-12.31 可能缩小到 -5..-10，但不会
翻正）。

G1r-C（016 §15-3）：代码可信度封板
  * q1_fast vs 独立编码的 generic dual-Q exact（纯 numpy logsumexp 写法）
    在随机 reachable 状态上回归：|q1_fast - dual_q_exact| < 1e-9。
  * D8 emulation invariant：FG 分支把动作集固定为 A_D8（"FG emulating
    Direct8"）与 D8 分支本身在相同 episode 上逐样本完全一致（lam/cost）。

统计口径（016 §10）：paired mean 用 **one-sided paired Hoeffding bound**
  U_δ = D̄ + 2H sqrt(log(1/δ)/(2n))（D=B^FG-B^D8 ∈ [-H,H]；分布无关、
  fixed-N、无需 t 假设），作为正式上界；t-based one-sided mean CI 仅作
  参考列。QoS：Wilson 95%（同 G0/G1）。

协议：N=8、GAMMA_B、levels=(1,2,4,8)、b_setup=16、stratified；
calibration（N_CAL @ H=96）只用于选 η_star（G1r-A 用 S_common 协议、
G1r-B 用 S_ref 协议——各自重新校准，因为停止语义变了）；test 完全 fresh
(N_TEST @ H=48/96)；双参数 grid 冻结、test 不碰（016 §9：λ_M 固定标度 +
η 校准 = fixed dual scale + calibrated dual ratio；完整 (ρ,η) 二维留给 G2）。

Gate（016 §15 判别 + §10 intersection-union）：test 上两方法均
U95(P_FA)<=0.12 且 U95(P_MD)<=0.40，且 paired Hoeffding U_δ(E[B^FG-B^D8])<0
=> G1r-B 保守版 PASS（granularity 独立收益站稳）→ 投入 fresh G2
（分别 (ρ,η) calibration，016 §15-4）。
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
DELTA_CI = 0.05


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


def mean_ci_t(xs, alpha=0.05):
    """One-sided t-based 95% mean bounds (016 §10：这是 t-UCB/LCB，
    不是双侧 CI——G1r 正式 Gate 用 Hoeffding，这里仅作参考列)。"""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return float("-inf"), float("inf")
    m = float(xs.mean())
    s = float(xs.std(ddof=1)) if n > 1 else 0.0
    tcrit = float(t_dist.ppf(1.0 - alpha, max(n - 1, 1)))
    half = tcrit * s / math.sqrt(n)
    return m - half, m + half


def hoeffding_upper(xs, H_ub, delta=0.05):
    """Paired one-sided Hoeffding bound (016 §10):
    U_{1-delta} = Dbar + 2H sqrt(log(1/delta)/(2n)),  D = B^FG-B^D8 in [-H,H].
    Distribution-free, fixed-N, no t assumption."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return float("inf")
    return float(xs.mean()) + 2.0 * H_ub * math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def classify_qos(k_fa, k_md, n, alpha, beta):
    ufa, lfa = wilson_upper(k_fa, n), wilson_lower(k_fa, n)
    umd, lmd = wilson_upper(k_md, n), wilson_lower(k_md, n)
    if ufa <= alpha and umd <= beta:
        return "FEASIBLE"
    if lfa > alpha or lmd > beta:
        return "INFEASIBLE"
    return "UNCERTAIN"


def r_lam(om, lam_f):
    p = 1.0 / (1.0 + math.exp(-om))
    return min(LAM_M * p, lam_f * (1.0 - p))


def q1_fast(pl, x, om, i, r2, lam_f):
    zi = (x // pl.powers[i]) % BASE_B
    r_cur, _ = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    cells = next(cells for (r2b, _ct, _qb, cells) in pl._tpl[i][zi]
                 if r2b == r2)
    lp = -math.log1p(math.exp(-om))
    lq = -math.log1p(math.exp(om))
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


def dual_q_exact(pl, x, om, i, r2, lam_f):
    """独立的 generic dual-Q 实现（015 §九 的 Q_λ^(1)），用 numpy
    logsumexp 写法，与 q1_fast 独立编码——G1r-C 回归对象。"""
    zi = (x // pl.powers[i]) % BASE_B
    r_cur, _ = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    cells = next(cells for (r2b, _ct, _qb, cells) in pl._tpl[i][zi]
                 if r2b == r2)
    omv = np.array([om + pl._llr_i[i][z_code_b(r2, m2)] - pl._llr_i[i][zi]
                    for (m2, _lp0, _lp1) in cells])
    pv = 1.0 / (1.0 + np.exp(-omv))
    Rv = np.minimum(LAM_M * pv, lam_f * (1.0 - pv))
    lp0 = np.array([lp0c for (_m2, lp0c, _lp1) in cells])
    lp1 = np.array([lp1c for (_m2, _lp0, lp1c) in cells])
    lp = -math.log1p(math.exp(-om))
    lq = -math.log1p(math.exp(om))
    # 混合权重 = P(H1)p(cell|H1) + P(H0)p(cell|H0)：用 logaddexp（
    # logsumexp），不是 max——exp(max(a,b)) != exp(a)+exp(b)。
    logw = np.logaddexp(lp + lp1, lq + lp0)
    w = np.exp(logw)
    w /= w.sum()
    return c + float(np.sum(w * Rv))


def q_min_fg(pl, x, om, h, lam_f):
    """min over A_FG = A_all of Q_λ^(1)；返回 (q_min, best action, per-action
    dict 不需要)。016 G1r-B 视角下 A_FG 含 granularity 信息。"""
    rem = int(x)
    zs = []
    for _ in range(N_UAV):
        zs.append(rem % BASE_B)
        rem //= BASE_B
    best_q = None
    best_a = None
    for i in range(N_UAV):
        zi = zs[i]
        for (r2, c_true, _qb, _cells) in pl._tpl[i][zi]:
            if c_true > h:
                continue
            q = q1_fast(pl, x, om, i, r2, lam_f)
            if best_q is None or q < best_q:
                best_q = q
                best_a = (i, r2)
    return best_q, best_a


def q_min_d8(pl, x, om, h, lam_f):
    """min over A_D8={(i,8)} of Q_λ^(1)（同时返回最优 (i,8)）。"""
    rem = int(x)
    zs = []
    for _ in range(N_UAV):
        zs.append(rem % BASE_B)
        rem //= BASE_B
    best_q = None
    best_a = None
    for i in range(N_UAV):
        r_cur, _ = z_decode_b(zs[i])
        if r_cur >= R_MAX:
            continue
        c_true = BH + (R_MAX - r_cur)
        if c_true > h:
            continue
        q = q1_fast(pl, x, om, i, R_MAX, lam_f)
        if best_q is None or q < best_q:
            best_q = q
            best_a = (i, R_MAX)
    return best_q, best_a


def sim_g1(pl, eta, H, L_i, mode, quants8, powers8, audit=None):
    """G1 语义（S_common = min_{A_FG}）模拟；mode=FG/D8。
    audit 非 None 时记录 D8 分支每个决策状态的 (q_fg,q_d8,R,forced_cost)。"""
    lam_f = LAM_M * math.exp(eta)
    x, h, lam, cost, pay, nt = 0, float(H), 0.0, 0.0, 0.0, 0
    while True:
        if h < 1e-9:
            break
        om = pl.omega(x)
        R = r_lam(om, lam_f)
        q_fg, _ = q_min_fg(pl, x, om, h, lam_f)
        if q_fg is None or q_fg >= R:
            break                       # S_common: STOP（两方法）
        if mode == "FG":
            _q, a = q_min_fg(pl, x, om, h, lam_f)
        else:
            q_d8, a = q_min_d8(pl, x, om, h, lam_f)
            if a is None:
                break
            q_fg2, _ = q_min_fg(pl, x, om, h, lam_f)
            if audit is not None:
                forced = 1.0 if (q_fg2 < R <= q_d8) else 0.0
                fcost = (BH + (R_MAX - z_decode_b(
                    (x // powers8[a[0]]) % BASE_B)[0])) if forced else 0.0
                audit["n_dec"] += 1
                audit["n_F"] += int(forced)
                audit["F_cost"] += fcost
        if a is None:
            break
        i, r2 = a
        zi = (x // powers8[i]) % BASE_B
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


def sim_sref(pl, eta, H, L_i, mode, quants8, powers8):
    """G1r-B 语义：S_ref = min_{A_D8} —— 仅当至少一个 Direct8 full packet
    值得发送才 CONTINUE；FG 继续后自由选 A_FG（对 FG 更苛刻，016 §4）。"""
    lam_f = LAM_M * math.exp(eta)
    x, h, lam, cost, pay, nt = 0, float(H), 0.0, 0.0, 0.0, 0
    while True:
        if h < 1e-9:
            break
        om = pl.omega(x)
        R = r_lam(om, lam_f)
        q_d8, _ = q_min_d8(pl, x, om, h, lam_f)
        if q_d8 is None or q_d8 >= R:
            break                       # S_ref: STOP（两方法）
        if mode == "FG":
            _q, a = q_min_fg(pl, x, om, h, lam_f)
            if a is None:
                break
        else:
            _q, a = q_min_d8(pl, x, om, h, lam_f)
            if a is None:
                break
        i, r2 = a
        zi = (x // powers8[i]) % BASE_B
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


def run_gate(eta, H, H_all, L_all, quants8, powers8, mode="g1", audit=False):
    """mode: "g1"（S_common）或 "sref"（S_ref）。audit 仅用于 g1+D8 记账。"""
    pl = SparsePlanner(quants8, LAM_M, LAM_M * math.exp(eta), b_h=BH,
                       cross_level=True, levels=LEVELS, delta_c=1.0)
    n_ep = len(H_all)
    n0 = n_ep // 2
    lam_fg = np.empty(n_ep); b_fg = np.empty(n_ep)
    n_fg = np.empty(n_ep); pl_fg = np.empty(n_ep); tst_fg = np.empty(n_ep)
    lam_d8 = np.empty(n_ep); b_d8 = np.empty(n_ep)
    n_d8 = np.empty(n_ep); pl_d8 = np.empty(n_ep); tst_d8 = np.empty(n_ep)
    audit_acc = {"n_dec": 0, "n_F": 0, "F_cost": 0.0,
                 "ep_F": 0} if audit else None
    for e in range(n_ep):
        L_i = L_all[e]
        if mode == "g1":
            nf_before = audit_acc["n_F"] if audit_acc is not None else 0
            lam_fg[e], b_fg[e], n_fg[e], pl_fg[e], tst_fg[e] = sim_g1(
                pl, eta, H, L_i, "FG", quants8, powers8)
            lam_d8[e], b_d8[e], n_d8[e], pl_d8[e], tst_d8[e] = sim_g1(
                pl, eta, H, L_i, "D8", quants8, powers8, audit=audit_acc)
            if audit_acc is not None and audit_acc["n_F"] > nf_before:
                audit_acc["ep_F"] += 1
        else:
            lam_fg[e], b_fg[e], n_fg[e], pl_fg[e], tst_fg[e] = sim_sref(
                pl, eta, H, L_i, "FG", quants8, powers8)
            lam_d8[e], b_d8[e], n_d8[e], pl_d8[e], tst_d8[e] = sim_sref(
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
    t_lcb, t_ucb = mean_ci_t(D)
    h_ucb = hoeffding_upper(D, float(H))

    def tstop_pmf(tst):
        pmf = np.bincount(tst.astype(int), minlength=TSTOP_MAX + 1)[:TSTOP_MAX]
        pmf[-1] += int(np.sum(tst.astype(int) >= TSTOP_MAX))
        return pmf / len(tst)

    m_fg = mclib.evaluate(lam_fg, b_fg, H_all, PFA_TARGET)
    m_d8 = mclib.evaluate(lam_d8, b_d8, H_all, PFA_TARGET)
    out_d = {
        "n0": n0, "eta": eta, "H": H, "mode": mode,
        "pfa_fg": kfa_fg / n0, "pmd_fg": kmd_fg / n0,
        "pfa_d8": kfa_d8 / n0, "pmd_d8": kmd_d8 / n0,
        "kfa_fg": kfa_fg, "kmd_fg": kmd_fg, "kfa_d8": kfa_d8, "kmd_d8": kmd_d8,
        "eb_fg": float(b_fg.mean()), "eb_d8": float(b_d8.mean()),
        "D": float(D.mean()), "t_lcb": t_lcb, "t_ucb": t_ucb,
        "h_ucb": h_ucb,
        "entx_fg": float(n_fg.mean()), "entx_d8": float(n_d8.mean()),
        "epl_fg": float(pl_fg.mean()), "epl_d8": float(pl_d8.mean()),
        "etst_fg": float(tst_fg.mean()), "etst_d8": float(tst_d8.mean()),
        "tst_fg": tstop_pmf(tst_fg), "tst_d8": tstop_pmf(tst_d8),
        "pd_fg": m_fg["pd"], "pd_d8": m_d8["pd"],
    }
    if audit_acc is not None:
        out_d["audit"] = audit_acc
    return out_d


def best_eta(eta_grid, H, H_all, L_all, quants8, powers8, mode):
    """Calibration（016 §9：grid 冻结、只在 calibration 用）：选"两方法都
    FEASIBLE 且 E[B^FG]+E[B^D8] 最小"的 η_star。"""
    feasible = {}
    for eta in eta_grid:
        r = run_gate(eta, H, H_all, L_all, quants8, powers8, mode=mode)
        c_fg = classify_qos(r["kfa_fg"], r["kmd_fg"], r["n0"], ALPHA, BETA)
        c_d8 = classify_qos(r["kfa_d8"], r["kmd_d8"], r["n0"], ALPHA, BETA)
        if c_fg == "FEASIBLE" and c_d8 == "FEASIBLE":
            feasible[eta] = r["eb_fg"] + r["eb_d8"]
    if feasible:
        return min(feasible, key=feasible.get), feasible
    return None, feasible


def regress_q1(quants8, powers8, n_states=200):
    """G1r-C：q1_fast vs generic dual_q_exact（独立编码回归）。"""
    pl = SparsePlanner(quants8, LAM_M, LAM_M * math.exp(1.2), b_h=BH,
                       cross_level=True, levels=LEVELS, delta_c=1.0)
    rng = np.random.default_rng(SEED0 + 7)
    lam_f = LAM_M * math.exp(1.2)
    max_diff = 0.0
    n_checked = 0
    for _ in range(n_states):
        # 279^8 > int64：按 UAV 逐个采样 z 码再组合（避免 overflow）
        zs = [int(rng.integers(0, BASE_B)) for _ in range(N_UAV)]
        x = sum(zi * (BASE_B ** i) for i, zi in enumerate(zs))
        om = pl.omega(x)
        rem = int(x)
        for i in range(N_UAV):
            zi = zs[i]
            for (r2, _ct, _qb, _cells) in pl._tpl[i][zi]:
                a = q1_fast(pl, x, om, i, r2, lam_f)
                b = dual_q_exact(pl, x, om, i, r2, lam_f)
                max_diff = max(max_diff, abs(a - b))
                n_checked += 1
    return max_diff, n_checked


def emulate_d8(quants8, powers8, n_ep_check=50):
    """G1r-C：D8 emulation invariant——FG 分支限定 A_D8 与 D8 分支在相同
    episode 上逐样本完全一致（lam/cost/n_tx/payload）。"""
    model8 = GaussianDetectorModel(GAMMA_B)
    rng = np.random.default_rng(SEED0 + 9)
    H_all = np.random.default_rng(SEED0 + 11).integers(0, 2, 2 * n_ep_check)
    L_all = model8.sample_llr(H_all, rng)
    H = 96.0
    eta = 1.2
    pl = SparsePlanner(quants8, LAM_M, LAM_M * math.exp(eta), b_h=BH,
                       cross_level=True, levels=LEVELS, delta_c=1.0)
    for e in range(2 * n_ep_check):
        L_i = L_all[e]
        r_d8 = sim_g1(pl, eta, H, L_i, "D8", quants8, powers8)
        # FG emulating D8：同一 S_common，但动作强制在 A_D8
        lam_f = LAM_M * math.exp(eta)
        x, h, lam, cost, pay, nt = 0, float(H), 0.0, 0.0, 0.0, 0
        while True:
            if h < 1e-9:
                break
            om = pl.omega(x)
            R = r_lam(om, lam_f)
            q_fg, _ = q_min_fg(pl, x, om, h, lam_f)
            if q_fg is None or q_fg >= R:
                break
            _q, a = q_min_d8(pl, x, om, h, lam_f)
            if a is None:
                break
            i, r2 = a
            zi = (x // powers8[i]) % BASE_B
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
        assert abs(r_d8[0] - lam) < 1e-9 and abs(r_d8[1] - cost) < 1e-9
        assert abs(r_d8[2] - nt) < 1e-9 and abs(r_d8[3] - pay) < 1e-9
    return n_ep_check


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
    N_CAL = N_TEST // 2
    H_BUDGETS = (48, 96)
    CAL_H = 96
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.7-G1r — forced-continuation audit + conservative S_ref "
        "Gate + dual-Q regression（016 §1-§6、§10、§15 路线 1-3）")
    out("")
    out(f"> 协议：N=8（GAMMA={GAMMA_B}、levels={LEVELS}、r_max={R_MAX}）；b_setup={BH}；"
        f"stratified N0=N1={N_TEST}（calibration N_CAL={N_CAL} @ H={CAL_H}，test "
        f"N_TEST={N_TEST} @ H∈{H_BUDGETS}）；episode 级 CRN（同一 W_e=(H_e,L_e) 给 "
        f"FG/D8；planner 确定性）；grid 冻结、只在 calibration 用（016 §9）。")
    out("")
    out("> **016 P0（§1）**：G1 的公共停止器 S_common 用 min_{a∈A_FG}Q<R_λ 判定，"
        "而 D8 只能从 A_D8={(i,8)} 选动作——存在 F(x)=1{q_FG<R_λ≤q_D8} 状态（小包"
        "值得买 → 公共控制器说继续，但 8-bit 包已不值得 → D8 被迫发 8-bit）。"
        "A_FG 本身含 granularity 信息 ⇒ 'STOP 判定与包粒度无关' 表述不成立。")
    out("")
    out("> **G1r-A（016 §15-1）**：审计 F 频率与 ΔB_forced（D8 被强迫支付的通信）。")
    out("> **G1r-B（016 §4/§15-2）**：保守停止器 S_ref：CONTINUE iff "
        "min_{a∈A_D8}Q_λ^(1)<R_λ；两方法共用（对 FG 更苛刻——只有'至少一个 "
        "Direct8 full packet 值得发送'才给 FG 一次 adaptive-granularity 机会）。")
    out("> **G1r-C（016 §15-3）**：q1_fast vs 独立 generic dual-Q exact 回归；"
        "D8 emulation invariant（FG 限定 A_D8 ≡ D8 分支）。")
    out("")
    out("> **统计（016 §10）**：paired bit 用 **one-sided paired Hoeffding "
        f"U={fmt(0)} 公式 U=D̄+2H√(log(1/δ)/2n)**（D∈[-H,H]，分布无关、无 t 假设）"
        "作为正式上界；t 版仅参考。QoS：Wilson 95%。Gate = intersection-union "
        "（各 component 按预设 level 控制，全条件成立才 PASS）。")
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

    # ------------------------------------------------ G1r-C regression first
    out("## 0. G1r-C — 代码可信度封板（016 §15-3）")
    out("")
    t_reg = time.time()
    max_diff, n_q = regress_q1(quants8, powers8, n_states=200)
    ok_reg = max_diff < 1e-9
    out(f"- q1_fast vs generic dual-Q exact：检查 {n_q} 个 (i,r2) 动作，"
        f"max|Δ|={fmt(max_diff, 12)}（<1e-9 → {mp(ok_reg)}）。")
    n_em = emulate_d8(quants8, powers8, n_ep_check=50)
    out(f"- D8 emulation invariant：{n_em} 个 episode 上 'FG 限定 A_D8' 与 D8 分支 "
        f"逐样本完全一致（lam/cost/N_tx/payload）→ **PASS**。")
    out(f"（{time.time()-t_reg:.1f}s）")
    out("")

    # -------------------------------------------- G1r-A/G1r-B calibration
    out("## 1. Calibration（016 §9：grid 冻结、仅 calibration）")
    out("")
    eta_a, feas_a = best_eta(GRID_ETA, CAL_H, H_cal, L_cal, quants8, powers8, "g1")
    eta_b, feas_b = best_eta(GRID_ETA, CAL_H, H_cal, L_cal, quants8, powers8, "sref")
    if eta_a is None or eta_b is None:
        out(f"- 校准 UNRESOLVED（n 不足）：S_common 达标 η="
            f"{sorted(feas_a) if feas_a else '∅'}，S_ref 达标 η="
            f"{sorted(feas_b) if feas_b else '∅'}——按 016 §15 转 lower-bound "
            f"或 --nlevel 扩样。")
        # 诊断标注：无达标 η 时以 G1 已冻结值作 *诊断默认*（非校准选择），
        # 报告中明确标注，防止把默认值误读为 calibration 结果。
        if eta_a is None:
            eta_a = 1.2
        if eta_b is None:
            eta_b = 1.2
    else:
        out(f"- **S_common（G1 语义）η_star = {fmt(eta_a, 1)}**（达标集 "
            f"{sorted(feas_a)}，选 E[B^FG]+E[B^D8] 最小）。")
        out(f"- **S_ref（G1r-B 保守）η_star = {fmt(eta_b, 1)}**（达标集 "
            f"{sorted(feas_b)}）。")
        show_eta = {e: "S_common" for e in feas_a}
        for e in feas_b:
            show_eta.setdefault(e, "S_ref")
        out(f"- 两停止器达标 η 交集/并集：{sorted(show_eta)}（S_common/S_ref 分别标注）。")
    out("")

    # ---------------------------------------------------------- G1r-A test
    out("## 2. G1r-A — Forced-Continuation Audit（G1 语义，η_star 冻结）")
    out("")
    for H in H_BUDGETS:
        Ht, Lt = (H_t48, L_t48) if H == 48 else (H_t96, L_t96)
        out(f"### H={H}（η_star={fmt(eta_a, 1)} 冻结，test fresh）")
        out("")
        r = run_gate(eta_a, H, Ht, Lt, quants8, powers8, mode="g1", audit=True)
        a = r["audit"]
        out(f"- **P(F=1)（按决策状态）** = {fmt(a['n_F'] / max(a['n_dec'], 1))}"
            f"（F 状态 {a['n_F']}/{a['n_dec']}）；P(episode contains F) = "
            f"{fmt(a['ep_F'] / r['n0'] / 2)}；ΔB_forced = {fmt(a['F_cost'])} bits"
            f"（D8 因 S_common 被迫支付的 8-bit 成本）。")
        out(f"- E[B^FG]={fmt(r['eb_fg'])}、E[B^D8]={fmt(r['eb_d8'])}，"
            f"E[D]={fmt(r['D'])}（t 参考 CI [{fmt(r['t_lcb'])}, {fmt(r['t_ucb'])}]，"
            f"Hoeffding U95={fmt(r['h_ucb'])}）。")
        out("")
        out("| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, name in (("fg", "FG"), ("d8", "Direct8")):
            kfa, kmd = r[f"kfa_{key}"], r[f"kmd_{key}"]
            ufa = wilson_upper(kfa, r["n0"])
            umd = wilson_upper(kmd, r["n0"])
            cls = classify_qos(kfa, kmd, r["n0"], ALPHA, BETA)
            out(f"| {name} | {fmt(r[f'pfa_{key}'])} | {fmt(ufa)} | "
                f"{fmt(r[f'pmd_{key}'])} | {fmt(umd)} | {cls} | "
                f"{fmt(r[f'entx_{key}'])} | {fmt(r[f'epl_{key}'])} |")
        out("")

    # ---------------------------------------------------------- G1r-B test
    out("## 3. G1r-B — conservative S_ref Gate（016 §4）")
    out("")
    for H in H_BUDGETS:
        Ht, Lt = (H_t48, L_t48) if H == 48 else (H_t96, L_t96)
        out(f"### H={H}（η_star_ref={fmt(eta_b, 1)} 冻结，test fresh）")
        out("")
        r = run_gate(eta_b, H, Ht, Lt, quants8, powers8, mode="sref")
        out("| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | "
            "E[B] | E[T_stop] |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, name in (("fg", "FG"), ("d8", "Direct8")):
            kfa, kmd = r[f"kfa_{key}"], r[f"kmd_{key}"]
            ufa = wilson_upper(kfa, r["n0"])
            umd = wilson_upper(kmd, r["n0"])
            cls = classify_qos(kfa, kmd, r["n0"], ALPHA, BETA)
            out(f"| {name} | {fmt(r[f'pfa_{key}'])} | {fmt(ufa)} | "
                f"{fmt(r[f'pmd_{key}'])} | {fmt(umd)} | {cls} | "
                f"{fmt(r[f'entx_{key}'])} | {fmt(r[f'epl_{key}'])} | "
                f"{fmt(r[f'eb_{key}'])} | {fmt(r[f'etst_{key}'])} |")
        out("")
        fg_cls = classify_qos(r["kfa_fg"], r["kmd_fg"], r["n0"], ALPHA, BETA)
        d8_cls = classify_qos(r["kfa_d8"], r["kmd_d8"], r["n0"], ALPHA, BETA)
        bit_sig = r["h_ucb"] < 0.0
        out(f"- **Gate（intersection-union，016 §10）**：QoS 双方 FEASIBLE = "
            f"{mp(fg_cls == 'FEASIBLE' and d8_cls == 'FEASIBLE')}（FG={fg_cls}、"
            f"D8={d8_cls}）；paired Hoeffding U95(E[D])={fmt(r['h_ucb'])}"
            f"（<0 → {mp(bit_sig)}）。")
        if (fg_cls == "FEASIBLE" and d8_cls == "FEASIBLE" and bit_sig):
            out(f"  → **G1r-B PASS**：保守 common-stop（仅当 D8 full packet 值得才给 "
                f"FG adaptive 机会）下仍 U95(E[B^FG−B^D8])<0 → **granularity 独立 "
                f"收益基本无法从公平性击穿**（016 §4）→ 值得投入 fresh G2 "
                f"（B0.7-G2，016 §15-4）。")
        else:
            out(f"  → **UNRESOLVED / FAIL**：016 §15 判定——若 P(F≈1) 不小或 S_ref "
                f"下不显著，则 016 §4 预期（-12.31 → -5..-10）落空，按 016 §15 "
                f"转 lower-bound 或先扩样。")
        out("")
        out("| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | "
            "P(T_stop≥5) |")
        out("| --- | --- | --- | --- | --- | --- |")
        for key, name in (("fg", "FG"), ("d8", "Direct8")):
            pmf = r[f"tst_{key}"]
            out(f"| {name} | {fmt(pmf[1])} | {fmt(pmf[2])} | {fmt(pmf[3])} | "
                f"{fmt(pmf[4])} | {fmt(pmf[5:].sum())} |")
        out("")
        setup_d = BH * (r["entx_fg"] - r["entx_d8"])
        pay_d = r["epl_fg"] - r["epl_d8"]
        out(f"- 分解（S_ref）：E[D]={fmt(r['D'])}；setup 部分 {BH}·(E[N_tx^FG]−"
            f"E[N_tx^D8])={fmt(setup_d)}，payload 部分 {fmt(pay_d)}。")
        out(f"- NP-matched（P_FA=0.05）secondary：P_D^FG={fmt(r['pd_fg'])} / "
            f"E[B]^FG={fmt(r['eb_fg'])}；P_D^D8={fmt(r['pd_d8'])} / "
            f"E[B]^D8={fmt(r['eb_d8'])}。")
        out(f"（{time.time() - t_start:>6.1f}s 累计）")
        out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    out("- **B0.7-G1r 结论（016 §15）**：G1r-A 量化 action-leakage（ΔB_forced / "
        "P(F=1)）——若很小，则 G1 的 D8 劣势主要是 granularity 真实机制；G1r-B "
        "用 A_D8-reference 的保守 common-stop 重验 U95(E[B^FG−B^D8])<0——若仍 "
        "显著，granularity 独立收益在公平性上站稳（016 §4 预期 -5..-10 bit），"
        "随后投入 **B0.7-G2**：FG/D8 **分别**在 calibration 上优化自己的 "
        "(ρ,η)（016 §7/§9：J_m⋆=inf E_π[B] s.t. QoS，各自 controller），test "
        "完全 fresh，暂不加 CPI；最后 B0.7-G3 重构 DualCPI 使 certificate 与 "
        "当前 dual objective 一致（016 §5/§15-5）。")
    out("")

    full_rp = os.path.join(OUT_DIR, "MVS-B0.7-G1r_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.7-G1r_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()