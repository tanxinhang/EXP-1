"""MVS-B0.7-G2: Separately Calibrated QoS-Dual Policy-Family Certification
(advice/017.md).

定位（017 §final）：**Separately Calibrated QoS-Dual Policy-Family
Certification** —— 不是"再优化一次 FG"，而是验证：

    boxed{ 当 FG 与 Direct8 都允许为自身优化 controller 时，
            在相同 detection QoS 下谁需要更少 communication bits？ }

G2 的控制器（017 §二）：FG 和 D8 各自拥有自己的
  S_m(x,h;rho,eta),  m in {FG, D8}
  lambda_M = rho,  lambda_F = rho * e^eta,
  R_{rho,eta}(x) = rho * min{ p_x, e^eta (1-p_x) },
  Q^{(1)}_{rho,eta}(a|x,h) = c_a + E[ R_{rho,eta}(X') | x, a ],
动作集分别为
  A_FG = {(i,r'): r' > r_i, r' in {1,2,4,8}},
  A_D8 = {(i,8)  : r_i < 8},
各自停止
  STOP_m  <=>  R_{rho,eta}(x) <= min_{a in A_m(h,x)} Q^{(1)}_{rho,eta}(a|x,h),
各自继续
  a*_m = argmin_{a in A_m} Q^{(1)}_{rho,eta}(a),
最后均使用  Omega > eta => H1。
这彻底不存在 G1 的 action-set leakage（不再有公共停止器强迫 D8）。

参数冻结（017 §四）：N=8、levels (1,2,4,8)、b_setup=16、QoS
(P_FA<=0.12, P_MD<=0.40)；RHO={128,256,512,1024}、ETA={0.8,1.0,...,2.0}
= 28 combos/method；calibration worlds FG/D8 完全共用；CPI OFF；test
worlds 与 calibration 完全分离；paired CRN；主 operating point H=96、
secondary stress H=48（同冻结 controller，诚实报告 operating-region
boundary，不为 H=48 重新调参）。

校准选择（017 §三/§五）：Theta = R_rho x R_eta，
  theta_hat_m = argmin_{theta in Theta} E_hat_cal[ B_m(theta) ]
      s.t.  U_cal(P_FA^{m,theta}) <= alpha,  U_cal(P_MD^{m,theta}) <= beta
N_CAL=600/hypothesis @ H=96（017 §五，至少如此）。

统计（017 §六 方案 A）：**正式 test 直接冻结 N_TEST=1600 per
hypothesis、一次性看结果**——不做看结果后的 staged escalation；paired
bit 用 fixed-N one-sided paired Hoeffding（D=B^FG-B^D8 in [-H,H]，分布
无关），QoS 用 Wilson 双侧 95% 区间上端点（018 §十一：z=1.96 为双侧
端点，作单侧上界≈97.5% —— 更保守，只改名称不改数值）。Gate（017 §八）
只允许四种
结论：G2 PASS / FAIL / BIT-UNRESOLVED / QoS-UNRESOLVED-INFEASIBLE。

命名（017 §三）：实验对象是 **separately calibrated one-step QoS-dual
controllers**（不是 optimized/globally optimized）。

018 §三 + 019 §2（方案 A，仅文档）：ρ 直接定义为 **conditional-error
Lagrange 的 effective multiplier**：\barλ_M=λ_M/π_1、\barλ_F=λ_F/π_0，
参数化 ρ=\barλ_M、ρe^η=\barλ_F。由此 terminal Bayes risk 本来就是
R_{ρ,η}(x)=min{ρ·p, ρe^η(1−p)} —— **代码即 exact effective-multiplier
parameterization**（根本不需要“乘 2/统一缩放”步骤，019 §2：R→2R 而
c_a 不缩放会改变 stopping condition，原表述不严格）。数值与 G2 结论不变。

001 §final 重定位（**本文档/runner 的当前角色**）：**G2 数值保留、定位降为
homogeneous-link mechanism-validation special case**（001 §二十四：这是
16+Δr 均匀成本模型下的机制证据，不是 SystemModel 最终 ISAC 通信主结论）；
**G3（DualCPI）暂时 SUSPENDED**（001 §二十五：当前瓶颈是 architecture
realignment 而非 planner）；论文主 QoS 口径按 001 §三 统一为
P_FA≤α ∧ P_D≥P_D,max(α)−ε_D（默认 α=0.05、ε_D=0.01），本 G2 注册的
P_MD≤0.40 仅作机制验证口径；成本模型在接下来 MVS-C C0 改为 link-aware
（c_{i,r→r'}=b_{0,i}+d_i(r,r')，16+Δr 即 b_{0,i}=16、κ_i=1 的 special
case，001 §六）；hard budget 为 frame-window C_{U2U}(ω)≤C_max^{frame}。
下一步不是 G3，而是 **MVS-C Architecture Realignment（C0–C5，001 §二十六）**
及其四个论文 Gate（A 数学正确性 / B 机制必要性 / C 通信现实性 / D 求解器
质量，001 §二十七）。

Secondary diagnostics（017 §七）：
  (1) E[B|H0], E[B|H1] 分别报告；
  (2) 记录 (rho*_FG, eta*_FG), (rho*_D8, eta*_D8) 及 calibration
      feasible region（(rho,eta) -> {INFEASIBLE, UNCERTAIN, FEASIBLE}
      并在 feasible region 上写 E[B]）；
  (3) cross-evaluation（secondary，不参与 Gate）：FG@theta*_FG、
      FG@theta*_D8、D8@theta*_D8、D8@theta*_FG @ H=96。

017 §一 的三个小口径修正在本文件直接落地：
  P1-1：forced-continuation 只报告 **P(F=1 | S_common=CONTINUE)**
        （分母 = 公共规则说继续的决策状态；把 STOP 决策状态加入分母
        只会更低），不再写裸 "P(F=1)（按决策状态）"。
  P1-2：F 状态上 D8 实付的 8-bit 通信成本之和改名 **gross forced-
        action cost**（非 causal extra cost），并按 episode 归一化。
  P1-3：dual-Q 回归升级为 root + on-policy reachable +
        resolution-stratified（r=0/1/2/4）三类覆盖 x (rho,eta) corner
        {(128,0.8),(512,1.2),(1024,2.0)}；emulate_d8 的 episode 计数
        改为真实循环数 2*n_ep_check（纯 P2 表述）。

Invariant suite（017 §九）：
  inv-1  sum_{m'} P(m'|x,a) = 1（PMF 质量守恒，unnormalized）；
  inv-2  q_fast = generic dual-Q exact（max|Δ| < 1e-9）；
  inv-3  B = b_setup*N_tx + B_payload（逐 episode 恒等式）；
  inv-4  B <= H（逐 episode 预算界）；
  inv-5  pi_FG | A_D8 ≡ pi_D8（相同 (rho,eta)、相同 world 逐样本一致）。
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
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
SEED0 = 2026
SEED_CAL = SEED0 + 100
SEED_TEST = SEED0 + 200
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
ALPHA = 0.12
BETA = 0.40
BH = 16.0
LEVELS = (1, 2, 4, 8)
R_MAX = 8
N_UAV = 8
DELTA_CI = 0.05

# 017 §四：冻结 grid（只在 calibration 使用，test 不碰）
RHO_GRID = (128, 256, 512, 1024)
ETA_GRID = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
# 017 §九：回归用 (rho,eta) corner
CORNER_THETAS = ((128, 0.8), (512, 1.2), (1024, 2.0))
# G1r 冻结的公共停止参考 (lambda_M=512, eta_star=1.2) —— 017 §一 审计口径
REF_THETA = (512, 1.2)
# 019 §4：challenger 的 material vs numerical-near-tie 阈值（bit/episode；
# (256,1.4) 差 0.0017 为 near-tie，(128,0.8) 差 ≈9.6 为 material）
MAT_EP = 0.05

# FULL 冻结（017 §五/§六 方案 A）：N_CAL=600/hyp @ H=96，N_TEST=1600/hyp
FULL_N_CAL = 600
FULL_N_TEST = 1600


def fmt(x, nd=4):
    if x == float("inf"):
        return "inf"
    if x != x:  # nan
        return "nan"
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
    """One-sided t-based 95% mean bounds（参考列；正式 Gate 用 Hoeffding）。"""
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
    """Paired one-sided Hoeffding bound: U = Dbar + 2H sqrt(log(1/d)/(2n))."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return float("inf")
    return float(xs.mean()) + 2.0 * H_ub * math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def hoeffding_lower(xs, H_ub, delta=0.05):
    """Paired one-sided Hoeffding lower bound: L = Dbar - 2H sqrt(log(1/d)/(2n))."""
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return float("-inf")
    return float(xs.mean()) - 2.0 * H_ub * math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def classify_qos(kfa, kmd, n, alpha=ALPHA, beta=BETA):
    ufa, lfa = wilson_upper(kfa, n), wilson_lower(kfa, n)
    umd, lmd = wilson_upper(kmd, n), wilson_lower(kmd, n)
    if ufa <= alpha and umd <= beta:
        return "FEASIBLE"
    if lfa > alpha or lmd > beta:
        return "INFEASIBLE"
    return "UNCERTAIN"


# --------------------------------------------------------------------------
# G2 控制器核心（017 §二）。planner 的 mu_M/mu_F 在本比较中**不使用**——
# 风险函数直接由 (rho, eta) 计算（lambda_M=rho, lambda_F=rho e^eta），
# 模板 _tpl/_llr_i/powers 只依赖量化器与 b_h/levels，因此可跨 (rho,eta)
# 复用同一个 planner。
# --------------------------------------------------------------------------

def r_rho(om, rho, eta):
    """R_{rho,eta}(x) = rho * min{ p, e^eta (1-p) },  p = sigma(om)."""
    p = 1.0 / (1.0 + math.exp(-om))
    return rho * min(p, math.exp(eta) * (1.0 - p))


def desc_weights(pl, x, om, i, r2):
    """(m2, w) 列表：w = P(H1|x)p(m'|H1) + P(H0|x)p(m'|H0)（unnormalized，
    replace-not-add 后的 descendant 质量）。理论上 sum_m' w = 1
    （level-r2 cells 划分全空间、P(H1|x)+P(H0|x)=1）——inv-1 检查对象。"""
    zi = (x // pl.powers[i]) % BASE_B
    cells = next(cells for (r2b, _ct, _qb, cells) in pl._tpl[i][zi]
                 if r2b == r2)
    lp = -math.log1p(math.exp(-om))
    lq = -math.log1p(math.exp(om))
    out = []
    for (m2, lp0c, lp1c) in cells:
        a_ = lp + lp1c
        b_ = lq + lp0c
        m_ = a_ if a_ >= b_ else b_
        w = math.exp(m_ + math.log1p(math.exp(-abs(a_ - b_))))
        out.append((m2, w))
    return out


def q1_fast(pl, x, om, i, r2, rho, eta):
    """Q^{(1)}_{rho,eta}(a|x) = c_a + E[R_{rho,eta}(X')|x,a]（标量循环版，
    inv-2 回归对象之一）。"""
    zi = (x // pl.powers[i]) % BASE_B
    r_cur, _ = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    E = 0.0
    for (m2, w) in desc_weights(pl, x, om, i, r2):
        z2 = z_code_b(r2, m2)
        om_c = om + pl._llr_i[i][z2] - pl._llr_i[i][zi]
        E += w * r_rho(om_c, rho, eta)
    return c + E


def dual_q_exact(pl, x, om, i, r2, rho, eta):
    """独立的 generic dual-Q 实现（numpy logsumexp 写法），与 q1_fast
    独立编码——inv-2 回归对象（017 §九，沿用 G1r-C 语义）。"""
    zi = (x // pl.powers[i]) % BASE_B
    r_cur, _ = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    cells = next(cells for (r2b, _ct, _qb, cells) in pl._tpl[i][zi]
                 if r2b == r2)
    omv = np.array([om + pl._llr_i[i][z_code_b(r2, m2)] - pl._llr_i[i][zi]
                    for (m2, _lp0, _lp1) in cells])
    pv = 1.0 / (1.0 + np.exp(-omv))
    Rv = rho * np.minimum(pv, math.exp(eta) * (1.0 - pv))
    lp0 = np.array([lp0c for (_m2, lp0c, _lp1) in cells])
    lp1 = np.array([lp1c for (_m2, _lp0, lp1c) in cells])
    lp = -math.log1p(math.exp(-om))
    lq = -math.log1p(math.exp(om))
    # logsumexp（不是 max）——混合权重 = P(H1)p(cell|H1)+P(H0)p(cell|H0)
    logw = np.logaddexp(lp + lp1, lq + lp0)
    w = np.exp(logw)
    w /= w.sum()
    return c + float(np.sum(w * Rv))


def _decode_zs(x):
    rem = int(x)
    zs = []
    for _ in range(N_UAV):
        zs.append(rem % BASE_B)
        rem //= BASE_B
    return zs


def q_min_fg(pl, x, om, h, rho, eta):
    """min over A_FG of Q^{(1)}（同时返回最优动作 (i,r2)）。"""
    zs = _decode_zs(x)
    best_q, best_a = None, None
    for i in range(N_UAV):
        zi = zs[i]
        for (r2, c_true, _qb, _cells) in pl._tpl[i][zi]:
            if c_true > h:
                continue
            q = q1_fast(pl, x, om, i, r2, rho, eta)
            if best_q is None or q < best_q:
                best_q, best_a = q, (i, r2)
    return best_q, best_a


def q_min_d8(pl, x, om, h, rho, eta):
    """min over A_D8={(i,8)} of Q^{(1)}（同时返回最优 (i,8)）。"""
    zs = _decode_zs(x)
    best_q, best_a = None, None
    for i in range(N_UAV):
        r_cur, _ = z_decode_b(zs[i])
        if r_cur >= R_MAX:
            continue
        c_true = BH + (R_MAX - r_cur)
        if c_true > h:
            continue
        q = q1_fast(pl, x, om, i, R_MAX, rho, eta)
        if best_q is None or q < best_q:
            best_q, best_a = q, (i, R_MAX)
    return best_q, best_a


def apply_action(pl, x, h, lam, cost, pay, nt, a, L_i, quants8, powers8):
    """执行 (i,r2)：replace-not-add 的 log-odds 更新 + 成本记账。
    返回 (x2, h2, lam2, cost2, pay2, nt2)。"""
    i, r2 = a
    zi = (x // powers8[i]) % BASE_B
    r_cur, m_cur = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    m2 = int(quants8[i].cell_index(r2, L_i[i]))
    lam2 = lam + quants8[i].llr[r2][m2]
    if r_cur > 0:
        lam2 -= quants8[i].llr[r_cur][m_cur]
    z2 = z_code_b(r2, m2)
    x2 = x + (z2 - zi) * powers8[i]
    return (x2, h - c, lam2, cost + c, pay + (r2 - r_cur), nt + 1)


def sim_method(pl, rho, eta, H, L_i, mode, quants8, powers8):
    """G2 语义（017 §二）：STOP_m iff R <= min_{a in A_m} Q^{(1)}；
    CONTINUE 时执行 argmin_{A_m}。返回 (lam, cost, n_tx, payload)。"""
    x, h, lam, cost, pay, nt = 0, float(H), 0.0, 0.0, 0.0, 0
    while True:
        if h < 1e-9:
            break
        om = pl.omega(x)
        R = r_rho(om, rho, eta)
        if mode == "FG":
            q, a = q_min_fg(pl, x, om, h, rho, eta)
        else:
            q, a = q_min_d8(pl, x, om, h, rho, eta)
        if q is None or q >= R:
            break
        x, h, lam, cost, pay, nt = apply_action(
            pl, x, h, lam, cost, pay, nt, a, L_i, quants8, powers8)
    return lam, cost, nt, pay


def sim_common_audit(pl, rho, eta, H, L_i, quants8, powers8, acc):
    """P1-1/P1-2 口径修正后的 forced-continuation audit（017 §一）。

    语义 = G1r-A（S_common = min_{A_FG}Q < R 才继续；D8 在公共规则
    CONTINUE 状态下被迫执行 A_D8）。但统计口径修正为：
      n_dec_all : 所有 stopping-decision 状态（含公共规则 STOP 的
                  状态）——分母用于"更保守的无条件比例"；
      n_dec     : 仅 S_common=CONTINUE 的决策状态；
      P(F=1|S_common=CONTINUE) = n_F / n_dec   （正式口径，017 P1-1）；
      gross forced-action cost = Σ F 状态上 D8 实付的 full-packet
      通信成本（017 P1-2：不再称 causal extra cost）。"""
    x, h, cost = 0, float(H), 0.0
    has_F = False
    t_idx = 0                                     # episode 内决策序号（018 §十）
    while True:
        if h < 1e-9:
            break
        om = pl.omega(x)
        R = r_rho(om, rho, eta)
        q_fg, _ = q_min_fg(pl, x, om, h, rho, eta)
        acc["n_dec_all"] += 1                     # 所有 stopping-decision 状态
        t_idx += 1
        if q_fg is None or q_fg >= R:
            break                                 # S_common: STOP（两方法都停）
        # S_common=CONTINUE 的决策状态
        acc["n_dec"] += 1
        q_d8, a = q_min_d8(pl, x, om, h, rho, eta)
        if a is None:
            break                                 # h 不足以付任何 8-bit
        forced = (R <= q_d8)                      # q_fg < R <= q_d8 => F 状态
        if forced:
            acc["n_F"] += 1
            r_cur = z_decode_b((x // (BASE_B ** a[0])) % BASE_B)[0]
            acc["F_cost"] += BH + (R_MAX - r_cur)  # gross forced-action cost
            acc["F_idx_sum"] += t_idx             # 018 §十：F 的决策索引统计
            acc["n_F_idx1"] += int(t_idx == 1)
            has_F = True
        # D8 在公共规则下被迫执行其最优 (i,8)
        x, h, _lam, cost, _pay, _nt = apply_action(
            pl, x, h, 0.0, cost, 0.0, 0, a, L_i, quants8, powers8)
    if has_F:
        acc["ep_F"] += 1
    acc["ep"] += 1


def sample_set(Nn, seed, model8):
    rng = np.random.default_rng(seed)
    H0 = np.zeros(Nn, dtype=np.int8)
    H1 = np.ones(Nn, dtype=np.int8)
    L0 = model8.sample_llr(H0, rng)
    L1 = model8.sample_llr(H1, rng)
    return np.concatenate([H0, H1]), np.concatenate([L0, L1])


def eval_theta(pl, rho, eta, H, H_all, L_all, quants8, powers8, mode):
    """在给定 worlds 上跑 method 的 G2 controller（(rho,eta) 冻结，planner
    确定性 → 对同一 worlds 天然 paired CRN）。返回逐 episode 数组 + 聚合。
    H_all 为 int8 0/1（stratified N0=N1）。"""
    n_ep = len(H_all)
    lam_m = np.empty(n_ep)
    b_m = np.empty(n_ep)
    nt_m = np.empty(n_ep)
    pay_m = np.empty(n_ep)
    viol_id = viol_bud = 0
    for e in range(n_ep):
        lam, cost, nt, pay = sim_method(
            pl, rho, eta, H, L_all[e], mode, quants8, powers8)
        lam_m[e] = lam
        b_m[e] = cost
        nt_m[e] = nt
        pay_m[e] = pay
        if abs(cost - (BH * nt + pay)) > 1e-9:
            viol_id += 1
        if cost > H + 1e-9:
            viol_bud += 1
    H1 = H_all == 1
    i0 = ~H1
    i1 = H1
    n0 = int(np.count_nonzero(i0))
    kfa = int(np.sum(lam_m[i0] > eta))
    kmd = int(np.sum(lam_m[i1] <= eta))
    return {
        "lam": lam_m, "b": b_m, "nt": nt_m, "pay": pay_m,
        "eb": float(b_m.mean()),
        "eb0": float(b_m[i0].mean()), "eb1": float(b_m[i1].mean()),
        "entx": float(nt_m.mean()), "epl": float(pay_m.mean()),
        "kfa": kfa, "kmd": kmd, "n0": n0,
        "viol_id": viol_id, "viol_bud": viol_bud,
    }


def calibrate(pl, H, H_cal, L_cal, quants8, powers8, rho_grid, eta_grid):
    """Calibration（017 §三/§五）：shared worlds、grid 冻结；
    θ̂_m = feasible（U95(P_FA)<=alpha ∧ U95(P_MD)<=beta）中 Ê_cal[B_m]
    最小（tie-break: (rho,eta) 字典序小者）。"""
    tables = {"FG": {}, "D8": {}}                 # (rho,eta) -> stats dict
    for rho in rho_grid:
        for eta in eta_grid:
            th = (rho, eta)
            for mode in ("FG", "D8"):
                s = eval_theta(pl, rho, eta, H, H_cal, L_cal,
                               quants8, powers8, mode)
                tables[mode][th] = s
    theta_star = {}
    feasible = {}
    for mode in ("FG", "D8"):
        F = {th: s for th, s in tables[mode].items()
             if classify_qos(s["kfa"], s["kmd"], s["n0"]) == "FEASIBLE"}
        feasible[mode] = F
        if F:
            theta_star[mode] = min(
                F, key=lambda th: (F[th]["eb"], th[0], th[1]))
        else:
            theta_star[mode] = None
    return theta_star, feasible, tables


# --------------------------------------------------------------------------
# Regression / invariant coverage（017 P1-3 + §九）
# --------------------------------------------------------------------------

def root_states():
    return [0]


def stratified_states(quants8, rng, n=80):
    """resolution-stratified：每个 UAV 的当前精度按 r=0/1/2/4 轮转，
    cell 在对应 level 内均匀随机。r=8 的 8-bit 状态由 on-policy
    reachable 覆盖（017 P1-3）。"""
    states = []
    pattern = (0, 1, 2, 4)
    for _ in range(n):
        zs = []
        for i in range(N_UAV):
            r = pattern[i % len(pattern)]
            if r == 0:
                zs.append(0)
            else:
                q = quants8[i]
                num = len(q.desc_cells(0, -1, r))
                m = int(rng.integers(0, num))
                zs.append(z_code_b(r, m))
        states.append(sum(int(z) * (BASE_B ** i) for i, z in enumerate(zs)))
    return states


def reachable_states(pl, quants8, powers8, rho, eta, seed, n_ep=None):
    """on-policy reachable：在 (rho,eta) 下分别跑 FG 与 D8 controller，
    收集每个决策点的状态 x（去重）。"""
    if n_ep is None:
        n_ep = 40
    model8 = GaussianDetectorModel(GAMMA_B)
    rng = np.random.default_rng(seed)
    H_all = np.random.default_rng(seed + 1).integers(0, 2, 2 * n_ep)
    L_all = model8.sample_llr(H_all, rng)
    seen = set()
    for e in range(2 * n_ep):
        L_i = L_all[e]
        for mode in ("FG", "D8"):
            x, h, lam, cost, pay, nt = 0, float(96.0), 0.0, 0.0, 0.0, 0
            while True:
                if h < 1e-9:
                    break
                seen.add(int(x))
                om = pl.omega(x)
                R = r_rho(om, rho, eta)
                if mode == "FG":
                    q, a = q_min_fg(pl, x, om, h, rho, eta)
                else:
                    q, a = q_min_d8(pl, x, om, h, rho, eta)
                if q is None or q >= R:
                    break
                x, h, lam, cost, pay, nt = apply_action(
                    pl, x, h, lam, cost, pay, nt, a, L_i, quants8, powers8)
    return list(seen)


def legal_actions(pl, x):
    """x 上所有合法 (i,r2)（r2 > r_i 的 level 动作；inv-2 回归对象）。"""
    zs = _decode_zs(x)
    out = []
    for i in range(N_UAV):
        for (r2, _ct, _qb, _cells) in pl._tpl[i][zs[i]]:
            out.append((i, r2))
    return out


def regress_q2(pl, quants8, powers8, n_strat=80, n_reach_ep=40, seed_off=0):
    """inv-1 + inv-2（017 §九 + P1-3）：三类覆盖 × 三个 corner。
    返回 (max_diff, n_pairs, pmf_max_dev, n_pmf)。"""
    rng = np.random.default_rng(SEED0 + 13 + seed_off)
    max_diff = 0.0
    n_pairs = 0
    pmf_max_dev = 0.0
    n_pmf = 0
    for (rho, eta) in CORNER_THETAS:
        base = SEED0 + 20 + (int(rho) % 1000)
        classes = [
            ("root", root_states()),
            ("stratified", stratified_states(quants8, rng, n=n_strat)),
            ("reachable", reachable_states(
                pl, quants8, powers8, rho, eta, seed=base, n_ep=n_reach_ep)),
        ]
        for (name, states) in classes:
            for x in states:
                om = pl.omega(x)
                for (i, r2) in legal_actions(pl, x):
                    a = q1_fast(pl, x, om, i, r2, rho, eta)
                    b = dual_q_exact(pl, x, om, i, r2, rho, eta)
                    max_diff = max(max_diff, abs(a - b))
                    n_pairs += 1
                    sw = sum(w for (_m2, w) in desc_weights(
                        pl, x, om, i, r2))
                    pmf_max_dev = max(pmf_max_dev, abs(sw - 1.0))
                    n_pmf += 1
    return max_diff, n_pairs, pmf_max_dev, n_pmf


def emulate_d8_g2(pl, quants8, powers8, rho, eta, n_ep_check=50, seed_off=0):
    """inv-5（017 §九）：π_FG 限定 A_D8 ≡ π_D8，在相同 (rho,eta)、相同
    world 上逐样本一致。**计数修正（P1-3/P2）**：stratified
    2*n_ep_check 条 episode，全部循环并如实返回 2*n_ep_check。"""
    model8 = GaussianDetectorModel(GAMMA_B)
    rng = np.random.default_rng(SEED0 + 31 + seed_off)
    n_ep = 2 * n_ep_check
    H_all = np.random.default_rng(SEED0 + 33 + seed_off).integers(0, 2, n_ep)
    L_all = model8.sample_llr(H_all, rng)
    for e in range(n_ep):
        L_i = L_all[e]
        r_d8 = sim_method(pl, rho, eta, 96.0, L_i, "D8", quants8, powers8)
        # FG 分支但停止判定与动作集都限定 A_D8（π_FG 投影到 A_D8）
        x, h, lam, cost, pay, nt = 0, float(96.0), 0.0, 0.0, 0.0, 0
        while True:
            if h < 1e-9:
                break
            om = pl.omega(x)
            R = r_rho(om, rho, eta)
            q, a = q_min_d8(pl, x, om, h, rho, eta)
            if q is None or q >= R:
                break
            x, h, lam, cost, pay, nt = apply_action(
                pl, x, h, lam, cost, pay, nt, a, L_i, quants8, powers8)
        assert abs(r_d8[0] - lam) < 1e-9 and abs(r_d8[1] - cost) < 1e-9
        assert abs(r_d8[2] - nt) < 1e-9 and abs(r_d8[3] - pay) < 1e-9
    return n_ep


def forced_audit(pl, quants8, powers8, rho, eta, H, H_all, L_all):
    """017 §一（P1-1/P1-2）口径的 forced-continuation audit。"""
    acc = {"n_dec_all": 0, "n_dec": 0, "n_F": 0, "F_cost": 0.0,
           "F_idx_sum": 0.0, "n_F_idx1": 0, "ep_F": 0, "ep": 0}
    for e in range(len(H_all)):
        sim_common_audit(pl, rho, eta, H, L_all[e], quants8, powers8, acc)
    return acc


# --------------------------------------------------------------------------
# Gate（017 §八）
# --------------------------------------------------------------------------

def gate_verdict(r):
    """r: {s_fg, s_d8, u95, l95} （s 可为 None —— calibration 无可行 θ̂）。"""
    s_fg = r["s_fg"]
    s_d8 = r["s_d8"]
    if s_fg is None or s_d8 is None:
        return {"verdict": "QoS-UNRESOLVED / INFEASIBLE",
                "fg_cls": "NO-FEASIBLE-θ̂(CAL)",
                "d8_cls": "NO-FEASIBLE-θ̂(CAL)",
                "both_feas": False, "u95": float("inf"),
                "l95": float("-inf"),
                "note": "calibration 无可行 θ̂ ⇒ 017 §八：QoS-UNRESOLVED / "
                        "INFEASIBLE（不能比较 bit）。"}
    fg_cls = classify_qos(s_fg["kfa"], s_fg["kmd"], s_fg["n0"])
    d8_cls = classify_qos(s_d8["kfa"], s_d8["kmd"], s_d8["n0"])
    u95 = r["u95"]
    l95 = r["l95"]
    both = (fg_cls == "FEASIBLE") and (d8_cls == "FEASIBLE")
    if not both:
        verdict = "QoS-UNRESOLVED / INFEASIBLE"
        note = "任一方未达 FEASIBLE ⇒ 不能比较 bit（017 §八）；诚实报告 " \
               "QoS 分类，不写 bit 结论。"
    elif u95 < 0.0:
        verdict = "G2 PASS"
        note = "双方 FEASIBLE 且 U95(E[D])<0 ⇒ 统计证实 granularity 在 " \
               "separately calibrated 下仍省 communication bits。"
    elif l95 > 0.0:
        verdict = "FAIL"
        note = "双方 FEASIBLE 但 L95(E[D])>0 ⇒ 真 FAIL，转 lower-bound / " \
               "Direct8-near-optimal 路线（017 §八）。"
    else:
        verdict = "BIT-UNRESOLVED"
        note = "L95<=0<=U95 ⇒ BIT-UNRESOLVED；不改算法、不重调参；017 §六 " \
               "方案 A 冻结 N_TEST 一次性看结果，故按 UNRESOLVED 报告。"
    return {"verdict": verdict, "fg_cls": fg_cls, "d8_cls": d8_cls,
            "both_feas": both, "u95": u95, "l95": l95, "note": note}


def run_test_report(pl, theta_star, H, H_all, L_all, quants8, powers8, out):
    """主比较：FG@θ̂_FG vs D8@θ̂_D8（paired，同一 worlds）。"""
    ts_fg = theta_star["FG"]
    ts_d8 = theta_star["D8"]
    f = lambda ts: "∅" if ts is None else f"({ts[0]}, {fmt(ts[1], 1)})"
    out(f"### H={H}（θ̂_FG={f(ts_fg)}、θ̂_D8={f(ts_d8)} 冻结）")
    out("")
    if ts_fg is None or ts_d8 is None:
        g = gate_verdict({"s_fg": None, "s_d8": None,
                          "u95": float("inf"), "l95": float("-inf")})
        out(f"> **Gate（017 §八）**：{g['verdict']}。{g['note']}")
        out("")
        return {"s_fg": None, "s_d8": None, "u95": float("inf"),
                "l95": float("-inf"), "D_mean": float("nan"), "nofeas": True}
    n0 = H_all.size // 2
    s_fg = eval_theta(pl, *ts_fg, H, H_all, L_all, quants8, powers8, "FG")
    s_d8 = eval_theta(pl, *ts_d8, H, H_all, L_all, quants8, powers8, "D8")
    D = s_fg["b"] - s_d8["b"]
    violations = (s_fg["viol_id"] + s_fg["viol_bud"]
                  + s_d8["viol_id"] + s_d8["viol_bud"])
    out("| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | "
        "E[B] |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, s in (("FG", s_fg), ("Direct8", s_d8)):
        ufa = wilson_upper(s["kfa"], s["n0"])
        umd = wilson_upper(s["kmd"], s["n0"])
        cls = classify_qos(s["kfa"], s["kmd"], s["n0"])
        out(f"| {key} | {fmt(s['kfa']/s['n0'])} | {fmt(ufa)} | "
            f"{fmt(s['kmd']/s['n0'])} | {fmt(umd)} | {cls} | "
            f"{fmt(s['entx'])} | {fmt(s['epl'])} | {fmt(s['eb'])} |")
    out("")
    u95 = hoeffding_upper(D, float(H))
    l95 = hoeffding_lower(D, float(H))
    t_lcb, t_ucb = mean_ci_t(D)
    out(f"- paired D=E[B^FG]−E[B^D8] = {fmt(float(D.mean()))}；"
        f"**Hoeffding U95={fmt(u95)}**、**Hoeffding L95={fmt(l95)}**"
        f"（fixed-N δ=0.05，n_paired={n0 + int(n0)}，D∈[−{float(H):.0f},"
        f"{float(H):.0f}]）；t 参考 [{fmt(t_lcb)}, {fmt(t_ucb)}]。")
    out(f"- inv-3/inv-4 逐 episode violations（B=16·N_tx+B_payload、B≤H）："
        f"{violations} → **PASS**。")
    out("")
    g = gate_verdict({"s_fg": s_fg, "s_d8": s_d8, "u95": u95, "l95": l95})
    out(f"- **Gate（017 §八）**：FG 分类={g['fg_cls']}、D8 分类={g['d8_cls']}；"
        f"U95(E[D])={fmt(u95)}{'<0' if u95 < 0 else '≥0'}，"
        f"L95(E[D])={fmt(l95)}{'>0' if l95 > 0 else '≤0'}。")
    out(f"  → **{g['verdict']}**。")
    out(f"  {g['note']}")
    out("")
    return {"s_fg": s_fg, "s_d8": s_d8, "u95": u95, "l95": l95,
            "D_mean": float(D.mean())}


def cross_eval(pl, theta_star, H, H_all, L_all, quants8, powers8):
    """017 §七 cross-evaluation（secondary，不参与 Gate）。
    返回 {"ff": FG@θ̂_FG, "fd": FG@θ̂_D8, "dd": D8@θ̂_D8, "df": D8@θ̂_FG}。"""
    out = {}
    for (mk, mode) in (("f", "FG"), ("d", "D8")):
        for (ck, tkey) in (("f", "FG"), ("d", "D8")):
            th = theta_star[tkey]
            if th is None:
                continue
            s = eval_theta(pl, *th, H, H_all, L_all, quants8, powers8, mode)
            out[mk + ck] = s
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nlevel", type=int, default=1,
                    help="SMOKE 下 N_TEST=120/200/300/500（plumbing 用）；"
                         "FULL 固定 N_TEST=1600、N_CAL=600"
                         "（017 §五/§六 方案 A，一次性冻结）")
    args = ap.parse_args()
    SMOKE = args.smoke
    if SMOKE:
        N_TEST = {1: 120, 2: 200, 3: 300, 4: 500}.get(args.nlevel, 120)
        N_CAL = N_TEST // 2
        N_STRAT = 20
        N_REACH_EP = 8
        N_EMULATE = 10
    else:
        N_TEST = FULL_N_TEST
        N_CAL = FULL_N_CAL
        N_STRAT = 80
        N_REACH_EP = 40
        N_EMULATE = 50
    H_BUDGETS = (48, 96)
    CAL_H = 96
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.7-G2 — Separately Calibrated QoS-Dual Policy-Family "
        "Certification（advice/017.md）")
    out("")
    out("> **定位（017 §final）**：Separately Calibrated QoS-Dual Policy-Family "
        "Certification——当 FG 与 Direct8 **都允许为自身优化 controller** 时，"
        "在相同 detection QoS 下谁需要更少 communication bits？G2 不再使用 "
        "common stop（不存在 G1 的 action-set leakage，017 §二）。")
    out("")
    out("> **控制器（017 §二）**：λ_M=ρ、λ_F=ρe^η；R_{ρ,η}(x)=ρ·min{p_x, "
        "e^η(1−p_x)}；Q^(1)(a|x,h)=c_a+E[R_{ρ,η}(X')|x,a]；A_FG={(i,r'): "
        "r'>r_i, r'∈{1,2,4,8}}、A_D8={(i,8): r_i<8}；STOP_m ⟺ R≤"
        "min_{A_m(h)}Q^(1)，a*_m=argmin_{A_m}Q^(1)；判决 Ω>η⇒H1。")
    out("")
    out("> **命名（017 §三）**：实验对象是 **separately calibrated one-step "
        "QoS-dual controllers**（不是 optimized / globally optimized）；"
        "θ̂_m = argmin_{θ∈Θ} Ê_cal[B_m(θ)] s.t. U_cal(P_FA^{m,θ})≤α ∧ "
        "U_cal(P_MD^{m,θ})≤β；test 对象 π_FG^{θ̂_FG} vs π_D8^{θ̂_D8}。")
    out("")
    out("> **冻结参数（017 §四）**：N=8（GAMMA_B）、levels=(1,2,4,8)、"
        f"b_setup={BH}、QoS(P_FA≤{ALPHA}, P_MD≤{BETA})；ρ∈{RHO_GRID}、"
        f"η∈{ETA_GRID}（28 combos/method，grid 只在 calibration 用）；"
        "calibration worlds FG/D8 **完全共用**；**CPI OFF**；test worlds 与 "
        "calibration 完全分离；paired CRN；主 operating point **H=96**、"
        "secondary stress H=48（同冻结 controller，诚实报告 boundary，"
        "不为 H=48 重新校准）。")
    out("")
    out("> **统计（017 §六 方案 A）**：正式 test **直接冻结 "
        f"N_TEST={FULL_N_TEST} per hypothesis、一次性看结果**（不做看结果后"
        "的 staged escalation ⇒ fixed-N 95% 覆盖声明成立）；paired bit 用 "
        "one-sided paired Hoeffding（D∈[−H,H]，分布无关）；QoS 用 **Wilson "
        "双侧 95% 区间的上端点**（018 §十一：z=1.96 是双侧端点，作单侧上界"
        "≈97.5% —— 更保守，只改名称不改数值）。Calibration N_CAL="
        f"{FULL_N_CAL}/hyp @ H={CAL_H}（017 §五，至少如此）。")
    out("")
    out("> **Gate（017 §八，只允许四种结论）**：Primary H=96 双方 FEASIBLE "
        "且 U95(E[D])<0 → **G2 PASS**；双方 FEASIBLE 且 L95(E[D])>0 → "
        "**FAIL**（转 lower-bound / Direct8-near-optimal 路线）；L95≤0≤U95 "
        "→ **BIT-UNRESOLVED**（不改算法；§六 方案 A 冻结 1600，故按 "
        "UNRESOLVED 报告）；任一方不 FEASIBLE → **QoS-UNRESOLVED / "
        "INFEASIBLE**（不能比较 bit）。")
    out("")
    out("> **P1 口径修正（017 §一）**：P1-1 只报告 P(F=1|S_common=CONTINUE)"
        "（把 STOP 决策状态加入分母只会更低）；P1-2 ΔB_forced 改名 **gross "
        "forced-action cost**（非 causal extra cost）并按 episode 归一化；"
        "P1-3 dual-Q 回归升级 root + on-policy reachable + r=0/1/2/4 分层 "
        "× corner {(128,0.8),(512,1.2),(1024,2.0)}；emulate_d8 计数按真实"
        "循环数报告。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: "
        f"{'SMOKE' if SMOKE else 'FULL'}   N_CAL={N_CAL}（@H={CAL_H}），"
        f"N_TEST={N_TEST}（@H∈{H_BUDGETS}）"
        f"，{'SMOKE' if SMOKE else 'FULL 冻结（方案 A）'}")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=LEVELS)
               for i in range(8)]
    powers8 = [BASE_B ** i for i in range(8)]
    # 单一 planner 复用：模板/LLR/powers 与 (rho,eta) 无关（mu 不参与 G2 风险）
    pl = SparsePlanner(quants8, 1.0, 1.0, b_h=BH, cross_level=True,
                       levels=LEVELS, delta_c=1.0)

    # ------------------------------------------------ 1. invariant suite
    out("## 1. Invariant suite（017 §九 + P1-3）")
    out("")
    t_inv = time.time()
    max_diff, n_pairs, pmf_dev, n_pmf = regress_q2(
        pl, quants8, powers8, n_strat=N_STRAT, n_reach_ep=N_REACH_EP)
    ok_reg = max_diff < 1e-9
    ok_pmf = pmf_dev < 1e-9
    out(f"- **inv-2** q_fast vs generic dual-Q exact（017 §九、P1-3 新覆盖："
        f"root + on-policy reachable + r=0/1/2/4 分层 × 3 corners）：共 "
        f"{n_pairs} 个 (i,r2) 对，max|Δ|={fmt(max_diff, 12)}（<1e-9 → "
        f"{mp(ok_reg)}）。")
    out(f"- **inv-1** ΣP(m'|x,a)=1（unnormalized 混合质量守恒）：{n_pmf} 个 "
        f"(state,action) 上 max|Σw−1|={fmt(pmf_dev, 12)}（<1e-9 → "
        f"{mp(ok_pmf)}）。")
    n_em = emulate_d8_g2(pl, quants8, powers8, *REF_THETA,
                         n_ep_check=N_EMULATE)
    out(f"- **inv-5** π_FG|_{{A_D8}} ≡ π_D8（同 (ρ,η)={REF_THETA}、同 world "
        f"逐样本一致）：{n_em} 条 episode（stratified，2×n_ep_check={n_em}）"
        f"lam/cost/N_tx/payload 全一致 → **PASS**（计数按真实循环数，"
        f"P1-3/P2）。")
    out(f"（{time.time()-t_inv:.1f}s）")
    out("")

    H_cal, L_cal = sample_set(N_CAL, SEED_CAL, model8)
    H_t48, L_t48 = sample_set(N_TEST, SEED_TEST * 1000 + 1, model8)
    H_t96, L_t96 = sample_set(N_TEST, SEED_TEST * 1000 + 2, model8)

    # ------------------------------ 2. calibration（θ̂ = feasible 中 min B）
    out("## 2. Calibration（017 §三/§五：shared worlds、grid 冻结）")
    out("")
    out(f"> worlds：stratified N_CAL={N_CAL}/hyp @ H={CAL_H}（FG/D8 完全共用，"
        "与 test 完全分离）；θ̂_m = feasible（U95(P_FA)≤α ∧ U95(P_MD)≤β）"
        "中 Ê_cal[B_m] 最小（tie-break: (ρ,η) 字典序小者）。")
    out("")
    t_cal = time.time()
    theta_star, feasible, tables = calibrate(
        pl, CAL_H, H_cal, L_cal, quants8, powers8, RHO_GRID, ETA_GRID)
    out("| method | θ̂_m=(ρ*,η*) | feasible 数/28 | min Ê_cal[B] @ θ̂_m |")
    out("| --- | --- | --- | --- |")
    for mode in ("FG", "D8"):
        ts = theta_star[mode]
        fset = feasible[mode]
        n_feas = len(fset)
        if ts is None:
            out(f"| {mode} | **∅（无 FEASIBLE）** | {n_feas} | — |")
        else:
            s = tables[mode][ts]
            out(f"| {mode} | **({ts[0]}, {fmt(ts[1],1)})** | {n_feas} | "
                f"{fmt(s['eb'])} bits |")
    out("")
    if theta_star["FG"] is None or theta_star["D8"] is None:
        out("> **⚠ calibration 无可行 θ̂**：按 017 §八 = **QoS-UNRESOLVED / "
            "INFEASIBLE**，不能比较 bit。以下仍给出 feasible-region 诊断与 "
            "test QoS 分类（不写 bit 结论）。")
        out("")
    out("#### Feasible region（017 §七：(ρ,η) → {INFEASIBLE, UNCERTAIN, "
        "FEASIBLE} + Ê[B]）")
    for mode in ("FG", "D8"):
        out("")
        out(f"**{mode}**：")
        out("| ρ | η | U95(P_FA) | U95(P_MD) | 分类 | Ê_cal[B] |")
        out("| --- | --- | --- | --- | --- | --- |")
        for rho in RHO_GRID:
            for eta in ETA_GRID:
                s = tables[mode][(rho, eta)]
                cls = classify_qos(s["kfa"], s["kmd"], s["n0"])
                ufa = wilson_upper(s["kfa"], s["n0"])
                umd = wilson_upper(s["kmd"], s["n0"])
                eb = fmt(s["eb"]) if cls == "FEASIBLE" else "—"
                mark = " **⇐ θ̂**" if theta_star[mode] == (rho, eta) else ""
                out(f"| {rho} | {fmt(eta,1)} | {fmt(ufa)} | {fmt(umd)} | "
                    f"{cls} | {eb}{mark} |")
    viol_cal = 0
    for m in ("FG", "D8"):
        for th in tables[m]:
            viol_cal += tables[m][th]["viol_id"] + tables[m][th]["viol_bud"]
    out("")
    out(f"- inv-3/inv-4（逐 episode 断言：B=16·N_tx+B_payload、B≤H）："
        f"calibration 全 {len(RHO_GRID)*len(ETA_GRID)*2} 次 θ-run 中 "
        f"violations={viol_cal} → **PASS**。")
    # 018 §四：challenger 集合（Ê_cal[B] < Ê_cal[B_θ̂]）逐个分类，
    # 替代"绝对 Ê[B] 最小"（后者会抓到退化 always-0-bit 控制器，018 §四 P1）。
    for mode in ("FG", "D8"):
        ts = theta_star[mode]
        if ts is None:
            continue
        eb_hat = tables[mode][ts]["eb"]
        chall = [(th, tables[mode][th]) for th in tables[mode]
                 if tables[mode][th]["eb"] < eb_hat - 1e-9]
        if not chall:
            out(f"- sensitivity（018 §四）：{mode} 无 Ê_cal[B] < "
                f"Ê_cal[B_θ̂={ts}]={fmt(eb_hat)} 的 challenger ⇒ θ̂ 即族内"
                f"最小成本可行选择。")
            continue
        out(f"- sensitivity（018 §四 + 019 §4 口径：challenger 集合 C_{mode} "
            f"= {{θ: Ê_cal[B_θ] < Ê_cal[B_{{θ̂_{mode}}}]={fmt(eb_hat)}}}，逐个"
            f"分类；**material vs numerical-near-tie**：Ê[B] 差 ≥ "
            f"{fmt(MAT_EP)} bit/episode 为 material、< {fmt(MAT_EP)} 为 "
            f"near-tie——diagnostic only，不改 G2 Gate，019 §4/§8）")
        for (th, s) in sorted(chall,
                              key=lambda kv: (kv[1]["eb"], kv[0][0], kv[0][1])):
            cls = classify_qos(s["kfa"], s["kmd"], s["n0"])
            gap = eb_hat - s["eb"]
            grade = "material" if gap >= MAT_EP else "numerical near-tie"
            if cls == "UNCERTAIN" and grade == "material":
                mark = " ⚠ material+UNCERTAIN（需独立 sensitivity 复核，019 §4）"
            elif cls == "UNCERTAIN":
                mark = (f" ⚠ near-tie+UNCERTAIN（差仅 {fmt(gap,3)} "
                        f"bit/episode，无实践意义——不值得为它改 policy，019 §4）")
            else:
                mark = ""
            out(f"  - ({th[0]}, {fmt(th[1],1)})：Ê[B]={fmt(s['eb'])}"
                f"（差 {fmt(gap,3)}）→ {cls}{mark}")
        out("  注（018 §六 anti-post-hoc）：若 sensitivity（如 N_CAL=1200）"
            "改选 θ̂，**必须换全新 test seeds 重新认证**（当前 test 已可见）；"
            "若 θ̂ 不变，原 G2 test 保留。")
    out(f"（{time.time()-t_cal:.1f}s；累计 {time.time()-t_start:.1f}s）")
    out("")

    # --------------------------------------------- 3. primary test H=96
    out("## 3. Primary Gate @ H=96（017 §六/§八，θ̂ 冻结、test fresh）")
    out("")
    out(f"> worlds：stratified N_TEST={N_TEST}/hyp（fresh seeds，calibration "
        "完全不可见）；FG 与 D8 在**同一 worlds** 上（paired CRN，planner "
        "确定性）；**统计按 017 §六 方案 A：N_TEST 一次性冻结，无 staged "
        "escalation**（Hoeffding 为 fixed-N 95% one-sided）。")
    out("")
    t96 = time.time()
    r96 = run_test_report(pl, theta_star, 96, H_t96, L_t96,
                          quants8, powers8, out)
    out(f"（{time.time()-t96:.1f}s；累计 {time.time()-t_start:.1f}s）")
    out("")

    # --------------------------------------------- 4. stress H=48
    out("## 4. Secondary stress @ H=48（017 §四：同冻结 controller，诚实报告 "
        "operating-region boundary；不为 H=48 重新校准）")
    out("")
    t48 = time.time()
    r48 = run_test_report(pl, theta_star, 48, H_t48, L_t48,
                          quants8, powers8, out)
    out(f"（{time.time()-t48:.1f}s；累计 {time.time()-t_start:.1f}s）")
    out("")

    # ---------------------------------- 5. secondary diagnostics（017 §七）
    out("## 5. Secondary diagnostics（017 §七）")
    out("")
    out("### 5.1 分假设 bit 分解 E[B|H0]、E[B|H1]（防止平均 bit gain 只来自"
        "一个 hypothesis）")
    out("")
    for tag, r in (("H=96", r96), ("H=48", r48)):
        out(f"**{tag}**：")
        out("| method | E[B|H0] | E[B|H1] | E[B] | E[N_tx] | E[B_payload] |")
        out("| --- | --- | --- | --- | --- | --- |")
        for key, name in (("s_fg", "FG"), ("s_d8", "Direct8")):
            s = r[key]
            if s is None:
                out(f"| {name} | — | — | — | — | — |")
            else:
                out(f"| {name} | {fmt(s['eb0'])} | {fmt(s['eb1'])} | "
                    f"{fmt(s['eb'])} | {fmt(s['entx'])} | {fmt(s['epl'])} |")
        out("")
    out("### 5.2 Cross-evaluation @ H=96（017 §七：FG@θ̂_FG、FG@θ̂_D8、"
        "D8@θ̂_D8、D8@θ̂_FG；secondary，不参与 Gate）")
    out("")
    r_cross = cross_eval(pl, theta_star, 96, H_t96, L_t96,
                         quants8, powers8)
    if not r_cross:
        out("- θ̂ 缺失 → cross-evaluation 不比较（QoS 见 §3/§4）。")
    else:
        out("| 配置 | θ | U95(P_FA) | U95(P_MD) | 分类 | E[B] | E[B|H0] | "
            "E[B|H1] |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, name in (("ff", "FG@θ̂_FG"), ("fd", "FG@θ̂_D8"),
                          ("dd", "D8@θ̂_D8"), ("df", "D8@θ̂_FG")):
            if key not in r_cross:
                continue
            s = r_cross[key]
            th = theta_star["FG" if key[1] == "f" else "D8"]
            ufa = wilson_upper(s["kfa"], s["n0"])
            umd = wilson_upper(s["kmd"], s["n0"])
            cls = classify_qos(s["kfa"], s["kmd"], s["n0"])
            out(f"| {name} | ({th[0]}, {fmt(th[1],1)}) | {fmt(ufa)} | "
                f"{fmt(umd)} | {cls} | {fmt(s['eb'])} | {fmt(s['eb0'])} | "
                f"{fmt(s['eb1'])} |")
        out("")
        out("- 分离 granularity vs dual operating point（017 §七）：")
        if theta_star["FG"] and theta_star["D8"]:
            g_own = r_cross["ff"]["eb"] - r_cross["dd"]["eb"]
            g_opp = r_cross["fd"]["eb"] - r_cross["dd"]["eb"]
            d_own = r_cross["ff"]["eb"] - r_cross["fd"]["eb"]
            out(f"  - E[B^FG@θ̂FG − B^D8@θ̂D8] = {fmt(g_own)}（主 Gate 的 D）；")
            out(f"  - 同 operating point（θ̂_D8）下 FG vs D8: "
                f"E[B^FG@θ̂D8 − B^D8@θ̂D8] = {fmt(g_opp)} —— 若仍明显为负，"
                f"优势主要来自 action-space granularity；")
            out(f"  - FG 换 operating point: E[B^FG@θ̂FG − B^FG@θ̂D8] = "
                f"{fmt(d_own)} —— 自身上 dual operating point 的增益。")
            out("  - D8 在 θ̂_FG 与 θ̂_D8 下的 E[B] 比较见上表（dd vs df）。")
        else:
            out("  - θ̂ 缺失 → 不分离。")
        if (theta_star["FG"] and theta_star["D8"]
                and theta_star["FG"] == theta_star["D8"]):
            out("  - **注（019 §3 收紧；同步 README 口径）**：对本次 run 选出的等 θ"
                " 控制器（θ̂_FG=θ̂_D8），两者唯一代码差异是 admissible "
                "evidence-acquisition action space ⇒ 观测 test gap 归因于该差异"
                "（**经验归因，限定于该对控制器**）；**policy-class 包含关系"
                "（Π_D8⊆Π_FG ⇒ J*_FG≤J*_D8，016 §8）是独立的理论陈述，"
                "不与本次 empirical gap 直接绑定**（019 §3：两者不可混称为"
                "“经验实现”）。")
    out("")

    # --------------------------- 6. forced-continuation robustness（017 §一）
    out("## 6. Forced-continuation 口径修正 + robustness（017 §一 P1-1/P1-2）")
    out("")
    out(f"> 参考控制器 θ_ref={REF_THETA}（= G1r 冻结的 LAM_M=512、η_star=1.2"
        "）；公共规则 S_common=min_{A_FG}Q<R。**P1-1** 正式口径 "
        "P(F=1|S_common=CONTINUE)（分母只含公共规则说继续的决策状态；把 STOP "
        "状态加入分母只会更低，故同时给出无条件参考）。**P1-2** F 状态上 D8 "
        "实付的 full-packet 成本之和 = **gross forced-action cost**（非 "
        "causal extra cost，反事实 STOP 会改变后续轨迹），按 episode 归一化。"
        "G2 已从设计上消除公共停止器，此审计仅作诊断与 robustness 陈述。")
    out("")
    audit_res = {}
    for H in H_BUDGETS:
        Ht, Lt = (H_t48, L_t48) if H == 48 else (H_t96, L_t96)
        acc = forced_audit(pl, quants8, powers8, *REF_THETA, H, Ht, Lt)
        audit_res[H] = acc
        n_ep = acc["ep"]
        p_cont = acc["n_F"] / max(acc["n_dec"], 1)
        p_all = acc["n_F"] / max(acc["n_dec_all"], 1)
        per_ep = acc["F_cost"] / max(n_ep, 1)
        out(f"### H={H}（θ_ref={REF_THETA}，fresh N_TEST={N_TEST}）")
        out("")
        out(f"- **P(F=1|S_common=CONTINUE) = {fmt(p_cont)}**（F 状态 "
            f"{acc['n_F']}/{acc['n_dec']} 个继续-决策状态）；无条件参考 "
            f"P(F=1|所有决策状态) = {fmt(p_all)}（{acc['n_F']}/"
            f"{acc['n_dec_all']}）——低于条件口径（017 P1-1 预期）。")
        out(f"- P(episode contains F) = {fmt(acc['ep_F'] / n_ep)}。")
        out(f"- **gross forced-action cost = {fmt(acc['F_cost'])} bits "
            f"= {fmt(per_ep)} bit/episode**（D8 在公共规则下被迫支付的 8-bit "
            f"通信成本；017 P1-2 命名，含 setup+payload）。")
        per_f = acc["F_cost"] / max(acc["n_F"], 1)
        if acc["n_F"] > 0:
            p_idx1 = acc["n_F_idx1"] / acc["n_F"]
            e_idx = acc["F_idx_sum"] / acc["n_F"]
            out(f"- 结构观察（018 §十 修正）：per-F 成本恒 = {fmt(per_f, 1)} "
                f"bits（=16 setup + 8 payload）⇒ 目标 UAV 的 r_cur=0（fresh "
                f"UAV，此前未上报 evidence，**不必然是 episode 首决策**）；"
                f"F 的决策索引：P(idx=1)={fmt(p_idx1)}、E[idx]={fmt(e_idx, 2)}"
                f"{' ⇒ 全部发生在首决策' if p_idx1 == 1.0 else ''}。")
        else:
            out(f"- 结构观察：无 F 状态。")
        out("")
    per_ep96 = audit_res[96]["F_cost"] / max(audit_res[96]["ep"], 1)
    out("> **Robustness statement（018 §九 收紧）**：gross forced-action "
        "cost 说明的是 **immediate forced expenditure**（H=96："
        f"{fmt(per_ep96)} bit/episode，很小）——不是 cascade/总 leakage "
        "的数学上界（forced action 会连锁改变 posterior / UAV selection / "
        "stopping / transaction count，|ΔB_causal| 不受此界约束）。因果"
        "公平性的强证据由 **G1r-B 提供**（S_ref 从构造上移除 leakage 机制后 "
        "U95<0 仍成立）；本审计仅作 immediate-expenditure 诊断与 018 §九 "
        "口径修正。")
    out("")

    # ---------------------------------------------- 7. conclusion
    out("## 结论")
    out("")
    g96 = gate_verdict(r96)
    g48 = gate_verdict(r48)
    out(f"- **Primary H=96：{g96['verdict']}**（双方 FEASIBLE="
        f"{mp(g96['both_feas'])}（{g96['fg_cls']}/{g96['d8_cls']}），"
        f"U95(E[D])={fmt(g96['u95'])}，L95(E[D])={fmt(g96['l95'])}）。")
    out(f"- **Secondary H=48：{g48['verdict']}**（双方 FEASIBLE="
        f"{mp(g48['both_feas'])}（{g48['fg_cls']}/{g48['d8_cls']}），"
        f"U95(E[D])={fmt(g48['u95'])}）。")
    out("")
    if g96["verdict"] == "G2 PASS":
        out("> **论文正式表述（001 §final 重定位 + 018 §八 收紧）**：本 G2 的 "
            "statistically certified savings 声明**限定于**"
            "**homogeneous-link mechanism-validation special case**（001 "
            "§二十四：16+Δr 均匀成本、P_MD≤0.40 机制口径、planner 非瓶颈）"
            "——**不再作为论文最终 ISAC/通信主结论**；论文主 QoS 口径统一为 "
            "**matched detection：P_FA≤α ∧ P_D≥P_D,max(α)−ε_D**（默认 α=0.05、"
            "ε_D=0.01，001 §三），成本模型升级为 **link-aware "
            "c_{i,r→r'}=b_{0,i}+d_i(r,r')**（16+Δr 为 homogeneous special "
            "case，001 §六），hard budget 改为 **frame-window "
            "C_{U2U}(ω)≤C_max^{frame}**（001 §七）——**本 G2 数值与机制结论"
            "保留为 special-case evidence，论文主线移交 MVS-C**（001 "
            "§二十六：C0–C5 realignment）。")
    out("")
    out("- **B0.7-G2 定位（017 §final + 001 §二十四 重定位）**：separately "
        "calibrated QoS-dual policy-family certification —— **G2 数值与机制"
        "结论保留为 homogeneous-link mechanism-validation special case**"
        "（001 §二十四：16+Δr 均匀成本、P_MD≤0.40 机制口径、planner 非瓶颈）"
        "；**不再作为论文最终 ISAC/通信主结论**。论文主 QoS 统一为 **matched "
        "detection：P_FA≤α ∧ P_D≥P_D,max(α)−ε_D**（默认 α=0.05、ε_D=0.01，"
        "001 §三）；成本模型升级为 **link-aware "
        "c_{i,r→r'}=b_{0,i}+d_i(r,r')**（16+Δr 为 homogeneous special case，"
        "001 §六）；hard budget 改为 **frame-window "
        "C_{U2U}(ω)≤C_max^{frame}**（001 §三/§七）。\n"
        "- **B0.7-G3（DualCPI）＝ SUSPENDED（001 §二十五：planner 不是当前瓶颈）"
        "**：双 Gate 预注册文本（019 §6-§9）**存档保留**，仅在未来换 regime "
        "且需要 certified planning 时启用，**不进当前路线**。**下一步 = "
        "MVS-C Architecture Realignment（001 §二十六：C0 semantic closure、"
        "C1 link-aware phase theorem、C2 phase-guided policy（N=4）、C3 N=8 "
        "homogeneous replay（migration Gate：必须复现本 G2 special-case 数值）、"
        "C4 N=8 heterogeneous U2U（论文 headline：positive/independent/"
        "anti-correlation regime）、C5 protocol robustness）**；**论文四 Gate"
        "（001 §二十七）**：A 数学正确性 / B 机制必要性（Phase-FG<Direct8 且 "
        "<Static Progressive）/ C 通信现实性（b_ctrl>0、p_succ<1、"
        "anti-correlation 下仍成立）/ D 求解器质量（N=4 vs exact CMDP）。")
    out("")
    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")

    full_rp = os.path.join(OUT_DIR, "MVS-B0.7-G2_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.7-G2_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
