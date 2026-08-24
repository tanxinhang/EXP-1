"""MVS-C C2: phase-guided conditional-refinement policy + exact budgeted-CMDP
oracle (001 §二十六.2-§26.3，001 §九-§十四，001 §二十七 Gate D)。

定位（001 §二十六.2）：在 C1（link-aware phase theorem 数值验证已完成）之后，
实现 **phase-guided policy（先 N=4）**：

    Q_i^{prog}(s) = c_i(r,r_+) + E[ min{ R(S^+), c_i(r_+,r_max) + E[R(S^max)|S^+] } ]
    Q_i^{dir}(s)  = c_i(r,r_max) + E[R(S^max)|s]
    Q_i^{local}(s)= min{ Q_i^{dir}, Q_i^{prog} }
    i* = argmin_i Q_i^{local}(s)； R(s) <= min_i Q_i^{local}(s)  => STOP

**theory-certified action pruning（001 §十四）**：对 UAV i，若
g_{s,i}(b_i)=E[min{D_{s,i}-d_{2,i}, b_i}] >= 0 ⇒ Q_i^{prog} >= Q_i^{dir}
⇒ probe 剔除，A_keep(s) = {a_i^{dir}} ∪ {a_i^{probe} : g<0} ——每个 UAV 最多
2 个 action（probe/jump），O(2N) 而非 O(N|R|)。

每个决策后收到真实 message 再对**所有 UAV** receding 重规划（001 §十）。

Gate D（001 §二十七，C2 主 Gate）：N=4 下与 **exact budgeted CMDP**（001
§二十一：exact memoized backward Bellman 仅 N=4）比对
rel_gap = (C_PhaseFG - C_CMDP*)/C_CMDP*，预注册阈值 0.10。
比较对象（C2 审计确定）：C_CMDP* = π* 的**通信成本 E[B]**（用
exact_policy_cost + oracle_decision，memo 已由 solve 预热）；Lagrangian 值
V_lag 含终端 dual risk，不能当通信成本直接用。

口径（C0 已封）：成本 link-aware homogeneous special case
c_{i,r->r'} = b_{0,i} + d_i(r,r') = 16 + (r'-r)（001 §六，κ_i=1）；
hard budget = frame-window C_{U2U}(ω) <= C_max^{frame}（H=48/96）；
主 QoS = matched detection：P_FA<=α ∧ P_D>=P_D,max(α)-ε_D（α=0.05、
ε_D=0.01，001 §三），P_D,max 用**精确量化全融合**（卷积，非连续近似）。
8-bit 精确向后归纳在此 N=4 下 reachable 状态 ~4.6e9 ⇒ 不可行（本文档
给出 profile 计数估计）；Gate D 的 oracle 在 {1,2,4} 粒度（23^4=279,841，
MVS-A ExactDP 规模，可行）上做 ——两个粒度各自 apples-to-apples。

matched 口径统计不可认证诊断（C2 审计）：N=4 弱感知下 P_D,max^q(0.05)=0.8482
⇒ 目标 P_D>=0.8382（β=0.1618）；量化全融合实际可达 ~0.845-0.848 ⇒
P_D,max−P_D 余量 ~0.005-0.008，而 Wilson 95% 半宽 n=400 → ±0.035、n=800 →
±0.025 ⇒ **ε_D=0.01 的认证余量被 CI 消耗殆尽**：matched FEASIBLE 在该系统
规模下统计不可认证（任何 θ 都只会 UNCERTAIN/INFEASIBLE —— 不是搜索失败，
是与 reference 的余量 < CI 半宽的固有事实）。机制口径（β=0.40）余量 ~0.20
≫ CI，是 C2 的可认证比较面（017 §四 同口径）。
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time

import numpy as np

from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA4 = [-1.0, 1.0, 3.0, 5.0]
LEVELS8 = (1, 2, 4, 8)
LEVELS4 = (1, 2, 4)
BH = 16.0
ALPHA = 0.05
EPS_D = 0.01
H_BUDGETS = (48, 96)
RHO_GRID = (128, 256, 512, 1024)
# C2 冻结网格（本 runner 自有协议参数）：η 增补 0.85 —— matched band
# η∈[0.81,0.91] 落在 0.8/1.0 之间的空档（N=4 全融合 P_FA(η)=1−Φ((η+3.605)/2.685)、
# P_MD(η)=Φ((η−3.605)/2.685)）。
ETA_GRID = (0.8, 0.85, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
GATE_D_CORNERS = ((256, 1.2), (512, 1.2), (1024, 1.6))
GATE_D_REL = 0.10            # rel_gap 阈值（预注册）
SEED0 = 2026
SEED_CAL = SEED0 + 100
SEED_TEST = SEED0 + 200
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
SMOKE_OUT_DIR = os.path.join(OUT_DIR, "smoke")
DELTA = 0.05                 # paired Hoeffding δ
Z95 = 1.96                   # Wilson 双侧端点（018 §十一 口径）

FULL_N_CAL = 400             # per hypothesis @ H=96
FULL_N_TEST = 800            # per hypothesis @ H in {48, 96}


def fmt(x, nd=4):
    if x == float("inf"):
        return "inf"
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def mp(flag):
    return "PASS" if flag else "FAIL"


def wilson_upper(k, n, z=Z95):
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return min(1.0, center + half)


def wilson_lower(k, n, z=Z95):
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half)


def hoeffding_upper(xs, H_ub, delta=DELTA):
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return float("inf")
    return float(xs.mean()) + 2.0 * H_ub * math.sqrt(math.log(1.0 / delta) / (2.0 * n))


# ------------------------------------------------------------ dual risk helpers
def r_dual(om, rho, eta):
    """R_{rho,eta}(x) = rho * min{ p_x, e^eta (1-p_x) }（018 §三：ρ 即
    conditional-error Lagrange 的 effective multiplier，无缩放步骤）。"""
    p = 1.0 / (1.0 + math.exp(-om))
    return rho * min(p, math.exp(eta) * (1.0 - p))


def _w(lp, lq, lp1c, lp0c):
    a_ = lp + lp1c
    b_ = lq + lp0c
    m_ = a_ if a_ >= b_ else b_
    return math.exp(m_ + math.log1p(math.exp(-abs(a_ - b_))))


def _logp(om):
    return -math.log1p(math.exp(-om))


def _logq(om):
    return -math.log1p(math.exp(om))


# -------------------------------------------------------------- phase support
def phase_support(pl, x, om, i, rho, eta):
    """UAV i 在状态 x（om 为 posterior log-odds）上的 probe/direct 支撑。

    probe  r -> r_next -> r_max（两次 transaction：c1=16+Δ1、c2=16+Δ2）
    dir    r -> r_max    （一次：c_dir = 16 + (r_max - r)）
    分支权重 w1 = Pr(X_1 branch | x)；R1 = R_{rho,eta}(x1)；
    E_R = E[R(X_2)|x1]（r_next -> r_max 的 tower 目标）。
    Y = (R1 - E_R) - d2，g(b) = E[min{Y,b}]；g0 := g(16)（setup 成本）。
    恒等式（013 §1）：Q_prog - Q_dir = E[min{Y,16}] + (E_R_sum - E_dir)，
    tower: E_R_sum == E_dir（nested 精确成立）——T1 检查对象。

    返回 None（leaf）或 dict（含 branches、Q_prog、Q_dir、g0、E_dir、E_R_sum）。
    """
    zi = (x // pl.powers[i]) % BASE_B
    r, _m = z_decode_b(zi)
    if r >= pl.r_max:
        return None
    r_next = next((r2 for r2 in pl.levels if r2 > r), None)
    r_max = pl.r_max
    if r_next is None:
        return None
    tpl_i = pl._tpl[i][zi]
    dir_tpl = next((a for a in tpl_i if a[0] == r_max), None)
    prog_tpl = next((a for a in tpl_i if a[0] == r_next), None)
    if dir_tpl is None or prog_tpl is None:
        return None
    d2 = r_max - r_next
    lp = _logp(om)
    lq = _logq(om)
    E_dir = 0.0
    for (m2, lp0c, lp1c) in dir_tpl[3]:
        w = _w(lp, lq, lp1c, lp0c)
        z2 = z_code_b(r_max, m2)
        om2 = om + pl._llr_i[i][z2] - pl._llr_i[i][zi]
        E_dir += w * r_dual(om2, rho, eta)
    c1 = BH + (r_next - r)
    c2 = BH + d2
    branches = []
    for (m1, lp0c1, lp1c1) in prog_tpl[3]:
        w1 = _w(lp, lq, lp1c1, lp0c1)
        z1 = z_code_b(r_next, m1)
        x1 = x + (z1 - zi) * pl.powers[i]
        om1 = om + pl._llr_i[i][z1] - pl._llr_i[i][zi]
        R1 = r_dual(om1, rho, eta)
        E_R = 0.0
        ref = next((a for a in pl._tpl[i][z1] if a[0] == r_max), None)
        if ref is not None:
            lp1 = _logp(om1)
            lq1 = _logq(om1)
            for (m2, lp0c, lp1c) in ref[3]:
                w2 = _w(lp1, lq1, lp1c, lp0c)
                z2 = z_code_b(r_max, m2)
                om2 = om1 + pl._llr_i[i][z2] - pl._llr_i[i][z1]
                E_R += w2 * r_dual(om2, rho, eta)
        D = R1 - E_R
        Y = D - d2
        branches.append((w1, x1, R1, E_R, D, Y))
    wsum = sum(br[0] for br in branches)
    E_R_sum = sum(br[0] * br[3] for br in branches) / wsum
    g0 = sum(br[0] * min(br[5], BH) for br in branches) / wsum
    Q_prog = c1 + sum(br[0] * min(br[2], c2 + br[3]) for br in branches) / wsum
    Q_dir = (BH + (r_max - r)) + E_dir
    EY = sum(br[0] * br[5] for br in branches) / wsum
    return {"r": r, "r_next": r_next, "r_max": r_max, "d2": d2,
            "c1": c1, "c2": c2, "branches": branches, "wsum": wsum,
            "E_dir": E_dir, "E_R_sum": E_R_sum, "g0": g0,
            "Q_prog": Q_prog, "Q_dir": Q_dir, "EY": EY}


# --------------------------------------------------------------- q1 (myopic)
def q1(pl, x, om, i, r2, rho, eta):
    zi = (x // pl.powers[i]) % BASE_B
    r_cur, _ = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    tpl = next(a for a in pl._tpl[i][zi] if a[0] == r2)
    lp = _logp(om)
    lq = _logq(om)
    E = 0.0
    for (m2, lp0c, lp1c) in tpl[3]:
        w = _w(lp, lq, lp1c, lp0c)
        z2 = z_code_b(r2, m2)
        om2 = om + pl._llr_i[i][z2] - pl._llr_i[i][zi]
        E += w * r_dual(om2, rho, eta)
    return c + E


# ----------------------------------------------------------------- decisions
def phase_decision(pl, x, om, h, rho, eta):
    diag = {"full_actions": 0, "kept_actions": 0, "pruned": 0}
    cands = []
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        sup = phase_support(pl, x, om, i, rho, eta)
        if sup is None:
            continue
        c_dir = BH + (sup["r_max"] - sup["r"])
        diag["full_actions"] += len([r2 for r2 in pl.levels if r2 > sup["r"]])
        if sup["g0"] >= 0.0:
            diag["pruned"] += 1
            if c_dir <= h:
                cands.append((sup["Q_dir"], ("ACT", i, "JUMP", sup["r_max"])))
                diag["kept_actions"] += 1
        else:
            if sup["c1"] <= h:
                cands.append((sup["Q_prog"], ("ACT", i, "PROBE", sup["r_next"])))
                diag["kept_actions"] += 1
            if c_dir <= h:
                cands.append((sup["Q_dir"], ("ACT", i, "JUMP", sup["r_max"])))
                diag["kept_actions"] += 1
    if not cands:
        return ("STOP",), diag
    best_q, best_a = min(cands, key=lambda t: t[0])
    R = r_dual(om, rho, eta)
    if R <= best_q:
        return ("STOP",), diag
    return best_a, diag


def myopic_decision(pl, x, om, h, rho, eta):
    diag = {"full_actions": 0, "kept_actions": 0, "pruned": 0}
    cands = []
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        for r2 in pl.levels:
            if r2 <= r:
                continue
            c = BH + (r2 - r)
            diag["full_actions"] += 1
            if c <= h:
                cands.append((q1(pl, x, om, i, r2, rho, eta),
                              ("ACT", i, "ANY", r2)))
                diag["kept_actions"] += 1
    if not cands:
        return ("STOP",), diag
    best_q, best_a = min(cands, key=lambda t: t[0])
    R = r_dual(om, rho, eta)
    if R <= best_q:
        return ("STOP",), diag
    return best_a, diag


def direct_decision(pl, x, om, h, rho, eta):
    diag = {"full_actions": 0, "kept_actions": 0, "pruned": 0}
    cands = []
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        c = BH + (pl.r_max - r)
        diag["full_actions"] += 1
        if c <= h:
            cands.append((q1(pl, x, om, i, pl.r_max, rho, eta),
                          ("ACT", i, "ANY", pl.r_max)))
            diag["kept_actions"] += 1
    if not cands:
        return ("STOP",), diag
    best_q, best_a = min(cands, key=lambda t: t[0])
    R = r_dual(om, rho, eta)
    if R <= best_q:
        return ("STOP",), diag
    return best_a, diag


def oracle_decision(pl_oracle, x, om, h, rho, eta):
    _val, act = pl_oracle.solve(int(x), float(h))
    if act is None:
        return ("STOP",), {}
    i, r2 = act
    return ("ACT", i, "ANY", r2), {}


# ---------------------------------------------------------------- evaluation
def sample_set(n, seed, model):
    rng = np.random.default_rng(seed)
    H = model.sample_hypotheses(n, rng)
    L = model.sample_llr(H, rng)
    return H, L


def run_episode(model, quants, pl, H_e, L_e, h_budget, decide, rho, eta,
                prior_log_odds=0.0):
    x = 0
    om = prior_log_odds
    cost = 0.0
    nt = 0
    while True:
        dec, _d = decide(pl, x, om, h_budget - cost, rho, eta)
        if dec[0] == "STOP":
            break
        i, kind, r2 = dec[1], dec[2], dec[3]
        zi = (x // pl.powers[i]) % BASE_B
        r_cur, _ = z_decode_b(zi)
        m2 = int(quants[i].cell_index(r2, float(L_e[i])))
        z2 = z_code_b(r2, m2)
        x += (z2 - zi) * pl.powers[i]
        om += pl._llr_i[i][z2] - pl._llr_i[i][zi]
        cost += BH + (r2 - r_cur)
        nt += 1
        if cost > h_budget + 1e-9:
            raise AssertionError(f"budget violation: cost={cost} > H={h_budget}")
    return cost, cost - BH * nt, nt, 1.0 if om > eta else 0.0


def run_sets(model, quants, pl, H_set, L_set, h_budget, decide, rho, eta):
    n = len(H_set)
    B = np.empty(n)
    BP = np.empty(n)
    NT = np.empty(n)
    DEC = np.empty(n)
    for e in range(n):
        b, bp, nt, dec = run_episode(model, quants, pl, int(H_set[e]),
                                     L_set[e], h_budget, decide, rho, eta)
        B[e], BP[e], NT[e], DEC[e] = b, bp, nt, dec
    return B, BP, NT, DEC


def exact_policy_cost(pl, decide, rho, eta, x_int, h, memo=None):
    if memo is None:
        memo = {}
    key = (int(x_int), int(h))
    hit = memo.get(key)
    if hit is not None:
        return hit
    om = pl.omega(x_int)
    dec, _d = decide(pl, x_int, om, h, rho, eta)
    if dec[0] == "STOP":
        memo[key] = 0.0
        return 0.0
    i, kind, r2 = dec[1], dec[2], dec[3]
    zi = (x_int // pl.powers[i]) % BASE_B
    r_cur, _ = z_decode_b(zi)
    c = BH + (r2 - r_cur)
    tpl = next(a for a in pl._tpl[i][zi] if a[0] == r2)
    lp = _logp(om)
    lq = _logq(om)
    E = 0.0
    for (m2, lp0c, lp1c) in tpl[3]:
        w = _w(lp, lq, lp1c, lp0c)
        z2 = z_code_b(r2, m2)
        x2 = x_int + (z2 - zi) * pl.powers[i]
        E += w * (c + exact_policy_cost(pl, decide, rho, eta, x2, h - c, memo))
    memo[key] = E
    return E


# ------------------------------------------------------- exact full-fusion ref
def full_fusion_ref(model, quants, level):
    """P_D,max^q(α)：**精确量化全融合**参考 —— 各 UAV 的 level-消息 LLR 在
    H0/H1 下独立 ⇒ Ω 的精确分布 = 4 个 per-UAV LLR 质量函数的卷积。
    返回 (pfa_fn, pmd_fn, om_grid, w0, w1)。"""
    state = {0.0: (1.0, 1.0)}
    for i in range(len(quants)):
        q = quants[i]
        lv = q.llr[level]
        w0 = np.exp(q.logP0[level])
        w1 = np.exp(q.logP1[level])
        nxt = {}
        for (v, (a, b)) in state.items():
            for k in range(2 ** level):
                vk = v + float(lv[k])
                wa, wb = a * float(w0[k]), b * float(w1[k])
                if vk in nxt:
                    pa, pb = nxt[vk]
                    nxt[vk] = (pa + wa, pb + wb)
                else:
                    nxt[vk] = (wa, wb)
        state = nxt
    om_grid = np.array(sorted(state.keys()))
    w0g = np.array([state[v][0] for v in om_grid])
    w1g = np.array([state[v][1] for v in om_grid])
    c0 = np.concatenate([[0.0], np.cumsum(w0g)])
    c1 = np.concatenate([[0.0], np.cumsum(w1g)])
    t0 = c0[-1]
    t1 = c1[-1]

    def pfa(thr):
        idx = int(np.searchsorted(om_grid, thr, side="right"))
        return 1.0 - c0[idx] / t0

    def pmd(thr):
        idx = int(np.searchsorted(om_grid, thr, side="right"))
        return c1[idx] / t1

    return pfa, pmd, om_grid, w0g, w1g


def pd_max_at_alpha(pfa_fn, pmd_fn, alpha, lo=-10.0, hi=10.0):
    lo, hi = float(lo), float(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pfa_fn(mid) > alpha:
            lo = mid
        else:
            hi = mid
    eta = hi
    return eta, 1.0 - pmd_fn(eta), pmd_fn(eta)


# ------------------------------------------------- 8-bit oracle infeasibility
def reachable_profile_count(N, levels, b0, H):
    """reachable z-state 数上界：profile (r_1..r_N) 满足 Σ_i (b0*[r_i>0] + r_i)
    <= H；每个 profile 的 message cell 组合 = Π_i 2^{r_i}。纯组合计数。"""

    def rec(acc, rem, li):
        nonlocal total
        if li == N:
            cells = 1
            for r in acc:
                cells *= 1 if r == 0 else 2 ** r
            total += cells
            return
        for r in levels:
            cost_add = 0 if r == 0 else (b0 + r)
            if cost_add <= rem:
                rec(acc + [r], rem - cost_add, li + 1)

    total = 0
    rec([], H, 0)
    return int(total)


# --------------------------------------------------------------------- stats
def qos_stats(kfa, n0, kmd, n1, alpha, beta_t):
    """QoS 分类（015 §5 口径）：FEASIBLE / INFEASIBLE / UNCERTAIN。
    **分母修正（C2 审计）**：P_FA 分母 = 实测 H0 数 n0、P_MD 分母 = 实测 H1
    数 n1（binomial 使实际数略异；用 2*n 当两者分母会把 n 虚增、U_MD 压小）。"""
    ufa, lfa = wilson_upper(kfa, n0), wilson_lower(kfa, n0)
    umd, lmd = wilson_upper(kmd, n1), wilson_lower(kmd, n1)
    if ufa <= alpha and umd <= beta_t:
        return "FEASIBLE", ufa, umd
    if lfa > alpha or lmd > beta_t:
        return "INFEASIBLE", ufa, umd
    return "UNCERTAIN", ufa, umd


def calibrate(method, rho_grid, eta_grid, H_cal, H_set, L_set, n_hyp,
              alpha, beta_t, model, quants, pl, verbose=True):
    """扫 (rho,eta) grid 校准：记录 QoS U95 与 E[B]，选 θ̂
    （FEASIBLE 中 min E[B]；无 FEASIBLE 则 U_FA<=α 中 min U_MD；否则
    min U_FA —— 如实记录选择理由；α/β_t 按口径传入）。"""
    rows = []
    best = None
    for rho in rho_grid:
        for eta in eta_grid:
            Bs = []
            n0 = n1 = 0
            kfa = kmd = 0
            for hh in (0, 1):
                idx = np.where(H_set == hh)[0][:n_hyp]
                B, _bp, _nt, DEC = run_sets(model, quants, pl, H_set[idx],
                                            L_set[idx], H_cal, method,
                                            rho, eta)
                Bs.append(B)
                if hh == 0:
                    n0 = len(idx)
                    kfa = int(np.sum(DEC == 1.0))
                else:
                    n1 = len(idx)
                    kmd = int(np.sum(DEC == 0.0))
            B_all = np.concatenate(Bs)
            cls, ufa, umd = qos_stats(kfa, n0, kmd, n1, alpha, beta_t)
            eb = float(B_all.mean())
            rows.append({"rho": rho, "eta": eta, "cls": cls, "ufa": ufa,
                         "umd": umd, "eb": eb})
            if verbose:
                print(f"    [cal] {method.__name__} (rho={rho},eta={eta}) "
                      f"cls={cls} ufa={ufa:.4f} umd={umd:.4f} E[B]={eb:.3f}")
    feasible = [r for r in rows if r["cls"] == "FEASIBLE"]
    if feasible:
        best = min(feasible, key=lambda r: r["eb"])
        reason = "FEASIBLE 中 min E[B]"
    else:
        ufa_ok = [r for r in rows if r["ufa"] <= alpha]
        if ufa_ok:
            best = min(ufa_ok, key=lambda r: r["umd"])
            reason = "无 FEASIBLE：U_FA<=α 中 min U_MD"
        else:
            best = min(rows, key=lambda r: r["ufa"])
            reason = ("无 FEASIBLE 且无 U_FA<=α：min U_FA（注意：可能选中退化 "
                      "stop-at-root 的 θ —— 该口径下网格无可行点，如实报告）")
    return best, rows, reason


def run(method, pl, quants, model, H_set, L_set, theta, H_budget,
        alpha, beta_t, n_test):
    Bs, BPs, NTs, DECs = [], [], [], []
    kfa = kmd = 0
    n0 = n1 = 0
    for hh in (0, 1):
        idx = np.where(H_set == hh)[0][:n_test]
        B, BP, NT, DEC = run_sets(model, quants, pl, H_set[idx], L_set[idx],
                                  H_budget, method, theta[0], theta[1])
        Bs.append(B)
        BPs.append(BP)
        NTs.append(NT)
        DECs.append(DEC)
        if hh == 0:
            n0 = len(idx)
            kfa = int(np.sum(DEC == 1.0))
        else:
            n1 = len(idx)
            kmd = int(np.sum(DEC == 0.0))
    cls, ufa, umd = qos_stats(kfa, n0, kmd, n1, alpha, beta_t)
    return {"cls": cls, "ufa": ufa, "umd": umd,
            "eb": float(np.concatenate(Bs).mean()),
            "eb0": float(Bs[0].mean()), "eb1": float(Bs[1].mean()),
            "ep": float(np.concatenate(BPs).mean()),
            "ent": float(np.concatenate(NTs).mean()),
            "kfa": kfa, "kmd": kmd, "n0": n0, "n1": n1}


def paired_diff(a, b, methods, model, quants, pl, thetamap, H_set, L_set,
                H_budget, n_test):
    fn_a = next(fn for (nm, fn, _pl, _q, _g) in methods if nm == a)
    fn_b = next(fn for (nm, fn, _pl, _q, _g) in methods if nm == b)
    D = []
    for hh in (0, 1):
        idx = np.where(H_set == hh)[0][:n_test]
        for e in idx:
            Ba, _x, _y, _ = run_episode(model, quants, pl, int(H_set[e]),
                                        L_set[e], H_budget, fn_a,
                                        thetamap[a][0], thetamap[a][1])
            Bb, _x, _y, _ = run_episode(model, quants, pl, int(H_set[e]),
                                        L_set[e], H_budget, fn_b,
                                        thetamap[b][0], thetamap[b][1])
            D.append(Ba - Bb)
    return np.asarray(D, dtype=np.float64)


# ============================================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE：N_CAL=60/hyp、N_TEST=120/hyp、ρ/η 子网格")
    args = ap.parse_args()
    SMOKE = args.smoke
    if SMOKE:
        rho_grid = (128, 512)
        eta_grid = (0.8, 1.2, 2.0)
        n_cal = 60
        n_test = 120
        out_dir = SMOKE_OUT_DIR
        tag = "SMOKE"
    else:
        rho_grid = RHO_GRID
        eta_grid = ETA_GRID
        n_cal = FULL_N_CAL
        n_test = FULL_N_TEST
        out_dir = OUT_DIR
        tag = "FULL"

    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"# MVS-C C2 — phase-guided conditional-refinement policy + exact "
        f"budgeted-CMDP oracle（{tag}）")
    out("")
    out("> 依据 001 §二十六.2-§26.3 / §九-§十四 / §二十七 Gate D；C0 语义封板后"
        "的第一个**主算法**模块。冻结参数：N=4、GAMMA4=[-1,1,3,5] dB、"
        f"levels (1,2,4,8)、b_{{0,i}}=16（κ=1 homogeneous special case）、"
        f"QoS matched detection α={ALPHA}、ε_D={EPS_D}、H∈{H_BUDGETS}、"
        f"ρ∈{rho_grid}、η∈{eta_grid}、N_CAL={n_cal}/hyp @ H=96、"
        f"N_TEST={n_test}/hyp。")

    model4 = GaussianDetectorModel(GAMMA4, prior=(0.5, 0.5))
    quants8 = [NestedQuantizer(i, model4, r_max=8, levels=LEVELS8)
               for i in range(4)]
    quants4 = [NestedQuantizer(i, model4, r_max=4, levels=LEVELS4)
               for i in range(4)]
    pl8 = SparsePlanner(quants8, 1.0, 1.0, b_h=BH, cross_level=True,
                        levels=LEVELS8, direct_only=False, delta_c=1.0)
    pl4 = SparsePlanner(quants4, 1.0, 1.0, b_h=BH, cross_level=True,
                        levels=LEVELS4, direct_only=False, delta_c=1.0)

    # ----------------------------------------------------- 0. exact ref
    out("")
    out("## 0. 精确量化全融合参考（C0 语义：P_D,max 而非连续近似）")
    out("")
    pfa4, pmd4, _g4, _w4, _v4 = full_fusion_ref(model4, quants4, 4)
    eta4, pd4, pmd4v = pd_max_at_alpha(pfa4, pmd4, ALPHA)
    cont_pd = model4.raw_fusion_pd(ALPHA)
    BETA8 = 1.0 - cont_pd + EPS_D
    BETA4 = 1.0 - pd4 + EPS_D
    out(f"- 4-bit 全融合（精确卷积）：P_D,max^q(0.05) = {pd4:.4f}（η*="
        f"{eta4:.4f}）、P_MD={pmd4v:.4f} → matched 目标 P_MD ≤ {BETA4:.4f}；"
        f"support=16^4={16 ** 4}（trivial）")
    out(f"- 8-bit：精确卷积不可行（distinct Ω 上界 256^4≈{256 ** 4}，与 "
        f"exact DP 同源爆炸）→ C0 参考 = 连续 full-fusion P_D,raw={cont_pd:.4f} "
        f"→ matched 目标 P_MD ≤ {BETA8:.4f}；未覆盖的量化损失 ~1e-3 量级"
        f"（4-bit 精确 0.8482 级，见上行）——报告为参考面 limitation。")

    # ------------------------------------------------------ 1. theory gates
    out("")
    out("## 1. 理论 Gate（001 §九-§十四，reachable 状态上验证）")
    out("")
    th = (512, 1.2)
    H_c, L_c = sample_set(80, SEED_CAL + 7, model4)
    seen = set()
    for e in range(80):
        x, om = 0, 0.0
        cost = 0.0
        for _ in range(30):
            seen.add((x, int(96 - cost)))
            dec, _d = phase_decision(pl8, x, om, 96 - cost, th[0], th[1])
            if dec[0] == "STOP":
                break
            i, kind, r2 = dec[1], dec[2], dec[3]
            zi = (x // pl8.powers[i]) % BASE_B
            r_cur, _ = z_decode_b(zi)
            m2 = int(quants8[i].cell_index(r2, float(L_c[e][i])))
            z2 = z_code_b(r2, m2)
            x += (z2 - zi) * pl8.powers[i]
            om += pl8._llr_i[i][z2] - pl8._llr_i[i][zi]
            cost += BH + (r2 - r_cur)
    n_t1 = n_t3 = 0
    t1_bad = t3_bad = 0
    t4_max_full = t4_max_eval = 0
    t4_sum_full = t4_sum_eval = 0
    t4_n = 0
    for (x, h) in seen:
        if h <= 0:
            continue
        om = pl8.omega(x)
        for i in range(4):
            sup = phase_support(pl8, x, om, i, th[0], th[1])
            if sup is None:
                continue
            n_t1 += 1
            dev1 = abs(sup["Q_prog"] - sup["Q_dir"] - sup["g0"]
                       - (sup["E_R_sum"] - sup["E_dir"]))
            if dev1 > 1e-8:
                t1_bad += 1
            g0 = sup["g0"]
            if g0 >= 0.0 and sup["Q_prog"] + 1e-9 < sup["Q_dir"]:
                t3_bad += 1
            if g0 < 0.0 and sup["Q_dir"] + 1e-9 < sup["Q_prog"]:
                t3_bad += 1
            n_t3 += 1
        dec, diag = phase_decision(pl8, x, om, h, th[0], th[1])
        t4_n += 1
        t4_max_full = max(t4_max_full, diag["full_actions"])
        t4_max_eval = max(t4_max_eval, diag["kept_actions"] + diag["pruned"])
        t4_sum_full += diag["full_actions"]
        t4_sum_eval += diag["kept_actions"] + diag["pruned"]
    out(f"- **T1（支撑恒等式，013 §1）**：{n_t1} 个 UAV 支撑检查，"
        f"Q_prog−Q_dir−g0−(E_R_sum−E_dir) 偏差 >1e-8 的个数 {t1_bad} "
        f"→ {mp(t1_bad == 0)}")
    out(f"- **T3（pruning 自洽，001 §十四）**：{n_t3} 检查，符号矛盾 {t3_bad} "
        f"→ {mp(t3_bad == 0)}（g≥0 ⇒ Q_prog≥Q_dir）")
    out(f"- **T4（复杂度，001 §十四 O(2N) 非 O(N|R|)）**：{t4_n} 决策，"
        f"full-FG 每决策 max {t4_max_full}（≤N|R|=16: {mp(t4_max_full <= 16)}）、"
        f"Phase-FG 评估动作 max {t4_max_eval}（≤2N=8: {mp(t4_max_eval <= 8)}）、"
        f"总量比 {t4_sum_eval / max(1, t4_sum_full):.3f}")

    out("")
    out("- **T2（cond-refinement sandwich，001 §十/§十一）**：")

    def v1_self(pl, x, om, i, rho, eta):
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        best = r_dual(om, rho, eta)
        for r2 in pl.levels:
            if r2 <= r:
                continue
            c = BH + (r2 - r)
            q = c + sum(
                _w(_logp(om), _logq(om), lp1c, lp0c) *
                r_dual(om + pl._llr_i[i][z_code_b(r2, m2)]
                       - pl._llr_i[i][zi], rho, eta)
                for (m2, lp0c, lp1c) in
                next(a for a in pl._tpl[i][zi] if a[0] == r2)[3])
            best = min(best, q)
        return best

    def v1_global(pl, x, om, rho, eta):
        best = r_dual(om, rho, eta)
        for j in range(pl.N):
            zj = (x // pl.powers[j]) % BASE_B
            r, _ = z_decode_b(zj)
            if r >= pl.r_max:
                continue
            for r2 in pl.levels:
                if r2 <= r:
                    continue
                c = BH + (r2 - r)
                q = c + sum(
                    _w(_logp(om), _logq(om), lp1c, lp0c) *
                    r_dual(om + pl._llr_i[j][z_code_b(r2, m2)]
                           - pl._llr_i[j][zj], rho, eta)
                    for (m2, lp0c, lp1c) in
                    next(a for a in pl._tpl[j][zj] if a[0] == r2)[3])
                best = min(best, q)
        return best

    t2_checks = t2_bad = 0
    for (x, h) in list(seen)[:120]:
        if h <= 0:
            continue
        om = pl8.omega(x)
        for i in range(4):
            sup = phase_support(pl8, x, om, i, th[0], th[1])
            if sup is None:
                continue
            c1 = sup["c1"]
            q1v = c1 + sum(br[0] * br[2] for br in sup["branches"]) / sup["wsum"]
            q_prog = sup["Q_prog"]
            s2 = 0.0
            g2 = 0.0
            for br in sup["branches"]:
                x1 = br[1]
                om1 = pl8.omega(x1)
                s2 += br[0] * v1_self(pl8, x1, om1, i, th[0], th[1])
                g2 += br[0] * v1_global(pl8, x1, om1, th[0], th[1])
            q_self2 = c1 + s2 / sup["wsum"]
            q_glob2 = c1 + g2 / sup["wsum"]
            t2_checks += 1
            if not (q_glob2 <= q_self2 + 1e-8 and q_self2 <= q_prog + 1e-8
                    and q_prog <= q1v + 1e-8):
                t2_bad += 1
    out(f"    T2 检查 {t2_checks} 个（Q{{global-2}} ≤ Q{{self-2}} ≤ Q_prog ≤ "
        f"Q{{(1)}}），矛盾 {t2_bad} → {mp(t2_bad == 0)}")

    # ------------------------------------------------- 2. oracle feasibility
    out("")
    out("## 2. exact budgeted CMDP oracle 的粒度可行性（001 §二十一）")
    out("")
    est8_96 = reachable_profile_count(4, LEVELS8, 16, 96)
    est8_48 = reachable_profile_count(4, LEVELS8, 16, 48)
    est4_96 = reachable_profile_count(4, LEVELS4, 16, 96)
    est4_48 = reachable_profile_count(4, LEVELS4, 16, 48)
    out(f"- reachable z-state 数估计（profile 计数，不含 budget-layer 重复）："
        f"8-bit/H=96: {est8_96}、8-bit/H=48: {est8_48}、"
        f"4-bit/H=96: {est4_96}、4-bit/H=48: {est4_48}")
    out(f"- **结论**：8-bit 粒度 exact backward 不可行（H=96 已 ≥ 1e9）→ "
        f"Gate D oracle 冻结在 **4-bit {LEVELS4} 粒度**（{est4_96}，MVS-A "
        f"ExactDP 规模）；8-bit 与 Direct8/Myopic 的机制比较走 MC（§3）。")

    # ---------------------------------------------------------- 3. mechanism
    out("")
    out("## 3. 机制比较（双口径：matched 主口径 / legacy mechanism 参考口径）")
    out("")
    out("> 017 §三：G2 注册的 (P_FA≤0.12, P_MD≤0.40) 是 mechanism-dialect；001 §三 的"
        "**matched detection**（α=0.05、P_MD≤1−P_D,max+ε_D）是 paper 主口径。"
        "C2 两个口径都跑：matched 是裁判口径（统计不可认证时如实报 UNRESOLVED，"
        "见下），legacy 供 granularity-vs-D8/Myopic 的机制方向性对照（与 G2 "
        "−5.33 同口径）。")
    out("> **matched 统计不可认证诊断（C2 审计）**：N=4 弱感知下 "
        f"P_D,max^q(0.05)={pd4:.4f}、目标 P_D≥{pd4 - EPS_D:.4f}（β="
        f"{BETA4:.4f}）；量化全融合实际可达 ~0.845-0.848 ⇒ P_D,max−P_D 余量 "
        f"~0.005-0.008，而 Wilson 95% 半宽 n={n_cal} → ±0.036、n={n_test} → "
        "±0.025 ⇒ **ε_D=0.01 的认证余量被 CI 消耗殆尽**：matched FEASIBLE 在"
        "该系统规模下**统计不可认证**（任何 θ 只会 UNCERTAIN/INFEASIBLE —— "
        "不是搜索失败）。机制口径（β=0.40）余量 ~0.20 ≫ CI，是 C2 的可认证"
        "比较面。")
    H_cal, L_cal = sample_set(2 * n_cal, SEED_CAL, model4)
    H_t96, L_t96 = sample_set(2 * n_test, SEED_TEST + 1000, model4)
    H_t48, L_t48 = sample_set(2 * n_test, SEED_TEST + 2000, model4)

    methods = [
        ("Phase-FG(8-bit)", phase_decision, pl8, quants8, "8"),
        ("Myopic-FG(8-bit)", myopic_decision, pl8, quants8, "8"),
        ("Direct8", direct_decision, pl8, quants8, "8"),
        ("Phase-FG(4-bit)", phase_decision, pl4, quants4, "4"),
    ]
    DIALECTS = [
        ("matched（001 §三）", 0.05, {"8": BETA8, "4": BETA4}),
        ("legacy mechanism（017 §四）", 0.12, {"8": 0.40, "4": 0.40}),
    ]
    all_dial = {}
    for (dname, alpha_d, betamap) in DIALECTS:
        out("")
        out(f"### 3.{DIALECTS.index((dname, alpha_d, betamap)) + 1} 口径："
            f"{dname}（α={alpha_d}、β 按粒度）")
        cal_res = {}
        for (nm, fn, pl_, qu_, gran) in methods:
            out(f"- 校准 {nm}：")
            best, rows, reason = calibrate(fn, rho_grid, eta_grid, 96,
                                           H_cal, L_cal, n_cal, alpha_d,
                                           betamap[gran], model4, qu_, pl_)
            cal_res[nm] = {"theta": (best["rho"], best["eta"]), "rows": rows,
                           "reason": reason, "best": best}
            out(f"    θ̂ = ({best['rho']},{best['eta']})  (E[B]={best['eb']:.3f}, "
                f"cls={best['cls']}, ufa={best['ufa']:.4f}, "
                f"umd={best['umd']:.4f})；理由：{reason}")
        out("")
        out("| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | "
            "E[B|H1] | E[payload] | E[N_tx] |  （@H=96）")
        out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        test_res = {}
        for (nm, fn, pl_, qu_, gran) in methods:
            thm = cal_res[nm]["theta"]
            r = run(fn, pl_, qu_, model4, H_t96, L_t96, thm, 96,
                    alpha_d, betamap[gran], n_test)
            test_res[(nm, 96)] = r
            out(f"| {nm} | ({thm[0]},{thm[1]}) | {r['cls']} | {r['ufa']:.4f} | "
                f"{r['umd']:.4f} | {r['eb']:.3f} | {r['eb0']:.3f} | "
                f"{r['eb1']:.3f} | {r['ep']:.3f} | {r['ent']:.3f} |")
        out("")
        out("| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |"
            "  （@H=48，同冻结 θ̂，operating-region boundary）")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for (nm, fn, pl_, qu_, gran) in methods:
            thm = cal_res[nm]["theta"]
            r = run(fn, pl_, qu_, model4, H_t48, L_t48, thm, 48,
                    alpha_d, betamap[gran], n_test)
            test_res[(nm, 48)] = r
            out(f"| {nm} | {r['cls']} | {r['ufa']:.4f} | {r['umd']:.4f} | "
                f"{r['eb']:.3f} | {r['ep']:.3f} | {r['ent']:.3f} |")
        out("")
        thmap = {nm: cal_res[nm]["theta"] for (nm, _f, _p, _q, _g) in methods}
        for (a, b) in [("Phase-FG(8-bit)", "Direct8"),
                       ("Phase-FG(8-bit)", "Myopic-FG(8-bit)")]:
            ra = test_res[(a, 96)]
            rb = test_res[(b, 96)]
            D = paired_diff(a, b, methods, model4, quants8, pl8, thmap, H_t96,
                            L_t96, 96, n_test)
            uh = hoeffding_upper(D, 96.0)
            both = ra["cls"] == "FEASIBLE" and rb["cls"] == "FEASIBLE"
            verdict = "比较有效（双方 FEASIBLE）" if both else "no-compare"
            out(f"- {a} − {b}（H=96，paired CRN，n={len(D)}）：E[D]={D.mean():.3f}"
                f"、Hoeffding U95={uh:.3f}"
                f"{'（< 0 ⇒ 该口径下前者更省，可认证）' if both and uh < 0 else ''}"
                f" | {verdict}；E[B] 各 {ra['eb']:.3f} vs {rb['eb']:.3f}")
        all_dial[dname] = {"cal": cal_res, "test": test_res}
    out("")
    out("### 3.4 双口径小结")
    out("")
    for dname in (d for (d, _, _) in DIALECTS):
        tr = all_dial[dname]["test"]
        cls96 = {nm: tr[(nm, 96)]["cls"] for (nm, _, _, _, _) in methods}
        eb96 = {nm: tr[(nm, 96)]["eb"] for (nm, _, _, _, _) in methods}
        out(f"- {dname}：@H=96 分类 {cls96}；E[B]（8-bit 方法）"
            f"Phase-FG={eb96['Phase-FG(8-bit)']:.3f}、Myopic="
            f"{eb96['Myopic-FG(8-bit)']:.3f}、Direct8={eb96['Direct8']:.3f}。")

    # ---------------------------------------------------------- 4. Gate D
    out("")
    out("## 4. Gate D — 求解器质量：Phase-FG vs exact budgeted CMDP*（4-bit）")
    out("")
    mt_cal = all_dial[DIALECTS[0][0]]["cal"]
    thm4 = mt_cal["Phase-FG(4-bit)"]["theta"]
    thm4_cls = mt_cal["Phase-FG(4-bit)"]["best"]["cls"]
    thetas_d = list(GATE_D_CORNERS) + [tuple(thm4)]
    out(f"> Gate D 的 θ 集合：固定 corners {GATE_D_CORNERS} + matched 口径下 "
        f"Phase-FG(4-bit) 的 θ̂={tuple(thm4)}（matched 分类 {thm4_cls}——"
        f"非 FEASIBLE 时注明：θ-fixed 比较仍有效，但 θ̂ 不是认证可行点，仅作"
        f"同 θ 求解器质量比对）。比较对象：CMDP* 的 E[B]（用 exact_policy_cost "
        f"+ oracle_decision，memo 已由 solve 预热）。")
    rows_d = []
    for theta in thetas_d:
        rho, eta = theta
        pl_o = SparsePlanner(quants4, rho * 0.5, rho * math.exp(eta) * 0.5,
                             b_h=BH, cross_level=True, levels=LEVELS4,
                             direct_only=False, delta_c=1.0)
        for H in (48, 96):
            t_s = time.time()
            _val_lag, _act = pl_o.solve(0, float(H))
            t_o = time.time() - t_s
            c_cmdp_b = exact_policy_cost(pl_o, oracle_decision, rho, eta, 0, H)
            c_phase = exact_policy_cost(pl4, phase_decision, rho, eta, 0, H)
            c_myo = exact_policy_cost(pl4, myopic_decision, rho, eta, 0, H)
            c_dir = exact_policy_cost(pl4, direct_decision, rho, eta, 0, H)
            rel_phase = (c_phase - c_cmdp_b) / max(c_cmdp_b, 1e-12)
            rel_myo = (c_myo - c_cmdp_b) / max(c_cmdp_b, 1e-12)
            rel_dir = (c_dir - c_cmdp_b) / max(c_cmdp_b, 1e-12)
            rows_d.append({"theta": theta, "H": H, "V_lag": _val_lag,
                           "C_cmdp_b": c_cmdp_b, "C_phase": c_phase,
                           "rel_phase": rel_phase, "rel_myo": rel_myo,
                           "rel_dir": rel_dir, "memo": len(pl_o.memo),
                           "exp": pl_o.n_expansions, "t_oracle": t_o})
            out(f"- θ={theta} H={H}：CMDP* E[B]={c_cmdp_b:.3f}（V_lag="
                f"{_val_lag:.1f} 仅作 solve 证书；memo={len(pl_o.memo)}、"
                f"expansions={pl_o.n_expansions}、{t_o:.1f}s）；"
                f"Phase-FG exact C={c_phase:.3f}（rel={rel_phase * 100:.2f}%）、"
                f"Myopic {c_myo:.3f}（{rel_myo * 100:.2f}%）、"
                f"Direct4 {c_dir:.3f}（{rel_dir * 100:.2f}%）")
    max_rel = max((r["rel_phase"] for r in rows_d), default=0.0)
    gate_d_pass = max_rel <= GATE_D_REL
    out(f"- **Gate D 判决**：Phase-FG 相对 gap 最大值 {max_rel * 100:.2f}% ≤ "
        f"预注册阈值 {GATE_D_REL * 100:.0f}% → {mp(gate_d_pass)}"
        f"（001 §二十七：D 良好则不再做 CPI）。注：H=96 行的 rel 为负是 "
        f"**dual trade**（Phase-FG 以更高终端风险换更少 bits —— 同一 θ 下 "
        f"Lagrangian 最优性由 V_lag 保证，Phase-FG 的 E[B]+E[R] ≥ V_lag "
        f"恒成立，负 rel 不是求解器缺陷），判决取 rel 的 max。")

    out("")
    out(f"总耗时: {time.time() - t0:.1f}s")
    out("")
    rp = os.path.join(out_dir, "MVS-C_C2_report.md")
    os.makedirs(out_dir, exist_ok=True)
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()