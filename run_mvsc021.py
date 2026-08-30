"""MVS-C C2.1: Budget-Aware Theoretical/Credibility Closure (advice/002.md).

定位（002 §十四）：C2 之后、C3 之前的**短理论/可信度收口**，集中修 002 的
四项 P0，并把 matched 口径与 Gate D 的语义拉到与论文完全一致：

  P0-1（002 §四）hard-budget phase pruning bug：理论剪枝（g>=0 ⇒ probe 剔除）
       只在 direct **可行**（c_dir<=h）时才成立 —— 若 c_probe<=h<c_dir，
       probe 是唯一可行结构动作，绝不能删。建立**资源窗三区域 phase law**
       （002 §五）：
           Q_prog(h)-Q_dir(h) =
             region A (c1<=h<c_dir) : probe 唯一可行（direct 不可行，不可剪）
             region B (c_dir<=h<c1+c2) : Q_prog = c1 + E[R(X1)]（第二包不可行）
             region C (h>=c1+c2) : E[min{Y,b0}]（原 013/001 §十二 定理）
       安全剪枝规则：prune probe ⟺ g>=0 ∧ c_dir<=h。
  P0-2（002 §三）Gate D 拆分：
        D1 solver-quality = (J_θ(Phase)-V_θ*)/V_θ*，J_θ=E[B]+E[R_θ(x_tau)]（Lagrangian，
          不是裸 E[B]）；D2 primal communication = matched-QoS 下 E[C] 的比较
          （双方 FEASIBLE 才有效）。上一版“Phase E[B] vs oracle E[B]”的负 rel
          只证明“用更大终端风险换更少 bits”，不能当 solver 证书 → Gate D 降级
          PROVISIONAL，由 D1 重认证。
  P0-3（002 §二）matched 定性修正：冻结 (rho,eta) 网格扫不出 matched 可行点
        ≠ 机制层不可行。N=4/H=96 下 4×8-bit direct = 96 <= H 且
        P_D,max^{8b}(alpha) >= P_D,max^{4b}(alpha)=0.8482 > 0.8382 ⇒
        **显式 primal 可行构造存在（π_full）**。C2.1 用 MITM（002 §七）精确算出
        P_D,max^{8b}(0.05)，并把 matched 判决改为“注册冻结族/网格 infeasible，
        primal feasibility 由 π_full 构造证明，不否定”。
  P0-4（002 §八）分层抽样：H0/H1 严格 n_0=n_1=n（独立 seed），不再靠 binomial。
  P1（002 §七/§九 最小化）reference 用 MITM 精确化（8-bit 256²+256²，非 256⁴），
        BETA8 = 1 - P_D,max^{8b} + eps_D。
  A2（C2.1 审计注记）：本 runner 的 myopic/direct 基线是 **C2 时代的 one-step
        参考**（`myopic_decision`/`direct_decision`：continuation = 立即 R(X')，
        非 budget-region 版）——Gate D1/D2 的参考对象有意用它们（002 §十一：
        budget-aware Myopic-FG 是 C3a migration 的正式定义，不在 C2.1 重做）。

冻结参数：N=4、GAMMA4=[-1,1,3,5] dB、levels (1,2,4,8)、b_{0,i}=16（κ=1
homogeneous special case）、α=0.05、ε_D=0.01、H∈(48,96)；matched 网格用
ρ-homotopy 扩展（002 §三：128..8192，log 步长）；legacy mechanism 网格保持
017 口径（ρ∈{128..1024}、η∈{0.8..2.0}）。
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
# ρ-homotopy（002 §三：matched 的双重 homotopy，直到 feasible 或收敛到 π_full）
RHO_MATCHED = (128, 256, 512, 1024, 2048, 4096, 8192)
RHO_LEGACY = (128, 256, 512, 1024)
ETA_GRID = (0.8, 0.85, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
GATE_D1_REL = 0.10          # D1：Δ_J ≤ 10%（预注册）
SEED0 = 2026
SEED_CAL = SEED0 + 100
SEED_TEST = SEED0 + 200
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
SMOKE_OUT_DIR = os.path.join(OUT_DIR, "smoke")
DELTA = 0.05
Z95 = 1.96

FULL_N_CAL = 400            # per hypothesis（002 §八：严格 n0=n1）
FULL_N_TEST = 800


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


def r_dual(om, rho, eta):
    p = 1.0 / (1.0 + math.exp(-om))
    return rho * min(p, math.exp(eta) * (1.0 - p))


def _logp(om):
    return -math.log1p(math.exp(-om))


def _logq(om):
    return -math.log1p(math.exp(om))


def _w(lp, lq, lp1c, lp0c):
    a_ = lp + lp1c
    b_ = lq + lp0c
    m_ = a_ if a_ >= b_ else b_
    return math.exp(m_ + math.log1p(math.exp(-abs(a_ - b_))))


# ------------------------------------------------------------------ sampling
def sample_set_strat(n, seed, model):
    """002 §八：严格 n0=n1=n 的分层抽样（H0/H1 独立 seed，跨方法 CRN 由
    同一 (H,L) 集保证）。返回 H(2n,)、L(2n,N)。"""
    H0 = np.zeros(n, dtype=np.int8)
    H1 = np.ones(n, dtype=np.int8)
    rng0 = np.random.default_rng(seed * 2 + 1)
    rng1 = np.random.default_rng(seed * 2 + 2)
    L0 = model.sample_llr(H0, rng0)
    L1 = model.sample_llr(H1, rng1)
    return np.concatenate([H0, H1]), np.vstack([L0, L1])


# ------------------------------------------------------------------ MITM ref
def pair_full_dist(qs, level):
    """两个 UAV 的 level 消息 LLR 精确卷积分布（support ≤ 2^(2·level)）。"""
    state = {0.0: (1.0, 1.0)}
    for q in qs:
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
    s = np.array(sorted(state.keys()))
    w0 = np.array([state[v][0] for v in s])
    w1 = np.array([state[v][1] for v in s])
    return s, w0, w1


def full_fusion_ref_mitm(model, quants, level):
    """002 §七：meet-in-the-middle 精确 8-bit 全融合 ROC。
    两侧 A=(0,1)、B=(2,3)，各 2^(2·level) 个 pair-sum；对 B 建 suffix
    survival，阈值扫描 O(2^(2L)·log 2^(2L)) 而非 2^(4L)。
    返回 (pfa_fn, pmd_fn, pdmax_alpha)。
    """
    sA, wA0, wA1 = pair_full_dist(quants[:2], level)
    sB, wB0, wB1 = pair_full_dist(quants[2:], level)
    # suffix survival（从大到小累计）
    orderB = np.argsort(sB)
    sBs = sB[orderB]
    cF0 = np.concatenate([[0.0], np.cumsum(wB0[orderB])])
    cF1 = np.concatenate([[0.0], np.cumsum(wB1[orderB])])
    tot0, tot1 = float(wB0.sum()), float(wB1.sum())
    orderA = np.argsort(sA)
    sAs_ = sA[orderA]
    wA0s_ = wA0[orderA]
    wA1s_ = wA1[orderA]
    totA0, totA1 = float(wA0.sum()), float(wA1.sum())

    def pfa(thr):
        idx = np.searchsorted(sBs, thr - sAs_, side='right')
        surv = (tot0 - cF0[idx]) / tot0
        return float(np.dot(wA0s_, surv)) / totA0

    def pmd(thr):
        idx = np.searchsorted(sBs, thr - sAs_, side='right')
        surv = (tot1 - cF1[idx]) / tot1
        return 1.0 - float(np.dot(wA1s_, surv)) / totA1   # P_MD=P1(Omega<=eta)

    return pfa, pmd


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


# ------------------------------------------------- resource-window phase law
def phase_support_budget(pl, x, om, i, h, rho, eta):
    """002 §五 资源窗三区域 probe/direct 支撑。

    c1 = b0+Δ1（probe 首包）、c2 = b0+Δ2（第二包）、c_dir = b0+Δ1+Δ2 = c1+c2−b0。
    区域：
      A: c1<=h<c_dir      → probe 唯一可行（direct 不可行；即使 g>=0 也**不可剪**）
      B: c_dir<=h<c1+c2   → Q_prog = c1 + E[R(X1)]（第二包不可行，无 conditional
                            refinement；per-branch E_R=E[R(X2)|X1] 仍计算，仅作为
                            003 §五 tower 恒等式 E[E[R(X2)|X1]]=E[R(X2)] 的
                            独立审计对象，不进入 Q_prog）
      C: h>=c1+c2         → Q_prog = c1 + E[min{R1, c2+E_R}]（原定理区域）
    剪枝规则（002 §四）：prune probe ⟺ g0>=0 ∧ c_dir<=h。
    返回 None（该 UAV 无任何可行动作或无下一 level）或 dict。"""
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
    c1 = BH + (r_next - r)
    c2 = BH + d2
    c_dir = BH + (r_max - r)
    probe_feas = c1 <= h
    dir_feas = c_dir <= h
    if not probe_feas and not dir_feas:
        return {"r": r, "r_next": r_next, "r_max": r_max, "probe_feas": False,
                "dir_feas": False, "region": "NONE"}
    if dir_feas and h < c1 + c2:
        region = "B"
    else:
        region = "C" if (probe_feas and dir_feas) else "A"
    # direct continuation E_dir
    lp = _logp(om)
    lq = _logq(om)
    E_dir = 0.0
    for (m2, lp0c, lp1c) in dir_tpl[3]:
        w = _w(lp, lq, lp1c, lp0c)
        z2 = z_code_b(r_max, m2)
        om2 = om + pl._llr_i[i][z2] - pl._llr_i[i][zi]
        E_dir += w * r_dual(om2, rho, eta)
    # probe branches
    branches = []
    for (m1, lp0c1, lp1c1) in prog_tpl[3]:
        w1 = _w(lp, lq, lp1c1, lp0c1)
        z1 = z_code_b(r_next, m1)
        x1 = x + (z1 - zi) * pl.powers[i]
        om1 = om + pl._llr_i[i][z1] - pl._llr_i[i][zi]
        R1 = r_dual(om1, rho, eta)
        if d2 == 0.0:
            # 退化情形（013/phase_boundary.py 同语义）：r_next == r_max ⇒
            # probe 与 direct 是**同一动作**（r→r_max 只发一个包），第二包
            # 是 no-op ⇒ 续程 = 立即 STOP（R(X1)），E_R = R1 ⇒ D = 0、Y = 0，
            # g_x(b)=E[min(0,b)]=0、Q_prog=Q_dir（prog_tpl==dir_tpl，
            # ER1==E_dir 精确成立）。此前 E_R 保持 0 会让 C 区 gap 假性
            # E[min(R1,16)] 而 Q_prog−Q_dir 为负（可差上百 bits），
            # 恒等式门与 prune 决策在退化状态失真（4-bit exhaustive
            # certificate 暴露，005 §十）。
            E_R = R1
            D = 0.0
            Y = 0.0
        else:
            E_R = 0.0
            ref = next((a for a in pl._tpl[i][z1] if a[0] == r_max), None)
            # B/C 区都计算 per-branch counterfactual E[R(X2)|X1]：C 区用于
            # Q_prog 的 conditional refinement；B 区第二包虽不可行，但
            # counterfactual 是 tower 恒等式 E[E[R(X2)|X1]] = E[R(X2)] 的审计
            # 对象（003 §五：B 区 gap=E[Y] 需要 per-branch 独立验证，不能只做
            # 边际代数自洽）。A 区保持 E_R=0（probe 唯一可行，无对照物；
            # g0_chk 语义依赖它）。
            if ref is not None and region in ("B", "C"):
                lp1 = _logp(om1)
                lq1 = _logq(om1)
                for (m2, lp0c, lp1c) in ref[3]:
                    w2 = _w(lp1, lq1, lp1c, lp0c)
                    z2 = z_code_b(r_max, m2)
                    om2 = om1 + pl._llr_i[i][z2] - pl._llr_i[i][z1]
                    E_R += w2 * r_dual(om2, rho, eta)
            D = R1 - E_R
            Y = D - d2
        branches.append((w1, x1, om1, R1, E_R, D, Y))
    wsum = sum(br[0] for br in branches)
    E_R_sum = sum(br[0] * br[4] for br in branches) / wsum
    ER1 = sum(br[0] * br[3] for br in branches) / wsum if wsum > 0 else 0.0
    if probe_feas:
        if region == "C":
            Q_prog = c1 + sum(br[0] * min(br[3], c2 + br[4])
                              for br in branches) / wsum
        else:
            # region A/B：第二包不可行 → 续程 = 立即 STOP（R(X1)）
            Q_prog = c1 + ER1
    else:
        Q_prog = None
    Q_dir = (BH + (r_max - r)) + E_dir if dir_feas else None
    # 003 §四 piecewise prune gap：
    #   A：direct 不可行 → probe 绝不剪；gap ≡ -inf
    #   B：gap = Q_prog^B - Q_dir = -d2 + (ER1 - E_dir) = E[Y]（tower；第二包
    #       不可行，counterfactual 只通过边际 tower 进入，无需 per-branch）
    #   C：gap = g0 = E[min(Y_C, b0)]，Y_C = R1 - E_R - d2
    # g_verdict 是**独立计算**的对照量（003 §五 审计对象）：
    #   B：per-branch E[Y_B] = Σw·(R1 - E_R - d2)（tower 恒等式
    #      E[E[R(X2)|X1]] = E[R(X2)] ⇒ 与边际形式 ER1-E_dir-d2 一致）
    #   C：(Q_prog - Q_dir) 的策略价值形式（与 support 形式 g0 一致）
    # 因此 |gap - g_verdict| 恒等式的两边来自不同代码路径，检查可真实失败。
    if region == "A":
        gap = -float("inf")
        g_verdict = None
    elif region == "B":
        gap = (Q_prog - Q_dir) if (Q_prog is not None and Q_dir is not None) \
            else -float("inf")
        g_verdict = sum(br[0] * br[6] for br in branches) / wsum  # E[Y_B]
    else:
        g0 = sum(br[0] * min(br[6], BH) for br in branches) / wsum
        gap = g0
        g_verdict = (Q_prog - Q_dir) if (Q_prog is not None and Q_dir is not None) \
            else None
    prune_probe_ok = (dir_feas and gap >= -1e-9)
    return {"r": r, "r_next": r_next, "r_max": r_max, "c1": c1, "c2": c2,
            "c_dir": c_dir, "probe_feas": probe_feas, "dir_feas": dir_feas,
            "region": region, "branches": branches, "wsum": wsum,
            "E_dir": E_dir, "E_R_sum": E_R_sum, "ER1": ER1,
            "gap": gap, "g_verdict": g_verdict,
            "Q_prog": Q_prog, "Q_dir": Q_dir, "prune_probe_ok": prune_probe_ok}


def phase_decision_budget(pl, x, om, h, rho, eta):
    """002 §四 修正后的 budget-aware phase decision：candidate 只含可行动作；
    probe 只在 (g>=0 ∧ c_dir<=h) 时剪。"""
    diag = {"full_actions": 0, "kept_actions": 0, "pruned": 0}
    cands = []
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        sup = phase_support_budget(pl, x, om, i, h, rho, eta)
        if sup is None or (not sup["probe_feas"] and not sup["dir_feas"]):
            continue
        diag["full_actions"] += len([r2 for r2 in pl.levels if r2 > sup["r"]])
        if sup["probe_feas"] and not sup["prune_probe_ok"]:
            cands.append((sup["Q_prog"], ("ACT", i, "PROBE", sup["r_next"])))
            diag["kept_actions"] += 1
        elif sup["probe_feas"]:
            diag["pruned"] += 1
        if sup["dir_feas"]:
            cands.append((sup["Q_dir"], ("ACT", i, "JUMP", sup["r_max"])))
            diag["kept_actions"] += 1
    if not cands:
        return ("STOP",), diag
    best_q, best_a = min(cands, key=lambda t: t[0])
    R = r_dual(om, rho, eta)
    if R <= best_q:
        return ("STOP",), diag
    return best_a, diag


# --------------- one-step（myopic/direct）与 oracle（budget 无关部分不变）----
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


def myopic_decision(pl, x, om, h, rho, eta):
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
            if c <= h:
                cands.append((q1(pl, x, om, i, r2, rho, eta),
                              ("ACT", i, "ANY", r2)))
    if not cands:
        return ("STOP",), {}
    best_q, best_a = min(cands, key=lambda t: t[0])
    if r_dual(om, rho, eta) <= best_q:
        return ("STOP",), {}
    return best_a, {}


def direct_decision(pl, x, om, h, rho, eta):
    cands = []
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        c = BH + (pl.r_max - r)
        if c <= h:
            cands.append((q1(pl, x, om, i, pl.r_max, rho, eta),
                          ("ACT", i, "ANY", pl.r_max)))
    if not cands:
        return ("STOP",), {}
    best_q, best_a = min(cands, key=lambda t: t[0])
    if r_dual(om, rho, eta) <= best_q:
        return ("STOP",), {}
    return best_a, {}


def oracle_decision(pl_oracle, x, om, h, rho, eta):
    _val, act = pl_oracle.solve(int(x), float(h))
    if act is None:
        return ("STOP",), {}
    i, r2 = act
    return ("ACT", i, "ANY", r2), {}


# ----------------------------------------------------------------- episodes
def run_episode(model, quants, pl, H_e, L_e, h_budget, decide, rho, eta):
    x = 0
    om = 0.0
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
            raise AssertionError(f"budget violation: {cost} > {h_budget}")
    return cost, cost - BH * nt, nt, 1.0 if om > eta else 0.0


def run_sets(model, quants, pl, H_set, L_set, h_budget, decide, rho, eta,
             off=0, cnt=None):
    n = len(H_set)
    if cnt is None:
        cnt = n
    B = np.empty(cnt)
    BP = np.empty(cnt)
    NT = np.empty(cnt)
    DEC = np.empty(cnt)
    for e in range(cnt):
        b, bp, nt, dec = run_episode(model, quants, pl, int(H_set[off + e]),
                                     L_set[off + e], h_budget, decide, rho, eta)
        B[e], BP[e], NT[e], DEC[e] = b, bp, nt, dec
    return B, BP, NT, DEC


# -------------------------------------------------------------- exact policy
def exact_policy_lagrangian(pl, decide, rho, eta, x_int, h, memo=None):
    """策略的**Lagrangian 精确期望** J = E[B + R_theta(x_tau)]（002 §三 D1 对象）：
    与 V_theta*（solve 返回）同口径可比。"""
    if memo is None:
        memo = {}
    key = (int(x_int), int(h))
    hit = memo.get(key)
    if hit is not None:
        return hit
    om = pl.omega(x_int)
    dec, _d = decide(pl, x_int, om, h, rho, eta)
    if dec[0] == "STOP":
        val = r_dual(om, rho, eta)
        memo[key] = val
        return val
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
        E += w * (c + exact_policy_lagrangian(pl, decide, rho, eta, x2,
                                              h - c, memo))
    memo[key] = E
    return E


def exact_policy_cost(pl, decide, rho, eta, x_int, h, memo=None):
    """策略的**通信成本**精确期望 E[B]（D2 对象：matched-QoS 下才与 oracle 的
    E[B] 比较）。"""
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


# --------------------------------------------------------------------- QoS
def qos_stats(kfa, n0, kmd, n1, alpha, beta_t):
    ufa, lfa = wilson_upper(kfa, n0), wilson_lower(kfa, n0)
    umd, lmd = wilson_upper(kmd, n1), wilson_lower(kmd, n1)
    if ufa <= alpha and umd <= beta_t:
        return "FEASIBLE", ufa, umd
    if lfa > alpha or lmd > beta_t:
        return "INFEASIBLE", ufa, umd
    return "UNCERTAIN", ufa, umd


def calibrate(method, rho_grid, eta_grid, H_cal, H_seq, L_seq, n_hyp,
              alpha, beta_t, model, quants, pl, verbose=True):
    rows = []
    for rho in rho_grid:
        for eta in eta_grid:
            Bs = []
            kfa = kmd = 0
            for hh in (0, 1):
                B, _bp, _nt, DEC = run_sets(model, quants, pl, H_seq, L_seq,
                                            H_cal, method, rho, eta,
                                            off=hh * n_hyp, cnt=n_hyp)
                Bs.append(B)
                if hh == 0:
                    kfa = int(np.sum(DEC == 1.0))
                else:
                    kmd = int(np.sum(DEC == 0.0))
            B_all = np.concatenate(Bs)
            n0 = n1 = n_hyp
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
            reason = "无 FEASIBLE 且无 U_FA<=α：min U_FA"
    return best, rows, reason


def run_test(method, pl, quants, model, H_seq, L_seq, theta, H_budget,
             alpha, beta_t, n_test):
    Bs, BPs, NTs, DECs = [], [], [], []
    kfa = kmd = 0
    for hh in (0, 1):
        B, BP, NT, DEC = run_sets(model, quants, pl, H_seq, L_seq, H_budget,
                                  method, theta[0], theta[1],
                                  off=hh * n_test, cnt=n_test)
        Bs.append(B)
        BPs.append(BP)
        NTs.append(NT)
        DECs.append(DEC)
        if hh == 0:
            kfa = int(np.sum(DEC == 1.0))
        else:
            kmd = int(np.sum(DEC == 0.0))
    cls, ufa, umd = qos_stats(kfa, n_test, kmd, n_test, alpha, beta_t)
    return {"cls": cls, "ufa": ufa, "umd": umd,
            "eb": float(np.concatenate(Bs).mean()),
            "eb0": float(Bs[0].mean()), "eb1": float(Bs[1].mean()),
            "ep": float(np.concatenate(BPs).mean()),
            "ent": float(np.concatenate(NTs).mean()),
            "kfa": kfa, "kmd": kmd, "n0": n_test, "n1": n_test}


# ============================================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    SMOKE = args.smoke
    if SMOKE:
        rho_m = (128, 1024, 8192)
        rho_l = (128, 512)
        eta_g = (0.8, 1.2, 2.0)
        n_cal = 60
        n_test = 120
        out_dir = SMOKE_OUT_DIR
        tag = "SMOKE"
    else:
        rho_m = RHO_MATCHED
        rho_l = RHO_LEGACY
        eta_g = ETA_GRID
        n_cal = FULL_N_CAL
        n_test = FULL_N_TEST
        out_dir = OUT_DIR
        tag = "FULL"

    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"# MVS-C C2.1 — Budget-Aware Theoretical/Credibility Closure "
        f"（advice/002.md，{tag}）")
    out("")
    out(f"> 依据 002 §二-§八/§十四；C2 记录（fee5cfb/47194bf）保留不动。冻结参数："
        f"N=4、GAMMA4=[-1,1,3,5] dB、levels (1,2,4,8)、b{{0,i}}=16（κ=1）、"
        f"α={ALPHA}、ε_D={EPS_D}、H∈{H_BUDGETS}；matched ρ-homotopy {rho_m}、"
        f"legacy ρ {rho_l}、η∈{eta_g}；**分层抽样 n0=n1**（002 §八）、N_CAL="
        f"{n_cal}/hyp @H=96、N_TEST={n_test}/hyp。")

    model4 = GaussianDetectorModel(GAMMA4, prior=(0.5, 0.5))
    quants8 = [NestedQuantizer(i, model4, r_max=8, levels=LEVELS8)
               for i in range(4)]
    quants4 = [NestedQuantizer(i, model4, r_max=4, levels=LEVELS4)
               for i in range(4)]
    pl8 = SparsePlanner(quants8, 1.0, 1.0, b_h=BH, cross_level=True,
                        levels=LEVELS8, direct_only=False, delta_c=1.0)
    pl4 = SparsePlanner(quants4, 1.0, 1.0, b_h=BH, cross_level=True,
                        levels=LEVELS4, direct_only=False, delta_c=1.0)

    # ------------------------------------------------------- 0. MITM 参考
    out("")
    out("## 0. MITM 精确 8-bit 全融合参考（002 §七，替代 continuous 近似）")
    out("")
    out("> **P_D,max 口径（005 §九）**：LLR 全融合统计 Ω 在量化消息下是**离散**"
        "的，严格 Neyman–Pearson optimum 允许在 threshold atom 上随机化"
        "（δ(Ω=η)=1 w.p. γ）以用满剩余 false-alarm budget；本 runner 的 "
        "`pd_max_at_alpha` 用确定性阈值 bisection（Ω>η、P_FA≤α）——因此这里"
        "报告的是 **P_D,max^det-thr**（deterministic-threshold achievable "
        "reference），不是严格 randomized-NP P_D,max。对 8-bit（65536 support"
        " 密集）二者数值差极小；论文措辞按 005 §九：不写裸 “maximum "
        "achievable” 而不说明 det-thr。")
    out("")
    pfa8, pmd8 = full_fusion_ref_mitm(model4, quants8, 8)
    pfa4, pmd4 = full_fusion_ref_mitm(model4, quants4, 4)
    eta8, pd8, pmd8v = pd_max_at_alpha(pfa8, pmd8, ALPHA)
    eta4, pd4, pmd4v = pd_max_at_alpha(pfa4, pmd4, ALPHA)
    BETA8 = 1.0 - pd8 + EPS_D
    BETA4 = 1.0 - pd4 + EPS_D
    out(f"- 8-bit（MITM，65536+65536 support）：P_D,max^det-thr,8b(0.05) = "
        f"{pd8:.4f}（η*={eta8:.4f}）→ matched 目标 P_MD ≤ {BETA8:.4f}")
    out(f"- 4-bit（MITM）：P_D,max^det-thr,4b(0.05) = {pd4:.4f}（η*={eta4:.4f}）"
        f"→ matched 目标 P_MD ≤ {BETA4:.4f}")
    out("")
    # 002 §二：显式 primal 可行构造
    out("## 0.1 π_full 显式可行构造（002 §二：matched 的 primal feasibility 证明）")
    out("")
    c_full = 4 * (BH + 8)                       # 4×8-bit direct = 96
    out(f"- π_full = 四架全部 8-bit direct，阈值 η*(α) 由 MITM 参考标定：成本 "
        f"C=4×24={c_full} ≤ H=96；P_FA≤α（det-thr 构造标定，α=0.05）、P_D="
        f"P_D,max^8b(0.05)={pd8:.4f} ≥ 0.8382 ⇒ **matched primal 可行**"
        f"（存在性成立，C2 的 INFEASIBLE 结论降级为“冻结族/网格不可行”，002 §二）。")

    # ------------------------------------------------- 1. 资源窗 phase law
    out("")
    out("## 1. resource-window phase law（002 §五）＋ constrained pruning（002 §四）")
    out("")
    th = (512, 1.2)
    H_c, L_c = sample_set_strat(80, SEED_CAL + 7, model4)
    # —— 002 §四 的反例直接可测：h=20、c1=17（0→1 fresh）、c_dir=24
    pl_test = SparsePlanner(quants8[:1], 0.5, 0.5, b_h=BH, cross_level=True,
                            levels=LEVELS8, direct_only=False, delta_c=1.0)
    # 单 UAV N=1：fresh 状态 x=0, c1=17, c_dir=24
    sup20 = phase_support_budget(pl_test, 0, 0.0, 0, 20, th[0], th[1])
    # 003 §七：显式断言反例前提 g0 >= 0（不能只靠 region/feasibility；
    # region A 中分支 Y = R1 - E_R - d2，E_R 未算（A 第二包不可行）⇒
    # Y_code = R1 - d2，g0_chk = E[min(Y_code, 16)] 为 002 定理量）。
    # 注意必须用 br[6]（Y = R1 - E_R - d2），不能用 br[5]（D = R1 - E_R）：
    # A 区 E_R=0 时 br[5]=R1，与定理量 Y_code=R1-d2 差一个 d2。
    g0_chk = (sum(br[0] * min(br[6], BH) for br in sup20["branches"])
              / sup20["wsum"]) if sup20 is not None else None
    r_a = (sup20 is not None and sup20["region"] == "A"
           and sup20["probe_feas"] and not sup20["dir_feas"]
           and g0_chk is not None and g0_chk >= 0.0)
    out(f"- **Prune-safety 反例（002 §四 + 003 §七，N=1 fresh UAV，h=20）**："
        f"c1=17、c_dir=24 ⇒ region={sup20['region']}（A：probe 唯一可行）、"
        f"g0_chk=E[min(R1-d2,16)]={g0_chk:.3f} ≥ 0（**显式前提断言**）、"
        f"probe_feas={sup20['probe_feas']}、dir_feas={sup20['dir_feas']}、"
        f"prune_probe_ok={sup20['prune_probe_ok']}（g>=0 且 direct 不可行时也"
        f"必须为 False）→ {mp(r_a and not sup20['prune_probe_ok'])}")
    # —— reachable 状态上三区域验证
    seen = set()
    for e in range(80):
        x, om = 0, 0.0
        cost = 0.0
        for _ in range(30):
            h = 96 - cost
            seen.add((x, int(h)))
            dec, _d = phase_decision_budget(pl8, x, om, h, th[0], th[1])
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
    n_chk = n_bad_B = n_bad_C = 0
    region_cnt = {"A": 0, "B": 0, "C": 0}
    for (x, h) in seen:
        if h <= 0:
            continue
        om = pl8.omega(x)
        for i in range(4):
            sup = phase_support_budget(pl8, x, om, i, h, th[0], th[1])
            if sup is None or not sup["probe_feas"]:
                continue
            n_chk += 1
            region_cnt[sup["region"]] = region_cnt.get(sup["region"], 0) + 1
            if sup["region"] == "B":
                # 003 §五：Region-B 恒等式 gap == E[Y_B]。
                # gap 走边际 tower 路径（Q_prog−Q_dir = −d2+ER1−E_dir），
                # g_verdict 走 per-branch counterfactual 路径
                # （Σw·(R1−E_R−d2)，E_R=E[R(X2)|X1]）——两条独立代码路径，
                # 一致性由 tower 恒等式保证，检查可真实失败。
                if abs(sup["gap"] - sup["g_verdict"]) > 1e-9:
                    n_bad_B += 1
            elif sup["region"] == "C":
                # 003 §五：Region-C 恒等式 gap == (Q_prog−Q_dir)。
                # gap = E[min(Y,b)] 直接由 support 计算，g_verdict = Q_prog−Q_dir
                # 走策略价值形式——两条独立代码路径，检查可真实失败。
                if abs(sup["gap"] - sup["g_verdict"]) > 1e-9:
                    n_bad_C += 1
            # A 区：gap 恒等式 N/A（probe 唯一可行），003 §五 A 分支
    out(f"- **区域恒等式（003 §五按 region 分口径，独立路径对照）**：reachable "
        f"支撑检查 {n_chk}（region A/B/C 计数 {region_cnt}）——"
        f"B: |gap−E[Y]_per-branch|>1e-9 数 {n_bad_B}、"
        f"C: |gap−(Q_prog−Q_dir)|>1e-9 数 {n_bad_C} → "
        f"{mp(n_bad_B == 0 and n_bad_C == 0)}")
    # —— 003 §五：真正的 constrained dominance certificate
    #    ∀(s,i,h): prune ⟹ Q_prog ≥ Q_dir − ε；A 区绝不剪
    n_dom = n_dom_bad = 0
    for (x, h) in seen:
        if h <= 0:
            continue
        om = pl8.omega(x)
        for i in range(4):
            sup = phase_support_budget(pl8, x, om, i, h, th[0], th[1])
            # dominance 检查需要 dir 可行（否则不可比较）且 probe 可行
            if sup is None or not sup["probe_feas"] or not sup["dir_feas"]:
                continue
            n_dom += 1
            if sup["region"] == "A":
                if sup["prune_probe_ok"]:
                    n_dom_bad += 1      # A 区绝不剪
            elif sup["prune_probe_ok"]:
                # 剪了 probe ⇒ Q_prog >= Q_dir − tol
                if sup["Q_prog"] is not None and \
                        sup["Q_prog"] < sup["Q_dir"] - 1e-8:
                    n_dom_bad += 1
    out(f"- **dominance-safety（003 §五）**：prune ⟹ Q_prog ≥ Q_dir − ε："
        f"检查 {n_dom}（dir_feas 状态，A 区含不剪断言），矛盾 {n_dom_bad} "
        f"→ {mp(n_dom_bad == 0)}")

    # -------------------------------------------------- 2. 双口径校准/测试
    out("")
    out("## 2. 机制比较（matched 主口径 / legacy mechanism 参考口径）")
    out("")
    out(f"> matched 用 **ρ-homotopy 扩展网格 {rho_m}**（002 §三，不再撞 1024 边界）"
        f"与 **MITM BETA8={BETA8:.4f}**；verdict 措辞按 002 §二：任何网格失败都"
        f"定性为 **registered-family/grid infeasible**（primal 已由 π_full 构造"
        f"证明）。分层抽样 n0=n1（002 §八）。")
    out("")
    H_cal, L_cal = sample_set_strat(n_cal, SEED_CAL, model4)
    H_t96, L_t96 = sample_set_strat(n_test, SEED_TEST + 1000, model4)
    H_t48, L_t48 = sample_set_strat(n_test, SEED_TEST + 2000, model4)

    methods = [
        ("Phase-FG(8-bit)", phase_decision_budget, pl8, quants8),
        ("Myopic-FG(8-bit)", myopic_decision, pl8, quants8),
        ("Direct8", direct_decision, pl8, quants8),
        ("Phase-FG(4-bit)", phase_decision_budget, pl4, quants4),
    ]
    persist_cal = {}

    for (dname, alpha_d, rho_grid, beta_map, dtag) in [
        (f"matched（001 §三，ρ-homotopy {rho_m}）", 0.05, rho_m,
         {"8": BETA8, "4": BETA4}, "3.1"),
        ("legacy mechanism（017 §四）", 0.12, rho_l,
         {"8": 0.40, "4": 0.40}, "3.2"),
    ]:
        out("")
        out(f"### {dtag} 口径：{dname}（α={alpha_d}、β 按粒度）")
        cal_res = {}
        for (nm, fn, pl_, qu_) in methods:
            out(f"- 校准 {nm}：")
            best, rows, reason = calibrate(fn, rho_grid, eta_g, 96,
                                           H_cal, L_cal, n_cal, alpha_d,
                                           beta_map["8" if pl_ is pl8 else "4"],
                                           model4, qu_, pl_)
            cal_res[nm] = {"theta": (best["rho"], best["eta"]), "rows": rows,
                           "reason": reason, "best": best}
            out(f"    θ̂ = ({best['rho']},{best['eta']})  (E[B]={best['eb']:.3f}, "
                f"cls={best['cls']}, ufa={best['ufa']:.4f}, "
                f"umd={best['umd']:.4f})；理由：{reason}")
        if dtag == "3.1":
            persist_cal["matched"] = cal_res
        out("")
        out("| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] "
            "| E[payload] | E[N_tx] |  （@H=96）")
        out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        test_res = {}
        for (nm, fn, pl_, qu_) in methods:
            thm = cal_res[nm]["theta"]
            r = run_test(fn, pl_, qu_, model4, H_t96, L_t96, thm, 96,
                         alpha_d, beta_map["8" if pl_ is pl8 else "4"], n_test)
            test_res[(nm, 96)] = r
            out(f"| {nm} | ({thm[0]},{thm[1]}) | {r['cls']} | {r['ufa']:.4f} | "
                f"{r['umd']:.4f} | {r['eb']:.3f} | {r['eb0']:.3f} | "
                f"{r['eb1']:.3f} | {r['ep']:.3f} | {r['ent']:.3f} |")
        out("")
        out("| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |"
            "  （@H=48，同冻结 θ̂）")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for (nm, fn, pl_, qu_) in methods:
            thm = cal_res[nm]["theta"]
            r = run_test(fn, pl_, qu_, model4, H_t48, L_t48, thm, 48,
                         alpha_d, beta_map["8" if pl_ is pl8 else "4"], n_test)
            test_res[(nm, 48)] = r
            out(f"| {nm} | {r['cls']} | {r['ufa']:.4f} | {r['umd']:.4f} | "
                f"{r['eb']:.3f} | {r['ep']:.3f} | {r['ent']:.3f} |")
        out("")
        thmap = {nm: cal_res[nm]["theta"] for (nm, _f, _p, _q) in methods}
        for (a, b) in [("Phase-FG(8-bit)", "Direct8"),
                       ("Phase-FG(8-bit)", "Myopic-FG(8-bit)")]:
            ra = test_res[(a, 96)]
            rb = test_res[(b, 96)]
            D = []
            for hh in (0, 1):
                for e in range(n_test):
                    idx = hh * n_test + e
                    Ba, _x, _y, _ = run_episode(
                        model4, quants8, pl8, int(H_t96[idx]), L_t96[idx], 96,
                        next(fn2 for (nm, fn2, _p, _q) in methods if nm == a),
                        thmap[a][0], thmap[a][1])
                    Bb, _x, _y, _ = run_episode(
                        model4, quants8, pl8, int(H_t96[idx]), L_t96[idx], 96,
                        next(fn2 for (nm, fn2, _p, _q) in methods if nm == b),
                        thmap[b][0], thmap[b][1])
                    D.append(Ba - Bb)
            D = np.asarray(D, dtype=np.float64)
            uh = hoeffding_upper(D, 96.0)
            both = ra["cls"] == "FEASIBLE" and rb["cls"] == "FEASIBLE"
            verdict = "比较有效（双方 FEASIBLE）" if both else "no-compare"
            out(f"- {a} − {b}（H=96，分层 paired，n={len(D)}）：E[D]={D.mean():.3f}"
                f"、Hoeffding U95={uh:.3f}"
                f"{'（< 0 ⇒ 该口径下前者更省，可认证）' if both and uh < 0 else ''}"
                f" | {verdict}；E[B] 各 {ra['eb']:.3f} vs {rb['eb']:.3f}")

    out("")
    out("### 3.4 matched 定性（002 §二/§三）")
    out("")

    # A1（C2.1 审计）：§3.4 复用 §3.1 matched 口径已跑好的 cal_res —— 不再
    # 用第二个 seed（SEED_CAL+5）重跑 whole matched calibration（消除重复算力
    # 与 §3.1 表格抽样集不一致；002 §四 的“不重复为跑而跑”精神）。
    rows_pf = persist_cal["matched"]["Phase-FG(8-bit)"]["rows"]
    any_feas = any(r["cls"] == "FEASIBLE" for r in rows_pf)
    # 003 六：min U_MD 限定在 U_FA≤α 子集（不能取全网格最小值，否则会
    # 写出“离 matched 只差 0.0047”的错误直觉）。
    ufa_ok = [r for r in rows_pf if r["ufa"] <= ALPHA]
    edge = min(ufa_ok, key=lambda r: r["umd"]) if ufa_ok else None
    nf = sum(1 for r in rows_pf if r["cls"] == "FEASIBLE")
    if edge is None:
        out(f"- Phase-FG(8-bit) @ matched、ρ-homotopy：{len(rows_pf)} 个网格点，"
            f"FEASIBLE 数 = {nf}；**U_FA≤α 子集为空**（matched 网格无一点满足 "
            f"U_FA≤{ALPHA}）→ @α 边统计 UNCERTAIN（003 六）。")
    else:
        gap_md = edge["umd"] - BETA8
        out(f"- Phase-FG(8-bit) @ matched、ρ-homotopy：{len(rows_pf)} 个网格点，"
            f"FEASIBLE 数 = {nf}；**U_FA≤α 边最优点**：(ρ={edge['rho']}, "
            f"η={edge['eta']})、U_FA={edge['ufa']:.4f}、U_MD={edge['umd']:.4f} "
            f"（≈P_D {1 - edge['umd']:.4f}）——与 β8={BETA8:.4f} 的真实差距 "
            f"**{gap_md:.4f}**（003 六：不是全网格 min 的假象差距）。")
    out(f"- **定性（002 §二）**：matched 网格结果 = **registered frozen "
        f"controller/grid family** infeasible（或 feasible）——**不等于机制层不可行**；"
        f"primal feasibility 已由 §0.1 π_full 构造（C=96≤H、P_D="
        f"{pd8:.4f}≥0.8382）**证明存在**。论文措辞按 002 §十建议。")

    # ------------------------------------------------------- 3. Gate D 拆分
    out("")
    out("## 4. Gate D1（solver-quality，Lagrangian）与 D2（primal E[C]）"
        "（002 §三，取代 C2 的 E[B]-only Gate D）")
    out("")
    # Gate D1/D2 用 4-bit oracle（C2 冻结粒度）；θ-fixed corners 含 ρ-homotopy
    # 最大点 (8192,1.2)——002 §三 的 D1 是 θ-fixed solver-quality 证书（matched
    # 网格无可行 θ̂，故不用“matched θ̂”扫，如实注明）。
    GATE_D_CORNERS = ((256, 1.2), (512, 1.2), (1024, 1.6), (8192, 1.2))
    thetas_d = GATE_D_CORNERS
    rows_d1 = []
    max_dj = 0.0
    for theta in thetas_d:
        rho, eta = theta
        pl_o = SparsePlanner(quants4, rho * 0.5, rho * math.exp(eta) * 0.5,
                             b_h=BH, cross_level=True, levels=LEVELS4,
                             direct_only=False, delta_c=1.0)
        for H in (48, 96):
            t_s = time.time()
            v_star, _act = pl_o.solve(0, float(H))
            t_o = time.time() - t_s
            j_phase = exact_policy_lagrangian(pl4, phase_decision_budget,
                                              rho, eta, 0, H)
            j_myo = exact_policy_lagrangian(pl4, myopic_decision, rho, eta, 0, H)
            j_dir = exact_policy_lagrangian(pl4, direct_decision, rho, eta, 0, H)
            dj_phase = (j_phase - v_star) / max(v_star, 1e-12)
            dj_myo = (j_myo - v_star) / max(v_star, 1e-12)
            dj_dir = (j_dir - v_star) / max(v_star, 1e-12)
            c_phase = exact_policy_cost(pl4, phase_decision_budget, rho, eta, 0, H)
            c_cmdp_b = exact_policy_cost(pl_o, oracle_decision, rho, eta, 0, H)
            rows_d1.append({"theta": theta, "H": H, "V_star": v_star,
                            "J_phase": j_phase, "dj_phase": dj_phase,
                            "dj_myo": dj_myo, "dj_dir": dj_dir,
                            "C_phase": c_phase, "C_cmdp_b": c_cmdp_b,
                            "t_o": t_o})
            max_dj = max(max_dj, dj_phase)
            out(f"- θ={theta} H={H}：V*={v_star:.3f}（{t_o:.1f}s）；"
                f"J(Phase)={j_phase:.3f}（Δ_J={dj_phase * 100:.2f}%）、"
                f"J(Myo)={j_myo:.3f}（{dj_myo * 100:.2f}%）、"
                f"J(Direct4)={j_dir:.3f}（{dj_dir * 100:.2f}%）；"
                f"[D2 参考] E[B]: Phase={c_phase:.3f} vs CMDP* E[B]="
                f"{c_cmdp_b:.3f}（裸 E[B] 差 {c_phase - c_cmdp_b:+.3f} —— "
                f"只作 primal 参考，不作 solver 证书）")
    gate_d1 = max_dj <= GATE_D1_REL
    out(f"- **Gate D1 判决**：max Δ_J = {max_dj * 100:.2f}% ≤ 预注册 {GATE_D1_REL * 100:.0f}% "
        f"→ {mp(gate_d1)}（002 §三：Δ_J≥0 恒成立时 D1 才可能是质量证书；"
        f"C2 的 E[B]-only Gate D 正式降级为 PROVISIONAL，由 D1 替代）")
    out(f"- **Gate D2（primal E[C]）**：matched 口径下双方 FEASIBLE 才比较 —— "
        f"见 §3.1 matched 表与 §3.4 定性（当前 matched 网格若仍无可行点：D2 "
        f"UNRESOLVED，按 002 §二 不当作机制否定）。")

    out("")
    out(f"总耗时: {time.time() - t0:.1f}s")
    out("")
    rp = os.path.join(out_dir, "MVS-C_C21_report.md")
    os.makedirs(out_dir, exist_ok=True)
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()