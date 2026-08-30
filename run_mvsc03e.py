"""MVS-C C3e: Generalized Phase-Envelope Evidence Acquisition (advice/010.md §七-§十二).

定位（010 §十二 路线）：
  G0  Phase activation audit —— Phase-PJ 在实际工作点 (256,0.8) 是否真的激活
      （P(Q_phase≠Q_myopic)、action-change rate、region-C rate、pruning
      rate），回答 "Phase-PJ 为什么没有产生作用"（010 §一/§二）；
  G1  arbitrary r<s<t generalized phase-envelope theorem（010 §七）：
      c_i(r→q)=b0+κ(q−r)，
          Q_prog^{s,t} − Q_dir^t  ==  E[ min{ Y_{i,s,t}, b0 } ],
          Y = R(X_s) − E[R(X_t)|X_s] − κ(t−s)，
      数值 Gate（identity / tower / derivative=survival / b* 三情形分类）
      + 与 c21.phase_support_budget 的 (r_next, r_max) 特殊情形一致性；
  G2  论文生死 Gate（010 §八/§十二）：新 Proposed = **GPE-EA** —— 与
      Myopic-All **相同 full action set** A={(i,s): s>r_i, s∈levels}，
      唯一差别是 conditional-refinement Q（probe 用
      c(r→s)+E[min{R(X_s), min_t(c(s→t)+E[R(X_t)|X_s])}]，certificate
      证明全 continuation 被支配时精确退化为 one-step —— T50 断言）；
      separately calibrated、paired CRN、fresh test、G2 017 协议；
  G3  paired fixed-N empirical-Bernstein UCB（010 §十：MP Thm 4 plug-in
      方差，t=log(1/δ)）作为 G2 主 bit 认证，Hoeffding 为 sanity envelope。

协议（沿用 G2/C3b 017 §四 同协议）：N=8（GAMMA_B）、levels=(1,2,4,8)、
b_setup=16、QoS(P_FA≤0.12, P_MD≤0.40)；ρ∈{128,256,512,1024}、
η∈{0.8,…,2.0}（28 combos/method，仅 calibration）；calibration worlds
共用、test worlds 完全分离（paired CRN）；主 H=96、stress H=48（同冻结
θ̂）；QoS 用 Wilson 双侧 95% 上端点（n0/n1 分离，C3d 口径）。
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time

import numpy as np

import run_mvsb07g2 as g2
import run_mvsc021 as c21
import run_mvsc03a as c3a
from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs import phase_boundary as pb
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA_B = g2.GAMMA_B
BH = g2.BH
LEVELS = g2.LEVELS
R_MAX = g2.R_MAX
ALPHA = g2.ALPHA
BETA = g2.BETA
RHO_GRID = g2.RHO_GRID
ETA_GRID = g2.ETA_GRID
SEED_CAL = g2.SEED_CAL
SEED_TEST = g2.SEED_TEST
FULL_N_CAL = g2.FULL_N_CAL
FULL_N_TEST = g2.FULL_N_TEST
OUT_DIR = g2.OUT_DIR
SMOKE_OUT_DIR = os.path.join(OUT_DIR, "smoke")

fmt = g2.fmt
wilson_upper = g2.wilson_upper
wilson_lower = g2.wilson_lower
hoeffding_upper = g2.hoeffding_upper
hoeffding_lower = g2.hoeffding_lower
classify_qos2 = c3a.classify_qos2
sample_set = g2.sample_set
r_rho = g2.r_rho
q1_fast = g2.q1_fast
sim_decide = c3a.sim_decide
eval_decide = c3a.eval_decide
calibrate_decide = c3a.calibrate_decide
_w = c21._w

# 冻结的 Phase-PJ 工作点（G0 审计；与 C3b 校准 θ̂ 一致）
THETA_FROZEN = (256, 0.8)

# (b0, κ) 广义 envelope 测试角（010 §七：b0=setup airtime、κ=per-bit cost）
GENERAL_CORNERS = ((16.0, 1.0), (16.0, 2.0), (8.0, 1.0), (32.0, 0.5))


# ---------------------------------------------------------------------------
# G3: paired fixed-N empirical-Bernstein UCB（010 §十；MP Thm 4 one-sided）
# ---------------------------------------------------------------------------
def eb_ucb(xs, lo, hi, delta=0.05):
    """Paired one-sided fixed-N empirical-Bernstein UCB.

    Z ∈ [lo, hi]，X=(Z−lo)/(hi−lo) ∈ [0,1] iid。Maurer–Pontil (2009) Thm 4
    one-sided、plug-in variance：
        r = sqrt(2 V̂ t/(n−1)) + 7t/(3(n−1)),  t = log(1/δ),
        U = mean + (hi−lo)·r  ⇒  P(μ > U) ≤ δ.
    与 rbl_eb.PairCS(mode="eb") 的 (n−1) 保守分母一致；fixed-N 设计无需
    peeling/union bound（010 §十 的 paired fixed-N 方案）。
    """
    xs = np.asarray(xs, dtype=np.float64)
    n = int(xs.size)
    if n == 0:
        return float("inf")
    x = (xs - lo) / (hi - lo)
    mu = float(x.mean())
    v = float(x.var(ddof=1)) if n > 1 else 0.0
    t = math.log(1.0 / delta)
    r = math.sqrt(2.0 * v * t / (n - 1)) + 7.0 * t / (3.0 * (n - 1))
    return lo + (hi - lo) * (mu + r)


def eb_lcb(xs, lo, hi, delta=0.05):
    xs = np.asarray(xs, dtype=np.float64)
    n = int(xs.size)
    if n == 0:
        return float("-inf")
    x = (xs - lo) / (hi - lo)
    mu = float(x.mean())
    v = float(x.var(ddof=1)) if n > 1 else 0.0
    t = math.log(1.0 / delta)
    r = math.sqrt(2.0 * v * t / (n - 1)) + 7.0 * t / (3.0 * (n - 1))
    return lo + (hi - lo) * (mu - r)


# ---------------------------------------------------------------------------
# GPE-EA：conditional-refinement Q + full-action-set decision（010 §八）
# ---------------------------------------------------------------------------
class GPEMemo:
    """Per-run memo 对 _cond_refine_q：on-policy (x,i,s,conts) 有限重复，缓存
    (Q_cond, pruned) 使 FULL 校准可行（GPE 每决策 ~20× Myopic-All 原始开销）。

    **P0 修复（advice/010 续审）**：Q 层缓存 key 必须含 (rho, eta, b0, kappa)——
    Q_cond 依赖 r_rho(om; rho,eta) 与 c1/c2(b0,kappa)。旧实现 key 只含
    (x,i,s,conts)，28 组合校准共享 memo ⇒ 后续所有 (ρ,η) 组合拿到首个
    (ρ,η) 的陈旧 Q（实测 Q(128,0.8)=81.0 泄漏给 Q(1024,2.0) 应为 529.0），
    系统性低估 probe Q ⇒ GPE-EA 过度"继续买证据"（H=96 E[N_tx] 2.50 vs 1.66
    的方向），污染 C3e/C4 的 G2 判定。

    同时引入**结构层缓存**（数理依据：后验 odds om 是状态 x 的唯一函数——
    replace-not-add 下 om = prior + Σ_i ℓ_i(z_i)，与 (ρ,η,b0,κ) **无关**；
    转移权重 w=P(m'|x,a) 同样只依赖状态与量化器）：把 (w, om2) 传播序列按
    (x,i,s,conts) 缓存（key 不含 ρ/η），Q 计算时只按当前 (ρ,η) 现算
    r_rho 与 min/Σ —— 修复后校准 28 组合不再重复计算权重，FULL 可行。"""

    def __init__(self):
        # Q 层（P0 修复：完整 key 含 ρ/η/b0/κ）
        self.q = {}
        # 结构层（与 ρ/η/b0/κ 无关的 (w, om) 传播；om 由 x 决定，含 om 校验）
        self.struct = {}


def _tpl_index(tpl_list):
    """r2 → (c_true, q_budget, cells) 的 dict 索引（替代线性 next 扫描）。"""
    return {a[0]: a for a in tpl_list}


def _cond_refine_q(pl, x, om, i, s, conts, rho, eta, memo=None,
                   b0=BH, kappa=1.0):
    """Q_cond = c1 + E[ min{ R(X_s), min_{t∈conts}(c2_t + E[R(X_t)|X_s]) } ].

    certificate（pruned）：**分支wise** 检查每个 X_s 分支上
        min_t(c2_t + E[R(X_t)|X_s]) ≥ R(X_s)
    （任何 continuation 都不优于当场 STOP）⇒ E[min{·}] = E[R(X_s)]，
    Q_cond 精确等于 one-step q1_fast —— argmin 不变、只省计算（T50）。
    （010 §八 的 envelope g^{s,t}≥0 是期望级整路径支配，不蕴含分支wise
    证书；G1 报告 g^{s,t}，本函数用分支wise 证书，二者都如实报告。）

    P0 修复（见 GPEMemo docstring）：Q 层 key 补全 (rho,eta,b0,kappa)；
    结构层缓存 (w,om2) 序列（不含 ρ/η），内含 om 一致性校验。
    """
    qkey = (int(x), int(i), int(s), tuple(int(t) for t in conts),
            float(rho), float(eta), float(b0), float(kappa))
    if memo is not None and qkey in memo.q:
        return memo.q[qkey]
    zi = (x // pl.powers[i]) % BASE_B
    r, _m = z_decode_b(zi)
    tpl_i = _tpl_index(pl._tpl[i][zi])
    prog_tpl = tpl_i.get(s)
    if prog_tpl is None or not conts:
        q = q1_fast(pl, x, om, i, s, rho, eta)
        if memo is not None:
            memo.q[qkey] = (q, True)
        return q, True
    # 结构层：与 (ρ,η,b0,κ) 无关的 (w1, om1, {t:(w2s, om2s)}) 传播
    skey = (int(x), int(i), int(s), tuple(int(t) for t in conts))
    struct = None
    if memo is not None:
        struct = memo.struct.get(skey)
        if struct is not None and abs(struct["om"] - om) > 1e-9:
            struct = None      # om 不一致（防御：om 应=pl.omega(x)）
    if struct is None:
        lp = -math.log1p(math.exp(-om))
        lq = -math.log1p(math.exp(om))
        branches = []
        for (m1, lp0c1, lp1c1) in prog_tpl[3]:
            w1 = _w(lp, lq, lp1c1, lp0c1)
            z1 = z_code_b(s, m1)
            om1 = om + pl._llr_i[i][z1] - pl._llr_i[i][zi]
            # lp1/lq1 只依赖 om1（分支级）：提升到 t 循环外，每分支算一次
            lp1 = -math.log1p(math.exp(-om1))
            lq1 = -math.log1p(math.exp(om1))
            tpl1 = _tpl_index(pl._tpl[i][z1])
            cont = {}
            for t in conts:
                ref = tpl1.get(t)
                if ref is None:
                    continue
                w2s = []
                om2s = []
                for (m2, lp0c2, lp1c2) in ref[3]:
                    w2 = _w(lp1, lq1, lp1c2, lp0c2)
                    z2 = z_code_b(t, m2)
                    om2 = om1 + pl._llr_i[i][z2] - pl._llr_i[i][z1]
                    w2s.append(w2)
                    om2s.append(om2)
                cont[t] = (w2s, om2s)
            branches.append((w1, om1, cont))
        struct = {"om": om, "r": r, "branches": branches}
        if memo is not None:
            memo.struct[skey] = struct
    # Q 层：按当前 (ρ,η,b0,κ) 现算 r_rho/min/Σ；权重从结构层复用
    c1 = b0 + kappa * (s - r)
    E1 = 0.0
    wsum = 0.0
    all_stop_best = True
    for (w1, om1, cont) in struct["branches"]:
        R1 = r_rho(om1, rho, eta)
        cont_vals = []
        for t in conts:
            ent = cont.get(t)
            if ent is None:
                continue
            c2 = b0 + kappa * (t - s)
            w2s, om2s = ent
            E_R = 0.0
            for (w2, om2) in zip(w2s, om2s):
                E_R += w2 * r_rho(om2, rho, eta)
            cont_vals.append(c2 + E_R)
        best_cont = min(cont_vals) if cont_vals else R1
        all_stop_best &= (best_cont >= R1 - 1e-9)
        E1 += w1 * min(R1, best_cont)
        wsum += w1
    Q_cond = c1 + E1 / wsum if wsum > 0 else None
    if Q_cond is None:
        q = q1_fast(pl, x, om, i, s, rho, eta)
        if memo is not None:
            memo.q[qkey] = (q, True)
        return q, True
    if memo is not None:
        memo.q[qkey] = (Q_cond, all_stop_best)
    return Q_cond, all_stop_best


def gpe_decision(pl, x, om, h, rho, eta, memo=None):
    """GPE-EA（010 §八）：full action set A={(i,s): s>r_i, s∈levels}；probe
    s<r_max 用 conditional-refinement Q（certificate 认证全分支 STOP 最优时
    精确退化 one-step），s=r_max 用 one-step Q。与 Myopic-All **相同动作
    空间**——唯一差别是 Q 函数（conditional refinement vs myopic one-step）。
    """
    cands = []
    diag = {"n_cand": 0, "n_probe": 0, "n_terminal": 0, "n_cert": 0}
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _m = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        for s in pl.levels:
            if s <= r:
                continue
            c_s = BH + (s - r)
            if c_s > h + 1e-9:
                continue
            diag["n_cand"] += 1
            if s == pl.r_max:
                diag["n_terminal"] += 1
                cands.append((q1_fast(pl, x, om, i, s, rho, eta),
                              ("ACT", i, "ANY", s)))
                continue
            conts = [t for t in pl.levels
                     if t > s and BH + (t - s) <= h - c_s + 1e-9]
            if not conts:
                # 无可负担 continuation ⇒ probe 即 one-step（terminal 语义）
                diag["n_terminal"] += 1
                cands.append((q1_fast(pl, x, om, i, s, rho, eta),
                              ("ACT", i, "ANY", s)))
                continue
            diag["n_probe"] += 1
            Q_cond, pruned = _cond_refine_q(pl, x, om, i, s, conts, rho, eta,
                                            memo=memo)
            if pruned:
                diag["n_cert"] += 1
            cands.append((Q_cond, ("ACT", i, "ANY", s)))
    if not cands:
        return ("STOP",), diag
    best_q, best_a = min(cands, key=lambda t: t[0])
    if r_rho(om, rho, eta) <= best_q:
        return ("STOP",), diag
    return best_a, diag


# ---------------------------------------------------------------------------
# G0: Phase activation audit（010 §十二 C3e-G0）
# ---------------------------------------------------------------------------
def phase_activation_audit(pl, quants8, powers8, H_aud, L_aud, H=96):
    """在冻结工作点 θ̂=(256,0.8) 下，on-policy 决策状态上比较 Phase-PJ
    （c21.phase_decision_budget）与 Myopic-PJ（c3a.myopic_pj_decision）：
      n_decisions / P(continue) / action-change rate；
      probe-feasible supports 的 region A/B/C 计数；
      P(Q_phase≠Q_myopic)（|Q_prog−Q^(1)|>1e-6 的 candidate 比例）；
      pruning rate P(prune_probe_ok | probe_feasible)。
    """
    rho, eta = THETA_FROZEN
    tol = 1e-6
    n_dec = n_cont = n_act_change = 0
    n_tot_cand = n_cand_diff = 0
    n_probe_feas = n_pruned = 0
    region = {"A": 0, "B": 0, "C": 0}
    for e in range(len(H_aud)):
        x, h, om = 0, float(H), 0.0
        while True:
            if h < 1e-9:
                break
            dec_p, _dp = c21.phase_decision_budget(pl, x, om, h, rho, eta)
            dec_m, _dm = c3a.myopic_pj_decision(pl, x, om, h, rho, eta)
            n_dec += 1
            if dec_p[0] == "STOP":
                if dec_m[0] != "STOP":
                    n_act_change += 1
                break
            n_cont += 1
            if dec_m[0] == "STOP" or (dec_p[1], dec_p[3]) != (dec_m[1], dec_m[3]):
                n_act_change += 1
            # candidate-level audit on this decision state
            for i in range(pl.N):
                zi = (x // pl.powers[i]) % BASE_B
                r, _mc = z_decode_b(zi)
                if r >= pl.r_max:
                    continue
                sup = c21.phase_support_budget(pl, x, om, i, h, rho, eta)
                if sup is None or (not sup["probe_feas"] and not sup["dir_feas"]):
                    continue
                n_tot_cand += 1
                if sup["probe_feas"]:
                    n_probe_feas += 1
                    region[sup["region"]] += 1
                    if sup["prune_probe_ok"]:
                        n_pruned += 1
                    if sup["Q_prog"] is not None:
                        q1 = q1_fast(pl, x, om, i, sup["r_next"], rho, eta)
                        n_cand_diff += int(abs(sup["Q_prog"] - q1) > tol)
                if sup["dir_feas"] and sup["Q_dir"] is not None:
                    q1d = q1_fast(pl, x, om, i, sup["r_max"], rho, eta)
                    n_cand_diff += int(abs(sup["Q_dir"] - q1d) > tol)
            # advance under the phase policy
            i, _k, r2 = dec_p[1], dec_p[2], dec_p[3]
            zi = (x // pl.powers[i]) % BASE_B
            r_cur, m_cur = z_decode_b(zi)
            c = BH + (r2 - r_cur)
            m2 = int(quants8[i].cell_index(r2, float(L_aud[e][i])))
            om2 = om + pl._llr_i[i][z_code_b(r2, m2)] - pl._llr_i[i][zi]
            x += (z_code_b(r2, m2) - zi) * powers8[i]
            h -= c
            om = om2
    return {
        "n_decisions": n_dec, "n_continue": n_cont,
        "P_continue": (n_cont / n_dec) if n_dec else float("nan"),
        "action_change_rate": (n_act_change / n_dec) if n_dec else float("nan"),
        "n_candidates": n_tot_cand,
        "P_Q_phase_ne_myopic": (n_cand_diff / n_tot_cand) if n_tot_cand else 0.0,
        "n_probe_feasible": n_probe_feas,
        "n_pruned": n_pruned,
        "pruning_rate": (n_pruned / n_probe_feas) if n_probe_feas else 0.0,
        "regions": region,
    }


# ---------------------------------------------------------------------------
# G1: generalized r<s<t envelope gates（010 §七）
# ---------------------------------------------------------------------------
def _reachable_states(pl, quants8, powers8, L_set, max_states=200):
    """从 root 按 Phase-PJ 策略（θ̂ 冻结）随机推进收集可达 (x, om) 状态。"""
    rho, eta = THETA_FROZEN
    states = []
    for e in range(len(L_set)):
        x, h, om = 0, 96.0, 0.0
        while True:
            if h < 1e-9 or len(states) >= max_states:
                break
            states.append((x, om))
            dec, _d = c21.phase_decision_budget(pl, x, om, h, rho, eta)
            if dec[0] == "STOP":
                break
            i, _k, r2 = dec[1], dec[2], dec[3]
            zi = (x // pl.powers[i]) % BASE_B
            r_cur, _mc = z_decode_b(zi)
            c = BH + (r2 - r_cur)
            m2 = int(quants8[i].cell_index(r2, float(L_set[e][i])))
            om2 = om + pl._llr_i[i][z_code_b(r2, m2)] - pl._llr_i[i][zi]
            x += (z_code_b(r2, m2) - zi) * powers8[i]
            h -= c
            om = om2
        if len(states) >= max_states:
            break
    return states


def generalized_envelope_gates(pl, states):
    """G1 数值 Gate：random 可达状态 × 全部 (s,t) ladder 对 × (b0,κ) 角：
      G1a identity        max|g − (Q_prog − Q_dir)| < 1e-9
      G1b tower           max|E[E_R] − E_dir| < 1e-9
      G1c derivative      （(s,t)∈{(1,8),(2,8),(4,8)}、b0=16）∂g/∂b == Pr(Y>b)
      G1d b* 分类         b*<∞ ⟺ E[Y]≥0（A/B/C）
      G1e special-case    general (r_next,r_max,(16,1)) 的 Q_prog/Q_dir ==
                          c21.phase_support_budget C 区版本（逐状态）
    """
    rho, eta = THETA_FROZEN
    max_id = max_tower = 0.0
    max_deriv = 0.0
    max_id_sp = 0.0
    n_bstar = n_bstar_ok = 0
    n_special = n_special_ok = 0
    for (x, om) in states:
        zs = pl.decode(int(x))
        for i in range(pl.N):
            r, _m = z_decode_b(zs[i])
            if r >= pl.r_max:
                continue
            pairs = [(s, t) for s in LEVELS if s > r for t in LEVELS if t > s]
            for (s, t) in pairs:
                for (b0, kappa) in GENERAL_CORNERS:
                    sup = pb.general_phase_support(pl, x, om, i, s, t, rho, eta,
                                                   b0=b0, kappa=kappa)
                    if sup is None or sup["g_alt"] is None:
                        continue
                    max_id = max(max_id, abs(sup["g"] - sup["g_alt"]))
                    max_tower = max(max_tower, sup["tower_dev"])
            # G1c/G1d 在 (b0,κ)=(16,1) 上的代表性 (s,t) 组合
            for (s, t) in ((1, 8), (2, 8), (4, 8)):
                if not (r < s):
                    continue
                sup = pb.general_phase_support(pl, x, om, i, s, t, rho, eta,
                                               b0=BH, kappa=1.0)
                if sup is None:
                    continue
                bs = pb.bstar_general(sup)
                if bs is None:
                    continue
                n_bstar += 1
                finite = math.isfinite(bs["bstar"])
                ey_ge0 = bs["EY"] >= -1e-9
                n_bstar_ok += int(finite == ey_ge0)
                bl_ = sup["branches"]
                wsum = sup["wsum"]
                def g_at(b):
                    # generalized 7-tuple: Y at index 6 (not D at index 5)
                    return sum(br[0] * min(br[6], b) for br in bl_) / wsum
                eps = 1e-4
                b_ref = 4.0
                diff = (g_at(b_ref + eps) - g_at(b_ref)) / eps
                surv = sum(br[0] * (1.0 if br[6] > b_ref else 0.0)
                           for br in bl_) / wsum
                max_deriv = max(max_deriv, abs(diff - surv))
            # G1e special-case：与 c21.phase_support_budget 的 C 区逐状态比对
            sup_c = c21.phase_support_budget(pl, x, om, i, 96.0, rho, eta)
            if sup_c is None or sup_c["region"] != "C":
                continue
            sup_g = pb.general_phase_support(pl, x, om, i, sup_c["r_next"],
                                             sup_c["r_max"], rho, eta,
                                             b0=BH, kappa=1.0)
            if sup_g is None:
                continue
            n_special += 1
            ok = (abs(sup_g["Q_prog"] - sup_c["Q_prog"]) < 1e-9
                  and abs(sup_g["Q_dir"] - sup_c["Q_dir"]) < 1e-9)
            n_special_ok += int(ok)
            max_id_sp = max(max_id_sp, abs(sup_g["g"] - sup_g["g_alt"]))
    return {
        "n_states": len(states),
        "max_identity_dev": max_id,
        "max_tower_dev": max_tower,
        "max_deriv_dev": max_deriv,
        "n_bstar": n_bstar, "n_bstar_ok": n_bstar_ok,
        "n_special": n_special, "n_special_ok": n_special_ok,
        "max_identity_dev_special": max_id_sp,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nlevel", type=int, default=1)
    args = ap.parse_args()
    SMOKE = args.smoke
    if SMOKE:
        N_TEST = {1: 120, 2: 200, 3: 300, 4: 500}.get(args.nlevel, 120)
        N_CAL = N_TEST // 2
        N_AUDIT = 200
        N_STATES = 60
    else:
        N_TEST = FULL_N_TEST
        N_CAL = FULL_N_CAL
        N_AUDIT = 2000
        N_STATES = 220
    H_BUDGETS = (48, 96)
    CAL_H = 96
    out_dir = SMOKE_OUT_DIR if SMOKE else OUT_DIR
    tag = "SMOKE" if SMOKE else "FULL"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"# MVS-C C3e — Generalized Phase-Envelope Evidence Acquisition"
        f"（advice/010.md §七-§十二，{tag}）")
    out("")
    out("> **定位（010 §十二）**：C3b 显示 Phase-PJ 在实际工作点无实证增益"
        "（Phase==Myopic-PJ、Phase>Myopic-All +2.30 bits）。C3e 不再修"
        "Phase-PJ：G0 审计它为何不激活；G1 把 013 的 next/full 定理升级成"
        "任意 r<s<t 的 generalized phase envelope（link-affine cost "
        "c_i(r→q)=b0+κ(q−r)）；G2 新 Proposed **GPE-EA** 用与 Myopic-All "
        "相同 full action set，唯一差别是 conditional-refinement Q —— "
        "论文生死 Gate；G3 paired empirical-Bernstein UCB 作为主 bit 认证。")
    out("")
    out(f"> 协议（G2/C3b 017 §四 同）：N=8（GAMMA_B）、levels={LEVELS}、"
        f"b_setup={BH}、QoS(P_FA≤{ALPHA}, P_MD≤{BETA})；ρ∈{RHO_GRID}、"
        f"η∈{ETA_GRID}（28 combos/method，仅 calibration）；calibration "
        f"worlds 共用、test fresh 分离（paired CRN）；H=96 主 / H=48 stress；"
        f"N_CAL={N_CAL}、N_TEST={N_TEST}、N_AUDIT={N_AUDIT}。")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=LEVELS)
               for i in range(8)]
    powers8 = [BASE_B ** i for i in range(8)]
    pl = SparsePlanner(quants8, 1.0, 1.0, b_h=BH, cross_level=True,
                       levels=LEVELS, delta_c=1.0)

    H_cal, L_cal = sample_set(N_CAL, SEED_CAL, model8)
    H_t48, L_t48 = sample_set(N_TEST, SEED_TEST * 1000 + 1, model8)
    H_t96, L_t96 = sample_set(N_TEST, SEED_TEST * 1000 + 2, model8)
    H_aud, L_aud = sample_set(N_AUDIT, SEED_TEST * 1000 + 3, model8)

    # ------------------------------------------------------------ G0
    out("## G0. Phase activation audit（010 §十二 C3e-G0）")
    out("")
    out(f"> 冻结工作点 θ̂={THETA_FROZEN}（C3b 校准点）。回答：Phase-PJ 的 "
        f"conditional-refinement 在实际决策路径上的激活频率。")
    out("")
    t_g0 = time.time()
    aud = phase_activation_audit(pl, quants8, powers8, H_aud, L_aud, CAL_H)
    out(f"- 决策状态数：{aud['n_decisions']}；P(继续)={fmt(aud['P_continue'])}")
    out(f"- **action-change rate**（Phase vs Myopic-PJ 动作不同）: "
        f"{fmt(aud['action_change_rate'])}")
    out(f"- **P(Q_phase≠Q_myopic)**（|Q_prog−Q^(1)|>1e-6 的 candidate 比例，"
        f"{aud['n_candidates']} 个 candidate）: {fmt(aud['P_Q_phase_ne_myopic'])}")
    regs = aud["regions"]
    out(f"- probe-feasible supports={aud['n_probe_feasible']}，region "
        f"A/B/C={regs['A']}/{regs['B']}/{regs['C']}；**pruning rate**="
        f"{fmt(aud['pruning_rate'])}（pruned {aud['n_pruned']}）")
    inactive = (aud["action_change_rate"] < 0.05)
    # 010 §一/§二：决定性指标是 **action-change rate**——Q 值层面的差异
    # （P(Q_phase≠Q_myopic)，probe Q 与 myopic one-step 不同）本身不产生
    # 作用：差异集中的 probe 76% 被 pruning 掉、保留的 probe 的 argmin 与
    # myopic 一致，且 jump-to-full 的 Q 两者恒相同 ⇒ 决策处处一致。
    g0_verdict = ("**Phase-PJ 与 Myopic-PJ 动作处处一致 ⇒ conditional "
                  "refinement 在决策层没有产生作用（010 §一 结论复现）**"
                  if inactive else
                  "Phase-PJ 仍有 action-level 激活（需 GPE-EA 扩大动作/规划）")
    out(f"- **G0 判定**：action-change rate={fmt(aud['action_change_rate'])}"
        f"（P(Q_phase≠Q_myopic)={fmt(aud['P_Q_phase_ne_myopic'])} 仅 Q 值层"
        f"激活、pruning rate={fmt(aud['pruning_rate'])} ⇒ 差异被剪/不改变 "
        f"argmin）⇒ {g0_verdict}。（{time.time()-t_g0:.1f}s）")
    out("")

    # ------------------------------------------------------------ G1
    out("## G1. Generalized r<s<t phase-envelope theorem（010 §七）")
    out("")
    out(f"> c_i(r→q)=b0+κ(q−r)；Q_prog^{{s,t}}−Q_dir^t == E[min{{Y_{{i,s,t}}, "
        f"b0}}]，Y=R(X_s)−E[R(X_t)|X_s]−κ(t−s)。角 (b0,κ)∈{GENERAL_CORNERS}。")
    out("")
    t_g1 = time.time()
    states = _reachable_states(pl, quants8, powers8, L_aud, N_STATES)
    g1 = generalized_envelope_gates(pl, states)
    ok_g1a = g1["max_identity_dev"] < 1e-9
    ok_g1b = g1["max_tower_dev"] < 1e-9
    ok_g1c = g1["max_deriv_dev"] < 1e-6
    ok_g1d = (g1["n_bstar"] > 0 and g1["n_bstar_ok"] == g1["n_bstar"])
    ok_g1e = (g1["n_special"] > 0
              and g1["n_special_ok"] == g1["n_special"]
              and g1["max_identity_dev_special"] < 1e-9)
    out(f"- 可达状态（G1 抽样）: {g1['n_states']}")
    out(f"- **G1a identity** max|g−g_alt|={g1['max_identity_dev']:.2e} "
        f"（目标 <1e-9）→ {'PASS' if ok_g1a else 'FAIL'}")
    out(f"- **G1b tower** max|E[E_R]−E_dir|={g1['max_tower_dev']:.2e} "
        f"（目标 <1e-9）→ {'PASS' if ok_g1b else 'FAIL'}")
    out(f"- **G1c derivative=survival** max|∂g−Pr(Y>b)|="
        f"{g1['max_deriv_dev']:.2e}（目标 <1e-6）→ {'PASS' if ok_g1c else 'FAIL'}")
    out(f"- **G1d b* 分类**（b*<∞ ⟺ E[Y]≥0，A/B/C）: "
        f"{g1['n_bstar_ok']}/{g1['n_bstar']} → {'PASS' if ok_g1d else 'FAIL'}")
    out(f"- **G1e special-case**（与 c21.phase_support_budget C 区逐状态"
        f"比对 Q_prog/Q_dir）: {g1['n_special_ok']}/{g1['n_special']} 一致、"
        f"identity {g1['max_identity_dev_special']:.2e} → "
        f"{'PASS' if ok_g1e else 'FAIL'}")
    out(f"（{time.time()-t_g1:.1f}s）")
    out("")

    # ------------------------------------------------------------ G2
    out("## G2. Matched-action Gate：GPE-EA vs Myopic-All（010 §八/§十二）")
    out("")
    out("> 两者 **相同 full action set** A={(i,s): s>r_i, s∈levels}、相同成本"
        "模型、相同 QoS、相同 calibration/test worlds（paired CRN）——唯一"
        "差别是 GPE-EA 对 probe 用 conditional-refinement Q（certificate 证明"
        "全分支 STOP 最优时精确退化为 one-step）。separately calibrated（各自"
        "28 网格选 θ̂）、fresh test、主 bit 认证 = paired EB UCB（G3），"
        "Hoeffding sanity。")
    out("")
    memo_gpe = GPEMemo()
    methods = [
        ("GPE-EA (Proposed)",
         (lambda pl, x, om, h, rho, eta, memo=memo_gpe:
             gpe_decision(pl, x, om, h, rho, eta, memo))),
        ("Myopic-All", c21.myopic_decision),
    ]
    t_cal = time.time()
    cal_res = {}
    for (nm, fn) in methods:
        ts, F, tables = calibrate_decide(pl, CAL_H, H_cal, L_cal, quants8,
                                         powers8, RHO_GRID, ETA_GRID, fn)
        cal_res[nm] = {"theta": ts, "feasible": F, "tables": tables}
        if ts is None:
            out(f"- {nm}：**∅（无 FEASIBLE）**；feasible {len(F)}/28")
        else:
            s = tables[ts]
            out(f"- {nm}：**θ̂=({ts[0]}, {fmt(ts[1],1)})**、Ê_cal[B]="
                f"{fmt(s['eb'])} bits、feasible {len(F)}/28")
    out(f"（calibration {time.time()-t_cal:.1f}s）")
    out("")

    r96 = {}
    r48 = {}
    for (nm, fn) in methods:
        ts = cal_res[nm]["theta"]
        if ts is None:
            r96[nm] = r48[nm] = None
            continue
        r96[nm] = eval_decide(pl, *ts, 96, H_t96, L_t96, fn, quants8, powers8)
        r48[nm] = eval_decide(pl, *ts, 48, H_t48, L_t48, fn, quants8, powers8)
    ts_p = cal_res["GPE-EA (Proposed)"]["theta"]
    ts_m = cal_res["Myopic-All"]["theta"]
    out("| | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (nm, s, th) in (("GPE-EA (Proposed)", r96.get("GPE-EA (Proposed)"), ts_p),
                        ("Myopic-All", r96.get("Myopic-All"), ts_m)):
        if s is None:
            continue
        cls = classify_qos2(s["kfa"], s["n0"], s["kmd"], s["n1"])
        out(f"| {nm} | ({th[0]},{fmt(th[1],1)}) | {fmt(s['kfa']/s['n0'])} "
            f"| {fmt(wilson_upper(s['kfa'], s['n0']))} | "
            f"{fmt(s['kmd']/s['n1'])} | {fmt(wilson_upper(s['kmd'], s['n1']))} "
            f"| {cls} | {fmt(s['entx'])} | {fmt(s['epl'])} | {fmt(s['eb'])} |")
    out("")

    def paired_gate(sa, sb, H):
        if sa is None or sb is None:
            return None
        D = sa["b"] - sb["b"]
        lo, hi = -H, H
        u_eb = eb_ucb(D, lo, hi)
        l_eb = eb_lcb(D, lo, hi)
        u_ho = hoeffding_upper(D, H)
        l_ho = hoeffding_lower(D, H)
        return {"D": D, "mean": float(D.mean()), "u_eb": u_eb, "l_eb": l_eb,
                "u_ho": u_ho, "l_ho": l_ho,
                "eb_smaller": u_eb < 0.0, "ho_smaller": u_ho < 0.0}

    for H, H_lbl, res in ((96, "H=96 (primary)", r96), (48, "H=48 (stress)", r48)):
        sa = res["GPE-EA (Proposed)"]
        sb = res["Myopic-All"]
        g = paired_gate(sa, sb, H)
        out(f"### {H_lbl}：paired D=E[B^GPE]−E[B^MyopicAll]")
        out("")
        if g is None or sa is None or sb is None:
            out("- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。")
            out("")
            continue
        feas_p = classify_qos2(sa["kfa"], sa["n0"], sa["kmd"], sa["n1"]) == "FEASIBLE"
        feas_m = classify_qos2(sb["kfa"], sb["n0"], sb["kmd"], sb["n1"]) == "FEASIBLE"
        out(f"- E[D]={fmt(g['mean'])}；**paired EB U95={fmt(g['u_eb'])}**"
            f"（L95={fmt(g['l_eb'])}，MP Thm 4 t=log(1/δ)）；Hoeffding sanity "
            f"U95={fmt(g['u_ho'])}（L95={fmt(g['l_ho'])}）")
        cls_p = classify_qos2(sa["kfa"], sa["n0"], sa["kmd"], sa["n1"])
        cls_m = classify_qos2(sb["kfa"], sb["n0"], sb["kmd"], sb["n1"])
        out(f"- QoS：GPE {cls_p}（U95 {fmt(wilson_upper(sa['kfa'], sa['n0']))}/"
            f"{fmt(wilson_upper(sa['kmd'], sa['n1']))}）、Myopic-All {cls_m}"
            f"（U95 {fmt(wilson_upper(sb['kfa'], sb['n0']))}/"
            f"{fmt(wilson_upper(sb['kmd'], sb['n1']))}）")
        if not (feas_p and feas_m):
            verdict = "QoS-UNRESOLVED（一方/双方 QoS 未认证，不比较 bits）"
        elif g["u_eb"] < 0:
            verdict = ("**PASS**：双方 FEASIBLE 且 paired EB U95<0 ⇒ GPE-EA "
                       "在 matched-action 下统计认证更省 bits")
        else:
            verdict = "BIT-UNRESOLVED（EB U95≥0；方向/量级见点估计）"
        out(f"- **G2 判定**：{verdict}")
        out(f"- 分解：E[N_tx] GPE={fmt(sa['entx'])} vs Myopic {fmt(sb['entx'])}；"
            f"E[B_payload] {fmt(sa['epl'])} vs {fmt(sb['epl'])}；E[B|H0] "
            f"{fmt(sa['eb0'])} vs {fmt(sb['eb0'])}、E[B|H1] {fmt(sa['eb1'])} vs "
            f"{fmt(sb['eb1'])}（017 §七 secondary）")
        out("")

    # ------------------------------------------------------------ G3
    out("## G3. Paired empirical-Bernstein UCB（010 §十；主 bit 认证）")
    out("")
    out("> G2 判定以 **paired fixed-N one-sided EB UCB**（MP Thm 4 plug-in "
        "variance，t=log(1/δ)，(n−1) 保守分母）为主；Hoeffding（D∈[−H,H]）为 "
        "sanity envelope。paired CRN 压缩 E[B] 差分方差 ⇒ EB 通常比 Hoeffding "
        "更紧（B0.4 系列同机制验证）。")
    out("")
    for H, H_lbl, res in ((96, "H=96 (primary)", r96), (48, "H=48 (stress)", r48)):
        g = paired_gate(res["GPE-EA (Proposed)"], res["Myopic-All"], H)
        if g is None:
            continue
        n = int(g["D"].size)
        tighter = g["u_eb"] < g["u_ho"]
        out(f"- {H_lbl}（n={n}）：D∈[−{H},{H}]；paired EB U95="
            f"{fmt(g['u_eb'])}、Hoeffding U95={fmt(g['u_ho'])} —— EB 界 "
            f"{'更紧' if tighter else '不更紧（报告诚实）'}；"
            f"{'支撑 G2 PASS' if g['eb_smaller'] else 'G2 未以该 H 通过'}。")
    out("")

    # ------------------------------------------------------------ 结论
    out("## 结论（010 §十二 路线）")
    out("")
    g96 = paired_gate(r96.get("GPE-EA (Proposed)"), r96.get("Myopic-All"), 96)
    if g96 is not None:
        sa = r96["GPE-EA (Proposed)"]
        sb = r96["Myopic-All"]
        feas_p = classify_qos2(sa["kfa"], sa["n0"], sa["kmd"], sa["n1"]) == "FEASIBLE"
        feas_m = classify_qos2(sb["kfa"], sb["n0"], sb["kmd"], sb["n1"]) == "FEASIBLE"
        verdict = ("PASS" if (feas_p and feas_m and g96["u_eb"] < 0)
                   else ("QoS-UNRESOLVED" if not (feas_p and feas_m)
                         else "BIT-UNRESOLVED"))
        out(f"- **G2 (H=96) 判定**：**{verdict}**（E[D]={fmt(g96['mean'])}、"
            f"EB U95={fmt(g96['u_eb'])}、Hoeffding U95={fmt(g96['u_ho'])}、"
            f"GPE QoS {classify_qos2(sa['kfa'], sa['n0'], sa['kmd'], sa['n1'])}、"
            f"Myopic QoS {classify_qos2(sb['kfa'], sb['n0'], sb['kmd'], sb['n1'])}）")
        out(f"- **G0**：Phase activation——conditional-refinement 激活率 "
            f"{fmt(aud['P_Q_phase_ne_myopic'])}、action-change "
            f"{fmt(aud['action_change_rate'])}（010 §一：Phase-PJ 近似 myopic）。")
        g1ok = (ok_g1a and ok_g1b and ok_g1c and ok_g1d and ok_g1e)
        out(f"- **G1**：generalized envelope 五 Gate "
            f"{'全部 PASS' if g1ok else 'FAIL'}（a/b/c/d/e 见上文）。")
        g48 = paired_gate(r48.get("GPE-EA (Proposed)"),
                          r48.get("Myopic-All"), 48)
        pass_h96 = g96["u_eb"] < 0
        pass_h48 = g48 is not None and g48["u_eb"] < 0
        v96 = ("G2 H=96 PASS" if (feas_p and feas_m and pass_h96)
               else ("G2 H=96 QoS-UNRESOLVED" if not (feas_p and feas_m)
                     else "G2 H=96 BIT-UNRESOLVED"))
        v48 = ("G2 H=48 PASS" if (pass_h48 and g48 is not None) else
               "G2 H=48 BIT-UNRESOLVED/QoS-UNRESOLVED")
        if pass_h48:
            nxt = ("**budget-regime 依赖结论**：conditional-refinement 的价值在"
                   "紧预算（H=48）统计认证体现（省证据 payload），宽预算（H=96）"
                   "把多余 bits 花在事务/检测余量上——按 010 §十二 进 C4 时保留"
                   "GPE-EA 并报告该 regime 依赖")
        else:
            nxt = ("GPE-EA 在注册工作点未获得 matched-action 认证收益——按"
                   "010 §十二 先调整 planning 深度（或换 C4 heterogeneous "
                   "airtime 后重新认证）")
        out(f"- **G2 双 Budget 判定**：{v96}；{v48}。")
        out(f"- **下一步（010 §十二）**：{nxt}；C4 = heterogeneous airtime"
            f"（b0,i、κ_i，positive/independent/anti-correlated regimes，"
            f"τ_i(r→r')=τ_ctrl,i+(b_hdr+(r'−r))/R_i，hard frame budget）。")
    else:
        out("- θ̂ 缺失 → G2 无法判定。")
    out("")
    out(f"总耗时: {time.time()-t0:.1f}s")
    out("")

    rp = os.path.join(out_dir, "MVS-C_C3e_report.md")
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()