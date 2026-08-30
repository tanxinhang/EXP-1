"""MVS-C C3a: Migration Gate + Definition/Control Hardening (advice/005.md).

定位（005 §十七）：C2.1 FREEZE 后，进入 **C3a Migration** —— 新架构的
budget-aware Myopic-FG 在 N=8 homogeneous special case 下必须**复现旧
B0.7-G2**（migration 判定只认 Myopic-All，不让 Phase-FG 参加判决，005 §十七；
Phase-PJ 是 C3b 的 proposed）。

等价性基础（已在 C3a 审计中验证）：run_mvsc021.py 的
`myopic_decision`（budget-aware Myopic-All，A={1,2,4,8} one-step QoS-dual）
与 G2 runner（run_mvsb07g2.py）的 `q_min_fg`/`sim_method` 在 N=8 homogeneous
（GAMMA_B、levels(1,2,4,8)、b_setup=16）上 **episode 级 cost/N_tx/payload
逐位一致**（开发期 300 episodes + FULL seeds 复算 × 3 θ corners ×
H∈{48,96} × FG/D8，max|Δ|=0；T40 回归用 40 episodes）——两者是同语义
controller 的两次独立实现。因此 migration Gate = 用新架构 controller 跑
G2 协议，逐项复现 G2 FULL 数值：

  θ̂_FG = θ̂_D8 = (256, 0.8)
  H=96: E[B_FG−B_D8] = −5.3250, Hoeffding U95 = −1.1710, 双方 FEASIBLE → G2 PASS
  H=48: E[B_FG−B_D8] = −5.0263, Hoeffding U95 = −2.9493, 双方 FEASIBLE → G2 PASS

Contract hardening（005 §七/§八，C3b 前必须完成，本 runner 一并落地）：
  (H1) **Myopic-PJ**：与 Phase-PJ 相同动作集 A={next, full}（r→r_next、
       r→r_max），continuation 用 one-step Q^(1)=c+E[R(X')] —— 消除
       005 §七 的 action-space confounding（旧 Phase 只 {next,full} 而旧
       Myopic 枚举 {1,2,4,8}，无法干净归因 conditional-refinement value）；
  (H2) **Static Progressive**：固定 SNR 顺序（argsort(−γ_i)）逐 UAV 渐进
       ladder（r→next），STOP 用同一 QoS-dual 规则（005 §八：Gate B 要求
       Phase-FG < StaticProg；reviewer 会问收益是否来自任意 progressive
       ladder 而非 adaptive precision）；
  (H3) 统一命名：phase A/B/C 以 phase_boundary.py 为准（A: E[Y]<0 ⇒ b*=∞；
       B: E[Y]=0；C: E[Y]>0），run_mvsc01.py 的旧命名已修正；
  (H4) P_D,max 明确标注 **P_D,max^det-thr**（005 §九：离散 LLR 下确定性
       threshold 不是严格 NP randomized optimum；本 runner 只引用，不重算）；
  (H5) **policy mixture / convex-hull 诊断**（005 §六）：把校准网格每个
       deterministic (ρ,η) 控制器映射为 v=(P_FA, P_MD, E[B])，报告
       randomized episode-level mixture 的可行域凸包是否与
       P_FA≤α ∧ P_MD≤β 相交（error probabilities 与 E[B] 对外部随机化
       线性 ⇒ convex hull 即可判断 policy-class feasibility，与
       deterministic-grid feasibility 区分开）。

冻结参数（沿用 G2 017 §四，migration 必须同协议）：N=8、GAMMA_B、
levels=(1,2,4,8)、b_setup=16、QoS(P_FA≤0.12, P_MD≤0.40)；ρ∈{128,256,512,1024}、
η∈{0.8,…,2.0}（28 combos/method，仅 calibration）；calibration worlds
FG/D8 共用、test worlds 完全分离（paired CRN）；主 operating point H=96、
secondary stress H=48（同冻结 controller）。统计：fixed-N paired one-sided
Hoeffding（D∈[−H,H]）+ Wilson QoS 双侧 95% 上端点。

命名（005 §三/§十七 对齐）：实验对象是 **separately calibrated one-step
QoS-dual controllers**；Proposed 正式命名 **Phase-Guided Self-Conditional-
Refinement policy（Phase-PJ）**（SystemModel/README 已同步），C3a 只做
migration，C3b 才做 Phase-PJ 算法主比较。
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
from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA_B = g2.GAMMA_B
BH = g2.BH
LEVELS = g2.LEVELS
R_MAX = g2.R_MAX
ALPHA = g2.ALPHA
BETA = g2.BETA
RHO_GRID = g2.RHO_GRID
ETA_GRID = g2.ETA_GRID
SEED0 = g2.SEED0
SEED_CAL = g2.SEED_CAL
SEED_TEST = g2.SEED_TEST
FULL_N_CAL = g2.FULL_N_CAL
FULL_N_TEST = g2.FULL_N_TEST
OUT_DIR = g2.OUT_DIR
SMOKE_OUT_DIR = os.path.join(OUT_DIR, "smoke")

MAT_EP = g2.MAT_EP

fmt = g2.fmt
mp = g2.mp
wilson_upper = g2.wilson_upper
wilson_lower = g2.wilson_lower
hoeffding_upper = g2.hoeffding_upper
hoeffding_lower = g2.hoeffding_lower
classify_qos = g2.classify_qos
sample_set = g2.sample_set
r_rho = g2.r_rho
q1_fast = g2.q1_fast
apply_action = g2.apply_action


# ---------------------------------------------------------------------------
# 控制器族（全部 budget-aware：c<=h 才可行；STOP ⟺ R ≤ min_{A(h)} Q^(1)）。
# 统一签名 decide(pl, x, om, h, rho, eta) -> (dec, diag)，dec 形如
# ("STOP",) 或 ("ACT", i, kind, r2)。复用 run_mvsc021 的 Myopic-All/Direct8
# （C3a 迁移对象，等价性已审计），新写 Myopic-PJ / StaticProg。
# ---------------------------------------------------------------------------

# (M-All) budget-aware Myopic-All —— G2 FG 语义，C3a migration 主对象
myopic_all_decision = c21.myopic_decision
# (D8) Direct8
direct_decision = c21.direct_decision


def myopic_pj_decision(pl, x, om, h, rho, eta):
    """Myopic-PJ（005 §七 H1）：A = {(i, r_next), (i, r_max)}（probe-jump
    动作集，与 Phase-PJ 完全相同），continuation 用 one-step
    Q^(1) = c_a + E[R(X')|x,a]（myopic，不做 conditional refinement）。
    STOP ⟺ R(x) ≤ min_{A(h)} Q^(1)。"""
    cands = []
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        r_next = next((r2 for r2 in pl.levels if r2 > r), None)
        targets = [t for t in (r_next, pl.r_max) if t is not None and t > r]
        for r2 in targets:
            c = BH + (r2 - r)
            if c <= h:
                cands.append((q1_fast(pl, x, om, i, r2, rho, eta),
                              ("ACT", i, "PJ", r2)))
    if not cands:
        return ("STOP",), {}
    best_q, best_a = min(cands, key=lambda t: t[0])
    if r_rho(om, rho, eta) <= best_q:
        return ("STOP",), {}
    return best_a, {}


def static_prog_decision(pl, x, om, h, rho, eta):
    """Static Progressive（005 §八 H2，007 审计修正停止语义）。

    固定 SNR 顺序 order = argsort(−γ_i)（静态，与 baselines.py B11 一致），
    按顺序逐 UAV 渐进 ladder（r → next level）；**停止用 |Ω| ≥ η**（与判决
    阈值一致，B11 的 |Ω|≥η_s early-stop 语义），**不用** QoS-dual 的
    R ≤ min_a Q^(1)（后者在 root 处 R<Q 恒成立 ⇒ 23/28 网格点 E[B]=0
    全停退化，007 审计指出）。rho 参数保留仅为与其它 QoS-dual 控制器
    的 θ̂=(ρ,η) 校准网格同构（η 即停止/判决阈值；ρ 在本决策中不参与）。
    —— 唯一缺的是 adaptive selection/precision：顺序固定、粒度固定。"""
    # 停止：|Ω| ≥ η（与判决阈值一致）。无动作可选也停（预算耗尽）。
    if abs(om) >= eta:
        return ("STOP",), {}
    order = np.argsort(-np.asarray(
        [q.model.gamma_db[q.i] for q in pl.quants]))
    cands = []
    for i in order:
        i = int(i)
        zi = (x // pl.powers[i]) % BASE_B
        r, _ = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        r_next = next((r2 for r2 in pl.levels if r2 > r), None)
        if r_next is None:
            continue
        c = BH + (r_next - r)
        if c <= h:
            # 固定顺序：第一个可行 ladder 动作即选（不比较 Q，
            # 无 adaptive selection —— 005 §八 的对照点）
            return ("ACT", i, "PROG", r_next), {}
    return ("STOP",), {}


# ---------------------------------------------------------------------------
# episode 模拟（与 G2 eval_theta 同协议；decide 为上述决策函数）
# ---------------------------------------------------------------------------
def sim_decide(pl, rho, eta, H, L_i, decide, quants8, powers8):
    x, h, lam, cost, pay, nt = 0, float(H), 0.0, 0.0, 0.0, 0
    while True:
        if h < 1e-9:
            break
        om = pl.omega(x)
        dec, _d = decide(pl, x, om, h, rho, eta)
        if dec[0] == "STOP":
            break
        i, _kind, r2 = dec[1], dec[2], dec[3]
        zi = (x // powers8[i]) % BASE_B
        r_cur, _m_cur = z_decode_b(zi)
        c = BH + (r2 - r_cur)
        if c > h + 1e-9:
            raise AssertionError(f"budget violation: c={c} > h={h}")
        m2 = int(quants8[i].cell_index(r2, L_i[i]))
        lam2 = lam + quants8[i].llr[r2][m2]
        if r_cur > 0:
            lam2 -= quants8[i].llr[r_cur][_m_cur]
        z2 = z_code_b(r2, m2)
        x = x + (z2 - zi) * powers8[i]
        h -= c
        lam = lam2
        cost += c
        pay += (r2 - r_cur)
        nt += 1
    return lam, cost, nt, pay


def eval_decide(pl, rho, eta, H, H_all, L_all, decide, quants8, powers8):
    n_ep = len(H_all)
    b_m = np.empty(n_ep)
    pay_m = np.empty(n_ep)
    nt_m = np.empty(n_ep)
    lam_m = np.empty(n_ep)
    viol = 0
    for e in range(n_ep):
        lam, cost, nt, pay = sim_decide(pl, rho, eta, H, L_all[e], decide,
                                        quants8, powers8)
        lam_m[e], b_m[e], nt_m[e], pay_m[e] = lam, cost, nt, pay
        if abs(cost - (BH * nt + pay)) > 1e-9 or cost > H + 1e-9:
            viol += 1
    i0 = H_all == 0
    i1 = H_all == 1
    n0 = int(np.count_nonzero(i0))
    kfa = int(np.sum(lam_m[i0] > eta))
    kmd = int(np.sum(lam_m[i1] <= eta))
    return {"b": b_m, "pay": pay_m, "nt": nt_m, "lam": lam_m,
            "eb": float(b_m.mean()),
            "eb0": float(b_m[i0].mean()), "eb1": float(b_m[i1].mean()),
            "entx": float(nt_m.mean()), "epl": float(pay_m.mean()),
            "kfa": kfa, "kmd": kmd, "n0": n0, "viol": viol}


def calibrate_decide(pl, H, H_cal, L_cal, quants8, powers8, rho_grid,
                     eta_grid, decide):
    tables = {}
    for rho in rho_grid:
        for eta in eta_grid:
            tables[(rho, eta)] = eval_decide(pl, rho, eta, H, H_cal, L_cal,
                                             decide, quants8, powers8)
    F = {th: s for th, s in tables.items()
         if classify_qos(s["kfa"], s["kmd"], s["n0"]) == "FEASIBLE"}
    ts = min(F, key=lambda th: (F[th]["eb"], th[0], th[1])) if F else None
    return ts, F, tables


# ---------------------------------------------------------------------------
# 4-bit N=4 exhaustive dominance-safety certificate（005 §十）
# ---------------------------------------------------------------------------
def exhaustive_dominance_4bit(rho=512.0, eta=1.2, H=96):
    """005 §十：C2.1 的 263 个 reachable support 是 **sampled on-policy
    certificate**（A 区甚至没被多 UAV 状态覆盖，A=0/B=16/C=247）。4-bit N=4
    全状态空间只有 23^4 = 279841（BASE_B=279 的 z-code 前 23 个 = level
    0/1/2/4 消息码），可以做**真正的 exhaustive budget-reachable**
    dominance-safety 检查：

      对每个从 root 在 H 内可达的 (x, cost)（BFS，成本为真实
      c=16+(r'-r)），对每个 UAV i 在剩余预算 h=H−cost 下：
        prune_probe_ok ⟹ Q_prog ≥ Q_dir − ε；A 区绝不剪。

    8-bit（279^4）不可行，留给 resolution-stratified + adversarial 抽样。"""
    mm4 = GaussianDetectorModel([-1.0, 1.0, 3.0, 5.0], (0.5, 0.5))
    qu4 = [NestedQuantizer(i, mm4, r_max=4, levels=(1, 2, 4))
           for i in range(4)]
    pl4 = SparsePlanner(qu4, 1.0, 1.0, b_h=16.0, cross_level=True,
                        levels=(1, 2, 4), direct_only=False, delta_c=1.0)
    # BFS over (x, h) 对：从 (0, H) 出发，动作 r→r' 花 c=16+(r'-r)，
    # 子状态 (x', h-c)。同一 x 可以以不同剩余预算到达（例如先发 1-bit
    # 再补 4-bit vs 直接发 4-bit），region 判定依赖 h —— 因此必须传播
    # (x,h) 对而不是只记 min cost（005 §十：budget-reachable 全覆盖，
    # 不能漏掉高成本访问的 A 区）。
    seen_pairs = {(0, H)}
    frontier = [(0, H)]
    while frontier:
        nxt = []
        for (x, h) in frontier:
            if h < 1e-9:
                continue
            zs = pl4.decode(x)
            om = pl4.omega(x)
            for i in range(4):
                zi = zs[i]
                r_cur, _ = z_decode_b(zi)
                for (r2, c_true, _qb, _cells) in pl4._tpl[i][zi]:
                    if r2 <= r_cur:
                        continue
                    c = BH + (r2 - r_cur)
                    if c > h + 1e-9:
                        continue
                    # 每个 cell 都是可达子状态（消息随机；对 (x,h) 可达性
                    # 只需存在即可，但为完整覆盖状态空间枚举全部 cell）
                    for m2 in range(2 ** r2):
                        z2 = z_code_b(r2, m2)
                        x2 = x + (z2 - zi) * pl4.powers[i]
                        pair = (x2, round(h - c, 9))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            nxt.append(pair)
        frontier = nxt
    n_chk = n_dom = n_bad = n_A = n_B = n_C = 0
    for (x, h) in seen_pairs:
        om = pl4.omega(x)
        for i in range(4):
            sup = c21.phase_support_budget(pl4, x, om, i, h, rho, eta)
            if sup is None or not sup["probe_feas"]:
                continue
            n_chk += 1
            if sup["region"] == "A":
                n_A += 1
                if sup["prune_probe_ok"]:
                    n_bad += 1          # A 区绝不剪
            else:
                if sup["region"] == "B":
                    n_B += 1
                else:
                    n_C += 1
                if not sup["dir_feas"]:
                    continue
                n_dom += 1
                if sup["prune_probe_ok"] and sup["Q_prog"] is not None \
                        and sup["Q_prog"] < sup["Q_dir"] - 1e-8:
                    n_bad += 1
    return {"n_pairs": len(seen_pairs), "n_chk": n_chk, "n_dom": n_dom,
            "n_bad": n_bad, "regions": {"A": n_A, "B": n_B, "C": n_C}}


# ---------------------------------------------------------------------------
# policy mixture / convex-hull 诊断（005 §六 H5）
# ---------------------------------------------------------------------------


def convex_hull_diag(tables, alpha=ALPHA, beta=BETA):
    """把每个 deterministic (ρ,η) 控制器映射为 v=(P_FA, P_MD, E[B])；
    error probabilities 与 E[B] 对 episode-level randomized mixture 线性 ⇒
    v(λ)=λv1+(1−λ)v2。这里不做全凸包计算，只报告两个决定性诊断：
      (a) 是否有单一可行点（deterministic feasible）；
      (b) 任意两网格点的随机混合能否进入 QoS 象限（二维 mixture 扫描），
          给出 policy-class（convexified）feasibility 的证据边界。

    **口径（007 审计注明）**：本诊断用 **点估计** pfa=kfa/n0（error prob 对
    mixture 线性，005 §六 的期望值论点成立）；正式 Gate 的 QoS 判定用
    **Wilson U95 上界**（classify_qos）。点估计 ≤ α 不保证 U95 ≤ α——
    mixture 的认证可行性需在 C3c 用 U95/convex-hull 正式计算。因此
    n_pair_enter 是**期望值证据**，不是统计认证。

    返回 (n_feas, n_pair_enter, note)。"""
    pts = [(th, s) for th, s in tables.items()]
    n_feas = sum(1 for _th, s in pts
                 if classify_qos(s["kfa"], s["kmd"], s["n0"]) == "FEASIBLE")
    n_pair_enter = 0
    n_pair_total = 0
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            (th1, s1), (th2, s2) = pts[a], pts[b]
            pfa1, pmd1 = s1["kfa"] / s1["n0"], s1["kmd"] / s1["n0"]
            pfa2, pmd2 = s2["kfa"] / s2["n0"], s2["kmd"] / s2["n0"]
            n_pair_total += 1
            for lam in (0.25, 0.5, 0.75):
                pfa = lam * pfa1 + (1 - lam) * pfa2
                pmd = lam * pmd1 + (1 - lam) * pmd2
                if pfa <= alpha and pmd <= beta:
                    n_pair_enter += 1
                    break
    note = ("deterministic grid feasible" if n_feas > 0 else
            "deterministic grid infeasible; convex-hull (2-point mixture, "
            f"POINT-ESTIMATE metric) enters QoS quadrant in "
            f"{n_pair_enter}/{n_pair_total} pairs"
            if n_pair_enter > 0 else
            "deterministic grid infeasible AND 2-point mixtures (point-"
            "estimate) never enter QoS quadrant (policy-class evidence, "
            "005 §六; U95 certification deferred to C3c)")
    return n_feas, n_pair_enter, note


# ---------------------------------------------------------------------------
# 报告
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
    else:
        N_TEST = FULL_N_TEST
        N_CAL = FULL_N_CAL
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

    out(f"# MVS-C C3a — Migration Gate + Contract Hardening（advice/005.md，{tag}）")
    out("")
    out("> **定位（005 §十七）**：C2.1 FREEZE 后进入 **C3a Migration**——"
        "新架构 budget-aware Myopic-FG（Myopic-All，one-step QoS-dual，"
        "A={1,2,4,8}）在 N=8 homogeneous special case 下**逐项复现旧 "
        "B0.7-G2**。migration 判决只认 Myopic-All（Phase-PJ 不参加，005 "
        "§十七）。controller 等价性已审计：run_mvsc021.myopic_decision vs "
        "G2 q_min_fg（开发期 300 episodes + FULL seeds 复算，T40 回归 40 "
        "episodes × 3 corners × H∈{{48,96}} × FG/D8，max|Δ(cost,N_tx,"
        "payload)|=0）。")
    out("")
    out(f"> 冻结参数（G2 017 §四 同协议）：N=8（GAMMA_B）、levels=(1,2,4,8)、"
        f"b_setup={BH}、QoS(P_FA≤{ALPHA}, P_MD≤{BETA})；ρ∈{RHO_GRID}、"
        f"η∈{ETA_GRID}（28 combos/method，仅 calibration）；calibration "
        f"worlds 共用、test fresh 分离（paired CRN）；主 operating point "
        f"H=96、secondary stress H=48（同冻结 controller）；fixed-N paired "
        f"one-sided Hoeffding + Wilson QoS 上端点。N_CAL={N_CAL}、"
        f"N_TEST={N_TEST}。")
    out("")
    out("> **Contract hardening（005 §七/§八/§六，C3b 前完成）**：(H1) "
        "**Myopic-PJ**（A={{next,full}}，one-step）——与 Phase-PJ 同动作集，"
        "消除旧 Phase-vs-Myopic 动作空间混杂；(H2) **Static Progressive**"
        "（固定 SNR 顺序 ladder）——Gate B 主基线回归；(H3) 统一 A/B/C 命名"
        "（phase_boundary 为准）；(H4) P_D,max 标注 **det-thr**；(H5) "
        "**policy-mixture/convex-hull 诊断**（deterministic-grid vs "
        "policy-class feasibility 分离，005 §六）。")
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

    # ------------------------------------------------- 1. migration calibration
    out("## 1. Migration calibration（Myopic-All vs Direct8，G2 同协议）")
    out("")
    t_cal = time.time()
    methods = [
        ("Myopic-All (G2-FG)", myopic_all_decision),
        ("Direct8", direct_decision),
    ]
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
                f"{fmt(s['eb'])} bits、feasible {len(F)}/28"
                f"{'  ← 与 G2 一致' if ts == (256, 0.8) else '  ← 与 G2 不一致！'}")
    out(f"（{time.time()-t_cal:.1f}s）")
    out("")

    # ------------------------------------------------- 2. migration test
    out("## 2. Migration test @ H=96（θ̂ 冻结、fresh worlds、paired）")
    out("")
    t96 = time.time()
    r96 = {}
    for (nm, fn) in methods:
        ts = cal_res[nm]["theta"]
        if ts is None:
            r96[nm] = None
            continue
        r96[nm] = eval_decide(pl, *ts, 96, H_t96, L_t96, fn, quants8, powers8)
    ts_fg = cal_res["Myopic-All (G2-FG)"]["theta"]
    ts_d8 = cal_res["Direct8"]["theta"]
    if ts_fg is not None and ts_d8 is not None:
        s_fg, s_d8 = r96["Myopic-All (G2-FG)"], r96["Direct8"]
        D = s_fg["b"] - s_d8["b"]
        u95 = hoeffding_upper(D, 96.0)
        l95 = hoeffding_lower(D, 96.0)
        out("| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | "
            "E[B_payload] | E[B] |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, s in (("Myopic-All (G2-FG)", s_fg), ("Direct8", s_d8)):
            ufa = wilson_upper(s["kfa"], s["n0"])
            umd = wilson_upper(s["kmd"], s["n0"])
            cls = classify_qos(s["kfa"], s["kmd"], s["n0"])
            th = ts_fg if "Myopic" in key else ts_d8
            out(f"| {key} | ({th[0]},{fmt(th[1],1)}) | {fmt(s['kfa']/s['n0'])} "
                f"| {fmt(ufa)} | {fmt(s['kmd']/s['n0'])} | {fmt(umd)} | "
                f"{cls} | {fmt(s['entx'])} | {fmt(s['epl'])} | {fmt(s['eb'])} |")
        out("")
        d_mean = float(D.mean())
        out(f"- paired D=E[B^FG]−E[B^D8] = {fmt(d_mean)}；"
            f"**Hoeffding U95={fmt(u95)}**、L95={fmt(l95)}"
            f"（n_paired={H_t96.size}，D∈[−96,96]）")
        g2_target = -5.3250
        fg_feas = classify_qos(s_fg["kfa"], s_fg["kmd"], s_fg["n0"]) == "FEASIBLE"
        d8_feas = classify_qos(s_d8["kfa"], s_d8["kmd"], s_d8["n0"]) == "FEASIBLE"
        out(f"- **Migration anchor 复现**：E[D] 目标 −5.3250、实测 "
            f"{fmt(d_mean)}（偏差 {fmt(abs(d_mean - g2_target), 3)}）"
            f"{'→ PASS' if abs(d_mean - g2_target) < 0.01 else '→ FAIL'}；"
            f"U95 目标 −1.1710、实测 {fmt(u95)}"
            f"{'→ PASS' if u95 < 0 else '→ FAIL'}；双方 FEASIBLE "
            f"{'→ PASS' if (fg_feas and d8_feas) else '→ FAIL'}")
    else:
        out("> θ̂ 缺失 → migration test 无法比较（QoS-UNRESOLVED）。")
    out(f"（{time.time()-t96:.1f}s）")
    out("")

    # ------------------------------------------------- 3. stress H=48
    out("## 3. Secondary stress @ H=48（同冻结 θ̂，诚实报告 boundary）")
    out("")
    t48 = time.time()
    r48 = {}
    for (nm, fn) in methods:
        ts = cal_res[nm]["theta"]
        if ts is None:
            r48[nm] = None
            continue
        r48[nm] = eval_decide(pl, *ts, 48, H_t48, L_t48, fn, quants8, powers8)
    if ts_fg is not None and ts_d8 is not None:
        s_fg, s_d8 = r48["Myopic-All (G2-FG)"], r48["Direct8"]
        D = s_fg["b"] - s_d8["b"]
        u95 = hoeffding_upper(D, 48.0)
        l95 = hoeffding_lower(D, 48.0)
        d_mean = float(D.mean())
        out(f"- paired D = {fmt(d_mean)}；Hoeffding U95={fmt(u95)}、"
            f"L95={fmt(l95)}；G2 目标 −5.0263（偏差 "
            f"{fmt(abs(d_mean + 5.0263), 3)}）")
        for key, s in (("Myopic-All (G2-FG)", s_fg), ("Direct8", s_d8)):
            cls = classify_qos(s["kfa"], s["kmd"], s["n0"])
            out(f"  - {key}：分类={cls}、E[B]={fmt(s['eb'])}")
    else:
        out("> θ̂ 缺失。")
    out(f"（{time.time()-t48:.1f}s）")
    out("")

    # ------------------------------------- 4. hardening: Myopic-PJ + StaticProg
    out("## 4. Contract hardening：Myopic-PJ / StaticProg（005 §七/§八）")
    out("")
    out("> **目的**：为 C3b 准备同动作集因果对照（Phase-PJ vs Myopic-PJ）与 "
        "Gate B 主基线（StaticProg）。本 runner 只报告 calibrated θ̂ 与 "
        "feasible 数（同 G2 协议），C3b 才做 Phase-PJ 算法主比较。")
    out("")
    hard_methods = [
        ("Myopic-PJ (A={next,full})", myopic_pj_decision),
        ("StaticProg (fixed SNR ladder)", static_prog_decision),
    ]
    for (nm, fn) in hard_methods:
        ts, F, tables = calibrate_decide(pl, CAL_H, H_cal, L_cal, quants8,
                                         powers8, RHO_GRID, ETA_GRID, fn)
        n_zero = sum(1 for s in tables.values() if s["eb"] < 0.5)
        if ts is None:
            out(f"- {nm}：**∅（无 FEASIBLE）**；feasible {len(F)}/28"
                + (f"；其中 {n_zero}/28 网格点为 E[B]=0 全停退化"
                   f"（QoS-dual 停止在 root 即触发，未发送任何消息）" if n_zero else ""))
        else:
            s = tables[ts]
            out(f"- {nm}：**θ̂=({ts[0]}, {fmt(ts[1],1)})**、Ê_cal[B]="
                f"{fmt(s['eb'])} bits、feasible {len(F)}/28"
                f"{f'；{n_zero}/28 网格点全停退化（E[B]=0）' if n_zero else ''}")
        out("")
    # static prog order check
    order = np.argsort(-np.asarray([q.model.gamma_db[q.i]
                                    for q in quants8]))
    out(f"- StaticProg 固定顺序（SNR 降序）: {list(map(int, order))} "
        f"（GAMMA_B={list(GAMMA_B)}）")
    out("")

    # --------------------------- 5. policy-mixture / convex-hull 诊断（H5）
    out("## 5. Policy-mixture / convex-hull 诊断（005 §六 H5）")
    out("")
    out("> 把 deterministic (ρ,η) 控制器映射为 v=(P_FA,P_MD,E[B])；error "
        "probabilities 与 E[B] 对 episode-level randomized mixture 线性 ⇒ "
        "若二维混合能进入 QoS 象限，则 deterministic-grid infeasible 不能"
        "直接推出 policy-class infeasible。本诊断是 C3c 三层 feasibility "
        "frontier 的前置证据。**口径（007 审计注明）**：进入判定用**点估计**"
        "（kfa/n0，期望值线性）；正式 Gate 用 **Wilson U95**，混合的 U95 "
        "认证可行性留待 C3c convex-hull 正式计算——点估计进入 ≠ 统计认证。")
    out("")
    for (nm, fn) in methods + hard_methods:
        tables = cal_res[nm]["tables"] if nm in cal_res else None
        if tables is None:
            ts, _F, tables = calibrate_decide(
                pl, CAL_H, H_cal, L_cal, quants8, powers8,
                RHO_GRID, ETA_GRID, fn)
        n_feas, n_pair_enter, note = convex_hull_diag(tables)
        out(f"- **{nm}**：deterministic feasible {n_feas}/28；"
            f"2-point mixture 进入 QoS 象限 {n_pair_enter} 对 → {note}")
    out("")

    # ------------------------------- 5b. exhaustive dominance-safety (005 §十)
    out("## 5b. 4-bit N=4 exhaustive dominance-safety certificate（005 §十）")
    out("")
    out("> C2.1 的 263 个 reachable support 是 **sampled on-policy "
        "certificate**（A 区未覆盖：A=0/B=16/C=247，A 主要靠 N=1 synthetic "
        "反例）。005 §十 建议 4-bit/N=4 做**真正 exhaustive**（23^4=279841）"
        "dominance-safety 检查：prune ⟹ Q_prog ≥ Q_dir − ε 且 A 区绝不剪；"
        "8-bit（279^4）留给 resolution-stratified + adversarial 抽样。")
    out("")
    t_ex = time.time()
    ex = exhaustive_dominance_4bit()
    out(f"- 4-bit N=4 **budget-reachable** (x,h) 对（BFS，H=96，真实成本 "
        f"c=16+Δr）：{ex['n_pairs']} 对；检查 {ex['n_chk']} 个 "
        f"(x,i,probe-feasible) 支撑（region A/B/C 计数 {ex['regions']}）；"
        f"dominance 检查 {ex['n_dom']}（dir_feas），矛盾 {ex['n_bad']} → "
        f"{'**PASS**（exhaustive budget-reachable certificate）' if ex['n_bad'] == 0 else '**FAIL**'}"
        f"（{time.time()-t_ex:.1f}s）")
    out(f"- **A 区覆盖说明**：exhaustive BFS 下 A 区（c1≤h<c_dir）在 N=4/H=96 "
        f"无可达实例。正确理由（枚举）：单 UAV 在 4-bit ladder（levels "
        f"1,2,4、b=16）下可达花费 ∈ {{17,18,20,34,36,52}}（0→1:17、0→2:18、"
        f"0→4:20、0→1→2:34、0→1→4/0→2→4:36、0→1→2→4:52），3 个其他 UAV 的"
        f"总花费 = 三者和 ∈ {{51..58,60,68..74,76,85..}}；A 区需花费∈(76,79]"
        f"（剩余 h∈[17,20)），该集合中 **(76,79] 为空**（74→76 间隙 2、76→85 "
        f"间隙 9；76 对应 h=20 恰是 B/C 边界）。故 A 区不可达，A 区安全由 "
        f"N=1 synthetic 反例 + “A 区绝不剪”代码路径 + dominance 检查显式"
        f"断言保证（与 005 §十 观察一致）。")
    out("")

    # ------------------------------------------------------------ 6. 结论
    out("## 结论")
    out("")
    mig_ok = (ts_fg == (256, 0.8) and ts_d8 == (256, 0.8))
    if ts_fg is not None and ts_d8 is not None and mig_ok:
        s_fg, s_d8 = r96["Myopic-All (G2-FG)"], r96["Direct8"]
        D = s_fg["b"] - s_d8["b"]
        u95 = hoeffding_upper(D, 96.0)
        d_mean = float(D.mean())
        anchor_ok = (abs(d_mean + 5.3250) < 0.01 and u95 < 0
                     and classify_qos(s_fg["kfa"], s_fg["kmd"], s_fg["n0"])
                     == "FEASIBLE"
                     and classify_qos(s_d8["kfa"], s_d8["kmd"], s_d8["n0"])
                     == "FEASIBLE")
    else:
        anchor_ok = False
    out(f"- **C3a Migration Gate**：θ̂_FG=θ̂_D8=(256,0.8) 复现 "
        f"{'PASS' if mig_ok else 'FAIL'}；H=96 anchor（E[D]=−5.3250、"
        f"U95<0、双方 FEASIBLE）复现 {'**PASS**' if anchor_ok else '**FAIL**'}"
        f" → migration {'通过，可进入 C3b' if (mig_ok and anchor_ok) else '未通过，先修 migration'}"
        f"（005 §十七：Phase-PJ 不参加判决）。")
    out("")
    out(f"- **Contract hardening 落地**：Myopic-PJ 已实现（A={{next,full}}，"
        f"one-step，与 Phase-PJ 同动作集）；StaticProg 已实现（固定 SNR "
        f"ladder）；A/B/C 命名统一 phase_boundary 为准；P_D,max 标注 det-thr；"
        f"policy-mixture 诊断见 §5（C3c 前正式 convex-hull frontier）。")
    out("")
    out(f"总耗时: {time.time()-t0:.1f}s")
    out("")

    rp = os.path.join(out_dir, "MVS-C_C3a_report.md")
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
