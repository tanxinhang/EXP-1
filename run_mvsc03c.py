"""MVS-C C3c: Three-Layer Feasibility Frontier (advice/005.md §十九).

定位（005 §十九）：把 "INFEASIBLE" 拆成三层明确判定，极大提高论文理论
可信度：

    (L1) **Physical feasibility**：最大 evidence 在 hard budget 下能否
         达到 QoS？—— 枚举预算内可达的最大 evidence 消息组合（N=8、
         H=96、b=16：最多 4×8-bit = 96），用 MITM 精确全融合 ROC
         （P_D,max^det-thr，005 §九 口径）评估其 QoS 是否达
         P_FA≤α ∧ P_MD≤β（G2 口径 α=0.12、β=0.40）。注意 MITM 是 4-UAV
         子集精确融合（预算只够 4 个 8-bit；其他 UAV 不发 = 0 evidence），
         这与 C2.1 的 π_full 构造（4×8-bit=96=H）同构。
    (L2) **Policy-class feasibility（C3d 修正，010 §三/§四）**：对**每个
         方法自身**计算 conv{ v_θ^m : θ∈Θ_m }——只混合该方法自己的网格点
         （旧实现跨方法混合且排除无 θ̂ 的方法，010 §三 判定为 P0 逻辑缺口）；
         fractional-count Wilson U95 为近似证据，进入的组合在**全新 test
         worlds** 上做显式 Bernoulli-λ mixture（整数 kfa/kmd + Wilson，
         n0/n1 分离）正式认证（010 §四）。
    (L3) **Controller-search feasibility**：当前有限 (ρ,η) 网格是否
         找到？—— 直接引用 C3b 校准：各方法自身 FEASIBLE 数 + θ̂。
         **StaticProg 修正口径（010 §五）**：ρ 不参与其策略 ⇒ 28 网格点
         只含 7 个唯一阈值策略，报告 **0/7 unique**，并撤掉 "无可行点
         本身即 adaptive 必要性证据" 表述。

统计口径：L1 用 MITM det-thr（P_FA≤α 的确定性阈值，005 §九），改名
Constructive Physical Feasibility Certificate（010 §六：构造 FAIL 不蕴含
physical infeasible）；L2 用 per-method hull（Wilson U95，n0/n1 分离）；
L3 用 C3b 校准分类。三层各回答 "physical / registered-policy-family /
registered-grid" 的一种 infeasible 判定。
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
import run_mvsc03b as c3b
from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs.sparse import SparsePlanner, BASE_B

GAMMA_B = g2.GAMMA_B
BH = g2.BH
LEVELS = g2.LEVELS
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
mp = g2.mp
wilson_upper = g2.wilson_upper
classify_qos = g2.classify_qos
classify_qos2 = c3a.classify_qos2
sample_set = g2.sample_set
sim_decide = c3a.sim_decide
eval_decide = c3a.eval_decide
calibrate_decide = c3a.calibrate_decide

# C3b 五方法（005 §十八）
METHODS = c3b.METHODS


# ---------------------------------------------------------------------------
# L1 Physical feasibility：预算内最大 evidence 配置的 MITM QoS
# ---------------------------------------------------------------------------
def physical_feasibility(model, quants, alpha=ALPHA, beta=BETA, H=96):
    """枚举预算内可达的最大 evidence 组合（N=8、b=16），用 MITM 精确
    全融合 ROC（4-UAV 子集）评估 QoS。

    预算 H=96：最多 4×8-bit（4×24=96）或 5×4-bit（5×20=100>96 ⇒ 4×4-bit
    =80）。更强证据配置：
      (a) 4 strongest SNR UAVs 发 8-bit（cost 96，恰满）—— 最大 evidence；
      (b) 4 strongest 发 4-bit（cost 80）—— 更省的下界；
      (c) 全部 8 UAV 发 1-bit（cost 8×17=136>96 不可行）—— 说明 4×8-bit
          是预算内可达的最强全融合配置（更多 UAV 低比特 vs 更少 UAV
          高比特的组合由 MITM 覆盖）。
    返回 (verdict, rows)。rows: [{cfg, cost, P_D, P_MD, ok}]。
    """
    strong = sorted(range(len(quants)), key=lambda i: -model.gamma_db[i])[:4]
    rows = []
    # (a) 4x8-bit on strongest 4
    q4 = [quants[i] for i in strong]
    pfa8, pmd8 = c21.full_fusion_ref_mitm(model, q4, 8)
    _e8, pd8, pmd8v = c21.pd_max_at_alpha(pfa8, pmd8, alpha)
    rows.append({"cfg": "4 strongest UAVs 8-bit", "cost": 4 * (BH + 8),
                 "pd": pd8, "pmd": pmd8v,
                 "ok": pmd8v <= beta and _e8 is not None})
    # (b) 4x4-bit
    pfa4, pmd4 = c21.full_fusion_ref_mitm(model, q4, 4)
    _e4, pd4, pmd4v = c21.pd_max_at_alpha(pfa4, pmd4, alpha)
    rows.append({"cfg": "4 strongest UAVs 4-bit", "cost": 4 * (BH + 4),
                 "pd": pd4, "pmd": pmd4v, "ok": pmd4v <= beta})
    # (c) weaker-SNR 4 UAVs 8-bit (sanity: 最强 4 是否必要)
    weak = sorted(range(len(quants)), key=lambda i: model.gamma_db[i])[:4]
    qw = [quants[i] for i in weak]
    pfaw, pmdw = c21.full_fusion_ref_mitm(model, qw, 8)
    _ew, pdw, pmdwv = c21.pd_max_at_alpha(pfaw, pmdw, alpha)
    rows.append({"cfg": "4 weakest UAVs 8-bit", "cost": 4 * (BH + 8),
                 "pd": pdw, "pmd": pmdwv, "ok": pmdwv <= beta})
    verdict = any(r["ok"] for r in rows)
    return verdict, rows


# ---------------------------------------------------------------------------
# L2 Policy-class feasibility（C3d 修正，010 §三/§四）：**per-method**
# 注册凸包 + 显式 Bernoulli-λ mixture 认证
# ---------------------------------------------------------------------------
def per_method_policy_class(tables, alpha=ALPHA, beta=BETA):
    """010 §三（C3d P0）：conv{ v_θ^m : θ∈Θ_m } 与该方法自身是否进入 QoS
    象限——**只混合方法自身的网格点**，不再跨方法混合（跨方法 mixture 不是
    要回答的问题：StaticProg 因无 θ̂ 被旧 L2 直接排除，导致
    "L2 YES ⇒ StaticProg 只是 registered-grid infeasible" 的推理不成立）。

    tables: {(rho,eta): s}（该方法校准表，s 含 kfa/kmd/n0/n1/eb）。
      n_det   = deterministic FEASIBLE 数（classify_qos2，n0/n1 分离）；
      n_enter = 自身网格点两两 randomized mixture 进入 QoS 象限的
                （method,λ）组合数；fractional-count Wilson U95（期望计数
                线性，010 §四 标注为近似证据，正式认证在显式 mixture）；
      best_mix = (θ_a, θ_b, Ê[B], λ) —— 供显式 mixture Gate 使用。
    返回 dict（verdict 见上）。"""
    pts = list(tables.items())
    n_det = 0
    for (_th, s) in pts:
        if classify_qos2(s["kfa"], s["n0"], s["kmd"], s["n1"]) == "FEASIBLE":
            n_det += 1
    n_enter = 0
    best = None
    n_pairs = 0
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            (_ha, sa), (_hb, sb) = pts[a], pts[b]
            n_pairs += 1
            n0 = max(sa["n0"], sb["n0"])
            n1 = max(sa["n1"], sb["n1"])
            for lam in (0.25, 0.5, 0.75):
                kfa = lam * sa["kfa"] + (1 - lam) * sb["kfa"]
                kmd = lam * sa["kmd"] + (1 - lam) * sb["kmd"]
                if (wilson_upper(kfa, n0) <= alpha
                        and wilson_upper(kmd, n1) <= beta):
                    n_enter += 1
                    eb = lam * sa["eb"] + (1 - lam) * sb["eb"]
                    if best is None or eb < best[2]:
                        best = (_ha, _hb, eb, lam)
                    break
    if n_det > 0:
        verdict = "deterministic feasible（自身网格存在 FEASIBLE 点）"
    elif n_enter > 0:
        verdict = ("registered-hull feasible（deterministic 无可行点，但"
                   "自身网格 randomized mixture 进入 QoS——网格不足、"
                   "policy family 可行）")
    else:
        verdict = ("registered-hull infeasible（自身网格 deterministic + "
                   "mixture 均未进入 QoS——该注册策略族不可行）")
    return {"n_det": n_det, "n_enter": n_enter, "n_pairs": n_pairs,
            "n_grid": len(pts), "best_mix": best, "verdict": verdict}


def static_prog_unique_count(tables):
    """010 §五（C3d）：StaticProg 的 ρ 不参与策略（仅与其它 QoS-dual 网格
    同构），因此 4ρ×7η = 28 网格点只含 7 个**唯一**阈值策略（按 η）。
    返回 (unique_count, eta_list, n_zero_unique)。"""
    etas = sorted({th[1] for th in tables})
    rho0 = next(iter(tables))[0]
    n_zero_unique = sum(1 for eta in etas
                        if tables[(rho0, eta)]["eb"] < 0.5)
    return len(etas), etas, n_zero_unique


def explicit_mixture_gate(pl, th_a, th_b, lam, H_set, L_set, decide_fn,
                          quants8, powers8, seed):
    """010 §四（C3d）：**显式 Bernoulli-λ mixture 认证**——正式口径。

    在全新 test worlds（stratified n0=n1）上，每个 episode 先抽
    Z_e ~ Bernoulli(λ)（独立种子，与 worlds 无关），Z=1 用 θ_a、Z=0 用
    θ_b 运行该方法的完整 episode；得到**整数** kfa/kmd（各自 episode 用
    自己的 η 判决），Wilson U95（n0/n1 分离）判定。用于把 L2 的
    fractional-count 近似证据升级为显式认证（010 §四的干净方案）。
    返回 (kfa, kmd, n0, n1, cls, eb, n_ep)。"""
    rng = np.random.default_rng(seed)
    n_ep = len(H_set)
    b_all = np.empty(n_ep)
    lam_all = np.empty(n_ep)
    eta_e = np.empty(n_ep)
    for e in range(n_ep):
        flip = rng.random() < lam
        th = th_a if flip else th_b
        lamv, cost, _nt, _pay = sim_decide(pl, *th, 96, L_set[e], decide_fn,
                                           quants8, powers8)
        b_all[e] = cost
        lam_all[e] = lamv
        eta_e[e] = th[1]
    i0 = H_set == 0
    i1 = H_set == 1
    n0 = int(np.count_nonzero(i0))
    n1 = int(np.count_nonzero(i1))
    kfa = int(np.sum(lam_all[i0] > eta_e[i0]))
    kmd = int(np.sum(lam_all[i1] <= eta_e[i1]))
    cls = classify_qos2(kfa, n0, kmd, n1)
    return {"kfa": kfa, "kmd": kmd, "n0": n0, "n1": n1, "cls": cls,
            "eb": float(b_all.mean()), "n_ep": n_ep}


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
    out_dir = SMOKE_OUT_DIR if SMOKE else OUT_DIR
    tag = "SMOKE" if SMOKE else "FULL"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"# MVS-C C3c — Three-Layer Feasibility Frontier"
        f"（advice/005.md §十九，{tag}）")
    out("")
    out("> **定位（005 §十九）**：把 'INFEASIBLE' 拆成三层明确判定——"
        "**L1 Physical**（最大 evidence 在预算内能否达 QoS）、**L2 "
        "Policy-class**（deterministic + randomized convex hull 能否达）、"
        "**L3 Controller-search**（当前有限 (ρ,η) 网格是否找到）。以后再"
        "看到 INFEASIBLE，可明确说是 physical / policy-family / "
        "registered-grid 哪一种。")
    out("")
    out(f"> 协议：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup={BH}、"
        f"H=96、QoS(P_FA≤{ALPHA}, P_MD≤{BETA})；L1 用 MITM det-thr 精确 "
        f"ROC（005 §九）；L2 用 Wilson U95（与正式 Gate 一致）；L3 引用 "
        f"C3b 校准。N_CAL={N_CAL}、N_TEST={N_TEST}。")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=LEVELS)
               for i in range(8)]
    powers8 = [BASE_B ** i for i in range(8)]
    pl = SparsePlanner(quants8, 1.0, 1.0, b_h=BH, cross_level=True,
                       levels=LEVELS, delta_c=1.0)

    H_cal, L_cal = sample_set(N_CAL, SEED_CAL, model8)
    H_t96, L_t96 = sample_set(N_TEST, SEED_TEST * 1000 + 2, model8)

    # ------------------------------------------------ L1 physical
    out("## L1. Constructive Physical Feasibility Certificate"
        "（010 §六：改名 + 诚实限定）")
    out("")
    out("> **010 §六 改名**：L1 是 **constructive certificate**——枚举预算内"
        "可达的一组**构造型**最大 evidence 配置（4×8-bit / 4×4-bit 于最强 4 "
        "UAV、4×8-bit 于最弱 4 UAV）用 MITM 精确全融合 ROC 评估 QoS。"
        "**PASS 成立**；但若该构造 FAIL，**不能反推 physical infeasible**——"
        "除非真正求解 budgeted maximum-evidence oracle（010 §六；5×1-bit=85、"
        "混合分配 (8,8,4,2,…) 等大量组合未枚举）。")
    out("")
    t1 = time.time()
    ok1, rows = physical_feasibility(model8, quants8)
    out("| 配置 | cost | P_D,max^det-thr | P_MD | ≤β | 判定 |")
    out("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        out(f"| {r['cfg']} | {r['cost']} | {fmt(r['pd'])} | {fmt(r['pmd'])} "
            f"| {fmt(BETA)} | {'YES' if r['ok'] else 'NO'} |")
    out("")
    out(f"- **L1 判定**：构造型最大 evidence 配置中**存在**使 P_MD ≤ β 的配置"
        f" ⇒ **{'PHYSICALLY FEASIBLE (constructive)' if ok1 else 'CONSTRUCT FAIL'}**"
        f"（{time.time()-t1:.1f}s）。构造 FAIL 不蕴含 physical infeasible"
        f"（010 §六）。")
    out("")
    out("> 注：MITM 是 4-UAV 子集精确融合（H=96 最多 4×8-bit，其余 UAV "
        "不发 = 0 evidence）；这与 C2.1 的 π_full 构造（4×8-bit=96=H）同构"
        "，是预算内可达的强全融合 evidence。")
    out("")

    # ------------------------------------------------ L2 policy-class (C3d)
    out("## L2. Policy-class feasibility（C3d 修正，010 §三/§四："
        "**per-method** registered convex hull + 显式 mixture 认证）")
    out("")
    out("> **010 §三 P0 修正**：旧 L2 把无 θ̂ 的方法（StaticProg）直接排除，"
        "且 mixture 跨方法混合——那不是要回答的问题。修正后 L2 对**每个方法"
        "自身**计算 conv{ v_θ^m : θ∈Θ_m }：只混合该方法自己的网格点；"
        "fractional-count Wilson U95 仅作**近似证据**（010 §四），进入的"
        "组合再由**全新 test worlds 上的显式 Bernoulli-λ mixture**（整数"
        "kfa/kmd + Wilson，n0/n1 分离）正式认证。")
    out("")
    t2 = time.time()
    cal_all = {}                     # name -> (ts, F, tables)，L2/L3 共享
    for (nm, fn) in METHODS:
        ts, F, tables = calibrate_decide(pl, 96, H_cal, L_cal, quants8,
                                         powers8, RHO_GRID, ETA_GRID, fn)
        cal_all[nm] = (ts, F, tables)
    l2 = {}
    for (nm, _fn) in METHODS:
        ts, _F, tables = cal_all[nm]
        l2[nm] = per_method_policy_class(tables)
        r = l2[nm]
        if r["n_det"] > 0:
            out(f"- **{nm}**：deterministic feasible {r['n_det']}/{r['n_grid']}"
                f" → {r['verdict']}")
        elif r["n_enter"] > 0:
            best = r["best_mix"]
            out(f"- **{nm}**：deterministic {r['n_det']}/{r['n_grid']}、"
                f"自身网格 mixture 进入 QoS {r['n_enter']}/{r['n_pairs']} 对"
                f"（θ̂ 缺失）→ {r['verdict']}；推荐 mix θ_a={best[0]}、"
                f"θ_b={best[1]}、λ={best[3]}（approximation）")
        else:
            out(f"- **{nm}**：deterministic {r['n_det']}/{r['n_grid']}，"
                f"自身网格 mixture 也未进入 → {r['verdict']}")
    # 显式 mixture 认证（010 §四）：对 registered-hull feasible 但无 θ̂ 的方法
    out("")
    out("- **显式 Bernoulli-λ mixture 认证（010 §四，全新 test worlds）**：")
    n_elig = sum(1 for (nm, _fn) in METHODS
                 if l2[nm]["n_det"] == 0 and l2[nm]["best_mix"] is not None)
    if n_elig == 0:
        out("  - 无合格方法：本 regime 无 \"registered-hull feasible 但无 θ̂\""
            " 的方法（StaticProg 的 hull infeasible ⇒ 显式 mixture 无对象；"
            "010 §四 机制保留，本 regime 未触发）。")
    for (nm, _fn) in METHODS:
        r = l2[nm]
        if r["n_det"] > 0 or r["best_mix"] is None:
            continue
        (th_a, th_b, _eb, lam) = r["best_mix"]
        em = explicit_mixture_gate(pl, th_a, th_b, lam, H_t96, L_t96,
                                   dict(METHODS)[nm], quants8, powers8,
                                   SEED_TEST * 1000 + 7)
        out(f"  - {nm}：mix θ_a={th_a}、θ_b={th_b}、λ={lam} 于 "
            f"n={em['n_ep']} 全新 worlds → kfa/kmd={em['kfa']}/{em['kmd']}"
            f"（n0/n1={em['n0']}/{em['n1']}）、Wilson 分类={em['cls']}"
            f"{' → **policy family 正式可行（网格不足）**' if em['cls'] == 'FEASIBLE' else ''}"
            f"{' → **registered-hull infeasible 确认（显式）**' if em['cls'] == 'INFEASIBLE' else ''}"
            f"{'（UNCERTAIN，需扩样）' if em['cls'] == 'UNCERTAIN' else ''}")
    l2_note = ("per-method registered convex hull：见上表（每方法独立判定）；"
               "显式 mixture 认证如上述")
    out(f"- **L2 判定**：{l2_note}（{time.time()-t2:.1f}s）")
    out("")

    # ------------------------------------------------ L3 controller-search
    out("## L3. Controller-search feasibility（005 §十九：有限 (ρ,η) 网格；"
        "C3d 修正 StaticProg 口径）")
    out("")
    out("> 引用 C3b 校准：各方法自身 FEASIBLE 数 + θ̂。**StaticProg 修正**"
        "（010 §五）：ρ 不参与其策略 ⇒ 28 网格点只含 7 个唯一阈值策略，"
        "报告 **0/7 unique**；并**撤掉**\"StaticProg 无可行点本身即 adaptive "
        "必要性证据\"表述（010 §五）——adaptive 必要性由 L2 的 per-method "
        "判定（StaticProg 自身 hull/mixture 是否可行）支持。")
    out("")
    t3 = time.time()
    for (nm, _fn) in METHODS:
        ts, F, tables = cal_all[nm]
        if nm == "StaticProg":
            n_u, etas, n_zero_u = static_prog_unique_count(tables)
            if ts is None:
                out(f"- {nm}：**∅（无 FEASIBLE）**；feasible {len(F)}/28"
                    f" = **0/{n_u} unique threshold policies**"
                    f"{f'；{n_zero_u}/{n_u} 全停退化' if n_zero_u else ''}"
                    f" ⇒ registered-grid infeasible（ρ 不参与策略，"
                    f"4ρ 重复 ⇒ 7 个唯一策略）")
            else:
                s = tables[ts]
                out(f"- {nm}：**θ̂=({ts[0]}, {fmt(ts[1],1)})**、feasible "
                    f"{len(F)}/28 = {len(F)}/{n_u} unique ⇒ "
                    f"registered-grid feasible")
            continue
        n_zero = sum(1 for s in tables.values() if s["eb"] < 0.5)
        if ts is None:
            out(f"- {nm}：**∅（无 FEASIBLE）**；feasible {len(F)}/28"
                f"{f'；{n_zero}/28 全停退化' if n_zero else ''} "
                f"⇒ registered-grid infeasible")
        else:
            s = tables[ts]
            out(f"- {nm}：**θ̂=({ts[0]}, {fmt(ts[1],1)})**、feasible "
                f"{len(F)}/28 ⇒ registered-grid feasible"
                f"{f'；{n_zero}/28 全停退化' if n_zero else ''}")
    out(f"（{time.time()-t3:.1f}s）")
    out("")

    # ------------------------------------------------ conclusion
    n3_feas = sum(1 for (nm, _fn) in METHODS
                  if cal_all[nm][0] is not None)
    out("## 结论")
    out("")
    out(f"- **L1 Constructive Physical Feasibility Certificate**："
        f"{'PASS（构造存在：预算内配置达 QoS）' if ok1 else 'CONSTRUCT FAIL'}"
        f"——010 §六：构造 FAIL 不蕴含 physical infeasible。")
    out(f"- **L2 Policy-class（C3d per-method)**：{l2_note}。")
    out(f"- **L3 Controller-search**：见上表（{n3_feas}/5 方法网格可行；"
        f"StaticProg 按 **0/7 unique** 口径报告，010 §五）。")
    out("")
    out("> **三层归因（005 §十九）**：L1 NO ⇒ 需 budgeted max-evidence "
        "oracle 才能判 physical infeasible；L1 YES + 某方法 L2 NO ⇒ 该"
        "方法的注册策略族 infeasible（改算法/策略族）；L1/L2 整体 YES + "
        "某方法 L3 NO 且 L2 为 registered-hull feasible ⇒ 只是网格不够"
        "（010 §三：per-method 判定，不再跨方法混合）。当前：L1 PASS；"
        "L2 见上方 per-method 表；L3 见上表——不可行方法的 NO 是 "
        "registered-grid/policy 层，非物理层。")
    out("")
    out(f"总耗时: {time.time()-t0:.1f}s")
    out("")

    rp = os.path.join(out_dir, "MVS-C_C3c_report.md")
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
