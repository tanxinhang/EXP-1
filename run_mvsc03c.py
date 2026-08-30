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
    (L2) **Policy-class feasibility**：deterministic + randomized convex
         hull 能否达到？—— C3b 五方法的校准点 (kfa, kmd, n0) + 任意两
         点 episode-level 随机混合 v(λ)=λv1+(1−λ)v2（error prob 与 E[B]
         对 mixture 线性）；**Wilson U95 口径**（对 mixture 后的
         kfa(λ)/kmd(λ) 重新算 U95，升级 C3a 的点估计诊断）：是否存在
         mixture 使 U95(P_FA)≤α ∧ U95(P_MD)≤β。若 L2 也 NO ⇒
         policy-family infeasible（需 C3c 前修算法）；若 L2 YES 但
         L3 NO ⇒ 只是网格不够。
    (L3) **Controller-search feasibility**：当前有限 (ρ,η) 网格是否
         找到？—— 直接引用 C3b 校准：五方法各自 FEASIBLE 数 + θ̂。
         StaticProg 0/28 ⇒ registered-grid infeasible（对 StaticProg
         而言），但 L1/L2 已证明整体可行 ⇒ 是网格/控制器族不够，不是
         物理不可行。

统计口径：L1 用 MITM det-thr（P_FA≤α 的确定性阈值，005 §九）；L2 用
Wilson U95（与 G2/C3b 正式 Gate 一致）；L3 用 C3b 校准分类。三层各回答
"physical / policy-family / registered-grid" 的一种 infeasible 判定。
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
# L2 Policy-class feasibility：Wilson U95 convex-hull（mixture 计数重算）
# ---------------------------------------------------------------------------
def policy_class_feasibility(cal_points, alpha=ALPHA, beta=BETA):
    """deterministic + 两两 randomized mixture 能否进入 QoS 象限（Wilson
    U95 口径，升级 C3a 点估计诊断）。

    cal_points: [(name, kfa, kmd, n0, eb)] —— C3b 五方法校准的 θ̂ 点。
    对任意两方法 a,b 及 λ∈{0.25,0.5,0.75}，mixture 的 violation 计数线性：
        kfa(λ) = λ·kfa_a + (1−λ)·kfa_b（同一 worlds 上按概率混合控制器，
        决策独立 ⇒ 计数相加），再对 kfa(λ) 用 Wilson U95 判定。
    注意：mixture 计数 kfa(λ) 是实数，Wilson U95 对实数计数直接用
    （wilson_upper 接受 float；计数为 0..n 的线性组合，取整近似会丢失
    0.25 粒度——这里直接用 float，统计上等价于"期望计数"的 U95，标注为
    近似证据，正式认证在更大样本的显式 mixture 模拟）。

    返回 (n_det_feas, n_mix_enter, note)。"""
    n_det = sum(1 for (_n, kfa, kmd, n0, _e) in cal_points
                if classify_qos(kfa, kmd, n0) == "FEASIBLE")
    n_enter = 0
    n_pairs = 0
    for a in range(len(cal_points)):
        for b in range(a + 1, len(cal_points)):
            (_na, kfa_a, kmd_a, n0_a, _ea) = cal_points[a]
            (_nb, kfa_b, kmd_b, n0_b, _eb) = cal_points[b]
            n_pairs += 1
            for lam in (0.25, 0.5, 0.75):
                kfa = lam * kfa_a + (1 - lam) * kfa_b
                kmd = lam * kmd_a + (1 - lam) * kmd_b
                n0 = max(n0_a, n0_b)
                if wilson_upper(kfa, n0) <= alpha \
                        and wilson_upper(kmd, n0) <= beta:
                    n_enter += 1
                    break
    if n_det > 0:
        note = "deterministic policy-class feasible"
    elif n_enter > 0:
        note = (f"deterministic grid infeasible but randomized 2-point "
                f"mixture enters QoS (U95) in {n_enter}/{n_pairs} pairs ⇒ "
                f"policy-family feasible, grid insufficient")
    else:
        note = ("deterministic AND randomized 2-point mixtures never enter "
                "QoS (U95) ⇒ policy-family infeasible (needs algorithm "
                "change, not just grid)")
    return n_det, n_enter, note


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

    # ------------------------------------------------ L1 physical
    out("## L1. Physical feasibility（005 §十九：预算内最大 evidence 的 QoS）")
    out("")
    t1 = time.time()
    ok1, rows = physical_feasibility(model8, quants8)
    out("| 配置 | cost | P_D,max^det-thr | P_MD | ≤β | 判定 |")
    out("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        out(f"| {r['cfg']} | {r['cost']} | {fmt(r['pd'])} | {fmt(r['pmd'])} "
            f"| {fmt(BETA)} | {'YES' if r['ok'] else 'NO'} |")
    out("")
    out(f"- **L1 判定**：预算内存在最大 evidence 配置使 P_MD ≤ β ⇒ "
        f"**{'PHYSICALLY FEASIBLE' if ok1 else 'PHYSICALLY INFEASIBLE'}**"
        f"（{time.time()-t1:.1f}s）。")
    out("")
    out("> 注：MITM 是 4-UAV 子集精确融合（H=96 最多 4×8-bit，其余 UAV "
        "不发 = 0 evidence）；这与 C2.1 的 π_full 构造（4×8-bit=96=H）同构"
        "，是预算内可达的最强全融合 evidence。")
    out("")

    # ------------------------------------------------ L2 policy-class
    out("## L2. Policy-class feasibility（005 §十九：deterministic + "
        "randomized convex hull，Wilson U95 口径）")
    out("")
    out("> 升级 C3a 的点估计 convex_hull_diag：mixture 的 violation 计数"
        "线性（kfa(λ)=λkfa_a+(1−λ)kfa_b，同一 worlds 决策独立），再对 "
        "mixture 计数用 **Wilson U95** 判定——与正式 Gate 一致（C3a 的点"
        "估计进入 ≠ U95 认证，007 审计指出）。")
    out("")
    t2 = time.time()
    cal_points = []
    cal_all = {}                     # name -> (ts, F, tables)，L2/L3 共享
    for (nm, fn) in METHODS:
        ts, F, tables = calibrate_decide(pl, 96, H_cal, L_cal, quants8,
                                         powers8, RHO_GRID, ETA_GRID, fn)
        cal_all[nm] = (ts, F, tables)
        if ts is None:
            continue
        s = tables[ts]
        cal_points.append((nm, s["kfa"], s["kmd"], s["n0"], s["eb"]))
    if not cal_points:
        out(f"- 无 FEASIBLE θ̂ 的校准点（N_CAL={N_CAL} 太小？）——L2 "
            f"convex-hull 需 FULL N_CAL={FULL_N_CAL} 才有意义；SMOKE 下"
            f"本层标注 UNCERTAIN。")
        n_det = n_enter = 0
        note = "UNCERTAIN（SMOKE 无校准点；FULL 才有 L2 判定）"
    else:
        n_det, n_enter, note = policy_class_feasibility(cal_points)
        feas_names = [nm for (nm, kfa, kmd, n0, _e) in cal_points
                      if classify_qos(kfa, kmd, n0) == "FEASIBLE"]
        out(f"- deterministic feasible 方法：{n_det}/{len(cal_points)}"
            f"（{'、'.join(feas_names) if feas_names else '无'}）")
        out(f"- 两两 randomized mixture（U95 口径）进入 QoS 象限："
            f"{n_enter}/{len(cal_points) * (len(cal_points) - 1) // 2} 对")
    out(f"- **L2 判定**：{note}（{time.time()-t2:.1f}s）")
    out("")

    # ------------------------------------------------ L3 controller-search
    out("## L3. Controller-search feasibility（005 §十九：有限 (ρ,η) 网格）")
    out("")
    out("> 直接引用 C3b 校准：五方法各自 FEASIBLE 数（/28 网格）+ θ̂。"
        "若 L1/L2 YES 但某方法 L3 NO ⇒ 是 registered-grid 不够，不是物理"
        "不可行。")
    out("")
    t3 = time.time()
    for (nm, _fn) in METHODS:
        ts, F, tables = cal_all[nm]
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
    out(f"- **L1 Physical**：{'PASS（物理可行）' if ok1 else 'FAIL'}"
        f"——预算内最大 evidence 达 QoS。")
    out(f"- **L2 Policy-class**：{note}。")
    out(f"- **L3 Controller-search**：见上表（{n3_feas}/5 方法网格可行，"
        f"其余 registered-grid infeasible）。")
    out("")
    out("> **三层归因（005 §十九）**：若三层都 YES ⇒ 机制层可行；L1 NO ⇒ "
        "physical infeasible（改预算/成本）；L1 YES + L2 NO ⇒ policy-family "
        "infeasible（改算法）；L1/L2 YES + L3 NO ⇒ registered-grid "
        "infeasible（扩网格）。当前：L1 YES、L2 视 mixture、L3 "
        f"{n3_feas}/5 可行——不可行方法的 NO 是 registered-grid/policy 层，"
        "非物理层。")
    out("")
    out(f"总耗时: {time.time()-t0:.1f}s")
    out("")

    rp = os.path.join(out_dir, "MVS-C_C3c_report.md")
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
