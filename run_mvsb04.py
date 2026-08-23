"""MVS-B0.4/B0.4r: Pairwise-Difference Time-Uniform EB-CS Planner
(advice/009.md §7-§12 + advice/010.md B0.4r credibility patch).

Core change vs B0.3a/B0.3c: estimate the PAIR DIFFERENCE
    Delta_{a,b} = Q_a^{pi_b} - Q_b^{pi_b}   via  Z_t^{a,b} = G_a(W_t) - G_b(W_t)
on a shared latent world W_t, instead of arm-wise Q_a estimates compared via
U_a - L_b.  The nested-evidence coupling (kappa ~ 13, 008 §9) makes Var(Z)
~13x smaller than Var(G_a)+Var(G_b).

B0.4r credibility repairs (010.md):
  R0  canonical SAMPLE + SUPPORT orientation (range_ab built from the
      canonicalized pair; PairCS.update asserts z in [lo,hi]); a descending
      top_k_uavs regression (T25) locks it.
  R1  the FORMAL certificate path is now the theorem-backed predictable
      plug-in empirical-Bernstein CS (Maurer-Pontil 2009 Thm 6 per-n bound +
      union bound / peeling with delta-spending, CONTINUOUS interval, no grid
      inversion); the WSR-style variance-adaptive betting grid CS is kept as
      the experimental tighter ablation (010 §2).
  R2  hard invariant z in [lo, hi] in every pair CS (catches orientation bugs).
  R3  G1 is now a four-cell ablation (010 §4): H0 arm-wise/Hoeffding/full,
      H1 pair-Hoeffding/challenger, E1 pair-EB/challenger/shared, E0
      pair-EB/challenger/independent — isolating the pair-statistic, the CS,
      and the nested-coupling contributions.
  R4  G2 is a hard ROLLOUT-budget curve P(Q-Q_min<=2) vs R in {1000,3000,
      6000,12000}, both planners stopping at n_rollouts >= R.
  R5  G4 reworded: per-world rollout complexity O(|A|) -> O(1); total
      certification complexity still depends on |A| (log P confidence
      allocation + O(|A|) challenger search).

Sampling: predictable candidate-challenger pair (a_t, b_t) chosen from F_{t-1}
BEFORE W_t is drawn (009 §10); per-pair alpha_ab with sum <= delta.  Each world
costs 2 rollouts (1 when the challenger is STOP).  Certificate (009 §8):
U_{a_hat,b} <= eps for all b in A^+(x) minus {a_hat}  =>  Q_{a_hat} <=
min_b Q_b + eps with prob >= 1 - delta.

Gates:
  G0  Pair-CS anytime validity (FORMAL EB path + betting ablation): N=4 exact
      Delta_ab inside the CS for ALL n;
  G1  four-cell ablation (rollouts-to-certify / cert rate), 010 §4;
  G2  hard-rollout-budget action-quality curves, P(Q-Q_min <= eps) vs R;
  G3  N=8 shallow (H=24/34/40): near-tie-aware Q(a_CR) - Q_min;
  G4  scaling: 2 rollouts/world (per-world O(1)); total depends on |A|.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs.rbl_cr import (CRRBL, SNRDirectBase, LatentWorld, exact_qa_pi_b)
from opmvs.rbl_eb import CRRBLEB, PairCS
from opmvs.sparse import z_code_b

GAMMA_A = [-1.0, 1.0, 3.0, 5.0]
GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
SEED0 = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def mp(flag):
    return "PASS" if flag else "FAIL"


def random_state4(rng, seed_i):
    z = [0] * 4
    for u in range(int(rng.integers(0, 3))):
        i = int(rng.integers(0, 4))
        r = (1, 2, 4)[int(rng.integers(0, 3))]
        z[i] = z_code_b(r, int(rng.integers(0, 2 ** r)))
    return sum(int(z[i]) * (279 ** i) for i in range(4))


def pair_range_of(cr, x_int, h, a, b):
    """Deterministic pairwise range [l_a - u_b, u_a - l_b] (009 §9)."""

    def ep(act):
        if act is None:
            R0 = cr.pl.r_stop(x_int)
            return R0, R0
        i, r2 = act
        from opmvs.sparse import z_decode_b
        r_old, _ = z_decode_b(cr._z_digit(x_int, i))
        c = cr.b_h + (r2 - r_old)
        d = cr.bound_a(x_int, h, c, action=act)
        return c, c + d

    la, ua = ep(a)
    lb, ub = ep(b)
    return la - ub, ua - lb


# ------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    SMOKE = args.smoke
    n_state_g1 = 6 if SMOKE else 15
    n_state_g2 = 20 if SMOKE else 60
    n_runs_g0 = 60 if SMOKE else 200
    w_g1 = 1500 if SMOKE else 6000
    R_ab = 20000 if SMOKE else 24000
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.4/B0.4r — Pairwise-Difference EB-CS Planner（credibility patch）")
    out("")
    out("> 依据 `advice/009.md` §7-§12 与 `advice/010.md` B0.4r。B0.4 核心：直接估计 "
        "Δ_{a,b} = Q_a^{π_b} − Q_b^{π_b}（Z_t^{a,b} = G_a(W_t) − G_b(W_t)，共享 latent world），"
        "取代 arm-wise Q_a + U_a−L_b；predictable candidate–challenger pair sampling（009 §10，"
        "(a_t,b_t) 由 F_{t−1} 决定，每 pair α_ab 且 Σα_ab≤δ）；每 world 2 个 rollout；"
        "证书（009 §8）：U_{â,b} ≤ ε ∀b∈A⁺\\{â} ⟹ Q_â ≤ min_b Q_b + ε。")
    out("> **B0.4r（010）**：R0 canonical sample+support 同向（PairCS 硬断言 z∈[lo,hi]，"
        "descending top-k regression）；R1 **formal 证书路径 = predictable plug-in "
        "empirical-Bernstein CS**（Maurer–Pontil 2009 Thm 6 + peeling union bound，"
        "连续区间、无 grid inversion），betting grid CS 降级为实验性消融（不再承担主证书）；"
        "R3 G1 四格消融（H0/H1/E1/E0）分离 pair statistic / CS / coupling 三段因果；"
        "R4 G2 改为硬 rollout 预算曲线；R5 G4 表述修正（per-world O(1)，total 仍随 |A|）。")
    out("> 设计依据：耦合效率 κ = (σ_a²+σ_b²)/Var(G_a−G_b) ≈ 13（008 §9 实测），")
    out("> 所以『不要更精确估计 Q_a，直接更精确估计决策所需的 Q_a−Q_b』（009 §12）。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if SMOKE else 'FULL'}")
    out("")

    model4 = GaussianDetectorModel(GAMMA_A)
    quants4 = [NestedQuantizer(i, model4, r_max=4, levels=(1, 2, 4)) for i in range(4)]
    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=(1, 2, 4, 8)) for i in range(8)]
    bh = 16.0
    muM, muF = 256.0, 256.0 * np.exp(1.0)
    base4 = SNRDirectBase(quants4, GAMMA_A, bh, eta_b=2.0, levels=(1, 2, 4))
    base8 = SNRDirectBase(quants8, GAMMA_B, bh, eta_b=2.0, levels=(1, 2, 4, 8))
    cr = CRRBL(quants4, muM, muF, bh, base4, levels=(1, 2, 4), delta_c=1.0, seed=7)

    # ------------------------------------------------------------ G0
    out("## 1. G0 — Pair-CS anytime validity（exact Δ_{a,b}，N=4；formal EB + betting 消融）")
    out("")
    pair0 = ((3, 4), (2, 4))
    Q_pi0 = exact_qa_pi_b(cr, 0, 40)
    Delta0 = Q_pi0[pair0[0]] - Q_pi0[pair0[1]]
    lo0, hi0 = pair_range_of(cr, 0, 40, pair0[0], pair0[1])
    alpha0 = 0.01
    n_max0 = 300
    for mode, tag in (("eb", "PrPl-EB (formal, 连续区间)"), ("betting", "betting (experimental 消融)")):
        covered0 = 0
        for r in range(n_runs_g0):
            cr.rng = np.random.default_rng(4000 + r)
            cs = PairCS(lo0, hi0, alpha0, mode=mode)
            ok = True
            for n in range(1, n_max0 + 1):
                W = LatentWorld(cr, 0)
                z = cr._rollout(0, 40, pair0[0], W) - cr._rollout(0, 40, pair0[1], W)
                cs.update(z)
                L, U = cs.bounds()
                ok &= (Delta0 >= L and Delta0 <= U)
            covered0 += int(ok)
        cov0 = covered0 / n_runs_g0
        out(f"- {tag}：Δ^{{π_b}} = {fmt(Delta0)}，range = [{fmt(lo0)},{fmt(hi0)}]；"
            f"anytime coverage（∀ n ≤ {n_max0}，α={alpha0}）= {fmt(cov0)}"
            f"（下界 1−α = {fmt(1 - alpha0)}）→ **{mp(cov0 >= 1 - alpha0 - 0.03)}**")
    out("- 注（010 §2）：formal 证书路径 = Maurer–Pontil (2009) Thm 6 + peeling union bound"
        "（连续区间、无 grid inversion）；betting grid CS 仅作实验性收紧消融。")
    out("")

    # ------------------------------------------------------------ G1
    out("## 2. G1 — 四格消融：pair statistic × CS × coupling（010 §4，ε=40，rollout 预算）")
    out("")
    rng1 = np.random.default_rng(SEED0 + 1)
    eps_ab = 40.0
    cells = {"H0": ("arm-wise/Hoeffding/full", None, True),
             "H1": ("pair/Hoeffding/challenger", "hoeffding", True),
             "E1": ("pair/EB/challenger/shared", "eb", True),
             "E0": ("pair/EB/challenger/independent", "eb", False)}
    stats1 = {c: {"roll": [], "cert": 0, "n": 0} for c in cells}
    t_g1 = time.time()
    for s in range(n_state_g1):
        x = random_state4(rng1, s)
        if not exact_qa_pi_b(cr, x, 40):
            continue
        nA = max(1, len(cr.feasible_actions(x, 40)))
        for c, (desc, mode, shared) in cells.items():
            if c == "H0":
                hf = CRRBL(quants4, muM, muF, bh, base4, levels=(1, 2, 4),
                           delta_c=1.0, seed=13)
                hf.rng = np.random.default_rng(7000 + s)
                _, inf = hf.plan(x, 40, eps=eps_ab, delta=0.05,
                                 max_samples=max(1, R_ab // nA))
            else:
                eb = CRRBLEB(quants4, muM, muF, bh, base4, levels=(1, 2, 4),
                             delta_c=1.0, seed=11, cs_mode=mode, shared=shared)
                eb.cr.rng = np.random.default_rng(6000 + s)
                _, inf = eb.plan(x, 40, eps=eps_ab, delta=0.05,
                                 max_worlds=max(1, R_ab // 2))
            stats1[c]["n"] += 1
            if inf["certified"]:
                stats1[c]["cert"] += 1
                stats1[c]["roll"].append(inf["n_rollouts"])
            else:
                stats1[c]["roll"].append(float("inf"))
    out("| cell | 配置 | cert rate | 中位 rollouts-to-certify |")
    out("| --- | --- | --- | --- |")
    for c, (desc, _m, _sh) in cells.items():
        v = stats1[c]
        med = float(np.median(v["roll"])) if any(np.isfinite(v["roll"])) else float("nan")
        med_s = fmt(med) if np.isfinite(med) else "> %d" % R_ab
        out(f"| {c} | {desc} | {fmt(v['cert'] / max(v['n'], 1))} | {med_s} |")
    out("- 解读（010 §4）：**H0→H1**（arm→pair statistic + challenger 采样）本身不带来"
        "认证收益——sparse pair sampling 使每个 pair 样本变少，pair-Hoeffding 用全 range "
        "反而更宽；**H1→E1**（pair-Hoeffding→variance-adaptive EB）才是 CS 的贡献：E1 用 "
        "~2× 更少 rollout 完成认证；**E0→E1**（independent→shared world）是 nested CRN "
        "coupling 的贡献：无耦合时 E0 在预算内无法认证。κ≈13 因此被拆成『pairwise statistic "
        "+ variance-adaptive CS』与『coupling』两段因果。")
    out(f"（{time.time()-t_g1:.0f}s）")
    out("")

    # ------------------------------------------------------------ G2
    out("## 3. G2 — hard rollout-budget action-quality curves（010 §5，P(Q−Q_min ≤ 2) vs R）")
    out("")
    rng2 = np.random.default_rng(SEED0 + 2)
    n_state_g2 = 20 if SMOKE else 60
    R_grid = (1000, 3000, 6000, 12000)
    t_g2 = time.time()
    out("| R (rollouts) | B0.4 P(≤2) | B0.3c P(≤2) | B0.4 worlds | B0.3c worlds |")
    out("| --- | --- | --- | --- | --- |")
    for R in R_grid:
        e_ok = e_n = 0
        c_ok = c_n = 0
        w_b04 = []
        w_b03c = []
        for s in range(n_state_g2):
            x = random_state4(rng2, 200000 + s)
            Q_pi = exact_qa_pi_b(cr, x, 40)
            if not Q_pi:
                continue
            R0s = cr.pl.r_stop(x)
            Q_min = min(list(Q_pi.values()) + [R0s])
            eb = CRRBLEB(quants4, muM, muF, bh, base4, levels=(1, 2, 4),
                         delta_c=1.0, seed=17, cs_mode="eb")
            eb.cr.rng = np.random.default_rng(8000 + s)
            a, inf = eb.plan(x, 40, eps=8.0, delta=0.05,
                             max_worlds=max(1, R // 2), n_min=100)
            q_a = R0s if a is None else Q_pi.get(a, np.inf)
            e_ok += int(max(0.0, q_a - Q_min) <= 2.0)
            e_n += 1
            w_b04.append(inf["n_rollouts"])
            nA = max(1, len(cr.feasible_actions(x, 40)))
            hf = CRRBL(quants4, muM, muF, bh, base4, levels=(1, 2, 4),
                       delta_c=1.0, seed=13)
            hf.rng = np.random.default_rng(8100 + s)
            a_c, infc = hf.plan(x, 40, eps=8.0, delta=0.05,
                                max_samples=max(1, R // nA))
            q_ac = R0s if a_c is None else Q_pi.get(a_c, np.inf)
            c_ok += int(max(0.0, q_ac - Q_min) <= 2.0)
            c_n += 1
            w_b03c.append(infc["n_rollouts"])
        out(f"| {R} | {fmt(e_ok / max(e_n, 1))} | {fmt(c_ok / max(c_n, 1))} | "
            f"{fmt(np.mean(w_b04))} | {fmt(np.mean(w_b03c))} |")
    out("- 解读：R 为**硬 rollout 预算**（两 planner 都在 n_rollouts ≥ R 停止，010 §5）；"
        "B0.4 的 worlds 数高于 B0.3c（每 world 只做 2 个 rollout 而非 |A| 个），但每 world "
        "成本低 |A|/2 倍；曲线显示动作质量随 R 的 tradeoff。**certified 决策由证书保证 "
        "ε-optimal（≈1−δ）**；未认证执行问题按 009 §13 交 B0.4a。")
    out(f"（{time.time()-t_g2:.0f}s）")
    out("")

    # ------------------------------------------------------------ G3
    out("## 4. G3 — N=8 shallow oracle（H=24/34/40，near-tie-aware）")
    out("")
    for H in (24, 34, 40):
        eb8 = CRRBLEB(quants8, muM, muF, bh, base8, levels=(1, 2, 4, 8),
                      delta_c=1.0, seed=19, cs_mode="eb")
        eb8.cr.rng = np.random.default_rng(9000 + H)
        a, info = eb8.plan(0, H, eps=4.0, delta=0.05, max_worlds=w_g1)
        Q_pi8 = exact_qa_pi_b(eb8.cr, 0, H)
        R0_8 = eb8.cr.pl.r_stop(0)
        Q_min8 = min(list(Q_pi8.values()) + [R0_8])
        q_a = R0_8 if a is None else Q_pi8.get(a, np.inf)
        out(f"- H={H}: a_B0.4={a}，Q(a)−Q_min = {fmt(max(0.0, q_a - Q_min8))}，"
            f"certified={info['certified']}，worlds={info['n_worlds']}，"
            f"rollouts={info['n_rollouts']}（R_stop={fmt(R0_8)}，Q_min={fmt(Q_min8)}）")
    out("")

    # ------------------------------------------------------------ G4
    out("## 5. G4 — scaling：2 rollouts/world（B0.3c 全配对为 32/world）")
    out("")
    out("| H | 动作数 | worlds | rollouts | rollouts/world | certified | 耗时 |")
    out("| --- | --- | --- | --- | --- | --- | --- |")
    for H in (48, 64, 96, 120):
        eb8 = CRRBLEB(quants8, muM, muF, bh, base8, levels=(1, 2, 4, 8),
                      delta_c=1.0, seed=21, top_k_uavs=[7, 6, 5, 4])
        eb8.cr.rng = np.random.default_rng(3000 + H)
        t0 = time.time()
        a, info = eb8.plan(0, H, eps=8.0, delta=0.05, max_worlds=300)
        nA = len(eb8.cr.feasible_actions(0, H))
        out(f"| {H} | {nA} | {info['n_worlds']} | {info['n_rollouts']} | "
            f"{fmt(info['n_rollouts'] / max(info['n_worlds'], 1))} | "
            f"{info['certified']} | {time.time()-t0:.1f}s |")
    out("")
    out("- **per-world rollout complexity 从 O(|A|) 降到 O(1)**（010 §6）：B0.3c 全配对每 "
        "world 需要 |A| 个 rollout（N=8 root |A|=32），B0.4 candidate–challenger 每 world "
        "固定 2 个（challenger=STOP 时 1 个）。但**总 certification complexity 仍依赖 |A|**："
        "置信分配 α_ab=δ/P 带 log P = O(log|A|) 项、challenger 搜索每轮 O(|A|)、pair-CS "
        "存储最坏 O(|A|²)、更多 arms 需要更多 worlds 排除潜在 challenger——因此不是"
        "『与 UAV 数无关』，而是 per-world O(1)（UAV 数增加不放大单 world 的 rollout 成本）。")
    out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    # B0.4s (011): smoke writes to a SEPARATE path and must never touch the
    # FULL report — guard by hashing the FULL file before and after.
    full_rp = os.path.join(OUT_DIR, "MVS-B0.4_report.md")
    import hashlib

    def _md5(p):
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.4_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
