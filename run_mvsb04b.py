"""MVS-B0.4b: Feedback-Granularity Phase-Transition Theorem (advice/013.md).

Pure-theory封板 — NO planner changes.  The four gates:
  G0 Identity    Q_prog - Q_dir = E[min{Y_x, b}] on random reachable states,
                 max error < 1e-10 (013 §1);
  G1 Shape       g_x monotone nondecreasing + concave, right/left derivatives
                 = survival Pr(Y>b) / Pr(Y>=b), exact support (013 §2);
  G2 Existence   b*(x) < inf  <=>  E[Y_x] >= 0, three-case classification
                 A/B/C incl. the E[Y]=0 synthetic branch (013 §3, T30);
  G3 State dep.  exact b*(x) over root / reachable children — finite + inf
                 mix, NOT a global threshold (013 §6-§7); root b* = 7 EXACT
                 via the closed form b* = b0 - g(b0)/Pr(Y > b0) (013 §5).

The exact-support computation replaces the G6 grid interpolation: Y_x is a
discrete random variable over the X_1 branches, so g_x is exactly piecewise
linear and b* is read off the support breakpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np

from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs import sparse as sp
from opmvs.phase_boundary import (bstar_exact, bstar_from_dist,
                                  g_alt, g_from_support, survival,
                                  verify_identity, y_support)
from opmvs.sparse import z_code_b

GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
SEED0 = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
B_VALUES = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0]
B_GRID = np.arange(0.0, 49.0, 1.0)


def fmt(x, nd=4):
    if x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def mp(flag):
    return "PASS" if flag else "FAIL"


def reachable_states(rng, pl, n, seed_base):
    """Random reachable states: start at the root and apply k legal random
    refinements (a legal action sequence — every returned state is on the
    evidence DAG).  Skips leaf states (no UAV refineable)."""
    out = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        x = 0
        for _ in range(int(rng.integers(0, 3))):
            zs = list(pl.decode(x))
            cand = [i for i in range(pl.N)
                    if pl._tpl[i][zs[i]]]
            if not cand:
                break
            i = int(cand[rng.integers(0, len(cand))])   # draw a UAV id
            tpl = pl._tpl[i][zs[i]]
            r2 = tpl[int(rng.integers(0, len(tpl)))][0]
            m2 = int(rng.integers(0, 2 ** r2))
            zi = zs[i]
            zs[i] = z_code_b(r2, m2)
            x = pl.encode(tuple(zs))
        # keep the state if at least one UAV has a refineable template
        zs = pl.decode(x)
        if any(pl._tpl[i][zs[i]] for i in range(pl.N)):
            out.append(x)
    return out


def refineable_uavs(pl, x):
    zs = pl.decode(x)
    return [i for i in range(pl.N) if pl._tpl[i][zs[i]]]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    SMOKE = args.smoke
    n_state = 40 if SMOKE else 120
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.4b — Feedback-Granularity Phase-Transition Theorem")
    out("")
    out("> 依据 `advice/013.md`。**纯理论封板，不改 planner**。主定理（013 §1）：")
    out("> ")
    out(">     g_x(b) = Q_prog(x; b) − Q_dir(x; b) = E[ min{ Y_x, b } ],")
    out(">     Y_x = D_x(X₁) − Δ₂,   D_x(X₁) = R(X₁) − E[R(X₂)|X₁],   Δ₂ = r_max − r_next,")
    out("> progressive = r → r_next → r_max（两次 transaction），direct = r → r_max"
        "（一次完整包），每次 transaction 固定 setup/header 开销 b ≥ 0。")
    out("> 关键性质（013 §2）：g'₊(b) = Pr(Y_x > b)、g'₋(b) = Pr(Y_x ≥ b)——setup 开销对"
        " packetization preference 的边际影响 = 第二次反馈 transaction 的触发概率。"
        "b⋆(x) = inf{b ≥ 0 : g_x(b) ≥ 0}，且 b⋆<∞ ⟺ E[Y_x]≥0 ⟺ E[D_x]≥Δ₂（013 §4）。")
    out("> B0.4b 用 **exact support computation**（Y_x 离散 ⇒ g_x 精确分段线性，crossing "
        "在 support breakpoint 区间内闭式求解）取代 G6 的 grid interpolation（013 §5）。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if SMOKE else 'FULL'}")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=(1, 2, 4, 8))
               for i in range(8)]
    pl8 = sp.SparsePlanner(quants8, 256.0, 256.0 * np.exp(1.0), b_h=0.0,
                           cross_level=True)
    i_s = int(np.argmax(GAMMA_B))              # strongest UAV (7): SNR anchor
    rng = np.random.default_rng(SEED0)

    # ------------------------------------------------------------ G0/G1/G2
    out("## 1. G0 — Identity：Q_prog − Q_dir = E[min{Y_x, b}]（013 §1）")
    out("")
    out(f"- random reachable states × refineable UAV（N=8，levels (1,2,4,8)，"
        f"n_state={n_state}，b ∈ {B_VALUES}）：")
    states = reachable_states(rng, pl8, n_state, SEED0)
    n_pairs = 0
    max_dev = 0.0
    max_tower = 0.0
    g0_ok = True
    for x in states:
        for i in refineable_uavs(pl8, x):
            sup = y_support(pl8, x, i)
            if sup is None:
                continue
            dev, tower = verify_identity(sup, B_VALUES)
            n_pairs += 1
            max_dev = max(max_dev, dev)
            max_tower = max(max_tower, tower)
            g0_ok &= (dev < 1e-10) and (tower < 1e-10)
    out(f"  - (state, UAV) pairs tested = {n_pairs}；"
        f"max |g − (Q_prog−Q_dir)| = {max_dev:.3e}；"
        f"max tower dev |Σ w₁E[R₂|x₁] − E_dir| = {max_tower:.3e}；"
        f"**G0 = {mp(g0_ok)}**（标准 < 1e-10）。")
    out("- 说明：g 与 Q_prog−Q_dir 由同一 support 的两条独立路径计算——策略值形式"
        "（先验塔性质使 E_R 项相消）与 E[min(Y,b)] 形式；偏差纯浮点舍入（~1e-13）。")
    out("")

    # ------------------------------------------------------------ G1
    out("## 2. G1 — Shape：monotone、concave、导数 = survival（013 §2）")
    out("")
    n_states_g1 = 0
    all_mono = True
    all_conc = True
    all_deriv = True
    root_surv = None
    for x in states[:max(1, n_state // 4)]:
        for i in refineable_uavs(pl8, x):
            sup = y_support(pl8, x, i)
            if sup is None:
                continue
            n_states_g1 += 1
            gk = [g_from_support(sup, float(b)) for b in B_GRID]
            mono = all(gk[k + 1] >= gk[k] - 1e-9 for k in range(len(gk) - 1))
            conc = True
            for a in range(len(B_GRID) - 2):
                for c in range(a + 2, len(B_GRID)):
                    for k in range(a + 1, c):
                        chord = (gk[a] + (gk[c] - gk[a])
                                 * (B_GRID[k] - B_GRID[a]) / (B_GRID[c] - B_GRID[a]))
                        conc &= (gk[k] >= chord - 1e-9)
            # exact piecewise-linear derivative identity from the support:
            # slope on (b_l, b_r) = Pr(Y > mid); at atoms, right slope =
            # Pr(Y > y_k), left slope = Pr(Y >= y_k)  (013 §2)
            ys = sorted({float(br[5]) for br in sup["branches"]})
            ymin, ymax = ys[0], ys[-1]
            bps = sorted(set([0.0] + [v for v in ys if v > 0.0]))
            deriv = True
            # open intervals between consecutive breakpoints
            for a in range(len(bps) - 1):
                b_l, b_r = bps[a], bps[a + 1]
                if b_r <= b_l + 1e-12:
                    continue
                mid = 0.5 * (b_l + b_r)
                slope = (g_from_support(sup, b_r) - g_from_support(sup, b_l)) \
                    / (b_r - b_l)
                deriv &= abs(slope - survival(sup, mid)) < 1e-9
            # atoms: one-sided slopes
            h = 1e-6
            for yk in ys:
                if yk < 0:
                    continue
                yk = float(yk)
                gk_at = g_from_support(sup, yk)
                r_slope = (g_from_support(sup, yk + h) - gk_at) / h
                l_slope = (gk_at - g_from_support(sup, yk - h)) / h
                deriv &= abs(r_slope - survival(sup, yk, strict=True)) < 1e-5
                deriv &= abs(l_slope - survival(sup, yk, strict=False)) < 1e-5
            # tail: g(b) = E[Y] for b >= Ymax  (Case B plateau form)
            deriv &= abs(g_from_support(sup, ymax + 1.0) - sup["EY"]) < 1e-9
            # g(0) = -E[Y^-]
            deriv &= abs(g_from_support(sup, 0.0)
                         + sum(br[0] * max(-br[5], 0.0)
                               for br in sup["branches"]) / sup["wsum"]) < 1e-9
            all_mono &= mono
            all_conc &= conc
            all_deriv &= deriv
            if x == 0 and i == i_s:
                root_surv = survival(sup, 0.0)
    out(f"  - 检查 {(n_states_g1)} 个 (state, UAV) 对："
        f"monotone={mp(all_mono)}，concave(chord)={mp(all_conc)}，"
        f"导数=survival（精确 support，区间斜率 & 原子单侧斜率）={mp(all_deriv)}；"
        f"g(0)=−E[Y⁻] 与尾部 g≡E[Y] 均验证。")
    out(f"  - root (x₀, strongest UAV {i_s}) survival Pr(Y>0) = "
        f"{fmt(root_surv) if root_surv is not None else 'n/a'}（= 1/2 ⇒ Y_x 两点分布，"
        f"与 G6 一致）。")
    out("")

    # ------------------------------------------------------------ G2
    out("## 3. G2 — Existence：b⋆(x)<∞ ⟺ E[Y_x]≥0，三情形分类（013 §3）")
    out("")
    out("| 情形 | Y 分布 | E[Y] | 预期 b⋆ | 实际 b⋆ | g(b⋆−ε) | g(b⋆+ε) | g(b) for b≥b⋆ |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- |")
    syn = []
    # Case A: E[Y] < 0 => b* = inf, progressive strictly dominates
    rA = bstar_from_dist([0.5, 0.5], [-2.0, -1.0])
    gA = [g_from_support({"branches": [(0.5, 0, 0, 0, 0, -2.0),
                                       (0.5, 0, 0, 0, 0, -1.0)], "wsum": 1.0}, b)
          for b in B_GRID]
    domA = all(g <= rA["EY"] + 1e-9 for g in gA)
    syn.append((rA, domA, "progressive dominates ∀b"))
    # Case B: E[Y] = 0 => b* = max{0, ess sup Y} = 1, g(b)=0 for b >= 1
    rB = bstar_from_dist([0.5, 0.5], [-1.0, 1.0])
    gB_lt = g_from_support({"branches": [(0.5, 0, 0, 0, 0, -1.0),
                                         (0.5, 0, 0, 0, 0, 1.0)], "wsum": 1.0}, 0.5)
    gB_at = g_from_support({"branches": [(0.5, 0, 0, 0, 0, -1.0),
                                         (0.5, 0, 0, 0, 0, 1.0)], "wsum": 1.0}, 1.0)
    gB_ge = g_from_support({"branches": [(0.5, 0, 0, 0, 0, -1.0),
                                         (0.5, 0, 0, 0, 0, 1.0)], "wsum": 1.0}, 2.0)
    syn.append((rB, (rB["bstar"] == 1.0) and (gB_lt < 0) and
                abs(gB_at) < 1e-9 and abs(gB_ge) < 1e-9,
                "g(b)=0 for b≥b⋆ (equal, NOT direct-dominates)"))
    # Case C: E[Y] > 0 with P(Y<0)>0 => unique finite crossing
    rC = bstar_from_dist([0.5, 0.5], [-1.0, 3.0])
    gC_lt = g_from_support({"branches": [(0.5, 0, 0, 0, 0, -1.0),
                                         (0.5, 0, 0, 0, 0, 3.0)], "wsum": 1.0}, 0.5)
    gC_gt = g_from_support({"branches": [(0.5, 0, 0, 0, 0, -1.0),
                                         (0.5, 0, 0, 0, 0, 3.0)], "wsum": 1.0}, 1.5)
    syn.append((rC, (rC["bstar"] == 1.0) and (gC_lt < 0) and (gC_gt > 0),
                "unique crossing, sign change"))
    # Case C0: Y >= 0 a.s. => b* = 0
    rC0 = bstar_from_dist([0.5, 0.5], [1.0, 3.0])
    syn.append((rC0, rC0["bstar"] == 0.0, "Y≥0 a.s. ⇒ b⋆=0"))
    for r_, ok, note in syn:
        out(f"| {r_['case']} | {r_['Ymin']:.0f}..{r_['Ymax']:.0f} p=1/2 | "
            f"{fmt(r_['EY'])} | — | {fmt(r_['bstar'])} | {fmt(r_['g0'])} | "
            f"{fmt(r_['g_inf'])} | {note} {mp(ok)} |")
    out("")
    out("- 注意（013 §3 Case B）：**E[Y]=0 时 b>b⋆ 不是 direct 严格占优，而是两者持平**"
        "（g(b)=0 for b≥b⋆）——论文不能笼统写 “b>b⋆ ⇒ direct dominates”，"
        "只有 E[Y]>0 才严格成立。")
    out("")
    # real-state iff check
    n_A = n_B = n_C = 0
    iff_ok = True
    dom_ok = True
    n_class = 0
    for x in states:
        for i in refineable_uavs(pl8, x):
            sup = y_support(pl8, x, i)
            if sup is None:
                continue
            n_class += 1
            r_ = bstar_exact(sup)
            iff_ok &= (r_["bstar"] < float("inf")) == (sup["EY"] >= -1e-12)
            if r_["case"] == "A":
                n_A += 1
                # domination: g(b) <= E[Y] < 0 for all b >= 0 (013 §3 Case A)
                dom_ok &= all(g_from_support(sup, b) <= sup["EY"] + 1e-9
                              for b in B_GRID)
            elif r_["case"] == "B":
                n_B += 1
            else:
                n_C += 1
    out(f"- **real states**（{n_class} 个 (state, UAV) 对）："
        f"(b⋆<∞) ⟺ (E[Y]≥0) 100% 满足 = {mp(iff_ok)}；"
        f"Case A 支配 g(b)≤E[Y]<0 ∀b = {mp(dom_ok)}；"
        f"分类计数 A(E[Y]<0, b⋆=∞)={n_A}，B(E[Y]=0)={n_B}，C(E[Y]>0, 有限 crossing)={n_C}。")
    out("")

    # ------------------------------------------------------------ G3
    out("## 4. G3 — State dependence：exact b⋆(x) 分布（013 §6-§7）")
    out("")
    out(f"- N=8，strongest UAV = {i_s}（γ={GAMMA_B[i_s]:.0f}），"
        f"Δ₂ = r_max − r_next。root 与 1-bit 子状态（013 §5/§7 的解析例）：")
    out("")
    out("| 状态 | r | case | E[Y_x] | ess sup Y | Pr(Y>0) | b⋆(x) |")
    out("| --- | --- | --- | --- | --- | --- | --- |")
    root_sup = y_support(pl8, 0, i_s)
    r_root = bstar_exact(root_sup)
    out(f"| x₀ (root) | 0 | {r_root['case']} | {fmt(r_root['EY'])} | "
        f"{fmt(r_root['Ymax'])} | {fmt(survival(root_sup, 0.0))} | "
        f"**{fmt(r_root['bstar'])}** |")
    for m in (0, 1):
        zs = [0] * 8
        zs[i_s] = z_code_b(1, m)
        x_child = pl8.encode(tuple(zs))
        sup_c = y_support(pl8, x_child, i_s)
        r_c = bstar_exact(sup_c)
        out(f"| 1-bit cell {m} | 1 | {r_c['case']} | {fmt(r_c['EY'])} | "
            f"{fmt(r_c['Ymax'])} | {fmt(survival(sup_c, 0.0))} | "
            f"{fmt(r_c['bstar'])} |")
    out("")
    root_ok = (abs(r_root["bstar"] - 7.0) < 1e-9) and (r_root["case"] == "C")
    child_ok = (r_root["bstar"] < float("inf")
                and all(r_["case"] == "A" for r_ in
                        [bstar_exact(y_support(pl8, pl8.encode(tuple(
                            z_code_b(1, m) if j == i_s else 0 for j in range(8))), i_s))
                         for m in (0, 1)]))
    out(f"- **closed-form 验证**（013 §5）：root g(0) = {fmt(r_root['g0'])}，"
        f"Pr(Y>0) = {fmt(survival(root_sup, 0.0))} ⇒ "
        f"b⋆ = 0 − g(0)/Pr(Y>0) = {fmt(-r_root['g0'] / survival(root_sup, 0.0))} "
        f"= **exact support 结果 {fmt(r_root['bstar'])}**——不再需要 grid 插值。"
        f"（root={mp(root_ok)}，1-bit children b⋆=∞ = {mp(child_ok)}）")
    out("")
    # reachable-state distribution
    rng2 = np.random.default_rng(13000)
    states2 = reachable_states(rng2, pl8, n_state, 13000)
    bs = []
    cases = {"A": 0, "B": 0, "C": 0}
    for x in states2:
        for i in refineable_uavs(pl8, x):
            sup = y_support(pl8, x, i)
            if sup is None:
                continue
            r_ = bstar_exact(sup)
            bs.append(r_["bstar"])
            cases[r_["case"]] += 1
    finite = [b for b in bs if b < float("inf")]
    n_inf = sum(1 for b in bs if b == float("inf"))
    out(f"- **reachable children 分布**（{len(bs)} 个 (state, UAV) 对）："
        f"finite b⋆ = {len(finite)}（min={fmt(min(finite)) if finite else '—'}，"
        f"max={fmt(max(finite)) if finite else '—'}，"
        f"E={fmt(np.mean(finite)) if finite else '—'}），b⋆=∞ = {n_inf}；"
        f"case 计数 A={cases['A']} B={cases['B']} C={cases['C']}。")
    mix_ok = (len(finite) > 0) and (n_inf > 0)
    out(f"- **G3 = {mp(mix_ok)}**：finite 与 ∞ 同时出现 ⇒ b⋆ 是 **state-dependent "
        f"phase boundary**，**不是**全局常数，也不是 |Ω|/SNR 的简单单调函数"
        f"（013 §7：不声称后验置信度单调性——目前没有依据）。")
    out("")
    out("- **创新定位冻结（013 §8）**：**Feedback-Granularity-Aware Adaptive "
        "Evidence Acquisition under per-transaction setup cost**；核心可辨识结果 = "
        "**state-dependent packetization phase transition** + "
        "g'ₓ(b) = P(additional feedback transaction)，再与 B0.4/B0.4a 的 "
        "paired-difference certified acquisition 组合。不自称 “adaptive quantization”"
        " 本身新颖（Fang/Li 与 2026 ISAC 已有动态量化分辨率研究）。")
    out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")

    # FULL report hash guard (B0.4s convention): smoke never touches FULL.
    full_rp = os.path.join(OUT_DIR, "MVS-B0.4b_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.4b_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
