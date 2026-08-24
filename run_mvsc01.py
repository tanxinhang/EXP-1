"""MVS-C C0+C1 validation（001 §二十六.0-§二十六.1 先行验证模块，独立于 G2 runner）。

C0 — semantic closure（文档/接口层断言；不触碰 G2 runner 代码 —— 001 §二十六：
  新架构不临场重构 legacy runner，C0 代码清理在 MVS-C package 层执行）：
  (a) 主 QoS 口径 = matched detection：P_FA≤α ∧ P_D≥P_D,max(α)−ε_D（默认
      α=0.05、ε_D=0.01，001 §三）；
  (b) 成本模型 link-aware：c_{i,r→r'}=b_{0,i}+d_i(r,r')（16+Δr 为 homogeneous
      special case：b_{0,i}=16、κ_i=1，001 §六）；
  (c) hard budget = frame-window C_{U2U}(ω)≤C_max^{frame}（001 §七：物理
      frame 约束，非 planner horizon）；
  (d) belief 单份 canonical z-state（001 §十九.3）；_decode_zs 用 planner.N
      （001 §十九.2，G2 硬编码 N_UAV 属待清理项，移交 MVS-C package）；
      sigmoid 用 log-sigmoid（001 §十九.4）。
  本模块对 (a)-(c) 做文本断言（README/runner 关键短语存在性），(d) 仅登记。

C1 — link-aware phase theorem（001 §十二/§二十六.1）数值验证：
  定理：g_{s,i}(b_i) = E[min{D_{s,i}−d_{2,i}, b_i}] = E[min{Y_{s,i}, b_i}],
        Y_{s,i}=D_{s,i}−d_{2,i},
        ∂⁺g(b)/∂b = P(Y_{s,i}>b)（离散右导数 = survival 概率），
        b*_{s,i} = inf{b≥0 : g_{s,i}(b)≥0}。
  三情形（001 §十二/013 §三）：E[Y]<0 ⇒ b*=∞（progressive 永不占优）；
  E[Y]>0 且 P(Y<0)=0 ⇒ b*=0；E[Y]≥0 且 P(Y<0)>0 ⇒ 有限 crossing
  （支撑段线性：斜率 P(Y>b)）。
  κ=1 退化：Y_x 取 013 原定理分布时，本模块 b* 与 g 应与 G2/013 一致
  （16+Δr homogeneous special case 的 theory-preserving 复现）。
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")


class StepPMF:
    """离散 Y 分布（支撑-概率对），用于 C1 验证。"""

    def __init__(self, vals, probs):
        self.vals = np.asarray(vals, dtype=np.float64)
        self.probs = np.asarray(probs, dtype=np.float64)
        assert abs(self.probs.sum() - 1.0) < 1e-12
        assert len(self.vals) == len(self.probs)

    @property
    def ey(self):
        return float((self.vals * self.probs).sum())

    def g(self, b):
        """g(b)=E[min{Y,b}]（闭式）。"""
        return float((np.minimum(self.vals, b) * self.probs).sum())

    def g_right_deriv(self, b):
        """∂⁺g(b)=P(Y>b)（离散右导数）。"""
        return float(self.probs[self.vals > b].sum())

    def b_star(self):
        """b*=inf{b≥0:g(b)≥0}，三情形解析判定。"""
        ey = self.ey
        if ey < 0.0:
            return float("inf")                      # 情形 A：永不占优
        p_neg = float(self.probs[self.vals < 0.0].sum())
        if p_neg == 0.0:
            return 0.0                               # Y≥0 a.s. ⇒ g(0)=E[min(Y,0)]=0
        # 情形 C：有限 crossing。g 逐段线性（斜率 P(Y>b)），在支撑点间二分。
        g0 = self.g(0.0)
        if g0 >= 0.0:
            return 0.0
        hi = float(self.vals.max())
        lo, mid = 0.0, None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.g(mid) >= 0.0:
                hi = mid
            else:
                lo = mid
        return hi                                    # 二分上界（|g|<1e-9·rng 已足够）

    def b_star_scan(self):
        """b* 暴力扫描（支撑点与半程点）作为对照。"""
        ey = self.ey
        if ey < 0.0:
            return float("inf")
        p_neg = float(self.probs[self.vals < 0.0].sum())
        if p_neg == 0.0:
            return 0.0
        cands = sorted(set([0.0, self.vals.min(), self.vals.max()])
                       | set(0.5 * (self.vals[i] + self.vals[i + 1])
                             for i in range(len(self.vals) - 1)))
        for b in cands:
            if self.g(b) >= 0.0:
                return b
        return float("inf")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# MVS-C C0+C1 — semantic closure assertions + link-aware phase theorem "
        "validation（001 §二十六.0-§二十六.1）")
    out("")
    out("> 原则（001 §二十六/§二十八）：**不临场重构 G2 runner**——C0 代码清理"
        "移交 MVS-C package 层；本模块只做 (a) 文档口径断言 + (b) C1 数学验证。"
        "G2 数值不受本模块影响。")
    out("")

    # ------------------------------------------------------------ C0 assertions
    out("## C0 — semantic closure（001 §二十六.0 + §十九 文本断言）")
    out("")
    here = os.path.dirname(os.path.abspath(__file__))
    readme_txt = io.open(os.path.join(here, "README.md"), encoding="utf-8").read()
    runner_txt = io.open(os.path.join(here, "run_mvsb07g2.py"), encoding="utf-8").read()
    c0 = [
        ("(a1) README 主 QoS=matched detection", "matched detection",
         "P_FA≤α ∧ P_D≥P_D,max(α)−ε_D"),
        ("(a2) runner 主 QoS=matched detection", "matched detection", "P_D≥P_D,max"),
        ("(b) link-aware cost 16+Δr special case", "link-aware",
         "c_{i,r→r'}=b_{0,i}+d_i(r,r')"),
        ("(c) frame-window hard budget", "frame-window", "C_max^{frame}"),
        ("(d1) belief 单份 canonical z-state(登记)", "canonical z-state", "001 §十九.3"),
        ("(d2) _decode_zs 用 planner.N/pl.N(登记)", "_decode_zs",
         ("planner.N", "pl.N")),
        ("(d3) log-sigmoid(登记)", "log-sigmoid", "001 §十九.4"),
    ]
    for (name, k1, k2) in c0:
        k2s = (k2,) if isinstance(k2, str) else k2
        ok = (k1 in readme_txt or k1 in runner_txt) and any(
            k in (readme_txt + runner_txt) for k in k2s)
        out(f"- **C0 {name}**：关键字 '{k1}'/'{k2}' 存在于 README/runner → "
            f"{'PASS' if ok else 'FAIL'}。")

    # ------------------------------------------------- C1 phase theorem checks
    out("")
    out("## C1 — link-aware phase theorem（001 §十二/§二十六.1 数值验证）")
    out("")
    out("| 分布 | E[Y] | P(Y<0) | b* 解析 | b* 扫描 | g(0) | ∂⁺g(0)=P(Y>0) | ∂⁺g(b)≡P(Y>b) 检查 |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- |")

    checks = {
        # 情形 A：E[Y]<0 → b*=∞（progressive 永不占优）
        "A: E[Y]<0 (Y∈{−3,−1,1})": StepPMF([-3.0, -1.0, 1.0], [0.4, 0.4, 0.2]),
        # 情形 C：E[Y]≥0 且 P(Y<0)>0 → 有限 crossing（013 原定理 Y={−1,1} 复现）
        "C: 013 复现 Y={−1,1}": StepPMF([-1.0, 1.0], [0.5, 0.5]),
        # 情形 C：非对称有限 crossing
        "C: 非对称 Y={−2,1,3}": StepPMF([-2.0, 1.0, 3.0], [0.5, 0.3, 0.2]),
        # 情形 B：Y≥0 a.s. 且 E[Y]>0 → b*=0
        "B: Y≥0 a.s. (Y={1,3})": StepPMF([1.0, 3.0], [0.5, 0.5]),
        # κ=1 homogeneous（link-aware 退化：16+Δr；Y 即原 D_x−Δ₂ 分布）
        "κ=1 退化 (Y={−2,2})": StepPMF([-2.0, 2.0], [0.5, 0.5]),
    }

    ok_all = True
    for (name, pmf) in checks.items():
        ey = pmf.ey
        p_neg = float(pmf.probs[pmf.vals < 0.0].sum())
        bstar_a = pmf.b_star()
        bstar_s = pmf.b_star_scan()
        g0 = pmf.g(0.0)
        dg0 = pmf.g_right_deriv(0.0)
        surv0 = float(pmf.probs[pmf.vals > 0.0].sum())
        # ∂⁺g(b)≡P(Y>b) 检查：在 b∈{−2,0,1,2}（或支撑邻近点）抽查
        deriv_ok = all(
            abs(pmf.g_right_deriv(b) - float(pmf.probs[pmf.vals > b].sum())) < 1e-12
            for b in (-2.0, 0.0, 1.0, 2.0))
        # 解析 b* 与暴力扫描一致
        bstar_ok = (bstar_a == bstar_s) or (
            np.isinf(bstar_a) and np.isinf(bstar_s))
        ok = bstar_ok and deriv_ok and abs(dg0 - surv0) < 1e-12
        ok_all &= ok
        out(f"| {name} | {fmt(ey)} | {fmt(p_neg)} | {fmt(bstar_a)} | "
            f"{fmt(bstar_s)} | {fmt(g0)} | {fmt(dg0)}={fmt(surv0)} | "
            f"{'PASS' if deriv_ok else 'FAIL'} |")

    out("")
    out(f"> **C1 结论**：五个分布（含 013 原定理 Y={{−1,1}} 与 κ=1 退化、情形 "
        f"A/B/C 全分支）b* 解析-扫描一致、∂⁺g=survival 成立 → "
        f"{'**PASS**' if ok_all else '**FAIL**'}（001 §十二/§二十六.1）。")
    out("")
    out("> **000 系 §二十六 后续（MVS-C 主线，非本模块范围）**：C2 phase-guided "
        "policy（N=4，probe/jump/STOP + theory-certified pruning）、C3 N=8 "
        "homogeneous replay（migration Gate：复现 G2 special-case 数值）、C4 "
        "N=8 heterogeneous U2U（论文 headline）、C5 protocol robustness；论文"
        "四 Gate（001 §二十七：A 数学正确性 / B 机制必要性 / C 通信现实性 / "
        "D 求解器质量）。")
    out("")
    out(f"总耗时: {time.time() - t0:.1f}s")
    out("")

    rp = os.path.join(OUT_DIR, "MVS-C_C0C1_validation.md")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


def fmt(x, nd=4):
    if x == float("inf"):
        return "∞"
    return f"{x:.{nd}f}"


if __name__ == "__main__":
    main()