"""MVS-C C0+C1 validation（001 §二十六.0-§二十六.1 先行验证模块，独立于 G2 runner）。

C0 — Specification/Semantic Registration（005 §三 改名；文档/接口层断言；
  不触碰 G2 runner 代码 —— 001 §二十六：新架构不临场重构 legacy runner，
  C0 代码清理在 MVS-C package 层执行）：
  (a) 主 QoS 口径 = matched detection：P_FA≤α ∧ P_D≥P_D,max(α)−ε_D（默认
      α=0.05、ε_D=0.01，001 §三）；
  (b) 成本模型 link-aware：c_{i,r→r'}=b_{0,i}+d_i(r,r')（16+Δr 为 homogeneous
      special case：b_{0,i}=16、κ_i=1，001 §六）；
  (c) hard budget = frame-window C_{U2U}(ω)≤C_max^{frame}（001 §七：物理
      frame 约束，非 planner horizon）；
  (d) belief 单份 canonical z-state（001 §十九.3）；_decode_zs 用 planner.N
      （001 §十九.2，G2 硬编码 N_UAV 属待清理项，移交 MVS-C package）；
      sigmoid 用 log-sigmoid（001 §十九.4）。
  本模块对 (a)-(c) 做文本断言（README/MVS-C runner 关键短语存在性，
  005 §三：这是 semantic *registration*，不是 implementation closure），
  (d) 仅登记。

C1 — link-aware phase theorem（001 §十二/§二十六.1）数值验证：
  定理：g_{s,i}(b_i) = E[min{D_{s,i}−d_{2,i}, b_i}] = E[min{Y_{s,i}, b_i}],
        Y_{s,i}=D_{s,i}−d_{2,i},
        ∂⁺g(b)/∂b = P(Y_{s,i}>b)（离散右导数 = survival 概率），
        b*_{s,i} = inf{b≥0 : g_{s,i}(b)≥0}。
  三情形分类**统一采用 phase_boundary.py 的约定（013 §3，005 §四）**：
        A  E[Y]<0 ⇒ b*=∞（progressive 永不占优）；
        B  E[Y]=0 ⇒ b*=max{0, ess sup Y}（b≥b* 后 g(b)=0 持平，
           **不是** direct-dominates）；
        C  E[Y]>0 ⇒ 唯一有限 crossing（Y≥0 a.s. ⇒ b*=0）。
  （旧 run_mvsc01 曾把 "Y≥0 a.s. 且 E[Y]>0" 叫 B、把 "E[Y]=0 且
   P(Y<0)>0" 的 crossing 叫 C，与 phase_boundary 冲突——005 §四 已修正，
   本模块现在动态输出 case 标签，不再出现两套 A/B/C。）
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
    """离散 Y 分布（支撑-概率对），用于 C1 验证。

    三情形分类与 phase_boundary.py（013 §3）完全一致（005 §四 统一）：
      case() 返回 "A"/"B"/"C"：
        A  E[Y]<0      ⇒ b*=∞
        B  E[Y]=0      ⇒ b*=max{0, ess sup Y}（g(b) 在 b≥b* 持平，非占优）
        C  E[Y]>0      ⇒ 唯一有限 crossing（Y≥0 a.s. ⇒ b*=0）
    """

    def __init__(self, vals, probs):
        self.vals = np.asarray(vals, dtype=np.float64)
        self.probs = np.asarray(probs, dtype=np.float64)
        assert abs(self.probs.sum() - 1.0) < 1e-12
        assert len(self.vals) == len(self.probs)

    @property
    def ey(self):
        return float((self.vals * self.probs).sum())

    def case(self, eps=1e-12):
        """phase_boundary.py 约定（013 §3 / 005 §四）：A/B/C 按 E[Y] 符号。
        返回 ("A"|"B"|"C", 说明)。"""
        ey = self.ey
        if ey < -eps:
            return "A", "E[Y]<0 ⇒ b*=∞（progressive 永不占优）"
        if abs(ey) <= eps:
            return "B", "E[Y]=0 ⇒ b*=max{0, ess sup Y}（持平，非占优）"
        return "C", "E[Y]>0 ⇒ 唯一有限 crossing"

    def g(self, b):
        """g(b)=E[min{Y,b}]（闭式）。"""
        return float((np.minimum(self.vals, b) * self.probs).sum())

    def g_right_deriv(self, b):
        """∂⁺g(b)=P(Y>b)（离散右导数）。

        实现用**独立路径**（支撑排序 + 逆序累积求和），与
        `g_right_deriv_check` 的布尔掩码求和不是同一表达式——测试同时
        用两者对照，避免"实现与测试共用同一代码"的恒真式（005 §四：
        C1 验证强度需提升，不能 P(Y>b)=P(Y>b) 自证）。
        """
        order = np.argsort(self.vals)
        sv = self.vals[order]
        sp = self.probs[order]
        # P(Y > b) = 1 - P(Y <= b)：逆序累积
        idx = int(np.searchsorted(sv, b, side="right"))
        return float(sp[idx:].sum())

    def g_right_deriv_check(self, b):
        """独立对照：布尔掩码直接求和（与 g_right_deriv 不同代码结构）。"""
        return float(self.probs[self.vals > b].sum())

    def b_star(self):
        """b*=inf{b≥0:g(b)≥0}，按 phase_boundary.py 的 A/B/C 三情形解析判定。"""
        ey = self.ey
        if ey < 0.0:
            return float("inf")                      # 情形 A：E[Y]<0 永不占优
        # B/C 情形：E[Y]>=0。若 g(0)=E[min(Y,0)]=E[Y·1{Y<0}]<0 则需 crossing；
        # 否则 Y>=0 a.s. ⇒ g(0)=0 ⇒ b*=0。
        g0 = self.g(0.0)
        if g0 >= -1e-12:
            return 0.0                               # Y≥0 a.s.（B 或 C，b*=0）
        # g(0)<0<=E[Y]：唯一 crossing。g 逐段线性（斜率 P(Y>b)），在支撑点间二分。
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
    out("## C0 — Specification/Semantic Registration（001 §二十六.0 + §十九 文本断言）")
    out("")
    out("> **命名（005 §三）**：本检查是 **Specification/Semantic Registration**"
        "（文档与 runner 已登记新语义），**不是** implementation semantic "
        "closure——后者要求新语义进入实际 MVS-C architecture（cost 模型、"
        "link 接口、frame budget），属 C4/C5 的 package 层工作。")
    out("")
    here = os.path.dirname(os.path.abspath(__file__))
    readme_txt = io.open(os.path.join(here, "README.md"), encoding="utf-8").read()
    # MVS-C 主 runner 集合：旧 G2（机制 special case）+ C0/C1 验证 + C2.1
    # budget-aware + C3a migration（005 §三：新语义的登记目标是整个 MVS-C
    # runner 族，不能只看旧 G2）
    runner_paths = ["run_mvsb07g2.py", "run_mvsc01.py", "run_mvsc021.py",
                    "run_mvsc03a.py"]
    runner_txt = "\n".join(
        io.open(os.path.join(here, p), encoding="utf-8").read()
        for p in runner_paths)
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
        out(f"- **C0 {name}**：关键字 '{k1}'/'{k2}' 存在于 README/MVS-C runner"
            f"（{runner_paths}）→ {'PASS' if ok else 'FAIL'}。")

    # ------------------------------------------------- C1 phase theorem checks
    out("")
    out("## C1 — link-aware phase theorem（001 §十二/§二十六.1 数值验证）")
    out("")
    out("> **A/B/C 命名（005 §四 统一）**：与 `phase_boundary.py`（013 §3）完全"
        "一致——**A**：E[Y]<0 ⇒ b*=∞（progressive 永不占优）；**B**：E[Y]=0 ⇒ "
        "b*=max{0, ess sup Y}（b≥b* 后 g(b)=0 **持平**，非 direct 占优）；"
        "**C**：E[Y]>0 ⇒ 唯一有限 crossing（Y≥0 a.s. ⇒ b*=0）。旧版本把 "
        "“Y≥0 a.s. 且 E[Y]>0”叫 B、把 “E[Y]=0 且 P(Y<0)>0”的 crossing 叫 C，"
        "与 phase_boundary 冲突（005 §四 指出 Y={−2,1,3} 实际 E[Y]=−0.1 却标 "
        "“C: 非对称有限 crossing”，最后 b*=∞）——已修正，本模块动态输出 case。")
    out("")
    out("| 分布 | case | E[Y] | b* 解析 | b* 扫描 | g(0) | ∂⁺g(0)=P(Y>0) | ∂⁺g(b)≡P(Y>b) 检查 |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- |")

    checks = {
        # 情形 A：E[Y]<0 → b*=∞（progressive 永不占优）
        "Y={−3,−1,1}": StepPMF([-3.0, -1.0, 1.0], [0.4, 0.4, 0.2]),
        # 情形 B：E[Y]=0 → b*=max{0, ess sup Y}（013 原定理 Y={−1,1} 复现；
        #   b* 落在 ess sup=1 而非 0——持平点，不是 direct 占优，013 §3 B）
        "Y={−1,1}（013 原定理）": StepPMF([-1.0, 1.0], [0.5, 0.5]),
        # 情形 A：E[Y]<0 的非对称分布——005 §四 修正点：旧表误标 “C”，
        #   实际 E[Y]=−0.1<0 ⇒ b*=∞（A）
        "Y={−2,1,3}（E[Y]<0 非对称）": StepPMF([-2.0, 1.0, 3.0], [0.5, 0.3, 0.2]),
        # 情形 C：E[Y]>0 且 Y≥0 a.s. → b*=0（005 §四：旧表误叫 B）
        "Y={1,3}（E[Y]>0, Y≥0 a.s.）": StepPMF([1.0, 3.0], [0.5, 0.5]),
        # 情形 B：κ=1 homogeneous（link-aware 退化：16+Δr；E[Y]=0 ⇒ b*=ess sup=2）
        "Y={−2,2}（κ=1 退化）": StepPMF([-2.0, 2.0], [0.5, 0.5]),
    }

    ok_all = True
    for (name, pmf) in checks.items():
        ey = pmf.ey
        case, _note = pmf.case()
        bstar_a = pmf.b_star()
        bstar_s = pmf.b_star_scan()
        g0 = pmf.g(0.0)
        dg0 = pmf.g_right_deriv(0.0)
        surv0 = float(pmf.probs[pmf.vals > 0.0].sum())
        # ∂⁺g(b)≡P(Y>b) 检查：三条**独立路径**对照（005 §四 防恒真式）——
        #   (i)  g_right_deriv：排序+逆序累积；
        #   (ii) g_right_deriv_check：布尔掩码求和；
        #   (iii) 数值中心差分 (g(b+h)−g(b−h))/2h（h 小于相邻支撑最小间隙），
        #         在非原子点验证离散右导数 = survival。
        # 三者来自不同代码结构，任一实现回归都会让对照失败。
        deriv_ok = True
        for b in (-2.0, 0.0, 1.0, 2.0):
            d1 = pmf.g_right_deriv(b)
            d2 = pmf.g_right_deriv_check(b)
            deriv_ok &= abs(d1 - d2) < 1e-12
            deriv_ok &= abs(d1 - float(pmf.probs[pmf.vals > b].sum())) < 1e-12
        # 数值导数在非原子点（b = 支撑间隙中点）验证
        for b_mid in (0.5, 1.5, 2.5):
            if any(abs(b_mid - v) < 1e-12 for v in pmf.vals):
                continue          # 原子点：右导数不连续，跳过
            h = 0.05
            num = (pmf.g(b_mid + h) - pmf.g(b_mid - h)) / (2.0 * h)
            deriv_ok &= abs(num - pmf.g_right_deriv(b_mid)) < 1e-6
        # 解析 b* 与暴力扫描一致
        bstar_ok = (bstar_a == bstar_s) or (
            np.isinf(bstar_a) and np.isinf(bstar_s))
        ok = bstar_ok and deriv_ok and abs(dg0 - surv0) < 1e-12
        ok_all &= ok
        out(f"| {name} | **{case}** | {fmt(ey)} | {fmt(bstar_a)} | "
            f"{fmt(bstar_s)} | {fmt(g0)} | {fmt(dg0)}={fmt(surv0)} | "
            f"{'PASS' if deriv_ok else 'FAIL'} |")

    out("")
    out(f"> **C1 结论**：五个分布（含 013 原定理 Y={{−1,1}} 与 κ=1 退化、情形 "
        f"A/B/C 全分支）b* 解析-扫描一致；∂⁺g=survival 经**三条独立路径**"
        f"（排序逆序累积 / 布尔掩码 / 数值中心差分）对照成立 → "
        f"{'**PASS**' if ok_all else '**FAIL**'}（001 §十二/§二十六.1；"
        f"分类命名与 phase_boundary.py 统一（005 §四），derivative 检查已"
        f"脱离恒真式——007 审计修复）。")
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