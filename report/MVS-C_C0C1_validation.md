# MVS-C C0+C1 — semantic closure assertions + link-aware phase theorem validation（001 §二十六.0-§二十六.1）

> 原则（001 §二十六/§二十八）：**不临场重构 G2 runner**——C0 代码清理移交 MVS-C package 层；本模块只做 (a) 文档口径断言 + (b) C1 数学验证。G2 数值不受本模块影响。

## C0 — Specification/Semantic Registration（001 §二十六.0 + §十九 文本断言）

> **命名（005 §三）**：本检查是 **Specification/Semantic Registration**（文档与 runner 已登记新语义），**不是** implementation semantic closure——后者要求新语义进入实际 MVS-C architecture（cost 模型、link 接口、frame budget），属 C4/C5 的 package 层工作。

- **C0 (a1) README 主 QoS=matched detection**：关键字 'matched detection'/'P_FA≤α ∧ P_D≥P_D,max(α)−ε_D' 存在于 README/MVS-C runner（['run_mvsb07g2.py', 'run_mvsc01.py', 'run_mvsc021.py', 'run_mvsc03a.py']）→ PASS。
- **C0 (a2) runner 主 QoS=matched detection**：关键字 'matched detection'/'P_D≥P_D,max' 存在于 README/MVS-C runner（['run_mvsb07g2.py', 'run_mvsc01.py', 'run_mvsc021.py', 'run_mvsc03a.py']）→ PASS。
- **C0 (b) link-aware cost 16+Δr special case**：关键字 'link-aware'/'c_{i,r→r'}=b_{0,i}+d_i(r,r')' 存在于 README/MVS-C runner（['run_mvsb07g2.py', 'run_mvsc01.py', 'run_mvsc021.py', 'run_mvsc03a.py']）→ PASS。
- **C0 (c) frame-window hard budget**：关键字 'frame-window'/'C_max^{frame}' 存在于 README/MVS-C runner（['run_mvsb07g2.py', 'run_mvsc01.py', 'run_mvsc021.py', 'run_mvsc03a.py']）→ PASS。
- **C0 (d1) belief 单份 canonical z-state(登记)**：关键字 'canonical z-state'/'001 §十九.3' 存在于 README/MVS-C runner（['run_mvsb07g2.py', 'run_mvsc01.py', 'run_mvsc021.py', 'run_mvsc03a.py']）→ PASS。
- **C0 (d2) _decode_zs 用 planner.N/pl.N(登记)**：关键字 '_decode_zs'/'('planner.N', 'pl.N')' 存在于 README/MVS-C runner（['run_mvsb07g2.py', 'run_mvsc01.py', 'run_mvsc021.py', 'run_mvsc03a.py']）→ PASS。
- **C0 (d3) log-sigmoid(登记)**：关键字 'log-sigmoid'/'001 §十九.4' 存在于 README/MVS-C runner（['run_mvsb07g2.py', 'run_mvsc01.py', 'run_mvsc021.py', 'run_mvsc03a.py']）→ PASS。

## C1 — link-aware phase theorem（001 §十二/§二十六.1 数值验证）

> **A/B/C 命名（005 §四 统一）**：与 `phase_boundary.py`（013 §3）完全一致——**A**：E[Y]<0 ⇒ b*=∞（progressive 永不占优）；**B**：E[Y]=0 ⇒ b*=max{0, ess sup Y}（b≥b* 后 g(b)=0 **持平**，非 direct 占优）；**C**：E[Y]>0 ⇒ 唯一有限 crossing（Y≥0 a.s. ⇒ b*=0）。旧版本把 “Y≥0 a.s. 且 E[Y]>0”叫 B、把 “E[Y]=0 且 P(Y<0)>0”的 crossing 叫 C，与 phase_boundary 冲突（005 §四 指出 Y={−2,1,3} 实际 E[Y]=−0.1 却标 “C: 非对称有限 crossing”，最后 b*=∞）——已修正，本模块动态输出 case。

| 分布 | case | E[Y] | b* 解析 | b* 扫描 | g(0) | ∂⁺g(0)=P(Y>0) | ∂⁺g(b)≡P(Y>b) 检查 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Y={−3,−1,1} | **A** | -1.4000 | ∞ | ∞ | -1.6000 | 0.2000=0.2000 | PASS |
| Y={−1,1}（013 原定理） | **B** | 0.0000 | 1.0000 | 1.0000 | -0.5000 | 0.5000=0.5000 | PASS |
| Y={−2,1,3}（E[Y]<0 非对称） | **A** | -0.1000 | ∞ | ∞ | -1.0000 | 0.5000=0.5000 | PASS |
| Y={1,3}（E[Y]>0, Y≥0 a.s.） | **C** | 2.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000=1.0000 | PASS |
| Y={−2,2}（κ=1 退化） | **B** | 0.0000 | 2.0000 | 2.0000 | -1.0000 | 0.5000=0.5000 | PASS |

> **C1 结论**：五个分布（含 013 原定理 Y={−1,1} 与 κ=1 退化、情形 A/B/C 全分支）b* 解析-扫描一致；∂⁺g=survival 经**三条独立路径**（排序逆序累积 / 布尔掩码 / 数值中心差分）对照成立 → **PASS**（001 §十二/§二十六.1；分类命名与 phase_boundary.py 统一（005 §四），derivative 检查已脱离恒真式——007 审计修复）。

> **000 系 §二十六 后续（MVS-C 主线，非本模块范围）**：C2 phase-guided policy（N=4，probe/jump/STOP + theory-certified pruning）、C3 N=8 homogeneous replay（migration Gate：复现 G2 special-case 数值）、C4 N=8 heterogeneous U2U（论文 headline）、C5 protocol robustness；论文四 Gate（001 §二十七：A 数学正确性 / B 机制必要性 / C 通信现实性 / D 求解器质量）。

总耗时: 0.0s

