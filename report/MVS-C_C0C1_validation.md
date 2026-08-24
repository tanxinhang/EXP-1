# MVS-C C0+C1 — semantic closure assertions + link-aware phase theorem validation（001 §二十六.0-§二十六.1）

> 原则（001 §二十六/§二十八）：**不临场重构 G2 runner**——C0 代码清理移交 MVS-C package 层；本模块只做 (a) 文档口径断言 + (b) C1 数学验证。G2 数值不受本模块影响。

## C0 — semantic closure（001 §二十六.0 + §十九 文本断言）

- **C0 (a1) README 主 QoS=matched detection**：关键字 'matched detection'/'P_FA≤α ∧ P_D≥P_D,max(α)−ε_D' 存在于 README/runner → PASS。
- **C0 (a2) runner 主 QoS=matched detection**：关键字 'matched detection'/'P_D≥P_D,max' 存在于 README/runner → PASS。
- **C0 (b) link-aware cost 16+Δr special case**：关键字 'link-aware'/'c_{i,r→r'}=b_{0,i}+d_i(r,r')' 存在于 README/runner → PASS。
- **C0 (c) frame-window hard budget**：关键字 'frame-window'/'C_max^{frame}' 存在于 README/runner → PASS。
- **C0 (d1) belief 单份 canonical z-state(登记)**：关键字 'canonical z-state'/'001 §十九.3' 存在于 README/runner → PASS。
- **C0 (d2) _decode_zs 用 planner.N/pl.N(登记)**：关键字 '_decode_zs'/'('planner.N', 'pl.N')' 存在于 README/runner → PASS。
- **C0 (d3) log-sigmoid(登记)**：关键字 'log-sigmoid'/'001 §十九.4' 存在于 README/runner → PASS。

## C1 — link-aware phase theorem（001 §十二/§二十六.1 数值验证）

| 分布 | E[Y] | P(Y<0) | b* 解析 | b* 扫描 | g(0) | ∂⁺g(0)=P(Y>0) | ∂⁺g(b)≡P(Y>b) 检查 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: E[Y]<0 (Y∈{−3,−1,1}) | -1.4000 | 0.8000 | ∞ | ∞ | -1.6000 | 0.2000=0.2000 | PASS |
| C: 013 复现 Y={−1,1} | 0.0000 | 0.5000 | 1.0000 | 1.0000 | -0.5000 | 0.5000=0.5000 | PASS |
| C: 非对称 Y={−2,1,3} | -0.1000 | 0.5000 | ∞ | ∞ | -1.0000 | 0.5000=0.5000 | PASS |
| B: Y≥0 a.s. (Y={1,3}) | 2.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000=1.0000 | PASS |
| κ=1 退化 (Y={−2,2}) | 0.0000 | 0.5000 | 2.0000 | 2.0000 | -1.0000 | 0.5000=0.5000 | PASS |

> **C1 结论**：五个分布（含 013 原定理 Y={−1,1} 与 κ=1 退化、情形 A/B/C 全分支）b* 解析-扫描一致、∂⁺g=survival 成立 → **PASS**（001 §十二/§二十六.1）。

> **000 系 §二十六 后续（MVS-C 主线，非本模块范围）**：C2 phase-guided policy（N=4，probe/jump/STOP + theory-certified pruning）、C3 N=8 homogeneous replay（migration Gate：复现 G2 special-case 数值）、C4 N=8 heterogeneous U2U（论文 headline）、C5 protocol robustness；论文四 Gate（001 §二十七：A 数学正确性 / B 机制必要性 / C 通信现实性 / D 求解器质量）。

总耗时: 0.0s

