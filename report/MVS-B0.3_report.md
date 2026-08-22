# O-PEF MVS-B0.3 — CR-RBL: Confidence-Certified Rollout RBL

> 依据 `advice/006.md` §8-§17：base-rollout Q_a^πb + anytime Hoeffding 证书（δ_{a,n}=6δ/(π²|A|n²)）+ LUCB challenger + nested-evidence CRN。
> 生成时间: 2026-08-22 21:56:33   模式: FULL

> **理论边界（§8）**：第一版证书是相对 base policy πb 的：P(Q_â^πb ≤ min_a Q_a^πb + ε) ≥ 1−δ，**不是** 相对 V*；V*-certificate（CR-RBL+）留待后续。

## 1. G0 — anytime CI empirical coverage

- N=4 状态、固定动作: Q_true=134.1565；200 次独立 CI 构建，覆盖率 = 1.0000（理论下界 1−δ/|A| = 0.9917）→ **PASS**

## 2. G1 — N=4 exact oracle 对比（相对 base policy πb 的精确 Q）

- 状态数 600：P(a_CR = a_πb*) = 0.0883，E[Q^πb(a_CR) − Q^πb(a_πb*)] = 4.1799，E[Q*(a_CR) − V*]（次要）= 5.1822（62s）

## 3. G2 — N=8 shallow oracle（H=24/34/40 vs exact sparse planner）

- H=24: a_πb*=(7, 2), a_CR=(7, 2)，匹配（相对 πb）=True，a_star(V*)=(7, 2)，certified=False，samples=2032
- H=34: a_πb*=(7, 2), a_CR=(7, 8)，匹配（相对 πb）=False，a_star(V*)=(6, 1)，certified=False，samples=2032
- H=40: a_πb*=None, a_CR=(7, 8)，匹配（相对 πb）=False，a_star(V*)=(7, 4)，certified=False，samples=2032

## 4. G3 — PAC action certificate 经验违规率（N=4，相对 πb 的精确 Q）

- 认证触发的决策数 34（2-action 简化问题，ε=40）：经验违规率 = 0.0000（δ = 0.05）→ **PASS**
- 注：全动作集下的 anytime Hoeffding 证书非常保守（返回域 B(x) 大），在 ε 较小时很少触发——经验违规恒为 0（保守）；variance-adaptive empirical-Bernstein CS（CR-RBL-EB）是下一步的收紧方向（006 §12）

## 5. G4 — scalability（H=48/64/96/120，N=8，无全 cone）

| H | 动作数 | samples | certified | 耗时 |
| --- | --- | --- | --- | --- |
| 48 | 32 | 532 | False | 0.1s |
| 64 | 32 | 532 | False | 0.1s |
| 96 | 32 | 532 | False | 0.1s |
| 120 | 32 | 532 | False | 0.1s |

- CR-RBL 只采样决策相关动作，无 279^8 全表、无 reachable-cone 枚举——深 horizon 可运行（B0 精确递归在 H≥48 已不可行）

## 6. G5 — matched QoS 方向性演示（CR-RBL receding vs Adaptive Direct-8）

| 方法 | H | P_D @ P_FA=0.05 | E[B_radio] |
| --- | --- | --- | --- |
| CR-RBL (receding) | 48 | 0.7049 | 177.4000 |
| Adaptive Direct-8 | 48 | 0.6888 | 74.9700 |
| CR-RBL (receding) | 96 | 0.7049 | 164.9000 |
| Adaptive Direct-8 | 96 | 0.6888 | 74.9700 |

- 方向性演示（n=800，MC 噪声下）；matched-QoS 的正式比较需更大样本

总耗时: 217.2s
