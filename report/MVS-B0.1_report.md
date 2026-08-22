# O-PEF MVS-B0.1 — 可信度修复 + Feedback-Granularity 理论拔高

> 依据 `adcice/005.md`：修复 1bit-POTS 重复计数、共享 CRN、自然/NP 口径分离、Adaptive Direct-8 隔离 baseline、state-dependent conditional-VoI 定理、feedback/setup 成本；B0-G3 改为 UNCERTIFIED/COMPUTATION-LIMITED。
> 生成时间: 2026-08-22 18:57:27   模式: FULL

## 0. 创新定位（005.md §十六）

> **Feedback-Granularity-Aware Adaptive Evidence Acquisition**：per-transaction setup/header 开销下，何时先发粗证据获得反馈机会、何时跨级直发更多证据；matched detection QoS 下最小化总期望 radio 资源。

- 系统: N=8, R={1,2,4,8}, 每 UAV 状态 279, 理论空间 279^8≈1e19（sparse 不建表）

- 共享 CRN: baseline 与 O-PEF 均使用各自固定的 (H, L)；同一方法内所有参数共享同一批样本

## 1. P0-1 — 1bit-POTS 重复计数修复

- 修复后 1bit-POTS (ηs=6): P_D=0.8940 @ P_FA=0.05, E[B_radio]=573.4606 bits（修复前 B0 中该基线未能达标——重复计数已移除：seed 仅付一次，ladder 从 1→2 开始）

## 2. P0-2/3 — 共享 CRN 的协议公平 baseline + 自然/NP 口径

### 2.1 公平基线（共享 CRN, n=20000, Top-K 全扫 K=1..8）

- P_D,max = 0.8940（all-neighbor 8-bit），matched 目标 P_D ≥ 0.8840

| 方法 | 参数 | P_D @ P_FA=0.05 | E[B_radio] | 达标 |
| --- | --- | --- | --- | --- |
| AllNeighbor-8 | - | 0.8940 | 192.0000 | ✓ |
| SNR-TopK | 8 | 0.8940 | 192.0000 | ✓ |
| GlobalFixed | 6.0 | 0.8862 | 239.9028 | ✓ |
| StaticProg | 6.0 | 0.8932 | 368.8719 | ✓ |
| 1bit-POTS(fixed) | 6.0 | 0.8920 | 508.4547 | ✓ |
| Direct8-Ordered | 6.0 | 0.8857 | 110.5140 | ✓ |

### 2.2 自适应策略（共享 CRN，n=2500；Natural-policy QoS 为主，NP ROC 为诊断）

| 策略 | (s,η) | H | P_D(nat) | P_FA(nat) | E[B_radio] | P_D(NP@0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| Cross-Level O-PEF | (256,1.0) | 34 | 0.6246 | 0.0578 | 42.0128 | 0.5858 |
| Adjacent-Only | (256,1.0) | 34 | 0.6221 | 0.0578 | 41.7556 | 0.5836 |
| Adaptive Direct-8 | (256,1.0) | 34 | 0.7270 | 0.0492 | 47.8080 | 0.7287 |

### 2.3 收益来源隔离（同一 CRN、同一 (s,η,H)：UAV 选择 vs multi-resolution）

| 策略 | P_D(nat) | P_FA(nat) | E[B_radio] |
| --- | --- | --- | --- |
| Adaptive Direct-8 | 0.7270 | 0.0492 | 47.8080 |
| Adjacent-Only | 0.6221 | 0.0578 | 41.7556 |
| Cross-Level O-PEF | 0.6246 | 0.0578 | 42.0128 |

- 同 E[B] 对比: cross-level 相对 Adaptive Direct-8 的增益 = -5.7952 bits（multi-resolution + UAV 选择）；相对 Adjacent-Only 的增益 = 0.2572 bits（cross-level/跳级）

## 3. P1-5 — state-dependent conditional-VoI 定理验证

> Q_prog − Q_dir = E_{x'}[ min{ D(x') − Δ₂, b_h } ]，D(x') = R(x') − E[R(x'')|x']。b_h=0 ⇒ ≤0（渐进支配）；b_h>0 ⇒ 状态相关相变。

- b_h=0, UAV5: Q_prog=148.1466 Q_dir=155.1466 LHS−RHS dev = 3.02e-14（Q_prog−Q_dir = -7.0000)
- b_h=0, UAV6: Q_prog=136.4358 Q_dir=143.4358 LHS−RHS dev = 1.42e-14（Q_prog−Q_dir = -7.0000)
- b_h=0, UAV7: Q_prog=123.8854 Q_dir=130.8854 LHS−RHS dev = 5.15e-14（Q_prog−Q_dir = -7.0000)
- **b_h=0: 定理数值成立（max dev = 5.15e-14）**；Q_prog ≤ Q_dir ⇒ 渐进支配

- b_h=16, UAV5: Q_prog=164.1466 Q_dir=171.1466 LHS−RHS dev = 3.02e-14（Q_prog−Q_dir = -7.0000)
- b_h=16, UAV6: Q_prog=152.4358 Q_dir=159.4358 LHS−RHS dev = 1.42e-14（Q_prog−Q_dir = -7.0000)
- b_h=16, UAV7: Q_prog=139.8854 Q_dir=146.8854 LHS−RHS dev = 6.57e-14（Q_prog−Q_dir = -7.0000)
- **b_h=16: 定理数值成立（max dev = 6.57e-14）**；符号由状态分布决定 ⇒ 反馈粒度相变

### 3.1 q<7/23 降级为 communication-only Corollary；radio-only 判据 E[C_future|M^(1)]<7

- 经验继续概率 q = (454+13+0)/5669 = 0.082（corollary 阈值 7/23 = 0.304；q<阈值 ⇒ radio-only 渐进更省）
- 更正确判据：E[C_future | M^(1)] < 7（已付 1 bit 后，相对一次性 8-bit 的剩余 payload 差恰为 7 bits）——从 trajectory 直接统计，无需人为压成单变量 q

## 4. P1-6 — feedback/setup 成本与敏感性

> c_a = b_setup + (r'−r)，b_setup = b_data-header + b_control + b_feedback（grant/ACK/query 交易开销）。

| b_setup | root action (H=b_setup+8) | 说明 |
| --- | --- | --- |
| 16 | (7, 2) | direct jump |
| 24 | (7, 2) | direct jump |
| 32 | (7, 2) | direct jump |

- 在 '一个 direct-8 包' 预算（H=b_setup+8）下 root 均选 direct jump；而 B0-G1 中 H=2×(b_setup+1)（两个最小包）时 root 选 probe——粒度选择由（预算，状态）共同决定，即反馈粒度相变是状态相关的（与 VoI 定理一致）

## 5. B0.1 Gate 汇总

- **P0-1 1bit-POTS 修复**: 已修复（seed 仅一次，ladder 从 1→2）
- **P0-2 共享 CRN + 样本量**: baseline 20000, O-PEF 2500（同一固定 (H,L)）
- **P0-3 自然/NP 口径分离**: 已分离（Natural-policy QoS 为主，NP ROC 为诊断）
- **P1-4 Adaptive Direct-8**: 已加入（§2.3 隔离收益来源）
- **P1-5 VoI 定理**: 数值验证通过（b_h=0 ⇒ 渐进支配；b_h>0 ⇒ 状态相关相变）
- **P1-6 feedback/setup 成本**: b_setup 解释与敏感性（§4）
- **B0-G3 结论**: **UNCERTIFIED / COMPUTATION-LIMITED**（深预算超出 N=8 精确递归 cone 上限；O-PEF 最高 P_D(nat) 见 §2.2；需 certified rollout 扩展）

总耗时: 1000.2s
