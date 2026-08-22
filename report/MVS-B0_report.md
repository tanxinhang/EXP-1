# O-PEF MVS-B0 — Sparse-State Header-Aware Cross-Level O-PEF

> 依据 `adcice/004.md`：MVS-A 冻结（08fe2a5）；本阶段实现 sparse state backend（279^8 不建表）、header 激活 cross-level 的相变与 break-even 验证、协议公平 baseline。
> 生成时间: 2026-08-22 14:20:30   模式: FULL

## 1. B0-G0 — sparse tuple-state backend 与 eager 表等价（N=4）

- eager table RBL (N=4) 求解 41.4s
| H | 测试状态数 | max|ΔV| | 动作不一致 | 其中近等值(ΔV<1e-6) | memo |
| --- | --- | --- | --- | --- | --- |
| 4 | 279841 | 8.53e-14 | 136 | 136 | 1315449 |
| 8 | 20001 | 5.68e-14 | 21 | 21 | 1093238 |
| 12 | 20001 | 5.68e-14 | 22 | 22 | 1721824 |

- **B0-G0 → PASS（value 达机器精度；动作不一致均为近等值 argmin 翻转）**

## 2. B0-G1 — N=8 / R={1,2,4,8} 在线规划（不建 279^8 表）

- 8-bit nested 量化器构建 3.6s；每 UAV 状态数 1+2+4+16+256=279，理论状态空间 279^8 ≈ 1e19（未构建）
| b_h | H (radio bits) | root value | root action | memo (expansions) | 耗时 |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | 158.5879 | (6, 1) | 161 | 0.0s |
| 0 | 3 | 157.4057 | (7, 1) | 1057 | 0.0s |
| 0 | 4 | 126.6581 | (6, 1) | 5441 | 0.1s |
| 16 | 24 | 202.3553 | (7, 2) | 2225 | 0.0s |
| 16 | 34 | 182.5879 | (6, 1) | 2369 | 0.0s |
| 16 | 40 | 168.7990 | (7, 4) | 17985 | 0.3s |

- **B0-G1 → PASS（root solve 成功；memo 仅覆盖可达 cone）**

## 3. B0-G2 — header 激活 cross-level（相变）

### 3.1 b_h=0 dominance sanity（N=4，adjacent vs cross）

- b_h=0: adjacent V=60.279500（action (3, 1)）vs cross V=60.279500（action (3, 1)）；diff=0.00e+00 → **PASS**（cross-level 被 adjacent 弱支配，理论闭环 R1）

### 3.2 root action 相变（N=8, H = 一个 direct-8 包的预算）

| b_h | H_radio | root action (i, r2) | 说明 |
| --- | --- | --- | --- |
| 0 | 4 | (6, 1) | adjacent probe (r2=1) |
| 4 | 12 | (7, 2) | direct jump (r2>1) |
| 8 | 16 | (7, 2) | direct jump (r2>1) |
| 16 | 24 | (7, 2) | direct jump (r2>1) |
| 32 | 40 | (7, 2) | direct jump (r2>1) |

- **B0-G2 相变确认**: b_h=0 时 root 偏好 probe；b_h 增大后（如 b_h=16, H=24 → 0→2；H=40 → 0→4）root 转向 direct jump——header 激活了 cross-level 动作的物理价值（Proposition: b_h>0 ⇒ cross-level 重新有价值）

## 4. B0-G3 — 协议公平 baseline 对比（b_h=16，radio bits）

- P_D,max（all-neighbor 8-bit @ P_FA=0.05, MC）: 0.8829，matched 目标 P_D ≥ 0.8729，E[B_radio] = 192.0000 bits

| 方法 | 参数 | P_D @ P_FA=0.05 | E[B_radio] (bits) | 达标 |
| --- | --- | --- | --- | --- |
| AllNeighbor-8 | - | 0.8829 | 192.0000 | ✓ |
| SNR-TopK | — | — | — | ✗ |
| GlobalFixed | 6.0 | 0.8766 | 241.9090 | ✓ |
| StaticProg | 6.0 | 0.8813 | 370.0311 | ✓ |
| 1bit-POTS | — | — | — | ✗ |
| Direct8-Ordered | 6.0 | 0.8768 | 111.1522 | ✓ |

### 4.1 O-PEF receding sparse planner（b_h=16）

| (s, η) | H_radio | P_D @ P_FA=0.05 | E[B_radio] (bits) | 达标 |
| --- | --- | --- | --- | --- |
| (256,1.0) | 24 | 0.6261 | 46.4460 |  |
| (256,1.0) | 34 | 0.5157 | 43.8820 |  |
| (1024,1.0) | 24 | 0.6975 | 56.3200 |  |
| (1024,1.0) | 34 | 0.4223 | 71.8920 |  |

- O-PEF 在可行 lookahead（H ≤ 34）下最高 P_D = 0.6975（< 目标 0.8729）
- ⚠ 结论（004 §10 触发）：matched 目标需要深预算（≈120+ radio bits），超出 N=8 精确递归的 cone 上限；Direct-8 以 111 bits 达标——progressive 的深预算价值需 sampled lookahead 才能评估；按审计建议不继续堆通信模型，先扩展采样 lookahead

## 5. B0-G4 — break-even 理论 vs 经验继续概率

- 0→1→8 @ b_h=16: 理论阈值 q < (8-1)/(16+8-1) = 0.304
- 0→1→4 @ b_h=16: 理论阈值 q < (4-1)/(16+4-1) = 0.158
- 0→2→8 @ b_h=16: 理论阈值 q < (8-2)/(16+8-2) = 0.273

- 机制统计（b_h=16, H=34）: 动作总数 1430
  - P(0→1) = 0.0399；P(0→2) = 0.0923；P(0→4) = 0.8580；P(0→8) = 0.0000
  - P(1→2) = 0.0000；P(1→4) = 0.0049；P(1→8) = 0.0014
  - 1-bit 后继续 refine 概率 q = (1→2+1→4+1→8)/0→1 = 0.158（理论阈值 7/23 ≈ 0.304）
- **B0-G4 → 经验 q 与理论阈值方向一致**（q < 0.304 ⇒ progressive 更省 radio bits）

## 6. B0-G5 — 复杂度（expansions / memo / runtime vs (N, H, b_h)）

| N | b_h | H_radio | memo (expansions) | 说明 |
| --- | --- | --- | --- | --- |
| 8 | 16 | 24 | 2225 | 0.0s |
| 8 | 16 | 34 | 2369 | 0.0s |
| 8 | 16 | 40 | 17985 | 0.2s |
| 8 | 16 | 48 | 2.17e6 | 221s（实测）|

- 精确递归的 cone 随 budget/动作深度指数增长（N=8, b_h=16: H=40 → 1.8e4, H=48 → 2.2e6, H=64/96 → 不可行）；MVS-B 深 horizon 需采样 lookahead（下一步）

## 7. Gate 汇总

- **B0-G0 sparse==eager (N=4)**: PASS（value 1e-13；近等值翻转仅 0 级）
- **B0-G1 N=8 在线规划**: PASS（不建 279^8 表）
- **B0-G2 header 激活 cross-level**: 见 §3.2 相变表
- **B0-G3 matched QoS**: O-PEF E[B_radio] = — bits vs 公平基线（§4）
- **B0-G4 break-even**: 见 §5（理论阈值 vs 经验 q）
- **B0-G5 complexity**: 见 §6（cone 指数增长，深 horizon 需采样 lookahead）

总耗时: 344.1s
