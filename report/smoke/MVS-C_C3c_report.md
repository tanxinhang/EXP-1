# MVS-C C3c — Three-Layer Feasibility Frontier（advice/005.md §十九，SMOKE）

> **定位（005 §十九）**：把 'INFEASIBLE' 拆成三层明确判定——**L1 Physical**（最大 evidence 在预算内能否达 QoS）、**L2 Policy-class**（deterministic + randomized convex hull 能否达）、**L3 Controller-search**（当前有限 (ρ,η) 网格是否找到）。以后再看到 INFEASIBLE，可明确说是 physical / policy-family / registered-grid 哪一种。

> 协议：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup=16.0、H=96、QoS(P_FA≤0.12, P_MD≤0.4)；L1 用 MITM det-thr 精确 ROC（005 §九）；L2 用 Wilson U95（与正式 Gate 一致）；L3 引用 C3b 校准。N_CAL=60、N_TEST=120。

## L1. Physical feasibility（005 §十九：预算内最大 evidence 的 QoS）

| 配置 | cost | P_D,max^det-thr | P_MD | ≤β | 判定 |
| --- | --- | --- | --- | --- | --- |
| 4 strongest UAVs 8-bit | 96.0 | 0.8928 | 0.1072 | 0.4000 | YES |
| 4 strongest UAVs 4-bit | 80.0 | 0.8907 | 0.1093 | 0.4000 | YES |
| 4 weakest UAVs 8-bit | 96.0 | 0.6367 | 0.3633 | 0.4000 | YES |

- **L1 判定**：预算内存在最大 evidence 配置使 P_MD ≤ β ⇒ **PHYSICALLY FEASIBLE**（0.5s）。

> 注：MITM 是 4-UAV 子集精确融合（H=96 最多 4×8-bit，其余 UAV 不发 = 0 evidence）；这与 C2.1 的 π_full 构造（4×8-bit=96=H）同构，是预算内可达的最强全融合 evidence。

## L2. Policy-class feasibility（005 §十九：deterministic + randomized convex hull，Wilson U95 口径）

> 升级 C3a 的点估计 convex_hull_diag：mixture 的 violation 计数线性（kfa(λ)=λkfa_a+(1−λ)kfa_b，同一 worlds 决策独立），再对 mixture 计数用 **Wilson U95** 判定——与正式 Gate 一致（C3a 的点估计进入 ≠ U95 认证，007 审计指出）。

- 无 FEASIBLE θ̂ 的校准点（N_CAL=60 太小？）——L2 convex-hull 需 FULL N_CAL=600 才有意义；SMOKE 下本层标注 UNCERTAIN。
- **L2 判定**：UNCERTAIN（SMOKE 无校准点；FULL 才有 L2 判定）（60.6s）

## L3. Controller-search feasibility（005 §十九：有限 (ρ,η) 网格）

> 直接引用 C3b 校准：五方法各自 FEASIBLE 数（/28 网格）+ θ̂。若 L1/L2 YES 但某方法 L3 NO ⇒ 是 registered-grid 不够，不是物理不可行。

- Phase-PJ (Proposed)：**∅（无 FEASIBLE）**；feasible 0/28；10/28 全停退化 ⇒ registered-grid infeasible
- Myopic-PJ：**∅（无 FEASIBLE）**；feasible 0/28；10/28 全停退化 ⇒ registered-grid infeasible
- Myopic-All：**∅（无 FEASIBLE）**；feasible 0/28；8/28 全停退化 ⇒ registered-grid infeasible
- Direct8：**∅（无 FEASIBLE）**；feasible 0/28；10/28 全停退化 ⇒ registered-grid infeasible
- StaticProg：**∅（无 FEASIBLE）**；feasible 0/28 ⇒ registered-grid infeasible
（0.0s）

## 结论

- **L1 Physical**：PASS（物理可行）——预算内最大 evidence 达 QoS。
- **L2 Policy-class**：UNCERTAIN（SMOKE 无校准点；FULL 才有 L2 判定）。
- **L3 Controller-search**：见上表（0/5 方法网格可行，其余 registered-grid infeasible）。

> **三层归因（005 §十九）**：若三层都 YES ⇒ 机制层可行；L1 NO ⇒ physical infeasible（改预算/成本）；L1 YES + L2 NO ⇒ policy-family infeasible（改算法）；L1/L2 YES + L3 NO ⇒ registered-grid infeasible（扩网格）。当前：L1 YES、L2 视 mixture、L3 0/5 可行——不可行方法的 NO 是 registered-grid/policy 层，非物理层。

总耗时: 64.8s

