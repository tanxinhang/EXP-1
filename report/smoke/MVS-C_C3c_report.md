# MVS-C C3c — Three-Layer Feasibility Frontier（advice/005.md §十九，SMOKE）

> **定位（005 §十九）**：把 'INFEASIBLE' 拆成三层明确判定——**L1 Physical**（最大 evidence 在预算内能否达 QoS）、**L2 Policy-class**（deterministic + randomized convex hull 能否达）、**L3 Controller-search**（当前有限 (ρ,η) 网格是否找到）。以后再看到 INFEASIBLE，可明确说是 physical / policy-family / registered-grid 哪一种。

> 协议：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup=16.0、H=96、QoS(P_FA≤0.12, P_MD≤0.4)；L1 用 MITM det-thr 精确 ROC（005 §九）；L2 用 Wilson U95（与正式 Gate 一致）；L3 引用 C3b 校准。N_CAL=60、N_TEST=120。

## L1. Constructive Physical Feasibility Certificate（010 §六：改名 + 诚实限定）

> **010 §六 改名**：L1 是 **constructive certificate**——枚举预算内可达的一组**构造型**最大 evidence 配置（4×8-bit / 4×4-bit 于最强 4 UAV、4×8-bit 于最弱 4 UAV）用 MITM 精确全融合 ROC 评估 QoS。**PASS 成立**；但若该构造 FAIL，**不能反推 physical infeasible**——除非真正求解 budgeted maximum-evidence oracle（010 §六；5×1-bit=85、混合分配 (8,8,4,2,…) 等大量组合未枚举）。

| 配置 | cost | P_D,max^det-thr | P_MD | ≤β | 判定 |
| --- | --- | --- | --- | --- | --- |
| 4 strongest UAVs 8-bit | 96.0 | 0.8928 | 0.1072 | 0.4000 | YES |
| 4 strongest UAVs 4-bit | 80.0 | 0.8907 | 0.1093 | 0.4000 | YES |
| 4 weakest UAVs 8-bit | 96.0 | 0.6367 | 0.3633 | 0.4000 | YES |

- **L1 判定**：构造型最大 evidence 配置中**存在**使 P_MD ≤ β 的配置 ⇒ **PHYSICALLY FEASIBLE (constructive)**（0.5s）。构造 FAIL 不蕴含 physical infeasible（010 §六）。

> 注：MITM 是 4-UAV 子集精确融合（H=96 最多 4×8-bit，其余 UAV 不发 = 0 evidence）；这与 C2.1 的 π_full 构造（4×8-bit=96=H）同构，是预算内可达的强全融合 evidence。

## L2. Policy-class feasibility（C3d 修正，010 §三/§四：**per-method** registered convex hull + 显式 mixture 认证）

> **010 §三 P0 修正**：旧 L2 把无 θ̂ 的方法（StaticProg）直接排除，且 mixture 跨方法混合——那不是要回答的问题。修正后 L2 对**每个方法自身**计算 conv{ v_θ^m : θ∈Θ_m }：只混合该方法自己的网格点；fractional-count Wilson U95 仅作**近似证据**（010 §四），进入的组合再由**全新 test worlds 上的显式 Bernoulli-λ mixture**（整数kfa/kmd + Wilson，n0/n1 分离）正式认证。

- **Phase-PJ (Proposed)**：deterministic 0/28，自身网格 mixture 也未进入 → registered-hull infeasible（自身网格 deterministic + mixture 均未进入 QoS——该注册策略族不可行）
- **Myopic-PJ**：deterministic 0/28，自身网格 mixture 也未进入 → registered-hull infeasible（自身网格 deterministic + mixture 均未进入 QoS——该注册策略族不可行）
- **Myopic-All**：deterministic 0/28，自身网格 mixture 也未进入 → registered-hull infeasible（自身网格 deterministic + mixture 均未进入 QoS——该注册策略族不可行）
- **Direct8**：deterministic 0/28，自身网格 mixture 也未进入 → registered-hull infeasible（自身网格 deterministic + mixture 均未进入 QoS——该注册策略族不可行）
- **StaticProg**：deterministic 0/28，自身网格 mixture 也未进入 → registered-hull infeasible（自身网格 deterministic + mixture 均未进入 QoS——该注册策略族不可行）

- **显式 Bernoulli-λ mixture 认证（010 §四，全新 test worlds）**：
  - 无合格方法：本 regime 无 "registered-hull feasible 但无 θ̂" 的方法（StaticProg 的 hull infeasible ⇒ 显式 mixture 无对象；010 §四 机制保留，本 regime 未触发）。
- **L2 判定**：per-method registered convex hull：见上表（每方法独立判定）；显式 mixture 认证如上述（57.2s）

## L3. Controller-search feasibility（005 §十九：有限 (ρ,η) 网格；C3d 修正 StaticProg 口径）

> 引用 C3b 校准：各方法自身 FEASIBLE 数 + θ̂。**StaticProg 修正**（010 §五）：ρ 不参与其策略 ⇒ 28 网格点只含 7 个唯一阈值策略，报告 **0/7 unique**；并**撤掉**"StaticProg 无可行点本身即 adaptive 必要性证据"表述（010 §五）——adaptive 必要性由 L2 的 per-method 判定（StaticProg 自身 hull/mixture 是否可行）支持。

- Phase-PJ (Proposed)：**∅（无 FEASIBLE）**；feasible 0/28；10/28 全停退化 ⇒ registered-grid infeasible
- Myopic-PJ：**∅（无 FEASIBLE）**；feasible 0/28；10/28 全停退化 ⇒ registered-grid infeasible
- Myopic-All：**∅（无 FEASIBLE）**；feasible 0/28；8/28 全停退化 ⇒ registered-grid infeasible
- Direct8：**∅（无 FEASIBLE）**；feasible 0/28；10/28 全停退化 ⇒ registered-grid infeasible
- StaticProg：**∅（无 FEASIBLE）**；feasible 0/28 = **0/7 unique threshold policies** ⇒ registered-grid infeasible（ρ 不参与策略，4ρ 重复 ⇒ 7 个唯一策略）
（0.0s）

## 结论

- **L1 Constructive Physical Feasibility Certificate**：PASS（构造存在：预算内配置达 QoS）——010 §六：构造 FAIL 不蕴含 physical infeasible。
- **L2 Policy-class（C3d per-method)**：per-method registered convex hull：见上表（每方法独立判定）；显式 mixture 认证如上述。
- **L3 Controller-search**：见上表（0/5 方法网格可行；StaticProg 按 **0/7 unique** 口径报告，010 §五）。

> **三层归因（005 §十九）**：L1 NO ⇒ 需 budgeted max-evidence oracle 才能判 physical infeasible；L1 YES + 某方法 L2 NO ⇒ 该方法的注册策略族 infeasible（改算法/策略族）；L1/L2 整体 YES + 某方法 L3 NO 且 L2 为 registered-hull feasible ⇒ 只是网格不够（010 §三：per-method 判定，不再跨方法混合）。当前：L1 PASS；L2 见上方 per-method 表；L3 见上表——不可行方法的 NO 是 registered-grid/policy 层，非物理层。

总耗时: 60.4s

