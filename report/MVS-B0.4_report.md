# O-PEF MVS-B0.4/B0.4r — Pairwise-Difference EB-CS Planner（credibility patch）

> 依据 `advice/009.md` §7-§12 与 `advice/010.md` B0.4r。B0.4 核心：直接估计 Δ_{a,b} = Q_a^{π_b} − Q_b^{π_b}（Z_t^{a,b} = G_a(W_t) − G_b(W_t)，共享 latent world），取代 arm-wise Q_a + U_a−L_b；predictable candidate–challenger pair sampling（009 §10，(a_t,b_t) 由 F_{t−1} 决定，每 pair α_ab 且 Σα_ab≤δ）；每 world 2 个 rollout；证书（009 §8）：U_{â,b} ≤ ε ∀b∈A⁺\{â} ⟹ Q_â ≤ min_b Q_b + ε。
> **B0.4r（010）**：R0 canonical sample+support 同向（PairCS 硬断言 z∈[lo,hi]，descending top-k regression）；R1 **formal 证书路径 = predictable plug-in empirical-Bernstein CS**（Maurer–Pontil 2009 Thm 6 + peeling union bound，连续区间、无 grid inversion），betting grid CS 降级为实验性消融（不再承担主证书）；R3 G1 四格消融（H0/H1/E1/E0）分离 pair statistic / CS / coupling 三段因果；R4 G2 改为硬 rollout 预算曲线；R5 G4 表述修正（per-world O(1)，total 仍随 |A|）。
> 设计依据：耦合效率 κ = (σ_a²+σ_b²)/Var(G_a−G_b) ≈ 13（008 §9 实测），
> 所以『不要更精确估计 Q_a，直接更精确估计决策所需的 Q_a−Q_b』（009 §12）。

> 生成时间: 2026-08-23 15:15:40   模式: FULL

## 1. G0 — Pair-CS anytime validity（exact Δ_{a,b}，N=4；formal EB + betting 消融）

- PrPl-EB (formal, 连续区间)：Δ^{π_b} = -4.8159，range = [-394.3020,394.3020]；anytime coverage（∀ n ≤ 300，α=0.01）= 1.0000（下界 1−α = 0.9900）→ **PASS**
- betting (experimental 消融)：Δ^{π_b} = -4.8159，range = [-394.3020,394.3020]；anytime coverage（∀ n ≤ 300，α=0.01）= 1.0000（下界 1−α = 0.9900）→ **PASS**
- 注（010 §2）：formal 证书路径 = Maurer–Pontil (2009) Thm 6 + peeling union bound（连续区间、无 grid inversion）；betting grid CS 仅作实验性收紧消融。

## 2. G1 — 四格消融：pair statistic × CS × coupling（010 §4，ε=40，rollout 预算）

| cell | 配置 | cert rate | 中位 rollouts-to-certify |
| --- | --- | --- | --- |
| H0 | arm-wise/Hoeffding/full | 0.3333 | > 24000 |
| H1 | pair/Hoeffding/challenger | 0.2667 | > 24000 |
| E1 | pair/EB/challenger/shared | 0.9333 | 17288.0000 |
| E0 | pair/EB/challenger/independent | 0.3333 | > 24000 |
- 解读（010 §4）：**H0→H1**（arm→pair statistic + challenger 采样）本身不带来认证收益——sparse pair sampling 使每个 pair 样本变少，pair-Hoeffding 用全 range 反而更宽；**H1→E1**（pair-Hoeffding→variance-adaptive EB）才是 CS 的贡献：E1 用 ~2× 更少 rollout 完成认证；**E0→E1**（independent→shared world）是 nested CRN coupling 的贡献：无耦合时 E0 在预算内无法认证。κ≈13 因此被拆成『pairwise statistic + variance-adaptive CS』与『coupling』两段因果。
（63s）

## 3. G2 — hard rollout-budget action-quality curves（010 §5，P(Q−Q_min ≤ 2) vs R）

| R (rollouts) | B0.4 P(≤2) | B0.3c P(≤2) | B0.4 worlds | B0.3c worlds |
| --- | --- | --- | --- | --- |
| 1000 | 0.6000 | 0.8167 | 891.4500 | 996.6500 |
| 3000 | 0.7500 | 0.9667 | 2713.1833 | 2998.1167 |
| 6000 | 0.8833 | 0.9667 | 5313.2833 | 5998.1500 |
| 12000 | 0.8833 | 0.9833 | 10923.3667 | 11997.5667 |
- 解读：R 为**硬 rollout 预算**（两 planner 都在 n_rollouts ≥ R 停止，010 §5）；B0.4 的 worlds 数高于 B0.3c（每 world 只做 2 个 rollout 而非 |A| 个），但每 world 成本低 |A|/2 倍；曲线显示动作质量随 R 的 tradeoff。**certified 决策由证书保证 ε-optimal（≈1−δ）**；未认证执行问题按 009 §13 交 B0.4a。
（91s）

## 4. G3 — N=8 shallow oracle（H=24/34/40，near-tie-aware）

- H=24: a_B0.4=(7, 4)，Q(a)−Q_min = 1.0862，certified=False，worlds=6000，rollouts=12000（R_stop=256.0000，Q_min=202.3553）
- H=34: a_B0.4=(7, 2)，Q(a)−Q_min = 0.0000，certified=False，worlds=6000，rollouts=12000（R_stop=256.0000，Q_min=202.3553）
- H=40: a_B0.4=(7, 8)，Q(a)−Q_min = 0.0000，certified=False，worlds=6000，rollouts=12000（R_stop=256.0000，Q_min=206.9019）

## 5. G4 — scaling：2 rollouts/world（B0.3c 全配对为 32/world）

| H | 动作数 | worlds | rollouts | rollouts/world | certified | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | 16 | 300 | 600 | 2.0000 | False | 0.1s |
| 64 | 16 | 300 | 600 | 2.0000 | False | 0.1s |
| 96 | 16 | 300 | 600 | 2.0000 | False | 0.1s |
| 120 | 16 | 300 | 600 | 2.0000 | False | 0.1s |

- **per-world rollout complexity 从 O(|A|) 降到 O(1)**（010 §6）：B0.3c 全配对每 world 需要 |A| 个 rollout（N=8 root |A|=32），B0.4 candidate–challenger 每 world 固定 2 个（challenger=STOP 时 1 个）。但**总 certification complexity 仍依赖 |A|**：置信分配 α_ab=δ/P 带 log P = O(log|A|) 项、challenger 搜索每轮 O(|A|)、pair-CS 存储最坏 O(|A|²)、更多 arms 需要更多 worlds 排除潜在 challenger——因此不是『与 UAV 数无关』，而是 per-world O(1)（UAV 数增加不放大单 world 的 rollout 成本）。

总耗时: 172.6s
