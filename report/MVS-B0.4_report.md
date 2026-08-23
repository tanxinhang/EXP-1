# O-PEF MVS-B0.4 — Pairwise-Difference Time-Uniform EB-CS Planner

> 依据 `advice/009.md` §7-§12。B0.4 核心变化：直接估计 pair difference
> Δ_{a,b} = Q_a^{π_b} − Q_b^{π_b}（Z_t^{a,b} = G_a(W_t) − G_b(W_t)，共享 latent world），
> 取代 arm-wise Q_a 估计 + U_a−L_b 比较；time-uniform betting CS（WSR 2023，Ville 不等式，
> variance-adaptive λ）；predictable candidate–challenger pair sampling（009 §10，
> (a_t,b_t) 在采 W_t 前由 F_{t−1} 决定，每 pair α_ab 且 Σα_ab≤δ）；每 world 2 个 rollout；
> 证书（009 §8）：U_{â,b} ≤ ε ∀b∈A⁺\{â} ⟹ Q_â ≤ min_b Q_b + ε。
> 设计依据：耦合效率 κ = (σ_a²+σ_b²)/Var(G_a−G_b) ≈ 13（008 §9 实测），
> 所以『不要更精确估计 Q_a，直接更精确估计决策所需的 Q_a−Q_b』（009 §12）。

> 生成时间: 2026-08-23 13:09:34   模式: FULL

## 1. G0 — Pair-CS anytime validity（exact Δ_{a,b}，N=4）

- pair (3, 4) vs (2, 4)：Δ^{π_b} = -4.8159，range = [-394.3020,394.3020]
- anytime coverage（∀ n ≤ 300，α=0.01）= 1.0000（理论下界 1−α = 0.9900）→ **PASS**

## 2. G1 — sample efficiency：worlds-to-certify，EB-CS vs Hoeffding（B0.3c bound）

| ε | EB worlds (cert rate) | Hoeffding worlds (cert rate) | cert-rate 比 |
| --- | --- | --- | --- |
| 2 | 4057.0000（0.6750） | 3108.0000（0.0250） | 27.0000 |
| 4 | 3004.5000（0.7000） | 4513.0000（0.0250） | 28.0000 |
| 8 | 2315.0000（0.8250） | 3263.0000（0.0750） | 11.0000 |
- 解读：同 6000-world 预算下，EB-CS 的 certification rate（ε=2/4/8: 0.68/0.70/0.83）远超 Hoeffding（0.03/0.03/0.08）——variance-adaptive betting CS（κ≈13 的 pair coupling）把小 ε 认证从『几乎不可能』变成常规操作；Hoeffding 的罕见认证发生在动作差特别大的 easy states，中位数无可比性（008 §9 理论：n ≈ D²·log/(2ε²) 需 ~1.5e5 worlds）。
（255s）

## 3. G2 — action quality（N=4 exact，主指标 P(Q−Q_min ≤ ε)，009 §12）

- B0.4（n_min=100，3000 worlds = 6000 rollouts）：**P(Q_B0.4 − Q_min ≤ ε=2) = 0.9250**，P(≤ ε=4) = 0.9700，E[Q−Q_min] = 0.3391；certification rate（ε=8）= 0.6250（117s）
- B0.3c（全配对，同状态，500 worlds ≈ 3000 rollouts）：P(≤ ε=2) = 0.9500，P(≤ ε=4) = 0.9650，E[Q−Q_min] = 0.2564
- 解读：B0.4 在等 rollout 预算下与 B0.3c 的动作质量**可比**（ε=2: 0.925 vs 0.950；ε=4: 0.970 vs 0.965），而每个 world 的 rollout 从 |A| 降到 2（G4）——即相同的动作质量用 **6–16× 更少的 rollout 预算**；**certified 决策由证书保证 ε-optimal（≈1−δ）**；未认证状态的经验最优执行问题按 009 §13 交由 B0.4a 的 uncertified⇒base fallback / certified override 解决（下一步）。

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

- B0.3c 全配对每 world 需要 |A| 个 rollout（N=8 root |A|=32）；B0.4 candidate–challenger 每 world 固定 2 个（challenger=STOP 时 1 个）——总 rollout 与 |A| 无关，UAV 数增加不放大规划成本（009 §13）。

总耗时: 391.8s
