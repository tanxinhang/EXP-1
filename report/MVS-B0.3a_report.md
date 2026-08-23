# O-PEF MVS-B0.3a/B0.3c — CR-RBL credibility + closure patch

> 依据 `advice/007.md` §1-§7, §10-§12 与 `advice/008.md` §1-§6。相对 B0.3（`9d877d0`）的修复：
> **P0-A** 真正跨 action CRN：每次 MC 迭代采样一个 latent world W_m=(H_m, M_1^(8), …, M_N^(8))|x，所有候选动作在同一 W_m 上求值（G_a(W_m)）。实现中发现两个关键点：(i) world 必须包含隐假设 H_m（先按后验采样 H，再按 H-条件分布采样各 UAV cell）——逐 UAV 边缘独立采样会破坏 H 诱导的跨 UAV 相关性，导致 rollout 估计**系统性偏高 ~12 bits**（对 N=4 exact oracle 验证到 1e-6）；(ii) 修正后 paired-CRN 的方差比 Var(G_a−G_b)/[Var(G_a)+Var(G_b)] ≈ 0.08–0.15，相对独立采样降低 7–12 倍；
> **P0-B** certificate 竞争集 A⁺(x)=A(x)∪{STOP}，STOP 用精确值 R_stop(x)=min{C₀₁p, C₁₀(1−p)}；
> **P0-C** G0 oracle 改为 Q_a^{π_b}（exact_qa_pi_b），不再是 Q_a^⋆；
> **P0-D** G1/G3 的 STOP oracle 改为 R_stop(x)，不再是 base_policy_value；
> **P0-E** G5 硬预算语义：h_t = H − C_t pathwise，保证 C_T ≤ H；
> **P1-A/B/C/D** anytime coverage gate、binomial U95 violation gate + certification rate、回归不变量 T15-T20、b⋆(x₀) root-state 理论表述。
> **B0.3c（008）**：natural 阈值 η_nat=log(μ_F/μ_M)=1.0（T21 锁死）；G5 改名 directional (unmatched) comparison；joint-H vs pairing 三格消融（G7）；Hoeffding range 收紧 B→B_a(x,h)=min{c_max_rem,h}+R_max−c_a（G1/G3/G4 重测）；T17 拆分（T17a 确定性 + T17b 经验审计）；E[Y_x] 存在性判据（G6，b⋆<∞⟺E[Y]≥0）。

> 生成时间: 2026-08-23 13:20:45   模式: FULL

## 1. G0 — anytime CI coverage（oracle = Q_a^{π_b}，P1-A）

- 固定 N=4 状态 x0、目标动作 (3, 4)（π_b 最优之一），Q_true^{π_b} = 136.9329；n_max=200，|A|=12，δ=0.1；Hoeffding diameter（B0.3c）= 394.3020（旧 loose bound = 582.3020）
- anytime coverage（∀ n ≤ 200 均覆盖）= 1.0000（理论下界 1−δ/|A| = 0.9917）→ **PASS**；固定 n=200 coverage = 1.0000（次诊断）

## 2. G1 — N=4 exact oracle（best = argmin{ R_stop(x), Q_a^{π_b} }，P0-D）

- 状态数 600：P(a_CR = a_{π_b}⋆) = 0.8700，E[Q^{π_b}(a_CR) − min{R_stop, Q^{π_b}}] = 0.4489，E[Q*(a_CR) − V*]（次要，V* 续值）= 1.1848；certification rate（ε=2，诊断）= 0.0000，其中 match = 0.0000（24s）
- 注：B0.3 同口径为 match=0.088 / gap=4.18；paired-CRN 全配对估计（含 H_m 隐假设，见头部 P0-A）大幅改善。B0.3c 将 Hoeffding range 收紧为 D_a(x,h)=min{c_max_rem,h}+R_max−c_a（008 §4），certification rate 随 range 收紧重新测量（G3 用 ε=40 测证书本身）。

## 3. G2 — N=8 shallow oracle（H=24/34/40 vs exact sparse planner）

- H=24: a_{π_b}⋆=(7, 2)（R_stop=256.0000），a_CR=(7, 4)，匹配（相对 π_b）= False，a_star(V*)=(7, 2)，certified=False，worlds=250，rollouts=8000
- H=34: a_{π_b}⋆=(7, 2)（R_stop=256.0000），a_CR=(7, 4)，匹配（相对 π_b）= False，a_star(V*)=(6, 1)，certified=False，worlds=250，rollouts=8000
- H=40: a_{π_b}⋆=(7, 8)（R_stop=256.0000），a_CR=(7, 8)，匹配（相对 π_b）= True，a_star(V*)=(7, 4)，certified=False，worlds=250，rollouts=8000
- 注：H=24/34 时 (7,2) 与 (7,4) 的精确 Q^{π_b} 差仅 1.08 bits（near-tie），MC 估计在其噪声内，属合理近似；H=40 的 a_πb⋆=(7,8) 已被匹配。

## 4. G3 — certified-violation gate（binomial U95，P1-B）

- 2-action 问题（UAV 3: 1→2 vs 1→4 + STOP），ε=40，δ=0.05，max_worlds=2000
- tested=500，certified=492，certification rate = 0.9840，violations=0
- 经验违规率 = 0.0000；单侧 95% binomial 上界 U95(p_viol) = 0.0061 ≤ δ=0.05 → **PASS**
- 注：证书同时包含 STOP 竞争项（P0-B）；0-violation 时需约 ≥59 个 certified 样本才能让 U95 ≤ 0.05（007 §12）。
- **B0.3c range 收紧的效果（008 §4）**：ε=40 下 certification rate 由 B0.3a（loose bound）的 0.93 升至 0.984；但 G1（ε=2）与 G4（ε=4）仍为 0——原因是所需 gap ≈ 2·rad−ε 在这些 ε 下远超动作间真实 Q 差，瓶颈是 sample complexity 而非 bound；收紧 bound 只是部分解锁证书（008 §4 的 0%→20% 预期在 ε=40 口径下成立），ε 小的场景仍需 B0.4 的 variance-adaptive EB-CS。

## 5. G4 — scalability（N=8，H=48/64/96/120，无全 cone）

| H | 动作数 | worlds | rollouts | certified | 耗时 |
| --- | --- | --- | --- | --- | --- |
| 48 | 32 | 150 | 4800 | False | 0.1s |
| 64 | 32 | 150 | 4800 | False | 0.1s |
| 96 | 32 | 150 | 4800 | False | 0.1s |
| 120 | 32 | 150 | 4800 | False | 0.1s |

- 每 world 全动作配对求值：总 rollout 数 = worlds × |A|；无 279^8 全表。

## 6. G5 — directional (unmatched) hard-budget operating-point comparison

| 方法 | H | P_D^NP | P_FA^NP | P_D^nat(η=1) | P_FA^nat(η=1) | E[B] | SE(E[B]) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CR-RBL (receding) | 48 | 0.4555 | 0.0500 | 0.6339 | 0.0922 | 36.4300 | 0.2243 |
| Adaptive Direct-8 | 48 | 0.5437 | 0.0500 | 0.6967 | 0.0899 | 41.0700 | 0.3848 |
| CR-RBL (receding) | 96 | 0.6257 | 0.0500 | 0.7787 | 0.0899 | 54.3162 | 0.8184 |
| Adaptive Direct-8 | 96 | 0.6530 | 0.0500 | 0.8251 | 0.0876 | 59.4300 | 1.0275 |

- **B0.3c（008 §1/§2）**：natural 判决阈值改为 η_nat = log(μ_F/μ_M) = 1.0000（与 eval_exact.py 的 objective-consistent natural decision 锁死，T21）；G5 是 **directional (unmatched) hard-budget comparison**：两个 operating point （P_D^NP、E[B]）同时不同，P_D^{CR} < P_D^{D8} 且 E[B]^{CR} < E[B]^{D8}，只有 Pareto 方向性、不是 matched-QoS 通信 gain；正式比较（CI 口径）留待 B0.6。
- H 是 episode **硬通信预算**（h_t = H − C_t pathwise，C_T ≤ H 已断言）。n=800。（36s）

## 7. G6 — state-dependent phase boundary b⋆(x)（P1-D，007 §4）

- 理论：g_x(b) = E[min{Y_x, b}]，Y_x = D(x') − Δ₂；b ↦ min(Y, b) 非减凹 ⇒ g_x 非减凹；无原子点 g_x'(b) = Pr(Y_x > b)；b⋆(x) = inf{ b : g_x(b) ≥ 0 } 为 state-dependent packetization phase boundary：b_h < b⋆(x) 时小 packet 渐进细化划算，b_h > b⋆(x) 时 setup 开销主导、应减少反馈次数提高单次粒度。

- x0 (root): g(b) 序列 = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 12.5, 20.5]；monotone=True，concave=True；g'(8)=0.500 vs Pr(Y>8)=0.500；g'(16)=0.500 vs Pr(Y>16)=0.500（Δ=2 中心差分）；b⋆(x) = 7.0000；E[Y_x] = 38.5594，ess sup Y = 84.1187（b⋆<∞ ⟺ E[Y_x]≥0：PASS；E[Y]<0 ⇒ g(b)≤E[Y]<0 ∀b：PASS；g≡E[Y] 平台（需 ess sup Y≤0）：False）
  - 根状态 b⋆(x₀) = 7.0000（线性插值；B0.1a 网格口径 ≈8）——与 B0.1a root-state 结论一致；且 g 恰为线性、survival≡0.5 ⇒ Y_x 为两点分布（≈-7.0000 以概率 1/2，>48 以概率 1/2），packetization phase transition 有精确解析结构。

- x: UAV7 at 1-bit cell 0: g(b) 序列 = [-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]；monotone=True，concave=True；g'(8)=0.000 vs Pr(Y>8)=0.000；g'(16)=0.000 vs Pr(Y>16)=0.000（Δ=2 中心差分）；b⋆(x) = inf；E[Y_x] = -6.0000，ess sup Y = -6.0000（b⋆<∞ ⟺ E[Y_x]≥0：PASS；E[Y]<0 ⇒ g(b)≤E[Y]<0 ∀b：PASS；g≡E[Y] 平台（需 ess sup Y≤0）：True）
  - 非根状态 b⋆(x) = inf：b⋆ 随状态显著变化（根 ≈7 → 1-bit 子状态 = ∞），reachable-state 平均在 b=32 仍为负（B0.1a §4），证明 b⋆ 是 **state-dependent phase boundary**，不是全局阈值（007 §4 P1-D）；且 E[Y_x]=-6.0000<0 直接给出 **analytic certificate**：progressive dominates direct for every b_h ≥ 0（008 §6），无需扫 b；此处 g≡E[Y] 平台成立是因为 ess sup Y=-6.0000≤0（009 §3：平台需 ess sup Y≤0，不能作为 E[Y]<0 的一般推论）。

- x: UAV7 at 1-bit cell 1: g(b) 序列 = [-3.093, -3.093, -3.093, -3.093, -3.093, -3.093, -3.093, -3.093, -3.093, -3.093, -3.093]；monotone=True，concave=True；g'(8)=0.000 vs Pr(Y>8)=0.000；g'(16)=0.000 vs Pr(Y>16)=0.000（Δ=2 中心差分）；b⋆(x) = inf；E[Y_x] = -3.0932，ess sup Y = -0.1864（b⋆<∞ ⟺ E[Y_x]≥0：PASS；E[Y]<0 ⇒ g(b)≤E[Y]<0 ∀b：PASS；g≡E[Y] 平台（需 ess sup Y≤0）：True）
  - 非根状态 b⋆(x) = inf：b⋆ 随状态显著变化（根 ≈7 → 1-bit 子状态 = ∞），reachable-state 平均在 b=32 仍为负（B0.1a §4），证明 b⋆ 是 **state-dependent phase boundary**，不是全局阈值（007 §4 P1-D）；且 E[Y_x]=-3.0932<0 直接给出 **analytic certificate**：progressive dominates direct for every b_h ≥ 0（008 §6），无需扫 b；此处 g≡E[Y] 平台成立是因为 ess sup Y=-0.1864≤0（009 §3：平台需 ess sup Y≤0，不能作为 E[Y]<0 的一般推论）。


## 8. G7 — bias correction vs variance reduction ablation（008 §3）

- 三格消融：**(marginal-product × independent)** = 原 B0.3（逐 UAV 边缘独立采样、动作各自采样）；**(joint-H × independent)** = 仅修正概率模型（H-联合 world，但动作各自采样）——单独测 **bias correction**；**(joint-H × paired)** = B0.3a（共享 world）——单独测 **variance reduction**。指标：E|Q̂_a − Q_a^{π_b}|、P(a_CR = a_{π_b}⋆)、Var(Δ̂_{a,b})。

- Part A（root，n=2000，pair (3, 4) vs (2, 4)）：Var(Δ̂)^{ind} = 10.9019，Var(Δ̂)^{paired} = 0.8308，耦合效率 κ = (σ_a²+σ_b²)/σ_ab² = 13.1225（008 §9：n_paired ≈ n_uncoupled/κ）

| config | E|Q̂−Q^{π_b}| | P(a_CR=a*_{π_b}) | Var(Δ̂) (pair) |
| --- | --- | --- | --- |
| marg×ind | 9.4146 | 0.5733 | 11.9216 |
| joint×ind | 5.4912 | 0.6200 | 10.9019 |
| joint×paired | 5.7125 | 0.8667 | 0.8308 |
- 解释：marg→joint 消除边缘独立导致的估计 bias（E|Q̂−Q| 下降），ind→paired 通过共享 world 降低 Var(Δ̂)（κ≈13.1225，008 §9：n_paired ≈ n_uncoupled/κ），P(match) 从 joint×ind 到 joint×paired 显著提升。注意：G1 的 0.088→0.870 还包含 B0.3a 的 oracle/gate 修正（P0-C/D），本消融只隔离 world 模型与配对耦合对固定估计器的影响（008 §3）。（50s）


