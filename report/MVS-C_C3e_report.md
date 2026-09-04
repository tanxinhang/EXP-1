# MVS-C C3e — Generalized Phase-Envelope Evidence Acquisition（advice/010.md §七-§十二，FULL）

> **定位（010 §十二）**：C3b 显示 Phase-PJ 在实际工作点无实证增益（Phase==Myopic-PJ、Phase>Myopic-All +2.30 bits）。C3e 不再修Phase-PJ：G0 审计它为何不激活；G1 把 013 的 next/full 定理升级成任意 r<s<t 的 generalized phase envelope（link-affine cost c_i(r→q)=b0+κ(q−r)）；G2 新 Proposed **GPE-EA** 用与 Myopic-All 相同 full action set，唯一差别是 conditional-refinement Q —— 论文生死 Gate；G3 paired empirical-Bernstein UCB 作为主 bit 认证。

> 协议（G2/C3b 017 §四 同）：N=8（GAMMA_B）、levels=(1, 2, 4, 8)、b_setup=16.0、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，仅 calibration）；calibration worlds 共用、test fresh 分离（paired CRN）；H=96 主 / H=48 stress；N_CAL=600、N_TEST=1600、N_AUDIT=2000。

## G0. Phase activation audit（010 §十二 C3e-G0）

> 冻结工作点 θ̂=(256, 0.8)（C3b 校准点）。回答：Phase-PJ 的 conditional-refinement 在实际决策路径上的激活频率。

- 决策状态数：10006；P(继续)=0.6017
- **action-change rate**（Phase vs Myopic-PJ 动作不同）: 0.0000
- **P(Q_phase≠Q_myopic)**（|Q_prog−Q^(1)|>1e-6 的 candidate 比例，45816 个 candidate）: 0.6079
- probe-feasible supports=45816，region A/B/C=56/795/44965；**pruning rate**=0.7607（pruned 34851）
- **G0 判定**：action-change rate=0.0000（P(Q_phase≠Q_myopic)=0.6079 仅 Q 值层激活、pruning rate=0.7607 ⇒ 差异被剪/不改变 argmin）⇒ **Phase-PJ 与 Myopic-PJ 动作处处一致 ⇒ conditional refinement 在决策层没有产生作用（010 §一 结论复现）**。（81.7s）

## G1. Generalized r<s<t phase-envelope theorem（010 §七）

> c_i(r→q)=b0+κ(q−r)；Q_prog^{s,t}−Q_dir^t == E[min{Y_{i,s,t}, b0}]，Y=R(X_s)−E[R(X_t)|X_s]−κ(t−s)。角 (b0,κ)∈((16.0, 1.0), (16.0, 2.0), (8.0, 1.0), (32.0, 0.5))。

- 可达状态（G1 抽样）: 220
- **G1a identity** max|g−g_alt|=2.20e-13 （目标 <1e-9）→ PASS
- **G1b tower** max|E[E_R]−E_dir|=2.27e-13 （目标 <1e-9）→ PASS
- **G1c derivative=survival** max|∂g−Pr(Y>b)|=1.13e-11（目标 <1e-6）→ PASS
- **G1d b* 分类**（b*<∞ ⟺ E[Y]≥0，A/B/C）: 4847/4847 → PASS
- **G1e special-case**（与 c21.phase_support_budget C 区逐状态比对 Q_prog/Q_dir）: 1622/1622 一致、identity 1.74e-13 → PASS
（36.8s）

## G2. Matched-action Gate：GPE-EA(global Depth-2) vs Myopic-All（011 P0：SystemModel §24 真 Depth-2）

> **011 §二/§三 P0 修正**：旧 GPE-EA（`gpe_decision`）的第二步 conts 只枚举**同一 UAV** 更高精度（self-refinement continuation），而 SystemModel §24 的 `min_b` 定义在 **A(X',h') 全局可行动作空间**上——允许跨 UAV switch，且 UAV 到最高精度后**不终止**（仍可请求其他 UAV）。本轮 Proposed 改用 **`gpe_decision_global`**（`_cond_refine_q_global`），实现精确恒等式 Q_2(a|x,h)=Q_1(a|x,h)−E[Δ(X',h−c_a)]，Δ=[R(X')−min_b Q_1(b|X')]_+；并实现 011 §六 的**严格剪枝**：c_b≥R(X') ⇒ Q_1(b)≥R(X') 不可能优于 STOP（exact prune，不展开 E）；候选按 c_b 升序、c_b≥V_best 后整段跳过（B&B，非启发式）。旧 `gpe_decision` 保留为 Phase-Envelope self-refinement 子问题解析机制（011 §五 重定位，不删）。

> **011 §十 gate 协议**：先跑 6 个机制统计（P(Δ_all>0)、E[Δ_all]、E[Δ_self]、G_switch=E[Δ_all−Δ_self]、P(j*≠i)、action-change rate）于冻结 θ̂ 的 on-policy 状态；**G_switch>0 才跑 FULL matched-QoS**；否则如实报告 self-refinement 未激活（GPE 维持 ≡ Myopic，不浪费 FULL）。

### G2-gate：global Depth-2 机制统计（011 §三/§四/§十）

- 决策状态：11300；候选 323102
- **P(Δ_all>0)**=0.4787；**E[Δ_all]**=14.0608 bits；**E[Δ_self]**=0.6558 bits
- **G_switch=E[Δ_all−Δ_self]**=13.4050（011 §四：跨 UAV switch 的严格增益，≥0）
- **P(j*≠i)**=0.8962（第二步 argmin 的 UAV 与第一步不同——跨 UAV switch 激活频率）
- **action-change rate**=0.2398（GPE-global 与 Myopic 的动作不同）
- **Gate 判定**：**PASS（跨 UAV switch 机制激活 ⇒ 跑 FULL matched-QoS）**（51797.1s）

- GPE-EA (Proposed)：**θ̂=(128, 0.8)**、Ê_cal[B]=24.0708 bits、feasible 15/28
- Myopic-All：**θ̂=(256, 0.8)**、Ê_cal[B]=31.7975 bits、feasible 8/28
（calibration 214514.0s）

| | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPE-EA (Proposed) | (128,0.8) | 0.0794 | 0.0937 | 0.3394 | 0.3629 | FEASIBLE | 1.3334 | 2.6634 | 23.9984 |
| Myopic-All | (256,0.8) | 0.0756 | 0.0896 | 0.2894 | 0.3121 | FEASIBLE | 1.6587 | 5.6575 | 32.1975 |

### H=96 (primary)：paired D=E[B^GPE]−E[B^MyopicAll]

- E[D]=-8.1991；**paired EB U95=-7.0901**（L95=-9.3081，MP Thm 4 t=log(1/δ)）；Hoeffding sanity U95=-4.0451（L95=-12.3530）
- QoS：GPE FEASIBLE（U95 0.0937/0.3629）、Myopic-All FEASIBLE（U95 0.0896/0.3121）
- **G2 判定**：**PASS**：双方 FEASIBLE 且 paired EB U95<0 ⇒ GPE-EA 在 matched-action 下统计认证更省 bits
- 分解：E[N_tx] GPE=1.3334 vs Myopic 1.6587；E[B_payload] 2.6634 vs 5.6575；E[B|H0] 22.0362 vs 27.8275、E[B|H1] 25.9606 vs 36.5675（017 §七 secondary）

### H=48 (stress)：paired D=E[B^GPE]−E[B^MyopicAll]

- E[D]=-5.8113；**paired EB U95=-5.2925**（L95=-6.3300，MP Thm 4 t=log(1/δ)）；Hoeffding sanity U95=-3.7343（L95=-7.8882）
- QoS：GPE FEASIBLE（U95 0.1170/0.3515）、Myopic-All FEASIBLE（U95 0.0822/0.3712）
- **G2 判定**：**PASS**：双方 FEASIBLE 且 paired EB U95<0 ⇒ GPE-EA 在 matched-action 下统计认证更省 bits
- 分解：E[N_tx] GPE=1.2550 vs Myopic 1.4384；E[B_payload] 2.2550 vs 5.1312；E[B|H0] 21.2300 vs 25.5994、E[B|H1] 23.4400 vs 30.6931（017 §七 secondary）

## G3. Paired empirical-Bernstein UCB（010 §十；主 bit 认证）

> G2 判定以 **paired fixed-N one-sided EB UCB**（MP Thm 4 plug-in variance，t=log(1/δ)，(n−1) 保守分母）为主；Hoeffding（D∈[−H,H]）为 sanity envelope。paired CRN 压缩 E[B] 差分方差 ⇒ EB 通常比 Hoeffding 更紧（B0.4 系列同机制验证）。

- H=96 (primary)（n=3200）：D∈[−96,96]；paired EB U95=-7.0901、Hoeffding U95=-4.0451 —— EB 界 更紧；支撑 G2 PASS。
- H=48 (stress)（n=3200）：D∈[−48,48]；paired EB U95=-5.2925、Hoeffding U95=-3.7343 —— EB 界 更紧；支撑 G2 PASS。

## 结论（010 §十二 路线）

- **G2 (H=96) 判定**：**PASS**（E[D]=-8.1991、EB U95=-7.0901、Hoeffding U95=-4.0451、GPE QoS FEASIBLE、Myopic QoS FEASIBLE）
- **G0**：Phase activation——conditional-refinement 激活率 0.6079、action-change 0.0000（010 §一：Phase-PJ 近似 myopic）。
- **G1**：generalized envelope 五 Gate 全部 PASS（a/b/c/d/e 见上文）。
- **G2 双 Budget 判定**：G2 H=96 PASS；G2 H=48 PASS。
- **下一步（010 §十二）**：**budget-regime 依赖结论**：conditional-refinement 的价值在紧预算（H=48）统计认证体现（省证据 payload），宽预算（H=96）把多余 bits 花在事务/检测余量上——按 010 §十二 进 C4 时保留GPE-EA 并报告该 regime 依赖；C4 = heterogeneous airtime（b0,i、κ_i，positive/independent/anti-correlated regimes，τ_i(r→r')=τ_ctrl,i+(b_hdr+(r'−r))/R_i，hard frame budget）。

总耗时: 302001.1s

