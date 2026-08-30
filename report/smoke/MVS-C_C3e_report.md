# MVS-C C3e — Generalized Phase-Envelope Evidence Acquisition（advice/010.md §七-§十二，SMOKE）

> **定位（010 §十二）**：C3b 显示 Phase-PJ 在实际工作点无实证增益（Phase==Myopic-PJ、Phase>Myopic-All +2.30 bits）。C3e 不再修Phase-PJ：G0 审计它为何不激活；G1 把 013 的 next/full 定理升级成任意 r<s<t 的 generalized phase envelope（link-affine cost c_i(r→q)=b0+κ(q−r)）；G2 新 Proposed **GPE-EA** 用与 Myopic-All 相同 full action set，唯一差别是 conditional-refinement Q —— 论文生死 Gate；G3 paired empirical-Bernstein UCB 作为主 bit 认证。

> 协议（G2/C3b 017 §四 同）：N=8（GAMMA_B）、levels=(1, 2, 4, 8)、b_setup=16.0、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，仅 calibration）；calibration worlds 共用、test fresh 分离（paired CRN）；H=96 主 / H=48 stress；N_CAL=100、N_TEST=200、N_AUDIT=200。

## G0. Phase activation audit（010 §十二 C3e-G0）

> 冻结工作点 θ̂=(256, 0.8)（C3b 校准点）。回答：Phase-PJ 的 conditional-refinement 在实际决策路径上的激活频率。

- 决策状态数：1007；P(继续)=0.6038
- **action-change rate**（Phase vs Myopic-PJ 动作不同）: 0.0000
- **P(Q_phase≠Q_myopic)**（|Q_prog−Q^(1)|>1e-6 的 candidate 比例，4630 个 candidate）: 0.6084
- probe-feasible supports=4630，region A/B/C=7/61/4562；**pruning rate**=0.7622（pruned 3529）
- **G0 判定**：action-change rate=0.0000（P(Q_phase≠Q_myopic)=0.6084 仅 Q 值层激活、pruning rate=0.7622 ⇒ 差异被剪/不改变 argmin）⇒ **Phase-PJ 与 Myopic-PJ 动作处处一致 ⇒ conditional refinement 在决策层没有产生作用（010 §一 结论复现）**。（7.4s）

## G1. Generalized r<s<t phase-envelope theorem（010 §七）

> c_i(r→q)=b0+κ(q−r)；Q_prog^{s,t}−Q_dir^t == E[min{Y_{i,s,t}, b0}]，Y=R(X_s)−E[R(X_t)|X_s]−κ(t−s)。角 (b0,κ)∈((16.0, 1.0), (16.0, 2.0), (8.0, 1.0), (32.0, 0.5))。

- 可达状态（G1 抽样）: 60
- **G1a identity** max|g−g_alt|=1.75e-13 （目标 <1e-9）→ PASS
- **G1b tower** max|E[E_R]−E_dir|=1.71e-13 （目标 <1e-9）→ PASS
- **G1c derivative=survival** max|∂g−Pr(Y>b)|=6.98e-12（目标 <1e-6）→ PASS
- **G1d b* 分类**（b*<∞ ⟺ E[Y]≥0，A/B/C）: 1313/1313 → PASS
- **G1e special-case**（与 c21.phase_support_budget C 区逐状态比对 Q_prog/Q_dir）: 440/440 一致、identity 1.74e-13 → PASS
（8.9s）

## G2. Matched-action Gate：GPE-EA vs Myopic-All（010 §八/§十二）

> 两者 **相同 full action set** A={(i,s): s>r_i, s∈levels}、相同成本模型、相同 QoS、相同 calibration/test worlds（paired CRN）——唯一差别是 GPE-EA 对 probe 用 conditional-refinement Q（certificate 证明全分支 STOP 最优时精确退化为 one-step）。separately calibrated（各自28 网格选 θ̂）、fresh test、主 bit 认证 = paired EB UCB（G3），Hoeffding sanity。

- GPE-EA (Proposed)：**θ̂=(256, 0.8)**、Ê_cal[B]=45.8250 bits、feasible 8/28
- Myopic-All：**θ̂=(512, 0.8)**、Ê_cal[B]=42.2950 bits、feasible 5/28
（calibration 53.6s）

| | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPE-EA (Proposed) | (256,0.8) | 0.0800 | 0.1260 | 0.2500 | 0.3143 | UNCERTAIN | 2.5225 | 4.7300 | 45.0900 |
| Myopic-All | (512,0.8) | 0.0800 | 0.1260 | 0.2550 | 0.3196 | UNCERTAIN | 1.9975 | 6.6375 | 38.5975 |

### H=96 (primary)：paired D=E[B^GPE]−E[B^MyopicAll]

- E[D]=6.4925；**paired EB U95=12.0162**（L95=0.9688，MP Thm 4 t=log(1/δ)）；Hoeffding sanity U95=18.2417（L95=-5.2567）
- QoS：GPE UNCERTAIN（U95 0.1260/0.3143）、Myopic-All UNCERTAIN（U95 0.1260/0.3196）
- **G2 判定**：QoS-UNRESOLVED（一方/双方 QoS 未认证，不比较 bits）
- 分解：E[N_tx] GPE=2.5225 vs Myopic 1.9975；E[B_payload] 4.7300 vs 6.6375；E[B|H0] 37.0650 vs 34.4550、E[B|H1] 53.1150 vs 42.7400（017 §七 secondary）

### H=48 (stress)：paired D=E[B^GPE]−E[B^MyopicAll]

- E[D]=-7.5300；**paired EB U95=-4.8378**（L95=-10.2222，MP Thm 4 t=log(1/δ)）；Hoeffding sanity U95=-1.6554（L95=-13.4046）
- QoS：GPE UNCERTAIN（U95 0.1949/0.3668）、Myopic-All UNCERTAIN（U95 0.1494/0.3875）
- **G2 判定**：QoS-UNRESOLVED（一方/双方 QoS 未认证，不比较 bits）
- 分解：E[N_tx] GPE=1.2525 vs Myopic 1.5250；E[B_payload] 2.2525 vs 5.4225；E[B|H0] 21.4000 vs 28.2150、E[B|H1] 23.1850 vs 31.4300（017 §七 secondary）

## G3. Paired empirical-Bernstein UCB（010 §十；主 bit 认证）

> G2 判定以 **paired fixed-N one-sided EB UCB**（MP Thm 4 plug-in variance，t=log(1/δ)，(n−1) 保守分母）为主；Hoeffding（D∈[−H,H]）为 sanity envelope。paired CRN 压缩 E[B] 差分方差 ⇒ EB 通常比 Hoeffding 更紧（B0.4 系列同机制验证）。

- H=96 (primary)（n=400）：D∈[−96,96]；paired EB U95=12.0162、Hoeffding U95=18.2417 —— EB 界 更紧；G2 未以该 H 通过。
- H=48 (stress)（n=400）：D∈[−48,48]；paired EB U95=-4.8378、Hoeffding U95=-1.6554 —— EB 界 更紧；支撑 G2 PASS。

## 结论（010 §十二 路线）

- **G2 (H=96) 判定**：**QoS-UNRESOLVED**（E[D]=6.4925、EB U95=12.0162、Hoeffding U95=18.2417、GPE QoS UNCERTAIN、Myopic QoS UNCERTAIN）
- **G0**：Phase activation——conditional-refinement 激活率 0.6084、action-change 0.0000（010 §一：Phase-PJ 近似 myopic）。
- **G1**：generalized envelope 五 Gate 全部 PASS（a/b/c/d/e 见上文）。
- **G2 双 Budget 判定**：G2 H=96 QoS-UNRESOLVED；G2 H=48 PASS。
- **下一步（010 §十二）**：**budget-regime 依赖结论**：conditional-refinement 的价值在紧预算（H=48）统计认证体现（省证据 payload），宽预算（H=96）把多余 bits 花在事务/检测余量上——按 010 §十二 进 C4 时保留GPE-EA 并报告该 regime 依赖；C4 = heterogeneous airtime（b0,i、κ_i，positive/independent/anti-correlated regimes，τ_i(r→r')=τ_ctrl,i+(b_hdr+(r'−r))/R_i，hard frame budget）。

总耗时: 79.3s

