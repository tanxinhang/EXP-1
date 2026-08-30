# MVS-C C5 — Protocol Robustness under Link/Calibration Stress（001 §二十六.1 C5，SMOKE）

> **定位（010 §十二 路线；001 §二十六.1）**：C4 已在 anti-regime 证明（P0 修复后）GPE-EA-het ≡ Myopic-All-het（E[D]=0.0000，matched-action）。C5 叠加四类协议鲁棒性 stress：**(1) packet success** p_succ∈{1,0.95,0.9,0.8}（ARQ collapsed，期望成本仍为 affine ⇒ 010 §七 envelope 精确保持；含 B1/B2 explicit 等价验证）；**(2) control overhead** b_ctrl∈{0,4,8}（并入 b0）；**(3) calibration mismatch** Δγ∈{-3,-1,0,1,3}dB（planner=model 量化器、世界=true 采样；审计证书剪枝保真度）；**(4) evidence correlation** ρ∈{0,0.3,0.6}（世界 common-factor、planner 保持独立假设）。

> **数理设计（P0 教训正面应用）**：决策侧参数经 extended_params 合成 (b0',κ') 传入 ⇒ 自动进入 C3e P0 修复后的 memo Q-key（含 rho/eta/b0/kappa）⇒ 跨 (p_succ,b_ctrl) 无陈旧缓存（T60 锁定）；世界侧参数 （Δγ 换 planner 实例、ρ 只改采样）不进决策 memo。

> 协议（与 C4-G2 同构）：anti regime、separately calibrated（28 网格）、paired CRN、fresh test、paired EB UCB + Hoeffding + Wilson n0/n1；主 H=96、stress H=48；N_CAL=60、N_TEST=120。

## B. ARQ collapsed(B1) vs explicit(B2) 期望成本等价验证（SystemModel §41）

> E[retries]=1/p_succ（几何分布）⇒ collapsed E[B]=E[B_explicit]/p_succ；预算截断/停止时机造成微小差异。violations=0 验证两记账都不超预算。

| p_succ | b_ctrl | E[B^collapsed] | E[B^explicit] | D=col−exp | viol |
| --- | --- | --- | --- | --- | --- |
| 0.95 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.95 | 4.0 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.9 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.9 | 4.0 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.8 | 0.0 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.8 | 4.0 | 0.0000 | 0.0000 | 0.0000 | 0 |

## A. Packet success / control overhead（成本侧，matched G2）

> 基线 p_succ=1、b_ctrl=0 ＝ C4 anti 记录（E[D]=0.0000，GPE-het ≡ Myopic-het），C5 不重跑。以下为单因素扫描。

### p_succ=0.95（b_ctrl=0）

- GPE-EA-het：**θ̂=(256, 1.2)**、Ê_cal[B]=40.7333 airtime、feasible 8/28
- Myopic-All-het：**θ̂=(256, 1.2)**、Ê_cal[B]=40.7333 airtime、feasible 8/28
（calibration 25.0s）

- H=96 (primary)：E[D]=0.0000、EB U95=5.6154、Hoeffding U95=15.1681；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=2.8077、Hoeffding U95=7.5841；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED

### p_succ=0.9（b_ctrl=0）

- GPE-EA-het：**θ̂=(256, 1.2)**、Ê_cal[B]=42.9677 airtime、feasible 6/28
- Myopic-All-het：**θ̂=(256, 1.2)**、Ê_cal[B]=42.9677 airtime、feasible 6/28
（calibration 23.8s）

- H=96 (primary)：E[D]=0.0000、EB U95=5.6154、Hoeffding U95=15.1681；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=2.8077、Hoeffding U95=7.5841；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED

### p_succ=0.8（b_ctrl=0）

- GPE-EA-het：**θ̂=(256, 1.2)**、Ê_cal[B]=37.7393 airtime、feasible 5/28
- Myopic-All-het：**θ̂=(256, 1.2)**、Ê_cal[B]=37.7393 airtime、feasible 5/28
（calibration 21.4s）

- H=96 (primary)：E[D]=0.0000、EB U95=5.6154、Hoeffding U95=15.1681；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=2.8077、Hoeffding U95=7.5841；GPE INFEASIBLE、Myopic INFEASIBLE ⇒ QoS-UNRESOLVED

### b_ctrl=4.0（p_succ=1）

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 21.8s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

### b_ctrl=8.0（p_succ=1）

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 20.6s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

## C. Calibration mismatch（Δγ∈{-3,-1,0,1,3} dB，planner=model）

> **语义（SystemModel §65）**：planner 量化器/消息-PMF/ℓ/证书全部按 γ_model=γ_true+Δγ；世界按 true γ 采样。基线 Δγ=0＝C4 anti 记录。

### Δγ=-3.0 dB（mismatch）

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 16.5s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

- **剪枝保真度审计**（60 状态）：检查 2880、model 剪 1080、假剪 fp=180（0.1667）、漏剪 fn=720（0.4000）

### Δγ=-1.0 dB（mismatch）

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 23.7s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

- **剪枝保真度审计**（60 状态）：检查 2667、model 剪 873、假剪 fp=40（0.0458）、漏剪 fn=175（0.0975）

### Δγ=0（基线）＝ C4 anti 记录（E[D]=0.0000）。

### Δγ=1.0 dB（mismatch）

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 27.9s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

- **剪枝保真度审计**（60 状态）：检查 2634、model 剪 1036、假剪 fp=95（0.0917）、漏剪 fn=45（0.0282）

### Δγ=3.0 dB（mismatch）

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 33.9s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

- **剪枝保真度审计**（60 状态）：检查 2559、model 剪 978、假剪 fp=287（0.2935）、漏剪 fn=185（0.1170）

## D. Evidence correlation（ρ∈{0,0.3,0.6}，planner 保持独立假设）

> 世界 L 采样 common-factor 相关（sample_set_corr）；planner 决策不变（独立模型）⇒ 决策 memo 跨 ρ 复用（数理正确）。基线 ρ=0＝C4 anti 记录。

### ρ=0（基线）＝ C4 anti 记录（E[D]=0.0000）。

### ρ=0.3

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 13.3s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

### ρ=0.6

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 12.8s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

## 总结（C5 位置：010 §十二 路线）

- **A（p_succ/b_ctrl）**：ARQ-collapsed 期望成本保持 affine ⇒ 010 §七 envelope 精确成立；matched G2 逐点报告（PASS/BIT-UNRESOLVED/QoS-UNRESOLVED 诚实口径）。
- **B（B1 vs B2）**：见上表（E[B^collapsed]≈E[B^explicit]，viol=0）。
- **C（mismatch）**：证书剪枝保真度 fp/fn 率见上（审计 true 分布下 model 剪枝的可靠性）。
- **D（correlation）**：ρ 只改世界采样、决策函数不变 ⇒ 报告中 matched G2 反映相关证据对融合 QoS 的影响。

总耗时: 283.5s

