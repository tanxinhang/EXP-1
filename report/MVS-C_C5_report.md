# MVS-C C5 — Protocol Robustness under Link/Calibration Stress（001 §二十六.1 C5，FULL）

> **定位（010 §十二 路线；001 §二十六.1）**：C4 已在 anti-regime 证明（P0 修复后）GPE-EA-het ≡ Myopic-All-het（E[D]=0.0000，matched-action）。C5 叠加四类协议鲁棒性 stress：**(1) packet success** p_succ∈{1,0.95,0.9,0.8}（ARQ collapsed，期望成本仍为 affine ⇒ 010 §七 envelope 精确保持；含 B1/B2 explicit 等价验证）；**(2) control overhead** b_ctrl∈{0,4,8}（并入 b0）；**(3) calibration mismatch** Δγ∈{-3,-1,0,1,3}dB（planner=model 量化器、世界=true 采样；审计证书剪枝保真度）；**(4) evidence correlation** ρ∈{0,0.3,0.6}（世界 common-factor、planner 保持独立假设）。

> **数理设计（P0 教训正面应用）**：决策侧参数经 extended_params 合成 (b0',κ') 传入 ⇒ 自动进入 C3e P0 修复后的 memo Q-key（含 rho/eta/b0/kappa）⇒ 跨 (p_succ,b_ctrl) 无陈旧缓存（T60 锁定）；世界侧参数 （Δγ 换 planner 实例、ρ 只改采样）不进决策 memo。

> 协议（与 C4-G2 同构）：anti regime、separately calibrated（28 网格）、paired CRN、fresh test、paired EB UCB + Hoeffding + Wilson n0/n1；主 H=96、stress H=48；N_CAL=600、N_TEST=1600。

## B. ARQ collapsed(B1) vs explicit(B2) 期望成本等价验证（SystemModel §41）

> E[retries]=1/p_succ（几何分布）⇒ collapsed E[B]=E[B_explicit]/p_succ；预算截断/停止时机造成微小差异。violations=0 验证两记账都不超预算。

| p_succ | b_ctrl | E[B^collapsed] | E[B^explicit] | D=col−exp | viol |
| --- | --- | --- | --- | --- | --- |
| 0.95 | 0.0 | 31.5661 | 31.4754 | 0.0907 | 0 |
| 0.95 | 4.0 | 33.4926 | 33.3504 | 0.1423 | 0 |
| 0.9 | 0.0 | 33.1217 | 32.9953 | 0.1263 | 0 |
| 0.9 | 4.0 | 35.3533 | 34.9392 | 0.4141 | 0 |
| 0.8 | 0.0 | 37.2619 | 35.8961 | 1.3658 | 0 |
| 0.8 | 4.0 | 39.7725 | 38.4297 | 1.3427 | 0 |

## A. Packet success / control overhead（成本侧，matched G2）

> 基线 p_succ=1、b_ctrl=0 ＝ C4 anti 记录（E[D]=0.0000，GPE-het ≡ Myopic-het），C5 不重跑。以下为单因素扫描。

### p_succ=0.95（b_ctrl=0）

- GPE-EA-het：**θ̂=(256, 1.0)**、Ê_cal[B]=33.2466 airtime、feasible 8/28
- Myopic-All-het：**θ̂=(256, 1.0)**、Ê_cal[B]=33.2466 airtime、feasible 8/28
（calibration 236.2s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED

### p_succ=0.9（b_ctrl=0）

- GPE-EA-het：**θ̂=(256, 1.0)**、Ê_cal[B]=35.0605 airtime、feasible 7/28
- Myopic-All-het：**θ̂=(256, 1.0)**、Ê_cal[B]=35.0605 airtime、feasible 6/28
（calibration 223.7s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED

### p_succ=0.8（b_ctrl=0）

- GPE-EA-het：**θ̂=(256, 1.0)**、Ê_cal[B]=38.0507 airtime、feasible 5/28
- Myopic-All-het：**θ̂=(256, 1.0)**、Ê_cal[B]=38.0507 airtime、feasible 5/28
（calibration 206.1s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE INFEASIBLE、Myopic INFEASIBLE ⇒ QoS-UNRESOLVED

### b_ctrl=4.0（p_succ=1）

- GPE-EA-het：**θ̂=(256, 1.0)**、Ê_cal[B]=35.4643 airtime、feasible 6/28
- Myopic-All-het：**θ̂=(256, 1.0)**、Ê_cal[B]=35.4643 airtime、feasible 6/28
（calibration 213.5s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED

### b_ctrl=8.0（p_succ=1）

- GPE-EA-het：**θ̂=(256, 1.2)**、Ê_cal[B]=40.7140 airtime、feasible 7/28
- Myopic-All-het：**θ̂=(256, 1.2)**、Ê_cal[B]=40.7140 airtime、feasible 7/28
（calibration 201.2s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE INFEASIBLE、Myopic INFEASIBLE ⇒ QoS-UNRESOLVED

## C. Calibration mismatch（Δγ∈{-3,-1,0,1,3} dB，planner=model）

> **语义（SystemModel §65）**：planner 量化器/消息-PMF/ℓ/证书全部按 γ_model=γ_true+Δγ；世界按 true γ 采样。基线 Δγ=0＝C4 anti 记录。

### Δγ=-3.0 dB（mismatch）

- GPE-EA-het：**θ̂=(512, 1.0)**、Ê_cal[B]=36.1156 airtime、feasible 5/28
- Myopic-All-het：**θ̂=(512, 1.0)**、Ê_cal[B]=36.1156 airtime、feasible 5/28
（calibration 162.2s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED

- **剪枝保真度审计**（120 状态）：检查 5760、model 剪 2160、假剪 fp=360（0.1667）、漏剪 fn=1440（0.4000）

### Δγ=-1.0 dB（mismatch）

- GPE-EA-het：**θ̂=(256, 1.0)**、Ê_cal[B]=38.7577 airtime、feasible 8/28
- Myopic-All-het：**θ̂=(256, 1.0)**、Ê_cal[B]=38.7577 airtime、feasible 8/28
（calibration 219.1s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED

- **剪枝保真度审计**（120 状态）：检查 5256、model 剪 1714、假剪 fp=83（0.0484）、漏剪 fn=298（0.0841）

### Δγ=0（基线）＝ C4 anti 记录（E[D]=0.0000）。

### Δγ=1.0 dB（mismatch）

- GPE-EA-het：**θ̂=(256, 1.0)**、Ê_cal[B]=39.7641 airtime、feasible 8/28
- Myopic-All-het：**θ̂=(256, 1.0)**、Ê_cal[B]=39.7641 airtime、feasible 8/28
（calibration 263.5s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED

- **剪枝保真度审计**（120 状态）：检查 5178、model 剪 2048、假剪 fp=192（0.0938）、漏剪 fn=86（0.0275）

### Δγ=3.0 dB（mismatch）

- GPE-EA-het：**θ̂=(128, 1.0)**、Ê_cal[B]=32.2467 airtime、feasible 19/28
- Myopic-All-het：**θ̂=(128, 1.0)**、Ê_cal[B]=32.2467 airtime、feasible 19/28
（calibration 307.8s）

- H=96 (primary)：E[D]=0.0000、EB U95=0.4195、Hoeffding U95=4.1540；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.2098、Hoeffding U95=2.0770；GPE FEASIBLE、Myopic FEASIBLE ⇒ BIT-UNRESOLVED

- **剪枝保真度审计**（120 状态）：检查 5235、model 剪 1853、假剪 fp=540（0.2914）、漏剪 fn=352（0.1041）

## D. Evidence correlation（ρ∈{0,0.3,0.6}，planner 保持独立假设）

> 世界 L 采样 common-factor 相关（sample_set_corr）；planner 决策不变（独立模型）⇒ 决策 memo 跨 ρ 复用（数理正确）。基线 ρ=0＝C4 anti 记录。

### ρ=0（基线）＝ C4 anti 记录（E[D]=0.0000）。

### ρ=0.3

- GPE-EA-het：**θ̂=(1024, 1.6)**、Ê_cal[B]=51.8300 airtime、feasible 1/28
- Myopic-All-het：**θ̂=(1024, 1.6)**、Ê_cal[B]=51.9710 airtime、feasible 1/28
（calibration 120.9s）

- H=96 (primary)：E[D]=-0.1367、EB U95=0.7378、Hoeffding U95=5.7379；GPE UNCERTAIN、Myopic UNCERTAIN ⇒ QoS-UNRESOLVED
- H=48 (stress)：E[D]=0.0000、EB U95=0.4197、Hoeffding U95=2.9373；GPE INFEASIBLE、Myopic INFEASIBLE ⇒ QoS-UNRESOLVED

### ρ=0.6

- GPE-EA-het：∅（无 FEASIBLE θ̂）feasible 0/28
- Myopic-All-het：∅（无 FEASIBLE θ̂）feasible 0/28
（calibration 120.2s）

- H=96 (primary)：θ̂ 缺失 → QoS-UNRESOLVED。
- H=48 (stress)：θ̂ 缺失 → QoS-UNRESOLVED。

## 总结（C5 位置：010 §十二 路线）

- **A（p_succ/b_ctrl）**：ARQ-collapsed 期望成本保持 affine ⇒ 010 §七 envelope 精确成立；matched G2 逐点报告（PASS/BIT-UNRESOLVED/QoS-UNRESOLVED 诚实口径）。
- **B（B1 vs B2）**：见上表（E[B^collapsed]≈E[B^explicit]，viol=0）。
- **C（mismatch）**：证书剪枝保真度 fp/fn 率见上（审计 true 分布下 model 剪枝的可靠性）。
- **D（correlation）**：ρ 只改世界采样、决策函数不变 ⇒ 报告中 matched G2 反映相关证据对融合 QoS 的影响。

总耗时: 2783.3s

