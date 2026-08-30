# MVS-C C4 — Link-Aware Heterogeneous U2U Airtime（010 §九/§十二，FULL）

> **定位（001 §十六 / README C4 / 010 §九）**：成本从 homogeneous "16+Δr bits" 升级为 **per-UAV airtime** τ_i(r→r')=b0,i+κ_i(r'−r) （b0,i≡τ_ctrl,i、κ_i≡1/R_i），hard frame budget Στ≤H。三 regime：positive / independent / **anti-correlated**（强 sensing = 坏链路，001 §十六 最重要的机制实验）。GPE-EA-het vs Myopic-All-het 做 **matched-action** 对比（相同 full action set、相同成本模型、相同 QoS，唯一差别 = conditional-refinement planning）——协议与 C3e-G2 同构（separately calibrated、paired CRN、fresh test、paired EB UCB 主认证 + Hoeffding sanity + Wilson n0/n1）。

> 协议：N=8（GAMMA_B）、levels=(1, 2, 4, 8)、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，仅 calibration）；calibration worlds 共用、test worlds 完全分离（paired CRN）；主 H=96、stress H=48（同冻结 θ̂）；N_CAL=600、N_TEST=1600。budget 单位为 airtime，homogeneous 特例=旧 16+Δr。

## Regime：positive

> 链路参数（sensing rank → q → b0/κ）：b0=[np.float64(20.0), np.float64(18.9), np.float64(17.7), np.float64(16.6), np.float64(15.4), np.float64(14.3), np.float64(13.1), np.float64(12.0)]，κ=[np.float64(1.2), np.float64(1.14), np.float64(1.09), np.float64(1.03), np.float64(0.97), np.float64(0.91), np.float64(0.86), np.float64(0.8)]；bounds b0∈(12.0, 20.0)、κ∈(0.8, 1.2)。

- GPE-EA-het：**θ̂=(256, 0.8)**、Ê_cal[B]=41.1101 airtime、feasible 18/28
- Myopic-All-het：**θ̂=(256, 0.8)**、Ê_cal[B]=26.9969 airtime、feasible 9/28
（calibration 363.1s）

| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPE-EA-het | (256,0.8) | 0.0631 | 0.0761 | 0.2188 | 0.2397 | FEASIBLE | 2.8178 | 5.4294 | 41.4774 |
| Myopic-All-het | (256,0.8) | 0.0781 | 0.0923 | 0.2587 | 0.2808 | FEASIBLE | 1.7822 | 5.9441 | 27.6668 |

### H=96 (primary)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- E[D]=13.8106；**paired EB U95=14.9617**（L95=12.6594，MP Thm 4）；Hoeffding U95=17.9645
- QoS：GPE FEASIBLE（U95 0.0761/0.2397）、Myopic FEASIBLE（U95 0.0923/0.2808）
- **G2 判定**：BIT-UNRESOLVED（EB U95≥0；方向/量级见点估计）
- 分解：E[N_tx] GPE=2.8178 vs Myopic 1.7822；E[B_payload] 5.4294 vs 5.9441；E[B|H0] 36.3750 vs 23.5628、E[B|H1] 46.5798 vs 31.7708（secondary）

### H=48 (stress)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- E[D]=7.2951；**paired EB U95=8.0022**（L95=6.5879，MP Thm 4）；Hoeffding U95=9.3720
- QoS：GPE FEASIBLE（U95 0.0849/0.3108）、Myopic FEASIBLE（U95 0.0903/0.3165）
- **G2 判定**：BIT-UNRESOLVED（EB U95≥0；方向/量级见点估计）
- 分解：E[N_tx] GPE=2.3369 vs Myopic 1.7069；E[B_payload] 4.5856 vs 5.7200；E[B|H0] 30.7513 vs 23.1834、E[B|H1] 36.2717 vs 29.2496（secondary）

## Regime：independent

> 链路参数（sensing rank → q → b0/κ）：b0=[np.float64(16.6), np.float64(15.4), np.float64(17.7), np.float64(20.0), np.float64(14.3), np.float64(13.1), np.float64(12.0), np.float64(18.9)]，κ=[np.float64(1.03), np.float64(0.97), np.float64(1.09), np.float64(1.2), np.float64(0.91), np.float64(0.86), np.float64(0.8), np.float64(1.14)]；bounds b0∈(12.0, 20.0)、κ∈(0.8, 1.2)。

- GPE-EA-het：**θ̂=(256, 0.8)**、Ê_cal[B]=45.8558 airtime、feasible 17/28
- Myopic-All-het：**θ̂=(256, 0.8)**、Ê_cal[B]=37.2438 airtime、feasible 9/28
（calibration 347.8s）

| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPE-EA-het | (256,0.8) | 0.0694 | 0.0829 | 0.2238 | 0.2448 | FEASIBLE | 2.8959 | 5.6544 | 45.7691 |
| Myopic-All-het | (256,0.8) | 0.0712 | 0.0849 | 0.2500 | 0.2718 | FEASIBLE | 2.2309 | 4.3644 | 36.8263 |

### H=96 (primary)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- E[D]=8.9427；**paired EB U95=9.9206**（L95=7.9649，MP Thm 4）；Hoeffding U95=13.0967
- QoS：GPE FEASIBLE（U95 0.0829/0.2448）、Myopic FEASIBLE（U95 0.0849/0.2718）
- **G2 判定**：BIT-UNRESOLVED（EB U95≥0；方向/量级见点估计）
- 分解：E[N_tx] GPE=2.8959 vs Myopic 2.2309；E[B_payload] 5.6544 vs 4.3644；E[B|H0] 40.2577 vs 30.0934、E[B|H1] 51.2804 vs 43.5593（secondary）

### H=48 (stress)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- E[D]=4.6038；**paired EB U95=5.2140**（L95=3.9935，MP Thm 4）；Hoeffding U95=6.6807
- QoS：GPE FEASIBLE（U95 0.0856/0.3832）、Myopic UNCERTAIN（U95 0.1335/0.3178）
- **G2 判定**：QoS-UNRESOLVED（一方/双方 QoS 未认证，不比较 cost）
- 分解：E[N_tx] GPE=2.3384 vs Myopic 1.7544；E[B_payload] 4.2459 vs 3.2634；E[B|H0] 30.9675 vs 25.6107、E[B|H1] 36.7786 vs 32.9279（secondary）

## Regime：anti

> 链路参数（sensing rank → q → b0/κ）：b0=[np.float64(12.0), np.float64(13.1), np.float64(14.3), np.float64(15.4), np.float64(16.6), np.float64(17.7), np.float64(18.9), np.float64(20.0)]，κ=[np.float64(0.8), np.float64(0.86), np.float64(0.91), np.float64(0.97), np.float64(1.03), np.float64(1.09), np.float64(1.14), np.float64(1.2)]；bounds b0∈(12.0, 20.0)、κ∈(0.8, 1.2)。

- GPE-EA-het：**θ̂=(256, 0.8)**、Ê_cal[B]=31.1122 airtime、feasible 11/28
- Myopic-All-het：**θ̂=(256, 1.0)**、Ê_cal[B]=30.8474 airtime、feasible 10/28
（calibration 289.3s）

| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPE-EA-het | (256,0.8) | 0.0850 | 0.0997 | 0.3406 | 0.3642 | FEASIBLE | 1.4656 | 2.7241 | 31.6217 |
| Myopic-All-het | (256,1.0) | 0.0881 | 0.1030 | 0.3281 | 0.3515 | FEASIBLE | 1.4250 | 3.4053 | 31.7526 |

### H=96 (primary)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- E[D]=-0.1309；**paired EB U95=0.7062**（L95=-0.9680，MP Thm 4）；Hoeffding U95=4.0231
- QoS：GPE FEASIBLE（U95 0.0997/0.3642）、Myopic FEASIBLE（U95 0.1030/0.3515）
- **G2 判定**：BIT-UNRESOLVED（EB U95≥0；方向/量级见点估计）
- 分解：E[N_tx] GPE=1.4656 vs Myopic 1.4250；E[B_payload] 2.7241 vs 3.4053；E[B|H0] 27.6687 vs 28.7036、E[B|H1] 35.5748 vs 34.8016（secondary）

### H=48 (stress)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- E[D]=3.9134；**paired EB U95=4.7312**（L95=3.0955，MP Thm 4）；Hoeffding U95=5.9903
- QoS：GPE FEASIBLE（U95 0.1064/0.3933）、Myopic FEASIBLE（U95 0.1130/0.3775）
- **G2 判定**：BIT-UNRESOLVED（EB U95≥0；方向/量级见点估计）
- 分解：E[N_tx] GPE=1.4956 vs Myopic 1.2406；E[B_payload] 2.7462 vs 2.9625；E[B|H0] 26.7794 vs 26.5000、E[B|H1] 37.1224 vs 29.5750（secondary）

### anti-correlated 的链路选择机制（001 §十六）

> 逐 UAV 报告 GPE-EA-het 在 test 上的 **E[N_tx,i]** 与 **E[B,i]**（airtime 占比）：验证 planner 自动把 budget 从 “强 sensing 但坏链路”UAV 转移到“好链路（哪怕中等 sensing）”UAV。

| UAV | γ^s (dB) | q(链路质量) | b0_i | κ_i | E[N_tx,i] | E[B,i] 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | -4.0000 | 1.00 | 12.0 | 0.80 | 0.0056 | 0.0023 |
| 1 | -3.0000 | 0.86 | 13.1 | 0.86 | 0.0050 | 0.0023 |
| 2 | -2.0000 | 0.71 | 14.3 | 0.91 | 0.0000 | 0.0000 |
| 3 | -1.0000 | 0.57 | 15.4 | 0.97 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.43 | 16.6 | 1.03 | 0.0488 | 0.0306 |
| 5 | 1.0000 | 0.29 | 17.7 | 1.09 | 0.1472 | 0.0926 |
| 6 | 2.0000 | 0.14 | 18.9 | 1.14 | 0.2591 | 0.1639 |
| 7 | 3.0000 | 0.00 | 20.0 | 1.20 | 1.0000 | 0.7084 |

- **corr(E[N_tx,i], κ_i)=0.7418**：**正相关——与 001 §十六 的简单预期相反**：matched-QoS 下 planner 仍把 airtime 集中在最强 sensing 的坏链路 UAV7（γ=3dB 占 E[B] 约 0.7084），因为弱 sensing 好链路 UAV 的组合无法 QoS-FEASIBLE（evidence 不足以 达 P_MD≤β）——**sensing QoS 可行性约束优先于链路成本**，anti-correlation 不自动诱导重路由。

## 总结（C4 位置：010 §十二 路线）

- **C4 = heterogeneous link-aware airtime**：τ_i=b0,i+κ_i(r'−r)，与 010 §七 的 generalized envelope 参数（b0,i、κ_i）**天然同一组**——因此 G1 定理直接覆盖异质链路的相变结构，C4 验证的是 planner 在 per-UAV 成本下的 matched-action 收益与 anti-regime 重路由机制。
- **G2 认证（matched-action，per-regime）**：见上方各 regime 判定（PASS / BIT-UNRESOLVED / QoS-UNRESOLVED 诚实报告）。
- **anti-correlated 机制**：corr(E[N_tx,i], κ_i) 见上（负 ⇒ planner 避开坏链路 UAV，001 §十六 的机制直接证据）。

总耗时: 1170.3s

