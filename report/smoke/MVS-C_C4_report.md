# MVS-C C4 — Link-Aware Heterogeneous U2U Airtime（010 §九/§十二，SMOKE）

> **定位（001 §十六 / README C4 / 010 §九）**：成本从 homogeneous "16+Δr bits" 升级为 **per-UAV airtime** τ_i(r→r')=b0,i+κ_i(r'−r) （b0,i≡τ_ctrl,i、κ_i≡1/R_i），hard frame budget Στ≤H。三 regime：positive / independent / **anti-correlated**（强 sensing = 坏链路，001 §十六 最重要的机制实验）。GPE-EA-het vs Myopic-All-het 做 **matched-action** 对比（相同 full action set、相同成本模型、相同 QoS，唯一差别 = conditional-refinement planning）——协议与 C3e-G2 同构（separately calibrated、paired CRN、fresh test、paired EB UCB 主认证 + Hoeffding sanity + Wilson n0/n1）。

> 协议：N=8（GAMMA_B）、levels=(1, 2, 4, 8)、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，仅 calibration）；calibration worlds 共用、test worlds 完全分离（paired CRN）；主 H=96、stress H=48（同冻结 θ̂）；N_CAL=60、N_TEST=120。budget 单位为 airtime，homogeneous 特例=旧 16+Δr。

## Regime：positive

> 链路参数（sensing rank → q → b0/κ）：b0=[np.float64(20.0), np.float64(18.9), np.float64(17.7), np.float64(16.6), np.float64(15.4), np.float64(14.3), np.float64(13.1), np.float64(12.0)]，κ=[np.float64(1.2), np.float64(1.14), np.float64(1.09), np.float64(1.03), np.float64(0.97), np.float64(0.91), np.float64(0.86), np.float64(0.8)]；bounds b0∈(12.0, 20.0)、κ∈(0.8, 1.2)。

- GPE-EA-het：**θ̂=(512, 1.6)**、Ê_cal[B]=63.0133 airtime、feasible 3/28
- Myopic-All-het：**∅（无 FEASIBLE θ̂）**；feasible 0/28
（calibration 38.2s）

| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPE-EA-het | (512,1.6) | 0.0333 | 0.0826 | 0.2917 | 0.3784 | FEASIBLE | 3.8667 | 7.7500 | 57.1888 |

### H=96 (primary)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。

### H=48 (stress)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。

## Regime：independent

> 链路参数（sensing rank → q → b0/κ）：b0=[np.float64(16.6), np.float64(15.4), np.float64(17.7), np.float64(20.0), np.float64(14.3), np.float64(13.1), np.float64(12.0), np.float64(18.9)]，κ=[np.float64(1.03), np.float64(0.97), np.float64(1.09), np.float64(1.2), np.float64(0.91), np.float64(0.86), np.float64(0.8), np.float64(1.14)]；bounds b0∈(12.0, 20.0)、κ∈(0.8, 1.2)。

- GPE-EA-het：**∅（无 FEASIBLE θ̂）**；feasible 0/28
- Myopic-All-het：**∅（无 FEASIBLE θ̂）**；feasible 0/28
（calibration 37.4s）

| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### H=96 (primary)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。

### H=48 (stress)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。

## Regime：anti

> 链路参数（sensing rank → q → b0/κ）：b0=[np.float64(12.0), np.float64(13.1), np.float64(14.3), np.float64(15.4), np.float64(16.6), np.float64(17.7), np.float64(18.9), np.float64(20.0)]，κ=[np.float64(0.8), np.float64(0.86), np.float64(0.91), np.float64(0.97), np.float64(1.03), np.float64(1.09), np.float64(1.14), np.float64(1.2)]；bounds b0∈(12.0, 20.0)、κ∈(0.8, 1.2)。

- GPE-EA-het：**∅（无 FEASIBLE θ̂）**；feasible 0/28
- Myopic-All-het：**∅（无 FEASIBLE θ̂）**；feasible 0/28
（calibration 29.4s）

| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### H=96 (primary)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。

### H=48 (stress)：paired D=E[B^GPE]−E[B^Myopic]（airtime）

- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。

## 总结（C4 位置：010 §十二 路线）

- **C4 = heterogeneous link-aware airtime**：τ_i=b0,i+κ_i(r'−r)，与 010 §七 的 generalized envelope 参数（b0,i、κ_i）**天然同一组**——因此 G1 定理直接覆盖异质链路的相变结构，C4 验证的是 planner 在 per-UAV 成本下的 matched-action 收益与 anti-regime 重路由机制。
- **G2 认证（matched-action，per-regime）**：见上方各 regime 判定（PASS / BIT-UNRESOLVED / QoS-UNRESOLVED 诚实报告）。
- **anti-correlated 机制**：corr(E[N_tx,i], κ_i) 见上（负 ⇒ planner 避开坏链路 UAV，001 §十六 的机制直接证据）。

总耗时: 110.6s

