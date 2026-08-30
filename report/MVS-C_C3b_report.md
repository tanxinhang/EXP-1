# MVS-C C3b — Causal Four-Layer Algorithm Comparison（advice/005.md §十八，FULL）

> **定位（005 §十八）**：五方法 separately calibrated、paired CRN、四层因果对照——每对回答一个明确问题：**Phase-PJ vs Myopic-PJ**＝conditional-refinement planning 价值；**vs Direct8**＝adaptive granularity 价值；**vs StaticProg**＝realized-message feedback 价值；**vs Myopic-All**＝强 greedy baseline 下仍成立。

> 协议（G2 017 §四 同）：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup=16.0、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，仅 calibration）；calibration worlds 共用、test fresh 分离（paired CRN）；主 H=96、stress H=48 （同冻结 θ̂）；fixed-N paired one-sided Hoeffding + Wilson U95。N_CAL=600、N_TEST=1600。

> **StaticProg 语义（007 审计修正）**：固定 SNR 顺序 ladder + |Ω|≥η early-stop（B11 语义），不再用 QoS-dual R≤min Q（后者 root 即停导致全停退化）；rho 仅作 θ̂ 网格同构。

## 1. Calibration（五方法 separately calibrated，G2 协议）

- Phase-PJ (Proposed)：**θ̂=(256, 0.8)**、Ê_cal[B]=34.0033 bits、feasible 10/28；10/28 全停退化
- Myopic-PJ：**θ̂=(256, 0.8)**、Ê_cal[B]=34.0033 bits、feasible 10/28；10/28 全停退化
- Myopic-All：**θ̂=(256, 0.8)**、Ê_cal[B]=31.7975 bits、feasible 8/28；8/28 全停退化
- Direct8：**θ̂=(256, 0.8)**、Ê_cal[B]=36.8800 bits、feasible 10/28；10/28 全停退化
- StaticProg：**∅（无 FEASIBLE）**；feasible 0/28
（530.3s）

## 2. Test @ H=96（θ̂ 冻结、fresh worlds、paired）

| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-PJ (Proposed) | (256,0.8) | 0.0869 | 0.1017 | 0.3000 | 0.3229 | FEASIBLE | 1.5247 | 10.0975 | 34.4925 |
| Myopic-PJ | (256,0.8) | 0.0869 | 0.1017 | 0.3000 | 0.3229 | FEASIBLE | 1.5247 | 10.0975 | 34.4925 |
| Myopic-All | (256,0.8) | 0.0756 | 0.0896 | 0.2894 | 0.3121 | FEASIBLE | 1.6587 | 5.6575 | 32.1975 |
| Direct8 | (256,0.8) | 0.0712 | 0.0849 | 0.2913 | 0.3140 | FEASIBLE | 1.5634 | 12.5075 | 37.5225 |
| StaticProg | ∅ | — | — | — | — | NO-FEASIBLE-θ̂ | — | — | — |
（60.2s）

## 3. Stress @ H=48（同冻结 θ̂，诚实报告 boundary）

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[B] |
| --- | --- | --- | --- | --- | --- | --- |
| Phase-PJ (Proposed) | 0.0819 | 0.0963 | 0.3344 | 0.3579 | FEASIBLE | 31.5647 |
| Myopic-PJ | 0.0819 | 0.0963 | 0.3344 | 0.3579 | FEASIBLE | 31.5647 |
| Myopic-All | 0.0688 | 0.0822 | 0.3475 | 0.3712 | FEASIBLE | 28.1462 |
| Direct8 | 0.0781 | 0.0923 | 0.3281 | 0.3515 | FEASIBLE | 33.1725 |
| StaticProg | — | — | — | — | NO-FEASIBLE-θ̂ | — |
（49.5s）

## 4. 四层因果对照（Phase-PJ vs 各 baseline，paired D + Hoeffding）

> paired D=E[B^{m1}]−E[B^{m2}]（同 worlds，θ̂ 各自冻结）；Hoeffding U95<0 ⇒ 统计证实 m1 更省 bits（fixed-N δ=0.05）；任一方 NO-FEASIBLE ⇒ 标注对照不可比（QoS 未认证）。

- **Phase-PJ (Proposed) vs Myopic-PJ**（conditional-refinement planning value）：E[B] 34.4925 vs 34.4925，D=0.0000、U95=4.1540、L95=-4.1540 → **UNRESOLVED**（L95≤0≤U95）（FEASIBLE/FEASIBLE）
- **Phase-PJ (Proposed) vs Direct8**（adaptive evidence granularity value）：E[B] 34.4925 vs 37.5225，D=-3.0300、U95=1.1240、L95=-7.1840 → **UNRESOLVED**（L95≤0≤U95）（FEASIBLE/FEASIBLE）
- **Phase-PJ (Proposed) vs StaticProg**（realized-message-dependent feedback value）：StaticProg 无 FEASIBLE θ̂ → **对照不可比**（QoS-UNRESOLVED）。
- **Phase-PJ (Proposed) vs Myopic-All**（proposed still valuable under strong greedy baseline）：E[B] 34.4925 vs 32.1975，D=2.2950、U95=6.4490、L95=-1.8590 → **UNRESOLVED**（L95≤0≤U95）（FEASIBLE/FEASIBLE）
（0.0s）

## 结论

- **C3b 五方法 H=96**：4/5 方法达到 FEASIBLE（Phase-PJ、Myopic-PJ、Myopic-All、Direct8 若达标；StaticProg 语义修正后仍 无 FEASIBLE θ̂——固定顺序简单渐进无法同时满足 α=0.12/β=0.4，本身即 adaptive 必要性的证据，005 §十八 对照含义保留）。

- **四层因果对照**见 §4：Phase-PJ vs Myopic-PJ 隔离 conditional-refinement value（同动作集）；vs Direct8 隔离 adaptive granularity；vs StaticProg 隔离 realized-message feedback；vs Myopic-All 检验强 baseline 下仍成立。

总耗时: 643.5s

