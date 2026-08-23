# O-PEF MVS-B0.6 — matched-QoS：CR vs optimized Direct8（论文生死 Gate，014 §4-§6）

> 协议（014 §5-§6 冻结）：**stratified N0=N1=600**（H0/H1 独立采样）；episode 级 **CRN**——同一 physical world W_e=(H_e,L_e) 给 CR/Direct8/POTS，planner RNG 独立；**radio cost 与 planning cost 分离**（B_radio=Σ(b_setup+Δr_t)，CPI worlds 只算 compute）；判决阈值 η_nat=log(μ_F/μ_M)=1.0000（T21）；CR = 冻结的 SNR anchor + Operational-CPI（betting，w_ep=1000，δ_t 按决策序号）；Direct8 = **optimized**（SNR-order full packets + η_nat stop；direct_only planner 在 H=96 不可行——4 个全包 → 256⁴ cells）；POTS = round-robin 渐进（第二 comparator）。Gate：U95(P_FA^CR)≤0.12、U95(P_MD^CR)≤0.4、U95(E[D_e^D8])<0（D_e^D8=B_e^CR−B_e^D8，episode-paired）。
- α=0.12, β=0.4 的选取：natural Bayes 判决的工作区（G5 实测 P_FA^nat≈0.09、P_MD^nat≈0.22-0.37 @ η=1）——Gate 检验 QoS-viability 而非重调阈值。

> 生成时间: 2026-08-23 23:02:34   模式: FULL   nlevel=1（N0=N1=600）  w_ep=1000  map=True

## H=48 — matched-QoS gate（b_setup=16，N0=N1=600，w_ep=1000）

| 方法 | P_FA^nat | P_MD^nat | E[B] | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 0.0767 | 0.3533 | 41.4883 | 0.3079 |
| Direct8 (opt) | 0.1133 | 0.3117 | 34.5600 | 0.3441 |
| POTS | 0.2483 | 0.2500 | 17.0000 | 0.0000 |

- **QoS（CR，η_nat）**：P_FA^CR=0.0767，U95=0.1008（≤0.12 → PASS）；P_MD^CR=0.3533，U95=0.3924（≤0.4 → PASS）。
- **Bit（episode-paired）**：E[D_e^D8] = 6.9283 bits（CR−Direct8），95% CI [6.4097, 7.4470]（<0 → FAIL）；E[D_e^POTS] = 24.4883（[23.9815, 24.9951]）。
- **H=48 判定：FAIL**（QoS=PASS，bit=FAIL；CI 已证明不满足 → 诚实 FAIL）

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.5633 / E[B]^CR=41.4883；P_D^D8=0.4400 / E[B]^D8=34.5600；P_D^POTS=0.1510 / E[B]^POTS=17.0000。
（454s）

## H=96 — matched-QoS gate（b_setup=16，N0=N1=600，w_ep=1000）

| 方法 | P_FA^nat | P_MD^nat | E[B] | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 0.0717 | 0.2183 | 60.3850 | 0.7924 |
| Direct8 (opt) | 0.1400 | 0.2033 | 41.6400 | 0.6896 |
| POTS | 0.2483 | 0.2500 | 17.0000 | 0.0000 |

- **QoS（CR，η_nat）**：P_FA^CR=0.0717，U95=0.0951（≤0.12 → PASS）；P_MD^CR=0.2183，U95=0.2531（≤0.4 → PASS）。
- **Bit（episode-paired）**：E[D_e^D8] = 18.7450 bits（CR−Direct8），95% CI [17.6484, 19.8416]（<0 → FAIL）；E[D_e^POTS] = 43.3850（[42.0807, 44.6893]）。
- **H=96 判定：FAIL**（QoS=PASS，bit=FAIL；CI 已证明不满足 → 诚实 FAIL）

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.7517 / E[B]^CR=60.3850；P_D^D8=0.4883 / E[B]^D8=41.6400；P_D^POTS=0.1510 / E[B]^POTS=17.0000。
（727s）

## 5. b_setup regime map（secondary analysis，014 §7 预先声明）

- 理论预测（B0.4b）：root b⋆(x₀)=7；b_setup 小 ⇒ setup 便宜 ⇒ progressive/CR 赢；b_setup 大 ⇒ direct 赢；crossover 应在 b_setup≈b⋆ 附近。验证 E[D_e^D8]=E[B^CR−B^D8] 随 b_setup 的走向。H=96，N0=N1=300，w_ep=250（map 为 secondary，用轻量预算）。

| b_setup | E[B^CR] | E[B^D8] | E[B^POTS] | E[D^D8] | 95% CI | CR 省 bits? |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 27.0267 | 15.1867 | 1.0000 | 11.8400 | [10.8247, 12.8553] | no |
| 4 | 40.4633 | 22.7800 | 5.0000 | 17.6833 | [16.1797, 19.1870] | no |
| 8 | 49.2600 | 29.7067 | 9.0000 | 19.5533 | [17.9762, 21.1305] | no |
| 16 | 62.7600 | 42.4400 | 17.0000 | 20.3200 | [18.7456, 21.8944] | no |
| 32 | 70.6667 | 58.6000 | 33.0000 | 12.0667 | [10.8309, 13.3024] | no |

- **crossover 验证**：D(b_setup) 单调不减 = FAIL；存在 b_setup 使 CR 省 bits（D<0）= FAIL（b_setup=16/32 时 CR 贵 = 与主 Gate 一致；小 b_setup 是否翻转为 CR 省 bits 即理论 crossover）。

总耗时: 1661.8s

- **B0.6 结论（014 §4/§7 诚实口径）**：若 E[B]^CR ≥ E[B]^Direct8——phase transition 与 state-dependent adaptive packetization 理论成立，但在当前 b_setup=16 regime 下 optimized direct packetization 已接近或达到最优通信工作区间；CR 的保守 certified acquisition（base by default）会过度投资证据（bits），QoS 反而不输。不再改算法‘调赢’；regime map 是预先声明的 secondary analysis。

