# O-PEF MVS-B0.6 — matched-QoS：CR vs optimized Direct8（论文生死 Gate，014 §4-§6）

> 协议（014 §5-§6 冻结）：**stratified N0=N1=120**（H0/H1 独立采样）；episode 级 **CRN**——同一 physical world W_e=(H_e,L_e) 给 CR/Direct8/POTS，planner RNG 独立；**radio cost 与 planning cost 分离**（B_radio=Σ(b_setup+Δr_t)，CPI worlds 只算 compute）；判决阈值 η_nat=log(μ_F/μ_M)=1.0000（T21）；CR = 冻结的 SNR anchor + Operational-CPI（betting，w_ep=250，δ_t 按决策序号）；Direct8 = **optimized**（SNR-order full packets + η_nat stop；direct_only planner 在 H=96 不可行——4 个全包 → 256⁴ cells）；POTS = round-robin 渐进（第二 comparator）。Gate：U95(P_FA^CR)≤0.12、U95(P_MD^CR)≤0.4、U95(E[D_e^D8])<0（D_e^D8=B_e^CR−B_e^D8，episode-paired）。
- α=0.12, β=0.4 的选取：natural Bayes 判决的工作区（G5 实测 P_FA^nat≈0.09、P_MD^nat≈0.22-0.37 @ η=1）——Gate 检验 QoS-viability 而非重调阈值。

> 生成时间: 2026-08-23 23:36:12   模式: SMOKE   nlevel=1（N0=N1=120）  w_ep=250  map=True

## H=48 — matched-QoS gate（b_setup=16，N0=N1=120，w_ep=250）

| 方法 | P_FA^nat | P_MD^nat | E[B] | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 0.0833 | 0.3500 | 42.7000 | 0.6440 |
| Direct8 (opt) | 0.1250 | 0.3167 | 35.1000 | 0.7740 |
| POTS | 0.2833 | 0.2833 | 17.0000 | 0.0000 |

- **QoS（CR，η_nat）**：P_FA^CR=0.0833，U95=0.1466（≤0.12 → FAIL）；P_MD^CR=0.3500，U95=0.4388（≤0.4 → FAIL）。
- **Bit（episode-paired）**：E[D_e^D8] = 7.6000 bits（CR−Direct8），95% CI [6.4075, 8.7925]（<0 → FAIL）；E[D_e^POTS] = 25.7000（[24.6367, 26.7633]）。
- **H=48 判定：FAIL**（QoS=FAIL，bit=FAIL；CI 已证明不满足 → 诚实 FAIL）

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.5333 / E[B]^CR=42.7000；P_D^D8=0.4583 / E[B]^D8=35.1000；P_D^POTS=0.1265 / E[B]^POTS=17.0000。
（24s）

## H=96 — matched-QoS gate（b_setup=16，N0=N1=120，w_ep=250）

| 方法 | P_FA^nat | P_MD^nat | E[B] | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 0.0667 | 0.2333 | 64.0000 | 1.8459 |
| Direct8 (opt) | 0.1417 | 0.2167 | 42.4000 | 1.5554 |
| POTS | 0.2833 | 0.2833 | 17.0000 | 0.0000 |

- **QoS（CR，η_nat）**：P_FA^CR=0.0667，U95=0.1261（≤0.12 → FAIL）；P_MD^CR=0.2333，U95=0.3166（≤0.4 → PASS）。
- **Bit（episode-paired）**：E[D_e^D8] = 21.6000 bits（CR−Direct8），95% CI [18.9753, 24.2247]（<0 → FAIL）；E[D_e^POTS] = 47.0000（[43.9519, 50.0481]）。
- **H=96 判定：FAIL**（QoS=FAIL，bit=FAIL；CI 已证明不满足 → 诚实 FAIL）

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.7500 / E[B]^CR=64.0000；P_D^D8=0.5167 / E[B]^D8=42.4000；P_D^POTS=0.1265 / E[B]^POTS=17.0000。
（36s）

## 5. b_setup regime map（secondary analysis，014 §7 预先声明）

- 理论预测（B0.4b）：root b⋆(x₀)=7；b_setup 小 ⇒ setup 便宜 ⇒ progressive/CR 赢；b_setup 大 ⇒ direct 赢；crossover 应在 b_setup≈b⋆ 附近。验证 E[D_e^D8]=E[B^CR−B^D8] 随 b_setup 的走向。H=96，N0=N1=120，w_ep=250（map 为 secondary，用轻量预算）。

| b_setup | E[B^CR] | E[B^D8] | E[B^POTS] | E[D^D8] | 95% CI | CR 省 bits? |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 29.1667 | 15.1667 | 1.0000 | 14.0000 | [12.3322, 15.6678] | no |
| 4 | 42.9083 | 22.7500 | 5.0000 | 20.1583 | [17.7114, 22.6052] | no |
| 8 | 50.7500 | 29.6000 | 9.0000 | 21.1500 | [18.6375, 23.6625] | no |
| 16 | 64.0000 | 42.4000 | 17.0000 | 21.6000 | [18.9753, 24.2247] | no |
| 32 | 71.1667 | 58.5000 | 33.0000 | 12.6667 | [10.6792, 14.6541] | no |

- **crossover 验证**：D(b_setup) 单调不减 = FAIL；存在 b_setup 使 CR 省 bits（D<0）= FAIL（b_setup=16/32 时 CR 贵 = 与主 Gate 一致；小 b_setup 是否翻转为 CR 省 bits 即理论 crossover）。

总耗时: 254.4s

- **B0.6 结论（014 §4/§7 诚实口径）**：若 E[B]^CR ≥ E[B]^Direct8——phase transition 与 state-dependent adaptive packetization 理论成立，但在当前 b_setup=16 regime 下 optimized direct packetization 已接近或达到最优通信工作区间；CR 的保守 certified acquisition（base by default）会过度投资证据（bits），QoS 反而不输。不再改算法‘调赢’；regime map 是预先声明的 secondary analysis。

