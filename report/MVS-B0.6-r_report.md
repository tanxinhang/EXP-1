# O-PEF MVS-B0.6-r — matched-QoS 口径纠偏 + bit 成本分解（依据 015）

> 协议（014 §5-§6 冻结，015 §5 修正语义）：stratified N0=N1=600；episode 级 CRN（同一 W_e=(H_e,L_e) 给 CR/Direct8/POTS）；radio/planning cost 分离（B_radio=Σ(b_setup+Δr_t)，CPI worlds 只算 compute）；η_nat=log(μ_F/μ_M)=1.0000（T21）；CR = 冻结 SNR anchor + Operational-CPI（betting，w_ep=1000，δ_t 按决策序号）；Direct8 = SNR-order full packets + η_nat stop；POTS = round-robin 渐进。α=0.12、β=0.4（natural Bayes 工作区，同 B0.6）。

> **015 口径修正（本版本只改语义与记账，不碰 planner/CPI/阈值）**：
- **matched-QoS 语义**：QoS CI（Wilson 95%）对 **三个方法** 都计算并分类 FEASIBLE / INFEASIBLE / UNCERTAIN；**只有双方都 FEASIBLE 才允许比较 E[B^A]−E[B^B]**（015 §5：A≺B ⟺ A,B∈F_QoS ∧ U95(E[B^A−B^B])<0）。
- **判定措辞**：CR 的 bit 结论降级为 **COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**——Direct8 有更低 raw cost 但未被认证 QoS-feasible 时，只写 'Direct8 has lower raw communication cost but is not certified QoS-feasible at this operating point'，不写论文生死 FAIL。
- **regime-map 表述**：B0.4b 只证明状态局部的 packetization 相变 （g_x(b)=Q_prog−Q_dir=E[min{Y_x,b}]，b*_x state-dependent），**不证明 episode 级全局 D(b)=E[B^CR−B^D8] 单调或必在 b_setup≈b*₀ 附近 crossover**（015 §2：全局量混入 state occupancy / stopping time / UAV selection / remaining budget / CPI override）。地图改称 **system-level regime diagnostic: no global crossover observed**。
- **B0.6-d 成本分解（015 §六）**：逐 episode 记账 N_tx（反馈包数）与 B_payload（ΣΔr），断言恒等式 B = b_setup·N_tx + B_payload；报告 E[N_tx]、E[B_payload]、E[T_stop]、P(T_stop=k)——验证 'CR 贵在过多 evidence payload，而非 transaction 碎片'。

> 生成时间: 2026-08-24 00:19:40   模式: FULL   nlevel=1（N0=N1=600）  w_ep=1000  map=True

## H=48 — QoS 口径纠偏 gate（b_setup=16，N0=N1=600，w_ep=1000）

### QoS 三态分类（Wilson 95%，015 §5——三方法都算，不只 CR）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CR | 0.0767 | 0.0580 | 0.1008 | 0.3533 | 0.3161 | 0.3924 | **FEASIBLE** |
| Direct8(opt) | 0.1133 | 0.0904 | 0.1412 | 0.3117 | 0.2759 | 0.3498 | **UNCERTAIN** |
| POTS | 0.2483 | 0.2154 | 0.2844 | 0.2500 | 0.2170 | 0.2862 | **INFEASIBLE** |

- CR 分类 = **FEASIBLE**；Direct8 分类 = **UNCERTAIN**。（B0.6 只认 CR 的 QoS → 本次把 D8/POTS 也纳入 Gate。）

### Bit Gate（015 §5：仅当双方 FEASIBLE 才可比 E[B]）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 1.7292 | 13.8217 | 41.4883 | 0.3079 |
| Direct8 (opt) | 1.4400 | 11.5200 | 34.5600 | 0.3441 |
| POTS | 1.0000 | 1.0000 | 17.0000 | 0.0000 |

- **比较被 Gate 拦住**：CR=FEASIBLE，Direct8=UNCERTAIN。按 015 §5，双方未都 FEASIBLE 时 **不允许输出 'CR bit FAIL'**，只能写：

  > **Direct8 has lower raw communication cost but is not certified QoS-feasible at this operating point.**

- 判定：**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**（B0.6 的 FAIL 降级——先证明 D8 的 QoS 达标，matched 比较才成立；见 015 §1）。

### B0.6-d 停止与成本结构（015 §六）

| 方法 | E[T_stop] | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- | --- |
| CR | 1.7292 | 0.2708 | 0.7292 | 0.0000 | 0.0000 | 0.0000 |
| Direct8(opt) | 1.4400 | 0.5600 | 0.4400 | 0.0000 | 0.0000 | 0.0000 |
| POTS | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **分解诊断**：b_setup=16。CR E[N_tx]=1.7292 vs D8 1.4400 → setup 部分 16·E[N_tx] 分别 27.6667 / 23.0400；payload 部分 13.8217 / 11.5200。若 ΔB^CR−D8 中 payload 贡献 > setup 贡献，即证实 **CR 贵不是因为包太碎（transaction/setup 重复付费），而是由于总共采集了过多 evidence payload**（015 §六预测）。

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.5633 / E[B]^CR=41.4883；P_D^D8=0.4400 / E[B]^D8=34.5600；P_D^POTS=0.1510 / E[B]^POTS=17.0000。
（463s）

## H=96 — QoS 口径纠偏 gate（b_setup=16，N0=N1=600，w_ep=1000）

### QoS 三态分类（Wilson 95%，015 §5——三方法都算，不只 CR）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CR | 0.0717 | 0.0536 | 0.0951 | 0.2183 | 0.1871 | 0.2531 | **FEASIBLE** |
| Direct8(opt) | 0.1400 | 0.1145 | 0.1701 | 0.2033 | 0.1731 | 0.2374 | **UNCERTAIN** |
| POTS | 0.2483 | 0.2154 | 0.2844 | 0.2500 | 0.2170 | 0.2862 | **INFEASIBLE** |

- CR 分类 = **FEASIBLE**；Direct8 分类 = **UNCERTAIN**。（B0.6 只认 CR 的 QoS → 本次把 D8/POTS 也纳入 Gate。）

### Bit Gate（015 §5：仅当双方 FEASIBLE 才可比 E[B]）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 2.5208 | 20.0517 | 60.3850 | 0.7924 |
| Direct8 (opt) | 1.7350 | 13.8800 | 41.6400 | 0.6896 |
| POTS | 1.0000 | 1.0000 | 17.0000 | 0.0000 |

- **比较被 Gate 拦住**：CR=FEASIBLE，Direct8=UNCERTAIN。按 015 §5，双方未都 FEASIBLE 时 **不允许输出 'CR bit FAIL'**，只能写：

  > **Direct8 has lower raw communication cost but is not certified QoS-feasible at this operating point.**

- 判定：**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**（B0.6 的 FAIL 降级——先证明 D8 的 QoS 达标，matched 比较才成立；见 015 §1）。

### B0.6-d 停止与成本结构（015 §六）

| 方法 | E[T_stop] | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- | --- |
| CR | 2.5208 | 0.2325 | 0.3092 | 0.1633 | 0.2950 | 0.0000 |
| Direct8(opt) | 1.7350 | 0.5600 | 0.2475 | 0.0900 | 0.1025 | 0.0000 |
| POTS | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **分解诊断**：b_setup=16。CR E[N_tx]=2.5208 vs D8 1.7350 → setup 部分 16·E[N_tx] 分别 40.3333 / 27.7600；payload 部分 20.0517 / 13.8800。若 ΔB^CR−D8 中 payload 贡献 > setup 贡献，即证实 **CR 贵不是因为包太碎（transaction/setup 重复付费），而是由于总共采集了过多 evidence payload**（015 §六预测）。

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.7517 / E[B]^CR=60.3850；P_D^D8=0.4883 / E[B]^D8=41.6400；P_D^POTS=0.1510 / E[B]^POTS=17.0000。
（673s）

## 5. b_setup regime map — system-level regime diagnostic（015 §2、§六）

- **表述修正（015 §2）**：B0.4b 证明的是**状态局部**相变 （b*₀(x₀)=7, g'ₓ(b)=P(additional transaction)），并**不**蕴含 episode 级全局 D(b)=E[B^CR−B^D8] 单调、更不蕴含必在 b_setup≈b*₀ 发生全局 crossover（全局量混入 state occupancy / stopping time / UAV selection / remaining budget / CPI override）。因此本图是 **system-level regime diagnostic**，只观察 'no global crossover observed'，不写成 'theory-predicted crossover failed'。
- 同时给出分解（015 §六）：B = b_setup·N_tx + B_payload，验证 ΔE[B] 的构成。H=96，N0=N1=300，w_ep=250（map 为 secondary，轻量预算）。

| b_setup | E[N_tx^CR] | E[N_tx^D8] | E[B_pay^CR] | E[B_pay^D8] | E[D^D8] | 95% CI | CR 省 bits? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.3883 | 1.8983 | 27.0267 | 15.1867 | 11.8400 | [10.8247, 12.8553] | no |
| 4 | 3.3783 | 1.8983 | 26.9500 | 15.1867 | 17.6833 | [16.1797, 19.1870] | no |
| 8 | 3.0800 | 1.8567 | 24.6200 | 14.8533 | 19.5533 | [17.9762, 21.1305] | no |
| 16 | 2.6150 | 1.7683 | 20.9200 | 14.1467 | 20.3200 | [18.7456, 21.8944] | no |
| 32 | 1.7667 | 1.4650 | 14.1333 | 11.7200 | 12.0667 | [10.8309, 13.3024] | no |

- **system-level 观察**：样本路径上 D(b_setup) 未出现符号翻转 （存在 b_setup 使 CR 省 bits，D<0 = FAIL；单调非减 = FAIL）→ **system-level regime diagnostic: no global crossover observed**（015 §2：不写成 'theory-predicted crossover failed'，因为 B0.4b 只证明状态局部相变）。
- **分解读法（015 §六）**：b_setup=0 行 setup 完全免费，ΔE[B] 只剩 payload 差——若该行 E[D^D8]>0 且 E[B_pay^CR]>E[B_pay^D8]，即证实 **CR 贵在 evidence payload 过量而非 transaction 碎片**（setup 归一后 仍多付）。
总耗时: 1591.3s

- **B0.6-r 结论（015 诚实口径）**：matched-QoS 比较的先决条件是 **所有 参与比较的方法都 certified QoS-feasible**。当前 Direct8/POTS 的 QoS CI 显示它们未被认证（或在边界），因此 B0.6 的结论严格重述为：**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**——Direct8 有 更低 raw communication cost 但未被认证 QoS-feasible。成本分解（B0.6-d）将进一步定位 CR 多支出的来源（setup 重复 vs evidence payload 过量），为 B0.7 common-stop Gate（015 §十）提供前置证据。

