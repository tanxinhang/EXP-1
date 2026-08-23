# O-PEF MVS-B0.6-r — matched-QoS 口径纠偏 + bit 成本分解（依据 015）

> 协议（014 §5-§6 冻结，015 §5 修正语义）：stratified N0=N1=120；episode 级 CRN（同一 W_e=(H_e,L_e) 给 CR/Direct8/POTS）；radio/planning cost 分离（B_radio=Σ(b_setup+Δr_t)，CPI worlds 只算 compute）；η_nat=log(μ_F/μ_M)=1.0000（T21）；CR = 冻结 SNR anchor + Operational-CPI（betting，w_ep=250，δ_t 按决策序号）；Direct8 = SNR-order full packets + η_nat stop；POTS = round-robin 渐进。α=0.12、β=0.4（natural Bayes 工作区，同 B0.6）。

> **015 口径修正（本版本只改语义与记账，不碰 planner/CPI/阈值）**：
- **matched-QoS 语义**：QoS CI（Wilson 95%）对 **三个方法** 都计算并分类 FEASIBLE / INFEASIBLE / UNCERTAIN；**只有双方都 FEASIBLE 才允许比较 E[B^A]−E[B^B]**（015 §5：A≺B ⟺ A,B∈F_QoS ∧ U95(E[B^A−B^B])<0）。
- **判定措辞**：CR 的 bit 结论降级为 **COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**——Direct8 有更低 raw cost 但未被认证 QoS-feasible 时，只写 'Direct8 has lower raw communication cost but is not certified QoS-feasible at this operating point'，不写论文生死 FAIL。
- **regime-map 表述**：B0.4b 只证明状态局部的 packetization 相变 （g_x(b)=Q_prog−Q_dir=E[min{Y_x,b}]，b*_x state-dependent），**不证明 episode 级全局 D(b)=E[B^CR−B^D8] 单调或必在 b_setup≈b*₀ 附近 crossover**（015 §2：全局量混入 state occupancy / stopping time / UAV selection / remaining budget / CPI override）。地图改称 **system-level regime diagnostic: no global crossover observed**。
- **B0.6-d 成本分解（015 §六）**：逐 episode 记账 N_tx（反馈包数）与 B_payload（ΣΔr），断言恒等式 B = b_setup·N_tx + B_payload；报告 E[N_tx]、E[B_payload]、E[T_stop]、P(T_stop=k)——验证 'CR 贵在过多 evidence payload，而非 transaction 碎片'。

> 生成时间: 2026-08-24 00:16:31   模式: SMOKE   nlevel=1（N0=N1=120）  w_ep=250  map=False

## H=48 — QoS 口径纠偏 gate（b_setup=16，N0=N1=120，w_ep=250）

### QoS 三态分类（Wilson 95%，015 §5——三方法都算，不只 CR）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CR | 0.0833 | 0.0459 | 0.1466 | 0.3500 | 0.2705 | 0.4388 | **UNCERTAIN** |
| Direct8(opt) | 0.1250 | 0.0772 | 0.1960 | 0.3167 | 0.2402 | 0.4045 | **UNCERTAIN** |
| POTS | 0.2833 | 0.2104 | 0.3697 | 0.2833 | 0.2104 | 0.3697 | **INFEASIBLE** |

- CR 分类 = **UNCERTAIN**；Direct8 分类 = **UNCERTAIN**。（B0.6 只认 CR 的 QoS → 本次把 D8/POTS 也纳入 Gate。）

### Bit Gate（015 §5：仅当双方 FEASIBLE 才可比 E[B]）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 1.7792 | 14.2333 | 42.7000 | 0.6440 |
| Direct8 (opt) | 1.4625 | 11.7000 | 35.1000 | 0.7740 |
| POTS | 1.0000 | 1.0000 | 17.0000 | 0.0000 |

- **比较被 Gate 拦住**：CR=UNCERTAIN，Direct8=UNCERTAIN。按 015 §5，双方未都 FEASIBLE 时 **不允许输出 'CR bit FAIL'**，只能写：

  > **Direct8 has lower raw communication cost but is not certified QoS-feasible at this operating point.**

- 判定：**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**（B0.6 的 FAIL 降级——先证明 D8 的 QoS 达标，matched 比较才成立；见 015 §1）。

### B0.6-d 停止与成本结构（015 §六）

| 方法 | E[T_stop] | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- | --- |
| CR | 1.7792 | 0.2208 | 0.7792 | 0.0000 | 0.0000 | 0.0000 |
| Direct8(opt) | 1.4625 | 0.5375 | 0.4625 | 0.0000 | 0.0000 | 0.0000 |
| POTS | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **分解诊断**：b_setup=16。CR E[N_tx]=1.7792 vs D8 1.4625 → setup 部分 16·E[N_tx] 分别 28.4667 / 23.4000；payload 部分 14.2333 / 11.7000。若 ΔB^CR−D8 中 payload 贡献 > setup 贡献，即证实 **CR 贵不是因为包太碎（transaction/setup 重复付费），而是由于总共采集了过多 evidence payload**（015 §六预测）。

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.5333 / E[B]^CR=42.7000；P_D^D8=0.4583 / E[B]^D8=35.1000；P_D^POTS=0.1265 / E[B]^POTS=17.0000。
（25s）

## H=96 — QoS 口径纠偏 gate（b_setup=16，N0=N1=120，w_ep=250）

### QoS 三态分类（Wilson 95%，015 §5——三方法都算，不只 CR）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CR | 0.0667 | 0.0342 | 0.1261 | 0.2333 | 0.1667 | 0.3166 | **UNCERTAIN** |
| Direct8(opt) | 0.1417 | 0.0904 | 0.2152 | 0.2167 | 0.1524 | 0.2985 | **UNCERTAIN** |
| POTS | 0.2833 | 0.2104 | 0.3697 | 0.2833 | 0.2104 | 0.3697 | **INFEASIBLE** |

- CR 分类 = **UNCERTAIN**；Direct8 分类 = **UNCERTAIN**。（B0.6 只认 CR 的 QoS → 本次把 D8/POTS 也纳入 Gate。）

### Bit Gate（015 §5：仅当双方 FEASIBLE 才可比 E[B]）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE(E[B]) |
| --- | --- | --- | --- | --- |
| CR | 2.6667 | 21.3333 | 64.0000 | 1.8459 |
| Direct8 (opt) | 1.7667 | 14.1333 | 42.4000 | 1.5554 |
| POTS | 1.0000 | 1.0000 | 17.0000 | 0.0000 |

- **比较被 Gate 拦住**：CR=UNCERTAIN，Direct8=UNCERTAIN。按 015 §5，双方未都 FEASIBLE 时 **不允许输出 'CR bit FAIL'**，只能写：

  > **Direct8 has lower raw communication cost but is not certified QoS-feasible at this operating point.**

- 判定：**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**（B0.6 的 FAIL 降级——先证明 D8 的 QoS 达标，matched 比较才成立；见 015 §1）。

### B0.6-d 停止与成本结构（015 §六）

| 方法 | E[T_stop] | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- | --- |
| CR | 2.6667 | 0.2208 | 0.2667 | 0.1375 | 0.3750 | 0.0000 |
| Direct8(opt) | 1.7667 | 0.5375 | 0.2667 | 0.0875 | 0.1083 | 0.0000 |
| POTS | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **分解诊断**：b_setup=16。CR E[N_tx]=2.6667 vs D8 1.7667 → setup 部分 16·E[N_tx] 分别 42.6667 / 28.2667；payload 部分 21.3333 / 14.1333。若 ΔB^CR−D8 中 payload 贡献 > setup 贡献，即证实 **CR 贵不是因为包太碎（transaction/setup 重复付费），而是由于总共采集了过多 evidence payload**（015 §六预测）。

- **NP-matched（P_FA=0.05）secondary**：P_D^CR=0.7500 / E[B]^CR=64.0000；P_D^D8=0.5167 / E[B]^D8=42.4000；P_D^POTS=0.1265 / E[B]^POTS=17.0000。
（36s）

总耗时: 65.3s

- **B0.6-r 结论（015 诚实口径）**：matched-QoS 比较的先决条件是 **所有 参与比较的方法都 certified QoS-feasible**。当前 Direct8/POTS 的 QoS CI 显示它们未被认证（或在边界），因此 B0.6 的结论严格重述为：**COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**——Direct8 有 更低 raw communication cost 但未被认证 QoS-feasible。成本分解（B0.6-d）将进一步定位 CR 多支出的来源（setup 重复 vs evidence payload 过量），为 B0.7 common-stop Gate（015 §十）提供前置证据。

