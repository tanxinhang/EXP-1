# O-PEF MVS-B0.7-G0 — common-stop Gate：granularity 独立收益（015 §十/§十三）

> 协议（015 §十 冻结）：**N=4 exact 小系统**（GAMMA=[-1.0, 1.0, 3.0, 5.0]、levels=(1, 2, 4, 8)、r_max=8，同 B0.4a 配置加 8-bit 全包）；stratified N0=N1=120；episode 级 CRN（同一 W_e=(H_e,L_e) 驱动两分支，planner 无 随机——one-step exact 确定性）；判决阈值 η_nat=log(μ_F/μ_M)=1.0000（两方法相同）；radio cost B=Σ(b_setup+Δr_t)。

> **公共 stopping controller（015 §九/§十）**：S_λ(x,h)：CONTINUE ⟺ min_{a∈A_all} Q_λ^(1)(a|x,h) < R_λ(x)，Q_λ^(1)=c_a+E[R_λ(X')|x,a] （one-step approx，=VoIBase.q1），R_λ(x)=min{λ_M p, λ_F(1−p)}，λ=μ 为 自然工作点（μ_M/π_1、μ_F/π_0）。STOP 判定与包粒度无关（A_all 含全部 粒度），对 FG 和 Direct8 完全一致。CONTINUE 时：Direct8 限 A_D8={(i,8)}；FG 用 A_FG={(i,1),(i,2),(i,4),(i,8)}；两者都按**同一 one-step Q greedy** 选 UAV i——唯一系统差异 = feedback granularity。

> **Gate（015 §十三 G0 + B0.6-r 口径）**：两方法都算 Wilson 95% FEASIBLE/INFEASIBLE/UNCERTAIN；双方 FEASIBLE 才比较 U95(E[B^FG−B^D8])<0 → PASS（granularity 有独立收益，主线可进 G1）；否则 UNRESOLVED；若 FG 连 exact 小系统都不能降低 objective → **STOP，关闭 B0.7 主线**（015 §十三），转 015 §十四 Direct8-近优下界。

> 生成时间: 2026-08-24 01:02:46   模式: SMOKE   nlevel=1（N0=N1=120）  map=False

## H=48 — common-stop gate（b_setup=16，N0=N1=120）

### QoS 三态分类（Wilson 95%）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG (adaptive) | 0.1250 | 0.0772 | 0.1960 | 0.2250 | 0.1595 | 0.3076 | **UNCERTAIN** |
| Direct8 | 0.1250 | 0.0772 | 0.1960 | 0.2333 | 0.1667 | 0.3166 | **UNCERTAIN** |

### Bit Gate：FG vs Direct8（仅双方 FEASIBLE 可比）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE | E[T_stop] |
| --- | --- | --- | --- | --- | --- |
| FG | 1.2375 | 4.8333 | 24.6333 | — | 1.2375 |
| Direct8 | 1.2542 | 10.0333 | 30.1000 | — | 1.2542 |

- **比较被 Gate 拦住**：FG=UNCERTAIN，Direct8=UNCERTAIN——不是双方 FEASIBLE，matched 比较不成立（B0.6-r 口径）。
  → 判定 **UNRESOLVED**；按 015 §十三 G0，granularity 未在 exact 小系统上证明独立收益 → **STOP（暂）**，转 015 §十四 lower-bound 路线或以 --nlevel 扩样。

### B0.6-d 停止结构（015 §六 记账复用）

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.7625 | 0.2375 | 0.0000 | 0.0000 | 0.0000 |
| Direct8 | 0.7458 | 0.2542 | 0.0000 | 0.0000 | 0.0000 |

- 分解：b_setup=16。FG E[N_tx]=1.2375 vs D8 1.2542；payload 4.8333 vs 10.0333——ΔB^FG−D8 中 setup/payload 各自贡献，直接定位 granularity 的 成本结构效应。

- **NP-matched（P_FA=0.05）secondary**：P_D^FG=0.5625 / E[B]^FG=24.6333；P_D^D8=0.5583 / E[B]^D8=30.1000。
（11s）

## H=96 — common-stop gate（b_setup=16，N0=N1=120）

### QoS 三态分类（Wilson 95%）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG (adaptive) | 0.1500 | 0.0970 | 0.2247 | 0.1833 | 0.1243 | 0.2620 | **UNCERTAIN** |
| Direct8 | 0.1583 | 0.1038 | 0.2341 | 0.1750 | 0.1174 | 0.2528 | **UNCERTAIN** |

### Bit Gate：FG vs Direct8（仅双方 FEASIBLE 可比）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | SE | E[T_stop] |
| --- | --- | --- | --- | --- | --- |
| FG | 1.3167 | 5.1042 | 26.1708 | — | 1.3167 |
| Direct8 | 1.3458 | 10.7667 | 32.3000 | — | 1.3458 |

- **比较被 Gate 拦住**：FG=UNCERTAIN，Direct8=UNCERTAIN——不是双方 FEASIBLE，matched 比较不成立（B0.6-r 口径）。
  → 判定 **UNRESOLVED**；按 015 §十三 G0，granularity 未在 exact 小系统上证明独立收益 → **STOP（暂）**，转 015 §十四 lower-bound 路线或以 --nlevel 扩样。

### B0.6-d 停止结构（015 §六 记账复用）

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.7625 | 0.1708 | 0.0542 | 0.0125 | 0.0000 |
| Direct8 | 0.7458 | 0.1750 | 0.0667 | 0.0125 | 0.0000 |

- 分解：b_setup=16。FG E[N_tx]=1.3167 vs D8 1.3458；payload 5.1042 vs 10.7667——ΔB^FG−D8 中 setup/payload 各自贡献，直接定位 granularity 的 成本结构效应。

- **NP-matched（P_FA=0.05）secondary**：P_D^FG=0.5500 / E[B]^FG=26.1708；P_D^D8=0.5333 / E[B]^D8=32.3000。
（12s）

总耗时: 24.5s

- **B0.7-G0 结论（015 §十三）**：common-stop Gate 把 granularity 从 stopping/UAV-selection/budget/decision-threshold 中隔离出来。若 FG 在 exact 小系统上双方 QoS FEASIBLE 且 U95(E[B^FG−B^D8])<0 → granularity 有独立价值，继续 B0.7-G1（N=8 held-out QoS-dual calibration）；否则 **STOP，关闭 performance-improvement 主线**，B0.5 换用途为 Direct8-近优下界 V_LB≤V⋆≤V^D8（015 §十四）。

