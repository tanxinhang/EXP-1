# O-PEF MVS-B0.7-G1r — forced-continuation audit + conservative S_ref Gate + dual-Q regression（016 §1-§6、§10、§15 路线 1-3）

> 协议：N=8（GAMMA=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]、levels=(1, 2, 4, 8)、r_max=8）；b_setup=16.0；stratified N0=N1=600（calibration N_CAL=300 @ H=96，test N_TEST=600 @ H∈(48, 96)）；episode 级 CRN（同一 W_e=(H_e,L_e) 给 FG/D8；planner 确定性）；grid 冻结、只在 calibration 用（016 §9）。

> **016 P0（§1）**：G1 的公共停止器 S_common 用 min_{a∈A_FG}Q<R_λ 判定，而 D8 只能从 A_D8={(i,8)} 选动作——存在 F(x)=1{q_FG<R_λ≤q_D8} 状态（小包值得买 → 公共控制器说继续，但 8-bit 包已不值得 → D8 被迫发 8-bit）。A_FG 本身含 granularity 信息 ⇒ 'STOP 判定与包粒度无关' 表述不成立。

> **G1r-A（016 §15-1）**：审计 F 频率与 ΔB_forced（D8 被强迫支付的通信）。
> **G1r-B（016 §4/§15-2）**：保守停止器 S_ref：CONTINUE iff min_{a∈A_D8}Q_λ^(1)<R_λ；两方法共用（对 FG 更苛刻——只有'至少一个 Direct8 full packet 值得发送'才给 FG 一次 adaptive-granularity 机会）。
> **G1r-C（016 §15-3）**：q1_fast vs 独立 generic dual-Q exact 回归；D8 emulation invariant（FG 限定 A_D8 ≡ D8 分支）。

> **统计（016 §10）**：paired bit 用 **one-sided paired Hoeffding U=0.0000 公式 U=D̄+2H√(log(1/δ)/2n)**（D∈[-H,H]，分布无关、无 t 假设）作为正式上界；t 版仅参考。QoS：Wilson 95%。Gate = intersection-union （各 component 按预设 level 控制，全条件成立才 PASS）。

> 生成时间: 2026-08-24 03:14:08   模式: FULL   nlevel=1（N_TEST=600，N_CAL=300）

## 0. G1r-C — 代码可信度封板（016 §15-3）

- q1_fast vs generic dual-Q exact：检查 168 个 (i,r2) 动作，max|Δ|=0.000000000000（<1e-9 → PASS）。
- D8 emulation invariant：50 个 episode 上 'FG 限定 A_D8' 与 D8 分支 逐样本完全一致（lam/cost/N_tx/payload）→ **PASS**。
（1.6s）

## 1. Calibration（016 §9：grid 冻结、仅 calibration）

- **S_common（G1 语义）η_star = 1.2**（达标集 [1.0, 1.2]，选 E[B^FG]+E[B^D8] 最小）。
- **S_ref（G1r-B 保守）η_star = 1.0**（达标集 [1.0]）。
- 两停止器达标 η 交集/并集：[1.0, 1.2]（S_common/S_ref 分别标注）。

## 2. G1r-A — Forced-Continuation Audit（G1 语义，η_star 冻结）

### H=48（η_star=1.2 冻结，test fresh）

- **P(F=1)（按决策状态）** = 0.0221（F 状态 41/1856）；P(episode contains F) = 0.0342；ΔB_forced = 984.0000 bits（D8 因 S_common 被迫支付的 8-bit 成本）。
- E[B^FG]=27.0967、E[B^D8]=37.1200，E[D]=-10.0233（t 参考 CI [-10.4716, -9.5750]，Hoeffding U95=-6.6316）。

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0500 | 0.0705 | 0.4083 | 0.4481 | UNCERTAIN | 1.4800 | 3.4167 |
| Direct8 | 0.0483 | 0.0686 | 0.3733 | 0.4127 | UNCERTAIN | 1.5467 | 12.3733 |

### H=96（η_star=1.2 冻结，test fresh）

- **P(F=1)（按决策状态）** = 0.0277（F 状态 65/2350）；P(episode contains F) = 0.0542；ΔB_forced = 1560.0000 bits（D8 因 S_common 被迫支付的 8-bit 成本）。
- E[B^FG]=34.6900、E[B^D8]=47.0000，E[D]=-12.3100（t 参考 CI [-13.2604, -11.3596]，Hoeffding U95=-5.5266）。

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0333 | 0.0509 | 0.3567 | 0.3958 | FEASIBLE | 1.8808 | 4.5967 |
| Direct8 | 0.0400 | 0.0588 | 0.3133 | 0.3515 | FEASIBLE | 1.9583 | 15.6667 |

## 3. G1r-B — conservative S_ref Gate（016 §4）

### H=48（η_star_ref=1.0 冻结，test fresh）

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] | E[T_stop] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0517 | 0.0724 | 0.3783 | 0.4178 | UNCERTAIN | 1.4800 | 3.4167 | 27.0967 | 1.4800 |
| Direct8 | 0.0600 | 0.0819 | 0.3317 | 0.3703 | FEASIBLE | 1.5267 | 12.2133 | 36.6400 | 1.5267 |

- **Gate（intersection-union，016 §10）**：QoS 双方 FEASIBLE = FAIL（FG=UNCERTAIN、D8=FEASIBLE）；paired Hoeffding U95(E[D])=-6.1516（<0 → PASS）。
  → **UNRESOLVED / FAIL**：016 §15 判定——若 P(F≈1) 不小或 S_ref 下不显著，则 016 §4 预期（-12.31 → -5..-10）落空，按 016 §15 转 lower-bound 或先扩样。

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.5200 | 0.4800 | 0.0000 | 0.0000 | 0.0000 |
| Direct8 | 0.4733 | 0.5267 | 0.0000 | 0.0000 | 0.0000 |

- 分解（S_ref）：E[D]=-9.5433；setup 部分 16.0·(E[N_tx^FG]−E[N_tx^D8])=-0.7467，payload 部分 -8.7967。
- NP-matched（P_FA=0.05）secondary：P_D^FG=0.5917 / E[B]^FG=27.0967；P_D^D8=0.6200 / E[B]^D8=36.6400。
（ 140.8s 累计）

### H=96（η_star_ref=1.0 冻结，test fresh）

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] | E[T_stop] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0467 | 0.0666 | 0.3317 | 0.3703 | FEASIBLE | 1.8125 | 4.3600 | 33.3600 | 1.8125 |
| Direct8 | 0.0550 | 0.0762 | 0.2817 | 0.3190 | FEASIBLE | 1.8875 | 15.1000 | 45.3000 | 1.8875 |

- **Gate（intersection-union，016 §10）**：QoS 双方 FEASIBLE = PASS（FG=FEASIBLE、D8=FEASIBLE）；paired Hoeffding U95(E[D])=-5.1566（<0 → PASS）。
  → **G1r-B PASS**：保守 common-stop（仅当 D8 full packet 值得才给 FG adaptive 机会）下仍 U95(E[B^FG−B^D8])<0 → **granularity 独立 收益基本无法从公平性击穿**（016 §4）→ 值得投入 fresh G2 （B0.7-G2，016 §15-4）。

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.5075 | 0.2875 | 0.1058 | 0.0833 | 0.0158 |
| Direct8 | 0.4883 | 0.2600 | 0.1275 | 0.1242 | 0.0000 |

- 分解（S_ref）：E[D]=-11.9400；setup 部分 16.0·(E[N_tx^FG]−E[N_tx^D8])=-1.2000，payload 部分 -10.7400。
- NP-matched（P_FA=0.05）secondary：P_D^FG=0.6700 / E[B]^FG=33.3600；P_D^D8=0.7017 / E[B]^D8=45.3000。
（ 154.5s 累计）

总耗时: 154.5s

- **B0.7-G1r 结论（016 §15）**：G1r-A 量化 action-leakage（ΔB_forced / P(F=1)）——若很小，则 G1 的 D8 劣势主要是 granularity 真实机制；G1r-B 用 A_D8-reference 的保守 common-stop 重验 U95(E[B^FG−B^D8])<0——若仍 显著，granularity 独立收益在公平性上站稳（016 §4 预期 -5..-10 bit），随后投入 **B0.7-G2**：FG/D8 **分别**在 calibration 上优化自己的 (ρ,η)（016 §7/§9：J_m⋆=inf E_π[B] s.t. QoS，各自 controller），test 完全 fresh，暂不加 CPI；最后 B0.7-G3 重构 DualCPI 使 certificate 与 当前 dual objective 一致（016 §5/§15-5）。

