# O-PEF MVS-B0.7-G1r — forced-continuation audit + conservative S_ref Gate + dual-Q regression（016 §1-§6、§10、§15 路线 1-3）

> 协议：N=8（GAMMA=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]、levels=(1, 2, 4, 8)、r_max=8）；b_setup=16.0；stratified N0=N1=120（calibration N_CAL=60 @ H=96，test N_TEST=120 @ H∈(48, 96)）；episode 级 CRN（同一 W_e=(H_e,L_e) 给 FG/D8；planner 确定性）；grid 冻结、只在 calibration 用（016 §9）。

> **016 P0（§1）**：G1 的公共停止器 S_common 用 min_{a∈A_FG}Q<R_λ 判定，而 D8 只能从 A_D8={(i,8)} 选动作——存在 F(x)=1{q_FG<R_λ≤q_D8} 状态（小包值得买 → 公共控制器说继续，但 8-bit 包已不值得 → D8 被迫发 8-bit）。A_FG 本身含 granularity 信息 ⇒ 'STOP 判定与包粒度无关' 表述不成立。

> **G1r-A（016 §15-1）**：审计 F 频率与 ΔB_forced（D8 被强迫支付的通信）。
> **G1r-B（016 §4/§15-2）**：保守停止器 S_ref：CONTINUE iff min_{a∈A_D8}Q_λ^(1)<R_λ；两方法共用（对 FG 更苛刻——只有'至少一个 Direct8 full packet 值得发送'才给 FG 一次 adaptive-granularity 机会）。
> **G1r-C（016 §15-3）**：q1_fast vs 独立 generic dual-Q exact 回归；D8 emulation invariant（FG 限定 A_D8 ≡ D8 分支）。

> **统计（016 §10）**：paired bit 用 **one-sided paired Hoeffding U=0.0000 公式 U=D̄+2H√(log(1/δ)/2n)**（D∈[-H,H]，分布无关、无 t 假设）作为正式上界；t 版仅参考。QoS：Wilson 95%。Gate = intersection-union （各 component 按预设 level 控制，全条件成立才 PASS）。

> 生成时间: 2026-08-24 03:13:06   模式: SMOKE   nlevel=1（N_TEST=120，N_CAL=60）

## 0. G1r-C — 代码可信度封板（016 §15-3）

- q1_fast vs generic dual-Q exact：检查 168 个 (i,r2) 动作，max|Δ|=0.000000000000（<1e-9 → PASS）。
- D8 emulation invariant：50 个 episode 上 'FG 限定 A_D8' 与 D8 分支 逐样本完全一致（lam/cost/N_tx/payload）→ **PASS**。
（1.5s）

## 1. Calibration（016 §9：grid 冻结、仅 calibration）

- 校准 UNRESOLVED（n 不足）：S_common 达标 η=∅，S_ref 达标 η=∅——按 016 §15 转 lower-bound 或 --nlevel 扩样。

## 2. G1r-A — Forced-Continuation Audit（G1 语义，η_star 冻结）

### H=48（η_star=1.2 冻结，test fresh）

- **P(F=1)（按决策状态）** = 0.0136（F 状态 5/369）；P(episode contains F) = 0.0208；ΔB_forced = 120.0000 bits（D8 因 S_common 被迫支付的 8-bit 成本）。
- E[B^FG]=27.0000、E[B^D8]=36.9000，E[D]=-9.9000（t 参考 CI [-10.9388, -8.8612]，Hoeffding U95=-2.3159）。

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0583 | 0.1155 | 0.4500 | 0.5392 | UNCERTAIN | 1.4750 | 3.4000 |
| Direct8 | 0.0417 | 0.0938 | 0.4333 | 0.5227 | UNCERTAIN | 1.5375 | 12.3000 |

### H=96（η_star=1.2 冻结，test fresh）

- **P(F=1)（按决策状态）** = 0.0281（F 状态 13/463）；P(episode contains F) = 0.0542；ΔB_forced = 312.0000 bits（D8 因 S_common 被迫支付的 8-bit 成本）。
- E[B^FG]=33.0625、E[B^D8]=46.3000，E[D]=-13.2375（t 参考 CI [-15.3303, -11.1447]，Hoeffding U95=1.9306）。

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0250 | 0.0709 | 0.3833 | 0.4727 | UNCERTAIN | 1.7958 | 4.3292 |
| Direct8 | 0.0417 | 0.0938 | 0.3333 | 0.4217 | UNCERTAIN | 1.9292 | 15.4333 |

## 3. G1r-B — conservative S_ref Gate（016 §4）

### H=48（η_star_ref=1.2 冻结，test fresh）

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] | E[T_stop] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0583 | 0.1155 | 0.4500 | 0.5392 | UNCERTAIN | 1.4750 | 3.4000 | 27.0000 | 1.4750 |
| Direct8 | 0.0417 | 0.0938 | 0.4333 | 0.5227 | UNCERTAIN | 1.5167 | 12.1333 | 36.4000 | 1.5167 |

- **Gate（intersection-union，016 §10）**：QoS 双方 FEASIBLE = FAIL（FG=UNCERTAIN、D8=UNCERTAIN）；paired Hoeffding U95(E[D])=-1.8159（<0 → PASS）。
  → **UNRESOLVED / FAIL**：016 §15 判定——若 P(F≈1) 不小或 S_ref 下不显著，则 016 §4 预期（-12.31 → -5..-10）落空，按 016 §15 转 lower-bound 或先扩样。

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.5250 | 0.4750 | 0.0000 | 0.0000 | 0.0000 |
| Direct8 | 0.4833 | 0.5167 | 0.0000 | 0.0000 | 0.0000 |

- 分解（S_ref）：E[D]=-9.4000；setup 部分 16.0·(E[N_tx^FG]−E[N_tx^D8])=-0.6667，payload 部分 -8.7333。
- NP-matched（P_FA=0.05）secondary：P_D^FG=0.5292 / E[B]^FG=27.0000；P_D^D8=0.5750 / E[B]^D8=36.4000。
（  32.5s 累计）

### H=96（η_star_ref=1.2 冻结，test fresh）

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] | E[T_stop] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0250 | 0.0709 | 0.4083 | 0.4978 | UNCERTAIN | 1.7417 | 4.2000 | 32.0667 | 1.7417 |
| Direct8 | 0.0333 | 0.0826 | 0.3500 | 0.4388 | UNCERTAIN | 1.8583 | 14.8667 | 44.6000 | 1.8583 |

- **Gate（intersection-union，016 §10）**：QoS 双方 FEASIBLE = FAIL（FG=UNCERTAIN、D8=UNCERTAIN）；paired Hoeffding U95(E[D])=2.6348（<0 → FAIL）。
  → **UNRESOLVED / FAIL**：016 §15 判定——若 P(F≈1) 不小或 S_ref 下不显著，则 016 §4 预期（-12.31 → -5..-10）落空，按 016 §15 转 lower-bound 或先扩样。

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.5500 | 0.2667 | 0.0875 | 0.0833 | 0.0125 |
| Direct8 | 0.5083 | 0.2417 | 0.1333 | 0.1167 | 0.0000 |

- 分解（S_ref）：E[D]=-12.5333；setup 部分 16.0·(E[N_tx^FG]−E[N_tx^D8])=-1.8667，payload 部分 -10.6667。
- NP-matched（P_FA=0.05）secondary：P_D^FG=0.6417 / E[B]^FG=32.0667；P_D^D8=0.6833 / E[B]^D8=44.6000。
（  35.1s 累计）

总耗时: 35.1s

- **B0.7-G1r 结论（016 §15）**：G1r-A 量化 action-leakage（ΔB_forced / P(F=1)）——若很小，则 G1 的 D8 劣势主要是 granularity 真实机制；G1r-B 用 A_D8-reference 的保守 common-stop 重验 U95(E[B^FG−B^D8])<0——若仍 显著，granularity 独立收益在公平性上站稳（016 §4 预期 -5..-10 bit），随后投入 **B0.7-G2**：FG/D8 **分别**在 calibration 上优化自己的 (ρ,η)（016 §7/§9：J_m⋆=inf E_π[B] s.t. QoS，各自 controller），test 完全 fresh，暂不加 CPI；最后 B0.7-G3 重构 DualCPI 使 certificate 与 当前 dual objective 一致（016 §5/§15-5）。

