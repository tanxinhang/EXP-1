# O-PEF MVS-B0.7-G1 — N=8 held-out QoS-dual calibrated common-stop Gate（015 §七-§九/§十三）

> 协议（015 §十三 G1 冻结）：**N=8**（GAMMA=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]、levels=(1, 2, 4, 8)、r_max=8）；b_setup=16.0；stratified N0=N1=600（calibration N_CAL=300 @ H=96，test N_TEST=600 @ H∈(48, 96)）；episode 级 CRN（同一 W_e=(H_e,L_e) 给 FG/D8；planner 确定性）。
> **QoS-dual 校准（015 §七-§九）**：λ_M=512.0（标度固定），扫 η_dec=log(λ_F/λ_M) ∈ (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)；终止风险 R_λ(x)=min{λ_M p, λ_F(1-p)}，单步继续值 Q_λ^(1)(a|x,h)=c_a+E[R_λ(X')|x,a]；**STOP ⟺ R_λ(x) ≤ min_a Q_λ^(1)**（无 |Ω|≥κ 对称停止，015 §三）；判决 Ω>η_dec→H1。两方法共用同一 S_λ/判决；只选 calibration 上双方 FEASIBLE 且 E[B^FG]+E[B^D8] 最小的 η_star 冻结（**双参数不在 test 上触碰**——015 §七 的 anti-post-hoc 结构）。

> **Gate（015 §十三 G1）**：test 上两方法均 U95(P_FA)≤0.12 且 U95(P_MD)≤0.4（FEASIBLE）才比较 U95(E[B^FG−B^D8])<0 → PASS；任一 INFEASIBLE → 该行不具 matched 地位；双方 UNCERTAIN → UNRESOLVED （--nlevel 扩样）。
> 记账：B=b_setup·N_tx+payload（逐样本断言）；E[N_tx]、E[B_payload]、E[T_stop]、P(T_stop=k)。

> 生成时间: 2026-08-24 01:46:49   模式: FULL   nlevel=1（N_TEST=600，N_CAL=300）

## 1. Calibration（calibration seeds，H=96）— 求 η_star

| η_dec | 方法 | P_FA | U95(P_FA) | P_MD | U95(P_MD) | 分类 | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | FG | 0.0333 | 0.0603 | 0.2200 | 0.2703 | FEASIBLE | 42.3683 |
| 1.0 | D8 | 0.0433 | 0.0727 | 0.2300 | 0.2809 | FEASIBLE | 46.6800 |

| 1.2 | FG | 0.0333 | 0.0603 | 0.3333 | 0.3885 | FEASIBLE | 34.7833 |
| 1.2 | D8 | 0.0400 | 0.0686 | 0.2733 | 0.3264 | FEASIBLE | 46.5200 |

| 1.4 | FG | 0.0233 | 0.0474 | 0.3600 | 0.4158 | UNCERTAIN | 36.9233 |
| 1.4 | D8 | 0.0267 | 0.0517 | 0.3167 | 0.3713 | FEASIBLE | 45.8400 |

| 1.6 | FG | 0.0233 | 0.0474 | 0.3433 | 0.3987 | FEASIBLE | 38.5400 |
| 1.6 | D8 | 0.0200 | 0.0429 | 0.3467 | 0.4022 | UNCERTAIN | 45.6000 |

| 1.8 | FG | 0.0167 | 0.0384 | 0.4067 | 0.4631 | UNCERTAIN | 38.2550 |
| 1.8 | D8 | 0.0133 | 0.0338 | 0.3933 | 0.4496 | UNCERTAIN | 44.7200 |

| 2.0 | FG | 0.0100 | 0.0290 | 0.4233 | 0.4799 | UNCERTAIN | 40.0683 |
| 2.0 | D8 | 0.0100 | 0.0290 | 0.4433 | 0.4999 | UNCERTAIN | 44.0400 |

- **η_star = 1.2**（校准集上双方 FEASIBLE 且 E[B^FG]+E[B^D8]=81.3033 最小；其余达标 η：1.0(89.0483), 1.2(81.3033)）。冻结，test 上**不再触碰**。

## 2. Test（fresh seeds，H=48，η_star=1.2 冻结）

### QoS 三态分类（Wilson 95%，test）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG (adaptive) | 0.0500 | 0.0352 | 0.0705 | 0.4083 | 0.3697 | 0.4481 | **UNCERTAIN** |
| Direct8 | 0.0483 | 0.0339 | 0.0686 | 0.3733 | 0.3356 | 0.4127 | **UNCERTAIN** |

### Bit Gate（015 §十三 G1：双方 FEASIBLE 才比较）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | E[T_stop] |
| --- | --- | --- | --- | --- |
| FG | 1.4800 | 3.4167 | 27.0967 | 1.4800 |
| Direct8 | 1.5467 | 12.3733 | 37.1200 | 1.5467 |

- **Gate 拦截**：FG=UNCERTAIN、Direct8=UNCERTAIN——未双方 FEASIBLE，matched 比较不成立（B0.6-r 口径）。
  → 判定 **UNRESOLVED**（--nlevel 扩样；或 QoS 未达标时转 015 §十四 lower-bound 路线/G2 fresh Gate）。

### 停止结构（B0.6-d 记账，test）

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.5200 | 0.4800 | 0.0000 | 0.0000 | 0.0000 |
| Direct8 | 0.4533 | 0.5467 | 0.0000 | 0.0000 | 0.0000 |

- 分解：E[D]=-10.0233，其中 setup 部分 16.0·(E[N_tx^FG]−E[N_tx^D8])=-1.0667，payload 部分 -8.9567。

- **NP-matched（P_FA=0.05）secondary**：P_D^FG=0.5917 / E[B]^FG=27.0967；P_D^D8=0.6267 / E[B]^D8=37.1200。
（10s）

## 2. Test（fresh seeds，H=96，η_star=1.2 冻结）

### QoS 三态分类（Wilson 95%，test）

| 方法 | P_FA | L95 | U95 | P_MD | L95 | U95 | 分类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG (adaptive) | 0.0333 | 0.0217 | 0.0509 | 0.3567 | 0.3194 | 0.3958 | **FEASIBLE** |
| Direct8 | 0.0400 | 0.0270 | 0.0588 | 0.3133 | 0.2775 | 0.3515 | **FEASIBLE** |

### Bit Gate（015 §十三 G1：双方 FEASIBLE 才比较）

| 方法 | E[N_tx] | E[B_payload] | E[B]=b_setup·N_tx+payload | E[T_stop] |
| --- | --- | --- | --- | --- |
| FG | 1.8808 | 4.5967 | 34.6900 | 1.8808 |
| Direct8 | 1.9583 | 15.6667 | 47.0000 | 1.9583 |

- 双方 FEASIBLE → paired 比较：E[D]=E[B^FG−B^D8] = -12.3100，95% CI [-13.2604, -11.3596]（<0 → PASS）→ **PASS**。
  → **matched-QoS 下 granularity 有独立收益**：N=8 held-out、λ 只由 calibration 定，FG 显著省 bits → 主线可进 B0.7-G2（frozen CPI override / fresh Gate）或直接作为论文主线证据。

### 停止结构（B0.6-d 记账，test）

| 方法 | P(T_stop=1) | P(T_stop=2) | P(T_stop=3) | P(T_stop=4) | P(T_stop≥5) |
| --- | --- | --- | --- | --- | --- |
| FG | 0.5075 | 0.2858 | 0.0908 | 0.0500 | 0.0658 |
| Direct8 | 0.4658 | 0.2625 | 0.1192 | 0.1525 | 0.0000 |

- 分解：E[D]=-12.3100，其中 setup 部分 16.0·(E[N_tx^FG]−E[N_tx^D8])=-1.2400，payload 部分 -11.0700。

- **NP-matched（P_FA=0.05）secondary**：P_D^FG=0.6550 / E[B]^FG=34.6900；P_D^D8=0.7100 / E[B]^D8=47.0000。
（12s）

总耗时: 60.6s

- **B0.7-G1 结论（015 §十三）**：held-out 协议下 matched-QoS 双认证 + paired bit 比较。若 test 双方 FEASIBLE 且 U95(E[B^FG−B^D8])<0 → **granularity 在正式 QoS 口径下有独立收益**（论文主线证据）；否则 **STOP / UNRESOLVED**，B0.5 换用途为 Direct8-近优下界 （V_LB≤V⋆≤V^D8，015 §十四）。

