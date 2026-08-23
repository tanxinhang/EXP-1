# O-PEF MVS-B0.7-G1 — N=8 held-out QoS-dual calibrated common-stop Gate（015 §七-§九/§十三）

> 协议（015 §十三 G1 冻结）：**N=8**（GAMMA=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]、levels=(1, 2, 4, 8)、r_max=8）；b_setup=16.0；stratified N0=N1=120（calibration N_CAL=60 @ H=96，test N_TEST=120 @ H∈(48, 96)）；episode 级 CRN（同一 W_e=(H_e,L_e) 给 FG/D8；planner 确定性）。
> **QoS-dual 校准（015 §七-§九）**：λ_M=512.0（标度固定），扫 η_dec=log(λ_F/λ_M) ∈ (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)；终止风险 R_λ(x)=min{λ_M p, λ_F(1-p)}，单步继续值 Q_λ^(1)(a|x,h)=c_a+E[R_λ(X')|x,a]；**STOP ⟺ R_λ(x) ≤ min_a Q_λ^(1)**（无 |Ω|≥κ 对称停止，015 §三）；判决 Ω>η_dec→H1。两方法共用同一 S_λ/判决；只选 calibration 上双方 FEASIBLE 且 E[B^FG]+E[B^D8] 最小的 η_star 冻结（**双参数不在 test 上触碰**——015 §七 的 anti-post-hoc 结构）。

> **Gate（015 §十三 G1）**：test 上两方法均 U95(P_FA)≤0.12 且 U95(P_MD)≤0.4（FEASIBLE）才比较 U95(E[B^FG−B^D8])<0 → PASS；任一 INFEASIBLE → 该行不具 matched 地位；双方 UNCERTAIN → UNRESOLVED （--nlevel 扩样）。
> 记账：B=b_setup·N_tx+payload（逐样本断言）；E[N_tx]、E[B_payload]、E[T_stop]、P(T_stop=k)。

> 生成时间: 2026-08-24 01:44:59   模式: SMOKE   nlevel=1（N_TEST=120，N_CAL=60）

## 1. Calibration（calibration seeds，H=96）— 求 η_star

| η_dec | 方法 | P_FA | U95(P_FA) | P_MD | U95(P_MD) | 分类 | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | FG | 0.1167 | 0.2218 | 0.1667 | 0.2803 | UNCERTAIN | 44.9333 |
| 1.0 | D8 | 0.1000 | 0.2015 | 0.2167 | 0.3362 | UNCERTAIN | 48.8000 |

| 1.2 | FG | 0.1000 | 0.2015 | 0.2333 | 0.3544 | UNCERTAIN | 37.7250 |
| 1.2 | D8 | 0.0833 | 0.1807 | 0.2833 | 0.4077 | UNCERTAIN | 50.6000 |

| 1.4 | FG | 0.0833 | 0.1807 | 0.3000 | 0.4251 | UNCERTAIN | 40.5250 |
| 1.4 | D8 | 0.0667 | 0.1593 | 0.3000 | 0.4251 | UNCERTAIN | 50.0000 |

| 1.6 | FG | 0.0667 | 0.1593 | 0.3000 | 0.4251 | UNCERTAIN | 42.9333 |
| 1.6 | D8 | 0.0500 | 0.1370 | 0.3000 | 0.4251 | UNCERTAIN | 49.6000 |

| 1.8 | FG | 0.0333 | 0.1136 | 0.3500 | 0.4764 | UNCERTAIN | 41.5750 |
| 1.8 | D8 | 0.0333 | 0.1136 | 0.3833 | 0.5098 | UNCERTAIN | 49.2000 |

| 2.0 | FG | 0.0333 | 0.1136 | 0.3667 | 0.4932 | UNCERTAIN | 44.0083 |
| 2.0 | D8 | 0.0333 | 0.1136 | 0.4000 | 0.5263 | UNCERTAIN | 47.4000 |

- **校准 UNRESOLVED**：GRID_ETA 中无 η_dec 使两方法在 calibration 上同时 FEASIBLE。诚实报告（015 §十三：无法认证 matched 比较 → 转 015 §十四 lower-bound 路线）。

## 2. Test — 跳过（无冻结 η_star）

- 判定：**UNRESOLVED（校准失败）**——B0.7-G1 Gate 未建立，按 015 §十三 转 lower-bound 路线。

总耗时: 10.5s

- **B0.7-G1 结论（015 §十三）**：held-out 协议下 matched-QoS 双认证 + paired bit 比较。若 test 双方 FEASIBLE 且 U95(E[B^FG−B^D8])<0 → **granularity 在正式 QoS 口径下有独立收益**（论文主线证据）；否则 **STOP / UNRESOLVED**，B0.5 换用途为 Direct8-近优下界 （V_LB≤V⋆≤V^D8，015 §十四）。

