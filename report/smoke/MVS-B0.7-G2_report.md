# O-PEF MVS-B0.7-G2 — Separately Calibrated QoS-Dual Policy-Family Certification（advice/017.md）

> **定位（017 §final）**：Separately Calibrated QoS-Dual Policy-Family Certification——当 FG 与 Direct8 **都允许为自身优化 controller** 时，在相同 detection QoS 下谁需要更少 communication bits？G2 不再使用 common stop（不存在 G1 的 action-set leakage，017 §二）。

> **控制器（017 §二）**：λ_M=ρ、λ_F=ρe^η；R_{ρ,η}(x)=ρ·min{p_x, e^η(1−p_x)}；Q^(1)(a|x,h)=c_a+E[R_{ρ,η}(X')|x,a]；A_FG={(i,r'): r'>r_i, r'∈{1,2,4,8}}、A_D8={(i,8): r_i<8}；STOP_m ⟺ R≤min_{A_m(h)}Q^(1)，a*_m=argmin_{A_m}Q^(1)；判决 Ω>η⇒H1。

> **命名（017 §三）**：实验对象是 **separately calibrated one-step QoS-dual controllers**（不是 optimized / globally optimized）；θ̂_m = argmin_{θ∈Θ} Ê_cal[B_m(θ)] s.t. U_cal(P_FA^{m,θ})≤α ∧ U_cal(P_MD^{m,θ})≤β；test 对象 π_FG^{θ̂_FG} vs π_D8^{θ̂_D8}。

> **冻结参数（017 §四）**：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup=16.0、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，grid 只在 calibration 用）；calibration worlds FG/D8 **完全共用**；**CPI OFF**；test worlds 与 calibration 完全分离；paired CRN；主 operating point **H=96**、secondary stress H=48（同冻结 controller，诚实报告 boundary，不为 H=48 重新校准）。

> **统计（017 §六 方案 A）**：正式 test **直接冻结 N_TEST=1600 per hypothesis、一次性看结果**（不做看结果后的 staged escalation ⇒ fixed-N 95% 覆盖声明成立）；paired bit 用 one-sided paired Hoeffding（D∈[−H,H]，分布无关）；QoS 用 Wilson 95% one-sided upper。Calibration N_CAL=600/hyp @ H=96（017 §五，至少如此）。

> **Gate（017 §八，只允许四种结论）**：Primary H=96 双方 FEASIBLE 且 U95(E[D])<0 → **G2 PASS**；双方 FEASIBLE 且 L95(E[D])>0 → **FAIL**（转 lower-bound / Direct8-near-optimal 路线）；L95≤0≤U95 → **BIT-UNRESOLVED**（不改算法；§六 方案 A 冻结 1600，故按 UNRESOLVED 报告）；任一方不 FEASIBLE → **QoS-UNRESOLVED / INFEASIBLE**（不能比较 bit）。

> **P1 口径修正（017 §一）**：P1-1 只报告 P(F=1|S_common=CONTINUE)（把 STOP 决策状态加入分母只会更低）；P1-2 ΔB_forced 改名 **gross forced-action cost**（非 causal extra cost）并按 episode 归一化；P1-3 dual-Q 回归升级 root + on-policy reachable + r=0/1/2/4 分层 × corner {(128,0.8),(512,1.2),(1024,2.0)}；emulate_d8 计数按真实循环数报告。

> 生成时间: 2026-08-24 03:45:06   模式: SMOKE   N_CAL=60（@H=96），N_TEST=120（@H∈(48, 96)），SMOKE）

## 1. Invariant suite（017 §九 + P1-3）

- **inv-2** q_fast vs generic dual-Q exact（017 §九、P1-3 新覆盖：root + on-policy reachable + r=0/1/2/4 分层 × 3 corners）：共 3932 个 (i,r2) 对，max|Δ|=0.000000000001（<1e-9 → PASS）。
- **inv-1** ΣP(m'|x,a)=1（unnormalized 混合质量守恒）：3932 个 (state,action) 上 max|Σw−1|=0.000000000000（<1e-9 → PASS）。
- **inv-5** π_FG|_{A_D8} ≡ π_D8（同 (ρ,η)=(512, 1.2)、同 world 逐样本一致）：20 条 episode（stratified，2×n_ep_check=20）lam/cost/N_tx/payload 全一致 → **PASS**（计数按真实循环数，P1-3/P2）。
（0.9s）

## 2. Calibration（017 §三/§五：shared worlds、grid 冻结）

> worlds：stratified N_CAL=60/hyp @ H=96（FG/D8 完全共用，与 test 完全分离）；θ̂_m = feasible（U95(P_FA)≤α ∧ U95(P_MD)≤β）中 Ê_cal[B_m] 最小（tie-break: (ρ,η) 字典序小者）。

| method | θ̂_m=(ρ*,η*) | feasible 数/28 | min Ê_cal[B] @ θ̂_m |
| --- | --- | --- | --- |
| FG | **∅（无 FEASIBLE）** | 0 | — |
| D8 | **∅（无 FEASIBLE）** | 0 | — |

> **⚠ calibration 无可行 θ̂**：按 017 §八 = **QoS-UNRESOLVED / INFEASIBLE**，不能比较 bit。以下仍给出 feasible-region 诊断与 test QoS 分类（不写 bit 结论）。

#### Feasible region（017 §七：(ρ,η) → {INFEASIBLE, UNCERTAIN, FEASIBLE} + Ê[B]）

**FG**：
| ρ | η | U95(P_FA) | U95(P_MD) | 分类 | Ê_cal[B] |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.8 | 0.2417 | 0.4077 | UNCERTAIN | — |
| 128 | 1.0 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.2 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.4 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.6 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.8 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 2.0 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 256 | 0.8 | 0.2218 | 0.3362 | UNCERTAIN | — |
| 256 | 1.0 | 0.2015 | 0.3362 | UNCERTAIN | — |
| 256 | 1.2 | 0.2218 | 0.3544 | UNCERTAIN | — |
| 256 | 1.4 | 0.1807 | 0.4764 | UNCERTAIN | — |
| 256 | 1.6 | 0.1593 | 0.5263 | UNCERTAIN | — |
| 256 | 1.8 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 256 | 2.0 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 512 | 0.8 | 0.2218 | 0.2992 | UNCERTAIN | — |
| 512 | 1.0 | 0.2218 | 0.2803 | UNCERTAIN | — |
| 512 | 1.2 | 0.2015 | 0.3544 | UNCERTAIN | — |
| 512 | 1.4 | 0.1807 | 0.4251 | UNCERTAIN | — |
| 512 | 1.6 | 0.1593 | 0.4251 | UNCERTAIN | — |
| 512 | 1.8 | 0.1136 | 0.4764 | UNCERTAIN | — |
| 512 | 2.0 | 0.1136 | 0.4932 | UNCERTAIN | — |
| 1024 | 0.8 | 0.2218 | 0.2611 | UNCERTAIN | — |
| 1024 | 1.0 | 0.2218 | 0.2611 | UNCERTAIN | — |
| 1024 | 1.2 | 0.2015 | 0.3362 | UNCERTAIN | — |
| 1024 | 1.4 | 0.1807 | 0.3723 | UNCERTAIN | — |
| 1024 | 1.6 | 0.0886 | 0.4423 | UNCERTAIN | — |
| 1024 | 1.8 | 0.1136 | 0.4594 | UNCERTAIN | — |
| 1024 | 2.0 | 0.1136 | 0.4423 | UNCERTAIN | — |

**D8**：
| ρ | η | U95(P_FA) | U95(P_MD) | 分类 | Ê_cal[B] |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.8 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.0 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.2 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.4 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.6 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 1.8 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 128 | 2.0 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 256 | 0.8 | 0.2417 | 0.3178 | UNCERTAIN | — |
| 256 | 1.0 | 0.2218 | 0.3362 | UNCERTAIN | — |
| 256 | 1.2 | 0.2015 | 0.3901 | UNCERTAIN | — |
| 256 | 1.4 | 0.1807 | 0.4077 | UNCERTAIN | — |
| 256 | 1.6 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 256 | 1.8 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 256 | 2.0 | 0.0602 | 1.0000 | INFEASIBLE | — |
| 512 | 0.8 | 0.2218 | 0.2611 | UNCERTAIN | — |
| 512 | 1.0 | 0.2015 | 0.3362 | UNCERTAIN | — |
| 512 | 1.2 | 0.1807 | 0.3901 | UNCERTAIN | — |
| 512 | 1.4 | 0.1593 | 0.4251 | UNCERTAIN | — |
| 512 | 1.6 | 0.1593 | 0.4251 | UNCERTAIN | — |
| 512 | 1.8 | 0.1136 | 0.5098 | UNCERTAIN | — |
| 512 | 2.0 | 0.1136 | 0.5427 | UNCERTAIN | — |
| 1024 | 0.8 | 0.2417 | 0.2611 | UNCERTAIN | — |
| 1024 | 1.0 | 0.2015 | 0.3723 | UNCERTAIN | — |
| 1024 | 1.2 | 0.1593 | 0.3901 | UNCERTAIN | — |
| 1024 | 1.4 | 0.1370 | 0.4251 | UNCERTAIN | — |
| 1024 | 1.6 | 0.1136 | 0.4423 | UNCERTAIN | — |
| 1024 | 1.8 | 0.0886 | 0.4764 | UNCERTAIN | — |
| 1024 | 2.0 | 0.0886 | 0.5098 | UNCERTAIN | — |

- inv-3/inv-4（逐 episode 断言：B=16·N_tx+B_payload、B≤H）：calibration 全 56 次 θ-run 中 violations=0 → **PASS**。
（21.8s；累计 26.2s）

## 3. Primary Gate @ H=96（017 §六/§八，θ̂ 冻结、test fresh）

> worlds：stratified N_TEST=120/hyp（fresh seeds，calibration 完全不可见）；FG 与 D8 在**同一 worlds** 上（paired CRN，planner 确定性）；**统计按 017 §六 方案 A：N_TEST 一次性冻结，无 staged escalation**（Hoeffding 为 fixed-N 95% one-sided）。

### H=96（θ̂_FG=∅、θ̂_D8=∅ 冻结）

> **Gate（017 §八）**：QoS-UNRESOLVED / INFEASIBLE。calibration 无可行 θ̂ ⇒ 017 §八：QoS-UNRESOLVED / INFEASIBLE（不能比较 bit）。

（0.0s；累计 26.2s）

## 4. Secondary stress @ H=48（017 §四：同冻结 controller，诚实报告 operating-region boundary；不为 H=48 重新校准）

### H=48（θ̂_FG=∅、θ̂_D8=∅ 冻结）

> **Gate（017 §八）**：QoS-UNRESOLVED / INFEASIBLE。calibration 无可行 θ̂ ⇒ 017 §八：QoS-UNRESOLVED / INFEASIBLE（不能比较 bit）。

（0.0s；累计 26.2s）

## 5. Secondary diagnostics（017 §七）

### 5.1 分假设 bit 分解 E[B|H0]、E[B|H1]（防止平均 bit gain 只来自一个 hypothesis）

**H=96**：
| method | E[B|H0] | E[B|H1] | E[B] | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- |
| FG | — | — | — | — | — |
| Direct8 | — | — | — | — | — |

**H=48**：
| method | E[B|H0] | E[B|H1] | E[B] | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- |
| FG | — | — | — | — | — |
| Direct8 | — | — | — | — | — |

### 5.2 Cross-evaluation @ H=96（017 §七：FG@θ̂_FG、FG@θ̂_D8、D8@θ̂_D8、D8@θ̂_FG；secondary，不参与 Gate）

- θ̂ 缺失 → cross-evaluation 不比较（QoS 见 §3/§4）。

## 6. Forced-continuation 口径修正 + robustness（017 §一 P1-1/P1-2）

> 参考控制器 θ_ref=(512, 1.2)（= G1r 冻结的 LAM_M=512、η_star=1.2）；公共规则 S_common=min_{A_FG}Q<R。**P1-1** 正式口径 P(F=1|S_common=CONTINUE)（分母只含公共规则说继续的决策状态；把 STOP 状态加入分母只会更低，故同时给出无条件参考）。**P1-2** F 状态上 D8 实付的 full-packet 成本之和 = **gross forced-action cost**（非 causal extra cost，反事实 STOP 会改变后续轨迹），按 episode 归一化。G2 已从设计上消除公共停止器，此审计仅作诊断与 robustness 陈述。

### H=48（θ_ref=(512, 1.2)，fresh N_TEST=120）

- **P(F=1|S_common=CONTINUE) = 0.0136**（F 状态 5/369 个继续-决策状态）；无条件参考 P(F=1|所有决策状态) = 0.0104（5/480）——低于条件口径（017 P1-1 预期）。
- P(episode contains F) = 0.0208。
- **gross forced-action cost = 120.0000 bits = 0.5000 bit/episode**（D8 在公共规则下被迫支付的 8-bit 通信成本；017 P1-2 命名，含 setup+payload）。

### H=96（θ_ref=(512, 1.2)，fresh N_TEST=120）

- **P(F=1|S_common=CONTINUE) = 0.0281**（F 状态 13/463 个继续-决策状态）；无条件参考 P(F=1|所有决策状态) = 0.0194（13/671）——低于条件口径（017 P1-1 预期）。
- P(episode contains F) = 0.0542。
- **gross forced-action cost = 312.0000 bits = 1.3000 bit/episode**（D8 在公共规则下被迫支付的 8-bit 通信成本；017 P1-2 命名，含 setup+payload）。

> **Robustness statement（017 §一）**：即使极端地把 gross forced-action cost 全部视为 leakage bias（H=96：1.3000 bit/episode），也远低于 G1/G1r 的 −12.31 bit gap （H=96）——G1 的 granularity 收益不可能主要由 action-set leakage 解释；G2 已从设计上消除 public/common 停止器。

## 结论

- **Primary H=96：QoS-UNRESOLVED / INFEASIBLE**（双方 FEASIBLE=FAIL（NO-FEASIBLE-θ̂(CAL)/NO-FEASIBLE-θ̂(CAL)），U95(E[D])=inf，L95(E[D])=-inf）。
- **Secondary H=48：QoS-UNRESOLVED / INFEASIBLE**（双方 FEASIBLE=FAIL（NO-FEASIBLE-θ̂(CAL)/NO-FEASIBLE-θ̂(CAL)），U95(E[D])=inf）。


- **B0.7-G2 定位（017 §final）**：separately calibrated QoS-dual policy-family certification。若 **G2 PASS**，论文核心 performance 主线闭环；G3（DualCPI 是否还有独立增益）应变成“是否值得纳入主算法”的 Gate，而不是必做的性能增强——避免系统从“反馈粒度这一核心科学问题”跑回复杂 planner/sample-complexity 工程。

总耗时: 29.1s

