# O-PEF MVS-B0.7-G2 — Separately Calibrated QoS-Dual Policy-Family Certification（advice/017.md）

> **定位（017 §final）**：Separately Calibrated QoS-Dual Policy-Family Certification——当 FG 与 Direct8 **都允许为自身优化 controller** 时，在相同 detection QoS 下谁需要更少 communication bits？G2 不再使用 common stop（不存在 G1 的 action-set leakage，017 §二）。

> **控制器（017 §二）**：λ_M=ρ、λ_F=ρe^η；R_{ρ,η}(x)=ρ·min{p_x, e^η(1−p_x)}；Q^(1)(a|x,h)=c_a+E[R_{ρ,η}(X')|x,a]；A_FG={(i,r'): r'>r_i, r'∈{1,2,4,8}}、A_D8={(i,8): r_i<8}；STOP_m ⟺ R≤min_{A_m(h)}Q^(1)，a*_m=argmin_{A_m}Q^(1)；判决 Ω>η⇒H1。

> **命名（017 §三）**：实验对象是 **separately calibrated one-step QoS-dual controllers**（不是 optimized / globally optimized）；θ̂_m = argmin_{θ∈Θ} Ê_cal[B_m(θ)] s.t. U_cal(P_FA^{m,θ})≤α ∧ U_cal(P_MD^{m,θ})≤β；test 对象 π_FG^{θ̂_FG} vs π_D8^{θ̂_D8}。

> **冻结参数（017 §四）**：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup=16.0、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，grid 只在 calibration 用）；calibration worlds FG/D8 **完全共用**；**CPI OFF**；test worlds 与 calibration 完全分离；paired CRN；主 operating point **H=96**、secondary stress H=48（同冻结 controller，诚实报告 boundary，不为 H=48 重新校准）。

> **统计（017 §六 方案 A）**：正式 test **直接冻结 N_TEST=1600 per hypothesis、一次性看结果**（不做看结果后的 staged escalation ⇒ fixed-N 95% 覆盖声明成立）；paired bit 用 one-sided paired Hoeffding（D∈[−H,H]，分布无关）；QoS 用 **Wilson 双侧 95% 区间的上端点**（018 §十一：z=1.96 是双侧端点，作单侧上界≈97.5% —— 更保守，只改名称不改数值）。Calibration N_CAL=600/hyp @ H=96（017 §五，至少如此）。

> **Gate（017 §八，只允许四种结论）**：Primary H=96 双方 FEASIBLE 且 U95(E[D])<0 → **G2 PASS**；双方 FEASIBLE 且 L95(E[D])>0 → **FAIL**（转 lower-bound / Direct8-near-optimal 路线）；L95≤0≤U95 → **BIT-UNRESOLVED**（不改算法；§六 方案 A 冻结 1600，故按 UNRESOLVED 报告）；任一方不 FEASIBLE → **QoS-UNRESOLVED / INFEASIBLE**（不能比较 bit）。

> **P1 口径修正（017 §一）**：P1-1 只报告 P(F=1|S_common=CONTINUE)（把 STOP 决策状态加入分母只会更低）；P1-2 ΔB_forced 改名 **gross forced-action cost**（非 causal extra cost）并按 episode 归一化；P1-3 dual-Q 回归升级 root + on-policy reachable + r=0/1/2/4 分层 × corner {(128,0.8),(512,1.2),(1024,2.0)}；emulate_d8 计数按真实循环数报告。

> 生成时间: 2026-08-24 15:38:02   模式: SMOKE   N_CAL=60（@H=96），N_TEST=120（@H∈(48, 96)），SMOKE

## 1. Invariant suite（017 §九 + P1-3）

- **inv-2** q_fast vs generic dual-Q exact（017 §九、P1-3 新覆盖：root + on-policy reachable + r=0/1/2/4 分层 × 3 corners）：共 3932 个 (i,r2) 对，max|Δ|=0.000000000001（<1e-9 → PASS）。
- **inv-1** ΣP(m'|x,a)=1（unnormalized 混合质量守恒）：3932 个 (state,action) 上 max|Σw−1|=0.000000000000（<1e-9 → PASS）。
- **inv-5** π_FG|_{A_D8} ≡ π_D8（同 (ρ,η)=(512, 1.2)、同 world 逐样本一致）：20 条 episode（stratified，2×n_ep_check=20）lam/cost/N_tx/payload 全一致 → **PASS**（计数按真实循环数，P1-3/P2）。
（1.1s）

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
（26.7s；累计 31.8s）

## 3. Primary Gate @ H=96（017 §六/§八，θ̂ 冻结、test fresh）

> worlds：stratified N_TEST=120/hyp（fresh seeds，calibration 完全不可见）；FG 与 D8 在**同一 worlds** 上（paired CRN，planner 确定性）；**统计按 017 §六 方案 A：N_TEST 一次性冻结，无 staged escalation**（Hoeffding 为 fixed-N 95% one-sided）。

### H=96（θ̂_FG=∅、θ̂_D8=∅ 冻结）

> **Gate（017 §八）**：QoS-UNRESOLVED / INFEASIBLE。calibration 无可行 θ̂ ⇒ 017 §八：QoS-UNRESOLVED / INFEASIBLE（不能比较 bit）。

（0.0s；累计 31.8s）

## 4. Secondary stress @ H=48（017 §四：同冻结 controller，诚实报告 operating-region boundary；不为 H=48 重新校准）

### H=48（θ̂_FG=∅、θ̂_D8=∅ 冻结）

> **Gate（017 §八）**：QoS-UNRESOLVED / INFEASIBLE。calibration 无可行 θ̂ ⇒ 017 §八：QoS-UNRESOLVED / INFEASIBLE（不能比较 bit）。

（0.0s；累计 31.8s）

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
- 结构观察（018 §十 修正）：per-F 成本恒 = 24.0 bits（=16 setup + 8 payload）⇒ 目标 UAV 的 r_cur=0（fresh UAV，此前未上报 evidence，**不必然是 episode 首决策**）；F 的决策索引：P(idx=1)=0.0000、E[idx]=2.00。

### H=96（θ_ref=(512, 1.2)，fresh N_TEST=120）

- **P(F=1|S_common=CONTINUE) = 0.0281**（F 状态 13/463 个继续-决策状态）；无条件参考 P(F=1|所有决策状态) = 0.0194（13/671）——低于条件口径（017 P1-1 预期）。
- P(episode contains F) = 0.0542。
- **gross forced-action cost = 312.0000 bits = 1.3000 bit/episode**（D8 在公共规则下被迫支付的 8-bit 通信成本；017 P1-2 命名，含 setup+payload）。
- 结构观察（018 §十 修正）：per-F 成本恒 = 24.0 bits（=16 setup + 8 payload）⇒ 目标 UAV 的 r_cur=0（fresh UAV，此前未上报 evidence，**不必然是 episode 首决策**）；F 的决策索引：P(idx=1)=0.0000、E[idx]=2.46。

> **Robustness statement（018 §九 收紧）**：gross forced-action cost 说明的是 **immediate forced expenditure**（H=96：1.3000 bit/episode，很小）——不是 cascade/总 leakage 的数学上界（forced action 会连锁改变 posterior / UAV selection / stopping / transaction count，|ΔB_causal| 不受此界约束）。因果公平性的强证据由 **G1r-B 提供**（S_ref 从构造上移除 leakage 机制后 U95<0 仍成立）；本审计仅作 immediate-expenditure 诊断与 018 §九 口径修正。

## 结论

- **Primary H=96：QoS-UNRESOLVED / INFEASIBLE**（双方 FEASIBLE=FAIL（NO-FEASIBLE-θ̂(CAL)/NO-FEASIBLE-θ̂(CAL)），U95(E[D])=inf，L95(E[D])=-inf）。
- **Secondary H=48：QoS-UNRESOLVED / INFEASIBLE**（双方 FEASIBLE=FAIL（NO-FEASIBLE-θ̂(CAL)/NO-FEASIBLE-θ̂(CAL)），U95(E[D])=inf）。


- **B0.7-G2 定位（017 §final + 001 §二十四 重定位）**：separately calibrated QoS-dual policy-family certification —— **G2 数值与机制结论保留为 homogeneous-link mechanism-validation special case**（001 §二十四：16+Δr 均匀成本、P_MD≤0.40 机制口径、planner 非瓶颈）；**不再作为论文最终 ISAC/通信主结论**。论文主 QoS 统一为 **matched detection：P_FA≤α ∧ P_D≥P_D,max(α)−ε_D**（默认 α=0.05、ε_D=0.01，001 §三）；成本模型升级为 **link-aware c_{i,r→r'}=b_{0,i}+d_i(r,r')**（16+Δr 为 homogeneous special case，001 §六）；hard budget 改为 **frame-window C_{U2U}(ω)≤C_max^{frame}**（001 §三/§七）。
- **B0.7-G3（DualCPI）＝ SUSPENDED（001 §二十五：planner 不是当前瓶颈）**：双 Gate 预注册文本（019 §6-§9）**存档保留**，仅在未来换 regime 且需要 certified planning 时启用，**不进当前路线**。**下一步 = MVS-C Architecture Realignment（001 §二十六：C0 semantic closure、C1 link-aware phase theorem、C2 phase-guided policy（N=4）、C3 N=8 homogeneous replay（migration Gate：必须复现本 G2 special-case 数值）、C4 N=8 heterogeneous U2U（论文 headline：positive/independent/anti-correlation regime）、C5 protocol robustness）**；**论文四 Gate（001 §二十七）**：A 数学正确性 / B 机制必要性（Phase-FG<Direct8 且 <Static Progressive）/ C 通信现实性（b_ctrl>0、p_succ<1、anti-correlation 下仍成立）/ D 求解器质量（N=4 vs exact CMDP）。

总耗时: 35.3s

