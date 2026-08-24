# O-PEF MVS-B0.7-G2 — Separately Calibrated QoS-Dual Policy-Family Certification（advice/017.md）

> **定位（017 §final）**：Separately Calibrated QoS-Dual Policy-Family Certification——当 FG 与 Direct8 **都允许为自身优化 controller** 时，在相同 detection QoS 下谁需要更少 communication bits？G2 不再使用 common stop（不存在 G1 的 action-set leakage，017 §二）。

> **控制器（017 §二）**：λ_M=ρ、λ_F=ρe^η；R_{ρ,η}(x)=ρ·min{p_x, e^η(1−p_x)}；Q^(1)(a|x,h)=c_a+E[R_{ρ,η}(X')|x,a]；A_FG={(i,r'): r'>r_i, r'∈{1,2,4,8}}、A_D8={(i,8): r_i<8}；STOP_m ⟺ R≤min_{A_m(h)}Q^(1)，a*_m=argmin_{A_m}Q^(1)；判决 Ω>η⇒H1。

> **命名（017 §三）**：实验对象是 **separately calibrated one-step QoS-dual controllers**（不是 optimized / globally optimized）；θ̂_m = argmin_{θ∈Θ} Ê_cal[B_m(θ)] s.t. U_cal(P_FA^{m,θ})≤α ∧ U_cal(P_MD^{m,θ})≤β；test 对象 π_FG^{θ̂_FG} vs π_D8^{θ̂_D8}。

> **冻结参数（017 §四）**：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup=16.0、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，grid 只在 calibration 用）；calibration worlds FG/D8 **完全共用**；**CPI OFF**；test worlds 与 calibration 完全分离；paired CRN；主 operating point **H=96**、secondary stress H=48（同冻结 controller，诚实报告 boundary，不为 H=48 重新校准）。

> **统计（017 §六 方案 A）**：正式 test **直接冻结 N_TEST=1600 per hypothesis、一次性看结果**（不做看结果后的 staged escalation ⇒ fixed-N 95% 覆盖声明成立）；paired bit 用 one-sided paired Hoeffding（D∈[−H,H]，分布无关）；QoS 用 Wilson 95% one-sided upper。Calibration N_CAL=600/hyp @ H=96（017 §五，至少如此）。

> **Gate（017 §八，只允许四种结论）**：Primary H=96 双方 FEASIBLE 且 U95(E[D])<0 → **G2 PASS**；双方 FEASIBLE 且 L95(E[D])>0 → **FAIL**（转 lower-bound / Direct8-near-optimal 路线）；L95≤0≤U95 → **BIT-UNRESOLVED**（不改算法；§六 方案 A 冻结 1600，故按 UNRESOLVED 报告）；任一方不 FEASIBLE → **QoS-UNRESOLVED / INFEASIBLE**（不能比较 bit）。

> **P1 口径修正（017 §一）**：P1-1 只报告 P(F=1|S_common=CONTINUE)（把 STOP 决策状态加入分母只会更低）；P1-2 ΔB_forced 改名 **gross forced-action cost**（非 causal extra cost）并按 episode 归一化；P1-3 dual-Q 回归升级 root + on-policy reachable + r=0/1/2/4 分层 × corner {(128,0.8),(512,1.2),(1024,2.0)}；emulate_d8 计数按真实循环数报告。

> 生成时间: 2026-08-24 09:46:24   模式: FULL   N_CAL=600（@H=96），N_TEST=1600（@H∈(48, 96)），FULL 冻结（方案 A）

## 1. Invariant suite（017 §九 + P1-3）

- **inv-2** q_fast vs generic dual-Q exact（017 §九、P1-3 新覆盖：root + on-policy reachable + r=0/1/2/4 分层 × 3 corners）：共 15085 个 (i,r2) 对，max|Δ|=0.000000000002（<1e-9 → PASS）。
- **inv-1** ΣP(m'|x,a)=1（unnormalized 混合质量守恒）：15085 个 (state,action) 上 max|Σw−1|=0.000000000000（<1e-9 → PASS）。
- **inv-5** π_FG|_{A_D8} ≡ π_D8（同 (ρ,η)=(512, 1.2)、同 world 逐样本一致）：100 条 episode（stratified，2×n_ep_check=100）lam/cost/N_tx/payload 全一致 → **PASS**（计数按真实循环数，P1-3/P2）。
（3.9s）

## 2. Calibration（017 §三/§五：shared worlds、grid 冻结）

> worlds：stratified N_CAL=600/hyp @ H=96（FG/D8 完全共用，与 test 完全分离）；θ̂_m = feasible（U95(P_FA)≤α ∧ U95(P_MD)≤β）中 Ê_cal[B_m] 最小（tie-break: (ρ,η) 字典序小者）。

| method | θ̂_m=(ρ*,η*) | feasible 数/28 | min Ê_cal[B] @ θ̂_m |
| --- | --- | --- | --- |
| FG | **(256, 0.8)** | 8 | 31.7975 bits |
| D8 | **(256, 0.8)** | 10 | 36.8800 bits |

#### Feasible region（017 §七：(ρ,η) → {INFEASIBLE, UNCERTAIN, FEASIBLE} + Ê[B]）

**FG**：
| ρ | η | U95(P_FA) | U95(P_MD) | 分类 | Ê_cal[B] |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.8 | 0.1229 | 0.3737 | UNCERTAIN | — |
| 128 | 1.0 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.2 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.4 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.6 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.8 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 2.0 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 256 | 0.8 | 0.1026 | 0.3567 | FEASIBLE | 31.7975 **⇐ θ̂** |
| 256 | 1.0 | 0.0838 | 0.4178 | UNCERTAIN | — |
| 256 | 1.2 | 0.0800 | 0.4296 | UNCERTAIN | — |
| 256 | 1.4 | 0.0705 | 0.4532 | UNCERTAIN | — |
| 256 | 1.6 | 0.0469 | 0.5033 | INFEASIBLE | — |
| 256 | 1.8 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 256 | 2.0 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 512 | 0.8 | 0.0970 | 0.3052 | FEASIBLE | 38.9892 |
| 512 | 1.0 | 0.0876 | 0.3258 | FEASIBLE | 43.2800 |
| 512 | 1.2 | 0.0705 | 0.4178 | UNCERTAIN | — |
| 512 | 1.4 | 0.0588 | 0.4279 | UNCERTAIN | — |
| 512 | 1.6 | 0.0489 | 0.4212 | UNCERTAIN | — |
| 512 | 1.8 | 0.0449 | 0.4699 | UNCERTAIN | — |
| 512 | 2.0 | 0.0429 | 0.4883 | INFEASIBLE | — |
| 1024 | 0.8 | 0.0989 | 0.2792 | FEASIBLE | 46.8042 |
| 1024 | 1.0 | 0.0876 | 0.3138 | FEASIBLE | 46.5967 |
| 1024 | 1.2 | 0.0724 | 0.3327 | FEASIBLE | 47.9475 |
| 1024 | 1.4 | 0.0686 | 0.3481 | FEASIBLE | 48.5917 |
| 1024 | 1.6 | 0.0509 | 0.3924 | FEASIBLE | 45.2058 |
| 1024 | 1.8 | 0.0429 | 0.4229 | UNCERTAIN | — |
| 1024 | 2.0 | 0.0388 | 0.4548 | UNCERTAIN | — |

**D8**：
| ρ | η | U95(P_FA) | U95(P_MD) | 分类 | Ê_cal[B] |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.8 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.0 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.2 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.4 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.6 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 1.8 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 128 | 2.0 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 256 | 0.8 | 0.0970 | 0.3464 | FEASIBLE | 36.8800 **⇐ θ̂** |
| 256 | 1.0 | 0.0819 | 0.3873 | FEASIBLE | 36.9000 |
| 256 | 1.2 | 0.0724 | 0.4212 | UNCERTAIN | — |
| 256 | 1.4 | 0.0627 | 0.4616 | UNCERTAIN | — |
| 256 | 1.6 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 256 | 1.8 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 256 | 2.0 | 0.0064 | 1.0000 | INFEASIBLE | — |
| 512 | 0.8 | 0.0819 | 0.3138 | FEASIBLE | 44.2000 |
| 512 | 1.0 | 0.0800 | 0.3481 | FEASIBLE | 44.9000 |
| 512 | 1.2 | 0.0724 | 0.3737 | FEASIBLE | 45.1200 |
| 512 | 1.4 | 0.0666 | 0.4026 | UNCERTAIN | — |
| 512 | 1.6 | 0.0569 | 0.4380 | UNCERTAIN | — |
| 512 | 1.8 | 0.0469 | 0.4733 | UNCERTAIN | — |
| 512 | 2.0 | 0.0408 | 0.5083 | INFEASIBLE | — |
| 1024 | 0.8 | 0.0914 | 0.2862 | FEASIBLE | 52.0200 |
| 1024 | 1.0 | 0.0857 | 0.3190 | FEASIBLE | 51.8200 |
| 1024 | 1.2 | 0.0762 | 0.3515 | FEASIBLE | 50.7800 |
| 1024 | 1.4 | 0.0647 | 0.3805 | FEASIBLE | 50.3400 |
| 1024 | 1.6 | 0.0569 | 0.3992 | FEASIBLE | 49.8600 |
| 1024 | 1.8 | 0.0489 | 0.4296 | UNCERTAIN | — |
| 1024 | 2.0 | 0.0408 | 0.4632 | UNCERTAIN | — |

- inv-3/inv-4（逐 episode 断言：B=16·N_tx+B_payload、B≤H）：calibration 全 56 次 θ-run 中 violations=0 → **PASS**。
- sensitivity（017 §五 校准边界噪声）：FG 的 Ê_cal[B] 最小候选 = (128, 1.0)（分类 INFEASIBLE，Ê[B]=0.0000）≠ θ̂_FG —— 硬 INFEASIBLE（L95 已超限）⇒ θ̂ 即族内可行最小。正式协议不临场放宽（017 §六），论文以 sensitivity 处理。
- sensitivity（017 §五 校准边界噪声）：D8 的 Ê_cal[B] 最小候选 = (128, 0.8)（分类 INFEASIBLE，Ê[B]=0.0000）≠ θ̂_D8 —— 硬 INFEASIBLE（L95 已超限）⇒ θ̂ 即族内可行最小。正式协议不临场放宽（017 §六），论文以 sensitivity 处理。
（238.2s；累计 245.3s）

## 3. Primary Gate @ H=96（017 §六/§八，θ̂ 冻结、test fresh）

> worlds：stratified N_TEST=1600/hyp（fresh seeds，calibration 完全不可见）；FG 与 D8 在**同一 worlds** 上（paired CRN，planner 确定性）；**统计按 017 §六 方案 A：N_TEST 一次性冻结，无 staged escalation**（Hoeffding 为 fixed-N 95% one-sided）。

### H=96（θ̂_FG=(256, 0.8)、θ̂_D8=(256, 0.8) 冻结）

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0756 | 0.0896 | 0.2894 | 0.3121 | FEASIBLE | 1.6587 | 5.6575 | 32.1975 |
| Direct8 | 0.0712 | 0.0849 | 0.2913 | 0.3140 | FEASIBLE | 1.5634 | 12.5075 | 37.5225 |

- paired D=E[B^FG]−E[B^D8] = -5.3250；**Hoeffding U95=-1.1710**、**Hoeffding L95=-9.4790**（fixed-N δ=0.05，n_paired=3200，D∈[−96,96]）；t 参考 [-5.7084, -4.9416]。
- inv-3/inv-4 逐 episode violations（B=16·N_tx+B_payload、B≤H）：0 → **PASS**。

- **Gate（017 §八）**：FG 分类=FEASIBLE、D8 分类=FEASIBLE；U95(E[D])=-1.1710<0，L95(E[D])=-9.4790≤0。
  → **G2 PASS**。
  双方 FEASIBLE 且 U95(E[D])<0 ⇒ 统计证实 granularity 在 separately calibrated 下仍省 communication bits。

（25.2s；累计 270.6s）

## 4. Secondary stress @ H=48（017 §四：同冻结 controller，诚实报告 operating-region boundary；不为 H=48 重新校准）

### H=48（θ̂_FG=(256, 0.8)、θ̂_D8=(256, 0.8) 冻结）

| 方法 | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | E[B_payload] | E[B] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FG | 0.0688 | 0.0822 | 0.3475 | 0.3712 | FEASIBLE | 1.4384 | 5.1312 | 28.1462 |
| Direct8 | 0.0781 | 0.0923 | 0.3281 | 0.3515 | FEASIBLE | 1.3822 | 11.0575 | 33.1725 |

- paired D=E[B^FG]−E[B^D8] = -5.0263；**Hoeffding U95=-2.9493**、**Hoeffding L95=-7.1032**（fixed-N δ=0.05，n_paired=3200，D∈[−48,48]）；t 参考 [-5.1823, -4.8702]。
- inv-3/inv-4 逐 episode violations（B=16·N_tx+B_payload、B≤H）：0 → **PASS**。

- **Gate（017 §八）**：FG 分类=FEASIBLE、D8 分类=FEASIBLE；U95(E[D])=-2.9493<0，L95(E[D])=-7.1032≤0。
  → **G2 PASS**。
  双方 FEASIBLE 且 U95(E[D])<0 ⇒ 统计证实 granularity 在 separately calibrated 下仍省 communication bits。

（20.7s；累计 291.3s）

## 5. Secondary diagnostics（017 §七）

### 5.1 分假设 bit 分解 E[B|H0]、E[B|H1]（防止平均 bit gain 只来自一个 hypothesis）

**H=96**：
| method | E[B|H0] | E[B|H1] | E[B] | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- |
| FG | 27.8275 | 36.5675 | 32.1975 | 1.6587 | 5.6575 |
| Direct8 | 32.8800 | 42.1650 | 37.5225 | 1.5634 | 12.5075 |

**H=48**：
| method | E[B|H0] | E[B|H1] | E[B] | E[N_tx] | E[B_payload] |
| --- | --- | --- | --- | --- | --- |
| FG | 25.5994 | 30.6931 | 28.1462 | 1.4384 | 5.1312 |
| Direct8 | 30.6000 | 35.7450 | 33.1725 | 1.3822 | 11.0575 |

### 5.2 Cross-evaluation @ H=96（017 §七：FG@θ̂_FG、FG@θ̂_D8、D8@θ̂_D8、D8@θ̂_FG；secondary，不参与 Gate）

| 配置 | θ | U95(P_FA) | U95(P_MD) | 分类 | E[B] | E[B|H0] | E[B|H1] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FG@θ̂_FG | (256, 0.8) | 0.0896 | 0.3121 | FEASIBLE | 32.1975 | 27.8275 | 36.5675 |
| FG@θ̂_D8 | (256, 0.8) | 0.0896 | 0.3121 | FEASIBLE | 32.1975 | 27.8275 | 36.5675 |
| D8@θ̂_D8 | (256, 0.8) | 0.0849 | 0.3140 | FEASIBLE | 37.5225 | 32.8800 | 42.1650 |
| D8@θ̂_FG | (256, 0.8) | 0.0849 | 0.3140 | FEASIBLE | 37.5225 | 32.8800 | 42.1650 |

- 分离 granularity vs dual operating point（017 §七）：
  - E[B^FG@θ̂FG − B^D8@θ̂D8] = -5.3250（主 Gate 的 D）；
  - 同 operating point（θ̂_D8）下 FG vs D8: E[B^FG@θ̂D8 − B^D8@θ̂D8] = -5.3250 —— 若仍明显为负，优势主要来自 action-space granularity；
  - FG 换 operating point: E[B^FG@θ̂FG − B^FG@θ̂D8] = 0.0000 —— 自身上 dual operating point 的增益。
  - D8 在 θ̂_FG 与 θ̂_D8 下的 E[B] 比较见上表（dd vs df）。
  - **注（017 §七 通道分离）**：θ̂_FG=θ̂_D8 ⇒ operating-point 通道为 0，E[D] 全部归因于 action-space granularity（同一 dual operating point 下 FG 严格更省）——016 §8 的 policy-class 包含关系（Π_D8⊆Π_FG）在该 operating point 上的经验实现。

## 6. Forced-continuation 口径修正 + robustness（017 §一 P1-1/P1-2）

> 参考控制器 θ_ref=(512, 1.2)（= G1r 冻结的 LAM_M=512、η_star=1.2）；公共规则 S_common=min_{A_FG}Q<R。**P1-1** 正式口径 P(F=1|S_common=CONTINUE)（分母只含公共规则说继续的决策状态；把 STOP 状态加入分母只会更低，故同时给出无条件参考）。**P1-2** F 状态上 D8 实付的 full-packet 成本之和 = **gross forced-action cost**（非 causal extra cost，反事实 STOP 会改变后续轨迹），按 episode 归一化。G2 已从设计上消除公共停止器，此审计仅作诊断与 robustness 陈述。

### H=48（θ_ref=(512, 1.2)，fresh N_TEST=1600）

- **P(F=1|S_common=CONTINUE) = 0.0202**（F 状态 99/4911 个继续-决策状态）；无条件参考 P(F=1|所有决策状态) = 0.0155（99/6400）——低于条件口径（017 P1-1 预期）。
- P(episode contains F) = 0.0309。
- **gross forced-action cost = 2376.0000 bits = 0.7425 bit/episode**（D8 在公共规则下被迫支付的 8-bit 通信成本；017 P1-2 命名，含 setup+payload）。
- 结构观察：per-F 成本恒 = 24.0 bits（=16 setup+8 payload ⇒ 所有 F 状态发生在 r_cur=0 的首决策状态；与 G1r 的 1560/65=24.0 逐位一致，审计实现与 G1r 语义吻合）。

### H=96（θ_ref=(512, 1.2)，fresh N_TEST=1600）

- **P(F=1|S_common=CONTINUE) = 0.0313**（F 状态 198/6319 个继续-决策状态）；无条件参考 P(F=1|所有决策状态) = 0.0220（198/9017）——低于条件口径（017 P1-1 预期）。
- P(episode contains F) = 0.0597。
- **gross forced-action cost = 4752.0000 bits = 1.4850 bit/episode**（D8 在公共规则下被迫支付的 8-bit 通信成本；017 P1-2 命名，含 setup+payload）。
- 结构观察：per-F 成本恒 = 24.0 bits（=16 setup+8 payload ⇒ 所有 F 状态发生在 r_cur=0 的首决策状态；与 G1r 的 1560/65=24.0 逐位一致，审计实现与 G1r 语义吻合）。

> **Robustness statement（017 §一）**：即使极端地把 gross forced-action cost 全部视为 leakage bias（H=96：1.4850 bit/episode），也远低于 G1/G1r 的 −12.31 bit gap （H=96）——G1 的 granularity 收益不可能主要由 action-set leakage 解释；G2 已从设计上消除 public/common 停止器。

## 结论

- **Primary H=96：G2 PASS**（双方 FEASIBLE=PASS（FEASIBLE/FEASIBLE），U95(E[D])=-1.1710，L95(E[D])=-9.4790）。
- **Secondary H=48：G2 PASS**（双方 FEASIBLE=PASS（FEASIBLE/FEASIBLE），U95(E[D])=-2.9493）。

> **论文正式表述（017 §八）**：Under separately calibrated QoS-dual controllers and fresh held-out testing, adaptive feedback granularity achieves statistically certified communication savings over the Direct8 policy family considered.

- **B0.7-G2 定位（017 §final）**：separately calibrated QoS-dual policy-family certification。若 **G2 PASS**，论文核心 performance 主线闭环；G3（DualCPI 是否还有独立增益）应变成“是否值得纳入主算法”的 Gate，而不是必做的性能增强——避免系统从“反馈粒度这一核心科学问题”跑回复杂 planner/sample-complexity 工程。

总耗时: 383.2s

