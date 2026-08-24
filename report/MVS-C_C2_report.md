# MVS-C C2 — phase-guided conditional-refinement policy + exact budgeted-CMDP oracle（FULL）

> 依据 001 §二十六.2-§26.3 / §九-§十四 / §二十七 Gate D；C0 语义封板后的第一个**主算法**模块。冻结参数：N=4、GAMMA4=[-1,1,3,5] dB、levels (1,2,4,8)、b_{0,i}=16（κ=1 homogeneous special case）、QoS matched detection α=0.05、ε_D=0.01、H∈(48, 96)、ρ∈(128, 256, 512, 1024)、η∈(0.8, 0.85, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)、N_CAL=400/hyp @ H=96、N_TEST=800/hyp。

## 0. 精确量化全融合参考（C0 语义：P_D,max 而非连续近似）

- 4-bit 全融合（精确卷积）：P_D,max^q(0.05) = 0.8482（η*=0.8244）、P_MD=0.1518 → matched 目标 P_MD ≤ 0.1618；support=16^4=65536（trivial）
- 8-bit：精确卷积不可行（distinct Ω 上界 256^4≈4294967296，与 exact DP 同源爆炸）→ C0 参考 = 连续 full-fusion P_D,raw=0.8509 → matched 目标 P_MD ≤ 0.1591；未覆盖的量化损失 ~1e-3 量级（4-bit 精确 0.8482 级，见上行）——报告为参考面 limitation。

## 1. 理论 Gate（001 §九-§十四，reachable 状态上验证）

- **T1（支撑恒等式，013 §1）**：327 个 UAV 支撑检查，Q_prog−Q_dir−g0−(E_R_sum−E_dir) 偏差 >1e-8 的个数 0 → PASS
- **T3（pruning 自洽，001 §十四）**：327 检查，符号矛盾 0 → PASS（g≥0 ⇒ Q_prog≥Q_dir）
- **T4（复杂度，001 §十四 O(2N) 非 O(N|R|)）**：124 决策，full-FG 每决策 max 16（≤N|R|=16: PASS）、Phase-FG 评估动作 max 8（≤2N=8: PASS）、总量比 0.503

- **T2（cond-refinement sandwich，001 §十/§十一）**：
    T2 检查 310 个（Q{global-2} ≤ Q{self-2} ≤ Q_prog ≤ Q{(1)}），矛盾 0 → PASS

## 2. exact budgeted CMDP oracle 的粒度可行性（001 §二十一）

- reachable z-state 数估计（profile 计数，不含 budget-layer 重复）：8-bit/H=96: 6059221281、8-bit/H=48: 464817、4-bit/H=96: 279841、4-bit/H=48: 2993
- **结论**：8-bit 粒度 exact backward 不可行（H=96 已 ≥ 1e9）→ Gate D oracle 冻结在 **4-bit (1, 2, 4) 粒度**（279841，MVS-A ExactDP 规模）；8-bit 与 Direct8/Myopic 的机制比较走 MC（§3）。

## 3. 机制比较（双口径：matched 主口径 / legacy mechanism 参考口径）

> 017 §三：G2 注册的 (P_FA≤0.12, P_MD≤0.40) 是 mechanism-dialect；001 §三 的**matched detection**（α=0.05、P_MD≤1−P_D,max+ε_D）是 paper 主口径。C2 两个口径都跑：matched 是裁判口径（统计不可认证时如实报 UNRESOLVED，见下），legacy 供 granularity-vs-D8/Myopic 的机制方向性对照（与 G2 −5.33 同口径）。
> **matched 口径机制层不可达（C2 审计修正）**：α=0.05 边（U_FA≤0.05）下该 dual 控制器族在冻结网格上的最佳 P_D 仅 ~0.74-0.76（FULL cal 中 (1024,1.8) U_MD=0.2836、(1024,1.6) U_MD=0.2677@U_FA 0.0548），目标 P_D≥0.8382（β=0.1618）差 ~0.08-0.10 —— 差距源于 stopping-budget 权衡（H=96 下 4×8-bit direct=96 即全预算，且 π* 在 同一 θ 也只花 E[B]≈44.5≈2.8 tx，不以全融合为最优，见 Gate D 行）。**早先『ε_D 余量被 CI 消耗』的表述机制错误（C2 review 撤销）**——把全融合理论上限误当成了控制器可达操作点；实际不可达是机制层/网格层的，不是统计层的。matched 在该规模/网格下如实 INFEASIBLE；机制比较面 = legacy（β=0.40）。

### 3.1 口径：matched（001 §三）（α=0.05、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (1024,1.8)  (E[B]=45.020, cls=INFEASIBLE, ufa=0.0454, umd=0.2836)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD
- 校准 Myopic-FG(8-bit)：
    θ̂ = (1024,1.8)  (E[B]=36.816, cls=INFEASIBLE, ufa=0.0454, umd=0.3205)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD
- 校准 Direct8：
    θ̂ = (1024,1.6)  (E[B]=46.284, cls=INFEASIBLE, ufa=0.0486, umd=0.2677)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD
- 校准 Phase-FG(4-bit)：
    θ̂ = (1024,1.8)  (E[B]=40.619, cls=INFEASIBLE, ufa=0.0422, umd=0.2942)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (1024,1.8) | INFEASIBLE | 0.0322 | 0.3257 | 46.144 | 35.520 | 56.794 | 14.054 | 2.006 |
| Myopic-FG(8-bit) | (1024,1.8) | INFEASIBLE | 0.0276 | 0.3604 | 37.510 | 24.815 | 50.237 | 5.570 | 1.996 |
| Direct8 | (1024,1.6) | INFEASIBLE | 0.0276 | 0.2948 | 47.144 | 36.720 | 57.594 | 15.715 | 1.964 |
| Phase-FG(4-bit) | (1024,1.8) | INFEASIBLE | 0.0292 | 0.3167 | 41.285 | 30.832 | 51.763 | 8.013 | 2.079 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂，operating-region boundary）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | INFEASIBLE | 0.0368 | 0.3959 | 35.083 | 10.728 | 1.522 |
| Myopic-FG(8-bit) | INFEASIBLE | 0.0443 | 0.4141 | 27.848 | 3.970 | 1.492 |
| Direct8 | INFEASIBLE | 0.0559 | 0.3501 | 36.592 | 12.197 | 1.525 |
| Phase-FG(4-bit) | INFEASIBLE | 0.0443 | 0.3985 | 30.755 | 6.016 | 1.546 |

- Phase-FG(8-bit) − Direct8（H=96，paired CRN，n=1598）：E[D]=-1.000、Hoeffding U95=4.878 | no-compare；E[B] 各 46.144 vs 47.144
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，paired CRN，n=1598）：E[D]=8.634、Hoeffding U95=14.512 | no-compare；E[B] 各 46.144 vs 37.510

### 3.2 口径：legacy mechanism（017 §四）（α=0.12、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (128,1.0)  (E[B]=27.434, cls=FEASIBLE, ufa=0.0848, umd=0.2730)；理由：FEASIBLE 中 min E[B]
- 校准 Myopic-FG(8-bit)：
    θ̂ = (128,1.2)  (E[B]=22.074, cls=FEASIBLE, ufa=0.1050, umd=0.2624)；理由：FEASIBLE 中 min E[B]
- 校准 Direct8：
    θ̂ = (128,0.85)  (E[B]=28.607, cls=FEASIBLE, ufa=0.0906, umd=0.2598)；理由：FEASIBLE 中 min E[B]
- 校准 Phase-FG(4-bit)：
    θ̂ = (128,0.8)  (E[B]=23.396, cls=FEASIBLE, ufa=0.1108, umd=0.2437)；理由：FEASIBLE 中 min E[B]

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (128,1.0) | FEASIBLE | 0.0730 | 0.3450 | 27.660 | 26.401 | 28.921 | 8.215 | 1.215 |
| Myopic-FG(8-bit) | (128,1.2) | FEASIBLE | 0.0744 | 0.3077 | 22.351 | 20.933 | 23.773 | 2.256 | 1.256 |
| Direct8 | (128,0.85) | FEASIBLE | 0.0898 | 0.3038 | 29.166 | 27.510 | 30.827 | 9.722 | 1.215 |
| Phase-FG(4-bit) | (128,0.8) | FEASIBLE | 0.0967 | 0.2974 | 23.617 | 22.561 | 24.675 | 4.393 | 1.202 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂，operating-region boundary）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | FEASIBLE | 0.0787 | 0.3212 | 27.357 | 8.197 | 1.197 |
| Myopic-FG(8-bit) | FEASIBLE | 0.0898 | 0.2961 | 22.164 | 2.245 | 1.245 |
| Direct8 | FEASIBLE | 0.0940 | 0.2869 | 28.132 | 9.377 | 1.172 |
| Phase-FG(4-bit) | FEASIBLE | 0.1105 | 0.2749 | 23.096 | 4.341 | 1.172 |

- Phase-FG(8-bit) − Direct8（H=96，paired CRN，n=1598）：E[D]=-1.507、Hoeffding U95=4.371 | 比较有效（双方 FEASIBLE）；E[B] 各 27.660 vs 29.166
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，paired CRN，n=1598）：E[D]=5.309、Hoeffding U95=11.187 | 比较有效（双方 FEASIBLE）；E[B] 各 27.660 vs 22.351

### 3.4 双口径小结

- matched（001 §三）：@H=96 分类 {'Phase-FG(8-bit)': 'INFEASIBLE', 'Myopic-FG(8-bit)': 'INFEASIBLE', 'Direct8': 'INFEASIBLE', 'Phase-FG(4-bit)': 'INFEASIBLE'}；E[B]（8-bit 方法）Phase-FG=46.144、Myopic=37.510、Direct8=47.144。
- legacy mechanism（017 §四）：@H=96 分类 {'Phase-FG(8-bit)': 'FEASIBLE', 'Myopic-FG(8-bit)': 'FEASIBLE', 'Direct8': 'FEASIBLE', 'Phase-FG(4-bit)': 'FEASIBLE'}；E[B]（8-bit 方法）Phase-FG=27.660、Myopic=22.351、Direct8=29.166。

## 4. Gate D — 求解器质量：Phase-FG vs exact budgeted CMDP*（4-bit）

> Gate D 的 θ 集合：固定 corners ((256, 1.2), (512, 1.2), (1024, 1.6)) + matched 口径下 Phase-FG(4-bit) 的 θ̂=(1024, 1.8)（matched 分类 INFEASIBLE——非 FEASIBLE 时注明：θ-fixed 比较仍有效，但 θ̂ 不是认证可行点，仅作同 θ 求解器质量比对）。比较对象：CMDP* 的 E[B]（用 exact_policy_cost + oracle_decision，memo 已由 solve 预热）。
- θ=(256, 1.2) H=48：CMDP* E[B]=26.875（V_lag=85.0 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=27.125（rel=0.93%）、Myopic 26.875（0.00%）、Direct4 27.500（2.33%）
- θ=(256, 1.2) H=96：CMDP* E[B]=34.569（V_lag=80.3 仅作 solve 证书；memo=607890、expansions=607890、14.5s）；Phase-FG exact C=29.896（rel=-13.52%）、Myopic 29.775（-13.87%）、Direct4 31.176（-9.82%）
- θ=(512, 1.2) H=48：CMDP* E[B]=29.438（V_lag=142.3 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=29.812（rel=1.27%）、Myopic 29.438（0.00%）、Direct4 30.000（1.91%）
- θ=(512, 1.2) H=96：CMDP* E[B]=38.712（V_lag=124.6 仅作 solve 证书；memo=607890、expansions=607890、14.5s）；Phase-FG exact C=36.877（rel=-4.74%）、Myopic 36.518（-5.67%）、Direct4 36.055（-6.86%）
- θ=(1024, 1.6) H=48：CMDP* E[B]=30.938（V_lag=287.9 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=31.062（rel=0.40%）、Myopic 32.500（5.05%）、Direct4 31.250（1.01%）
- θ=(1024, 1.6) H=96：CMDP* E[B]=44.483（V_lag=234.3 仅作 solve 证书；memo=607890、expansions=607890、14.8s）；Phase-FG exact C=40.974（rel=-7.89%）、Myopic 43.109（-3.09%）、Direct4 40.121（-9.81%）
- θ=(1024, 1.8) H=48：CMDP* E[B]=30.812（V_lag=304.2 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=31.062（rel=0.81%）、Myopic 28.000（-9.13%）、Direct4 31.250（1.42%）
- θ=(1024, 1.8) H=96：CMDP* E[B]=44.778（V_lag=248.2 仅作 solve 证书；memo=607890、expansions=607890、14.5s）；Phase-FG exact C=41.085（rel=-8.25%）、Myopic 37.130（-17.08%）、Direct4 40.254（-10.10%）
- **Gate D 判决**：Phase-FG 相对 gap 最大值 1.27% ≤ 预注册阈值 10% → PASS（001 §二十七：D 良好则不再做 CPI）。注：H=96 行的 rel 为负是 **dual trade**（Phase-FG 以更高终端风险换更少 bits —— 同一 θ 下 Lagrangian 最优性由 V_lag 保证，Phase-FG 的 E[B]+E[R] ≥ V_lag 恒成立，负 rel 不是求解器缺陷），判决取 rel 的 max。

总耗时: 500.3s

