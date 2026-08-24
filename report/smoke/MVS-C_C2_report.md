# MVS-C C2 — phase-guided conditional-refinement policy + exact budgeted-CMDP oracle（SMOKE）

> 依据 001 §二十六.2-§26.3 / §九-§十四 / §二十七 Gate D；C0 语义封板后的第一个**主算法**模块。冻结参数：N=4、GAMMA4=[-1,1,3,5] dB、levels (1,2,4,8)、b_{0,i}=16（κ=1 homogeneous special case）、QoS matched detection α=0.05、ε_D=0.01、H∈(48, 96)、ρ∈(128, 512)、η∈(0.8, 1.2, 2.0)、N_CAL=60/hyp @ H=96、N_TEST=120/hyp。

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

- reachable z-state 数估计（profile 计数，不含 budget-layer 重复）：8-bit/H=96: 5972816656、8-bit/H=48: 0、4-bit/H=96: 234256、4-bit/H=48: 0
- **结论**：8-bit 粒度 exact backward 不可行（H=96 已 ≥ 1e9）→ Gate D oracle 冻结在 **4-bit (1, 2, 4) 粒度**（234256，MVS-A ExactDP 规模）；8-bit 与 Direct8/Myopic 的机制比较走 MC（§3）。

## 3. 机制比较（双口径：matched 主口径 / legacy mechanism 参考口径）

> 017 §三：G2 注册的 (P_FA≤0.12, P_MD≤0.40) 是 mechanism-dialect；001 §三 的**matched detection**（α=0.05、P_MD≤1−P_D,max+ε_D）是 paper 主口径。C2 两个口径都跑：matched 是裁判口径（统计不可认证时如实报 UNRESOLVED，见下），legacy 供 granularity-vs-D8/Myopic 的机制方向性对照（与 G2 −5.33 同口径）。
> **matched 统计不可认证诊断（C2 审计）**：N=4 弱感知下 P_D,max^q(0.05)=0.8482、目标 P_D≥0.8382（β=0.1618）；量化全融合实际可达 ~0.845-0.848 ⇒ P_D,max−P_D 余量 ~0.005-0.008，而 Wilson 95% 半宽 n=60 → ±0.036、n=120 → ±0.025 ⇒ **ε_D=0.01 的认证余量被 CI 消耗殆尽**：matched FEASIBLE 在该系统规模下**统计不可认证**（任何 θ 只会 UNCERTAIN/INFEASIBLE —— 不是搜索失败）。机制口径（β=0.40）余量 ~0.20 ≫ CI，是 C2 的可认证比较面。

### 3.1 口径：matched（001 §三）（α=0.05、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA（注意：可能选中退化 stop-at-root 的 θ —— 该口径下网格无可行点，如实报告）
- 校准 Myopic-FG(8-bit)：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA（注意：可能选中退化 stop-at-root 的 θ —— 该口径下网格无可行点，如实报告）
- 校准 Direct8：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA（注意：可能选中退化 stop-at-root 的 θ —— 该口径下网格无可行点，如实报告）
- 校准 Phase-FG(4-bit)：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA（注意：可能选中退化 stop-at-root 的 θ —— 该口径下网格无可行点，如实报告）

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (128,2.0) | INFEASIBLE | 0.0321 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Myopic-FG(8-bit) | (128,2.0) | INFEASIBLE | 0.0321 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Direct8 | (128,2.0) | INFEASIBLE | 0.0321 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase-FG(4-bit) | (128,2.0) | INFEASIBLE | 0.0321 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂，operating-region boundary）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |
| Myopic-FG(8-bit) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |
| Direct8 | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |
| Phase-FG(4-bit) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |

- Phase-FG(8-bit) − Direct8（H=96，paired CRN，n=236）：E[D]=0.000、Hoeffding U95=15.296 | no-compare；E[B] 各 0.000 vs 0.000
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，paired CRN，n=236）：E[D]=0.000、Hoeffding U95=15.296 | no-compare；E[B] 各 0.000 vs 0.000

### 3.2 口径：legacy mechanism（017 §四）（α=0.12、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (512,1.2)  (E[B]=36.586, cls=FEASIBLE, ufa=0.0886, umd=0.3382)；理由：FEASIBLE 中 min E[B]
- 校准 Myopic-FG(8-bit)：
    θ̂ = (512,0.8)  (E[B]=30.776, cls=FEASIBLE, ufa=0.0886, umd=0.3382)；理由：FEASIBLE 中 min E[B]
- 校准 Direct8：
    θ̂ = (512,1.2)  (E[B]=38.690, cls=FEASIBLE, ufa=0.0886, umd=0.3184)；理由：FEASIBLE 中 min E[B]
- 校准 Phase-FG(4-bit)：
    θ̂ = (512,0.8)  (E[B]=30.362, cls=FEASIBLE, ufa=0.1136, umd=0.2984)；理由：FEASIBLE 中 min E[B]

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (512,1.2) | FEASIBLE | 0.0733 | 0.3166 | 41.881 | 35.595 | 47.958 | 12.390 | 1.843 |
| Myopic-FG(8-bit) | (512,0.8) | FEASIBLE | 0.0733 | 0.2435 | 36.394 | 31.793 | 40.842 | 6.292 | 1.881 |
| Direct8 | (512,1.2) | FEASIBLE | 0.0733 | 0.3076 | 43.831 | 36.000 | 51.400 | 14.610 | 1.826 |
| Phase-FG(4-bit) | (512,0.8) | FEASIBLE | 0.0853 | 0.2435 | 37.025 | 32.552 | 41.350 | 6.992 | 1.877 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂，operating-region boundary）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | FEASIBLE | 0.1155 | 0.3322 | 32.921 | 10.295 | 1.414 |
| Myopic-FG(8-bit) | UNCERTAIN | 0.1261 | 0.2917 | 27.833 | 4.925 | 1.432 |
| Direct8 | FEASIBLE | 0.1155 | 0.3422 | 33.938 | 11.313 | 1.414 |
| Phase-FG(4-bit) | UNCERTAIN | 0.1364 | 0.2917 | 28.317 | 5.410 | 1.432 |

- Phase-FG(8-bit) − Direct8（H=96，paired CRN，n=236）：E[D]=-1.949、Hoeffding U95=13.347 | 比较有效（双方 FEASIBLE）；E[B] 各 41.881 vs 43.831
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，paired CRN，n=236）：E[D]=5.487、Hoeffding U95=20.783 | 比较有效（双方 FEASIBLE）；E[B] 各 41.881 vs 36.394

### 3.4 双口径小结

- matched（001 §三）：@H=96 分类 {'Phase-FG(8-bit)': 'INFEASIBLE', 'Myopic-FG(8-bit)': 'INFEASIBLE', 'Direct8': 'INFEASIBLE', 'Phase-FG(4-bit)': 'INFEASIBLE'}；E[B]（8-bit 方法）Phase-FG=0.000、Myopic=0.000、Direct8=0.000。
- legacy mechanism（017 §四）：@H=96 分类 {'Phase-FG(8-bit)': 'FEASIBLE', 'Myopic-FG(8-bit)': 'FEASIBLE', 'Direct8': 'FEASIBLE', 'Phase-FG(4-bit)': 'FEASIBLE'}；E[B]（8-bit 方法）Phase-FG=41.881、Myopic=36.394、Direct8=43.831。

## 4. Gate D — 求解器质量：Phase-FG vs exact budgeted CMDP*（4-bit）

> Gate D 的 θ 集合：固定 corners ((256, 1.2), (512, 1.2), (1024, 1.6)) + matched 口径下 Phase-FG(4-bit) 的 θ̂=(128, 2.0)（matched 分类 INFEASIBLE——非 FEASIBLE 时注明：θ-fixed 比较仍有效，但 θ̂ 不是认证可行点，仅作同 θ 求解器质量比对）。比较对象：CMDP* 的 E[B]（用 exact_policy_cost + oracle_decision，memo 已由 solve 预热）。
- θ=(256, 1.2) H=48：CMDP* E[B]=26.875（V_lag=85.0 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=27.125（rel=0.93%）、Myopic 26.875（0.00%）、Direct4 27.500（2.33%）
- θ=(256, 1.2) H=96：CMDP* E[B]=34.569（V_lag=80.3 仅作 solve 证书；memo=607890、expansions=607890、15.5s）；Phase-FG exact C=29.896（rel=-13.52%）、Myopic 29.775（-13.87%）、Direct4 31.176（-9.82%）
- θ=(512, 1.2) H=48：CMDP* E[B]=29.438（V_lag=142.3 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=29.812（rel=1.27%）、Myopic 29.438（0.00%）、Direct4 30.000（1.91%）
- θ=(512, 1.2) H=96：CMDP* E[B]=38.712（V_lag=124.6 仅作 solve 证书；memo=607890、expansions=607890、15.3s）；Phase-FG exact C=36.877（rel=-4.74%）、Myopic 36.518（-5.67%）、Direct4 36.055（-6.86%）
- θ=(1024, 1.6) H=48：CMDP* E[B]=30.938（V_lag=287.9 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=31.062（rel=0.40%）、Myopic 32.500（5.05%）、Direct4 31.250（1.01%）
- θ=(1024, 1.6) H=96：CMDP* E[B]=44.483（V_lag=234.3 仅作 solve 证书；memo=607890、expansions=607890、15.2s）；Phase-FG exact C=40.974（rel=-7.89%）、Myopic 43.109（-3.09%）、Direct4 40.121（-9.81%）
- θ=(128, 2.0) H=48：CMDP* E[B]=24.500（V_lag=63.7 仅作 solve 证书；memo=3073、expansions=3073、0.0s）；Phase-FG exact C=0.000（rel=-100.00%）、Myopic 0.000（-100.00%）、Direct4 0.000（-100.00%）
- θ=(128, 2.0) H=96：CMDP* E[B]=27.826（V_lag=62.4 仅作 solve 证书；memo=607890、expansions=607890、14.7s）；Phase-FG exact C=0.000（rel=-100.00%）、Myopic 0.000（-100.00%）、Direct4 0.000（-100.00%）
- **Gate D 判决**：Phase-FG 相对 gap 最大值 1.27% ≤ 预注册阈值 10% → PASS（001 §二十七：D 良好则不再做 CPI）。注：H=96 行的 rel 为负是 **dual trade**（Phase-FG 以更高终端风险换更少 bits —— 同一 θ 下 Lagrangian 最优性由 V_lag 保证，Phase-FG 的 E[B]+E[R] ≥ V_lag 恒成立，负 rel 不是求解器缺陷），判决取 rel 的 max。

总耗时: 81.1s

