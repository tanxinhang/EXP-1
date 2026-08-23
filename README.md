# O-PEF MVS-A/B0 — 最小可实现系统（R1→R2.1→MVS-B0 演进）

依据 [SystemModel.md](./SystemModel.md)（O-PEF v2.1）搭建的**可运行、可复现**实现。
当前版本完成 **MVS-A（v0→R1→R1.1→R2→R2.1，已封板）** 与 **MVS-B0
（Sparse-State Header-Aware Cross-Level O-PEF）**：

- 单目标二元检测，`N = 4` 架 UAV，Gaussian 局部检测器（§30–§32）；
- 每 UAV 一个 **nested 多精度量化器**（0/1/2/4 bit，§7、§14）；
- **log-domain 软融合**与 message-LLR，粗→细证据的 *replace-not-add* 更新（§8、§9、§11）；
- 证据状态 `z = (z_1,…,z_N)` 作为有限充分 Markov 状态（§10）；
- **Exact DAG-DP**（23^4 = 279,841 状态，memoized backward recursion，非 value iteration，§18、§20、§21）；
- **O-PEF-1**（depth-1 消融，§23）、**O-PEF-2E**（depth-2 精确期望主算法，§24、§27）
  与 **O-PEF-3**（depth-3 诊断性 solver 改进）；
- 基准：Raw Fusion / All-Neighbor / Random-K / SNR Top-K / Censoring / OTS-F / P-OTS /
  Global Fixed Progressive / Static Cost-Aware Progressive（§49–§53）；
- **R1（依据 adcice/001.md 审计）**：双乘子 (μ_M, μ_F) sweep + 策略自身终端判决、
  adjacent-only 动作族（0→1,1→2,2→4）、Lagrangian G2（J=E[B]+μ_M P_M+μ_F P_FA）、
  精确前向概率传播（主结果无需 MC）、G1a/G1b 独立认证、公平基线 1-bit-seeded P-OTS、
  2×2 solver 实验（cross-level × depth-2/3）。
- **R1.1+R2（依据 adcice/002.md）**：R1.1 三个小修复（1bit_POTS 冻结 1-bit 排序、
  exact_np_roc 按假设分别归一化、"不减"文字修正）+ full-precision constrained
  policy-mixture LP（`scipy.linprog`）冻结 **B_DP^CMDP** oracle；R2 实现
  **resource-bounded lookahead**（horizon = 未来 payload bits）`V_h(x)=min{R_stop,
  min_{a:c_a≤h}[c_a+E V_{h−c_a}]}`，硬认证 V_16(x)=V*(x) 机器精度。
- **R2.1（依据 adcice/003.md，MVS-A 封板）**：CMDP **column generation**
  （LP master + ExactDP pricing oracle）认证全局最优 **B_CMDP\***；hard-budget
  （RB-HardBudget，ablation）与 **receding RBL-RH** 分离（每状态重新读取
  policies[H]）；online sparse planner（memoized Solve(x,h)，不建全表）与
  eager 表全等价审计；H<16 才计入 scalability 评选；η_rec 用 B_CMDP\* 重定义。
- **MVS-B0.1a（依据 advice/006.md）**：1bit-POTS seed 后立即停止 + seed-aware LLR；
  真前缀 CRN（paired 对比）；Wilson CI 正式输出；resource lattice 保守化
  （true cost + ceil budget，离散化界 0≤C̃−C<N_txn·Δc）；**reachable-state ΔQ sweep**
  验证 critical feedback-granularity threshold（b*≈8：P(ΔQ>0): 0→0.69）。
- **MVS-B0.3（CR-RBL）**：Confidence-Certified Rollout RBL——base-rollout Q_a^πb
  + anytime Hoeffding 证书（δ_{a,n}=6δ/(π²|A|n²)）+ LUCB challenger + nested-evidence
  CRN（耦合 latent rollout）；return 有界 0≤G≤C_max^rem+R_max；Gates G0（CI 覆盖）、
  G1/G2（N=4/N=8 exact oracle 对比）、G3（certification error≤δ）、G4（scalability
  H=48/64/96/120 无全 cone）、G5（matched QoS 方向性）；创新定位：
  Confidence-certified feedback-granularity-aware evidence acquisition over a
  variable-cost nested-evidence DAG。
- **MVS-B0.3a（依据 advice/007.md，credibility patch）**：修复 B0.3 的口径问题——
  **P0-A 真正跨 action CRN**：每次 MC 迭代采样一个 latent world
  W_m=(H_m, M_1^(8), …, M_N^(8))|x，所有候选动作在同一 W_m 上求值 G_a(W_m)。
  实现发现 world 必须含隐假设 H_m（先按后验采 H、再按 H-条件分布采各 UAV cell）——
  逐 UAV 边缘独立采样破坏 H 诱导的跨 UAV 相关性，会使 rollout 估计系统性偏高
  ~12 bits（对 N=4 exact oracle 验证）；修正后 paired-CRN 方差比
  Var(G_a−G_b)/[Var(G_a)+Var(G_b)] ≈ 0.08–0.15（相对独立采样降 7–12 倍）；
  **P0-B certificate 竞争集含 STOP**（R_stop(x)=min{C₀₁p,C₁₀(1−p)} 精确项：
  reporting 动作需 U_â ≤ min{R_stop, min L_b}+ε）；
  **P0-C G0 oracle = Q_a^{π_b}**（exact_qa_pi_b，非 Q_a^⋆）；**P0-D G1/G3 STOP oracle =
  R_stop(x)**（非 base_policy_value）；**P0-E G5 硬预算** h_t=H−C_t pathwise、C_T≤H 逐样本
  成立；**P1-A anytime coverage gate**（∀n≤n_max）；**P1-B binomial U95 violation gate +
  certification rate**（0-violation 需 ≥59 certified 样本使 U95≤0.05）；**P1-C 回归不变量
  T15–T20**；**P1-D b⋆ 重表述为 root-state threshold**：g_x(b)=E[min{Y_x,b}] 非减凹、
  g_x'(b)=Pr(Y_x>b)，b⋆(x)=inf{b:g_x(b)≥0} 为 state-dependent packetization phase
  boundary（根状态 b⋆(x₀)=7.0，1-bit 子状态 b⋆=∞ ⇒ direct regime）。
- **MVS-B0.3c（依据 advice/008.md，收口补丁）**：**(1) natural 阈值修正**——
  G5 的 natural 指标改为 Ω>η_nat=log(μ_F/μ_M)=1.0（原来错用 Ω>0；与 eval_exact.py
  锁死，新增 T21）；**(2) G5 改名** directional (unmatched) hard-budget comparison
  （P_D 与 E[B] 同时不同，只有 Pareto 方向性，正式 matched-QoS 留 B0.6）；
  **(3) 三格消融**（G7）分离 bias correction 与 variance reduction：marg×ind
  E|Q̂−Q|=8.4 / P(match)=0.70 → joint×ind 5.6/0.73 → joint×paired 5.6/**0.95**，
  耦合效率 κ≈13（n_paired≈n_uncoupled/κ）；**(4) Hoeffding range 收紧**（008 §4）
  B→B_a(x,h)=min{c_max_rem,h}+R_max−c_a（root N=8: ~950→~422@H=48），G3
  certification rate 75.5%→94.5%（ε=40）；**(5) T17 拆分**为 T17a 确定性证书蕴含
  （100% PASS）+ T17b 经验审计，T19 标注为统计 sanity；**(6) E[Y_x] 存在性判据**
  （008 §6）：b⋆(x)<∞ ⟺ E[Y_x]≥0，E[Y_x]<0 ⇒ progressive dominates direct for
  every b_h≥0（根 EY=38.56>0↔b⋆=7；1-bit 子状态 EY<0↔b⋆=∞，analytically）。
- **MVS-B0.1（依据 adcice/005.md，可信度修复+理论拔高）**：修复 1bit-POTS 重复计数；
  共享 CRN + 置信区间；Natural-policy QoS 与 NP ROC 双口径分离；Adaptive Direct-8 最优
  baseline（隔离 UAV 选择与 multi-resolution 收益）；**state-dependent conditional-VoI 定理**
  （Q_prog−Q_dir = E[min{D(x')−Δ₂, b_h}]，b_h=0 ⇒ 渐进支配）；q<7/23 降为 Corollary 并新增
  E[C_future|M^(1)]<7 判据；feedback/setup 成本（b_setup）敏感性；正式 regression test suite
  （30 项全过）；创新定位：**Feedback-Granularity-Aware Adaptive Evidence Acquisition**。
- **MVS-B0（依据 adcice/004.md）**：**sparse tuple-state backend**
  （状态 x=(z_1..z_N)、Ω 现场计算、动作/PMF on-demand、memo key (x,h)、
  279^8 不建全表）；N=4 与旧 eager 表等价认证（B0-G0）；N=8/R={1,2,4,8}
  在线规划（B0-G1）；**header b_h 激活 cross-level 相变**（b_h=0 probe →
  b_h≥4 direct jump；B0-G2）；协议公平 baseline（含 **Direct-8 Ordered**）
  与 break-even 理论 q<(r''−r')/(b_h+r''−r')（B0-G3/G4）。

## 目录结构

```
Exp-1/
├── SystemModel.md          # 系统模型文档（只读参考）
├── adcice/001.md           # 交叉审计意见（R1 依据）
├── run_mvsa.py             # v0 流水线（diagnostic，G2/G3/G4 审计后 REOPEN）
├── run_mvsa_r1.py          # R1 流水线：目标一致的有约束策略审计
├── run_mvsa_r11.py         # R1.1+R2 流水线：CMDP LP oracle + RBL
├── run_mvsa_r21.py         # R2.1 流水线：column generation 证书 + receding RBL + online solver
├── run_mvsb0.py            # MVS-B0 流水线：sparse backend + header 相变 + 协议公平 baseline
├── run_mvsb01.py           # MVS-B0.1 流水线：可信度修复 + VoI 定理 + Adaptive Direct-8
├── run_mvsb01a.py          # MVS-B0.1a 补丁：seed-aware POTS + ΔQ 相变 + 保守 lattice
├── run_mvsb03.py           # B0.3 CR-RBL：认证 rollout 规划器（主交付）
├── run_mvsb03a.py          # B0.3a credibility patch：paired CRN + STOP 证书 + 硬预算 + b⋆(x) 相变
├── test_regressions.py     # 正式 regression test suite（30 项数学不变量）
├── smoke_test.py           # 核心模块快速冒烟测试
├── requirements.txt
├── opmvs/                  # MVS-A 实现包
│   ├── model.py            # Gaussian 检测模型（LLR 分布/采样/解析 ROC）
│   ├── quantizer.py        # nested 二分树量化器 + message PMF/LLR
│   ├── state.py            # 证据状态编码与状态空间（cross_level 选项）
│   ├── fusion.py           # log-domain 工具（softplus/logsumexp）
│   ├── dp.py               # Exact DAG-DP + Bellman residual 审计
│   ├── opef.py             # O-PEF-1 / O-PEF-2E / O-PEF-3
│   ├── rbl.py              # R2: resource-bounded lookahead + (idx,h) 精确传播 + OnlinePlanner
│   ├── cmdp.py             # R2.1: CMDP column generation（LP master + ExactDP pricing）
│   ├── sparse.py           # MVS-B0: sparse tuple-state planner（279^8 不建表）
│   ├── rbl_cr.py           # B0.3/B0.3a: CR-RBL（LatentWorld paired CRN + STOP 证书 + exact_qa_pi_b）
│   ├── eval_exact.py       # R1: 精确前向概率传播 + G1a/G1b + 精确 P_D,max
│   ├── baselines.py        # B0–B11 + 公平基线精确 table-policy 构建
│   ├── mc.py               # 向量化 Monte Carlo + 随机化 Neyman-Pearson 评估
│   └── gates.py            # G0/G1/G2 Gate 检查
└── report/
    ├── MVS-A_report.md     # v0 诊断报告（G2/G3/G4 FAIL，审计后 REOPEN）
    ├── MVS-A-R1_report.md  # R1 目标一致审计报告
    ├── MVS-A-R1.1_R2_report.md  # R1.1+R2 CMDP oracle 与 RBL 报告
    ├── MVS-A-R2.1_report.md     # R2.1 认证 CMDP + receding online RBL 报告
    ├── MVS-B0_report.md        # MVS-B0 sparse-state header 报告
    ├── MVS-B0.1_report.md      # MVS-B0.1 可信度修复 + 理论拔高报告
    ├── MVS-B0.1a_report.md     # B0.1a 补丁报告
    ├── MVS-B0.3_report.md      # B0.3 CR-RBL 认证 rollout 报告（主交付）
    ├── MVS-B0.3a_report.md     # B0.3a credibility patch 报告（paired CRN/STOP 证书/硬预算/b⋆ 相变）
    └── figures/            # 量化器 / Pareto / 精度审计 / R1 / R2 / R2.1 图
```

## 运行

```bash
pip install -r requirements.txt

python run_mvsa.py           # v0 流水线（约 5–8 分钟）
python run_mvsa_r1.py        # R1 流水线（约 7–10 分钟）
python run_mvsa_r11.py       # R1.1+R2 流水线（约 10 分钟）
python run_mvsa_r21.py       # R2.1 流水线（约 10 分钟）
python run_mvsb0.py          # MVS-B0 流水线（约 6–15 分钟）
python run_mvsb01.py         # MVS-B0.1 流水线（约 20–30 分钟）
python run_mvsb01a.py        # MVS-B0.1a 补丁（约 1 分钟）
python run_mvsb03.py         # B0.3 CR-RBL（约 6–10 分钟，推荐）
python run_mvsb03a.py        # B0.3a/B0.3c credibility + closure patch（约 6–10 分钟，推荐）
python test_regressions.py   # regression suite（约 5–6 分钟，32 项：30 不变量/审计 + T17a 拆分 + T21）
python run_mvsa.py --smoke   # 快速冒烟
python run_mvsa_r1.py --smoke
python run_mvsa_r11.py --smoke
python run_mvsa_r21.py --smoke
python smoke_test.py         # 核心模块自检
```

## 关键设计（对应 SystemModel 章节）

| 组件 | 实现要点 | 章节 |
| --- | --- | --- |
| 局部 LLR | `L_i = a_i X_i − a_i²/2`，`L_i\|H_h ~ N(±a²/2, a²)` | §30 |
| 量化器 | 对 mixture `½p(L\|H0)+½p(L\|H1)` 做条件中位数递归二分；外层 cell 允许 ±∞（无截断） | §14 |
| Message PMF | 内层 cell 用线性 `Φ(hi)−Φ(lo)` 后取 log（不能用 logcdf 之差！） | §8 |
| Message-LLR | `ℓ_i^(r)(m) = log P(M=m\|H1) − log P(M=m\|H0)` | §8 |
| 渐进更新 | `Λ' = Λ − ℓ_i^(r)(m) + ℓ_i^(r')(m')`（replace，绝不 add） | §9 |
| 状态 | `z_i ∈ {∅}∪{(r,m)}`，混合进制编码，23^4 状态 | §10, §20 |
| 成本 | MVS-A 纯 payload：`c_a = r' − r` | §16.1 |
| DP | 按未揭示比特层做 memoized backward recursion；Bellman residual 审计 | §18, §21, §35 |
| 终端风险 | `C_01 = μ_M/π_1`，`C_10 = μ_F/π_0`，`R_stop = min(C_01 p, C_10(1−p))` | §17 |
| O-PEF-k | depth-k lookahead：`Q^(k) = c + E[V^(k−1)]`，`V^(k) = min{R_stop, min Q^(k)}` | §23–§27 |
| 双乘子 | R1: `μ_M=s, μ_F=s·e^η`；自然判决 = `Ω > η`；`J = E[B]+μ_M P_M+μ_F P_FA` | §17, §19 |
| 动作族 | R1 主版 adjacent-only `0→1,1→2,2→4`（cross-level 在 b_h=0 时被弱支配） | §25 修正 |
| 评估 | v0: 共享 (H,L) MC + 随机化 NP 阈值；R1: **精确前向概率传播**（table policy 无 MC） | §57–§58 |

## 数值正确性要点

1. **PMF 计算**：cell 概率 = `Φ(hi) − Φ(lo)`（线性域）再取 log；`logcdf(hi)−logcdf(lo) = log(a/b)` 是错误写法。
2. **随机化 NP 评估**：决策统计量 Ω 离散时，用 `P(Ω>η) ≤ α ≤ P(Ω≥η)` + 边界随机化，保证 P_FA ≡ 0.05。
3. **log-domain**：`log σ(Ω) = −softplus(−Ω)`，`log(1−σ) = −softplus(Ω)`，混合权重用 logsumexp。
4. **精确传播**：前向概率质量沿 policy DAG 传播（H0/H1 分别），STOP 时按自然判决或 NP 阈值累计；
   父状态质量必须清零（否则 mass 不守恒）。

## 结果

- **v0**（`report/MVS-A_report.md`，diagnostic）：G0/G1 PASS；G2/G3/G4 FAIL——
  经 adcice/001.md 审计判定为指标口径问题（REOPEN）。
- **R1**（`report/MVS-A-R1_report.md`）：G0/G1a/G1b/G2 全部 PASS（J(OPEF) ≥ J(DP) 精确成立；
  J(π_DP)(x0)=V*(x0) 达 1e-13）；v0 的 μ→∞ ceiling 确认为 Bayes/NP criterion mismatch 症状
  （DP 在 s=4096 时 P_D@P_FA=0.05 ≈ 0.847 ≈ P_D,max）；2×2 实验证实 cross-level 对有限深度
  lookahead 的偏置；adjacent-only 是消除 confounder 的规范化而非性能增强。
- **R1.1+R2**（`report/MVS-A-R1.1_R2_report.md`）：
  - R1.1：1bit_POTS 冻结 1-bit 排序修复；exact_np_roc 按假设归一化；
    **B_DP^CMDP**（constrained policy-mixture LP）冻结为 R2 oracle；
  - R2：resource-bounded lookahead，horizon=未来 payload bits；
    **V_16(x)=V*(x) 机器精度**（硬认证）。
- **R2.1**（`report/MVS-A-R2.1_report.md`，MVS-A 封板）：
  - **CMDP column generation**（LP master + ExactDP pricing）认证全局最优
    **B_CMDP\***（比 RMP 网格解更优——pricing 发现了网格外的策略列）；
  - **receding RBL-RH** 与 hard-budget（RB-HardBudget）分离；receding 在
    **H=6**（≪16）即 QoS 达标且 E[B] ≈ 5.2 bits < B9（7.57）——scalability 成立；
  - **online sparse planner**（memoized Solve(x,h)，不建全表）与 eager 表
    动作/值全等价（0 不一致）；部署时 root 求解稀疏率 ~3%；
  - η_rec（重定义，用 B_CMDP\* 与 receding H<16 的 B_RBL）达标硬 Gate ≥50%。

## 下一步（依据 advice/008.md 的新顺序，MVS-A 封板）

MVS-A 的 G0/G1a/G1b/G2 + R2.1-G0..G4 全部通过，按 003.md 冻结，不再继续优化。
MVS-B0/B0.1/B0.1a/B0.3/B0.3a/B0.3c 已按 004/005/006/007/008 完成。审计 008 的新顺序
（B0.5 移到 B0.6 之后，避免"理论漂亮但通信收益未定"）：

- **B0.4**：**paired-difference time-uniform EB-CS**——每次选一对 (a_t, b_t)，在共享
  world W_t 上得 Z_t^{a,b}=G_a(W_t)−G_b(W_t)，对 Δ_{a,b}=Q_a^{π_b}−Q_b^{π_b} 直接建
  EB 置信序列（U_{â,b}≤ε ∀b ⟹ Q_â≤min_b Q_b+ε；STOP 已知，只需对 G_a−R_stop 建单边 CS）；
  采样结构回到 candidate–challenger pair（2 rollouts/world，替代当前 32/world 的全配对）；
- **B0.4a**：uncertified ⇒ **fallback to base policy**；certified policy improvement
  （U(D_a)<0 才 override，D_a=Q_a^{π_b}−Q_{a_b}^{π_b}）；one-step conditional-VoI base
  a_b(x)=argmin_a[c_a+E[R_stop(X')|x,a]] 替换 SNR-base 做 ablation；
- **B0.4b**：正式 b⋆(x) 定理（B0.3c 已验证存在性判据：b⋆<∞ ⟺ E[Y_x]≥0；
  E[Y_x]<0 ⟹ progressive dominates direct for every b_h≥0）；
- **B0.6-pre**：N=4 exact / N=8 shallow sample-complexity gate（先验收算法）；
- **B0.6**：matched QoS + CI，CR vs Direct8/POTS（**论文生死 Gate**：P_FA^CR≤α、
  P_MD^CR≤β、E[B]^CR<E[B]^D8 with CI）——若 E[B]^CR≥E[B]^Direct8，论文诚实结论为
  "phase transition 存在、但当前 regime 最优策略落在 direct-packetization regime"；
- **B0.5**：**Bellman sandwich** L_k≤V⋆≤U_k（修正 006 不等式为
  V_genie≤V⋆≤min{V^{π_b}, R_stop}），把证书从 Q^{π_b} 提升到 Q⋆——只在 B0.6 显示
  算法有实际通信价值后做；
- **B1**（fading/packet errors）仍然最后。

关键算术修正（003.md §8）：MVS-B 每 UAV evidence states =
**1+2+4+16+256 = 279**（不是 47）；279⁸ ≈ 1e19 ⇒ **MVS-B 禁止复用全枚举 StateSpace**，
必须使用 R2.1-G3 的 sparse online planner。MVS-B 最重要的实验是
sensing/U2U **anti-correlation** regime（强 sensing ≠ 低通信成本）。
