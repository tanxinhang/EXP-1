# O-PEF MVS-A — 最小可实现系统（R1/R1.1/R2 演进）

依据 [SystemModel.md](./SystemModel.md)（O-PEF v2.1，§29–§36 定义 MVS-A 最小数学验证系统）
搭建的**可运行、可复现**的最小实现。当前版本完整实现并验证 **MVS-A**、**MVS-A-R1**
（目标一致纠偏）与 **MVS-A-R1.1+R2**（CMDP oracle + resource-bounded lookahead）：

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

## 目录结构

```
Exp-1/
├── SystemModel.md          # 系统模型文档（只读参考）
├── adcice/001.md           # 交叉审计意见（R1 依据）
├── run_mvsa.py             # v0 流水线（diagnostic，G2/G3/G4 审计后 REOPEN）
├── run_mvsa_r1.py          # R1 流水线：目标一致的有约束策略审计
├── run_mvsa_r11.py         # R1.1+R2 流水线：CMDP LP oracle + RBL
├── run_mvsa_r21.py         # R2.1 流水线：column generation 证书 + receding RBL + online solver（主交付）
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
│   ├── eval_exact.py       # R1: 精确前向概率传播 + G1a/G1b + 精确 P_D,max
│   ├── baselines.py        # B0–B11 + 公平基线精确 table-policy 构建
│   ├── mc.py               # 向量化 Monte Carlo + 随机化 Neyman-Pearson 评估
│   └── gates.py            # G0/G1/G2 Gate 检查
└── report/
    ├── MVS-A_report.md     # v0 诊断报告（G2/G3/G4 FAIL，审计后 REOPEN）
    ├── MVS-A-R1_report.md  # R1 目标一致审计报告
    ├── MVS-A-R1.1_R2_report.md  # R1.1+R2 CMDP oracle 与 RBL 报告
    ├── MVS-A-R2.1_report.md     # R2.1 认证 CMDP + receding online RBL 报告（主交付）
    └── figures/            # 量化器 / Pareto / 精度审计 / R1 / R2 / R2.1 图
```

## 运行

```bash
pip install -r requirements.txt

python run_mvsa.py           # v0 流水线（约 5–8 分钟）
python run_mvsa_r1.py        # R1 流水线（约 7–10 分钟）
python run_mvsa_r11.py       # R1.1+R2 流水线（约 10 分钟）
python run_mvsa_r21.py       # R2.1 流水线（约 10 分钟，推荐）
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

## 下一步（MVS-B0/B1/B2，MVS-A 封板）

MVS-A 的 G0/G1a/G1b/G2 + R2.1-G0..G4 全部通过，按 003.md 冻结，不再继续优化。
进入 MVS-B 分三步（每次只加一个新变量）：

- **MVS-B0**：`b_h = 16` + cross-level actions（验证 b_h>0 后跨级动作恢复价值，
  闭合 R1 的"cross-level 在 b_h=0 被弱支配"理论闭环）；
- **MVS-B1**：异构 ARQ-collapsed 成本 `c̄_a = (b_h + r'−r)/p_i^succ`（非整数成本 ⇒
  R2.1-G3 的稀疏递归 planner 需实数 budget 处理）；
- **MVS-B2**：显式 packet-loss（failure self-loop）与 collapsed 版本 A/B 对照。

关键算术修正（003.md §8）：MVS-B 每 UAV evidence states =
**1+2+4+16+256 = 279**（不是 47）；279⁸ ≈ 1e19 ⇒ **MVS-B 禁止复用全枚举 StateSpace**，
必须使用 R2.1-G3 的 sparse online planner。MVS-B 最重要的实验是
sensing/U2U **anti-correlation** regime（强 sensing ≠ 低通信成本）。
