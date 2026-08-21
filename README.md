# O-PEF MVS-A — 最小可实现系统（含 R1 目标一致纠偏）

依据 [SystemModel.md](./SystemModel.md)（O-PEF v2.1，§29–§36 定义 MVS-A 最小数学验证系统）
搭建的**可运行、可复现**的最小实现。当前版本完整实现并验证 **MVS-A** 与 **MVS-A-R1**：

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

## 目录结构

```
Exp-1/
├── SystemModel.md          # 系统模型文档（只读参考）
├── adcice/001.md           # 交叉审计意见（R1 依据）
├── run_mvsa.py             # v0 流水线（diagnostic，G2/G3/G4 审计后 REOPEN）
├── run_mvsa_r1.py          # R1 流水线：目标一致的有约束策略审计（主交付）
├── smoke_test.py           # 核心模块快速冒烟测试
├── requirements.txt
├── opmvs/                  # MVS-A 实现包
│   ├── model.py            # Gaussian 检测模型（LLR 分布/采样/解析 ROC）
│   ├── quantizer.py        # nested 二分树量化器 + message PMF/LLR
│   ├── state.py            # 证据状态编码与状态空间（cross_level 选项）
│   ├── fusion.py           # log-domain 工具（softplus/logsumexp）
│   ├── dp.py               # Exact DAG-DP + Bellman residual 审计
│   ├── opef.py             # O-PEF-1 / O-PEF-2E / O-PEF-3
│   ├── eval_exact.py       # R1: 精确前向概率传播 + G1a/G1b + 精确 P_D,max
│   ├── baselines.py        # B0–B11 + 公平基线精确 table-policy 构建
│   ├── mc.py               # 向量化 Monte Carlo + 随机化 Neyman-Pearson 评估
│   └── gates.py            # G0/G1/G2 Gate 检查
└── report/
    ├── MVS-A_report.md     # v0 诊断报告（G2/G3/G4 FAIL，审计后 REOPEN）
    ├── MVS-A-R1_report.md  # R1 目标一致审计报告（主交付）
    └── figures/            # 量化器 / Pareto / 精度审计 / R1 图
```

## 运行

```bash
pip install -r requirements.txt

python run_mvsa.py           # v0 流水线（约 5–8 分钟）
python run_mvsa_r1.py        # R1 流水线（约 10 分钟，推荐）
python run_mvsa.py --smoke   # 快速冒烟
python run_mvsa_r1.py --smoke
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
- **R1**（`report/MVS-A-R1_report.md`，主交付）：
  - G0/G1a/G1b/G2 全部 PASS（J(OPEF) ≥ J(DP) 精确成立；J(π_DP)(x0)=V*(x0) 达 1e-13）；
  - v0 的 μ→∞ ceiling 确认为 Bayes/NP criterion mismatch 症状：DP 在 s=4096 时
    P_D@P_FA=0.05 ≈ 0.847 ≈ P_D,max；
  - 2×2 实验证实 cross-level 对有限深度 lookahead 的偏置（OPEF-2 E[B] 8.93→2.32 bits）；
  - QoS-matched：Exact DP（≈6.0 bits）优于公平基线 B9（7.57）/ B11（9.16）/ 1-bit-POTS（9.03）；
  - OPEF-2E/3 仍受有限深度截断限制（过早停止），G3/G4 对 OPEF 未达标 → 下一步
    resource-bounded lookahead（按累计未来 bit 成本截断 horizon），再复验 G3/G4。

## 下一步（MVS-B）

前置条件（§66/§70）：R1 的 G0/G1a/G1b/G2 已过；G3/G4 对 OPEF 需先完成
resource-bounded lookahead 改进。MVS-B（N=8、0/1/2/4/8 bit、Rician U2U 信道、
header 16 bit、packet loss，§37–§42）在本包架构上扩展：`StateSpace` 的 `BASE=47`
（1+2+4+16+256）、`c_a^radio = b_h + r'−r` 与 ARQ-collapsed 成本；跨级动作在
b_h>0 后才有物理意义；Gate G5 在 MVS-B 上执行。
