# O-PEF MVS-A — 最小可实现系统

依据 [SystemModel.md](./SystemModel.md)（O-PEF v2.1，§29–§36 定义 MVS-A 最小数学验证系统）
搭建的**可运行、可复现**的最小实现。当前版本完整实现并验证 **MVS-A**：

- 单目标二元检测，`N = 4` 架 UAV，Gaussian 局部检测器（§30–§32）；
- 每 UAV 一个 **nested 多精度量化器**（0/1/2/4 bit，§7、§14）；
- **log-domain 软融合**与 message-LLR，粗→细证据的 *replace-not-add* 更新（§8、§9、§11）；
- 证据状态 `z = (z_1,…,z_N)` 作为有限充分 Markov 状态（§10）；
- **Exact DAG-DP**（23^4 = 279,841 状态，memoized backward recursion，非 value iteration，§18、§20、§21）；
- **O-PEF-1**（depth-1 消融，§23）、**O-PEF-2E**（depth-2 精确期望主算法，§24、§27）
  与 **O-PEF-3**（depth-3 诊断性 solver 改进，§36/§66 "Gap>20% 时先优化 solver" 的
  直接体现——加深 lookahead 截断以逼近 Exact DP）；
- 基准：Raw Fusion / All-Neighbor / Random-K / SNR Top-K / Censoring / OTS-F / P-OTS /
  Global Fixed Progressive / Static Cost-Aware Progressive（§49–§53）；
- Gate **G0 / G1 / G2** 自动检查 + 核心实验一/二/三 + 精度审计（§34–§36、§58–§61）。

## 目录结构

```
Exp-1/
├── SystemModel.md          # 系统模型文档（只读参考）
├── run_mvsa.py             # 主流水线：构建 → G0 → 求解 → MC → 报告
├── smoke_test.py           # 核心模块快速冒烟测试
├── requirements.txt
├── opmvs/                  # MVS-A 实现包
│   ├── model.py            # Gaussian 检测模型（LLR 分布/采样/解析 ROC）
│   ├── quantizer.py        # nested 二分树量化器 + message PMF/LLR
│   ├── state.py            # 证据状态编码与状态空间
│   ├── fusion.py           # log-domain 工具（softplus/logsumexp）
│   ├── dp.py               # Exact DAG-DP + Bellman residual 审计
│   ├── opef.py             # O-PEF-1 / O-PEF-2E
│   ├── baselines.py        # B0–B11 基准算法
│   ├── mc.py               # 向量化 Monte Carlo + 随机化 Neyman-Pearson 评估
│   └── gates.py            # G0/G1/G2 Gate 检查
└── report/
    ├── MVS-A_report.md     # 生成的实验报告
    └── figures/            # 量化器 / Pareto / 精度审计图
```

## 运行

```bash
pip install -r requirements.txt

python run_mvsa.py          # 完整流水线（约 5–8 分钟，输出 report/MVS-A_report.md）
python run_mvsa.py --smoke  # 快速冒烟（小样本、少 μ 点）
python smoke_test.py        # 核心模块自检
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
| 评估 | 共享 (H,L) 样本；**随机化 Neyman-Pearson 阈值**把每个方法都校准到 P_FA=0.05 | §57–§58 |

## 数值正确性要点

1. **PMF 计算**：cell 概率 = `Φ(hi) − Φ(lo)`（线性域）再取 log；`logcdf(hi)−logcdf(lo) = log(a/b)` 是错误写法。
2. **随机化 NP 评估**：决策统计量 Ω 离散时，用 `P(Ω>η) ≤ α ≤ P(Ω≥η)` + 边界随机化，保证 P_FA ≡ 0.05。
3. **log-domain**：`log σ(Ω) = −softplus(−Ω)`，`log(1−σ) = −softplus(Ω)`，混合权重用 logsumexp。

## 结果

运行后查看 `report/MVS-A_report.md`。预期（与文档一致）：

- 单 UAV P_D @ P_FA=0.05 ≈ [0.226, 0.301, 0.408, 0.553]；连续全融合 P_D,raw ≈ 0.851；
- Exact DP 的 Bellman residual 为 0（双精度）；
- Gate G0/G1 通过；G2/G3/G4 依据实际 sweep 结果如实报告（含相邻策略随机混合插值，§19）。

## 下一步（MVS-B）

MVS-B（N=8、0/1/2/4/8 bit、Rician U2U 信道、header 16 bit、packet loss，
§37–§42）在本包架构上扩展：`StateSpace` 的 `BASE=47`（1+2+4+16+256），
`baselines.py` 增加 `c_a^radio = b_h + r'−r` 与 ARQ-collapsed 成本；Gate G5 在 MVS-B 上执行。
