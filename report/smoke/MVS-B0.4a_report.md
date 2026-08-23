# O-PEF MVS-B0.4a — Certified Policy Improvement (base by default)

> 依据 `advice/011.md` §3-§8。核心：**Base by default; override only with certified evidence of improvement.** a_inc^(0) = a_b（one-step conditional-VoI base），challenger c 只有在 U_{c,a_inc} < 0（pairwise CS 证明 Q_c^{π_b} < Q_{a_inc}^{π_b}）时才替换 incumbent；episode 级 δ：决策 t 花 δ_t = 6δ_episode/(π²t²)，决策内每 pair α = δ_t/P ⇒ P(所有 override 有效) ≥ 1−δ_episode。
> 与 B0.4 全动作 ε-optimal 证书的区别：不再证明『该动作接近所有动作最优』，只需证明『该动作不比我会执行的 base 差』——base-anchored，O(|A|) 而非 O(|A|²) 的置信预算（011 §3/§9）。

> 生成时间: 2026-08-23 16:24:48   模式: SMOKE

## 1. G0 — fallback tail：empirical-best vs SNR-base vs VoI-base（011 §8-1）

| fallback | E[gap] | P(gap>2) | P(gap>4) |
| --- | --- | --- | --- |
| emp-best | 1.2446 | 0.2333 | 0.1667 |
| snr-base | 3.9316 | 0.2000 | 0.1667 |
| voi-base | 1.6958 | 0.2333 | 0.1000 |
- 解读：VoI-base 是否消除 uncertified tail（P(gap>2) 相比 empirical-best 是否下降）；empirical-best 是 B0.4 未认证分支的执行行为（011 指出应结束）。

## 2. G1/G2 — override 收益与 certified override 安全性（011 §8-2/§8-3）

- betting-mode CPI（1500 worlds/decision，δ_episode=0.05，决策 t 用 δ_t=6δ/(π²t²)）：
  - **P(override) = 0.1000**；**E[Q_{a_b} − Q_{a_override} | override] = 20.6330**
  - **certified override 安全性**（N=4 exact，matched base）：safe=3，violations=0；单侧 95% binomial U95 = 0.6316（certified override 只在 U<0 时执行——理论 P(violation) ≤ Σδ_t）
  - 执行质量：base E[gap]=5.1890（P(>2)=0.4000）→ CPI E[gap]=3.1257（P(>2)=0.3000）
  - formal PrPl-EB 路径（2000 worlds，n=6）：safe override rate = 0.0000——formal 证书保守，override 需更大预算（011 §9 预期：override 比 full best-arm 容易，但 EB 的 peeling 开销仍在）。

## 3. G3 — VoI-base 强度：VoI-base vs SNR-base vs CPI vs empirical-best（011 §8-4）

- 综合：若 VoI-base ≈ CPI，论文应强调 feedback granularity + conditional VoI；若 CPI 在 VoI-base 之上有明确增益，pairwise certified planning 才有独立算法价值。（G0 的 voi-base 行 = VoI-base 本身；G1 的 CPI 行 = certified improvement 之上；两者之差即 pairwise planner 的边际价值。）

总耗时: 179.2s
