# O-PEF MVS-B0.4a — Certified Policy Improvement (base by default)

> 依据 `advice/011.md` §3-§8。核心：**Base by default; override only with certified evidence of improvement.** a_inc^(0) = a_b（one-step conditional-VoI base），challenger c 只有在 U_{c,a_inc} < 0（pairwise CS 证明 Q_c^{π_b} < Q_{a_inc}^{π_b}）时才替换 incumbent；episode 级 δ：决策 t 花 δ_t = 6δ_episode/(π²t²)，决策内每 pair α = δ_t/P ⇒ P(所有 override 有效) ≥ 1−δ_episode。
> 与 B0.4 全动作 ε-optimal 证书的区别：不再证明『该动作接近所有动作最优』，只需证明『该动作不比我会执行的 base 差』——base-anchored，O(|A|) 而非 O(|A|²) 的置信预算（011 §3/§9）。

> 生成时间: 2026-08-23 17:15:30   模式: SMOKE

## 1. G0 — fallback tail：empirical-best vs SNR-base vs VoI-base（011 §8-1）

| fallback | E[gap] | P(gap>2) | P(gap>4) |
| --- | --- | --- | --- |
| emp-best | 1.2446 | 0.2333 | 0.1667 |
| snr-base | 3.9316 | 0.2000 | 0.1667 |
| voi-base | 1.6958 | 0.2333 | 0.1000 |
- 解读：VoI-base 是否消除 uncertified tail（P(gap>2) 相比 empirical-best 是否下降）；empirical-best 是 B0.4 未认证分支的执行行为（011 指出应结束）。

## 2. G1/G2 — 2×2 CPI matrix（base × mode）与 override 收益/安全（011 §8-2/§8-3）

- **Formal-CPI**（cs_mode=eb，theorem-backed PrPl-EB）承担 safety claim；**Operational-CPI**（cs_mode=betting，finite-grid 实验 CS）只承担性能探索，不承担严格置信保证（012 §1）。独立状态都是 episode 的第一个决策，统一用 δ_1 = 6δ_episode/π²（012 §3 修复 stale δ_t bug）。

- **Operational-CPI@VoI**（1500 worlds，δ₁=0.03）：P(override)=0.2000，E[gain|override]=18.0064，base E[gap]=5.1890（P(>2)=0.4000）→ CPI E[gap]=1.5877（P(>2)=0.3000）；overrides=6 safe / 0 violations，U95=0.3930
- **Formal-CPI@VoI**（2000 worlds，δ₁=0.03）：P(override)=0.0000，E[gain|override]=0.0000，base E[gap]=3.8482（P(>2)=0.1667）→ CPI E[gap]=3.8482（P(>2)=0.1667）；overrides=0 safe / 0 violations，U95=1.0000
- **Operational-CPI@SNR**（1500 worlds，δ₁=0.03）：P(override)=0.1333，E[gain|override]=22.2182，base E[gap]=4.1204（P(>2)=0.3333）→ CPI E[gap]=1.1580（P(>2)=0.2333）；overrides=4 safe / 0 violations，U95=0.5271
- **Formal-CPI@SNR**（2000 worlds，δ₁=0.03）：P(override)=0.0000，E[gain|override]=0.0000，base E[gap]=2.1745（P(>2)=0.1667）→ CPI E[gap]=2.1745（P(>2)=0.1667）；overrides=0 safe / 0 violations，U95=1.0000
- **G2 统计口径（012 §2）**：0 violation 时需 **≥59 个 overrides** 才能让单侧 95% binomial 上界 ≤ 0.05；当前 override 数不足以经验认证 5% violation rate——表述为 **0 observed violations, but insufficient override count for a 5% violation-rate certification（UNCERTAIN，非 FAIL）**；理论的 P(violation) ≤ Σδ_t 是 Formal-CPI 的保证，经验 U95 是 sanity check。

## 3. t-scan — episode 内决策序号 t 对 override rate 的影响（012 §3）

| t | δ_t = 6δ/(π²t²) | P(override) |
| --- | --- | --- |
| 1 | 0.0304 | 0.1333 |
| 2 | 0.0076 | 0.0667 |
| 4 | 0.0019 | 0.2000 |
| 8 | 0.0005 | 0.0667 |
- 真正的 episode-level δ_t 需在实际 receding episode trajectory 中测试（012 §3；此处是独立状态 × 决策序号 t 的近似）。

## 4. G3 — VoI-base 强度：VoI-base vs SNR-base vs CPI vs empirical-best（011 §8-4）

- 综合：若 VoI-base ≈ CPI，论文应强调 feedback granularity + conditional VoI；若 CPI 在 VoI-base 之上有明确增益，pairwise certified planning 才有独立算法价值。（G0 的 voi-base/snr-base 行 = 各 base 本身；G1 的 CPI 行 = certified improvement 之上；两者之差即 pairwise planner 的边际价值。）
- 012 §6 结论：VoI-base 目前是 **better theoretical anchor 而非 better performance anchor**（G0 显示 SNR-base 的 P(gap>2) 更低）——最终算法可用 VoI 作 candidate ranking、SNR 作 safe anchor，或反之，需 B0.6 前的 ablation 决定。

总耗时: 267.8s
