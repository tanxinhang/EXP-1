# O-PEF MVS-B0.6-pre — sample-complexity gate（N=4 exact / N=8 shallow）

> 依据 `advice/008.md` §14 与 `advice/013.md` 最终指示：进入 B0.6 前**先验收算法**。验收对象 = B0.4a/B0.4a-r 的 **CPI acquisition**（B0.6 每步决策实际部署的算法）：base-anchored O(|A|) 置信分配（012 §4 方案 A），cs_mode=betting 为 Operational-CPI（性能），cs_mode=eb 为 Formal-CPI（safety claim，012 §1）。G0 同时完成 012 §6 的 base ablation（VoI vs SNR 谁做 anchor）。

> 生成时间: 2026-08-23 20:46:56   模式: FULL

## 1. G0 — N=4 exact：P(gap≤2) 与 E[gap] vs 世界预算 W（base ablation，012 §6）

- 口径：random reachable N=4 states（n=30），H=40，δ₁=0.0304；oracle = **matched-base** exact Q^{π_b}；gap = Q^{π_b}(a) − min{R_stop, Q^{π_b}}；base 行 = a_b 本身，CPI 行 = CPI.decide(betting)。

| base | W | P(gap≤2) | E[gap] | P(override) | E[gain\|override] | mean worlds |
| --- | --- | --- | --- | --- | --- | --- |
| VoI | 250 | 0.7667 | 2.0297 (base 2.7588) | 0.0333 | 21.8729 | 250.0000 |
| VoI | 500 | 0.7667 | 2.0297 (base 2.7588) | 0.0333 | 21.8729 | 500.0000 |
| VoI | 1000 | 0.8000 | 0.9478 (base 2.7588) | 0.1000 | 18.1092 | 1000.0000 |
| VoI | 2000 | 0.8000 | 0.7821 (base 2.7588) | 0.1000 | 19.7669 | 2000.0000 |
| SNR | 250 | 0.8000 | 1.9656 (base 3.7038) | 0.0333 | 52.1445 | 250.0000 |
| SNR | 500 | 0.7667 | 2.1316 (base 3.7038) | 0.0333 | 47.1646 | 500.0000 |
| SNR | 1000 | 0.8000 | 1.9656 (base 3.7038) | 0.0333 | 52.1445 | 991.1000 |
| SNR | 2000 | 0.8333 | 1.0533 (base 3.7038) | 0.1000 | 26.5049 | 1948.6333 |
（G0 总耗时 212s；CPI-only 吞吐 1099 worlds/s）

- **G0 gate（best base = SNR @ W=2000）**：P(gap≤2)=0.8333 ≥ 0.80 → PASS；CPI E[gap]=1.0533 ≤ base E[gap]−1.0（=2.7038）→ PASS；**G0 = PASS**。（012 §6：VoI 作 candidate ranking / SNR 作 safe anchor 的 ablation 数据已出。）

## 2. G1 — N=4 formal certification：eb-mode CPI 的 sample cost（012 §1）

- 口径：**form 证书只在证据强时触发**。MP+peeling 半径含 range-scaling 项 7t(hi−lo)/(3(n−1))（pair range ≈ 2·(h−c_a+R_max) ≈ 790，B0.3c budget-aware diameter），故 gap g 需 n ≳ 7t(hi−lo)/(3g) 量级的世界数才认证（与 B0.4r/008 『收紧 bound 只部分解锁证书』一致）。按 base 真 gap 分层——base 明显次优（Q^{{π_b}}(a_b) − Q_min ≥ 15）时 override 才值得；base 已最优时无 override ⇒ 执行 base（安全，011 §3）——所以 eb decide 只跑在 stratified 状态上。

| W | P(override) \| base gap≥15 | (n_ov/n_strat) | mean worlds (stratified) | mean rollouts |
| --- | --- | --- | --- | --- |
| 4000 | 0.0000 | (0/15) | 4000.0000 | 7960.0000 |
| 8000 | 0.8667 | (13/15) | 8000.0000 | 15960.0000 |

- 分层状态：120 个 random reachable 中 base gap ≥ 15 的 有 15 个（gap 列表 = [21.9, 20.4, 17.2, 17.2, 20.8, 21.6, 21.5, 17.2, 17.2, 21.9, 15.2, 21.9, 21.5, 15.9, 21.5]）。未分层状态（base 已最优）eb decide 跑到 cap 也无 override ⇒ 执行 base（安全）。
- **G1 gate（W=8000，base gap ≥ 15 的状态）**：P(override) = 0.8667（13/15）≥ 0.5 且 n_strat≥2 → PASS；**G1 = PASS**（form 证书在值得 override 时于预算内触发；base 已最优时无 override ⇒ 安全执行 base）。

## 3. G2 — N=8 shallow oracle（H=40，oracle = sparse exact Q^{π_b}）

| mode | W | P(gap≤2) | E[gap] (base) | P(override) |
| --- | --- | --- | --- | --- |
| Operational | 2000 | 1.0000 | 0.1851 (1.6915) | 0.1000 |
| Formal | 2000 | 0.9000 | 1.6915 (1.6915) | 0.0000 |

- **G2 gate（Operational @ W=2000）**：P(gap≤2)=1.0000 ≥ 0.75 → PASS；CPI E[gap]=0.1851 ≤ base E[gap]=1.6915 → PASS；**G2 = PASS**（N=8 shallow 验收）。

## 4. G3 — sample-complexity accounting + B0.6 可行性（先验收算法）

- G0 曲线（SNR base）显示 W=250 即达 P(gap≤2) ≥ 0.80 ⇒ B0.6 per-decision 世界预算 w_ep = 250。
- B0.6 规模：n_ep=800 episodes × K=3 decisions/episode × w_ep=250 worlds/decision = **600000 worlds**；CPI-only 实测吞吐 1099 worlds/s ⇒ 预计 **9.1 min**（不含 oracle，oracle 仅用于事后评测）。
- **G3 gate**：预计 wall time ≤ 60 min → PASS；**G3 = PASS**（B0.6 的 episode 级 matched-QoS 模拟可承受）。

## 验收结论：**PASS — 算法验收通过，进入 B0.6**（G0=PASS，G1=PASS，G2=PASS，G3=PASS）

总耗时: 452.5s

