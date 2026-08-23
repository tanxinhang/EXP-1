# O-PEF MVS-B0.6-pre — sample-complexity gate（N=4 exact / N=8 shallow）

> 依据 `advice/008.md` §14 与 `advice/013.md` 最终指示：进入 B0.6 前**先验收算法**。验收对象 = B0.4a/B0.4a-r 的 **CPI acquisition**（B0.6 每步决策实际部署的算法）：base-anchored O(|A|) 置信分配（012 §4 方案 A），cs_mode=betting 为 Operational-CPI（性能），cs_mode=eb 为 Formal-CPI（safety claim，012 §1）。G0 同时完成 012 §6 的 base ablation（VoI vs SNR 谁做 anchor）。

> 生成时间: 2026-08-23 20:42:40   模式: SMOKE

## 1. G0 — N=4 exact：P(gap≤2) 与 E[gap] vs 世界预算 W（base ablation，012 §6）

- 口径：random reachable N=4 states（n=10），H=40，δ₁=0.0304；oracle = **matched-base** exact Q^{π_b}；gap = Q^{π_b}(a) − min{R_stop, Q^{π_b}}；base 行 = a_b 本身，CPI 行 = CPI.decide(betting)。

| base | W | P(gap≤2) | E[gap] | P(override) | E[gain\|override] | mean worlds |
| --- | --- | --- | --- | --- | --- | --- |
| VoI | 250 | 0.7000 | 3.0107 (base 3.0107) | 0.0000 | 0.0000 | 250.0000 |
| VoI | 500 | 0.7000 | 3.0107 (base 3.0107) | 0.0000 | 0.0000 | 500.0000 |
| VoI | 1000 | 0.7000 | 1.9177 (base 3.0107) | 0.1000 | 10.9292 | 1000.0000 |
| VoI | 2000 | 0.8000 | 0.9341 (base 3.0107) | 0.1000 | 20.7656 | 2000.0000 |
| SNR | 250 | 0.9000 | 0.3654 (base 5.5799) | 0.1000 | 52.1445 | 250.0000 |
| SNR | 500 | 0.8000 | 0.8634 (base 5.5799) | 0.1000 | 47.1646 | 500.0000 |
| SNR | 1000 | 0.9000 | 0.3654 (base 5.5799) | 0.1000 | 52.1445 | 973.3000 |
| SNR | 2000 | 0.8000 | 0.7895 (base 5.5799) | 0.1000 | 47.9035 | 1845.9000 |
（G0 总耗时 66s；CPI-only 吞吐 1162 worlds/s）

- **G0 gate（best base = SNR @ W=2000）**：P(gap≤2)=0.8000 ≥ 0.80 → PASS；CPI E[gap]=0.7895 ≤ base E[gap]−1.0（=4.5799）→ PASS；**G0 = PASS**。（012 §6：VoI 作 candidate ranking / SNR 作 safe anchor 的 ablation 数据已出。）

## 2. G1 — N=4 formal certification：eb-mode CPI 的 sample cost（012 §1）

- 口径：**form 证书只在证据强时触发**。MP+peeling 半径含 range-scaling 项 7t(hi−lo)/(3(n−1))（pair range ≈ 2·(h−c_a+R_max) ≈ 790，B0.3c budget-aware diameter），故 gap g 需 n ≳ 7t(hi−lo)/(3g) 量级的世界数才认证（与 B0.4r/008 『收紧 bound 只部分解锁证书』一致）。按 base 真 gap 分层——base 明显次优（Q^{{π_b}}(a_b) − Q_min ≥ 15）时 override 才值得；base 已最优时无 override ⇒ 执行 base（安全，011 §3）——所以 eb decide 只跑在 stratified 状态上。

| W | P(override) \| base gap≥15 | (n_ov/n_strat) | mean worlds (stratified) | mean rollouts |
| --- | --- | --- | --- | --- |
| 4000 | 0.0000 | (0/9) | 4000.0000 | 7960.0000 |
| 8000 | 1.0000 | (9/9) | 8000.0000 | 15960.0000 |

- 分层状态：48 个 random reachable 中 base gap ≥ 15 的 有 9 个（gap 列表 = [np.float64(21.9), np.float64(20.4), np.float64(17.2), np.float64(17.2), np.float64(20.8), np.float64(21.6), np.float64(21.5), np.float64(17.2), np.float64(17.2)]）。未分层状态（base 已最优）eb decide 跑到 cap 也无 override ⇒ 执行 base（安全）。
- **G1 gate（W=8000，base gap ≥ 15 的状态）**：P(override) = 1.0000（9/9）≥ 0.5 且 n_strat≥2 → PASS；**G1 = PASS**（form 证书在值得 override 时于预算内触发；base 已最优时无 override ⇒ 安全执行 base）。

## 3. G2 — N=8 shallow oracle（H=40，oracle = sparse exact Q^{π_b}）

| mode | W | P(gap≤2) | E[gap] (base) | P(override) |
| --- | --- | --- | --- | --- |
| Operational | 1000 | 1.0000 | 0.0000 (0.0000) | 0.0000 |
| Formal | 1000 | 1.0000 | 0.0000 (0.0000) | 0.0000 |

- **G2 gate（Operational @ W=1000）**：P(gap≤2)=1.0000 ≥ 0.75 → PASS；CPI E[gap]=0.0000 ≤ base E[gap]=0.0000 → PASS；**G2 = PASS**（N=8 shallow 验收）。

## 4. G3 — sample-complexity accounting + B0.6 可行性（先验收算法）

- G0 曲线（SNR base）显示 W=250 即达 P(gap≤2) ≥ 0.80 ⇒ B0.6 per-decision 世界预算 w_ep = 250。
- B0.6 规模：n_ep=800 episodes × K=3 decisions/episode × w_ep=250 worlds/decision = **600000 worlds**；CPI-only 实测吞吐 1162 worlds/s ⇒ 预计 **8.6 min**（不含 oracle，oracle 仅用于事后评测）。
- **G3 gate**：预计 wall time ≤ 60 min → PASS；**G3 = PASS**（B0.6 的 episode 级 matched-QoS 模拟可承受）。

## 验收结论：**PASS — 算法验收通过，进入 B0.6**（G0=PASS，G1=PASS，G2=PASS，G3=PASS）

总耗时: 241.0s

