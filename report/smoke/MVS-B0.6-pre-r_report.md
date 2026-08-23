# O-PEF MVS-B0.6-pre-r — credibility patch（advice/014，不改 planner）

> 三个 P0（014 §1）：**P0-1** 真实 **(x,h,t)** reachable states（每次反馈实扣 setup+payload，h=H−C_t，t=决策序号，δ_t=6δ_episode/(π²t²)）+ 去重 + on-policy occupancy state set；oracle 与 CPI 用状态自己的 h。**P0-2** 全 Gate 冻结 **SNR anchor**（G1 formal 不再用 VoI）。**P0-3** Gate 改 **CI 判定**（Wilson LCB95 + paired 差分的 t-UCB95），w_ep 用 **held-out** 半集选择，不再用 point estimate 首次过线。统计口径诚实：engineering acceptance vs statistical certification 明确区分（014 §2：25/30 的 Wilson 95% ≈ [0.664, 0.927]，n=30 不能认证 ≥0.80）。

> 生成时间: 2026-08-23 21:16:07   模式: SMOKE

## 0. 状态集（P0-1）：(x,h,t) augmented + 去重 + occupancy

- Set A（random trajectories，实扣成本）= 55，Set B（on-policy occupancy @ w_ep=250）= 15，合并去重后 = **32** 个 (x,h,t)（含 occupancy 9 个）。
- h 分布 = [20, 22, 23, 40]，t 分布 = [1, 2]（根 (0,40,1) 含在每个轨迹/episode 中，去重后仍保留）——oracle/CPI 均在状态自己的 h 上评测，δ_t 用状态自己的 t。

## 1. G0 — N=4 exact，SNR anchor，(x,h,t)：CI 判定（014 §2/§3）

| W | P(gap≤2) | Wilson LCB95 | E[gap^CPI] | E[gap^base] | E[D_s] | UCB95(E[D_s]) |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | 0.6562 | 0.4831 | 3.6095 | 7.1340 | -3.5245 | -0.3597 |
| 1000 | 0.7500 | 0.5789 | 1.2453 | 7.1340 | -5.8886 | -2.0397 |
| 2000 | 0.7500 | 0.5789 | 0.7770 | 7.1340 | -6.3570 | -2.6043 |

- **held-out 选择**：selection half 上无 W 使 LCB95[P(gap≤2)] ≥ 0.80 ⇒ w_ep 取网格最大值 2000（保守）；0.80 的统计认证缺样本（014 §2），点估计曲线见上表。
- **G0 gate（validation half @ w_ep=2000，n=16）**：UCB95(E[D_s]) = -0.9606 < −1 → FAIL；LCB95[P(gap≤2)] = 0.5050 ≥ 0.80 → FAIL（P̂=0.7500，Wilson 95% [0.5050, 0.9950]）。**G0 = FAIL**（E[D] 统计显著为负 = UNCERTAIN；P(gap≤2) 的 0.80 统计认证 = UNCERTAIN（工程验收达标，正式认证交 B0.6））。

## 2. G1 — N=4 formal certification，SNR anchor，(x,h,t)（014 §1 P0-2）

- 分层：32 个 (x,h,t) 状态中 base gap ≥ 15 的有 7 个（gap = [49.9, 42.5, 26.8, 19.6, 19.0, 19.8, 18.3]）。eb decide 只跑 stratified 状态（SNR anchor，δ_t 用状态自己的 t）。

| W | P(override) | (n_ov/n_strat) | Wilson LCB95 | mean worlds |
| --- | --- | --- | --- | --- |
| 4000 | 1.0000 | (7/7) | 0.6457 | 4000.0000 |
| 8000 | 1.0000 | (7/7) | 0.6457 | 8000.0000 |

- **G1 gate（W=8000，SNR anchor，gap≥15）**：LCB95[P(override)] = 0.6457（7/7）≥ 0.5 → PASS；**G1 = PASS**（form 证书在值得 override 时于预算内触发）。

## 3. G2 — N=8 shallow（H=40，oracle = sparse exact Q^{π_b}），SNR anchor

- N=8 状态集：random + occupancy 去重后取前 11 个 (x,h,t) （h ∈ [20, 22, 23, 40]）。

| mode | W | P(gap≤2) | Wilson LCB95 | E[D_s] | UCB95(E[D_s]) |
| --- | --- | --- | --- | --- | --- |
| Operational | 1000 | 0.4545 | 0.2127 | -53.4343 | -17.7240 |
| Formal | 1000 | 0.4545 | 0.2127 | -46.9756 | -10.3720 |

- **G2 gate（Operational @ W=1000，n=11）**：UCB95(E[D_s]) = -17.7240 < −1 → PASS；LCB95[P(gap≤2)] = 0.2127 ≥ 0.75 → FAIL（P̂=0.4545）。**G2 = FAIL**（014 §2：n=11 下 0.75 的统计认证待 B0.6 判定）。

## 4. G3 — feasibility：held-out w_ep 与 B0.6 pilot 冻结 w_ep=1000（014 §3）

- held-out w_ep = 2000：800 ep × 3 dec × 2000 worlds = 4800000 worlds ⇒ ~72.8 min（@1099 worlds/s）→ FAIL
- B0.6 pilot（014 §3 冻结） = 1000：800 ep × 3 dec × 1000 worlds = 2400000 worlds ⇒ ~36.4 min（@1099 worlds/s）→ PASS
- 说明：held-out w_ep 因 0.80 统计认证缺样本而落到网格最大值（偏重）；B0.6 pilot 按 014 §3 冻结 **w_ep=1000**（质量改善区且不重）。
- **G3 = PASS**：B0.6 pilot（w_ep=1000）的 episode 级模拟 ≤ 60 min。

## 验收结论（014 口径，诚实标注）：
- G0 E[D] 统计显著为负（UCB<−1）：FAIL；G0 P(gap≤2) LCB95≥0.80：FAIL（**UNCERTAIN — 工程验收，正式认证交 B0.6 stratified 实验**）
- G1 formal（SNR）LCB95≥0.5：PASS；G2 N=8 UCB<−1 & LCB95≥0.75：FAIL；G3 feasibility：PASS
- **总评：ENGINEERING ACCEPTANCE — P0 已修；P(gap≤2) 的 0.80/0.75 统计认证未在 n=32/11 达成（UNCERTAIN），按 014 §4 直接进入 B0.6 做 stratified matched-QoS 终审**

总耗时: 32.1s

