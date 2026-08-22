# O-PEF MVS-B0.1a — Credibility Patch（advice/006.md §17）

> 修复 1bit-POTS seed 后立即停止、真前缀 CRN、CI 输出、unmatched 措辞、reachable-state ΔQ 相变实证；resource lattice 已在 sparse.py 保守化。
> 生成时间: 2026-08-22 21:17:04   模式: FULL

- 真前缀 CRN: baseline n=20000；adaptive 使用前 2500 条 → 公共子集可做 paired 对比

## 1. P0-1b — 1bit-POTS：seed 后立即停止 + seed-aware cumulative LLR

| ηs | P_D @ P_FA=0.05 | E[B_radio] | 达标 |
| --- | --- | --- | --- |
- P_D,max = 0.8940，matched 目标 P_D ≥ 0.8840
| 0.5 | 0.7680 | 139.0858 |  |
| 1.0 | 0.7806 | 154.5789 |  |
| 1.5 | 0.8144 | 178.6593 |  |
| 2.0 | 0.8793 | 210.0695 |  |
| 3.0 | 0.8905 | 277.0960 | ✓ |
| 4.0 | 0.8939 | 347.6982 | ✓ |
| 6.0 | 0.8940 | 477.2699 | ✓ |
- **修复后 1bit-POTS matched: E[B_radio] = 277.0960 bits**（seed 后立即停止显著降低通信成本）

## 2. 公平基线（前缀 CRN，Wilson 95% CI）

| 方法 | 参数 | P_D @ 0.05 (95% CI) | P_FA (95% CI) | E[B_radio] |
| --- | --- | --- | --- | --- |
| AllNeighbor-8 | - | 0.8940 [0.888,0.900] | 0.0500 [0.046,0.054] | 192.0000 |
| SNR-TopK | 8 | 0.8940 [0.888,0.900] | 0.0500 [0.046,0.054] | 192.0000 |
| 1bit-POTS | matched ηs | 0.8905 [0.883,0.895] | 0.0500 [0.044,0.053] | 277.0960 |
| Direct8-Ordered | matched ηs | 0.8857 [0.878,0.890] | 0.0500 [0.042,0.050] | 110.5140 |

- 说明：P_FA 的 CI 与 0.05 的关系决定 'QoS 认证' 或 'UNCERTIFIED'（不可据此断言 FAIL）

## 3. unmatched operating-point decomposition（原'同 E[B] gain'改称）

- 该对比只是 same-(s,η,H) operating-point 下的 cost/performance 差异，P_D 与 E[B] 均未匹配——**不构成 gain**；真正的 multi-resolution gain 必须在 P_D, P_FA matched 后比较 E[B]（保持 UNCERTIFIED）

## 4. reachable-state ΔQ sweep 与 critical feedback-granularity threshold

> ΔQ(x;b) = Q_prog − Q_dir = E[min{D(x')−Δ₂, b}]；b*=inf{b:E[min(D−Δ₂,b)]≥0}。

| b_setup | P(ΔQ<0) | P(ΔQ>0) | E[ΔQ] |
| --- | --- | --- | --- |
| 0 | 0.9613 | 0.0000 | -14.2641 |
| 4 | 0.9613 | 0.0387 | -12.6595 |
| 8 | 0.5587 | 0.4412 | -11.1047 |
| 16 | 0.3463 | 0.6538 | -8.1391 |
| 32 | 0.3137 | 0.6863 | -2.6406 |

- 根状态 g(b)=E[min(D−Δ₂,b)]：[(0, np.float64(-3.5)), (4, np.float64(-1.5)), (8, np.float64(0.5)), (16, np.float64(4.5)), (32, np.float64(12.5)), (64, np.float64(28.5)), (128, np.float64(38.559))]
- **临界阈值 b* ≈ 8**（g(b*)≥0 ⇒ direct packetization 开始占优）——feedback-granularity phase transition

总耗时: 8.3s
