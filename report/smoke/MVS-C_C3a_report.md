# MVS-C C3a — Migration Gate + Contract Hardening（advice/005.md，SMOKE）

> **定位（005 §十七）**：C2.1 FREEZE 后进入 **C3a Migration**——新架构 budget-aware Myopic-FG（Myopic-All，one-step QoS-dual，A={1,2,4,8}）在 N=8 homogeneous special case 下**逐项复现旧 B0.7-G2**。migration 判决只认 Myopic-All（Phase-PJ 不参加，005 §十七）。controller 等价性已审计：run_mvsc021.myopic_decision vs G2 q_min_fg（开发期 300 episodes + FULL seeds 复算，T40 回归 40 episodes × 3 corners × H∈{{48,96}} × FG/D8，max|Δ(cost,N_tx,payload)|=0）。

> 冻结参数（G2 017 §四 同协议）：N=8（GAMMA_B）、levels=(1,2,4,8)、b_setup=16.0、QoS(P_FA≤0.12, P_MD≤0.4)；ρ∈(128, 256, 512, 1024)、η∈(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)（28 combos/method，仅 calibration）；calibration worlds 共用、test fresh 分离（paired CRN）；主 operating point H=96、secondary stress H=48（同冻结 controller）；fixed-N paired one-sided Hoeffding + Wilson QoS 上端点。N_CAL=60、N_TEST=120。

> **Contract hardening（005 §七/§八/§六，C3b 前完成）**：(H1) **Myopic-PJ**（A={{next,full}}，one-step）——与 Phase-PJ 同动作集，消除旧 Phase-vs-Myopic 动作空间混杂；(H2) **Static Progressive**（固定 SNR 顺序 ladder）——Gate B 主基线回归；(H3) 统一 A/B/C 命名（phase_boundary 为准）；(H4) P_D,max 标注 **det-thr**；(H5) **policy-mixture/convex-hull 诊断**（deterministic-grid vs policy-class feasibility 分离，005 §六）。

## 1. Migration calibration（Myopic-All vs Direct8，G2 同协议）

- Myopic-All (G2-FG)：**∅（无 FEASIBLE）**；feasible 0/28
- Direct8：**∅（无 FEASIBLE）**；feasible 0/28
（22.3s）

## 2. Migration test @ H=96（θ̂ 冻结、fresh worlds、paired）

> θ̂ 缺失 → migration test 无法比较（QoS-UNRESOLVED）。
（0.0s）

## 3. Secondary stress @ H=48（同冻结 θ̂，诚实报告 boundary）

> θ̂ 缺失。
（0.0s）

## 4. Contract hardening：Myopic-PJ / StaticProg（005 §七/§八）

> **目的**：为 C3b 准备同动作集因果对照（Phase-PJ vs Myopic-PJ）与 Gate B 主基线（StaticProg）。本 runner 只报告 calibrated θ̂ 与 feasible 数（同 G2 协议），C3b 才做 Phase-PJ 算法主比较。

- Myopic-PJ (A={next,full})：**∅（无 FEASIBLE）**；feasible 0/28；其中 10/28 网格点为 E[B]=0 全停退化（QoS-dual 停止在 root 即触发，未发送任何消息）

- StaticProg (fixed SNR ladder)：**∅（无 FEASIBLE）**；feasible 0/28；其中 23/28 网格点为 E[B]=0 全停退化（QoS-dual 停止在 root 即触发，未发送任何消息）

- StaticProg 固定顺序（SNR 降序）: [7, 6, 5, 4, 3, 2, 1, 0] （GAMMA_B=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]）

## 5. Policy-mixture / convex-hull 诊断（005 §六 H5）

> 把 deterministic (ρ,η) 控制器映射为 v=(P_FA,P_MD,E[B])；error probabilities 与 E[B] 对 episode-level randomized mixture 线性 ⇒ 若二维混合能进入 QoS 象限，则 deterministic-grid infeasible 不能直接推出 policy-class infeasible。本诊断是 C3c 三层 feasibility frontier 的前置证据。**口径（007 审计注明）**：进入判定用**点估计**（kfa/n0，期望值线性）；正式 Gate 用 **Wilson U95**，混合的 U95 认证可行性留待 C3c convex-hull 正式计算——点估计进入 ≠ 统计认证。

- **Myopic-All (G2-FG)**：deterministic feasible 0/28；2-point mixture 进入 QoS 象限 216 对 → deterministic grid infeasible; convex-hull (2-point mixture, POINT-ESTIMATE metric) enters QoS quadrant in 216/378 pairs
- **Direct8**：deterministic feasible 0/28；2-point mixture 进入 QoS 象限 178 对 → deterministic grid infeasible; convex-hull (2-point mixture, POINT-ESTIMATE metric) enters QoS quadrant in 178/378 pairs
- **Myopic-PJ (A={next,full})**：deterministic feasible 0/28；2-point mixture 进入 QoS 象限 169 对 → deterministic grid infeasible; convex-hull (2-point mixture, POINT-ESTIMATE metric) enters QoS quadrant in 169/378 pairs
- **StaticProg (fixed SNR ladder)**：deterministic feasible 0/28；2-point mixture 进入 QoS 象限 79 对 → deterministic grid infeasible; convex-hull (2-point mixture, POINT-ESTIMATE metric) enters QoS quadrant in 79/378 pairs

## 5b. 4-bit N=4 exhaustive dominance-safety certificate（005 §十）

> C2.1 的 263 个 reachable support 是 **sampled on-policy certificate**（A 区未覆盖：A=0/B=16/C=247，A 主要靠 N=1 synthetic 反例）。005 §十 建议 4-bit/N=4 做**真正 exhaustive**（23^4=279841）dominance-safety 检查：prune ⟹ Q_prog ≥ Q_dir − ε 且 A 区绝不剪；8-bit（279^4）留给 resolution-stratified + adversarial 抽样。

- 4-bit N=4 **budget-reachable** (x,h) 对（BFS，H=96，真实成本 c=16+Δr）：607121 对；检查 432644 个 (x,i,probe-feasible) 支撑（region A/B/C 计数 {'A': 0, 'B': 339776, 'C': 92868}）；dominance 检查 432644（dir_feas），矛盾 0 → **PASS**（exhaustive budget-reachable certificate）（15.1s）
- **A 区覆盖说明**：exhaustive BFS 下 A 区（c1≤h<c_dir）在 N=4/H=96 无可达实例。正确理由（枚举）：单 UAV 在 4-bit ladder（levels 1,2,4、b=16）下可达花费 ∈ {17,18,20,34,36,52}（0→1:17、0→2:18、0→4:20、0→1→2:34、0→1→4/0→2→4:36、0→1→2→4:52），3 个其他 UAV 的总花费 = 三者和 ∈ {51..58,60,68..74,76,85..}；A 区需花费∈(76,79]（剩余 h∈[17,20)），该集合中 **(76,79] 为空**（74→76 间隙 2、76→85 间隙 9；76 对应 h=20 恰是 B/C 边界）。故 A 区不可达，A 区安全由 N=1 synthetic 反例 + “A 区绝不剪”代码路径 + dominance 检查显式断言保证（与 005 §十 观察一致）。

## 结论

- **C3a Migration Gate**：θ̂_FG=θ̂_D8=(256,0.8) 复现 FAIL；H=96 anchor（E[D]=−5.3250、U95<0、双方 FEASIBLE）复现 **FAIL** → migration 未通过，先修 migration（005 §十七：Phase-PJ 不参加判决）。

- **Contract hardening 落地**：Myopic-PJ 已实现（A={next,full}，one-step，与 Phase-PJ 同动作集）；StaticProg 已实现（固定 SNR ladder）；A/B/C 命名统一 phase_boundary 为准；P_D,max 标注 det-thr；policy-mixture 诊断见 §5（C3c 前正式 convex-hull frontier）。

总耗时: 62.1s

