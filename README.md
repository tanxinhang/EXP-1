# O-PEF MVS-A/B0 — 最小可实现系统（R1→R2.1→MVS-B0 演进）

依据 [SystemModel.md](./SystemModel.md)（O-PEF v2.1）搭建的**可运行、可复现**实现。
当前版本完成 **MVS-A（v0→R1→R1.1→R2→R2.1，已封板）** 与 **MVS-B0
（Sparse-State Header-Aware Cross-Level O-PEF）**：

- 单目标二元检测，`N = 4` 架 UAV，Gaussian 局部检测器（§30–§32）；
- 每 UAV 一个 **nested 多精度量化器**（0/1/2/4 bit，§7、§14）；
- **log-domain 软融合**与 message-LLR，粗→细证据的 *replace-not-add* 更新（§8、§9、§11）；
- 证据状态 `z = (z_1,…,z_N)` 作为有限充分 Markov 状态（§10）；
- **Exact DAG-DP**（23^4 = 279,841 状态，memoized backward recursion，非 value iteration，§18、§20、§21）；
- **O-PEF-1**（depth-1 消融，§23）、**O-PEF-2E**（depth-2 精确期望主算法，§24、§27）
  与 **O-PEF-3**（depth-3 诊断性 solver 改进）；
- 基准：Raw Fusion / All-Neighbor / Random-K / SNR Top-K / Censoring / OTS-F / P-OTS /
  Global Fixed Progressive / Static Cost-Aware Progressive（§49–§53）；
- **R1（依据 advice/001.md 审计）**：双乘子 (μ_M, μ_F) sweep + 策略自身终端判决、
  adjacent-only 动作族（0→1,1→2,2→4）、Lagrangian G2（J=E[B]+μ_M P_M+μ_F P_FA）、
  精确前向概率传播（主结果无需 MC）、G1a/G1b 独立认证、公平基线 1-bit-seeded P-OTS、
  2×2 solver 实验（cross-level × depth-2/3）。
- **R1.1+R2（依据 advice/002.md）**：R1.1 三个小修复（1bit_POTS 冻结 1-bit 排序、
  exact_np_roc 按假设分别归一化、"不减"文字修正）+ full-precision constrained
  policy-mixture LP（`scipy.linprog`）冻结 **B_DP^CMDP** oracle；R2 实现
  **resource-bounded lookahead**（horizon = 未来 payload bits）`V_h(x)=min{R_stop,
  min_{a:c_a≤h}[c_a+E V_{h−c_a}]}`，硬认证 V_16(x)=V*(x) 机器精度。
- **R2.1（依据 advice/003.md，MVS-A 封板）**：CMDP **column generation**
  （LP master + ExactDP pricing oracle）认证全局最优 **B_CMDP\***；hard-budget
  （RB-HardBudget，ablation）与 **receding RBL-RH** 分离（每状态重新读取
  policies[H]）；online sparse planner（memoized Solve(x,h)，不建全表）与
  eager 表全等价审计；H<16 才计入 scalability 评选；η_rec 用 B_CMDP\* 重定义。
- **MVS-B0.1a（依据 advice/006.md）**：1bit-POTS seed 后立即停止 + seed-aware LLR；
  真前缀 CRN（paired 对比）；Wilson CI 正式输出；resource lattice 保守化
  （true cost + ceil budget，离散化界 0≤C̃−C<N_txn·Δc）；**reachable-state ΔQ sweep**
  验证 critical feedback-granularity threshold（b*≈8：P(ΔQ>0): 0→0.69）。
- **MVS-B0.3（CR-RBL）**：Confidence-Certified Rollout RBL——base-rollout Q_a^πb
  + anytime Hoeffding 证书（δ_{a,n}=6δ/(π²|A|n²)）+ LUCB challenger + nested-evidence
  CRN（耦合 latent rollout）；return 有界 0≤G≤C_max^rem+R_max；Gates G0（CI 覆盖）、
  G1/G2（N=4/N=8 exact oracle 对比）、G3（certification error≤δ）、G4（scalability
  H=48/64/96/120 无全 cone）、G5（matched QoS 方向性）；创新定位：
  Confidence-certified feedback-granularity-aware evidence acquisition over a
  variable-cost nested-evidence DAG。
- **MVS-B0.3a（依据 advice/007.md，credibility patch）**：修复 B0.3 的口径问题——
  **P0-A 真正跨 action CRN**：每次 MC 迭代采样一个 latent world
  W_m=(H_m, M_1^(8), …, M_N^(8))|x，所有候选动作在同一 W_m 上求值 G_a(W_m)。
  实现发现 world 必须含隐假设 H_m（先按后验采 H、再按 H-条件分布采各 UAV cell）——
  逐 UAV 边缘独立采样破坏 H 诱导的跨 UAV 相关性，会使 rollout 估计系统性偏高
  ~12 bits（对 N=4 exact oracle 验证）；修正后 paired-CRN 方差比
  Var(G_a−G_b)/[Var(G_a)+Var(G_b)] ≈ 0.08–0.15（相对独立采样降 7–12 倍）；
  **P0-B certificate 竞争集含 STOP**（R_stop(x)=min{C₀₁p,C₁₀(1−p)} 精确项：
  reporting 动作需 U_â ≤ min{R_stop, min L_b}+ε）；
  **P0-C G0 oracle = Q_a^{π_b}**（exact_qa_pi_b，非 Q_a^⋆）；**P0-D G1/G3 STOP oracle =
  R_stop(x)**（非 base_policy_value）；**P0-E G5 硬预算** h_t=H−C_t pathwise、C_T≤H 逐样本
  成立；**P1-A anytime coverage gate**（∀n≤n_max）；**P1-B binomial U95 violation gate +
  certification rate**（0-violation 需 ≥59 certified 样本使 U95≤0.05）；**P1-C 回归不变量
  T15–T20**；**P1-D b⋆ 重表述为 root-state threshold**：g_x(b)=E[min{Y_x,b}] 非减凹、
  g_x'(b)=Pr(Y_x>b)，b⋆(x)=inf{b:g_x(b)≥0} 为 state-dependent packetization phase
  boundary（根状态 b⋆(x₀)=7.0，1-bit 子状态 b⋆=∞ ⇒ direct regime）。
- **MVS-B0.3c（依据 advice/008.md，收口补丁）**：**(1) natural 阈值修正**——
  G5 的 natural 指标改为 Ω>η_nat=log(μ_F/μ_M)=1.0（原来错用 Ω>0；与 eval_exact.py
  锁死，新增 T21）；**(2) G5 改名** directional (unmatched) hard-budget comparison
  （P_D 与 E[B] 同时不同，只有 Pareto 方向性，正式 matched-QoS 留 B0.6）；
  **(3) 三格消融**（G7）分离 bias correction 与 variance reduction：marg×ind
  E|Q̂−Q|=8.4 / P(match)=0.70 → joint×ind 5.6/0.73 → joint×paired 5.6/**0.95**，
  耦合效率 κ≈13（n_paired≈n_uncoupled/κ）；**(4) Hoeffding range 收紧**（008 §4）
  B→B_a(x,h)=min{c_max_rem,h}+R_max−c_a（root N=8: ~950→~422@H=48），G3
  certification rate 75.5%→94.5%（ε=40）；**(5) T17 拆分**为 T17a 确定性证书蕴含
  （100% PASS）+ T17b 经验审计，T19 标注为统计 sanity；**(6) E[Y_x] 存在性判据**
  （008 §6）：b⋆(x)<∞ ⟺ E[Y_x]≥0，E[Y_x]<0 ⇒ progressive dominates direct for
  every b_h≥0（根 EY=38.56>0↔b⋆=7；1-bit 子状态 EY<0↔b⋆=∞，analytically）。
- **MVS-B0.4（依据 advice/009.md，主算法升级）**：**Pairwise-Difference Time-Uniform
  EB-CS Planner**——直接估计 Δ_{a,b}=Q_a^{π_b}−Q_b^{π_b}（Z_t^{a,b}=G_a(W_t)−G_b(W_t)，
  共享 latent world），取代 arm-wise Q_a+U_a−L_b；WSR-2023 betting martingale CS
  （Ville 不等式、variance-adaptive λ、grid inversion，G0 anytime coverage 1.0 验证）；
  **predictable candidate–challenger pair sampling**（(a_t,b_t) 由 F_{t−1} 决定、每 pair
  α_ab 且 Σα_ab≤δ、**2 rollouts/world**，替代 B0.3c 的 32/world 全配对）；证书
  U_{â,b}≤ε ∀b⟹Q_â≤min Q_b+ε（STOP 精确，只需 G_â−R_stop 单边）。实现中修复了 pair
  canonical-orientation bug（candidate 切换会混入反号样本，破坏 CS）。Gates：G0 Pair-CS
  validity PASS（coverage 1.0）；G1 sample efficiency——同 6000-world 预算下 EB certification
  rate 0.68/0.70/0.83（ε=2/4/8）vs Hoeffding 0.03/0.03/0.08；G2 action quality——等 rollout
  预算下与 B0.3c 可比（ε-opt(2)=0.925 vs 0.950，ε-opt(4)=0.970 vs 0.965，用 6–16× 更少
  rollout）；G3 N=8 shallow near-tie-aware（H=24/34/40 的 Q(a)−Q_min ≤ 1.09 bits）；
  G4 scaling（2 rollouts/world，总 rollout 与 |A| 无关）。
- **MVS-B0.4r（依据 advice/010.md，credibility patch）**：**R0** canonical sample+support
  同向——range_ab 先 canonicalize 再算 [l_c0−u_c1, u_c0−l_c1]，PairCS.update 硬断言
  z∈[lo,hi]（descending top-k 回归 T25 锁死，修掉非升序动作枚举下的 support 方向 bug）；
  **R1 formal 证书路径 = predictable plug-in empirical-Bernstein CS**（Maurer–Pontil 2009
  Thm 6 + peeling union bound，连续区间、无 grid inversion；betting grid CS 降级为实验性
  收紧消融，不再承担主证书）；**R3 G1 四格消融**（H0 arm/Hoeffding/full、H1 pair/Hoeffding/
  challenger、E1 pair/EB/challenger/shared、E0 pair/EB/challenger/independent）分离三段因果：
  H0→H1 无收益（sparse pair sampling + 全 range 反而更宽）、H1→E1 是 variance-adaptive CS
  的贡献（~2× 更少 rollout）、E0→E1 是 nested CRN coupling 的贡献（无耦合无法认证）；
  **R4 G2 改为硬 rollout 预算曲线** P(Q−Q_min≤2) vs R∈{1000,3000,6000,12000}；
  **R5 G4 表述修正**——per-world rollout O(|A|)→O(1)，但总 certification complexity 仍
  随 |A|（log P 置信分配 + O(|A|) challenger 搜索 + O(|A|²) pair 存储）。
- **MVS-B0.4s/B0.4a（依据 advice/011.md）**：**B0.4s 基础设施收口**——smoke 输出改到
  `report/smoke/` 独立路径 + FULL 文件 hash 不变断言（防覆盖回归）；删除 `z_code_b`
  未用 import 与 `last_cand` 死变量；Maurer–Pontil 引用修正为 **Theorem 4**（
  √(2V log(2/δ)/n)+7log(2/δ)/(3(n−1)) 是 MP Thm 4；代码用 (n−1) 分母更保守，纯引用
  编号修正）。**B0.4a Certified Policy Improvement**——`base by default; override
  only with certified evidence`：a_inc⁽⁰⁾=a_b（**one-step conditional-VoI base**，
  Q_a⁽¹⁾=c_a+E[R_stop(X')|x,a]，STOP 当 VoI≤0），challenger c 仅在 pairwise CS 证明
  U_{c,a_inc}<0 时替换 incumbent（monotone improvement chain ⇒ V^{π_CPI}≤V^{π_b}）；
  episode 级 δ：决策 t 花 δ_t=6δ_episode/(π²t²)，决策内每 pair α=δ_t/P ⇒
  P(所有 override 有效)≥1−δ_episode。实现修复：STOP 作为 challenger 时误用 base-rollout
  而非精确 R_stop（P0）、challenger 选择卡在"真 Δ>0 的 pair"（改 point-estimate 聚焦 +
  消除）。四 Gate：G0 fallback tail、G1/G2 override 收益与安全性（N=4 exact matched-base
  + binomial U95）、G3 VoI-base 强度。关键口径：CPI 的 exact oracle 必须与 rollout 的
  base 匹配（SNR-base oracle 检查 VoI-base rollouts 会产生假 violation）。
- **MVS-B0.4a-r（依据 advice/012.md，credibility closure）**：**budget-aware base**
  π_b(x,h) 全局贯通（012 §5：CPI anchor、MC rollouts、exact oracle 统一调用
  base.act(pl,x,om,h)）；**Formal vs Operational CPI 分离**（012 §1）——cs_mode="eb"
  是 theorem-backed PrPl-EB（承担 safety claim），cs_mode="betting" 是 finite-grid
  实验 CS（只做性能探索，无严格连续置信保证，此前 11 次 override 是 operational
  证据而非 formal 证书）；**base-anchored O(|A|) 置信分配**（012 §4 方案 A：每个
  candidate 只与原 a_b 比较，α_c=δ_t/n_cand，取代 all-pairs O(|A|²) persistent
  chain；修 resolve 的 pair-key bug + block-based focus B_focus=200）；**δ_1 修复**
  （012 §3：独立状态是 episode 第一个决策，δ_1=6δ_episode/π²，取代 stale δ_t）；
  **G2 UNCERTAIN 口径**（012 §2：0 violations 但 override 数 <59 ⇒ 不足以认证
  5% violation rate，不写 FAIL）；新增 t-scan P_override(t∈{1,2,4,8})。
- **MVS-B0.4b（依据 advice/013.md，Feedback-Granularity Phase-Transition Theorem，
  纯理论封板，不改 planner）**：主定理 **g_x(b) = Q_prog − Q_dir = E[min{Y_x, b}]**，
  Y_x = D_x(X₁) − Δ₂，D_x(X₁) = R(X₁) − E[R(X₂)|X₁]（第二阶段纯信息收益），Δ₂ =
  r_max − r_next；**导数即触发概率** g'₊(b)=Pr(Y_x>b)、g'₋(b)=Pr(Y_x≥b)——
  setup 开销对 packetization preference 的边际影响 = 第二次反馈 transaction 的
  触发概率（013 §2，比"存在 threshold"更像论文主定理）；**b⋆ 三情形完整分类**（013 §3）：
  A E[Y]<0 ⇒ b⋆=∞（progressive 严格占优 ∀b）；B E[Y]=0 ⇒ b⋆=max{0,ess sup Y}，
  b>b⋆ 是**持平**而非 direct 占优（synthetic Y={−1,+1} 用例防误写）；C E[Y]>0 ⇒
  唯一有限 crossing（Y≥0 a.s. ⇒ b⋆=0）；物理判据 **b⋆<∞ ⟺ E[D_x]≥Δ₂**（sensing/
  evidence value vs communication payload，013 §4）；**exact support computation**
  （013 §5：Y_x 离散 ⇒ g_x 精确分段线性，crossing 闭式 b⋆=b₀−g(b₀)/Pr(Y>b₀)，取代
  G6 grid 插值；root b⋆=7.0000 精确复现）；四 Gate G0 identity（random reachable
  states 最大误差 <1e-10，实测 3.9e-13）/G1 shape（monotone+concave+导数=survival）/
  G2 existence（三情形 100%）/G3 state dependence（finite+∞ 并存 ⇒ 非全局阈值，
  **不声称** |Ω|/SNR 单调性）；回归 T28–T31。创新定位冻结（013 §8）：**state-dependent
  packetization phase transition + g'ₓ(b)=P(additional feedback transaction)**，
  与 B0.4/B0.4a 的 paired-difference certified acquisition 组合；不自称
  "adaptive quantization" 本身新颖。
- **MVS-B0.6-pre（依据 advice/008.md §14 + 013 final，先验收算法）**：**N=4 exact /
  N=8 shallow sample-complexity gate**——验收 B0.6 每步决策部署的 CPI acquisition：
  G0 N=4 exact P(gap≤2)/E[gap] vs 世界预算 W（base ablation 定 012 §6 的 anchor：SNR
  略优、VoI 为理论 anchor）；G1 N=4 formal certification（eb-mode，按 base 真 gap≥15
  分层——MP+peeling 的 range-scaling 项 7t(hi−lo)/(3(n−1)) 使小 gap 认证昂贵，实测
  W=4000 时 0/9、W=8000 时 9/9 override，即 gap≥15 在 ~8k worlds 内可正式认证）；
  G2 N=8 shallow（H=40，oracle=sparse exact Q^{π_b}）；G3 sample-complexity 记账 +
  B0.6 可行性（G0 曲线取 w_ep ⇒ 800 episodes × 3 decisions 的 matched-QoS 模拟预计
  <10 min）。四 Gate 全 PASS ⇒ 算法验收通过。
- **MVS-B0.6-pre-r（依据 advice/014.md，credibility patch，不改 planner）**：
  修三个 P0 后算法方向才可信。**P0-1** 真实 **(x,h,t)** augmented states——random
  trajectories 与 on-policy occupancy episodes（实扣 setup+payload，h=H−C_t，t=决策
  序号）去重；oracle/CPI 用状态自己的 h，δ_t=6δ_episode/(π²t²)；budget-exhausted
  状态（无可行动作）不是决策点，剔除。**P0-2** 全 Gate 冻结 **SNR anchor**（G1
  formal 之前误用 VoI）。**P0-3** Gate 改 **CI 判定**（Wilson LCB95 + paired 差分
  t-UCB95），w_ep 用 **held-out** 半集选择。诚实口径（014 §2）：25/30 的 Wilson
  ≈ [0.66, 0.93]，n=32 下 P(gap≤2)≥0.80 **不能统计认证**（UNCERTAIN）——E[D_s] 统计
  显著为负、G1 formal LCB95≥0.5 PASS、G3 pilot（w_ep=1000 冻结，014 §3）≤60 min
  PASS ⇒ **ENGINEERING ACCEPTANCE，正式统计认证交 B0.6 stratified matched-QoS**。
  顺带修 CPI.decide 的 anchor clamp（012 §5 budget-aware：名义 base 动作在 h 下不可
  负担 ⇒ anchor=STOP，T32 回归）。
- **MVS-B0.6（依据 advice/014.md §4-§6，论文生死 Gate）**：**matched-QoS 下 CR vs
  optimized Direct8（POTS 第二 comparator）**。协议冻结：**stratified N0=N1**（H0/H1
  独立采样，初始 N0=N1=600、escalation 1000/1600/2500）；episode 级 **CRN**（同一
  physical world W_e=(H_e,L_e) 给三个算法，planner RNG 独立）；radio cost 与
  planning cost 分离（B_radio=Σ(b_setup+Δr_t)，CPI worlds 只算 compute）；判决阈值
  η_nat=log(μ_F/μ_M)=1（T21）；CR = 冻结的 SNR anchor + Operational-CPI
  （betting，w_ep=1000 pilot，δ_t 按决策序号）；Direct8 = optimized（SNR-order 全包
  + η_nat stop；direct_only planner 在 H=96 不可行——256⁴ cells）；POTS = round-robin
  渐进。Gate：U95(P_FA^CR)≤α=0.12、U95(P_MD^CR)≤β=0.40（natural Bayes 工作区）、
  **U95(E[D_e^D8])<0**（episode-paired D_e^D8=B_e^CR−B_e^D8）；三态
  PASS/FAIL/UNCERTAIN（扩样）。**结果（N0=N1=600，w_ep=1000）**：H=48/96 的 CR QoS
  全 PASS（P_FA U95 0.101/0.095、P_MD U95 0.392/0.253），但 **bit Gate FAIL 决定性**——
  CR 比 optimized Direct8 多花 +6.93（CI [6.41,7.45]）/+18.75（CI [17.65,19.84]）
  bits/episode；NP-matched（P_FA=0.05）视角 CR P_D=0.563/0.752 vs D8 0.440/0.488
  （CR 用更多 bits 买更好检测）。**b_setup∈{0,4,8,16,32} regime map（secondary，
  预先声明）**：CR 在所有 b_setup 都更贵（D=+11.8…+20.3），**理论预测的 crossover
  未出现**——瓶颈是冻结 base 的保守停止（|Ω|≥2，eta_b=2）vs Direct8 的 η_nat=1，
  而非 packetization granularity。**诚实结论（014 §7）：phase transition 与
  state-dependent adaptive packetization 理论成立（B0.4b 已验证），但在当前
  b_setup=16 regime 下 optimized direct packetization 已接近最优通信工作区间；
  不再改算法"调赢"。**
- **MVS-B0.6-r / B0.6-d（依据 advice/015.md，口径纠偏 + 成本分解，不改 planner）**：
  015 复核认定 B0.6 的 "matched-QoS" 只认证了 **CR 的 QoS**（Gate 只用 CR 的
  pfa/pmd 字段），Direct8/POTS 从未给 QoS CI，不能声称 "在相同 QoS 下比较 bit"。
  **P0-1 matched-QoS 语义**：对 **三个方法** 都算 Wilson 95% CI 并分类
  FEASIBLE / INFEASIBLE / UNCERTAIN（015 §5：A≺B ⟺ A,B∈F_QoS ∧
  U95(E[B^A−B^B])<0）——只有双方都 FEASIBLE 才允许比较 E[B]；**P0-2 判定降级**：
  CR 的 bit 结论降为 **COMMON-THRESHOLD BIT LOSS / MATCHED-QoS UNRESOLVED**——
  Direct8 有更低 raw cost 但未被认证 QoS-feasible 时只写 "Direct8 has lower raw
  communication cost but is not certified QoS-feasible at this operating point"，
  不写论文生死 FAIL；**P0-3 regime-map 表述**：B0.4b 只证明状态局部的
  packetization 相变（g_x(b)=E[min{Y_x,b}]、b*_x state-dependent），**不蕴含**
  episode 级全局 D(b) 单调或必在 b_setup≈b*₀ 全局 crossover（全局量混入 state
  occupancy / stopping time / UAV selection / remaining budget / CPI override）——
  地图改称 **system-level regime diagnostic: no global crossover observed**；
  **B0.6-d（015 §六）成本分解**：逐 episode 记账 N_tx 与 B_payload，断言恒等式
  B=b_setup·N_tx+B_payload，报告 E[N_tx]、E[B_payload]、E[T_stop]、P(T_stop=k)
  ——验证 "CR 贵在 evidence payload 过量而非 transaction 碎片"。
- **MVS-B0.7-G0（依据 advice/015.md §十/§十三，common-stop 机制 Gate）**：
  **把 STOP 与 GRANULARITY 拆开**——B0.6 的 CR-vs-D8 同时变了 stopping
  threshold / candidate policy / granularity / CPI override / transaction
  count，输赢都无法归因 multi-granularity 机制；G0 用 **完全相同的公共停止
  控制器** S_λ(x,h)（015 §九 one-step approx：CONTINUE ⟺ min_{a∈A_all}
  Q_λ^(1)(a|x,h) < R_λ(x)，Q_λ^(1)=c_a+E[R_λ(X')|x,a]，与包粒度无关，对两方法
  同一判定），只比较 **Direct8（A_D8={(i,8)}）vs Adaptive Granularity
  （A_FG={(i,1),(i,2),(i,4),(i,8)}）**，两者用同一 one-step Q greedy 选
  UAV——唯一系统差异 = feedback granularity。**N=4 exact 小系统**（GAMMA=
  [-1,1,3,5]、levels (1,2,4,8)、r_max=8）；stratified N0=N1；episode 级 CRN
  （planner 确定性 → CRN 自动）；记账复用 B0.6-d（B=b_setup·N_tx+payload、
  E[T_stop]、P(T_stop=k)）。Gate（015 §十三 G0）：FG 显著降低 paired E[B]
  （U95<0）且 QoS 未被证伪（非 INFEASIBLE）→ **granularity 有独立价值，进
  G1**（N=8 held-out QoS-dual calibration + 双方 Wilson U95 认证）；否则
  **STOP，关闭 performance-improvement 主线**，转 015 §十四 Direct8-近优
  lower-bound（V_LB≤V⋆≤V^D8）。冒烟（N0=N1=120，b_setup=16）：FG E[B]=
  24.63/26.17 vs D8 30.10/32.30（H=48/96）；分解显示 **N_tx 几乎相同而
  payload 4.83 vs 10.03 → 省的全是 evidence payload**（015 §六预测在
  common-stop 下成立）。
- **MVS-B0.7-G1（依据 advice/015.md §七-§九/§十三，held-out QoS-dual 认证）**：
  **双参数（λ_M 标度 + η_dec=log(λ_F/λ_M)）只在 calibration seeds 上确定**，
  test seeds 完全 fresh（015 §七 anti-post-hoc 结构）；**停止 = dual 风险**
  R_λ(x)=min{λ_M p, λ_F(1−p)}，**STOP ⟺ R_λ(x) ≤ min_a Q_λ^(1)(a|x,h)**
  （Q_λ^(1)=c_a+E[R_λ(X')|x,a]，015 §九 one-step；**移除 |Ω|≥κ 对称硬停**，
  015 §三指出其无 Bayes 理由）；判决 Ω>η_dec→H1；两方法共用同一 S_λ（公共
  停止），唯一差异 = granularity（A_FG={1,2,4,8} vs A_D8={8}——实现上
  A_FG=A_all，S 判定与 FG 动作合并单次遍历）。**N=8**（GAMMA_B、levels
  (1,2,4,8)、b_setup=16）；calibration N0=N1=300 @ H=96，扫 η_dec∈{1.0…2.0}
  选"双方 U95(FEASIBLE) 且 E[B^FG]+E[B^D8] 最小"的 **η_star=1.2** 冻结；
  test N0=N1=600 @ H∈{48,96}。**Gate（015 §十三 G1）**：test 上双方均
  U95(P_FA)≤0.12 且 U95(P_MD)≤0.40 才比较 U95(E[B^FG−B^D8])<0。
  **FULL 结果**：**H=96：双方 FEASIBLE（FG U95 0.0509/0.3958、D8 0.0588/
  0.3515）→ E[D]=−12.31（CI [−13.26, −11.36]）<0 → matched-QoS PASS**；
  分解 setup 差仅 −1.24、**payload 差 −11.07**（granularity 收益仍几乎全来自
  evidence payload）；**H=48：双方 P_MD U95 略超 0.40（0.4481/0.4127）→
  UNCERTAIN（诚实拦截，扩样可解）**，E[D]=−10.02 方向一致。**结论：held-out
  matched-QoS 口径下 granularity 有独立收益**（论文主线核心证据；B0.6 的 FAIL
  不是 granularity 机制失败，而是 stopping/selection 未解耦 + matched 口径缺
  认证——015 预判最有价值结果）。
  * **016 §1 P0 修正**：G1 的公共停止器 S_common 用 min_{A_FG}Q<R_λ 判定——
    但 D8 实际只能从 A_D8 选动作，存在 F(x)=1{q_FG<R_λ≤q_D8} 状态（小包值得买
    → 公共控制器说继续，但 8-bit 包已不值得 → D8 被迫发 8-bit）。A_FG 本身含
    granularity 信息 ⇒ "STOP 判定与包粒度无关"表述不成立（action-set leakage，
    会天然抬高 D8 成本）。G1 的 −12.31 方向来自真实 granularity 仍大概率成立
    （016 §2：E[N_tx] 只差 0.0775，payload 4.60 vs 15.67，优势几乎全在包粒度），
    但需要 G1r 审计与保守 Gate 封板。
- **MVS-B0.7-G1r（依据 advice/016.md，P0 审计 + 保守 Gate + 代码回归，不改 planner）**：
  **G1r-A（016 §15-1）Forced-Continuation Audit**：在 D8 每个决策状态记录
  q_FG=min_{A_FG}Q、q_D8=min_{A_D8}Q、R_λ；F(x)=1{q_FG<R_λ≤q_D8}；报告
  P(F=1)（按决策状态）、P(episode contains F)（按 episode）、
  ΔB_forced=Σ_{F 状态}(D8 被迫支付的 8-bit 通信成本)——回答"D8 的通信中到底
  有多少是 FG-action-defined common stop 强迫出来的"；
  **G1r-B（016 §4/§15-2）保守 S_ref Gate**：停止器改为
  S_ref(x,h): CONTINUE ⟺ min_{a∈A_D8}Q_λ^(1)(a)<R_λ(x)——两方法共用
  （只有"至少一个 Direct8 full packet 值得发送"才给 FG 一次 adaptive-granularity
  机会，对 FG 更苛刻）；若此保守版仍 U95(E[B^FG−B^D8])<0 且双方 FEASIBLE →
  granularity 独立收益基本无法从公平性击穿（016 §4 预期：−12.31 可能缩小到
  −5..−10，但不会翻正）；**G1r-C（016 §15-3）代码可信度封板**：q1_fast vs
  独立 generic dual-Q exact 回归（max|Δ|<1e-9，回归实际揪出并修复了
  dual_q_exact 的 np.maximum 权重 bug→logaddexp）+ D8 emulation invariant
  （FG 限定 A_D8 ≡ D8 分支，逐样本一致）；**统计口径（016 §10）**：paired bit
  正式用 one-sided paired Hoeffding U=D̄+2H√(log(1/δ)/2n)（D∈[−H,H]，
  分布无关、无 t 假设），t-based one-sided 仅参考；QoS 保留 Wilson；Gate 明确
  为 intersection-union test。**FULL 结果（N_TEST=600，calibration N_CAL=300）**：
  G1r-C 双回归 PASS（q1_fast vs generic dual-Q max|Δ|=0——回归实际揪出并修复了
  dual_q_exact 的 np.maximum 权重 bug，改为 logaddexp；D8 emulation 50 episode
  逐样本一致）。校准：S_common（G1 语义）η_star=1.2（达标 {1.0,1.2}）；S_ref
  （保守）η_star_ref=1.0（达标 {1.0}）——各自冻结。**G1r-A（S_common 审计）**：
  P(F=1) 很小——H=48: 0.0221（41/1856 决策状态）、P(episode F)=0.0342、
  ΔB_forced=984 bits；H=96: 0.0277（65/2350）、P(episode F)=0.0542、
  ΔB_forced=1560 bits——**D8 只有约 2-3% 的决策状态被 FG-action-defined stop
  强迫**，G1 的 D8 劣势主要来自真实 granularity（016 §15 判定）。**G1r-B（保守
  S_ref）**：H=96 **双方 FEASIBLE（FG 0.0666/0.3703、D8 0.0762/0.3190）→
  Hoeffding 95% U95(E[B^FG−B^D8])=−5.16 <0 → G1r-B PASS**；E[D]=−11.94、
  分解 setup 差仅 −1.20、**payload 差 −10.74**（016 §4 预期 −12.31→−5..−10
  精确命中：缩小但远未翻正）。H=48 双方 P_MD U95 略超/贴近 0.40 → UNRESOLVED
  （扩样可解），E[D]=−9.54、Hoeffding U95=−6.15 方向一致。**结论：在保守
  A_D8-reference common-stop 下 granularity 独立收益仍显著（U95<0），fairness
  无法击穿**——按 016 §15 投入 fresh **B0.7-G2**（FG/D8 分别 (ρ,η)
  calibration，test 完全 fresh、暂不加 CPI）。
- **MVS-B0.7-G2（依据 advice/017.md，Separately Calibrated QoS-Dual
  Policy-Family Certification，性能主线闭环）**：**FG/D8 分别拥有自己的
  (ρ,η) dual controller**（017 §二：λ_M=ρ、λ_F=ρe^η，R_{ρ,η}(x)=
  ρ·min{p_x,e^η(1−p_x)}，Q^(1)(a|x,h)=c_a+E[R_{ρ,η}(X')|x,a]，A_FG=
  {(i,r'):r'>r_i,r'∈{1,2,4,8}} vs A_D8={(i,8):r_i<8}，各自 STOP⟺
  R≤min_{A_m(h)}Q^(1)、各自 argmin、Ω>η⇒H1——彻底无 common-stop /
  action-set leakage）；**分别 calibration**（017 §五：N_CAL=600/hyp @
  H=96、FG/D8 **共用** worlds、grid 冻结 4ρ×7η=28/method，选 θ̂_m =
  calibration-feasible（Wilson U95≤α,β）中 Ê_cal[B] 最小；命名
  **separately calibrated one-step QoS-dual controllers**，017 §三）；
  test **完全 fresh**（017 §六 方案 A：**N_TEST=1600 一次性冻结**、无
  staged escalation ⇒ fixed-N Hoeffding 95% 声明成立；paired CRN；
  one-sided paired Hoeffding + Wilson QoS；**CPI OFF**）。**P1 口径修正
  （017 §一）落地**：forced-continuation 只报 **P(F=1|S_common=CONTINUE)**
  （不再写裸 P(F=1)；把 STOP 决策状态加入分母只会更低）、ΔB_forced→
  **gross forced-action cost**（按 episode 归一化；H=96 fresh N=1600
  下 ≈1.49 bit/ep = **immediate forced expenditure**（非 cascade/总
  leakage 上界；因果公平性强证据由 G1r-B 提供，018 §九）、emulate_d8 计数
  按真实循环数（P2）。**Invariant suite 升级（017 §九，全 PASS）**：
  inv-1 ΣP(m'|x,a)=1（max|Σw−1|=0）、inv-2 dual-Q 回归（root +
  on-policy reachable + r=0/1/2/4 分层 × corners {(128,0.8),(512,1.2),
  (1024,2.0)}：15085 对 max|Δ|=2e-12）、inv-3 B=16·N_tx+B_payload、
  inv-4 B≤H（56 θ-run 0 violations）、inv-5 π_FG|_A_D8 ≡ π_D8（100
  episodes 逐样本一致；**control-flow equivalence 检查，非独立实现互检**，
  018 §十二）。**FULL 结果**：θ̂_FG=θ̂_D8=**(256,0.8)**；
  **H=96 primary：G2 PASS**（双方 FEASIBLE：FG Wilson U95 0.0896/
  0.3121、D8 0.0849/0.3140；E[D]=**−5.33**、Hoeffding U95=**−1.17**<0、
  L95=−9.48）；**H=48 secondary：也 PASS**（同冻结 controller 双方
  FEASIBLE，E[D]=−5.03、U95=−2.95<0——operating-region boundary 未
  出现）；分解 setup 差 **+1.52**（FG 事务更多：16·(E[N_tx^FG]−E[N_tx^D8])
  =16·0.0953）、payload 差 −6.85（净省全来自 evidence payload），
  合计 −5.33）。**机制归因（018 §七 限定）**：对**本次 run 选出的等 θ
  控制器**（θ̂_FG=θ̂_D8），两者唯一代码差异是 admissible evidence-
  acquisition action space（同时影响 selection 与 stopping）⇒ 观测 test
  gap 归因于该差异——**不外推为一般结论**（D8 的 θ̂ 是 empirical
  minimizer，near-tie 见下）。**结论：G2
  回答 017 §二 主问题 = FG 胜**（E[B|H0]/E[B|H1] 两 hypothesis 均省，
  非平均掩盖）。**sensitivity（018 §四/§五 收紧；runner 已改为扫描所有
  Ê_cal[B] < Ê_cal[B_{θ̂}] 的 challenger 并逐个分类）**：FG@(128,0.8) 是
  **material + UNCERTAIN challenger**（U95(P_FA)=0.1229 仅超 0.12，
  Ê[B] 差 ≈9.6 bits/episode——**material**，019 §4）——
  “may potentially select a lower-cost FG controller; this requires an
  independent sensitivity calibration”（019 §5，不断言“只会加大”；
  **018/019：G2 冻结，不再跑 sensitivity，留作 limitation/sensitivity
  note**）；另有 **FG@(256,1.4) 是 numerical near-tie challenger**
  （Ê[B] 差仅 0.0017 bit/episode，无实践意义，不值得为它改 policy，
  019 §4）；D8 更低成本候选均硬 INFEASIBLE（ρ=128 ⇒ P_MD≈1.0）；
  **D8@(256,0.8) vs (256,1.0) 仅差 0.02 bit/episode ⇒ θ̂_D8 是 empirical
  minimizer**（018 §七 near-tie）。
  论文表述（018 §八 收紧）：Under separately calibrated QoS-dual
  controllers selected from the same pre-specified calibration grid and
  evaluated on fresh held-out trials, adaptive feedback granularity
  achieves statistically certified communication savings **relative to
  the calibrated Direct8 controller**（test 认证对象是 π̂_FG vs π̂_D8
  单对，非整个 Direct8 policy family）。
- **MVS-B0.1（依据 advice/005.md，可信度修复+理论拔高）**：修复 1bit-POTS 重复计数；
  共享 CRN + 置信区间；Natural-policy QoS 与 NP ROC 双口径分离；Adaptive Direct-8 最优
  baseline（隔离 UAV 选择与 multi-resolution 收益）；**state-dependent conditional-VoI 定理**
  （Q_prog−Q_dir = E[min{D(x')−Δ₂, b_h}]，b_h=0 ⇒ 渐进支配）；q<7/23 降为 Corollary 并新增
  E[C_future|M^(1)]<7 判据；feedback/setup 成本（b_setup）敏感性；正式 regression test suite
  （all checks PASS）；创新定位：**Feedback-Granularity-Aware Adaptive Evidence Acquisition**。
- **MVS-B0（依据 advice/004.md）**：**sparse tuple-state backend**
  （状态 x=(z_1..z_N)、Ω 现场计算、动作/PMF on-demand、memo key (x,h)、
  279^8 不建全表）；N=4 与旧 eager 表等价认证（B0-G0）；N=8/R={1,2,4,8}
  在线规划（B0-G1）；**header b_h 激活 cross-level 相变**（b_h=0 probe →
  b_h≥4 direct jump；B0-G2）；协议公平 baseline（含 **Direct-8 Ordered**）
  与 break-even 理论 q<(r''−r')/(b_h+r''−r')（B0-G3/G4）。

## 目录结构

```
Exp-1/
├── SystemModel.md          # 系统模型文档（只读参考）
├── advice/001.md           # 交叉审计意见（R1 依据）
├── run_mvsa.py             # v0 流水线（diagnostic，G2/G3/G4 审计后 REOPEN）
├── run_mvsa_r1.py          # R1 流水线：目标一致的有约束策略审计
├── run_mvsa_r11.py         # R1.1+R2 流水线：CMDP LP oracle + RBL
├── run_mvsa_r21.py         # R2.1 流水线：column generation 证书 + receding RBL + online solver
├── run_mvsb0.py            # MVS-B0 流水线：sparse backend + header 相变 + 协议公平 baseline
├── run_mvsb01.py           # MVS-B0.1 流水线：可信度修复 + VoI 定理 + Adaptive Direct-8
├── run_mvsb01a.py          # MVS-B0.1a 补丁：seed-aware POTS + ΔQ 相变 + 保守 lattice
├── run_mvsb03.py           # B0.3 CR-RBL：认证 rollout 规划器（主交付）
├── run_mvsb03a.py          # B0.3a/B0.3c credibility + closure patch（paired CRN + STOP 证书 + 硬预算 + b⋆ 相变）
├── run_mvsb04.py           # B0.4 pairwise-difference EB-CS planner（主算法升级）
├── run_mvsc01.py           # MVS-C C0+C1：semantic registration + link-aware phase theorem 验证
├── run_mvsc02.py           # MVS-C C2：phase-guided policy + exact budgeted-CMDP oracle（Gate D）
├── run_mvsc021.py          # MVS-C C2.1：budget-aware closure（matched ρ-homotopy + MITM + Gate D1/D2）
├── run_mvsc03a.py          # MVS-C C3a：migration gate（复现 G2 anchor）+ contract hardening（Myopic-PJ/StaticProg/convex-hull/exhaustive 4-bit certificate）
├── run_mvsc03b.py          # MVS-C C3b：五方法四层因果对照（Phase-PJ/Myopic-PJ/Myopic-All/Direct8/StaticProg）
├── run_mvsc03c.py          # MVS-C C3c：三层 feasibility frontier（C3d 修正：L1 constructive certificate / L2 per-method hull + 显式 mixture / L3 StaticProg 0-7 unique）
├── run_mvsc03e.py          # MVS-C C3e：Generalized Phase-Envelope（G0 activation audit + G1 generalized r<s<t 定理 + G2 GPE-EA vs Myopic-All matched Gate + G3 paired EB UCB）
├── run_mvsc04.py           # MVS-C C4：Link-Aware Heterogeneous U2U Airtime（τ_i=b0,i+κ_i(r'−r)，positive/independent/anti-correlated 三 regime，GPE-EA-het vs Myopic-All-het matched-action 认证 + anti 重路由机制报告）
├── run_mvsc05.py           # MVS-C C5：Protocol Robustness（p_succ ARQ-collapsed/control overhead 成本 stress + Δγ calibration mismatch + ρ evidence correlation；含 B1/B2 explicit 等价验证、mismatch 证书剪枝保真度审计）
├── test_regressions.py     # 正式 regression test suite（T01-T60，all checks PASS）
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
│   ├── sparse.py           # MVS-B0: sparse tuple-state planner（279^8 不建表）
│   ├── rbl_cr.py           # B0.3/B0.3c: CR-RBL（LatentWorld paired CRN + STOP 证书 + exact_qa_pi_b）
│   ├── rbl_eb.py           # B0.4: pairwise-difference EB-CS planner（betting CS + candidate-challenger）
│   ├── phase_boundary.py   # B0.4b/B0.7/C3e-G1: 反馈粒度相变定理 + generalized r&lt;s&lt;t envelope（010 §七）
│   ├── eval_exact.py       # R1: 精确前向概率传播 + G1a/G1b + 精确 P_D,max
│   ├── baselines.py        # B0–B11 + 公平基线精确 table-policy 构建
│   ├── mc.py               # 向量化 Monte Carlo + 随机化 Neyman-Pearson 评估
│   └── gates.py            # G0/G1/G2 Gate 检查
└── report/
    ├── MVS-A_report.md     # v0 诊断报告（G2/G3/G4 FAIL，审计后 REOPEN）
    ├── MVS-A-R1_report.md  # R1 目标一致审计报告
    ├── MVS-A-R1.1_R2_report.md  # R1.1+R2 CMDP oracle 与 RBL 报告
    ├── MVS-A-R2.1_report.md     # R2.1 认证 CMDP + receding online RBL 报告
    ├── MVS-B0_report.md        # MVS-B0 sparse-state header 报告
    ├── MVS-B0.1_report.md      # MVS-B0.1 可信度修复 + 理论拔高报告
    ├── MVS-B0.1a_report.md     # B0.1a 补丁报告
    ├── MVS-B0.3_report.md      # B0.3 CR-RBL 认证 rollout 报告（主交付）
    ├── MVS-B0.3a_report.md     # B0.3a/B0.3c credibility + closure patch 报告
    ├── MVS-B0.4_report.md      # B0.4 pairwise-difference EB-CS planner 报告（主算法升级）
    └── figures/            # 量化器 / Pareto / 精度审计 / R1 / R2 / R2.1 图
```

## 运行

```bash
pip install -r requirements.txt

python run_mvsa.py           # v0 流水线（约 5–8 分钟）
python run_mvsa_r1.py        # R1 流水线（约 7–10 分钟）
python run_mvsa_r11.py       # R1.1+R2 流水线（约 10 分钟）
python run_mvsa_r21.py       # R2.1 流水线（约 10 分钟）
python run_mvsb0.py          # MVS-B0 流水线（约 6–15 分钟）
python run_mvsb01.py         # MVS-B0.1 流水线（约 20–30 分钟）
python run_mvsb01a.py        # MVS-B0.1a 补丁（约 1 分钟）
python run_mvsb03.py         # B0.3 CR-RBL（约 6–10 分钟，推荐）
python run_mvsb03a.py        # B0.3a/B0.3c credibility + closure patch（约 6–10 分钟，推荐）
python run_mvsb04.py         # B0.4 pairwise-difference EB-CS planner（约 5–8 分钟，主算法升级）
python run_mvsb04a.py        # B0.4a Certified Policy Improvement（约 8–15 分钟）
python run_mvsb04b.py        # B0.4b phase-transition 定理封板（约 10 秒）
python run_mvsb06pre.py      # B0.6-pre sample-complexity gate（约 4–15 分钟）
python run_mvsb06prer.py     # B0.6-pre-r credibility patch（014，约 5–20 分钟）
python run_mvsb06.py         # B0.6 matched-QoS gate（约 25–40 分钟）
python run_mvsb06.py --map   # B0.6 + b_setup regime map（secondary，约 30–50 分钟）
python run_mvsb07g2.py       # B0.7-G2 separately calibrated QoS-dual certification（017；FULL 冻结 N_CAL=600/N_TEST=1600，约 15–30 分钟）
python run_mvsc01.py         # MVS-C C0+C1 semantic closure + link-aware phase theorem 验证（001 §二十六.0-1，约 <1 分钟）
python run_mvsc02.py         # MVS-C C2 phase-guided policy + exact budgeted-CMDP oracle（Gate D；FULL 约 8 分钟）
python run_mvsc02.py --smoke # C2 冒烟（约 1.5 分钟）
python run_mvsc021.py         # MVS-C C2.1 budget-aware closure（advice/002.md：matched ρ-homotopy + MITM + Gate D1/D2；FULL 约 16–30 分钟）
python run_mvsc021.py --smoke # C2.1 冒烟（约 2 分钟）
python run_mvsc03a.py         # MVS-C C3a Migration Gate + contract hardening（005.md：复现 G2 anchor + Myopic-PJ/StaticProg + exhaustive 4-bit certificate；FULL 约 8–10 分钟）
python run_mvsc03a.py --smoke # C3a 冒烟（约 1 分钟）
python run_mvsc03b.py         # MVS-C C3b Causal Four-Layer Comparison（005.md §十八：Phase-PJ/Myopic-PJ/Myopic-All/Direct8/StaticProg 四层对照；FULL 约 11 分钟）
python run_mvsc03b.py --smoke # C3b 冒烟（约 1.5 分钟）
python run_mvsc03c.py         # MVS-C C3c Three-Layer Feasibility Frontier（C3d 修正：L1 constructive certificate / L2 per-method hull + explicit mixture / L3 StaticProg 0/7 unique；FULL 约 9 分钟）
python run_mvsc03c.py --smoke # C3c 冒烟（约 1 分钟）
python run_mvsc03e.py         # MVS-C C3e Generalized Phase-Envelope（G0 activation audit + G1 generalized r<s<t 定理 + G2 GPE-EA vs Myopic-All matched Gate + G3 paired EB UCB；SMOKE 约 1–2 分钟、FULL 约 15–30 分钟）
python run_mvsc03e.py --smoke # C3e 冒烟（约 1–2 分钟）
python run_mvsc04.py         # MVS-C C4 Link-Aware Heterogeneous U2U Airtime（τ_i=b0,i+κ_i(r'−r)，positive/independent/anti 三 regime，GPE-EA-het vs Myopic-All-het matched-action 认证 + anti 机制报告；FULL 约 20–30 分钟）
python run_mvsc04.py --smoke # C4 冒烟（约 2–3 分钟）
python run_mvsc05.py         # MVS-C C5 Protocol Robustness（p_succ ARQ-collapsed/control overhead + Δγ mismatch + ρ correlation；B1/B2 等价验证 + 证书剪枝保真度审计；FULL 约 20–35 分钟）
python run_mvsc05.py --smoke # C5 冒烟（约 4–5 分钟）
python test_regressions.py   # regression suite（约 6–8 分钟；运行结束打印 all checks PASS）
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
  经 advice/001.md 审计判定为指标口径问题（REOPEN）。
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

## 下一步（依据 advice/008/009.md 的顺序，MVS-A 封板）

MVS-A 的 G0/G1a/G1b/G2 + R2.1-G0..G4 全部通过，按 003.md 冻结，不再继续优化。
MVS-B0/B0.1/B0.1a/B0.3/B0.3a/B0.3c/B0.4/B0.4r/B0.4s/B0.4a/B0.4a-r/B0.4b/B0.6-pre/B0.6-pre-r/B0.6
已按 004/005/006/007/008/009/010/011/012/013/014 完成。审计 011 确认 B0.4a 的 CPI
落地（B0.5 移到 B0.6 之后）；012 完成 B0.4a-r credibility closure；013 完成 B0.4b
纯理论封板（**B0.4 系列结束**）；B0.6-pre/B0.6-pre-r 算法验收（P0 已修、工程验收
通过）；**B0.6 matched-QoS 终审（014 口径）：CR 未能在 b_setup=16 regime 下省 bits
（诚实 FAIL，详见上文 B0.6 bullet）**；**015 复核后该 FAIL 降级为 COMMON-THRESHOLD
BIT LOSS / MATCHED-QoS UNRESOLVED（B0.6-r：D8/POTS 的 QoS 从未被认证，matched
比较不成立，详见上文 B0.6-r bullet）**：

- **B0.4a/B0.4a-r/B0.4b/B0.6-pre/B0.6-pre-r/B0.6/B0.6-r**（已完成）：CPI（base-anchored +
  Formal/Operational 分离）、phase-transition 主定理（exact b⋆=7）、sample-complexity
  gate、matched-QoS 终审（CR QoS 达标但 bits 多花 6.9-18.7/episode，regime map 无
  crossover ⇒ 014 结论：direct 在该 regime 已近最优）、B0.6-r 口径纠偏（三方法
  QoS CI + FEASIBLE/INFEASIBLE/UNCERTAIN 分类 + system-level regime diagnostic +
  B0.6-d 成本分解）；
- **B0.5**：**Bellman sandwich** L_k≤V⋆≤U_k（修正 006 不等式为
  V_genie≤V⋆≤min{V^{π_b}, R_stop}），把证书从 Q^{π_b} 提升到 Q⋆——**B0.6-r 显示
  当前算法在 b_setup=16 下有 raw bit loss（但 matched-QoS 未决），B0.5 的证书提升
  暂缓**（除非换 regime/算法；015 §十四：若 B0.7 common-stop 仍输，则把 B0.5 换用途
  为 Direct8 近优下界证明 V_LB≤V⋆≤V^D8）；
- **B0.7-G1**（已完成，详见上文 G1 bullet）：held-out QoS-dual 认证——H=96
  matched-QoS PASS（双方 FEASIBLE、E[D]=−12.31 CI [−13.26,−11.36]）；H=48
  UNCERTAIN（双方 P_MD U95 略超 0.40，扩样可解）。
- **B0.7-G2（已完成，017 定义 + FULL 结果详见上文 G2 bullet）**：按 016
  §15-4 定义、017 具体化——FG/D8 **分别** calibration 各自 (ρ,η)（017 §五：
  N_CAL=600/hyp @ H=96、4ρ×7η=28 grid 冻结），test 完全 fresh（017 §六
  方案 A：**N_TEST=1600 一次性冻结**、paired CRN、one-sided paired
  Hoeffding + Wilson QoS；CPI OFF）；**H=96 primary：G2 PASS**（θ̂_FG=
  θ̂_D8=(256,0.8)、双方 FEASIBLE、E[D]=−5.33、U95=−1.17<0）、**H=48
  stress 同冻结 controller 也 PASS**（E[D]=−5.03、U95=−2.95<0）；
  θ̂_FG=θ̂_D8 ⇒ −5.33 归因于 admissible action space 差异（**限定于本次
  选出的等 θ 控制器，018 §七**；operating-point 通道为 0）。**结论**：granularity 在“各自最公平 controller”下仍有统计
  认证的通信收益（≈−5.3 bit/ep @ H=96 且 H=48 也稳定），**论文核心
  performance 主线闭环**（先证 granularity 本身成立，016 §6 顺序）；
- **B0.7-G2r（018 §十五 credibility closure，已完成）**：不改主 Gate 数值，
  只修文档/诊断——**G3 预注册符号修正（P0，018 §二）**；rho 明确为
  **conditional-error Lagrange 的 effective multiplier**（\barλ_M=λ_M/π_1、
  \barλ_F=λ_F/π_0；代码即 exact parameterization，无“乘 2”步骤，019 §2）；
  sensitivity
  改为 **challenger 集合扫描**（E_cal[B]<E_cal[B_theta-hat] 逐个分类，018 §四）；
  **D8 (256,0.8) vs (256,1.0) near-tie => theta-hat_D8 是 empirical minimizer**
  （018 §七）；删“gross forced cost 是 leakage 上界”暗示（018 §九，强证据
  移交 G1r-B）；“24 bits ⇒ 首决策”改为 **r_cur=0 fresh UAV + F 决策索引
  统计**（018 §十）；论文句改 **relative to the calibrated Direct8
  controller**（018 §八）；“−5.33 全部来自 granularity”限定为 **for the
  selected equal-theta controllers in this run**（018 §七）。runner 同文件，
  报告重生成（数值与 G2 一致）。
- **B0.7-G3（DualCPI）＝ SUSPENDED（001 §二十五）**：双 Gate 预注册文本
  （019 §6-§9，含 Gate A/B、δ_G3 effect-size、主比较 FG_base↔FG_DualCPI）
  **存档保留**，仅在未来换 regime 且需要 certified planning 时启用，
  **不进当前路线**（001 §二十五：planner 不是瓶颈，不值得为它堆算力）。
- **MVS-C C0+C1（依据 advice/001.md §二十六.0-§二十六.1 先行验证）**：
  semantic closure + link-aware phase theorem 验证模块 `run_mvsc01.py`
  （独立于 G2 runner，不改 G2 数值）——**C0 文本断言 7/7 PASS**（README/
  runner 已含 matched detection QoS `P_FA≤α ∧ P_D≥P_D,max(α)−ε_D`、
  link-aware cost `c_{i,r→r'}=b_{0,i}+d_i(r,r')`、frame-window hard budget
  `C_max^{frame}` 口径；`_decode_zs` 改用 `pl.N`（001 §十九.2，runner
  已同步修复硬编码 N_UAV），belief canonical z-state / log-sigmoid 登记）；
  **C1 数值验证 PASS**（五分布，情形 A/B/C 全分支 + κ=1 退化复现 013 原定理：
  b* 解析=暴力扫描、∂⁺g(b)≡P(Y>b) 成立；报告 `report/MVS-C_C0C1_validation.md`）。
- **MVS-C C2（依据 advice/001.md §二十六.2-§二十六.3 / §九-§十四 / §二十七 Gate D）**：
  **phase-guided conditional-refinement policy（N=4 先行）+ exact budgeted-CMDP
  oracle**（`run_mvsc02.py`，SMOKE+FULL 双报告）。**理论 Gate 全 PASS**：T1
  支撑恒等式（013 §1 on reachable states，327 检查 0 偏差）、T2 cond-refinement
  sandwich（Q^{global-2} ≤ Q^{self-2} ≤ Q^{prog} ≤ Q^{(1)}，310 检查 0 矛盾）、
  T3 theory-certified pruning 自洽（g≥0 ⇒ Q_prog≥Q_dir，327 检查 0 矛盾）、
  T4 复杂度 O(2N)（每决策评估动作 max 8 ≤ 2N，full-FG max 16 = N|R|）。**oracle
  粒度可行性**：8-bit exact backward 不可行（reachable z-state 估计 **279⁴ ≈ 6.06e9** @
  H=96，与 8-bit 精确卷积参考 256⁴ 同源爆炸）→ Gate D oracle 冻结在
  **{1,2,4} 粒度**（23⁴=279,841，MVS-A ExactDP 规模，14–15s/θ 可解）。**双口径
  机制比较**：(a) **matched（001 §三）机制层不可达**——α=0.05 边（U_FA≤0.05）
   下该 dual 族冻结网格上最佳 P_D 仅 ~0.74-0.76（FULL cal (1024,1.8)
   U_MD=0.2836），目标 P_D≥0.8382（β=0.1618）差 ~0.08-0.10，源于
   **stopping-budget 权衡**（H=96 下 4×8-bit direct=96 即全预算，π* 同 θ 也仅
   E[B]≈44.5≈2.8 tx，不以全融合为最优，见 Gate D 行）——**早先“ε_D 余量被 CI
   消耗”是错误机制（C2 review 撤销）**，不可达是机制/网格层而非统计层；
   matched 在该规模下如实 INFEASIBLE）；
  **注（C2.1 修正，002 §二）**：该定性只对**注册冻结族/网格**成立（matched
  63-θ ρ-homotopy 无可行点）——**primal feasibility 已由 π_full 显式构造证明**
  （全部 8-bit direct，C=96≤H=96，P_D=P_D,max^8b(0.05)=0.8509≥0.8382），
  不再写"机制层不可行"；见下 C2.1 bullet。
  
  
  (b) **legacy mechanism（017 §四）全部 FEASIBLE**：
  @H=96 的 E[B]：**Phase-FG(8-bit)=27.66、Direct8=29.17（paired E[D]=−1.51，
  Hoeffding U95=4.37 未过 0 → 方向性未认证）、Myopic-FG=22.35（Phase-FG 比
  Myopic 贵 +5.31，未认证）**——granularity-vs-D8 的点估计方向在 N=4 成立但
  统计未认证（C2 诚实口径；机制主比较留 C4/N=8）。**Gate D PASS**：Phase-FG
  与 exact CMDP* 通信成本在全部冻结 θ 下 rel 最大 +1.27%（H=48 行，≤10% 预注册
  阈值）；H=96 行 rel 为负（−4.7…−13.5%）是 **dual trade**（Phase-FG 以更高终端
  risk 换更少 bits，Lagrangian 最优性由 V_lag 保证）——求解器质量认证成立，
  **不再做 CPI**（001 §二十七）。报告：`report/MVS-C_C2_report.md`（FULL）、
  `report/smoke/MVS-C_C2_report.md`。
- **MVS-C C2.1（依据 advice/002.md §二-§八/§十四，Budget-Aware Theoretical/Credibility Closure）**：
  002 审计的 **4 项 P0 全部闭环**（C2 记录保留不动）：
  **(1) hard-budget constrained pruning 修复（002 §四/§五）**：prune probe 安全条件改为
  **prune ⟺ g≥0 ∧ c_dir≤h**——region A（c1≤h<c_dir）时 direct 不可行、probe 是**唯一可行
  结构动作绝不剪**（002 §四 反例 PASS：N=1 fresh/h=20、c1=17、c_dir=24）；建资源窗三区域
  law：A（probe 唯一）/B（c_dir≤h<c1+c2，Q_prog=c1+E[R(X1)]）/C（h≥c1+c2，原 E[min{Y,b0}]）
  ——reachable 263 检查 0 偏差、constrained-pruning 自洽 0 矛盾。
  **(2) Gate D 拆分（002 §三）**：D1 solver-quality（Lagrangian Jθ=E[B]+E[Rθ(xτ)]
  vs exact Vθ*，Δ_J 最大 **1.89% ≤ 10% PASS**，取代裸 E[B] Gate——负 rel 只是 dual trade）；
  D2 primal E[C]（matched 双方 FEASIBLE 才比较）。
  **(3) matched 定性修正（002 §二）**：网格失败≠机制不可行——**π_full 显式可行构造**
  （全 8-bit direct，C=96≤H，P_D=P_D,max^8b(0.05)=**0.8509**≥0.8382 ⇒ matched primal 可行）；
  ρ-homotopy（128…8192）扫描仍无 matched 点 → 定性 **registered-grid infeasible**
  （不再写机制层不可行）。
  **(4) MITM 精确 8-bit 全融合 ROC（002 §七）**：256²+suffix merge、O(256² log)
  非 256⁴；P_D,max^8b(0.05)=0.8509（η*=0.8117）→ BETA8=0.1591（=1−0.8509+ε_D）；分层抽样 n0=n1（002 §八）。
  报告：`report/MVS-C_C21_report.md`（FULL）、`report/smoke/MVS-C_C21_report.md`。
  **FULL 主结果（N_CAL=400/N_TEST=800/hyp、757.6s）**：**Gate D1**（Lagrangian
  Jθ=E[B]+E[Rθ(x_τ)] vs exact Vθ*）**Δ_J max 1.89% ≤ 10% 全非负 → PASS**（正式
  solver-quality 证书；C2 的裸 E[B] Gate 正式降级为 provisional，负 rel 只是
  dual trade）；**D2** 裸 E[B] 仅作 primal 参考（matched 双方 FEASIBLE 才判决）。
  matched 63-θ ρ-homotopy 仍无网格可行点（定性 **registered-grid infeasible**；
  primal 由 π_full 构造证明）；legacy 全部 FEASIBLE：Phase-FG(8-bit) E[B]=27.570
  vs Direct8 29.175（paired E[D]=−1.605、U95=4.270 未认证方向性）、Myopic 22.027
  （更贵、未认证）、Phase-FG(4-bit) 24.349（β=0.40 下粗粒度更省）。
- **MVS-C C2.1a（依据 advice/003.md，credibility patch）**：003 审计的 4 项 P0/P1 全部闭环：
  **(1) Region-B pruning 判据修正（003 §2-§4）**：B 区 gap=(Q_prog−Q_dir)=E[Y]（tower，第二包不可行时 counterfactual 经边际 tower 进入），C 区 gap=E[min(Y,b)]；prune ⟺ dir_feas∧gap≥−tol；A 区**绝不剪probe**（003 §4 反例显式 assert g0≥0，PASS）。
  **(2) dominance-safety（003 §5）**：真正证书 prune ⟹ Q_prog ≥ Q_dir − ε，reachable 263 检查 + A 区不剪断言，0 矛盾 PASS（取代旧的 g0↔prune 自洽自检）。
  **(3) matched §3.4 统计修正（003 §6）**：min U_MD 限定 U_FA≤α 子集 → U_FA≤α 边最优点 (ρ=8192,η=1.6) U_MD=0.2446、与 β8=0.1591 的**真实差距 0.0855**（不再写全网格 min 0.1638 的 0.0047 假象）。
  **(4) regression 落库（003 §8）**：**T33–T39 入 test_regressions.py**（MITM==brute、Region A/B/C law、dominance、stratified n0=n1、J≥V*），**53/53 PASS**；P2：65536 typo、P_FA≤α（det-thr）措辞、duplicate orderB 删除、Gate D1 Δ_J max 1.89%→1.88%（003 §9 预期下降兑现）；FULL runtime 757.6s→642.1s。报告：`report/MVS-C_C21_report.md`（FULL）、`report/smoke/MVS-C_C21_report.md`。
  **[C2.1b（004 审计，独立路径对照修复）]**：区域恒等式门原为恒真式（C 区 gap 与 g_verdict 是同一变量、B 区代数恒等，永远不会失败）。修复：B 区 `phase_support_budget` 现在也计算 per-branch counterfactual E_R=E[R(X2)|X1]（仅作审计，不进入 Q_prog），g_verdict=Σw·(R1−E_R−d2) 独立于 gap 的边际 tower 路径；C 区 g_verdict=Q_prog−Q_dir 独立于 gap 的 support 形式——两侧来自不同代码路径，|gap−g_verdict|≤1e-9 可真实失败（sabotage 验证：Q_prog+5 / E_dir+5 / E_R 偏差均被捕获）。同时修复 main() 中 N=1 反例 g0_chk 的索引 bug（用 br[6]=Y=R1−d2，非 br[5]=D=R1）。T35/T36 同步为真实检查；T34 注释算术修正（c1+c2=36）。53/53 PASS；FULL/SMOKE 重跑数值与 C2.1a 逐项一致（FULL runtime 为墙钟、因机器而异：提交版 642.1s，后续重跑 665.6s/739.2s 等——**以报告"总耗时"为准**），仅恒等式门描述更新。
  **已完成（MVS-C C0→C2.1，见上各 bullet）**：C0 Specification/Semantic
  Registration（005 §三 改名：文档/runner 关键字登记，非 implementation
  closure）、C1 link-aware phase theorem 数值验证（**A/B/C 命名与
  phase_boundary.py 统一，005 §四**）、C2 phase-guided policy + 4-bit
  exact-oracle Gate D（已由 C2.1 修正为 D1/D2）、**C2.1 budget-aware
  closure（002.md）**、**C2.1b identity-gate 独立路径对照（004.md）**。
  **下一步（002 §十一 + 005 §十七，MVS-C 后续）＝ 修正路线**：
  - **C3a — Migration Gate（已完成，005 §十七）**：新架构 budget-aware
    Myopic-FG（Myopic-All，one-step QoS-dual，A={1,2,4,8}）在 N=8
    homogeneous 下**逐项复现 B0.7-G2**（θ̂_FG=θ̂_D8=(256,0.8)、H=96
    E[D]=−5.3250、U95=−1.1710、双方 FEASIBLE → **PASS**；H=48 E[D]=−5.0263、
    U95=−2.9493 也复现）。controller 等价性审计：run_mvsc021 决策族 vs G2
    q_min_fg/q_min_d8（开发期 300 episodes 审计 + T40 回归 40 episodes × 3
    corners × H∈{48,96}，FULL seeds 复算 max|Δ(cost,N_tx,payload)|=0）。**Contract hardening 同步落地**：
    (H1) **Myopic-PJ**（A={next,full}，one-step——与 Phase-PJ 同动作集，
    消除旧 Phase-vs-Myopic 动作空间混杂，005 §七）；(H2) **StaticProg**
    （固定 SNR ladder，Gate B 主基线回归，005 §八）；(H3) A/B/C 命名统一
    phase_boundary 为准（005 §四）；(H4) **P_D,max^det-thr** 标注（005 §九：
    离散 LLR 下确定性阈值 ≠ 严格 randomized-NP）；(H5) **policy-mixture/
    convex-hull 诊断**（deterministic-grid vs policy-class feasibility
    分离，005 §六）；(H6) **4-bit N=4 exhaustive budget-reachable
    dominance-safety certificate**（BFS 607121 个 (x,h) 对、432644 支撑、
    B/C 区 0 矛盾——**并暴露并修复 d2=0 退化 bug**：r_next==r_max 时
    probe==direct，Q_prog 必须=Q_dir、gap=0，T42；A 区在 N=4/H=96 数学
    上不可达，靠 N=1 反例 + “绝不剪”断言覆盖，005 §十）。报告：
    `report/MVS-C_C3a_report.md`（FULL）、`report/smoke/MVS-C_C3a_report.md`。
  - **C3b — Algorithm（已完成，005 §十八 四层因果对照）**：五方法 separately
    calibrated（θ̂ 均=(256,0.8)）+ paired CRN + Hoeffding/Wilson，四层对照：
    (1) **Phase-PJ ≡ Myopic-PJ**（同动作集 {next,full}，θ̂ 下 D=0.000，T43 锁定；
        conditional-refinement planning value ≈ 0——但 (1024,2.0) 极端点有
        4/40 差异且 Phase 更省，refinement 有 bite 只在 θ 边角）；
    (2) **Phase-PJ vs Direct8**：E[B] 34.49 vs 37.52，D=−3.03（U95=1.12，
        UNRESOLVED——方向对但 Hoeffding 界未认证）；
    (3) **Phase-PJ vs StaticProg**：StaticProg 无 FEASIBLE θ̂（固定顺序简单
        ladder 无法同时满足 α=0.12/β=0.40，本身即 adaptive 必要性证据）；
    (4) **Phase-PJ vs Myopic-All**：34.49 vs 32.20，D=+2.30（Myopic-All 更省，
        UNRESOLVED）。**结论**：Myopic-All（A={1,2,4,8} 全跳）< Phase-PJ <
    Direct8（32.20 < 34.49 < 37.52）；StaticProg 语义修正为 |Ω|≥η 渐进
    ladder（007 审计：消除 QoS-dual R≤min Q 的 root 全停退化，T44 锁定）。
    报告：`report/MVS-C_C3b_report.md`（FULL）、
    `report/smoke/MVS-C_C3b_report.md`。
  - **C3c — Feasibility frontier（已完成；C3d 依 advice/010.md §三-§六
    修正 L2/L1/StaticProg 口径）**：三层：**L1 Constructive Physical
    Feasibility Certificate**（010 §六 改名：构造型最大 evidence 配置
    4 强 SNR UAV 8-bit cost 96=H、MITM ROC P_MD=0.1072≤0.40；4 弱 UAV
    0.3633 也达标——PASS；**构造 FAIL 不蕴含 physical infeasible**，需
    budgeted max-evidence oracle）；**L2 per-method registered convex hull**
    （010 §三 P0 修正：对**每个方法自身** conv{v_θ^m: θ∈Θ_m}，旧版跨方法
    混合且直接排除无 θ̂ 的 StaticProg 是逻辑缺口；deterministic FEASIBLE 数
    + 自身网格 2-point mixture（fractional Wilson U95 仅作近似证据）+ 在
    **全新 test worlds 上显式 Bernoulli-λ mixture**（整数 kfa/kmd、Wilson
    n0/n1 分离）正式认证（010 §四））；**L3 Controller-search**：各方法
    自身 FEASIBLE 数 + θ̂，**StaticProg 改 0/7 unique 口径**（010 §五：ρ
    不参与策略 ⇒ 4ρ 重复 ⇒ 7 个唯一阈值策略），撤掉"无可行点本身即
    adaptive 必要性证据"表述。报告：`report/MVS-C_C3c_report.md`（FULL）、
    `report/smoke/MVS-C_C3c_report.md`。
  - **C3d — 并入 C3c runner（010 §十二：P0 立即，已完成）**：n0/n1 分离
    （eval_decide 返回 n1、kmd 用 n1 做 Wilson 分母，T47）、L1 改名
    constructive certificate、per-method hull（T51 单元测试）、StaticProg
    7-unique 口径（T48）。
  - **C3e — Generalized Phase-Envelope Evidence Acquisition（已完成，
    advice/010.md §七-§十二）**：
    - **G0 Phase activation audit**：θ̂=(256,0.8) 冻结下 Phase-PJ vs
      Myopic-PJ——**action-change rate = 0**（两方法决策处处一致，010 §一
      结论复现）；P(Q_phase≠Q_myopic)≈0.61 仅 Q 值层激活、probe 76% 被
      pruning（010 §二 机制解释：差异被剪/不改变 argmin）；
    - **G1 generalized r<s<t envelope 定理（010 §七）**：link-affine
      c_i(r→q)=b0+κ(q−r) 下 Q_prog^{s,t}−Q_dir^t == E[min{Y_{i,s,t}, b0}]，
      Y=R(X_s)−E[R(X_t)|X_s]−κ(t−s)；Gate 全 PASS：identity 1.75e-13、
      tower 1.71e-13、derivative=survival 6.98e-12、b* 三情形 1313/1313、
      与 c21.phase_support_budget 的 (next,max) 特殊情形 440/440 一致
      （T46）；开发中揪出 generalized support 7-tuple 的 Y 索引 bug
      （idx 6 而非 5）——G1a 由 14.0 修正到 1.75e-13；
    - **G2 matched-action 论文 Gate（010 §八/§十二）**：新 Proposed
      **GPE-EA** = full action set（与 Myopic-All 相同 A={(i,s): s>r_i}）
      + conditional-refinement Q（probe 用 c(r→s)+E[min{R(X_s),
      min_t(c(s→t)+E[R(X_t)|X_s])}]，certificate 证明全 continuation 被
      支配时精确退化为 one-step，T50）；separately calibrated、paired CRN、
      fresh test（G2 017 协议）；SMOKE 下 Myopic-All 无 FEASIBLE θ̂ ⇒
      QoS-UNRESOLVED（cal worlds 不足；**FULL N_CAL=600 才是主判定**）；
    - **G3 paired fixed-N empirical-Bernstein UCB（010 §十）**：MP Thm 4
      plug-in variance、t=log(1/δ)、(n−1) 保守分母——G2 主 bit 认证；
      Hoeffding（D∈[−H,H]）为 sanity envelope。
    **P0 修复（用户审计 + T56 锁死）**：`_cond_refine_q` 的 memo key 曾
    漏 (ρ,η,b0,κ)——28 组合校准共享 memo ⇒ 后续组合拿到首个 (ρ,η) 的陈旧
    Q_cond（实测 Q(128,0.8)=81.0 泄漏给 Q(1024,2.0) 应为 529.0），系统性
    低估 probe Q ⇒ GPE-EA 过度"继续买证据"（旧报告 H=96 E[N_tx] 2.50 vs
    1.66、+12.6 bits 正是此污染的方向）。修复：Q 层 key 补全 (x,i,s,conts,
    ρ,η,b0,κ)；**结构层缓存**（数理依据：om 是状态 x 的唯一函数、转移权重
    只依赖状态与量化器，均与 (ρ,η,b0,κ) 无关）按 (x,i,s,conts) 缓存
    (w,om2) 传播序列——修复后校准不再重复算权重。**FULL 重跑决定性结果：
    GPE-EA ≡ Myopic-All**（θ̂ 均 (256,0.8)；P_FA U95 0.0896/0.0896、P_MD
    0.3121/0.3121、E[N_tx]、E[B_payload]、E[B|H0/H1] **逐位一致**；
    E[D]=0.0000，EB U95=0.4195/0.2098 对称）——条件细化在均匀成本
    (b0=16,κ=1) 下与 one-step **决策级等价**（T50 certificate 全支配
    剪枝 ⇒ Q_cond≡q1，定理级而非数值巧合）；010 §一/§十二 的判断成立：
    refinement 价值只在 C4 异质链路成本下才能体现。
    报告：`report/MVS-C_C3e_report.md`（FULL 已生成，含 P0 修复后数值）。
  - **C4 — Link-Aware Heterogeneous U2U Airtime（已完成，010 §九/§十二；论文 headline）**：
    成本模型从 homogeneous "16+Δr bits" 升级为 **per-UAV airtime**：
        τ_i(r→r') = b0,i + κ_i·(r'−r),
    b0,i ≡ τ_ctrl,i（该 UAV 一次 transaction/control/setup airtime）、
    κ_i ≡ 1/R_i（每 evidence bit 的 airtime）；hard budget Στ ≤ H（airtime
    单位）。三 regime（001 §十六）：**positive**（强 sensing = 好链路）、
    **independent**（无关联）、**anti-correlated**（强 sensing = 坏链路，
    001 §十六 最重要的机制实验）。链路质量 q_i ∈ [0,1] → b0,i=12+8(1−q)、
    κ_i=0.8+0.4(1−q)（homogeneous (16,1) 是 q≡0.5 特例，T54 断言 GPE-het
    ≡ GPE）。
    新 Proposed = **GPE-EA-het**：full action set（与 Myopic-All-het **相同
    动作空间**），probe 用 conditional-refinement Q（per-UAV b0,i/κ_i 直接
    进入 010 §七 的 generalized envelope）——唯一差别仍是 conditional-
    refinement planning。G2 协议同 C3e-G2/G3（separately calibrated、
    paired CRN、fresh test、paired EB UCB 主认证 + Hoeffding sanity +
    Wilson n0/n1）。
    **FULL 结果（N_CAL=600/N_TEST=1600/hyp；P0 修复后重跑——决定性）**：
      - **三 regime 全部 GPE-EA-het ≡ Myopic-All-het 逐位一致**：θ̂ 均
        (256,0.8)，P_FA U95、P_MD U95、E[N_tx]、E[B_payload]、E[B|H0/H1]
        完全相同；E[D]=**0.0000**（EB U95=0.4195/0.2098 对称、Hoeffding
        sanity 4.1540/2.0770）→ **BIT-UNRESOLVED（零差）**。修复前报告的
        +13.81/+8.94/−0.13 与"GPE-EA-het 更贵/更省"全部是 P0 污染的产物。
      - **anti-correlated 机制（诚实，与 001 §十六 简单预期相反）**：
        corr(E[N_tx,i], κ_i)=**+0.74（正相关）**——planner 仍把 budget 集中
        在最强 sensing 的坏链路 UAV7（γ=3dB、b0=20、κ=1.2，约占 airtime
        70.8%），好链路弱 sensing UAV0–3 几乎不用（E[N_tx,i]≤0.005）。
        关键：**matched-QoS（P_MD≤0.40）下弱 sensing 好链路 UAV 组合无法
        FEASIBLE**——sensing QoS 可行性约束**优先于**链路成本（且该机制对
        GPE 与 Myopic 同样成立，与规划深度无关）→ **anti-correlation 不
        自动诱导重路由**（001 §十六 的机制在本注册刻度下被反例否证，诚实
        报告；010 §十二 下一步 C5 protocol robustness）。
    runner：`run_mvsc04.py`；报告：`report/MVS-C_C4_report.md`、
    `report/smoke/MVS-C_C4_report.md`。回归：T53（link_params 三 regime
    语义）、T54（homogeneous 极限 GPE-het ≡ GPE）、T55（het budget
    恒等式 B=Σ(b0_i N_tx,i+κ_i pay_i) ≤ H pathwise）。
  - **C4 — Link-Aware Heterogeneous U2U Airtime（已完成，010 §九/§十二；
    论文 headline，002 §十二/001 §十六）**：成本模型升级为 per-UAV
    airtime `τ_i(r→r')=b0,i+κ_i(r'−r)`（b0,i≡τ_ctrl,i、κ_i≡1/R_i），硬约束
    Στ≤H；新 Proposed **GPE-EA-het**（full action set + conditional-
    refinement Q，per-UAV b0,i/κ_i 直接进入 010 §七 generalized envelope）
    vs **Myopic-All-het**（相同动作空间、相同成本模型）matched-action 对比
    （各自 (ρ,η) 校准、paired CRN、fresh test、paired EB UCB 主认证 +
    Hoeffding sanity + Wilson n0/n1，协议同 C3e-G2/G3）。三 regime：
    **positive / independent / anti-correlated**（强 sensing = 坏链路，
    001 §十六 最重要的机制实验）。anti-regime 额外报告链路重路由机制
    （per-UAV E[N_tx,i] 与 E[B,i] 占比 vs 链路质量：planner 自动把 budget
    从"强 sensing 但坏链路"UAV 转移到"好链路（哪怕中等 sensing）"UAV）。
    runner：`run_mvsc04.py`；报告：`report/MVS-C_C4_report.md`（FULL/SMOKE）。
  - **C5 — Protocol Robustness（已完成，001 §二十六.1；runner `run_mvsc05.py`）**：
    四类 stress 叠加到 C4 同一 matched-action 协议（anti regime、GPE-EA-het
    vs Myopic-All-het、separately calibrated、paired CRN、fresh test、paired
    EB UCB + Hoeffding + Wilson n0/n1）：
    - **(1) packet success（ARQ collapsed）+ control overhead**：期望重传成本
      c̄(Δr)=(b0+b_ctrl+κΔr)/p_succ **仍为 affine**（b0'=(b0+b_ctrl)/p_succ、
      κ'=κ/p_succ）⇒ 010 §七 envelope 精确保持（T57）；B1(collapsed) vs
      B2(explicit) 期望成本等价（viol=0，E[col]≈E[exp]，几何重传）；全
      p_succ/b_ctrl 组合 **E[D]=0.0000**（GPE-het ≡ Myopic-het 维持，bit 级）。
    - **(2) calibration mismatch Δγ**：planner 用 model（γ+Δγ）量化器/消息
      PMF/ℓ/证书、世界 true 采样（SystemModel §65 部署语义）；matched G2 仍
      E[D]=0（refinement 零价值维持），但 **证书剪枝保真度方向性退化**：
      Δγ=+3dB 假剪率 **29.1%**（高估 sensing → 过度剪枝）、Δγ=+1dB 9.4%；
      Δγ=−3dB 漏剪率 **40%**（低估 → 保守漏剪）、Δγ=−1dB 9.8%（T59 锁
      mismatch LLR 分离）——**P0 教训的平行面：模型参数错误 ⇒ 剪枝证书失真，
      部署需审计/保守化**。
    - **(3) evidence correlation ρ**：世界 common-factor 相关、planner 保持
      独立模型假设（T58）；ρ=0.3 时 GPE 略省（E[D]=−0.137）但 EB U95=0.74
      未认证、双方 QoS UNCERTAIN；ρ=0.6 无可行 θ̂——证据相关削弱融合
      （独立性假设被破坏的代价，诚实记录）。
    - **数理设计（P0 教训正面应用）**：决策侧参数（ρ/η/b0/κ/p_succ/b_ctrl）
      经 (b0',κ') 合成进入 memo Q-key（T60；跨组合无陈旧缓存）；世界侧参数
      （Δγ 换 planner 实例、ρ 只改采样）不进决策 memo。
    报告：`report/MVS-C_C5_report.md`（FULL 2783s）、`report/smoke/…`。
  - **B1**（fading/packet errors）仍最后；
  **论文四 Gate（001 §二十七，经 002 §三 修正为 D1/D2）**：A 数学正确性 /
  B 机制必要性（Phase-FG<Direct8 且 <Static Progressive）/ C 通信现实性
  （b_ctrl>0、p_succ<1、anti-correlation 下仍成立）/ D 求解器质量
  （**D1：Lagrangian Δ_J=E[B]+E[R_θ] vs V_θ* ≤ 10%**；**D2：matched QoS 下
  E[C] 比较，双方 FEASIBLE 才判**）。**不再扩散 G0/G1/G2… 的几十个 Gate。**
- **B1**（fading/packet errors）仍然最后。

关键算术修正（003.md §8）：MVS-B 每 UAV evidence states =
**1+2+4+16+256 = 279**（不是 47）；279⁸ ≈ 1e19 ⇒ **MVS-B 禁止复用全枚举 StateSpace**，
必须使用 R2.1-G3 的 sparse online planner。MVS-B 最重要的实验是
sensing/U2U **anti-correlation** regime（强 sensing ≠ 低通信成本）。

