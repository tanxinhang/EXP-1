> # O-PEF v2.1：优化驱动的渐进式证据报告与软融合
>
> **Optimization-Driven Progressive Evidence Fusion for Distributed Multi-UAV ISAC**
>
> ## ——最终冻结版系统模型、算法与验证方案
>
> ------
>
> # 1. 研究定位
>
> ## 1.1 核心科学问题
>
> 考虑由 (N) 架 UAV 构成的分布式 ISAC 感知网络。每架 UAV 已经完成本地信号处理，并形成关于目标存在性的局部软证据。
>
> 如果所有 UAV 都将最高精度统计量完整发送至融合 UAV，可获得较强检测性能，但 U2U 报告成本随网络规模增长。
>
> 另一方面，固定 Top-(K)、censoring 或 ordered transmission 通常主要解决：
>
> [
> \text{“哪个节点发送？”}
> ]
>
> 而本文进一步研究：
>
> [
> \boxed{
> \text{哪个 UAV}
> +
> \text{发送多少精度}
> +
> \text{什么时候发送}
> +
> \text{什么时候停止}
> }
> ]
>
> 即：
>
> > **在局部证据已经产生之后，根据当前实际收到的证据、尚未揭示的信息价值以及异质 U2U 通信成本，自适应决定下一单位通信资源应该用于哪个 UAV、提升多少证据精度，或者直接停止通信。**
>
> 因此本文本质上研究：
>
> [
> \boxed{
> \textbf{QoS-constrained adaptive multi-resolution evidence acquisition}
> }
> ]
>
> 而不是单纯 sensor selection，也不是 certificate-first detection。
>
> ------
>
> # 2. 选题原因与文献依据
>
> ## 2.1 多 UAV 协同检测需要软信息融合
>
> 多 UAV 空间分布使不同节点具有不同：
>
> - sensing distance；
> - aspect angle；
> - RCS；
> - clutter；
> - interference；
> - sensing SINR。
>
> 单节点可能只能获得较弱证据，但多个弱证据联合后可以形成可靠检测。
>
> 近期 Fuse-then-Detect 工作进一步说明，在弱 UAV echo 条件下，过早对单链路信息做 thresholding 可能损失可以通过多视角联合积累得到的证据。
>
> 因此本文保留：
>
> [
> \boxed{\text{soft evidence before final decision}}
> ]
>
> 作为系统基本原则。
>
> ------
>
> ## 2.2 All-neighbor fusion 带来通信扩展性问题
>
> 若所有节点均发送 (B_{\max})-bit evidence，则：
>
> [
> B_{\rm all}=NB_{\max}.
> ]
>
> 随着：
>
> [
> N\rightarrow32,64,\ldots
> ]
>
> 报告负担近似线性增加。
>
> 因此真正需要优化的是：
>
> [
> \boxed{
> \text{Detection value per communication resource}.
> }
> ]
>
> ------
>
> ## 2.3 经典 censoring 已解决“是否发送”
>
> Rago、Willett 与 Bar-Shalom 已研究 censoring sensors：
>
> [
> L_i\in\mathcal C
> \Rightarrow
> \text{silent}.
> ]
>
> 因此本文不能把：
>
> > “低价值 sensor 不发送”
>
> 作为创新。
>
> ------
>
> ## 2.4 Ordered Transmission 已解决“谁先发送”
>
> Blum 和 Sadler 的 ordered transmission 根据局部 detection evidence 强度决定发送顺序，并允许 fusion center 提前结束通信。
>
> 因此：
>
> [
> \text{ordering}
> +
> \text{early stopping}
> ]
>
> 本身也不是创新。
>
> ------
>
> ## 2.5 FE-SRTS 已进一步考虑反馈、量化、衰落与重传
>
> 近期 FE-SRTS 已将：
>
> - feedback；
> - reordered transmission；
> - quantization；
> - fading；
> - retransmission；
> - power control；
>
> 引入 distributed detection。
>
> 因此 O-PEF 不能把：
>
> [
> \text{feedback/quantization/retransmission}
> ]
>
> 单独包装成创新。
>
> ------
>
> ## 2.6 Successive refinement 与 online sensor selection 也已有理论基础
>
> 已有信息论工作研究 successive-refinement hypothesis testing。
>
> 同时，也已有工作把：
>
> [
> \text{online sensor selection}
> +
> \text{optimal stopping}
> ]
>
> 建模为 Bayesian dynamic programming。
>
> 因此 O-PEF 真正的研究空间不是上述单一机制，而是：
>
> [
> \boxed{
> \begin{aligned}
> &\text{post-observation UAV evidence}\
> +&\text{per-UAV multi-resolution evidence}\
> +&\text{adaptive UAV selection}\
> +&\text{adaptive precision selection}\
> +&\text{realized-message-dependent adaptation}\
> +&\text{heterogeneous U2U reporting cost}.
> \end{aligned}}
> ]
>
> ------
>
> # 3. O-PEF 与已有问题的核心区别
>
> 传统 Top-(K) 的决策变量：
>
> [
> x_i\in{0,1}.
> ]
>
> O-PEF 的基本动作：
>
> [
> \boxed{
> a=(i,r_i\rightarrow r_i')
> }
> ]
>
> 其中：
>
> [
> r_i'>r_i.
> ]
>
> 因此同一个 UAV 可以经历：
>
> [
> 0\rightarrow1\rightarrow4
> ]
>
> 而另一个 UAV 可能只有：
>
> [
> 0\rightarrow2
> ]
>
> 第三个 UAV：
>
> [
> r_i=0
> ]
>
> 始终不发送。
>
> 更重要的是，第二步是否发生取决于第一步实际收到的 evidence。
>
> 所以：
>
> [
> \boxed{
> \text{O-PEF不是静态bit allocation，而是反馈式信息获取。}
> }
> ]
>
> ------
>
> # 4. 第一篇论文的系统边界
>
> 第一篇主动收敛为：
>
> - 单目标；
> - 二元检测；
> - 多 UAV 分布式 sensing；
> - 一个临时 target-owner/fusion UAV；
> - U2U progressive evidence reporting；
> - 固定 waveform；
> - 固定 sensing power；
> - 固定 communication power；
> - 固定轨迹/位置；
> - 不做 target association；
> - 不做 MARL；
> - 不做 fully decentralized consensus。
>
> 基本链路：
>
> [
> \boxed{
> \text{ISAC echo}
> \rightarrow
> \text{local detector}
> \rightarrow
> \text{soft statistic}
> \rightarrow
> \text{multi-resolution message}
> \rightarrow
> \text{adaptive U2U reporting}
> \rightarrow
> \text{soft fusion}
> \rightarrow
> \text{decision}.
> }
> ]
>
> ------
>
> # 5. 基础检测模型
>
> 考虑：
>
> [
> H_0:\text{目标不存在},
> ]
>
> [
> H_1:\text{目标存在}.
> ]
>
> UAV (i) 获取：
>
> [
> Y_i.
> ]
>
> 第一阶段假设：
>
> [
> Y_i\perp Y_j|H_h,
> \qquad i\neq j.
> ]
>
> 局部连续 LLR：
>
> # [ L_i
>
> \log
> \frac{
> p_i(Y_i|H_1)
> }{
> p_i(Y_i|H_0)
> }.
> ]
>
> 如果所有连续 evidence 均免费获得：
>
> # [ \Lambda_{\rm raw}
>
> \sum_{i=1}^{N}L_i.
> ]
>
> 这定义最理想的：
>
> [
> \boxed{\text{Raw Full Fusion}}
> ]
>
> 但不是 O-PEF 必须达到的实际通信 reference。
>
> ------
>
> # 6. 三档检测性能参考
>
> 本版正式区分三个性能层。
>
> ## 6.1 Raw continuous reference
>
> [
> P_{D,\rm raw}.
> ]
>
> 所有 UAV 的连续 LLR 被完整获得。
>
> 它只表示统计模型允许的理想上界。
>
> ------
>
> ## 6.2 Maximum-message reference
>
> 设论文允许最大精度为：
>
> [
> r_{\max}.
> ]
>
> 所有 UAV 均发送最高精度 message：
>
> [
> M_i^{(r_{\max})}.
> ]
>
> 对应检测性能：
>
> [
> \boxed{
> P_{D,\max}.
> }
> ]
>
> 这是 O-PEF 真正应该匹配的 achievable full-report reference。
>
> ------
>
> ## 6.3 Adaptive policy performance
>
> 策略 (\pi)：
>
> [
> P_{D,\pi}.
> ]
>
> 正式 matched-performance 约束：
>
> ## [ \boxed{ P_{D,\pi} \ge P_{D,\max}
>
> \epsilon_D.
> }
> ]
>
> 推荐：
>
> [
> \epsilon_D=0.005\sim0.01.
> ]
>
> ------
>
> ## 6.4 单独评价量化损失
>
> 定义：
>
> # [ \boxed{ \Delta_Q
>
> ## P_{D,\rm raw}
>
> P_{D,\max}.
> }
> ]
>
> 这样：
>
> [
> \text{quantizer loss}
> ]
>
> 与：
>
> [
> \text{adaptive reporting loss}
> ]
>
> 被严格分开。
>
> MVS-A：
>
> [
> r_{\max}=4.
> ]
>
> MVS-B 与完整系统：
>
> [
> r_{\max}=8.
> ]
>
> ------
>
> # 7. 多精度 Evidence Message
>
> 定义：
>
> # [ M_i^{(r)}
>
> Q_i^{(r)}(L_i).
> ]
>
> 精度集合：
>
> ### MVS-A
>
> # [ \mathcal R_A
>
> {0,1,2,4}.
> ]
>
> ### MVS-B / Full System
>
> # [ \mathcal R_B
>
> {0,1,2,4,8}.
> ]
>
> 要求：
>
> [
> Q_i^{(1)}
> \prec
> Q_i^{(2)}
> \prec
> Q_i^{(4)}
> \prec
> Q_i^{(8)}.
> ]
>
> 即形成 nested partition。
>
> ------
>
> # 8. Message-LLR
>
> 融合 UAV 不把 quantizer midpoint 直接当作 LLR。
>
> 对于：
>
> [
> M_i^{(r)}=m,
> ]
>
> 定义：
>
> # [ \boxed{ \ell_i^{(r)}(m)
>
> \log
> \frac{
> P(M_i^{(r)}=m|H_1)
> }{
> P(M_i^{(r)}=m|H_0)
> }.
> }
> ]
>
> 若当前 UAV (i) 的 evidence precision 为 (r_i)，则当前 fusion statistic：
>
> # [ \Lambda
>
> \sum_{i:r_i>0}
> \ell_i^{(r_i)}
> (m_i).
> ]
>
> ------
>
> # 9. Progressive Refinement 的正确 LLR 更新
>
> 假设当前：
>
> [
> M_i^{(r)}=m
> ]
>
> 并进一步获得：
>
> [
> M_i^{(r')}=m',
> \qquad
> r'>r.
> ]
>
> 不能执行：
>
> # [ \Lambda'
>
> \Lambda+\ell_i^{(r')}(m'),
> ]
>
> 否则 coarse evidence 被重复计算。
>
> 正确更新：
>
> # [ \boxed{ \Lambda'
>
> ## \Lambda
>
> \ell_i^{(r)}(m)
> +
> \ell_i^{(r')}(m').
> }
> ]
>
> 这是整个 progressive fusion implementation 必须满足的基本 invariant。
>
> ------
>
> # 10. 正确的 Markov 状态
>
> 完整 history：
>
> [
> \mathcal H_t
> ]
>
> 不适合作为 DP 状态。
>
> 但：
>
> [
> (\Lambda_t,\mathbf r_t)
> ]
>
> 也不充分，因为下一 refinement 的条件分布取决于 UAV 当前所在 coarse cell。
>
> 定义：
>
> [
> z_i=
> \begin{cases}
> \varnothing,&r_i=0,\
> (r_i,m_i),&r_i>0.
> \end{cases}
> ]
>
> 于是：
>
> # [ \mathbf z
>
> (z_1,\ldots,z_N).
> ]
>
> 系统状态定义为：
>
> [
> \boxed{
> x=(\Omega,\mathbf z)
> }
> ]
>
> 其中：
>
> # [ \Omega
>
> \log
> \frac{
> P(H_1|\mathbf z)
> }{
> P(H_0|\mathbf z)
> }.
> ]
>
> 事实上 (\Omega) 可以由 (\mathbf z) 唯一计算，因此理论上：
>
> [
> \boxed{\mathbf z}
> ]
>
> 就是充分 Markov 状态。
>
> 实现中保存：
>
> [
> (\Omega,\mathbf z)
> ]
>
> 只是为了避免重复计算。
>
> ------
>
> # 11. Log-domain Posterior
>
> 定义 prior：
>
> [
> \pi_h=P(H_h).
> ]
>
> 当前 log posterior odds：
>
> # [ \Omega
>
> \log
> \frac{\pi_1}{\pi_0}
> +
> \Lambda.
> ]
>
> 数值实现不直接计算：
>
> [
> p=\frac1{1+e^{-\Omega}}.
> ]
>
> 而采用：
>
> # [ \log p
>
> -\operatorname{softplus}(-\Omega),
> ]
>
> # [ \log(1-p)
>
> -\operatorname{softplus}(\Omega).
> ]
>
> 所有：
>
> - likelihood；
> - posterior；
> - message probability normalization；
>
> 均使用：
>
> [
> \operatorname{logsumexp}.
> ]
>
> 避免：
>
> [
> \Omega\gg0
> ]
>
> 或：
>
> [
> \Omega\ll0
> ]
>
> 导致浮点下溢/饱和。
>
> ------
>
> # 12. Message State Transition
>
> 假设当前 UAV (i)：
>
> [
> M_i^{(r)}=m.
> ]
>
> 动作：
>
> [
> a=(i,r\rightarrow r').
> ]
>
> 对于与 (m) 一致的 child message (m')：
>
> # [ P(m'|m,H_h)
>
> \frac{
> P(M_i^{(r')}=m'|H_h)
> }{
> P(M_i^{(r)}=m|H_h)
> }.
> ]
>
> 当前 posterior 下：
>
> # [ P(m'|x,a)
>
> p(x)P(m'|m,H_1)
> +
> [1-p(x)]P(m'|m,H_0).
> ]
>
> 由此可以准确构造：
>
> [
> x\rightarrow x'.
> ]
>
> ------
>
> # 13. Evidence Model Calibration
>
> 定义 context：
>
> # [ \xi_i
>
> (
> \gamma_i^s,
> \text{clutter class},
> \text{geometry class},
> \text{detector mode}
> ).
> ]
>
> 融合 UAV 需要已知：
>
> # [ \boxed{ \theta_{i,h}^{(r)}(m)
>
> P(M_i^{(r)}=m|H_h,\xi_i).
> }
> ]
>
> MVS-A/B 中认为该模型已知。
>
> 完整 ISAC 系统采用：
>
> [
> \boxed{
> \text{offline/slow-timescale calibration}
> }
> ]
>
> 建立 lookup table。
>
> 不能要求 UAV 每一个 sensing frame 重新上传完整 PMF。
>
> 最终需要增加 calibration mismatch stress。
>
> ------
>
> # 14. Quantizer 设计
>
> 第一篇不优化 quantizer。
>
> 使用 mixture distribution：
>
> # [ p_{\rm mix}(L)
>
> \frac12p(L|H_0)
> +
> \frac12p(L|H_1).
> ]
>
> 构造 binary nested partition tree。
>
> 外部 cells 直接允许：
>
> [
> (-\infty,\tau_1],
> ]
>
> 和：
>
> [
> [\tau_K,+\infty).
> ]
>
> 不对 Gaussian LLR 做人为有限截断。
>
> 因此不设置：
>
> [
> P(\text{tail})<1%
> ]
>
> 之类不合理 Gate。
>
> Quantizer quality 直接通过：
>
> # [ \Delta_Q^{(r)}
>
> ## P_{D,\rm raw}
>
> P_{D,rbit}
> ]
>
> 评价。
>
> 可辅助报告：
>
> [
> I(H;M^{(r)}).
> ]
>
> ------
>
> # 15. 主优化问题
>
> 定义：
>
> # [ P_M^\pi
>
> 1-P_D^\pi.
> ]
>
> 正式问题：
>
> [
> \boxed{
> \min_{\pi}
> \mathbb E[C_\pi]
> }
> ]
>
> subject to：
>
> [
> P_{FA}^{\pi}
> \le
> \alpha,
> ]
>
> [
> P_M^\pi
> \le
> \beta_M.
> ]
>
> 第一阶段：
>
> [
> \alpha=0.05.
> ]
>
> 也报告 dual formulation：
>
> [
> \max_\pi P_D^\pi
> ]
>
> subject to：
>
> [
> P_{FA}^\pi\le0.05,
> ]
>
> [
> E[C_\pi]\le C_{\max}.
> ]
>
> ------
>
> # 16. 通信成本
>
> ## 16.1 MVS-A：纯 Payload Cost
>
> # [ \boxed{ c_a
>
> r'-r.
> }
> ]
>
> 不考虑：
>
> - header；
> - packet loss；
> - MAC delay。
>
> 目的纯粹验证：
>
> [
> \text{progressive evidence optimization 是否存在基本收益}.
> ]
>
> ------
>
> ## 16.2 MVS-B：Total Radio Bits
>
> 动作：
>
> [
> a=(i,r\rightarrow r').
> ]
>
> 定义：
>
> # [ \boxed{ c_a^{radio}
>
> b_h+r'-r.
> }
> ]
>
> 主目标：
>
> [
> \boxed{
> E[B_{\rm radio}].
> }
> ]
>
> 这是第一篇主要通信指标。
>
> ------
>
> ## 16.3 Secondary Resource Metric
>
> 若 UAV (i) 的有效编码效率为：
>
> [
> \eta_i^{eff},
> ]
>
> 定义：
>
> # [ C_a^{res}
>
> \frac{
> b_h+r'-r
> }{
> \eta_i^{eff}
> }.
> ]
>
> 它解释为：
>
> [
> \text{effective radio resource occupation},
> ]
>
> 而不是直接声称：
>
> [
> \text{end-to-end latency}.
> ]
>
> ------
>
> # 17. Lagrangian 与 Bellman 风险
>
> 原 constrained problem：
>
> [
> \min_\pi E[C_\pi]
> ]
>
> subject to：
>
> [
> P_M\le\beta_M,
> ]
>
> [
> P_{FA}\le\alpha.
> ]
>
> Lagrangian：
>
> # [ \mathcal L
>
> E[C_\pi]
> +
> \mu_M(P_M-\beta_M)
> +
> \mu_F(P_{FA}-\alpha).
> ]
>
> 固定：
>
> [
> \mu_M,\mu_F\ge0.
> ]
>
> 与策略有关部分：
>
> [
> E[C_\pi]
> +
> \mu_MP_M
> +
> \mu_FP_{FA}.
> ]
>
> 对应 terminal costs：
>
> # [ \boxed{ C_{01}
>
> \frac{\mu_M}{\pi_1}
> }
> ]
>
> 以及：
>
> # [ \boxed{ C_{10}
>
> \frac{\mu_F}{\pi_0}.
> }
> ]
>
> 若当前 posterior：
>
> [
> p=P(H_1|x),
> ]
>
> 则 STOP-(H_0)：
>
> [
> R_0(x)=C_{01}p,
> ]
>
> STOP-(H_1)：
>
> [
> R_1(x)=C_{10}(1-p).
> ]
>
> 因此：
>
> # [ R_{\rm stop}(x)
>
> \min{
> R_0(x),R_1(x)
> }.
> ]
>
> ------
>
> # 18. Bellman Optimality Equation
>
> 动作集合：
>
> # [ \mathcal A(x)
>
> {
> (i,r_i\rightarrow r_i'):
> r_i'>r_i
> }.
> ]
>
> Bellman 方程：
>
> # [ \boxed{ V^\star(x)
>
> \min
> \left{
> R_{\rm stop}(x),
> \min_{a\in\mathcal A(x)}
> \left[
> c_a
> +
> E[V^\star(x')|x,a]
> \right]
> \right}.
> }
> ]
>
> 其核心含义：
>
> [
> \boxed{
> \text{Expected information value}
> \lessgtr
> \text{communication cost}.
> }
> ]
>
> 不是要求：
>
> > 所有未报告信息都绝对不可能改变判决。
>
> ------
>
> # 19. Constrained Policy 的严谨处理
>
> 固定 dual variables 时：
>
> [
> V^\star
> ]
>
> 可以产生 deterministic Markov policy。
>
> 但原 constrained problem 的最优 operating point 不一定恰好对应单个 deterministic policy。
>
> 因此正式实验采用：
>
> 1. sweep：
>    [
>    (\mu_M,\mu_F);
>    ]
> 2. 得到 deterministic policy Pareto points；
> 3. Monte Carlo 验证：
>    [
>    (P_D,P_{FA},E[B]);
>    ]
> 4. 必要时在两个相邻 policies 之间进行 episode-level randomized mixture。
>
> 不声称一般情况下必然具有 zero duality gap。
>
> ------
>
> # 20. Exact-DP 的有限状态结构
>
> MVS-A：
>
> # [ \mathcal R
>
> {0,1,2,4}.
> ]
>
> 单 UAV 可能的 evidence states：
>
> [
> 1+2+4+16=23.
> ]
>
> 因此：
>
> [
> N=4
> ]
>
> 时粗略全局上界：
>
> # [ 23^4
>
> 1. 
>
> ]
>
> 可作为 Exact-DP oracle。
>
> ------
>
> # 21. Exact-DP 不使用 Value Iteration
>
> MVS-A：
>
> - 无 packet loss；
> - refinement 不可逆；
> - 每个动作严格增加 precision。
>
> 因此 state transition graph 是：
>
> [
> \boxed{\text{DAG}}.
> ]
>
> Exact-DP 采用：
>
> [
> \boxed{
> \text{memoized backward recursion}
> }
> ]
>
> 或 topological backward induction。
>
> terminal / maximum-information states：
>
> [
> V(x)=R_{\rm stop}(x).
> ]
>
> 然后按 remaining refinement depth 反向求值。
>
> Bellman residual：
>
> [
> \max_x
> |V(x)-TV(x)|
> ]
>
> 只作为 correctness audit。
>
> ------
>
> # 22. Exact-DP 使用范围
>
> 正式冻结：
>
> [
> \boxed{
> N=4
> }
> ]
>
> 为 exact oracle。
>
> (N=5) 仅作为可选状态剪枝测试。
>
> 不再声称：
>
> [
> N=6
> ]
>
> 一定能够 exact solve。
>
> MVS-B：
>
> [
> N=8
> ]
>
> 只运行近似 O-PEF。
>
> ------
>
> # 23. O-PEF-1：Depth-1
>
> 定义：
>
> # [ Q_a^{(1)}(x)
>
> c_a+
> E[
> R_{\rm stop}(x')
> |x,a
> ].
> ]
>
> 若：
>
> [
> R_{\rm stop}(x)
> \le
> \min_aQ_a^{(1)}(x),
> ]
>
> 则 STOP。
>
> 否则：
>
> # [ a^\star
>
> \arg\min_aQ_a^{(1)}.
> ]
>
> O-PEF-1 作为：
>
> [
> \boxed{\text{low-complexity ablation}}
> ]
>
> 而非最终主算法。
>
> ------
>
> # 24. O-PEF-2：Depth-2 主算法
>
> 为了捕捉：
>
> [
> \text{probe}
> \rightarrow
> \text{conditional refinement}
> ]
>
> 价值，采用：
>
> # [ Q_a^{(2)}(x)
>
> c_a
> +
> E_{x'|x,a}
> \left[
> \min
> \left{
> R_{\rm stop}(x'),
> \min_b
> \left[
> c_b+
> E[
> R_{\rm stop}(x'')
> |x',b
> ]
> \right]
> \right}
> \right].
> ]
>
> 若：
>
> [
> R_{\rm stop}(x)
> \le
> \min_aQ_a^{(2)},
> ]
>
> 停止。
>
> 否则：
>
> # [ a^\star
>
> \arg\min_aQ_a^{(2)}.
> ]
>
> ------
>
> # 25. 允许跨级 Refinement
>
> 动作允许：
>
> [
> 0\rightarrow1,
> ]
>
> [
> 0\rightarrow2,
> ]
>
> [
> 0\rightarrow4,
> ]
>
> 甚至：
>
> [
> 0\rightarrow8.
> ]
>
> 同样允许：
>
> [
> 1\rightarrow4,
> ]
>
> [
> 2\rightarrow8.
> ]
>
> 原因是：
>
> [
> b_h
> ]
>
> 较大时连续多个小 packet 可能得不偿失。
>
> ------
>
> # 26. Depth-2 复杂度控制
>
> 对于：
>
> [
> r\rightarrow r',
> ]
>
> child 数量最大约为：
>
> [
> 2^{r'-r}.
> ]
>
> 因此大跨级动作成本较高。
>
> 采用以下机制：
>
> ### 1. Transition table 预计算
>
> 提前保存：
>
> [
> P(m'|m,H_h).
> ]
>
> ### 2. Candidate pruning
>
> 只保留 Top-(M)。
>
> 默认：
>
> [
> M=8.
> ]
>
> ### 3. State caching
>
> 相同：
>
> [
> (\mathbf z,\Omega)
> ]
>
> 重复利用。
>
> ### 4. Action family control
>
> MVS-B 可优先允许：
>
> [
> r\rightarrow r_{\rm next}
> ]
>
> 以及：
>
> [
> r\rightarrow r_{\max}
> ]
>
> 而不是所有组合。
>
> ------
>
> # 27. O-PEF-2E 与 O-PEF-2M
>
> 正式定义两个实现。
>
> ## O-PEF-2E
>
> Exact discrete expectation。
>
> 主要用于：
>
> - MVS-A；
> - MVS-B 机制验证。
>
> ## O-PEF-2M
>
> 使用 stratified / importance sampling 估计第二层 expectation。
>
> 当：
>
> [
> N
> ]
>
> 或：
>
> [
> r_{\max}
> ]
>
> 较大导致 O-PEF-2E 太慢时使用。
>
> 必须比较：
>
> [
> \text{policy cost error}
> ]
>
> 和：
>
> [
> \text{runtime reduction}.
> ]
>
> ------
>
> # 28. Optional Certificate Guard
>
> certificate 不再属于主算法理论。
>
> 如果某状态容易获得：
>
> [
> \Lambda_{\min}>\eta
> ]
>
> 或者：
>
> [
> \Lambda_{\max}<\eta,
> ]
>
> 可以直接 STOP。
>
> 但：
>
> [
> \boxed{
> \text{certificate only provides a sufficient stop}.
> }
> ]
>
> 没有 certificate 时完全不阻止 Bellman stopping。
>
> 如果消融证明 Guard 收益很小，直接放 Appendix。
>
> ------
>
> # 29. MVS-A：最小数学验证系统
>
> ## 29.1 系统规模
>
> [
> \boxed{
> N=4,\qquad Q=1.
> }
> ]
>
> 不加入 UAV geometry。
>
> 不加入无线通信 channel。
>
> 不加入 packet loss。
>
> ------
>
> # 30. MVS-A Sensing Statistical Model
>
> 定义：
>
> [
> H_0:
> X_i\sim\mathcal N(0,1),
> ]
>
> [
> H_1:
> X_i\sim\mathcal N(a_i,1).
> ]
>
> local LLR：
>
> # [ L_i
>
> a_iX_i
> -\frac{a_i^2}{2}.
> ]
>
> 因此：
>
> [
> L_i|H_0
> \sim
> \mathcal N
> \left(
> -\frac{a_i^2}{2},
> a_i^2
> \right),
> ]
>
> [
> L_i|H_1
> \sim
> \mathcal N
> \left(
> +\frac{a_i^2}{2},
> a_i^2
> \right).
> ]
>
> 定义：
>
> # [ a_i^2
>
> 10^{\gamma_i^s/10}.
> ]
>
> ------
>
> # 31. MVS-A 默认 Sensing Strength
>
> 正式冻结：
>
> # [ \boxed{ \gamma_i^s
>
> [-1,1,3,5]\ {\rm dB}.
> }
> ]
>
> 在：
>
> [
> P_{FA}=0.05
> ]
>
> 下，各单 UAV 大致：
>
> [
> P_D
> \approx
> [0.226,0.301,0.408,0.553].
> ]
>
> 因此：
>
> [
> \max_iP_{D,i}
> \approx0.553,
> ]
>
> 不存在单个强节点独立完成任务。
>
> 而 continuous full fusion：
>
> [
> \boxed{
> P_{D,\rm raw}
> \approx0.851.
> }
> ]
>
> 很好地形成：
>
> [
> \boxed{
> \text{weak individual evidence}
> \rightarrow
> \text{strong cooperative evidence}.
> }
> ]
>
> ------
>
> # 32. MVS-A Weak Stress
>
> 额外测试：
>
> [
> [-3,-1,1,3]\ {\rm dB}.
> ]
>
> 该场景不替换默认场景，仅用于观察：
>
> > 当所有 local evidence 更弱时，O-PEF 是否逐渐退化到接近 all-neighbor reporting。
>
> ------
>
> # 33. MVS-A 完整参数
>
> | 参数                 | Default           |
> | -------------------- | ----------------- |
> | UAV 数 (N)           | **4**             |
> | Target 数 (Q)        | **1**             |
> | Prior                | (0.5/0.5)         |
> | Local variance       | 1                 |
> | Sensing strength     | **[-1,1,3,5] dB** |
> | Evidence levels      | **0/1/2/4 bit**   |
> | Header               | **0**             |
> | Packet success       | **1**             |
> | Main cost            | **payload bits**  |
> | (P_{FA})             | **0.05**          |
> | Raw reference        | continuous LLR    |
> | Achievable reference | all-node 4-bit    |
> | Exact oracle         | **DAG DP**        |
> | Monte Carlo          | (10^5)            |
> | Independent runs     | ≥20               |
>
> ------
>
> # 34. MVS-A G0：Statistical Sanity
>
> 必须验证：
>
> ### G0.1
>
> 解析：
>
> [
> P_{D,\rm raw}
> ]
>
> 与 Monte Carlo 一致。
>
> ### G0.2
>
> 所有：
>
> [
> P(M_i^{(r)}=m|H_h)
> ]
>
> 满足：
>
> [
> \sum_mP(M_i^{(r)}=m|H_h)=1.
> ]
>
> ### G0.3
>
> Nested consistency：
>
> # [ P(m|H_h)
>
> \sum_{m'\in children(m)}
> P(m'|H_h).
> ]
>
> ### G0.4
>
> 验证：
>
> [
> P_{D,1bit},
> P_{D,2bit},
> P_{D,4bit}.
> ]
>
> 并报告：
>
> [
> \Delta_Q^{(r)}.
> ]
>
> ### G0.5
>
> Log-domain stress：
>
> [
> \Omega\in[-100,100].
> ]
>
> 要求：
>
> - 无 NaN；
> - 无 overflow；
> - 无非法 probability。
>
> ------
>
> # 35. MVS-A G1：Exact DP
>
> 实现：
>
> [
> \boxed{\text{acyclic backward DP}}
> ]
>
> 而不是 value iteration。
>
> 要求：
>
> [
> \max_x|V-TV|
> ]
>
> 达到双精度数值容差。
>
> 重点审计：
>
> - child mapping；
> - state hash；
> - coarse-to-fine PMF；
> - replace-not-add LLR；
> - legal actions；
> - terminal risk。
>
> ------
>
> # 36. MVS-A G2：Solver Quality
>
> 比较：
>
> - Exact DP；
> - O-PEF-1；
> - O-PEF-2E。
>
> 定义：
>
> # [ Gap
>
> \frac{
> C_{\rm OPEF2}-C_{\rm DP}
> }{
> C_{\rm DP}
> }.
> ]
>
> 目标：
>
> [
> \boxed{
> Gap\le10%.
> }
> ]
>
> 如果：
>
> [
> Gap>20%,
> ]
>
> 停止进入 MVS-B，先优化 solver。
>
> ------
>
> # 37. MVS-B：通信机制最小系统
>
> 系统：
>
> [
> \boxed{
> N=8,\qquad Q=1.
> }
> ]
>
> 继续采用 Gaussian local detector。
>
> 这样：
>
> [
> \text{sensing model}
> ]
>
> 保持可控，只单独加入：
>
> [
> \text{communication complexity}.
> ]
>
> ------
>
> # 38. MVS-B Sensing Parameters
>
> 默认：
>
> [
> \boxed{
> [-4,-3,-2,-1,0,1,2,3]\ {\rm dB}.
> }
> ]
>
> Evidence levels：
>
> [
> \boxed{
> {0,1,2,4,8}.
> }
> ]
>
> 8-bit all-neighbor：
>
> [
> P_{D,\max}
> ]
>
> 作为 achievable reference。
>
> ------
>
> # 39. MVS-B U2U Channel
>
> 采用准静态 air-to-air Rician channel：
>
> # [ h_{io}
>
> \sqrt{\beta_{io}}
> \left[
> \sqrt{
> \frac{K_R}{K_R+1}
> }
> e^{j\phi}
> +
> \sqrt{
> \frac1{K_R+1}
> }
> g_{io}
> \right],
> ]
>
> 其中：
>
> [
> g_{io}\sim\mathcal{CN}(0,1).
> ]
>
> large-scale gain：
>
> # [ \beta_{io}
>
> \beta_0
> \left(
> \frac{d_{io}}{d_0}
> \right)^{-\alpha_c}
> 10^{-S_{io}/10},
> ]
>
> [
> S_{io}
> \sim
> \mathcal N(0,\sigma_{\rm sh}^2).
> ]
>
> ------
>
> # 40. MVS-B Channel Parameters
>
> | 参数                   | Default   |
> | ---------------------- | --------- |
> | Carrier                | 5.9 GHz   |
> | Bandwidth              | 10 MHz    |
> | Link distance          | 100–800 m |
> | (P_c)                  | 10 dBm    |
> | Tx/Rx gain             | 3 dBi     |
> | Path-loss exponent     | 2.2       |
> | Rician (K)             | 8 dB      |
> | Shadowing std          | 4 dB      |
> | Header                 | 16 bit    |
> | Packet success nominal | 0.95      |
>
> 这些参数用于生成：
>
> [
> \text{communication heterogeneity}
> ]
>
> 而不是声称具体 5G/Wi-Fi 标准协议。
>
> ------
>
> # 41. Packet Loss
>
> MVS-B 同时提供两种实现。
>
> ## B1 — ARQ Collapsed
>
> 固定：
>
> [
> p_i^{succ}.
> ]
>
> retry-until-success：
>
> # [ \bar c_a
>
> \frac{
> c_a
> }{
> p_i^{succ}
> }.
> ]
>
> 主指标仍为：
>
> [
> E[B_{\rm radio}].
> ]
>
> ------
>
> ## B2 — Explicit Transition
>
> 一次发送：
>
> # [ x'
>
> \begin{cases}
> x_a^+,&p_i^{succ},\
> x,&1-p_i^{succ}.
> \end{cases}
> ]
>
> 用于验证 collapsed abstraction 是否改变 policy。
>
> ------
>
> # 42. Packet Loss 不强行塞入 Latency 加权和
>
> 不采用：
>
> [
> B+\lambda_TT.
> ]
>
> 另外报告：
>
> [
> E[N_{\rm attempts}],
> ]
>
> [
> E[N_{\rm rounds}],
> ]
>
> 必要时：
>
> [
> E[T_{\rm reporting}].
> ]
>
> 若未来正式研究 latency，则采用：
>
> [
> E[T]\le T_{\max}
> ]
>
> 作为约束，而不是任意加权。
>
> ------
>
> # 43. Sensing–Communication Coupling Stress
>
> MVS-B 设置三个 regime。
>
> ## Regime A：Positive Correlation
>
> 强 sensing UAV 同时拥有较好 U2U。
>
> ## Regime B：Independent
>
> [
> \gamma_i^s
> \perp
> \gamma_i^c.
> ]
>
> ## Regime C：Anti-Correlation
>
> 强 sensing UAV 恰好通信成本高。
>
> Regime C 是最重要的机制实验。
>
> 因为它直接测试：
>
> [
> \boxed{
> \text{sensing quality}
> \neq
> \text{communication value}.
> }
> ]
>
> ------
>
> # 44. 完整 UAV-ISAC 系统
>
> 只有 MVS-A/B 通过 Gate 后才进入。
>
> 默认：
>
> [
> \boxed{
> N=8,\qquad Q=1.
> }
> ]
>
> 空间区域：
>
> [
> 1{\rm km}\times1{\rm km}.
> ]
>
> UAV altitude：
>
> [
> 100\sim200{\rm m}.
> ]
>
> 默认：
>
> [
> 150{\rm m}.
> ]
>
> Target：
>
> [
> Q=1
> ]
>
> 放置于任务区域内部。
>
> ------
>
> # 45. 完整 Sensing Channel
>
> 对于 UAV (i)：
>
> # [ H_0: y_i[n]
>
> c_i[n]
> +
> u_i[n]
> +
> w_i[n],
> ]
>
> # [ H_1: y_i[n]
>
> \alpha_i
> s_i[n-\tau_i]
> e^{j2\pi\nu_i nT_s}
> +
> c_i[n]
> +
> u_i[n]
> +
> w_i[n].
> ]
>
> 其中：
>
> - (\alpha_i)：target reflection；
> - (\tau_i)：delay；
> - (\nu_i)：Doppler；
> - (c_i[n])：clutter；
> - (u_i[n])：residual interference；
> - (w_i[n])：thermal noise。
>
> ------
>
> # 46. Target Echo Power
>
> 采用 monostatic radar equation：
>
> # [ P_{r,i}
>
> \frac{
> P_sG_tG_r\lambda^2\sigma_T
> }{
> (4\pi)^3
> d_{iT}^4
> L_s
> }.
> ]
>
> local sensing SINR：
>
> # [ \gamma_i^s
>
> \frac{
> P_{r,i}
> }{
> P_{cl,i}
> +
> P_{int,i}
> +
> N_0B_sF
> }.
> ]
>
> 这样 UAV geometry 自然产生：
>
> [
> \text{sensing heterogeneity}.
> ]
>
> ------
>
> # 47. Full-System Sensing Parameters
>
> | 参数            | Default     | Sweep          |
> | --------------- | ----------- | -------------- |
> | UAV 数          | 8           | 4/8/16/32      |
> | Target 数       | **1**       | 第一篇固定     |
> | Altitude        | 150 m       | 100–200 m      |
> | Region          | 1 km × 1 km | 固定           |
> | Sensing carrier | 5.9 GHz     | 固定           |
> | Bandwidth       | 20 MHz      | 10–40 MHz      |
> | Sensing power   | 30 dBm      | 固定           |
> | Antenna gain    | 15 dBi      | 固定           |
> | Noise figure    | 6 dB        | 固定           |
> | RCS             | 0.1 m²      | 0.01–1 m²      |
> | CNR             | 5 dB        | 0–15 dB        |
> | Max evidence    | 8 bit       | 固定           |
> | (P_{FA})        | 0.05        | 0.01/0.05/0.10 |
>
> ------
>
> # 48. OTFS/OFDM 的正确位置
>
> O-PEF 不依赖 specific waveform。
>
> 若采用 OTFS：
>
> [
> Y_i[k,l]
> ]
>
> 经过：
>
> - DD matched filtering；
> - GLRT；
> - sparse detector；
>
> 产生：
>
> [
> T_i.
> ]
>
> 再通过 calibration 得到：
>
> # [ L_i
>
> \log
> \frac{
> p_i(T_i|H_1,\xi_i)
> }{
> p_i(T_i|H_0,\xi_i)
> }.
> ]
>
> 因此：
>
> [
> \boxed{
> \text{OTFS负责产生证据，O-PEF负责通信证据。}
> }
> ]
>
> 第一篇不再联合优化 OTFS waveform。
>
> ------
>
> # 49. Baseline：性能参考
>
> ## B0 — Raw Full Fusion
>
> 连续 LLR 全融合。
>
> 只做 ideal upper reference。
>
> ## B1 — Max-Bit All-Neighbor
>
> MVS-A：
>
> [
> 4N\ {\rm bits}.
> ]
>
> MVS-B：
>
> [
> 8N\ {\rm bits}.
> ]
>
> 这是：
>
> [
> \boxed{\text{achievable full-report reference}.}
> ]
>
> ------
>
> # 50. Static Selection Baselines
>
> ## B2 — Random-(K)
>
> ## B3 — Sensing-SNR Top-(K)
>
> ## B4 — Cost-Aware Top-(K)
>
> 例如：
>
> # [ score_i
>
> \frac{
> D_i
> }{
> c_i^{full}
> }.
> ]
>
> 这些算法一次选定节点，不根据实际 message 修改 schedule。
>
> ------
>
> # 51. Classical Distributed-Detection Baselines
>
> ## B5 — Censoring
>
> 根据：
>
> [
> |L_i|>\tau_c
> ]
>
> 决定是否报告。
>
> ## B6 — OTS-F
>
> Ordered Full Reporting。
>
> 按照经典 ordered-transmission 机制依次报告 full evidence。
>
> 仿真如果直接使用全部：
>
> [
> |L_i|
> ]
>
> 进行免费全局排序，则必须标记为：
>
> [
> \boxed{\text{OTS-Oracle-Order}}
> ]
>
> 表示这是一个偏强、未计 ordering-control cost 的基线。
>
> ------
>
> # 52. 更强 OTS Baselines
>
> ## B7 — OTS-C
>
> Ordered Transmission + Censoring。
>
> 弱 local evidence 可以不进入 full reporting。
>
> ## B8 — P-OTS
>
> Progressive Ordered Transmission。
>
> 保留 ordered structure，同时使用固定 progressive precision：
>
> [
> 1\rightarrow2\rightarrow4\rightarrow8.
> ]
>
> 但 precision schedule 不根据当前 fusion posterior 联合优化。
>
> 这是 O-PEF 极其重要的竞争 baseline。
>
> ------
>
> # 53. Progressive Mechanism Baselines
>
> ## B9 — Global Fixed Progressive
>
> 所有 UAV 同步：
>
> [
> 1\rightarrow2\rightarrow4\rightarrow8.
> ]
>
> ## B10 — Random-Subset Progressive
>
> 每轮随机选择 (K) 个 UAV refinement。
>
> ## B11 — Static Cost-Aware Progressive
>
> 根据：
>
> [
> \gamma_i^s
> ]
>
> 及：
>
> [
> c_i
> ]
>
> 预先设计 progressive schedule。
>
> 但收到实际 evidence 后不允许改变 schedule。
>
> ------
>
> # 54. Proposed 与 Oracle
>
> ## B12 — O-PEF-1
>
> Depth-1。
>
> ## Proposed — O-PEF-2
>
> Depth-2 + cross-level refinement + posterior-risk stopping。
>
> ## B13 — Exact DAG-DP
>
> 仅：
>
> [
> N=4.
> ]
>
> 作为最优 oracle。
>
> ------
>
> # 55. 完整 ISAC 阶段额外 Baseline
>
> 再加入：
>
> - FE-SRTS；
> - Nearest-(K)；
> - Sensing-SINR Top-(K)；
> - Hard Detect-then-Fuse。
>
> ------
>
> # 56. Baseline 机制矩阵
>
> | Method             | Node selection | Multi-resolution | Realized-message adaptation | Stopping |
> | ------------------ | -------------- | ---------------- | --------------------------- | -------- |
> | Top-K              | ✓              | ✗                | ✗                           | ✗        |
> | Censoring          | ✓              | ✗                | Local                       | ✓        |
> | OTS-F              | ✓              | ✗                | ✓                           | ✓        |
> | P-OTS              | ✓              | ✓                | 部分                        | ✓        |
> | Fixed Progressive  | ✗              | ✓                | ✗                           | ✓        |
> | Static Progressive | ✓              | ✓                | ✗                           | ✓        |
> | **O-PEF-2**        | **✓**          | **✓**            | **✓**                       | **✓**    |
>
> 因此真正需要证明的是：
>
> [
> \boxed{
> \text{selection}
> +
> \text{precision}
> +
> \text{realized evidence adaptation}
> }
> ]
>
> 的联合增益。
>
> ------
>
> # 57. 核心评价指标
>
> 主指标：
>
> [
> P_D,
> ]
>
> [
> P_{FA},
> ]
>
> [
> E[B_{\rm payload}],
> ]
>
> [
> E[B_{\rm radio}].
> ]
>
> 辅助指标：
>
> [
> E[N_{\rm query}],
> ]
>
> [
> E[N_{\rm attempts}],
> ]
>
> [
> E[C_{\rm resource}],
> ]
>
> [
> runtime.
> ]
>
> 机制指标：
>
> [
> P(r_i^\star=r).
> ]
>
> Oracle 指标：
>
> [
> Gap_{\rm oracle}.
> ]
>
> ------
>
> # 58. 核心实验一：Matched Detection
>
> 固定：
>
> [
> P_{FA}=0.05.
> ]
>
> 要求：
>
> ## [ P_{D,\pi} \ge P_{D,\max}
>
> 0.01.
> ]
>
> 比较：
>
> [
> E[B_{\rm radio}].
> ]
>
> 这是主结果。
>
> ------
>
> # 59. 核心实验二：Matched Communication
>
> 固定：
>
> # [ E[B_{\rm radio}]
>
> B_0.
> ]
>
> 比较：
>
> [
> P_D.
> ]
>
> 验证：
>
> > 同样通信资源下，O-PEF 是否提取更多 detection value。
>
> ------
>
> # 60. 核心实验三：Pareto Frontier
>
> 画：
>
> [
> \boxed{
> E[B_{\rm radio}]
> \quad vs\quad
> P_D
> }
> ]
>
> 统一：
>
> [
> P_{FA}=0.05.
> ]
>
> 理想结果是 O-PEF-2 的 Pareto frontier 优于：
>
> - OTS；
> - P-OTS；
> - Static Progressive；
> - Cost-Aware Top-K。
>
> ------
>
> # 61. Progressive Precision Audit
>
> 报告：
>
> [
> P(r_i^\star=0),
> P(r_i^\star=1),
> P(r_i^\star=2),
> P(r_i^\star=4),
> P(r_i^\star=8).
> ]
>
> 特别检查：
>
> [
> P(
> r_i^\star=r_{\max}
> |
> r_i^\star>0
> ).
> ]
>
> 若接近：
>
> [
> 1,
> ]
>
> 说明：
>
> > 一旦选择 UAV，最终还是必须 full report。
>
> 此时 progressive precision 缺乏必要性。
>
> ------
>
> # 62. Header Stress
>
> 设置：
>
> # [ b_h
>
> 0,8,16,32,64
> ]
>
> bits。
>
> 如果 Proposed 只在：
>
> [
> b_h=0
> ]
>
> 有效，应降低工程定位。
>
> 跨级动作的目标就是让算法随着 (b_h) 增加自动从：
>
> [
> 0\rightarrow1\rightarrow2
> ]
>
> 转向：
>
> [
> 0\rightarrow4
> ]
>
> 甚至：
>
> [
> 0\rightarrow8.
> ]
>
> ------
>
> # 63. Packet Reliability Stress
>
> 设置：
>
> # [ p^{succ}
>
> 1,
> 0.95,
> 0.9,
> 0.8.
> ]
>
> 观察：
>
> [
> \text{sensing strong/U2U bad}
> ]
>
> 节点是否逐渐被：
>
> [
> \text{sensing medium/U2U good}
> ]
>
> 节点替代。
>
> ------
>
> # 64. Network Scaling
>
> 设置：
>
> # [ N
>
> 4,8,16,32,64.
> ]
>
> 64 UAV 只作为 scalability stress。
>
> 报告：
>
> [
> E[B]/N,
> ]
>
> [
> E[N_{\rm query}]/N,
> ]
>
> [
> runtime.
> ]
>
> 如果出现：
>
> [
> E[B]/N\downarrow
> ]
>
> 可以称：
>
> > empirical sublinear reporting behavior。
>
> 没有证明前不能声称：
>
> [
> O(\log N).
> ]
>
> ------
>
> # 65. Calibration Robustness
>
> 完整系统增加：
>
> # [ \gamma_i^{model}
>
> \gamma_i^{true}
> +
> \Delta_\gamma.
> ]
>
> 设置：
>
> # [ \Delta_\gamma
>
> -3,-1,0,1,3
> \ {\rm dB}.
> ]
>
> 报告：
>
> [
> P_D,
> P_{FA},
> E[B].
> ]
>
> 用于检查 message-PMF model mismatch 是否破坏 O-PEF。
>
> ------
>
> # 66. 最终 Gate
>
> ## Gate G0 — Statistical Correctness
>
> 必须通过：
>
> - raw ROC；
> - message PMF；
> - nested probability；
> - (\Delta_Q)；
> - log-domain tests。
>
> ------
>
> ## Gate G1 — Exact DP
>
> 必须完成：
>
> [
> N=4
> ]
>
> DAG-DP，并验证 Bellman identity。
>
> ------
>
> ## Gate G2 — Solver Gap
>
> 要求：
>
> [
> Gap_{\rm OPEF2-DP}
> \le10%
> ]
>
> 为理想目标。
>
> 若：
>
> [
> Gap>20%,
> ]
>
> 停止扩大系统。
>
> ------
>
> ## Gate G3 — Adaptive Evidence Value
>
> O-PEF-2 必须稳定优于：
>
> [
> \boxed{\text{Static Cost-Aware Progressive}}
> ]
>
> 否则说明 realized evidence feedback 没有明显价值。
>
> ------
>
> ## Gate G4 — Progressive Precision Value
>
> O-PEF-2 必须优于：
>
> [
> \boxed{\text{OTS-F/P-OTS}}
> ]
>
> 否则说明 ordered reporting 已经足够。
>
> ------
>
> ## Gate G5 — Protocol Robustness
>
> 在：
>
> [
> b_h=16\sim32
> ]
>
> 以及：
>
> [
> p^{succ}=0.9\sim0.95
> ]
>
> 条件下仍应保留可测收益。
>
> ------
>
> # 67. 第一篇不继续扩展的内容
>
> 本阶段不加入：
>
> [
> \text{trajectory optimization},
> ]
>
> [
> \text{waveform optimization},
> ]
>
> [
> \text{sensing power allocation},
> ]
>
> [
> \text{communication power allocation},
> ]
>
> [
> \text{multi-target association},
> ]
>
> [
> \text{MAPPO/MARL}.
> ]
>
> 原因不是这些问题不重要，而是它们会模糊 O-PEF 最核心的问题：
>
> [
> \boxed{
> \text{adaptive evidence communication itself}.
> }
> ]
>
> ------
>
> # 68. 推荐创新点
>
> ## Contribution 1
>
> 提出 multi-UAV ISAC 中的 **post-observation multi-resolution evidence acquisition** 问题。
>
> 决策变量从：
>
> [
> i
> ]
>
> 提升为：
>
> [
> (i,r_i\rightarrow r_i').
> ]
>
> ------
>
> ## Contribution 2
>
> 建立基于 message likelihood 的 statistically consistent progressive soft fusion，并正确解决 coarse-to-fine evidence replacement。
>
> ------
>
> ## Contribution 3
>
> 证明 current per-UAV evidence cells 构成有限充分 Markov state，并把报告过程建立为 finite sequential decision problem。
>
> ------
>
> ## Contribution 4
>
> 建立：
>
> [
> \text{Exact DAG-DP}
> \rightarrow
> \text{O-PEF-1}
> \rightarrow
> \text{O-PEF-2}
> ]
>
> 完整最优—近似求解链。
>
> ------
>
> ## Contribution 5
>
> 显式区分：
>
> [
> \text{sensing quality}
> ]
>
> 与：
>
> [
> \text{communication cost},
> ]
>
> 并根据实际 evidence state 在线决定 UAV 与 precision。
>
> ------
>
> # 69. 不应主张的创新
>
> 不主张：
>
> - 首次 distributed detection；
> - 首次 censoring；
> - 首次 ordered transmission；
> - 首次 early stopping；
> - 首次 online sensor selection；
> - 首次 dynamic programming detection；
> - 首次 successive refinement；
> - 首次 quantized detection；
> - 首次反馈；
> - 首次降低 UAV 通信量。
>
> 创新来自这些机制在：
>
> [
> \boxed{
> \text{UAV-ISAC multi-resolution adaptive evidence allocation}
> }
> ]
>
> 问题中的联合构造。
>
> ------
>
> # 70. 推荐实验推进顺序
>
> 严格执行：
>
> [
> \boxed{
> G0
> \rightarrow
> G1
> \rightarrow
> G2
> \rightarrow
> G3
> \rightarrow
> G4
> \rightarrow
> G5
> }
> ]
>
> 即：
>
> ### Step 1
>
> Gaussian analytical detector。
>
> ### Step 2
>
> Nested quantizer/message PMF。
>
> ### Step 3
>
> 4-bit all-node reference。
>
> ### Step 4
>
> Exact DAG-DP。
>
> ### Step 5
>
> O-PEF-1。
>
> ### Step 6
>
> O-PEF-2E。
>
> ### Step 7
>
> 强 baseline：
>
> - OTS-F；
> - P-OTS；
> - Static Progressive。
>
> ### Step 8
>
> MVS-B。
>
> ### Step 9
>
> Full UAV-ISAC。
>
> 不能在 G0/G1 没闭环前直接接 OTFS。
>
> ------
>
> # 71. 最理想的论文结果
>
> 理想结果不是：
>
> > O-PEF 比 all-neighbor 少 50% bits。
>
> 更有说服力的是：
>
> > 在 (P_{FA}=0.05) 且 (P_D) 距最高精度全节点融合不超过 0.5–1 个百分点的条件下，O-PEF-2 相对于 OTS-F、P-OTS 和 static cost-aware progressive reporting 进一步降低 U2U radio-bit cost；在 (N=4) 的 exact-solvable 系统中，其平均成本接近 DAG-DP oracle，并且该优势在 sensing/U2U 异质、header overhead 和 packet-loss 条件下仍然存在。
>
> ------
>
> # 72. 失败时的明确解释
>
> 若：
>
> [
> OPEF\approx OTS,
> ]
>
> 关闭 multi-resolution 作为主创新。
>
> 若：
>
> [
> OPEF\approx POTS,
> ]
>
> 说明动态 UAV/precision 联合优化价值有限。
>
> 若：
>
> [
> OPEF\approx StaticProgressive,
> ]
>
> 说明 realized evidence feedback 价值有限。
>
> 若：
>
> [
> r_i^\star=r_{\max}
> ]
>
> 几乎总是成立，则 progressive precision 不值得继续。
>
> 若：
>
> [
> Gap_{\rm DP}>20%,
> ]
>
> 优先修改 solver，不引入 MARL。
>
> 若：
>
> [
> b_h\ge16
> ]
>
> 后收益完全消失，则重新评估 progressive protocol 的现实性。
>
> ------
>
> # 73. 后续扩展
>
> 第一篇完成以后再考虑：
>
> [
> Q>1.
> ]
>
> 动作扩展为：
>
> [
> a=(q,i,r\rightarrow r').
> ]
>
> 共享 U2U budget：
>
> [
> \sum_qC_q
> \le C_{\max}.
> ]
>
> 此时问题成为：
>
> [
> \boxed{
> \text{cross-target adaptive evidence resource allocation}.
> }
> ]
>
> 再进一步研究：
>
> - target-owner assignment；
> - multi-owner fusion；
> - distributed prices；
> - mobility；
> - power allocation；
> - decentralized implementation。
>
> ------
>
> # 74. 最终系统定义
>
> O-PEF 最终定义为：
>
> > **在多个 UAV 已经完成本地 ISAC 感知并产生软证据之后，将每个 UAV 的 evidence 建模为可逐级揭示的 nested message，融合 UAV 根据当前每个节点已暴露的 evidence cell、当前 posterior、未来潜在信息价值及异质 U2U 报告代价，自适应选择下一 UAV 与 evidence precision，并在继续获取信息的期望检测收益不足以抵偿通信成本时主动停止，从而在给定检测 QoS 下最小化多 UAV 协同感知的证据报告资源。**
>
> 其最简数学表达为：
>
> # [ \boxed{ a^\star
>
> \arg\min
> \left{
> R_{\rm stop}(x),
> ;
> c_a+
> E[V(x')|x,a]
> \right}.
> }
> ]
>
> 它解决的根本问题不是：
>
> [
> \text{“谁最好？”}
> ]
>
> 而是：
>
> [
> \boxed{
> \text{“当前状态下，下一单位通信资源究竟还值不值得花，以及应该花在哪里？”}
> }
> ]
>
> ------
>
> # 参考文献
>
> [1] C. Rago, P. Willett, and Y. Bar-Shalom, “Censoring Sensors: A Low-Communication-Rate Scheme for Distributed Detection,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. 32, no. 2, pp. 554–568, 1996.
>
> [2] W. P. Tay, J. N. Tsitsiklis, and M. Z. Win, “Asymptotic Performance of a Censoring Sensor Network,” *IEEE Transactions on Information Theory*, vol. 53, no. 11, pp. 4191–4209, 2007.
>
> [3] R. S. Blum and B. M. Sadler, “Energy Efficient Signal Detection in Sensor Networks Using Ordered Transmissions,” *IEEE Transactions on Signal Processing*, vol. 56, no. 7, pp. 3229–3235, 2008.
>
> [4] C. Tian and J. Chen, “Hypothesis Testing under Successive Refinement Communication Constraints,” *Proc. IEEE International Symposium on Information Theory*, 2007.
>
> [5] V. Srivastava, K. Plarre, and F. Bullo, “Adaptive Sensor Selection in Sequential Hypothesis Testing,” *Proc. IEEE Conference on Decision and Control*, 2011.
>
> [6] S. Li, X. Li, X. Wang, and J. Liu, “Sequential Hypothesis Test With Online Usage-Constrained Sensor Selection,” *IEEE Transactions on Information Theory*, vol. 65, no. 7, pp. 4392–4410, 2019.
>
> [7] S. Adhikary and N. B. Mehta, “Energy-Efficient Distributed Detection Through Feedback-Assisted Ordered Transmissions in the Presence of Fading and Quantization,” *IEEE Transactions on Communications*, vol. 73, no. 7, pp. 5264–5278, 2025.
>
> [8] W. Huang, N. González-Prelcic, V. Ratnam, M. Bayraktar, and C. J. Zhang, “Fuse-then-Detect for Passive UAV Localization Using Multi-UE 5G Uplink Signals,” arXiv:2607.11955, 2026.
>
> [9] C. Dickerson, W. Khawaja, and I. Guvenc, “Adaptive 5G Resource Allocation for Multistatic ISAC-Based UAV Detection and Tracking,” arXiv:2606.21677, 2026.
>
> [10] B. Li, M. Ye, H. Liu, et al., “Queue-Aware Graph Reinforcement Learning for UAV-ISAC-Assisted Maritime Data Collection,” arXiv:2607.00324, 2026.