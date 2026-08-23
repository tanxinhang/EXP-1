# O-PEF MVS-B0.4b — Feedback-Granularity Phase-Transition Theorem

> 依据 `advice/013.md`。**纯理论封板，不改 planner**。主定理（013 §1）：
> 
>     g_x(b) = Q_prog(x; b) − Q_dir(x; b) = E[ min{ Y_x, b } ],
>     Y_x = D_x(X₁) − Δ₂,   D_x(X₁) = R(X₁) − E[R(X₂)|X₁],   Δ₂ = r_max − r_next,
> progressive = r → r_next → r_max（两次 transaction），direct = r → r_max（一次完整包），每次 transaction 固定 setup/header 开销 b ≥ 0。
> 关键性质（013 §2）：g'₊(b) = Pr(Y_x > b)、g'₋(b) = Pr(Y_x ≥ b)——setup 开销对 packetization preference 的边际影响 = 第二次反馈 transaction 的触发概率。b⋆(x) = inf{b ≥ 0 : g_x(b) ≥ 0}，且 b⋆<∞ ⟺ E[Y_x]≥0 ⟺ E[D_x]≥Δ₂（013 §4）。
> B0.4b 用 **exact support computation**（Y_x 离散 ⇒ g_x 精确分段线性，crossing 在 support breakpoint 区间内闭式求解）取代 G6 的 grid interpolation（013 §5）。

> 生成时间: 2026-08-23 19:54:45   模式: SMOKE

## 1. G0 — Identity：Q_prog − Q_dir = E[min{Y_x, b}]（013 §1）

- random reachable states × refineable UAV（N=8，levels (1,2,4,8)，n_state=40，b ∈ [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0]）：
  - (state, UAV) pairs tested = 311；max |g − (Q_prog−Q_dir)| = 3.944e-13；max tower dev |Σ w₁E[R₂|x₁] − E_dir| = 3.411e-13；**G0 = PASS**（标准 < 1e-10）。
- 说明：g 与 Q_prog−Q_dir 由同一 support 的两条独立路径计算——策略值形式（先验塔性质使 E_R 项相消）与 E[min(Y,b)] 形式；偏差纯浮点舍入（~1e-13）。

## 2. G1 — Shape：monotone、concave、导数 = survival（013 §2）

  - 检查 76 个 (state, UAV) 对：monotone=PASS，concave(chord)=PASS，导数=survival（精确 support，区间斜率 & 原子单侧斜率）=PASS；g(0)=−E[Y⁻] 与尾部 g≡E[Y] 均验证。
  - root (x₀, strongest UAV 7) survival Pr(Y>0) = 0.5000（= 1/2 ⇒ Y_x 两点分布，与 G6 一致）。

## 3. G2 — Existence：b⋆(x)<∞ ⟺ E[Y_x]≥0，三情形分类（013 §3）

| 情形 | Y 分布 | E[Y] | 预期 b⋆ | 实际 b⋆ | g(b⋆−ε) | g(b⋆+ε) | g(b) for b≥b⋆ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | -2..-1 p=1/2 | -1.5000 | — | inf | -1.5000 | -1.5000 | progressive dominates ∀b PASS |
| B | -1..1 p=1/2 | 0.0000 | — | 1.0000 | -0.5000 | 0.0000 | g(b)=0 for b≥b⋆ (equal, NOT direct-dominates) PASS |
| C | -1..3 p=1/2 | 1.0000 | — | 1.0000 | -0.5000 | 1.0000 | unique crossing, sign change PASS |
| C | 1..3 p=1/2 | 2.0000 | — | 0.0000 | 0.0000 | 2.0000 | Y≥0 a.s. ⇒ b⋆=0 PASS |

- 注意（013 §3 Case B）：**E[Y]=0 时 b>b⋆ 不是 direct 严格占优，而是两者持平**（g(b)=0 for b≥b⋆）——论文不能笼统写 “b>b⋆ ⇒ direct dominates”，只有 E[Y]>0 才严格成立。

- **real states**（311 个 (state, UAV) 对）：(b⋆<∞) ⟺ (E[Y]≥0) 100% 满足 = PASS；Case A 支配 g(b)≤E[Y]<0 ∀b = PASS；分类计数 A(E[Y]<0, b⋆=∞)=79，B(E[Y]=0)=7，C(E[Y]>0, 有限 crossing)=225。

## 4. G3 — State dependence：exact b⋆(x) 分布（013 §6-§7）

- N=8，strongest UAV = 7（γ=3），Δ₂ = r_max − r_next。root 与 1-bit 子状态（013 §5/§7 的解析例）：

| 状态 | r | case | E[Y_x] | ess sup Y | Pr(Y>0) | b⋆(x) |
| --- | --- | --- | --- | --- | --- | --- |
| x₀ (root) | 0 | C | 38.5594 | 84.1187 | 0.5000 | **7.0000** |
| 1-bit cell 0 | 1 | A | -6.0000 | -6.0000 | 0.0000 | inf |
| 1-bit cell 1 | 1 | A | -3.0932 | -0.1864 | 0.0000 | inf |

- **closed-form 验证**（013 §5）：root g(0) = -3.5000，Pr(Y>0) = 0.5000 ⇒ b⋆ = 0 − g(0)/Pr(Y>0) = 7.0000 = **exact support 结果 7.0000**——不再需要 grid 插值。（root=PASS，1-bit children b⋆=∞ = PASS）

- **reachable children 分布**（309 个 (state, UAV) 对）：finite b⋆ = 194（min=0.0000，max=17.9367，E=7.3404），b⋆=∞ = 115；case 计数 A=115 B=9 C=185。
- **G3 = PASS**：finite 与 ∞ 同时出现 ⇒ b⋆ 是 **state-dependent phase boundary**，**不是**全局常数，也不是 |Ω|/SNR 的简单单调函数（013 §7：不声称后验置信度单调性——目前没有依据）。

- **创新定位冻结（013 §8）**：**Feedback-Granularity-Aware Adaptive Evidence Acquisition under per-transaction setup cost**；核心可辨识结果 = **state-dependent packetization phase transition** + g'ₓ(b) = P(additional feedback transaction)，再与 B0.4/B0.4a 的 paired-difference certified acquisition 组合。不自称 “adaptive quantization” 本身新颖（Fang/Li 与 2026 ISAC 已有动态量化分辨率研究）。

总耗时: 8.2s

