# MVS-C C2.1 — Budget-Aware Theoretical/Credibility Closure （advice/002.md，FULL）

> 依据 002 §二-§八/§十四；C2 记录（fee5cfb/47194bf）保留不动。冻结参数：N=4、GAMMA4=[-1,1,3,5] dB、levels (1,2,4,8)、b{0,i}=16（κ=1）、α=0.05、ε_D=0.01、H∈(48, 96)；matched ρ-homotopy (128, 256, 512, 1024, 2048, 4096, 8192)、legacy ρ (128, 256, 512, 1024)、η∈(0.8, 0.85, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)；**分层抽样 n0=n1**（002 §八）、N_CAL=400/hyp @H=96、N_TEST=800/hyp。

## 0. MITM 精确 8-bit 全融合参考（002 §七，替代 continuous 近似）

- 8-bit（MITM，65536+65536 support）：P_D,max^8b(0.05) = 0.8509（η*=0.8117）→ matched 目标 P_MD ≤ 0.1591
- 4-bit（MITM）：P_D,max^4b(0.05) = 0.8482（η*=0.8244）→ matched 目标 P_MD ≤ 0.1618

## 0.1 π_full 显式可行构造（002 §二：matched 的 primal feasibility 证明）

- π_full = 四架全部 8-bit direct，阈值 η*(α) 由 MITM 参考标定：成本 C=4×24=96.0 ≤ H=96；P_FA≤α（det-thr 构造标定，α=0.05）、P_D=P_D,max^8b(0.05)=0.8509 ≥ 0.8382 ⇒ **matched primal 可行**（存在性成立，C2 的 INFEASIBLE 结论降级为“冻结族/网格不可行”，002 §二）。

## 1. resource-window phase law（002 §五）＋ constrained pruning（002 §四）

- **Prune-safety 反例（002 §四 + 003 §七，N=1 fresh UAV，h=20）**：c1=17、c_dir=24 ⇒ region=A（A：probe 唯一可行）、g0_chk=E[min(R1-d2,16)]=16.000 ≥ 0（**显式前提断言**）、probe_feas=True、dir_feas=False、prune_probe_ok=False（g>=0 且 direct 不可行时也必须为 False）→ PASS
- **区域恒等式（003 §五按 region 分口径，独立路径对照）**：reachable 支撑检查 263（region A/B/C 计数 {'A': 0, 'B': 16, 'C': 247}）——B: |gap−E[Y]_per-branch|>1e-9 数 0、C: |gap−(Q_prog−Q_dir)|>1e-9 数 0 → PASS
- **dominance-safety（003 §五）**：prune ⟹ Q_prog ≥ Q_dir − ε：检查 263（dir_feas 状态，A 区含不剪断言），矛盾 0 → PASS

## 2. 机制比较（matched 主口径 / legacy mechanism 参考口径）

> matched 用 **ρ-homotopy 扩展网格 (128, 256, 512, 1024, 2048, 4096, 8192)**（002 §三，不再撞 1024 边界）与 **MITM BETA8=0.1591**；verdict 措辞按 002 §二：任何网格失败都定性为 **registered-family/grid infeasible**（primal 已由 π_full 构造证明）。分层抽样 n0=n1（002 §八）。


### 3.1 口径：matched（001 §三，ρ-homotopy (128, 256, 512, 1024, 2048, 4096, 8192)）（α=0.05、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (8192,1.6)  (E[B]=56.203, cls=INFEASIBLE, ufa=0.0422, umd=0.2446)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD
- 校准 Myopic-FG(8-bit)：
    θ̂ = (2048,1.4)  (E[B]=44.314, cls=UNCERTAIN, ufa=0.0486, umd=0.2287)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD
- 校准 Direct8：
    θ̂ = (1024,1.4)  (E[B]=47.160, cls=UNCERTAIN, ufa=0.0486, umd=0.2287)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD
- 校准 Phase-FG(4-bit)：
    θ̂ = (4096,1.4)  (E[B]=46.875, cls=UNCERTAIN, ufa=0.0486, umd=0.2206)；理由：无 FEASIBLE：U_FA<=α 中 min U_MD

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (8192,1.6) | INFEASIBLE | 0.0472 | 0.2656 | 55.919 | 44.329 | 67.510 | 17.169 | 2.422 |
| Myopic-FG(8-bit) | (2048,1.4) | INFEASIBLE | 0.0516 | 0.2474 | 43.548 | 33.566 | 53.530 | 8.178 | 2.211 |
| Direct8 | (1024,1.4) | INFEASIBLE | 0.0501 | 0.2630 | 46.440 | 36.600 | 56.280 | 15.480 | 1.935 |
| Phase-FG(4-bit) | (4096,1.4) | INFEASIBLE | 0.0530 | 0.2409 | 46.087 | 36.405 | 55.770 | 8.998 | 2.318 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | INFEASIBLE | 0.0516 | 0.3875 | 41.179 | 13.639 | 1.721 |
| Myopic-FG(8-bit) | INFEASIBLE | 0.0617 | 0.3531 | 32.320 | 6.120 | 1.637 |
| Direct8 | INFEASIBLE | 0.0602 | 0.3569 | 37.335 | 12.445 | 1.556 |
| Phase-FG(4-bit) | INFEASIBLE | 0.0617 | 0.3506 | 33.681 | 6.601 | 1.692 |

- Phase-FG(8-bit) − Direct8（H=96，分层 paired，n=1600）：E[D]=9.479、Hoeffding U95=15.354 | no-compare；E[B] 各 55.919 vs 46.440
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，分层 paired，n=1600）：E[D]=12.371、Hoeffding U95=18.246 | no-compare；E[B] 各 55.919 vs 43.548

### 3.2 口径：legacy mechanism（017 §四）（α=0.12、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (128,1.0)  (E[B]=27.995, cls=FEASIBLE, ufa=0.1050, umd=0.2816)；理由：FEASIBLE 中 min E[B]
- 校准 Myopic-FG(8-bit)：
    θ̂ = (128,1.2)  (E[B]=22.633, cls=FEASIBLE, ufa=0.1164, umd=0.2684)；理由：FEASIBLE 中 min E[B]
- 校准 Direct8：
    θ̂ = (128,0.9)  (E[B]=29.430, cls=FEASIBLE, ufa=0.1108, umd=0.2499)；理由：FEASIBLE 中 min E[B]
- 校准 Phase-FG(4-bit)：
    θ̂ = (128,1.0)  (E[B]=25.029, cls=FEASIBLE, ufa=0.0877, umd=0.2737)；理由：FEASIBLE 中 min E[B]

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (128,1.0) | FEASIBLE | 0.0773 | 0.3147 | 27.570 | 26.146 | 28.994 | 8.210 | 1.210 |
| Myopic-FG(8-bit) | (128,1.2) | FEASIBLE | 0.0856 | 0.2902 | 22.027 | 20.486 | 23.567 | 2.237 | 1.237 |
| Direct8 | (128,0.9) | FEASIBLE | 0.0870 | 0.3057 | 29.175 | 27.090 | 31.260 | 9.725 | 1.216 |
| Phase-FG(4-bit) | (128,1.0) | FEASIBLE | 0.0801 | 0.3005 | 24.349 | 22.675 | 26.024 | 4.409 | 1.246 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | FEASIBLE | 0.0829 | 0.3326 | 27.304 | 8.194 | 1.194 |
| Myopic-FG(8-bit) | FEASIBLE | 0.0954 | 0.3198 | 22.038 | 2.237 | 1.238 |
| Direct8 | FEASIBLE | 0.0842 | 0.3185 | 28.200 | 9.400 | 1.175 |
| Phase-FG(4-bit) | FEASIBLE | 0.0842 | 0.3288 | 24.221 | 4.421 | 1.238 |

- Phase-FG(8-bit) − Direct8（H=96，分层 paired，n=1600）：E[D]=-1.605、Hoeffding U95=4.270 | 比较有效（双方 FEASIBLE）；E[B] 各 27.570 vs 29.175
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，分层 paired，n=1600）：E[D]=5.543、Hoeffding U95=11.418 | 比较有效（双方 FEASIBLE）；E[B] 各 27.570 vs 22.027

### 3.4 matched 定性（002 §二/§三）

- Phase-FG(8-bit) @ matched、ρ-homotopy：63 个网格点，FEASIBLE 数 = 0；**U_FA≤α 边最优点**：(ρ=8192, η=1.6)、U_FA=0.0422、U_MD=0.2446 （≈P_D 0.7554）——与 β8=0.1591 的真实差距 **0.0855**（003 六：不是全网格 min 的假象差距）。
- **定性（002 §二）**：matched 网格结果 = **registered frozen controller/grid family** infeasible（或 feasible）——**不等于机制层不可行**；primal feasibility 已由 §0.1 π_full 构造（C=96≤H、P_D=0.8509≥0.8382）**证明存在**。论文措辞按 002 §十建议。

## 4. Gate D1（solver-quality，Lagrangian）与 D2（primal E[C]）（002 §三，取代 C2 的 E[B]-only Gate D）

- θ=(256, 1.2) H=48：V*=85.036（0.0s）；J(Phase)=85.233（Δ_J=0.23%）、J(Myo)=85.036（-0.00%）、J(Direct4)=85.496（0.54%）；[D2 参考] E[B]: Phase=27.125 vs CMDP* E[B]=26.875（裸 E[B] 差 +0.250 —— 只作 primal 参考，不作 solver 证书）
- θ=(256, 1.2) H=96：V*=80.348（16.6s）；J(Phase)=81.862（Δ_J=1.88%）、J(Myo)=81.920（1.96%）、J(Direct4)=80.703（0.44%）；[D2 参考] E[B]: Phase=29.896 vs CMDP* E[B]=34.569（裸 E[B] 差 -4.673 —— 只作 primal 参考，不作 solver 证书）
- θ=(512, 1.2) H=48：V*=142.253（0.0s）；J(Phase)=142.524（Δ_J=0.19%）、J(Myo)=142.253（0.00%）、J(Direct4)=142.691（0.31%）；[D2 参考] E[B]: Phase=29.812 vs CMDP* E[B]=29.438（裸 E[B] 差 +0.375 —— 只作 primal 参考，不作 solver 证书）
- θ=(512, 1.2) H=96：V*=124.641（16.5s）；J(Phase)=126.512（Δ_J=1.50%）、J(Myo)=126.892（1.81%）、J(Direct4)=125.559（0.74%）；[D2 参考] E[B]: Phase=36.877 vs CMDP* E[B]=38.712（裸 E[B] 差 -1.836 —— 只作 primal 参考，不作 solver 证书）
- θ=(1024, 1.6) H=48：V*=287.891（0.0s）；J(Phase)=287.984（Δ_J=0.03%）、J(Myo)=300.504（4.38%）、J(Direct4)=288.171（0.10%）；[D2 参考] E[B]: Phase=31.062 vs CMDP* E[B]=30.938（裸 E[B] 差 +0.125 —— 只作 primal 参考，不作 solver 证书）
- θ=(1024, 1.6) H=96：V*=234.269（16.4s）；J(Phase)=237.908（Δ_J=1.55%）、J(Myo)=241.394（3.04%）、J(Direct4)=236.987（1.16%）；[D2 参考] E[B]: Phase=40.974 vs CMDP* E[B]=44.483（裸 E[B] 差 -3.509 —— 只作 primal 参考，不作 solver 证书）
- θ=(8192, 1.2) H=48：V*=1813.290（0.1s）；J(Phase)=1813.415（Δ_J=0.01%）、J(Myo)=1813.290（0.00%）、J(Direct4)=1813.415（0.01%）；[D2 参考] E[B]: Phase=33.750 vs CMDP* E[B]=33.625（裸 E[B] 差 +0.125 —— 只作 primal 参考，不作 solver 证书）
- θ=(8192, 1.2) H=96：V*=1327.791（16.5s）；J(Phase)=1345.102（Δ_J=1.30%）、J(Myo)=1345.562（1.34%）、J(Direct4)=1344.983（1.29%）；[D2 参考] E[B]: Phase=46.683 vs CMDP* E[B]=53.463（裸 E[B] 差 -6.781 —— 只作 primal 参考，不作 solver 证书）
- **Gate D1 判决**：max Δ_J = 1.88% ≤ 预注册 10% → PASS（002 §三：Δ_J≥0 恒成立时 D1 才可能是质量证书；C2 的 E[B]-only Gate D 正式降级为 PROVISIONAL，由 D1 替代）
- **Gate D2（primal E[C]）**：matched 口径下双方 FEASIBLE 才比较 —— 见 §3.1 matched 表与 §3.4 定性（当前 matched 网格若仍无可行点：D2 UNRESOLVED，按 002 §二 不当作机制否定）。

总耗时: 739.2s

