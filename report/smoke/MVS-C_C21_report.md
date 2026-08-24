# MVS-C C2.1 — Budget-Aware Theoretical/Credibility Closure （advice/002.md，SMOKE）

> 依据 002 §二-§八/§十四；C2 记录（fee5cfb/47194bf）保留不动。冻结参数：N=4、GAMMA4=[-1,1,3,5] dB、levels (1,2,4,8)、b{0,i}=16（κ=1）、α=0.05、ε_D=0.01、H∈(48, 96)；matched ρ-homotopy (128, 1024, 8192)、legacy ρ (128, 512)、η∈(0.8, 1.2, 2.0)；**分层抽样 n0=n1**（002 §八）、N_CAL=60/hyp @H=96、N_TEST=120/hyp。

## 0. MITM 精确 8-bit 全融合参考（002 §七，替代 continuous 近似）

- 8-bit（MITM，65636+65636 support）：P_D,max^8b(0.05) = 0.8509（η*=0.8117）→ matched 目标 P_MD ≤ 0.1591
- 4-bit（MITM）：P_D,max^4b(0.05) = 0.8482（η*=0.8244）→ matched 目标 P_MD ≤ 0.1618

## 0.1 π_full 显式可行构造（002 §二：matched 的 primal feasibility 证明）

- π_full = 四架全部 8-bit direct，阈值 η*(α) 由 MITM 参考标定：成本 C=4×24=96.0 ≤ H=96；P_FA=α=0.05（构造标定）、P_D=P_D,max^8b(0.05)=0.8509 ≥ 0.8382 ⇒ **matched primal 可行**（存在性成立，C2 的 INFEASIBLE 结论降级为“冻结族/网格不可行”，002 §二）。

## 1. resource-window phase law（002 §五）＋ constrained pruning（002 §四）

- **Prune-safety 反例（002 §四，N=1 fresh UAV，h=20）**：c1=17、c_dir=24 ⇒ region=A（A：probe 唯一可行）、probe_feas=True、dir_feas=False、prune_probe_ok=False（g>=0 时也必须为 False，因为 direct 不可行）→ PASS（若 g0>=0 而 prune_probe_ok=False 才安全）。
- **区域一致性**：reachable 支撑检查 263（region A/B/C 计数 {'A': 0, 'B': 16, 'C': 247}），Q_prog 公式与区域语义偏差 >1e-6 的个数 0 → PASS
- **constrained-pruning 自洽**（prune ⟺ g>=0 ∧ c_dir<=h）：检查 263，矛盾 0 → PASS

## 2. 机制比较（matched 主口径 / legacy mechanism 参考口径）

> matched 用 **ρ-homotopy 扩展网格 (128, 1024, 8192)**（002 §三，不再撞 1024 边界）与 **MITM BETA8=0.1591**；verdict 措辞按 002 §二：任何网格失败都定性为 **registered-family/grid infeasible**（primal 已由 π_full 构造证明）。分层抽样 n0=n1（002 §八）。


### 3.1 口径：matched（001 §三，ρ-homotopy (128, 1024, 8192)）（α=0.05、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA
- 校准 Myopic-FG(8-bit)：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA
- 校准 Direct8：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA
- 校准 Phase-FG(4-bit)：
    θ̂ = (128,2.0)  (E[B]=0.000, cls=INFEASIBLE, ufa=0.0602, umd=1.0000)；理由：无 FEASIBLE 且无 U_FA<=α：min U_FA

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (128,2.0) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Myopic-FG(8-bit) | (128,2.0) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Direct8 | (128,2.0) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase-FG(4-bit) | (128,2.0) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |
| Myopic-FG(8-bit) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |
| Direct8 | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |
| Phase-FG(4-bit) | INFEASIBLE | 0.0310 | 1.0000 | 0.000 | 0.000 | 0.000 |

- Phase-FG(8-bit) − Direct8（H=96，分层 paired，n=240）：E[D]=0.000、Hoeffding U95=15.168 | no-compare；E[B] 各 0.000 vs 0.000
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，分层 paired，n=240）：E[D]=0.000、Hoeffding U95=15.168 | no-compare；E[B] 各 0.000 vs 0.000

### 3.2 口径：legacy mechanism（017 §四）（α=0.12、β 按粒度）
- 校准 Phase-FG(8-bit)：
    θ̂ = (512,1.2)  (E[B]=39.008, cls=FEASIBLE, ufa=0.1136, umd=0.3723)；理由：FEASIBLE 中 min E[B]
- 校准 Myopic-FG(8-bit)：
    θ̂ = (512,0.8)  (E[B]=33.333, cls=FEASIBLE, ufa=0.1136, umd=0.3723)；理由：FEASIBLE 中 min E[B]
- 校准 Direct8：
    θ̂ = (512,1.2)  (E[B]=40.200, cls=FEASIBLE, ufa=0.0886, umd=0.3723)；理由：FEASIBLE 中 min E[B]
- 校准 Phase-FG(4-bit)：
    θ̂ = (128,1.2)  (E[B]=24.867, cls=FEASIBLE, ufa=0.1136, umd=0.3544)；理由：FEASIBLE 中 min E[B]

| method | θ̂ | QoS cls | U_FA | U_MD | E[B] | E[B|H0] | E[B|H1] | E[payload] | E[N_tx] |  （@H=96）
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | (512,1.2) | FEASIBLE | 0.1155 | 0.3609 | 41.258 | 34.433 | 48.083 | 12.658 | 1.788 |
| Myopic-FG(8-bit) | (512,0.8) | FEASIBLE | 0.1155 | 0.3255 | 35.496 | 30.483 | 40.508 | 6.029 | 1.842 |
| Direct8 | (512,1.2) | FEASIBLE | 0.0938 | 0.3521 | 42.200 | 35.200 | 49.200 | 14.067 | 1.758 |
| Phase-FG(4-bit) | (128,1.2) | UNCERTAIN | 0.1155 | 0.4388 | 26.038 | 23.583 | 28.492 | 4.838 | 1.325 |

| method | QoS cls | U_FA | U_MD | E[B] | E[payload] | E[N_tx] |  （@H=48，同冻结 θ̂）
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase-FG(8-bit) | FEASIBLE | 0.1155 | 0.3697 | 35.300 | 11.767 | 1.471 |
| Myopic-FG(8-bit) | UNCERTAIN | 0.1667 | 0.3255 | 27.704 | 4.904 | 1.425 |
| Direct8 | FEASIBLE | 0.1155 | 0.3697 | 35.300 | 11.767 | 1.471 |
| Phase-FG(4-bit) | UNCERTAIN | 0.1261 | 0.3697 | 24.917 | 4.983 | 1.246 |

- Phase-FG(8-bit) − Direct8（H=96，分层 paired，n=240）：E[D]=-0.942、Hoeffding U95=14.226 | 比较有效（双方 FEASIBLE）；E[B] 各 41.258 vs 42.200
- Phase-FG(8-bit) − Myopic-FG(8-bit)（H=96，分层 paired，n=240）：E[D]=5.763、Hoeffding U95=20.931 | 比较有效（双方 FEASIBLE）；E[B] 各 41.258 vs 35.496

### 3.4 matched 定性（002 §二/§三）

- Phase-FG(8-bit) @ matched、ρ-homotopy：9 个网格点，FEASIBLE 数 = 0，最低 U_MD@α 边 = 0.2803、对应 P_D ≈ 0.7197
- **定性（002 §二）**：matched 网格结果 = **registered frozen controller/grid family** infeasible（或 feasible）——**不等于机制层不可行**；primal feasibility 已由 §0.1 π_full 构造（C=96≤H、P_D=0.8509≥0.8382）**证明存在**。论文措辞按 002 §十建议。

## 4. Gate D1（solver-quality，Lagrangian）与 D2（primal E[C]）（002 §三，取代 C2 的 E[B]-only Gate D）

- θ=(256, 1.2) H=48：V*=85.036（0.0s）；J(Phase)=85.496（Δ_J=0.54%）、J(Myo)=85.036（-0.00%）、J(Direct4)=85.496（0.54%）；[D2 参考] E[B]: Phase=27.500 vs CMDP* E[B]=26.875（裸 E[B] 差 +0.625 —— 只作 primal 参考，不作 solver 证书）
- θ=(256, 1.2) H=96：V*=80.348（17.4s）；J(Phase)=81.864（Δ_J=1.89%）、J(Myo)=81.920（1.96%）、J(Direct4)=80.703（0.44%）；[D2 参考] E[B]: Phase=29.899 vs CMDP* E[B]=34.569（裸 E[B] 差 -4.670 —— 只作 primal 参考，不作 solver 证书）
- θ=(512, 1.2) H=48：V*=142.253（0.0s）；J(Phase)=142.691（Δ_J=0.31%）、J(Myo)=142.253（0.00%）、J(Direct4)=142.691（0.31%）；[D2 参考] E[B]: Phase=30.000 vs CMDP* E[B]=29.438（裸 E[B] 差 +0.562 —— 只作 primal 参考，不作 solver 证书）
- θ=(512, 1.2) H=96：V*=124.641（17.3s）；J(Phase)=126.519（Δ_J=1.51%）、J(Myo)=126.892（1.81%）、J(Direct4)=125.559（0.74%）；[D2 参考] E[B]: Phase=36.885 vs CMDP* E[B]=38.712（裸 E[B] 差 -1.827 —— 只作 primal 参考，不作 solver 证书）
- θ=(1024, 1.6) H=48：V*=287.891（0.0s）；J(Phase)=288.171（Δ_J=0.10%）、J(Myo)=300.504（4.38%）、J(Direct4)=288.171（0.10%）；[D2 参考] E[B]: Phase=31.250 vs CMDP* E[B]=30.938（裸 E[B] 差 +0.312 —— 只作 primal 参考，不作 solver 证书）
- θ=(1024, 1.6) H=96：V*=234.269（16.6s）；J(Phase)=237.916（Δ_J=1.56%）、J(Myo)=241.394（3.04%）、J(Direct4)=236.987（1.16%）；[D2 参考] E[B]: Phase=40.981 vs CMDP* E[B]=44.483（裸 E[B] 差 -3.502 —— 只作 primal 参考，不作 solver 证书）
- θ=(8192, 1.2) H=48：V*=1813.290（0.0s）；J(Phase)=1813.415（Δ_J=0.01%）、J(Myo)=1813.290（0.00%）、J(Direct4)=1813.415（0.01%）；[D2 参考] E[B]: Phase=33.750 vs CMDP* E[B]=33.625（裸 E[B] 差 +0.125 —— 只作 primal 参考，不作 solver 证书）
- θ=(8192, 1.2) H=96：V*=1327.791（16.4s）；J(Phase)=1345.105（Δ_J=1.30%）、J(Myo)=1345.562（1.34%）、J(Direct4)=1344.983（1.29%）；[D2 参考] E[B]: Phase=46.686 vs CMDP* E[B]=53.463（裸 E[B] 差 -6.778 —— 只作 primal 参考，不作 solver 证书）
- **Gate D1 判决**：max Δ_J = 1.89% ≤ 预注册 10% → PASS（002 §三：Δ_J≥0 恒成立时 D1 才可能是质量证书；C2 的 E[B]-only Gate D 正式降级为 PROVISIONAL，由 D1 替代）
- **Gate D2（primal E[C]）**：matched 口径下双方 FEASIBLE 才比较 —— 见 §3.1 matched 表与 §3.4 定性（当前 matched 网格若仍无可行点：D2 UNRESOLVED，按 002 §二 不当作机制否定）。

总耗时: 91.7s

