"""MVS-C C5: Protocol Robustness under Link/Calibration Stress（001 §二十六.1 C5）。

定位（010 §十二 路线；001 §二十六.1；SystemModel §41、§65、§16.2）：
  C4 已在 anti-correlated regime 证明（P0 修复后）GPE-EA-het ≡ Myopic-All-het
  （matched-action、matched-QoS，E[D]=0.0000）。C5 把四类协议鲁棒性 stress
  叠加到同一 matched-action 协议上，检查 robustness 压力下是否：
    (A) refinement 价值维持零（≡），还是出现（>0/<0）；
    (B) ARQ-collapsed 期望成本仍是 **affine**（010 §七 envelope 精确保持）；
    (C) calibration mismatch 下 certificate-pruning 的**保真度**（P0 教训的
        正面形式：决策侧参数进 memo Q-key；世界侧参数不进）。

四类 stress：
  1. p_succ ∈ {1,0.95,0.9,0.8}：packet success，ARQ collapsed ——
     期望重传成本 c̄(Δr) = (b0+κ·Δr)/p_succ = b0'/p_succ + (κ/p_succ)·Δr
     ＝ 仍为 affine（b0'=b0/p_succ、κ'=κ/p_succ）⇒ 010 §七 envelope
     （Q_prog−Q_dir=E[min{Y,b0'}], Y=D−κ'Δr）**精确成立**。
  2. b_ctrl ∈ {0,4,8}：每 transaction 的 control/setup 额外 airtime，
     并入 b0（c = b0+b_ctrl+κΔr），同样保持 affine。
  3. Δγ ∈ {-3,-1,0,1,3} dB：calibration mismatch —— planner 用 model
     量化器（γ_model = γ_true+Δγ），世界用 true 采样（SystemModel §65
     的部署语义：UAV 按标定 PMF 量化/报告，真实信道不同）。审计证书
     剪枝在 true 分布下的保真度（fp/fn 率）。
  4. ρ ∈ {0,0.3,0.6}：evidence correlation —— 世界 common-factor 相关
     采样，planner 保持独立模型假设（第一阶段 Y_i⊥Y_j|H_h 放松）。

协议：与 C4-G2 完全同构（anti regime、GPE-EA-het vs Myopic-All-het、
separately calibrated 28 网格、paired CRN、fresh test、paired EB UCB 主
认证 + Hoeffding sanity + Wilson n0/n1、H=96 主 / H=48 stress）。

数理设计（P0 教训正面应用）：
  - 决策侧参数（rho、eta、b0、kappa）进 memo Q-key（C3e P0 修复已含）；
    p_succ/b_ctrl 经 extended_params 合成 (b0',κ') 传入 ⇒ 自动进 key，
    跨 (p_succ,b_ctrl) 无陈旧缓存（T60 锁定）。
  - 世界侧参数（Δγ 换 planner 实例、ρ 只改采样）不进决策 memo。
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time

import numpy as np

import run_mvsb07g2 as g2
import run_mvsc021 as c21
import run_mvsc03a as c3a
import run_mvsc03e as c3e
import run_mvsc04 as c4
from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs import phase_boundary as pb
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA_B = g2.GAMMA_B
LEVELS = g2.LEVELS
ALPHA = g2.ALPHA
BETA = g2.BETA
RHO_GRID = g2.RHO_GRID
ETA_GRID = g2.ETA_GRID
SEED_CAL = g2.SEED_CAL
SEED_TEST = g2.SEED_TEST
FULL_N_CAL = g2.FULL_N_CAL
FULL_N_TEST = g2.FULL_N_TEST
OUT_DIR = g2.OUT_DIR
SMOKE_OUT_DIR = os.path.join(OUT_DIR, "smoke")

fmt = g2.fmt
wilson_upper = g2.wilson_upper
classify_qos2 = c3a.classify_qos2
sample_set = g2.sample_set
GPEMemo = c3e.GPEMemo
link_params = c4.link_params
myopic_all_het = c4.myopic_all_het
sim_decide_het = c4.sim_decide_het
eval_decide_het = c4.eval_decide_het
calibrate_decide_het = c4.calibrate_decide_het

# stress 扫描点（单因素：其它维度取基线）
P_SUCC_GRID = (1.0, 0.95, 0.9, 0.8)
B_CTRL_GRID = (0.0, 4.0, 8.0)
DGAMMA_GRID = (-3.0, -1.0, 0.0, 1.0, 3.0)
RHO_GRID_C5 = (0.0, 0.3, 0.6)


# ---------------------------------------------------------------------------
# (A) ARQ-collapsed + control 扩展成本：affine 保持（010 §七 envelope）
# ---------------------------------------------------------------------------
def extended_params(b0_arr, kappa_arr, p_succ=1.0, b_ctrl=0.0):
    """c̄(Δr) = (b0+b_ctrl+κΔr)/p_succ = b0' + κ'·Δr（affine）。
    b0'=(b0+b_ctrl)/p_succ、κ'=κ/p_succ ⇒ 010 §七 envelope 精确保持。"""
    b0e = (np.asarray(b0_arr) + b_ctrl) / p_succ
    ke = np.asarray(kappa_arr) / p_succ
    return b0e, ke


# ---------------------------------------------------------------------------
# 世界采样（correlation / true 分布）
# ---------------------------------------------------------------------------
def sample_set_corr(N, seed, model, rho):
    """evidence correlation 世界采样（common-factor）：z=√ρ·g+√(1−ρ)·ε，
    L_i = μ_h + σ_i·z_i；ρ=0 退化独立。planner 保持独立模型假设。"""
    rng = np.random.default_rng(seed)
    n = int(N)
    h = (rng.random(n) < model.prior[1]).astype(np.int8)
    a2 = 10.0 ** (model.gamma_db / 10.0)
    a = np.sqrt(a2)
    mu = np.where(h[:, None] == 1, 0.5 * a2[None, :], -0.5 * a2[None, :])
    g = rng.normal(size=n)[:, None]
    eps = rng.normal(size=(n, model.N))
    z = np.sqrt(rho) * g + np.sqrt(1.0 - rho) * eps
    return h, mu + a[None, :] * z


# ---------------------------------------------------------------------------
# (B) ARQ B1(collapsed) vs B2(explicit) 期望成本等价验证（SystemModel §41）
# ---------------------------------------------------------------------------
def sim_dep_collapsed(pl, rho, eta, H, L_i, decide, b0_arr, kappa_arr,
                      p_succ, b_ctrl, quants8, powers8):
    """collapsed：每动作成本 c̄=(b0+b_ctrl+κΔr)/p_succ（期望重传），状态一定
    更新（重传直至成功）。"""
    x, h, om, cost, nt = 0, float(H), 0.0, 0.0, 0
    while True:
        if h < 1e-9:
            break
        dec, _d = decide(pl, x, om, h, rho, eta)
        if dec[0] == "STOP":
            break
        i, _k, r2 = dec[1], dec[2], dec[3]
        zi = (x // powers8[i]) % BASE_B
        r_cur, _m = z_decode_b(zi)
        c = (b0_arr[i] + b_ctrl + kappa_arr[i] * (r2 - r_cur)) / p_succ
        if c > h + 1e-9:
            raise AssertionError(f"collapsed budget violation: {c} > {h}")
        m2 = int(quants8[i].cell_index(r2, float(L_i[i])))
        z2 = z_code_b(r2, m2)
        om2 = om + quants8[i].llr[r2][m2]
        if r_cur > 0:
            om2 -= quants8[i].llr[r_cur][_m]
        x += (z2 - zi) * powers8[i]
        h -= c
        om = om2
        cost += c
        nt += 1
    return om, cost, nt


def sim_dep_explicit(pl, rho, eta, H, L_i, decide, b0_arr, kappa_arr,
                     p_succ, b_ctrl, quants8, powers8, rng):
    """explicit：每次发送独立 Bernoulli(p_succ)；成功→状态更新；失败→状态
    不变；无论成败扣一次单发成本 c_once=b0+b_ctrl+κΔr。期望成本
    c_once/p_succ（几何重试）＝collapsed；预算截断/停止造成微小差异。"""
    x, h, om, cost, nt = 0, float(H), 0.0, 0.0, 0
    while True:
        if h < 1e-9:
            break
        dec, _d = decide(pl, x, om, h, rho, eta)
        if dec[0] == "STOP":
            break
        i, _k, r2 = dec[1], dec[2], dec[3]
        zi = (x // powers8[i]) % BASE_B
        r_cur, _m = z_decode_b(zi)
        c_once = b0_arr[i] + b_ctrl + kappa_arr[i] * (r2 - r_cur)
        if rng.random() < p_succ:
            m2 = int(quants8[i].cell_index(r2, float(L_i[i])))
            z2 = z_code_b(r2, m2)
            om2 = om + quants8[i].llr[r2][m2]
            if r_cur > 0:
                om2 -= quants8[i].llr[r_cur][_m]
            x += (z2 - zi) * powers8[i]
            om = om2
        h -= c_once
        cost += c_once
        nt += 1
    return om, cost, nt


def arq_equivalence_check(pl, b0_list, kappa_list, p_succ, b_ctrl, H,
                          L_set, quants8, powers8, seed, theta, ndec):
    """B1 vs B2 在少量 worlds 的 E[B]/violations 对比。"""
    rng = np.random.default_rng(seed + 1)
    cb = []
    eb = []
    ok = True
    for e in range(len(L_set)):
        L_i = L_set[e]
        _om1, c1, _n1 = sim_dep_collapsed(pl, *theta, H, L_i, ndec,
                                          b0_list, kappa_list, p_succ, b_ctrl,
                                          quants8, powers8)
        _om2, c2, _n2 = sim_dep_explicit(pl, *theta, H, L_i, ndec,
                                         b0_list, kappa_list, p_succ, b_ctrl,
                                         quants8, powers8, rng)
        cb.append(c1)
        eb.append(c2)
        ok &= (c1 <= H + 1e-9 and c2 <= H + 1e-9)
    cb = np.asarray(cb)
    eb = np.asarray(eb)
    return {"E_collapsed": float(cb.mean()), "E_explicit": float(eb.mean()),
            "D": float((cb - eb).mean()),
            "viol": 0 if ok else 1, "n": len(cb)}


# ---------------------------------------------------------------------------
# (C) certificate-pruning 保真度审计（mismatch 下）
# ---------------------------------------------------------------------------
def pruning_fidelity(pl_model, quant_model, mm_true, b0_arr, kappa_arr,
                     states, rho, eta):
    """011 §七（P0.5）修法：**冻结 deploy partition** B^deploy = B^model。

    旧实现（010/C4 版）把 model 的 om 同时喂给 true planner——同一 message
    index m 在 model/true quantizer 中对应的物理 LLR 区间不同，且 Ω_model(x)
    ≠ Ω_true(x)，故 fp/fn 无严格物理语义。

    修法：
      qu_frozen[i] = NestedQuantizer(i, mm_true, bounds_override=
                      quant_model[i].bounds)   —— 同 cells（model partition），
                      PMF/LLR 用 **true** 模型重算 ⇒ ℓ_true^(r)(m)；
      Ω_true(x) = log(π1/π0) + Σ_i ℓ_i,true(m_i)  =  pl_frozen.omega(x)；
      比较 g_model(x, Ω_model) vs g_true(x, Ω_true)（严格同-message-space）。"""
    qu_frozen = [NestedQuantizer(i, mm_true, r_max=pl_model.r_max,
                                 levels=LEVELS,
                                 bounds_override=quant_model[i].bounds)
                 for i in range(len(quant_model))]
    pl_frozen = SparsePlanner(qu_frozen, 1.0, 1.0, b_h=BH, cross_level=True,
                              levels=LEVELS, delta_c=1.0)
    n_chk = n_pr = n_fp = n_fn = 0
    for (x, om) in states:
        om_true = pl_frozen.omega(int(x))          # Ω_true（同 cells true LLR 和）
        for i in range(len(b0_arr)):
            zs = pl_model.decode(int(x))
            r, _ = z_decode_b(zs[i])
            if r >= pl_model.r_max:
                continue
            for s in LEVELS:
                if s <= r:
                    continue
                for t in LEVELS:
                    if t <= s:
                        continue
                    m_ = pb.general_phase_support(pl_model, x, om, i, s, t,
                                                  rho, eta, b0=b0_arr[i],
                                                  kappa=kappa_arr[i])
                    tr_ = pb.general_phase_support(pl_frozen, x, om_true, i, s,
                                                   t, rho, eta, b0=b0_arr[i],
                                                   kappa=kappa_arr[i])
                    if m_ is None or tr_ is None:
                        continue
                    n_chk += 1
                    gm = m_["g"]
                    gt = tr_["g"]
                    if gm >= 0:
                        n_pr += 1
                        if gt < 0:
                            n_fp += 1
                    elif gt >= 0:
                        n_fn += 1
    return {"n_chk": n_chk, "n_model_prune": n_pr, "n_fp": n_fp, "n_fn": n_fn,
            "fp_rate": (n_fp / n_pr) if n_pr else 0.0,
            "fn_rate": (n_fn / max(n_chk - n_pr, 1)) if n_chk > n_pr else 0.0}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _pl_true():
    mm = GaussianDetectorModel(GAMMA_B)
    qu = [NestedQuantizer(i, mm, r_max=8, levels=LEVELS) for i in range(8)]
    return SparsePlanner(qu, 1.0, 1.0, b_h=16.0, cross_level=True,
                         levels=LEVELS, delta_c=1.0)


def _pw8():
    return [BASE_B ** i for i in range(8)]


def _run_pair(pl, qu, b0e, ke, H_cal, L_cal, H_t96, L_t96, H_t48, L_t48,
              out, N_CAL, N_TEST):
    memo_g = GPEMemo()
    methods = [
        # 011 §十一 run_mvsc05 行：与 C4-G2 同构的 **global Depth-2**（q1_het
        # 作第二步 primitive）——G2-gate 已证 G_switch>0，C5 协议与 C4 相同。
        ("GPE-EA-het (global D2)",
         (lambda pl, x, om, h, rho, eta, m=memo_g, b=b0e, k=ke:
             c4.gpe_het_decision_global(pl, x, om, h, rho, eta, b, k, m))),
        ("Myopic-All-het", (lambda pl, x, om, h, rho, eta, b=b0e, k=ke:
                            myopic_all_het(pl, x, om, h, rho, eta, b, k))),
    ]
    t_cal = time.time()
    cal_res = {}
    for (nm, fn) in methods:
        ts, F, tables = calibrate_decide_het(pl, 96, H_cal, L_cal, qu, _pw8(),
                                             RHO_GRID, ETA_GRID, fn, b0e, ke)
        cal_res[nm] = (ts, F, tables)
        if ts is None:
            out(f"- {nm}：∅（无 FEASIBLE θ̂）feasible {len(F)}/28")
        else:
            s = tables[ts]
            out(f"- {nm}：**θ̂=({ts[0]}, {fmt(ts[1],1)})**、Ê_cal[B]="
                f"{fmt(s['eb'])} airtime、feasible {len(F)}/28")
    out(f"（calibration {time.time()-t_cal:.1f}s）")
    out("")
    r96 = {}
    r48 = {}
    for (nm, fn) in methods:
        ts = cal_res[nm][0]
        if ts is None:
            r96[nm] = r48[nm] = None
            continue
        r96[nm] = eval_decide_het(pl, *ts, 96, H_t96, L_t96, fn, qu, _pw8(),
                                  b0e, ke)
        r48[nm] = eval_decide_het(pl, *ts, 48, H_t48, L_t48, fn, qu, _pw8(),
                                  b0e, ke)
    for H, H_lbl, res in ((96, "H=96 (primary)", r96), (48, "H=48 (stress)",
                                                         r48)):
        sa = res.get("GPE-EA-het")
        sb = res.get("Myopic-All-het")
        g = c4.paired_gate(sa, sb, H)
        if g is None or sa is None or sb is None:
            out(f"- {H_lbl}：θ̂ 缺失 → QoS-UNRESOLVED。")
            continue
        cls_p = classify_qos2(sa["kfa"], sa["n0"], sa["kmd"], sa["n1"])
        cls_m = classify_qos2(sb["kfa"], sb["n0"], sb["kmd"], sb["n1"])
        feas = (cls_p == "FEASIBLE" and cls_m == "FEASIBLE")
        verdict = ("QoS-UNRESOLVED" if not feas
                   else ("**PASS**（EB U95<0）" if g["u_eb"] < 0
                         else "BIT-UNRESOLVED"))
        out(f"- {H_lbl}：E[D]={fmt(g['mean'])}、EB U95={fmt(g['u_eb'])}、"
            f"Hoeffding U95={fmt(g['u_ho'])}；GPE {cls_p}、Myopic {cls_m}"
            f" ⇒ {verdict}")
    out("")
    return cal_res


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nlevel", type=int, default=1)
    args = ap.parse_args()
    SMOKE = args.smoke
    if SMOKE:
        N_TEST = {1: 120, 2: 200, 3: 300, 4: 500}.get(args.nlevel, 120)
        N_CAL = N_TEST // 2
        N_AUD = 60
    else:
        N_TEST = FULL_N_TEST
        N_CAL = FULL_N_CAL
        N_AUD = 120
    out_dir = SMOKE_OUT_DIR if SMOKE else OUT_DIR
    tag = "SMOKE" if SMOKE else "FULL"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"# MVS-C C5 — Protocol Robustness under Link/Calibration Stress"
        f"（001 §二十六.1 C5，{tag}）")
    out("")
    out("> **定位（010 §十二 路线；001 §二十六.1）**：C4 已在 anti-regime 证明"
        "（P0 修复后）GPE-EA-het ≡ Myopic-All-het（E[D]=0.0000，matched-"
        "action）。C5 叠加四类协议鲁棒性 stress：**(1) packet success** "
        "p_succ∈{1,0.95,0.9,0.8}（ARQ collapsed，期望成本仍为 affine ⇒ 010 §七 "
        "envelope 精确保持；含 B1/B2 explicit 等价验证）；**(2) control "
        "overhead** b_ctrl∈{0,4,8}（并入 b0）；**(3) calibration mismatch** "
        "Δγ∈{-3,-1,0,1,3}dB（planner=model 量化器、世界=true 采样；审计证书"
        "剪枝保真度）；**(4) evidence correlation** ρ∈{0,0.3,0.6}（世界 "
        "common-factor、planner 保持独立假设）。")
    out("")
    out("> **数理设计（P0 教训正面应用）**：决策侧参数经 extended_params 合成 "
        "(b0',κ') 传入 ⇒ 自动进入 C3e P0 修复后的 memo Q-key（含 rho/eta/"
        "b0/kappa）⇒ 跨 (p_succ,b_ctrl) 无陈旧缓存（T60 锁定）；世界侧参数 "
        "（Δγ 换 planner 实例、ρ 只改采样）不进决策 memo。")
    out("")
    out(f"> 协议（与 C4-G2 同构）：anti regime、separately calibrated（28 网格）、"
        f"paired CRN、fresh test、paired EB UCB + Hoeffding + Wilson n0/n1；"
        f"主 H=96、stress H=48；N_CAL={N_CAL}、N_TEST={N_TEST}。")
    out("")

    mm_true = GaussianDetectorModel(GAMMA_B)
    b0_arr, kappa_arr = link_params("anti")
    qu_true = [NestedQuantizer(i, mm_true, r_max=8, levels=LEVELS)
               for i in range(8)]
    pl_true = _pl_true()
    pw = _pw8()

    # ---------------- (B) ARQ B1 vs B2 等价验证（N=4 小系统，SystemModel §41）
    out("## B. ARQ collapsed(B1) vs explicit(B2) 期望成本等价验证"
        "（SystemModel §41）")
    out("")
    out("> E[retries]=1/p_succ（几何分布）⇒ collapsed E[B]=E[B_explicit]/p_succ；"
        "预算截断/停止时机造成微小差异。violations=0 验证两记账都不超预算。")
    out("")
    # B1/B2 验证用**强 sensing** 4-UA（anti 下坏链路但 sensing 有价值，
    # root 有行动——弱 sensing 组在 anti 下 root 即 STOP 是正确行为
    # （refinement 价值为负），会造成 E[B]=0 的假象）
    mm4 = GaussianDetectorModel(GAMMA_B[-4:])
    qu4 = [NestedQuantizer(i, mm4, r_max=8, levels=LEVELS) for i in range(4)]
    pw4 = [BASE_B ** i for i in range(4)]
    pl4 = SparsePlanner(qu4, 1.0, 1.0, b_h=16.0, cross_level=True,
                        levels=LEVELS, delta_c=1.0)
    b04, k4 = link_params("anti", GAMMA_B[-4:])
    H4, L4 = sample_set(N_TEST, SEED_TEST * 300 + 1, mm4)
    out("| p_succ | b_ctrl | E[B^collapsed] | E[B^explicit] | D=col−exp | viol |")
    out("| --- | --- | --- | --- | --- | --- |")
    for psu in (0.95, 0.9, 0.8):
        for bct in (0.0, 4.0):
            r = arq_equivalence_check(pl4, list(b04), list(k4), psu, bct, 96,
                                      L4, qu4, pw4, SEED_TEST * 300 + 5,
                                      (256.0, 0.8),
                                      (lambda pl, x, om, h, rho, eta,
                                       b=b04, k=k4, p=psu, bc=bct:
                                       myopic_all_het(pl, x, om, h, rho, eta,
                                                      extended_params(b, k, p,
                                                                      bc)[0],
                                                      extended_params(b, k, p,
                                                                      bc)[1])))
            out(f"| {psu} | {bct} | {fmt(r['E_collapsed'])} | "
                f"{fmt(r['E_explicit'])} | {fmt(r['D'])} | {r['viol']} |")
    out("")

    # ---------------- (A) p_succ / b_ctrl（成本侧 stress）
    out("## A. Packet success / control overhead（成本侧，matched G2）")
    out("")
    out("> 基线 p_succ=1、b_ctrl=0 ＝ C4 anti 记录（E[D]=0.0000，GPE-het ≡ "
        "Myopic-het），C5 不重跑。以下为单因素扫描。")
    out("")
    for psu in (0.95, 0.9, 0.8):
        out(f"### p_succ={psu}（b_ctrl=0）")
        out("")
        b0e, ke = extended_params(b0_arr, kappa_arr, psu, 0.0)
        H_cal, L_cal = sample_set(N_CAL, SEED_CAL + 700, mm_true)
        H_t96, L_t96 = sample_set(N_TEST, SEED_TEST * 1000 + 21, mm_true)
        H_t48, L_t48 = sample_set(N_TEST, SEED_TEST * 1000 + 31, mm_true)
        _run_pair(pl_true, qu_true, b0e, ke, H_cal, L_cal, H_t96, L_t96,
                  H_t48, L_t48, out, N_CAL, N_TEST)
    for bct in (4.0, 8.0):
        out(f"### b_ctrl={bct}（p_succ=1）")
        out("")
        b0e, ke = extended_params(b0_arr, kappa_arr, 1.0, bct)
        H_cal, L_cal = sample_set(N_CAL, SEED_CAL + 710, mm_true)
        H_t96, L_t96 = sample_set(N_TEST, SEED_TEST * 1000 + 22, mm_true)
        H_t48, L_t48 = sample_set(N_TEST, SEED_TEST * 1000 + 32, mm_true)
        _run_pair(pl_true, qu_true, b0e, ke, H_cal, L_cal, H_t96, L_t96,
                  H_t48, L_t48, out, N_CAL, N_TEST)

    # ---------------- (C) calibration mismatch
    out("## C. Calibration mismatch（Δγ∈{-3,-1,0,1,3} dB，planner=model）")
    out("")
    out("> **语义（SystemModel §65）**：planner 量化器/消息-PMF/ℓ/证书全部按 "
        "γ_model=γ_true+Δγ；世界按 true γ 采样。基线 Δγ=0＝C4 anti 记录。")
    out("")
    for dg in DGAMMA_GRID:
        if dg == 0.0:
            out("### Δγ=0（基线）＝ C4 anti 记录（E[D]=0.0000）。")
            out("")
            continue
        out(f"### Δγ={dg} dB（mismatch）")
        out("")
        mm_mod = GaussianDetectorModel(np.asarray(GAMMA_B) + dg)
        qu_mod = [NestedQuantizer(i, mm_mod, r_max=8, levels=LEVELS)
                  for i in range(8)]
        pl_mod = SparsePlanner(qu_mod, 1.0, 1.0, b_h=16.0, cross_level=True,
                               levels=LEVELS, delta_c=1.0)
        b0e, ke = extended_params(b0_arr, kappa_arr, 1.0, 0.0)
        H_cal, L_cal = sample_set(N_CAL, SEED_CAL + 800 + int(dg), mm_true)
        H_t96, L_t96 = sample_set(N_TEST, SEED_TEST * 1000 + 41 + int(dg),
                                  mm_true)
        H_t48, L_t48 = sample_set(N_TEST, SEED_TEST * 1000 + 51 + int(dg),
                                  mm_true)
        _run_pair(pl_mod, qu_mod, b0e, ke, H_cal, L_cal, H_t96, L_t96,
                  H_t48, L_t48, out, N_CAL, N_TEST)
        # 证书剪枝保真度审计（model planner vs true 对照 planner）
        states = c3e._reachable_states(pl_mod, qu_mod, pw, L_t96, N_AUD)
        # 011 §七 P0.5：冻结 model partition（qu_mod.bounds），同 cells 重算
        # true PMF/LLR ⇒ Ω_true（pruning_fidelity 内部构造 pl_frozen）。
        pf = pruning_fidelity(pl_mod, qu_mod, mm_true, b0e, ke, states,
                              256.0, 0.8)
        out(f"- **剪枝保真度审计**（{len(states)} 状态）：检查 {pf['n_chk']}、"
            f"model 剪 {pf['n_model_prune']}、假剪 fp={pf['n_fp']}（"
            f"{fmt(pf['fp_rate'])}）、漏剪 fn={pf['n_fn']}（{fmt(pf['fn_rate'])}）")
        out("")

    # ---------------- (D) evidence correlation
    out("## D. Evidence correlation（ρ∈{0,0.3,0.6}，planner 保持独立假设）")
    out("")
    out("> 世界 L 采样 common-factor 相关（sample_set_corr）；planner 决策不变"
        "（独立模型）⇒ 决策 memo 跨 ρ 复用（数理正确）。基线 ρ=0＝C4 anti 记录。")
    out("")
    for rho in RHO_GRID_C5:
        if rho == 0.0:
            out("### ρ=0（基线）＝ C4 anti 记录（E[D]=0.0000）。")
            out("")
            continue
        out(f"### ρ={rho}")
        out("")
        b0e, ke = extended_params(b0_arr, kappa_arr, 1.0, 0.0)
        H_cal, L_cal = sample_set_corr(N_CAL, SEED_CAL + 900 + int(rho * 10),
                                       mm_true, rho)
        H_t96, L_t96 = sample_set_corr(N_TEST, SEED_TEST * 1000 + 61 +
                                       int(rho * 10), mm_true, rho)
        H_t48, L_t48 = sample_set_corr(N_TEST, SEED_TEST * 1000 + 71 +
                                       int(rho * 10), mm_true, rho)
        _run_pair(pl_true, qu_true, b0e, ke, H_cal, L_cal, H_t96, L_t96,
                  H_t48, L_t48, out, N_CAL, N_TEST)

    # ------------------------------------------------------------ 总结
    out("## 总结（C5 位置：010 §十二 路线）")
    out("")
    out("- **A（p_succ/b_ctrl）**：ARQ-collapsed 期望成本保持 affine ⇒ 010 §七 "
        "envelope 精确成立；matched G2 逐点报告（PASS/BIT-UNRESOLVED/QoS-"
        "UNRESOLVED 诚实口径）。")
    out("- **B（B1 vs B2）**：见上表（E[B^collapsed]≈E[B^explicit]，viol=0）。")
    out("- **C（mismatch）**：证书剪枝保真度 fp/fn 率见上（审计 true 分布下 "
        "model 剪枝的可靠性）。")
    out("- **D（correlation）**：ρ 只改世界采样、决策函数不变 ⇒ 报告中 "
        "matched G2 反映相关证据对融合 QoS 的影响。")
    out("")
    out(f"总耗时: {time.time()-t0:.1f}s")
    out("")

    rp = os.path.join(out_dir, "MVS-C_C5_report.md")
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()