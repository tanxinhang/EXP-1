"""MVS-C C4: Link-Aware Heterogeneous U2U Airtime（010 §九/§十二；论文 headline）。

定位（001 §十六 / README C4 bullet / 010 §九）：
  C4 把成本模型从 homogeneous "16+(r'−r) bits" 升级为 **link-aware airtime**：
        τ_i(r→r') = b0,i + κ_i·(r'−r),
    b0,i ≡ τ_ctrl,i（该 UAV 一次 transaction/control/setup airtime），
    κ_i  ≡ 1/R_i（每 evidence bit 的 airtime，R_i 为该 U2U 链路数据速率）。
    hard budget：Σ_t τ_{a_t} ≤ H（frame 窗口 airtime）。
  这正是 010 §七 的 generalized phase theorem 参数本身（b0,i、κ_i 直接进入
  G1 的 envelope），因此理论层不用新建——C4 是**把异质链路接入 GPE-EA 决策**
  并在三 regime 下做 matched-action 认证。

三 regime（001 §十六：sensing/link positive / independent / anti-correlated）：
  按 GAMMA_B 的 sensing 强度排序 s_i∈[0,1]（归一化 rank）：
    positive      ：链路质量 q_i = s_i            （强 sensing = 好链路）
    independent   ：q_i = 固定种子 shuffle(s_i)    （无关联）
    anti-correlated：q_i = 1 − s_i                 （强 sensing = 坏链路，
                                                    001 §十六 最重要的机制实验）
  链路质量 → 成本：b0,i = 12 + 8(1−q_i) ∈ [12,20]，κ_i = 0.8 + 0.4(1−q_i)
  ∈ [0.8,1.2]（好链路 = 低 setup、低 per-bit）。homogeneous (b0=16,κ=1)
  是 q≡0.5 的特例（T54 断言 GPE-het ≡ GPE / q1_het ≡ q1_fast）。

G2 协议（与 C3e-G2/017 §四 完全同构，唯一差别是成本模型）：
  新 Proposed = **GPE-EA-het**：full action set A={(i,s): s>r_i}（与
  Myopic-All-het **相同动作空间**），probe 用 conditional-refinement Q
  （per-UAV b0,i/κ_i 进入 _cond_refine_q 与 envelope），Myopic-All-het 用
  one-step Q —— 唯一差别仍是 conditional-refinement planning。
  separately calibrated (ρ,η)（28 网格）、paired CRN、fresh test、主认证
  paired EB UCB（c3e.eb_ucb）+ Hoeffding sanity + Wilson QoS n0/n1。

anti-correlated regime 的机制报告（001 §十六）：
  统计每个 UAV 的 E[N_tx,i] 与 E[B,i]=b0,i·E[N_tx,i]+κ_i·ΣΔr 占比，
  验证 planner 自动把 budget 从"强 sensing 但坏链路"UAV 转移到
  "好链路（哪怕中等 sensing）"UAV —— sensing 质量 ≠ 通信价值的直接证据。
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
from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs import phase_boundary as pb
from opmvs.sparse import SparsePlanner, BASE_B, z_code_b, z_decode_b

GAMMA_B = g2.GAMMA_B
BH = g2.BH          # 名义 setup（homogeneous 特例 = 16）
LEVELS = g2.LEVELS
R_MAX = g2.R_MAX
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
wilson_lower = g2.wilson_lower
hoeffding_upper = g2.hoeffding_upper
hoeffding_lower = g2.hoeffding_lower
classify_qos2 = c3a.classify_qos2
sample_set = g2.sample_set
r_rho = g2.r_rho
_w = c21._w
# 复用 c3e 的 paired EB UCB 与 conditional-refinement Q 原语
eb_ucb = c3e.eb_ucb
eb_lcb = c3e.eb_lcb
GPEMemo = c3e.GPEMemo
_cond_refine_q = c3e._cond_refine_q

REGIMES = ("positive", "independent", "anti")
H_BUDGETS = (48, 96)

LINK_PARAM_BOUNDS = {"b0": (12.0, 20.0), "kappa": (0.8, 1.2)}


# ---------------------------------------------------------------------------
# 链路参数（per-UAV airtime 成本）
# ---------------------------------------------------------------------------
def sensing_rank(gamma_db):
    """s_i ∈ [0,1]：归一化 sensing 强度 rank（强 = 1）。"""
    g = np.asarray(gamma_db, dtype=float)
    order = np.argsort(np.argsort(g))            # 0=最弱 .. N−1=最强
    return order / max(len(g) - 1, 1)


def link_quality(regime, gamma_db, seed=2028):
    """q_i ∈ [0,1]：链路质量（1=最好）。三 regime 定义见文件头。"""
    s = sensing_rank(gamma_db)
    if regime == "positive":
        return s
    if regime == "anti":
        return 1.0 - s
    if regime == "independent":
        return np.random.default_rng(seed).permutation(np.linspace(0.0, 1.0,
                                                                   len(s)))
    raise ValueError(f"unknown regime {regime}")


def link_params(regime, gamma_db=None, seed=2028):
    """(b0_arr, kappa_arr) per-UAV。homogeneous 特例 (16,1) 对应 q=0.5。"""
    gamma_db = GAMMA_B if gamma_db is None else gamma_db
    q = link_quality(regime, gamma_db, seed=seed)
    b0 = 12.0 + 8.0 * (1.0 - q)
    kappa = 0.8 + 0.4 * (1.0 - q)
    return np.asarray(b0, dtype=float), np.asarray(kappa, dtype=float)


# ---------------------------------------------------------------------------
# heterogeneous 决策函数（与 GPE-EA / Myopic-All 同构，成本换 per-UAV）
# ---------------------------------------------------------------------------
def q1_het(pl, x, om, i, r2, rho, eta, b0, kappa):
    """one-step Q 的 heterogeneous 版本：c = b0 + κ·(r2−r_cur)。

    注意：`pl._tpl[i][zi]` 是 *list of 4-tuples* (r2, c_true, q_budget, cells)
    ——必须按 r2 选中该 tuple 再取 [3]=cells（g2.desc_weights 的模式）；
    直接 `pl._tpl[i][zi][3]` 会取到"第 4 个动作 tuple"而不是 cells。"""
    zi = (x // pl.powers[i]) % BASE_B
    r_cur, _m = z_decode_b(zi)
    c = b0 + kappa * (r2 - r_cur)
    lp = -math.log1p(math.exp(-om))
    lq = -math.log1p(math.exp(om))
    cells = next(cs for (r2b, _ct, _qb, cs) in pl._tpl[i][zi]
                 if r2b == int(r2))
    E = 0.0
    for (m2, lp0c, lp1c) in cells:
        w = _w(lp, lq, lp1c, lp0c)
        z2 = z_code_b(r2, m2)
        om2 = om + pl._llr_i[i][z2] - pl._llr_i[i][zi]
        E += w * r_rho(om2, rho, eta)
    return c + E


def gpe_het_decision(pl, x, om, h, rho, eta, b0_arr, kappa_arr, memo=None):
    """GPE-EA-het：full action set + per-UAV (b0_i,κ_i) 成本；probe 用
    conditional-refinement Q（certificate 认证全分支 STOP 最优时精确退化
    one-step），terminal 用 one-step。与 Myopic-All-het 相同动作空间。"""
    cands = []
    diag = {"n_cand": 0, "n_probe": 0, "n_terminal": 0, "n_cert": 0}
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _m = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        b0i, ki = b0_arr[i], kappa_arr[i]
        for s in pl.levels:
            if s <= r:
                continue
            c_s = b0i + ki * (s - r)
            if c_s > h + 1e-9:
                continue
            diag["n_cand"] += 1
            if s == pl.r_max:
                diag["n_terminal"] += 1
                cands.append((q1_het(pl, x, om, i, s, rho, eta, b0i, ki),
                              ("ACT", i, "ANY", s)))
                continue
            conts = [t for t in pl.levels
                     if t > s and b0i + ki * (t - s) <= h - c_s + 1e-9]
            if not conts:
                diag["n_terminal"] += 1
                cands.append((q1_het(pl, x, om, i, s, rho, eta, b0i, ki),
                              ("ACT", i, "ANY", s)))
                continue
            diag["n_probe"] += 1
            Q_cond, pruned = _cond_refine_q(pl, x, om, i, s, conts, rho, eta,
                                            memo=memo, b0=b0i, kappa=ki)
            if pruned:
                diag["n_cert"] += 1
            cands.append((Q_cond, ("ACT", i, "ANY", s)))
    if not cands:
        return ("STOP",), diag
    best_q, best_a = min(cands, key=lambda t: t[0])
    if r_rho(om, rho, eta) <= best_q:
        return ("STOP",), diag
    return best_a, diag


def myopic_all_het(pl, x, om, h, rho, eta, b0_arr, kappa_arr):
    """Myopic-All-het：与 GPE-EA-het 相同动作空间，one-step Q（per-UAV 成本）。"""
    cands = []
    for i in range(pl.N):
        zi = (x // pl.powers[i]) % BASE_B
        r, _m = z_decode_b(zi)
        if r >= pl.r_max:
            continue
        b0i, ki = b0_arr[i], kappa_arr[i]
        for s in pl.levels:
            if s <= r:
                continue
            c = b0i + ki * (s - r)
            if c > h + 1e-9:
                continue
            cands.append((q1_het(pl, x, om, i, s, rho, eta, b0i, ki),
                          ("ACT", i, "ANY", s)))
    if not cands:
        return ("STOP",), {}
    best_q, best_a = min(cands, key=lambda t: t[0])
    if r_rho(om, rho, eta) <= best_q:
        return ("STOP",), {}
    return best_a, {}


# ---------------------------------------------------------------------------
# heterogeneous episode sim / eval / calibrate（成本 = b0_i + κ_i·Δr）
# ---------------------------------------------------------------------------
def sim_decide_het(pl, rho, eta, H, L_i, decide, quants8, powers8,
                   b0_arr, kappa_arr):
    """与 c3a.sim_decide 同构，但每步成本 c=b0_arr[i]+κ_arr[i](r2−r_cur)；
    额外记账 per-UAV 事务数 nt_i 与 per-UAV payload。"""
    x, h, om, cost, pay, nt = 0, float(H), 0.0, 0.0, 0.0, 0
    n8 = pl.N
    nt_i = np.zeros(n8)
    pay_i = np.zeros(n8)
    while True:
        if h < 1e-9:
            break
        dec, _d = decide(pl, x, om, h, rho, eta)
        if dec[0] == "STOP":
            break
        i, _k, r2 = dec[1], dec[2], dec[3]
        zi = (x // powers8[i]) % BASE_B
        r_cur, _m = z_decode_b(zi)
        c = b0_arr[i] + kappa_arr[i] * (r2 - r_cur)
        if c > h + 1e-9:
            raise AssertionError(f"het budget violation: c={c} > h={h}")
        m2 = int(quants8[i].cell_index(r2, float(L_i[i])))
        z2 = z_code_b(r2, m2)
        om2 = om + quants8[i].llr[r2][m2]
        if r_cur > 0:
            om2 -= quants8[i].llr[r_cur][_m]
        x += (z2 - zi) * powers8[i]
        h -= c
        om = om2
        cost += c
        pay += (r2 - r_cur)
        nt += 1
        nt_i[i] += 1
        pay_i[i] += (r2 - r_cur)
    return om, cost, nt, pay, nt_i, pay_i


def eval_decide_het(pl, rho, eta, H, H_all, L_all, decide, quants8, powers8,
                    b0_arr, kappa_arr):
    n_ep = len(H_all)
    b_m = np.empty(n_ep)
    pay_m = np.empty(n_ep)
    nt_m = np.empty(n_ep)
    lam_m = np.empty(n_ep)
    nt_i_all = np.zeros((n_ep, pl.N))
    pay_i_all = np.zeros((n_ep, pl.N))
    viol = 0
    for e in range(n_ep):
        lam, cost, nt, pay, nt_i, pay_i = sim_decide_het(
            pl, rho, eta, H, L_all[e], decide, quants8, powers8,
            b0_arr, kappa_arr)
        lam_m[e], b_m[e], nt_m[e], pay_m[e] = lam, cost, nt, pay
        nt_i_all[e] = nt_i
        pay_i_all[e] = pay_i
        # 记账恒等式：B = Σ_i (b0_i·N_tx,i + κ_i·Pay_i)（deterministic）
        chk = float(np.sum(b0_arr * nt_i_all[e] + kappa_arr * pay_i_all[e]))
        if abs(chk - cost) > 1e-9 or cost > H + 1e-9:
            viol += 1
    i0 = H_all == 0
    i1 = H_all == 1
    n0 = int(np.count_nonzero(i0))
    n1 = int(np.count_nonzero(i1))
    kfa = int(np.sum(lam_m[i0] > eta))
    kmd = int(np.sum(lam_m[i1] <= eta))
    return {"b": b_m, "pay": pay_m, "nt": nt_m, "lam": lam_m,
            "nt_i": nt_i_all.mean(axis=0), "pay_i": pay_i_all.mean(axis=0),
            "eb": float(b_m.mean()),
            "eb0": float(b_m[i0].mean()), "eb1": float(b_m[i1].mean()),
            "entx": float(nt_m.mean()), "epl": float(pay_m.mean()),
            "kfa": kfa, "kmd": kmd, "n0": n0, "n1": n1, "viol": viol}


def calibrate_decide_het(pl, H, H_cal, L_cal, quants8, powers8, rho_grid,
                         eta_grid, decide, b0_arr, kappa_arr):
    tables = {}
    for rho in rho_grid:
        for eta in eta_grid:
            tables[(rho, eta)] = eval_decide_het(
                pl, rho, eta, H, H_cal, L_cal, decide, quants8, powers8,
                b0_arr, kappa_arr)
    F = {th: s for th, s in tables.items()
         if classify_qos2(s["kfa"], s["n0"], s["kmd"], s["n1"]) == "FEASIBLE"}
    ts = min(F, key=lambda th: (F[th]["eb"], th[0], th[1])) if F else None
    return ts, F, tables


def paired_gate(sa, sb, H):
    """与 c3e 同构：paired D ∈ [−H,H]，EB UCB 主 + Hoeffding sanity。"""
    if sa is None or sb is None:
        return None
    D = sa["b"] - sb["b"]
    lo, hi = -H, H
    u_eb = eb_ucb(D, lo, hi)
    l_eb = eb_lcb(D, lo, hi)
    u_ho = hoeffding_upper(D, H)
    l_ho = hoeffding_lower(D, H)
    return {"D": D, "mean": float(D.mean()), "u_eb": u_eb, "l_eb": l_eb,
            "u_ho": u_ho, "l_ho": l_ho, "eb_smaller": u_eb < 0.0,
            "ho_smaller": u_ho < 0.0}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nlevel", type=int, default=1)
    ap.add_argument("--regime", default=None,
                    help="positive|independent|anti（默认全部）")
    args = ap.parse_args()
    SMOKE = args.smoke
    if SMOKE:
        N_TEST = {1: 120, 2: 200, 3: 300, 4: 500}.get(args.nlevel, 120)
        N_CAL = N_TEST // 2
    else:
        N_TEST = FULL_N_TEST
        N_CAL = FULL_N_CAL
    regimes = (args.regime,) if args.regime else REGIMES
    out_dir = SMOKE_OUT_DIR if SMOKE else OUT_DIR
    tag = "SMOKE" if SMOKE else "FULL"
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out(f"# MVS-C C4 — Link-Aware Heterogeneous U2U Airtime"
        f"（010 §九/§十二，{tag}）")
    out("")
    out("> **定位（001 §十六 / README C4 / 010 §九）**：成本从 homogeneous "
        "\"16+Δr bits\" 升级为 **per-UAV airtime** τ_i(r→r')=b0,i+κ_i(r'−r) "
        "（b0,i≡τ_ctrl,i、κ_i≡1/R_i），hard frame budget Στ≤H。三 regime："
        "positive / independent / **anti-correlated**（强 sensing = 坏链路，"
        "001 §十六 最重要的机制实验）。GPE-EA-het vs Myopic-All-het 做 "
        "**matched-action** 对比（相同 full action set、相同成本模型、相同 "
        "QoS，唯一差别 = conditional-refinement planning）——协议与 C3e-G2 "
        "同构（separately calibrated、paired CRN、fresh test、paired EB UCB "
        "主认证 + Hoeffding sanity + Wilson n0/n1）。")
    out("")
    out(f"> 协议：N=8（GAMMA_B）、levels={LEVELS}、QoS(P_FA≤{ALPHA}, "
        f"P_MD≤{BETA})；ρ∈{RHO_GRID}、η∈{ETA_GRID}（28 combos/method，仅 "
        f"calibration）；calibration worlds 共用、test worlds 完全分离"
        f"（paired CRN）；主 H=96、stress H=48（同冻结 θ̂）；N_CAL={N_CAL}、"
        f"N_TEST={N_TEST}。budget 单位为 airtime，homogeneous 特例=旧 16+Δr。")
    out("")

    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=LEVELS)
               for i in range(8)]
    powers8 = [BASE_B ** i for i in range(8)]
    pl = SparsePlanner(quants8, 1.0, 1.0, b_h=BH, cross_level=True,
                       levels=LEVELS, delta_c=1.0)

    for reg in regimes:
        b0_arr, kappa_arr = link_params(reg)
        out(f"## Regime：{reg}")
        out("")
        out(f"> 链路参数（sensing rank → q → b0/κ）：b0={list(np.round(b0_arr,1))}，"
            f"κ={list(np.round(kappa_arr,2))}；bounds b0∈{LINK_PARAM_BOUNDS['b0']}、"
            f"κ∈{LINK_PARAM_BOUNDS['kappa']}。")
        out("")

        H_cal, L_cal = sample_set(N_CAL, SEED_CAL + (REGIMES.index(reg) * 100),
                                  model8)
        H_t96, L_t96 = sample_set(N_TEST,
                                  SEED_TEST * 1000 + 2 + REGIMES.index(reg),
                                  model8)
        H_t48, L_t48 = sample_set(N_TEST,
                                  SEED_TEST * 1000 + 11 + REGIMES.index(reg),
                                  model8)

        memo_g = GPEMemo()
        methods = [
            ("GPE-EA-het", (lambda pl, x, om, h, rho, eta, memo=memo_g:
                            gpe_het_decision(pl, x, om, h, rho, eta,
                                             b0_arr, kappa_arr, memo))),
            ("Myopic-All-het", (lambda pl, x, om, h, rho, eta:
                                myopic_all_het(pl, x, om, h, rho, eta,
                                               b0_arr, kappa_arr))),
        ]
        t_cal = time.time()
        cal_res = {}
        for (nm, fn) in methods:
            ts, F, tables = calibrate_decide_het(
                pl, 96, H_cal, L_cal, quants8, powers8, RHO_GRID, ETA_GRID,
                fn, b0_arr, kappa_arr)
            cal_res[nm] = (ts, F, tables)
            if ts is None:
                out(f"- {nm}：**∅（无 FEASIBLE θ̂）**；feasible {len(F)}/28")
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
            r96[nm] = eval_decide_het(pl, *ts, 96, H_t96, L_t96, fn,
                                      quants8, powers8, b0_arr, kappa_arr)
            r48[nm] = eval_decide_het(pl, *ts, 48, H_t48, L_t48, fn,
                                      quants8, powers8, b0_arr, kappa_arr)

        out("| 方法 | θ̂ | P_FA | U95 | P_MD | U95 | 分类 | E[N_tx] | "
            "E[B_payload] | E[B] |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for (nm, fn) in methods:
            s = r96.get(nm)
            if s is None:
                continue
            ts = cal_res[nm][0]
            cls = classify_qos2(s["kfa"], s["n0"], s["kmd"], s["n1"])
            out(f"| {nm} | ({ts[0]},{fmt(ts[1],1)}) | {fmt(s['kfa']/s['n0'])} "
                f"| {fmt(wilson_upper(s['kfa'], s['n0']))} | "
                f"{fmt(s['kmd']/s['n1'])} | "
                f"{fmt(wilson_upper(s['kmd'], s['n1']))} | {cls} | "
                f"{fmt(s['entx'])} | {fmt(s['epl'])} | {fmt(s['eb'])} |")
        out("")

        for H, H_lbl, res in ((96, "H=96 (primary)", r96),
                              (48, "H=48 (stress)", r48)):
            sa = res.get("GPE-EA-het")
            sb = res.get("Myopic-All-het")
            g = paired_gate(sa, sb, H)
            out(f"### {H_lbl}：paired D=E[B^GPE]−E[B^Myopic]（airtime）")
            out("")
            if g is None or sa is None or sb is None:
                out("- θ̂ 缺失 → QoS-UNRESOLVED（无法比较）。")
                out("")
                continue
            feas_p = classify_qos2(sa["kfa"], sa["n0"], sa["kmd"], sa["n1"]) \
                == "FEASIBLE"
            feas_m = classify_qos2(sb["kfa"], sb["n0"], sb["kmd"], sb["n1"]) \
                == "FEASIBLE"
            out(f"- E[D]={fmt(g['mean'])}；**paired EB U95={fmt(g['u_eb'])}**"
                f"（L95={fmt(g['l_eb'])}，MP Thm 4）；Hoeffding U95="
                f"{fmt(g['u_ho'])}")
            out(f"- QoS：GPE "
                f"{classify_qos2(sa['kfa'], sa['n0'], sa['kmd'], sa['n1'])}"
                f"（U95 {fmt(wilson_upper(sa['kfa'], sa['n0']))}/"
                f"{fmt(wilson_upper(sa['kmd'], sa['n1']))}）、Myopic "
                f"{classify_qos2(sb['kfa'], sb['n0'], sb['kmd'], sb['n1'])}"
                f"（U95 {fmt(wilson_upper(sb['kfa'], sb['n0']))}/"
                f"{fmt(wilson_upper(sb['kmd'], sb['n1']))}）")
            if not (feas_p and feas_m):
                verdict = "QoS-UNRESOLVED（一方/双方 QoS 未认证，不比较 cost）"
            elif g["u_eb"] < 0:
                verdict = ("**PASS**：双方 FEASIBLE 且 paired EB U95<0 ⇒ "
                           "GPE-EA-het 在 matched-action 下统计认证更省 airtime")
            else:
                verdict = "BIT-UNRESOLVED（EB U95≥0；方向/量级见点估计）"
            out(f"- **G2 判定**：{verdict}")
            out(f"- 分解：E[N_tx] GPE={fmt(sa['entx'])} vs Myopic "
                f"{fmt(sb['entx'])}；E[B_payload] {fmt(sa['epl'])} vs "
                f"{fmt(sb['epl'])}；E[B|H0] {fmt(sa['eb0'])} vs "
                f"{fmt(sb['eb0'])}、E[B|H1] {fmt(sa['eb1'])} vs "
                f"{fmt(sb['eb1'])}（secondary）")
            out("")

        # anti-regime 链路重路由机制（001 §十六 关键实验）
        if reg == "anti" and r96.get("GPE-EA-het") is not None:
            s = r96["GPE-EA-het"]
            out("### anti-correlated 的链路选择机制（001 §十六）")
            out("")
            out("> 逐 UAV 报告 GPE-EA-het 在 test 上的 **E[N_tx,i]** 与 "
                "**E[B,i]**（airtime 占比）：验证 planner 自动把 budget 从 "
                "“强 sensing 但坏链路”UAV 转移到“好链路（哪怕中等 sensing）”"
                "UAV。")
            out("")
            out("| UAV | γ^s (dB) | q(链路质量) | b0_i | κ_i | E[N_tx,i] | "
                "E[B,i] 占比 |")
            out("| --- | --- | --- | --- | --- | --- | --- |")
            total_b = float(np.sum(b0_arr * s["nt_i"] + kappa_arr * s["pay_i"]))
            order = np.argsort(GAMMA_B)
            for i in order:
                bi = b0_arr[i] * s["nt_i"][i] + kappa_arr[i] * s["pay_i"][i]
                share = bi / total_b if total_b > 0 else 0.0
                q = link_quality("anti", GAMMA_B)[i]
                out(f"| {i} | {fmt(GAMMA_B[i])} | {fmt(q,2)} | "
                    f"{fmt(b0_arr[i],1)} | {fmt(kappa_arr[i],2)} | "
                    f"{fmt(s['nt_i'][i])} | {fmt(share)} |")
            out("")
            # 相关性：E[N_tx,i] vs κ_i（链路质量 proxy）——基于实测符号的诚实判定
            nt = s["nt_i"]
            corr = float(np.corrcoef(nt, kappa_arr)[0, 1]) \
                if np.std(nt) > 0 and np.std(kappa_arr) > 0 else float("nan")
            share7 = float((b0_arr[7] * nt[7] + kappa_arr[7] * s["pay_i"][7])
                           / total_b) if total_b > 0 else 0.0
            if corr < -0.3:
                mech = (f"负相关 ⇒ planner 把 budget 从坏链路 UAV 转向好链路 "
                        f"UAV——001 §十六 的链路重路由机制成立")
            elif corr > 0.3:
                mech = (f"**正相关——与 001 §十六 的简单预期相反**：matched-QoS "
                        f"下 planner 仍把 airtime 集中在最强 sensing 的坏链路 "
                        f"UAV7（γ=3dB 占 E[B] 约 {fmt(share7)}），因为弱 sensing "
                        f"好链路 UAV 的组合无法 QoS-FEASIBLE（evidence 不足以 "
                        f"达 P_MD≤β）——**sensing QoS 可行性约束优先于链路成本**，"
                        f"anti-correlation 不自动诱导重路由")
            else:
                mech = f"近似无关（|corr|≤0.3）"
            out(f"- **corr(E[N_tx,i], κ_i)={fmt(corr)}**：{mech}。")
            out("")

    # ------------------------------------------------------------ 总结
    out("## 总结（C4 位置：010 §十二 路线）")
    out("")
    out("- **C4 = heterogeneous link-aware airtime**：τ_i=b0,i+κ_i(r'−r)，"
        "与 010 §七 的 generalized envelope 参数（b0,i、κ_i）**天然同一组**"
        "——因此 G1 定理直接覆盖异质链路的相变结构，C4 验证的是 planner "
        "在 per-UAV 成本下的 matched-action 收益与 anti-regime 重路由机制。")
    out("- **G2 认证（matched-action，per-regime）**：见上方各 regime 判定"
        "（PASS / BIT-UNRESOLVED / QoS-UNRESOLVED 诚实报告）。")
    out("- **anti-correlated 机制**：corr(E[N_tx,i], κ_i) 见上（负 ⇒ planner "
        "避开坏链路 UAV，001 §十六 的机制直接证据）。")
    out("")
    out(f"总耗时: {time.time()-t0:.1f}s")
    out("")

    rp = os.path.join(out_dir, "MVS-C_C4_report.md")
    with io.open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()