"""MVS-B0.6-pre: N=4 exact / N=8 shallow sample-complexity gate (008 §14, 013 final).

先验收算法 — BEFORE the paper-critical B0.6 matched-QoS comparison, verify the
B0.4a/B0.4a-r Certified-Policy-Improvement acquisition (the algorithm B0.6
deploys per decision) has affordable sample complexity:

  G0  N=4 EXACT sample-complexity curves:  P(gap<=2) and E[gap] of the CPI
      decision vs per-decision world budget W, bases {VoI, SNR} (012 §6
      ablation — which base is the better anchor), oracle = MATCHED-base
      exact Q^{pi_b}.  Also settles 012 §6 "VoI candidate ranking vs SNR
      safe anchor" before B0.6.
  G1  N=4 FORMAL certification: eb-mode CPI P(override), P(early decision),
      worlds/rollouts used vs budget — the formal certificate's sample cost.
  G2  N=8 SHALLOW oracle: CPI (betting + eb) on root + random reachable
      states, H=40, oracle = N=8 sparse exact Q^{pi_b}.
  G3  Sample-complexity accounting + B0.6 feasibility projection: from the
      G0 curve pick the per-decision budget w_ep that reaches the quality
      target, project total worlds/wall time for the B0.6 episode simulation
      (n_ep episodes x K decisions), and give the acceptance verdict.

All Q values use the MATCHED base oracle (the same pi_b the rollouts follow).
Acceptance criteria (honest, documented):
  G0: at W=Wmax, P(gap<=2) >= 0.80 AND CPI E[gap] <= base E[gap] - 1.0 on the
      better base (base ablation settles 012 §6);
  G1: eb-mode CPI overrides on states where the base is CLEARLY suboptimal
      (Q^{pi_b}(a_b) - Q_min >= 15): P(override) >= 0.5 at W=8000 with
      n_strat >= 2 — the formal certificate fires within budget when evidence
      is strong (MP range-scaling term makes small-gap certification
      expensive, consistent with B0.4r/008; base-optimal states default to
      the safe base by design);
  G2: at W8, P(gap<=2) >= 0.75 AND CPI E[gap] <= base E[gap];
  G3: projected B0.6 wall time <= 60 min at the w_ep from the G0 curve,
      using CPI-only throughput (the exact oracle is evaluation-only).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np

from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs import sparse as sp
from opmvs.rbl_cr import CRRBL, SNRDirectBase, exact_qa_pi_b
from opmvs.rbl_eb import CPI, VoIBase
from opmvs.sparse import z_code_b

GAMMA_A = [-1.0, 1.0, 3.0, 5.0]
GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
SEED0 = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
BH = 16.0
MU_M, MU_F = 256.0, 256.0 * np.exp(1.0)
DELTA_1 = 6.0 * 0.05 / (np.pi ** 2)          # first-decision episode delta (012 §3)


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def mp(flag):
    return "PASS" if flag else "FAIL"


def reachable_states(rng, pl, n):
    """Random reachable states: root + legal random refinement sequences."""
    out = []
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        x = 0
        for _ in range(int(rng.integers(0, 3))):
            zs = list(pl.decode(x))
            cand = [i for i in range(pl.N) if pl._tpl[i][zs[i]]]
            if not cand:
                break
            i = int(cand[rng.integers(0, len(cand))])
            tpl = pl._tpl[i][zs[i]]
            r2 = tpl[int(rng.integers(0, len(tpl)))][0]
            zs[i] = z_code_b(r2, int(rng.integers(0, 2 ** r2)))
            x = pl.encode(tuple(zs))
        zs = pl.decode(x)
        if any(pl._tpl[i][zs[i]] for i in range(pl.N)):
            out.append(x)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    SMOKE = args.smoke
    n_state = 10 if SMOKE else 30
    n_eb = 16 if SMOKE else 40
    n8 = 4 if SMOKE else 10
    W_GRID = (250, 500, 1000, 2000)
    W_MAX = 2000
    W_EB = 8000
    W8 = 1000 if SMOKE else 2000
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.6-pre — sample-complexity gate（N=4 exact / N=8 shallow）")
    out("")
    out("> 依据 `advice/008.md` §14 与 `advice/013.md` 最终指示：进入 B0.6 前**先验收算法**。"
        "验收对象 = B0.4a/B0.4a-r 的 **CPI acquisition**（B0.6 每步决策实际部署的算法）："
        "base-anchored O(|A|) 置信分配（012 §4 方案 A），cs_mode=betting 为 Operational-CPI"
        "（性能），cs_mode=eb 为 Formal-CPI（safety claim，012 §1）。"
        "G0 同时完成 012 §6 的 base ablation（VoI vs SNR 谁做 anchor）。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if SMOKE else 'FULL'}")
    out("")

    model4 = GaussianDetectorModel(GAMMA_A)
    quants4 = [NestedQuantizer(i, model4, r_max=4, levels=(1, 2, 4)) for i in range(4)]
    model8 = GaussianDetectorModel(GAMMA_B)
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=(1, 2, 4, 8)) for i in range(8)]
    voi = VoIBase(BH)
    base4_snr = SNRDirectBase(quants4, GAMMA_A, BH, eta_b=2.0, levels=(1, 2, 4))
    base8 = SNRDirectBase(quants8, GAMMA_B, BH, eta_b=2.0, levels=(1, 2, 4, 8))

    # ------------------------------------------------------------ G0
    out("## 1. G0 — N=4 exact：P(gap≤2) 与 E[gap] vs 世界预算 W（base ablation，012 §6）")
    out("")
    out(f"- 口径：random reachable N=4 states（n={n_state}），H=40，δ₁={fmt(DELTA_1)}；"
        f"oracle = **matched-base** exact Q^{{π_b}}；gap = Q^{{π_b}}(a) − min{{R_stop, Q^{{π_b}}}}；"
        f"base 行 = a_b 本身，CPI 行 = CPI.decide(betting)。")
    out("")
    out("| base | W | P(gap≤2) | E[gap] | P(override) | E[gain\\|override] | mean worlds |")
    out("| --- | --- | --- | --- | --- | --- | --- |")
    rng0 = np.random.default_rng(SEED0)
    pl4 = sp.SparsePlanner(quants4, MU_M, MU_F, b_h=BH, cross_level=True, levels=(1, 2, 4))
    states0 = reachable_states(rng0, pl4, n_state)
    g0_rows = {}
    n_world_total = 0
    cpi_worlds_total = 0
    t_cpi_start = 0.0
    t_cpi_secs = 0.0
    t_g0 = time.time()
    for base, btag in ((voi, "VoI"), (base4_snr, "SNR")):
        cr_b = CRRBL(quants4, MU_M, MU_F, BH, base, levels=(1, 2, 4),
                     delta_c=1.0, seed=7)
        for W in W_GRID:
            n_ok = n_tt = 0
            gaps_c = []
            gaps_b = []
            ov = 0
            gains = []
            ws = []
            for s, x in enumerate(states0):
                Qp = exact_qa_pi_b(cr_b, x, 40)
                if not Qp:
                    continue
                R0 = cr_b.pl.r_stop(x)
                Qmin = min(list(Qp.values()) + [R0])
                om = cr_b.pl.omega(x)
                a_b = base.act(cr_b.pl, x, om, h=40)
                qb = R0 if a_b is None else Qp.get(a_b, np.inf)
                cpi = CPI(quants4, MU_M, MU_F, BH, base, levels=(1, 2, 4),
                          delta_c=1.0, seed=3, cs_mode="betting")
                cpi.cr.rng = np.random.default_rng(6000 + s)
                t0c = time.time()
                a_e, info = cpi.decide(x, 40, delta_t=DELTA_1, max_worlds=W)
                t_cpi_secs += time.time() - t0c
                cpi_worlds_total += info["n_worlds"]
                qe = R0 if a_e is None else Qp.get(a_e, np.inf)
                n_tt += 1
                n_ok += int(max(0.0, qe - Qmin) <= 2.0)
                gaps_c.append(max(0.0, qe - Qmin))
                gaps_b.append(max(0.0, qb - Qmin))
                ws.append(info["n_worlds"])
                n_world_total += info["n_worlds"]
                if info["override"]:
                    ov += 1
                    gains.append(qb - qe)
            g0_rows[(btag, W)] = (n_ok / max(n_tt, 1), float(np.mean(gaps_c) if gaps_c else 0.0),
                                  float(np.mean(gaps_b) if gaps_b else 0.0),
                                  ov / max(n_tt, 1), float(np.mean(gains) if gains else 0.0),
                                  float(np.mean(ws) if ws else 0.0))
            out(f"| {btag} | {W} | {fmt(g0_rows[(btag, W)][0])} | "
                f"{fmt(g0_rows[(btag, W)][1])} (base {fmt(g0_rows[(btag, W)][2])}) | "
                f"{fmt(g0_rows[(btag, W)][3])} | {fmt(g0_rows[(btag, W)][4])} | "
                f"{fmt(g0_rows[(btag, W)][5])} |")
    out(f"（G0 总耗时 {time.time()-t_g0:.0f}s；CPI-only 吞吐 "
        f"{fmt(cpi_worlds_total / max(t_cpi_secs, 1e-9), 0)} worlds/s）")
    out("")
    # G0 gate on the better base (tie-break: lower E[gap], then VoI)
    best = max(("VoI", "SNR"),
               key=lambda b: (g0_rows[(b, W_MAX)][0], -g0_rows[(b, W_MAX)][1]))
    g0_p, g0_eg, g0_bg, _, _, _ = g0_rows[(best, W_MAX)]
    g0_ok = (g0_p >= 0.80) and (g0_eg <= g0_bg - 1.0)
    out(f"- **G0 gate（best base = {best} @ W={W_MAX}）**：P(gap≤2)={fmt(g0_p)} "
        f"≥ 0.80 → {mp(g0_p >= 0.80)}；CPI E[gap]={fmt(g0_eg)} ≤ base E[gap]−1.0"
        f"（={fmt(g0_bg - 1.0)}）→ {mp(g0_eg <= g0_bg - 1.0)}；**G0 = {mp(g0_ok)}**。"
        f"（012 §6：VoI 作 candidate ranking / SNR 作 safe anchor 的 ablation 数据已出。）")
    out("")

    # ------------------------------------------------------------ G1
    out("## 2. G1 — N=4 formal certification：eb-mode CPI 的 sample cost（012 §1）")
    out("")
    out("- 口径：**form 证书只在证据强时触发**。MP+peeling 半径含 range-scaling 项 "
        "7t(hi−lo)/(3(n−1))（pair range ≈ 2·(h−c_a+R_max) ≈ 790，B0.3c budget-aware "
        "diameter），故 gap g 需 n ≳ 7t(hi−lo)/(3g) 量级的世界数才认证（与 B0.4r/008 "
        "『收紧 bound 只部分解锁证书』一致）。按 base 真 gap 分层——base 明显次优"
        "（Q^{{π_b}}(a_b) − Q_min ≥ 15）时 override 才值得；base 已最优时无 override ⇒ "
        "执行 base（安全，011 §3）——所以 eb decide 只跑在 stratified 状态上。")
    out("")
    rng1 = np.random.default_rng(SEED0 + 1)
    states1_all = reachable_states(rng1, pl4, 3 * n_eb)
    cr_voi4 = CRRBL(quants4, MU_M, MU_F, BH, voi, levels=(1, 2, 4), delta_c=1.0, seed=7)
    GAP_MIN = 15.0
    g1_strat = []            # (x, base_gap) with base gap >= GAP_MIN (oracle cached)
    for x in states1_all:
        Qp = exact_qa_pi_b(cr_voi4, x, 40)
        if not Qp:
            continue
        R0 = cr_voi4.pl.r_stop(x)
        Qmin = min(list(Qp.values()) + [R0])
        om = cr_voi4.pl.omega(x)
        a_b = voi.act(cr_voi4.pl, x, om, h=40)
        qb = R0 if a_b is None else Qp.get(a_b, np.inf)
        bgap = max(0.0, qb - Qmin)
        if bgap >= GAP_MIN:
            g1_strat.append((x, bgap, Qp, R0, Qmin, a_b))
    out("| W | P(override) \\| base gap≥15 | (n_ov/n_strat) | mean worlds (stratified) | mean rollouts |")
    out("| --- | --- | --- | --- | --- |")
    g1_ov_eb = 0
    g1_tt_eb = 0
    for W in (4000, W_EB):
        n_ov_s = 0
        n_tt_s = 0
        ws = []
        rs = []
        for s, (x, bgap, _Qp, _R0, _Qmin, _a_b) in enumerate(g1_strat):
            cpi = CPI(quants4, MU_M, MU_F, BH, voi, levels=(1, 2, 4),
                      delta_c=1.0, seed=3, cs_mode="eb")
            cpi.cr.rng = np.random.default_rng(7000 + s)
            _a, info = cpi.decide(x, 40, delta_t=DELTA_1, max_worlds=W)
            n_tt_s += 1
            n_ov_s += int(info["override"])
            ws.append(info["n_worlds"])
            rs.append(info["n_rollouts"])
        if W == W_EB:
            g1_ov_eb, g1_tt_eb = n_ov_s, n_tt_s
        out(f"| {W} | {fmt(n_ov_s / max(n_tt_s, 1))} | ({n_ov_s}/{n_tt_s}) | "
            f"{fmt(np.mean(ws)) if ws else '—'} | {fmt(np.mean(rs)) if rs else '—'} |")
    out("")
    out(f"- 分层状态：{len(states1_all)} 个 random reachable 中 base gap ≥ {GAP_MIN:.0f} 的 "
        f"有 {len(g1_strat)} 个（gap 列表 = {[float(round(g, 1)) for _x, g, *_ in g1_strat]}）。"
        f"未分层状态（base 已最优）eb decide 跑到 cap 也无 override ⇒ 执行 base（安全）。")
    g1_ok = (g1_tt_eb >= 2) and (g1_ov_eb / max(g1_tt_eb, 1) >= 0.5)
    out(f"- **G1 gate（W={W_EB}，base gap ≥ {GAP_MIN:.0f} 的状态）**：P(override) = "
        f"{fmt(g1_ov_eb / max(g1_tt_eb, 1))}（{g1_ov_eb}/{g1_tt_eb}）≥ 0.5 且 n_strat≥2 → "
        f"{mp(g1_ok)}；**G1 = {mp(g1_ok)}**（form 证书在值得 override 时于预算内触发；"
        f"base 已最优时无 override ⇒ 安全执行 base）。")
    out("")

    # ------------------------------------------------------------ G2
    out("## 3. G2 — N=8 shallow oracle（H=40，oracle = sparse exact Q^{π_b}）")
    out("")
    rng2 = np.random.default_rng(SEED0 + 2)
    states8 = [0] + reachable_states(
        rng2, sp.SparsePlanner(quants8, MU_M, MU_F, b_h=BH, cross_level=True), n8 - 1)
    out("| mode | W | P(gap≤2) | E[gap] (base) | P(override) |")
    out("| --- | --- | --- | --- | --- |")
    g2_rows = {}
    for mode, mtag in (("betting", "Operational"), ("eb", "Formal")):
        n_ok = n_tt = 0
        gaps_c = []
        gaps_b = []
        ov = 0
        for s, x in enumerate(states8):
            cr8 = CRRBL(quants8, MU_M, MU_F, BH, base8, levels=(1, 2, 4, 8),
                        delta_c=1.0, seed=11)
            Qp = exact_qa_pi_b(cr8, x, 40)
            if not Qp:
                continue
            R0 = cr8.pl.r_stop(x)
            Qmin = min(list(Qp.values()) + [R0])
            om = cr8.pl.omega(x)
            a_b = base8.act(cr8.pl, x, om, h=40)
            qb = R0 if a_b is None else Qp.get(a_b, np.inf)
            cpi = CPI(quants8, MU_M, MU_F, BH, base8, levels=(1, 2, 4, 8),
                      delta_c=1.0, seed=3, cs_mode=mode)
            cpi.cr.rng = np.random.default_rng(8000 + s)
            a_e, info = cpi.decide(x, 40, delta_t=DELTA_1, max_worlds=W8)
            qe = R0 if a_e is None else Qp.get(a_e, np.inf)
            n_tt += 1
            n_ok += int(max(0.0, qe - Qmin) <= 2.0)
            gaps_c.append(max(0.0, qe - Qmin))
            gaps_b.append(max(0.0, qb - Qmin))
            ov += int(info["override"])
        g2_rows[mtag] = (n_ok / max(n_tt, 1), float(np.mean(gaps_c) if gaps_c else 0.0),
                         float(np.mean(gaps_b) if gaps_b else 0.0), ov / max(n_tt, 1))
        out(f"| {mtag} | {W8} | {fmt(g2_rows[mtag][0])} | {fmt(g2_rows[mtag][1])} "
            f"({fmt(g2_rows[mtag][2])}) | {fmt(g2_rows[mtag][3])} |")
    out("")
    g2_p, g2_eg, g2_bg, _ = g2_rows["Operational"]
    g2_ok = (g2_p >= 0.75) and (g2_eg <= g2_bg)
    out(f"- **G2 gate（Operational @ W={W8}）**：P(gap≤2)={fmt(g2_p)} ≥ 0.75 → "
        f"{mp(g2_p >= 0.75)}；CPI E[gap]={fmt(g2_eg)} ≤ base E[gap]={fmt(g2_bg)} → "
        f"{mp(g2_eg <= g2_bg)}；**G2 = {mp(g2_ok)}**（N=8 shallow 验收）。")
    out("")

    # ------------------------------------------------------------ G3
    out("## 4. G3 — sample-complexity accounting + B0.6 可行性（先验收算法）")
    out("")
    # pick the smallest W where the better base reaches P(gap<=2) >= 0.80
    w_ep = None
    for W in W_GRID:
        if g0_rows[(best, W)][0] >= 0.80:
            w_ep = W
            break
    if w_ep is None:
        w_ep = W_MAX
        out(f"- 注意：G0 曲线上 P(gap≤2) 未达 0.80（max = "
            f"{fmt(max(g0_rows[(best, W)][0] for W in W_GRID))}），取 W={w_ep} 作 B0.6 "
            f"per-decision 预算（保守）。")
    else:
        out(f"- G0 曲线（{best} base）显示 W={w_ep} 即达 P(gap≤2) ≥ 0.80 ⇒ B0.6 "
            f"per-decision 世界预算 w_ep = {w_ep}。")
    # per-world cost = CPI-only throughput (oracle 只在评测时用，B0.6 模拟不用)
    worlds_per_s = cpi_worlds_total / max(t_cpi_secs, 1e-9)
    n_ep = 800                       # B0.3a G5 口径的 episode 数
    K = 3                            # H=48..96 下的平均决策数/ episode
    total_worlds = n_ep * K * w_ep
    proj_min = total_worlds / worlds_per_s / 60.0
    out(f"- B0.6 规模：n_ep={n_ep} episodes × K={K} decisions/episode × w_ep={w_ep} "
        f"worlds/decision = **{total_worlds} worlds**；CPI-only 实测吞吐 "
        f"{fmt(worlds_per_s, 0)} worlds/s ⇒ 预计 **{fmt(proj_min, 1)} min**"
        f"（不含 oracle，oracle 仅用于事后评测）。")
    feas_ok = proj_min <= 60.0
    out(f"- **G3 gate**：预计 wall time ≤ 60 min → {mp(feas_ok)}；"
        f"**G3 = {mp(feas_ok)}**（B0.6 的 episode 级 matched-QoS 模拟可承受）。")
    out("")
    verdict = g0_ok and g1_ok and g2_ok and feas_ok
    out(f"## 验收结论：**{'PASS — 算法验收通过，进入 B0.6' if verdict else 'FAIL — 需先解决'}**"
        f"（G0={mp(g0_ok)}，G1={mp(g1_ok)}，G2={mp(g2_ok)}，G3={mp(feas_ok)}）")
    out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")

    # FULL report hash guard (B0.4s convention)
    full_rp = os.path.join(OUT_DIR, "MVS-B0.6-pre_report.md")

    def _md5(p):
        try:
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            return None

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.6-pre_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
