"""MVS-B0.4a: Certified Policy Improvement (advice/011.md §3-§8).

Base by default; override only with certified evidence of improvement:
    a_inc^(0) = a_b = pi_b(x, h),
    a_inc -> c only when U_{c,a_inc} < 0   (Q_c^{pi_b} < Q_{a_inc}^{pi_b}),
so on the confidence event the executed chain is strictly decreasing in
Q^{pi_b} and V^{pi_CPI} <= V^{pi_b} (policy improvement over the finite acyclic
evidence DAG).  Episode-level delta: decision t spends
delta_t = 6 delta_episode / (pi^2 t^2); within a decision every possible pair
gets alpha = delta_t / P, so P(all executed overrides valid) >= 1 - delta_episode.

The base is the one-step conditional-VoI base (011 §7):
    Q_a^(1) = c_a + E[R_stop(X')|x,a],  a_VoI = argmin over A_h^+(x),
which is objective-consistent and automatically uses the header/setup cost,
the current posterior and the message resolution.

Gates (011 §8 — the four scientific questions):
  G0  fallback tail: empirical-best vs SNR-base vs VoI-base — E[gap],
      P(gap>2), P(gap>4) (does the fallback eliminate the uncertified tail?);
  G1  override benefit: P(override) and E[Q_{a_b} - Q_{a_override} | override];
  G2  certified override safety on the N=4 exact oracle (matched base):
      Q_{a_override} <= Q_{a_b}, violations + binomial U95;
  G3  VoI-base strength: VoI-base vs SNR-base vs CPI-executed vs
      empirical-best action quality.
All Q values are evaluated under the SAME base (the VoI-base) that the CPI
rollouts follow — the B0.4a exact oracle must match the rollout base.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import beta as beta_dist

from opmvs import GaussianDetectorModel, NestedQuantizer
from opmvs.rbl_cr import CRRBL, SNRDirectBase, exact_qa_pi_b
from opmvs.rbl_eb import CRRBLEB, VoIBase, CPI
from opmvs.sparse import z_code_b

GAMMA_A = [-1.0, 1.0, 3.0, 5.0]
SEED0 = 2026
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def random_state4(rng, seed_i):
    z = [0] * 4
    for u in range(int(rng.integers(0, 3))):
        i = int(rng.integers(0, 4))
        r = (1, 2, 4)[int(rng.integers(0, 3))]
        z[i] = z_code_b(r, int(rng.integers(0, 2 ** r)))
    return sum(int(z[i]) * (279 ** i) for i in range(4))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    SMOKE = args.smoke
    n_state = 30 if SMOKE else 100
    w_cpi = 1500 if SMOKE else 3000          # betting-mode CPI worlds per decision
    w_eb = 2000 if SMOKE else 6000           # formal-EB override rate subset
    n_eb = 6 if SMOKE else 15
    delta_episode = 0.05
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# O-PEF MVS-B0.4a — Certified Policy Improvement (base by default)")
    out("")
    out("> 依据 `advice/011.md` §3-§8。核心：**Base by default; override only with "
        "certified evidence of improvement.** a_inc^(0) = a_b（one-step conditional-VoI "
        "base），challenger c 只有在 U_{c,a_inc} < 0（pairwise CS 证明 Q_c^{π_b} < "
        "Q_{a_inc}^{π_b}）时才替换 incumbent；episode 级 δ：决策 t 花 δ_t = "
        "6δ_episode/(π²t²)，决策内每 pair α = δ_t/P ⇒ P(所有 override 有效) ≥ 1−δ_episode。")
    out("> 与 B0.4 全动作 ε-optimal 证书的区别：不再证明『该动作接近所有动作最优』，"
        "只需证明『该动作不比我会执行的 base 差』——base-anchored，O(|A|) 而非 O(|A|²) 的"
        "置信预算（011 §3/§9）。")
    out("")
    out(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   模式: {'SMOKE' if SMOKE else 'FULL'}")
    out("")

    model4 = GaussianDetectorModel(GAMMA_A)
    quants4 = [NestedQuantizer(i, model4, r_max=4, levels=(1, 2, 4)) for i in range(4)]
    bh = 16.0
    muM, muF = 256.0, 256.0 * np.exp(1.0)
    voi = VoIBase(bh)
    snr = SNRDirectBase(quants4, GAMMA_A, bh, eta_b=2.0, levels=(1, 2, 4))
    cr_voi = CRRBL(quants4, muM, muF, bh, voi, levels=(1, 2, 4), delta_c=1.0, seed=7)
    rng = np.random.default_rng(SEED0 + 4)

    # ------------------------------------------------------------ G0
    out("## 1. G0 — fallback tail：empirical-best vs SNR-base vs VoI-base（011 §8-1）")
    out("")
    stats = {"emp-best": [], "snr-base": [], "voi-base": []}
    for s in range(n_state):
        x = random_state4(rng, 400000 + s)
        Qp = exact_qa_pi_b(cr_voi, x, 40)
        if not Qp:
            continue
        R0 = cr_voi.pl.r_stop(x)
        Qmin = min(list(Qp.values()) + [R0])
        # empirical best (B0.4 pairwise planner exhaustion)
        eb = CRRBLEB(quants4, muM, muF, bh, voi, levels=(1, 2, 4),
                     delta_c=1.0, seed=17, cs_mode="eb")
        eb.cr.rng = np.random.default_rng(9000 + s)
        a_e, _ = eb.plan(x, 40, eps=8.0, delta=0.05, max_worlds=1500, n_min=100)
        q_e = R0 if a_e is None else Qp.get(a_e, np.inf)
        stats["emp-best"].append(max(0.0, q_e - Qmin))
        # SNR base action
        om = cr_voi.pl.omega(x)
        a_s = snr.act(cr_voi.pl, x, om)
        q_s = R0 if a_s is None else Qp.get(a_s, np.inf)
        stats["snr-base"].append(max(0.0, q_s - Qmin))
        # VoI base action
        a_v = voi.act(cr_voi.pl, x, om, h=40)
        q_v = R0 if a_v is None else Qp.get(a_v, np.inf)
        stats["voi-base"].append(max(0.0, q_v - Qmin))
    out("| fallback | E[gap] | P(gap>2) | P(gap>4) |")
    out("| --- | --- | --- | --- |")
    for k in ("emp-best", "snr-base", "voi-base"):
        g = np.array(stats[k])
        out(f"| {k} | {fmt(g.mean())} | {fmt(np.mean(g > 2))} | {fmt(np.mean(g > 4))} |")
    out("- 解读：VoI-base 是否消除 uncertified tail（P(gap>2) 相比 empirical-best 是否下降）；"
        "empirical-best 是 B0.4 未认证分支的执行行为（011 指出应结束）。")
    out("")

    # ------------------------------------------------------------ G1/G2
    out("## 2. G1/G2 — override 收益与 certified override 安全性（011 §8-2/§8-3）")
    out("")
    n_ov = 0
    n_ov_eb = 0
    n_safe = 0
    n_viol = 0
    gains = []
    gap_base = []
    gap_cpi = []
    n_tested = 0
    for s in range(n_state):
        x = random_state4(rng, 500000 + s)
        Qp = exact_qa_pi_b(cr_voi, x, 40)          # matched base oracle
        if not Qp:
            continue
        R0 = cr_voi.pl.r_stop(x)
        Qmin = min(list(Qp.values()) + [R0])
        om = cr_voi.pl.omega(x)
        a_b = voi.act(cr_voi.pl, x, om, h=40)
        qb = R0 if a_b is None else Qp.get(a_b, np.inf)
        cpi = CPI(quants4, muM, muF, bh, voi, levels=(1, 2, 4),
                  delta_c=1.0, seed=3, cs_mode="betting")
        cpi.cr.rng = np.random.default_rng(6000 + s)
        delta_t = 6.0 * delta_episode / (np.pi ** 2 * (s + 1) ** 2)   # 011 §5
        a_e, info = cpi.decide(x, 40, delta_t=delta_t, max_worlds=w_cpi)
        qe = R0 if a_e is None else Qp.get(a_e, np.inf)
        n_tested += 1
        gap_base.append(max(0.0, qb - Qmin))
        gap_cpi.append(max(0.0, qe - Qmin))
        if info["override"]:
            n_ov += 1
            gains.append(qb - qe)
            n_safe += int(qe <= qb + 1e-9)
            n_viol += int(qe > qb + 1e-9)
    out(f"- betting-mode CPI（{w_cpi} worlds/decision，δ_episode={delta_episode}，"
        f"决策 t 用 δ_t=6δ/(π²t²)）：")
    out(f"  - **P(override) = {fmt(n_ov / max(n_tested, 1))}**；"
        f"**E[Q_{{a_b}} − Q_{{a_override}} | override] = {fmt(np.mean(gains) if gains else 0.0)}**")
    out(f"  - **certified override 安全性**（N=4 exact，matched base）：safe={n_safe}，"
        f"violations={n_viol}；单侧 95% binomial U95 = "
        f"{fmt(float(beta_dist.ppf(0.95, n_viol + 1, max(n_safe, 1))))}"
        f"（certified override 只在 U<0 时执行——理论 P(violation) ≤ Σδ_t）")
    out(f"  - 执行质量：base E[gap]={fmt(np.mean(gap_base))}（P(>2)={fmt(np.mean(np.array(gap_base) > 2))}）"
        f"→ CPI E[gap]={fmt(np.mean(gap_cpi))}（P(>2)={fmt(np.mean(np.array(gap_cpi) > 2))}）")
    # formal-EB override rate on a subset
    n_ov_eb = 0
    for s in range(n_eb):
        x = random_state4(rng, 550000 + s)
        Qp = exact_qa_pi_b(cr_voi, x, 40)
        if not Qp:
            continue
        R0 = cr_voi.pl.r_stop(x)
        om = cr_voi.pl.omega(x)
        a_b = voi.act(cr_voi.pl, x, om, h=40)
        qb = R0 if a_b is None else Qp.get(a_b, np.inf)
        cpi = CPI(quants4, muM, muF, bh, voi, levels=(1, 2, 4),
                  delta_c=1.0, seed=3, cs_mode="eb")
        cpi.cr.rng = np.random.default_rng(7000 + s)
        a_e, info = cpi.decide(x, 40, delta_t=delta_t, max_worlds=w_eb)
        qe = R0 if a_e is None else Qp.get(a_e, np.inf)
        n_ov_eb += int(info["override"] and qe <= qb + 1e-9)
    out(f"  - formal PrPl-EB 路径（{w_eb} worlds，n={n_eb}）：safe override rate = "
        f"{fmt(n_ov_eb / max(n_eb, 1))}——formal 证书保守，override 需更大预算"
        f"（011 §9 预期：override 比 full best-arm 容易，但 EB 的 peeling 开销仍在）。")
    out("")

    # ------------------------------------------------------------ G3
    out("## 3. G3 — VoI-base 强度：VoI-base vs SNR-base vs CPI vs empirical-best（011 §8-4）")
    out("")
    out("- 综合：若 VoI-base ≈ CPI，论文应强调 feedback granularity + conditional VoI；"
        "若 CPI 在 VoI-base 之上有明确增益，pairwise certified planning 才有独立算法价值。"
        "（G0 的 voi-base 行 = VoI-base 本身；G1 的 CPI 行 = certified improvement 之上；"
        "两者之差即 pairwise planner 的边际价值。）")
    out("")

    out(f"总耗时: {time.time() - t_start:.1f}s")
    out("")
    # B0.4s: smoke -> separate path; FULL hash guard
    full_rp = os.path.join(OUT_DIR, "MVS-B0.4a_report.md")
    import hashlib

    def _md5(p):
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    full_hash_before = _md5(full_rp)
    rp = os.path.join(OUT_DIR, "smoke", "MVS-B0.4a_report.md") if SMOKE else full_rp
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if SMOKE and full_hash_before is not None:
        assert _md5(full_rp) == full_hash_before, \
            "SMOKE run modified the FULL report — path separation broken"
    print(f"\n[report] -> {rp}")


if __name__ == "__main__":
    main()
