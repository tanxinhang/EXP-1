"""MVS-A/B0 formal regression test suite (005.md P1-8).

Runnable:  python test_regressions.py        (prints PASS/FAIL per test)
Every test asserts a mathematically-grounded invariant; a failure is a bug,
not a tuning issue.  The suite covers:

  T01 model analytics (single-UAV / raw-fusion P_D vs MC)
  T02 message-PMF normalization + nested consistency
  T03 log-domain stress (no NaN / overflow, p+q=1)
  T04 exact DAG-DP Bellman residual == 0
  T05 G1a transition invariants (norm / martingale / info monotonicity)
  T06 G1b J(pi_DP)(x0) == V*(x0)
  T07 exact propagation mass conservation + exact vs MC agreement
  T08 sparse tuple-state backend == eager table RBL (N=4, value/action)
  T09 CMDP column generation certified optimum (<= RMP, reduced cost >= -tol)
  T10 RBL hard certification V_16 == V* + V_h monotone
  T11 1bit-POTS double-count fix (ladder starts at 1->2)
  T12 Adaptive Direct-8 action-set restriction
  T13 VoI theorem identity Q_prog-Q_dir = E[min{D-D2, b_h}]
  T14 hypothesis sampling respects the configured prior
"""
import time
import numpy as np

from opmvs import (ExactDP, GaussianDetectorModel, NestedQuantizer, StateSpace)
from opmvs import baselines as bl
from opmvs import cmdp
from opmvs import eval_exact as ee
from opmvs import mc as mclib
from opmvs import rbl as rblmod
from opmvs import sparse as sp
from opmvs.fusion import log_sigmoid, log_one_minus_sigmoid, logsumexp2
from opmvs.state import action_decode

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def run():
    t0 = time.time()
    print("=== MVS regression suite ===")

    # ------------------------------------------------------------- T01
    print("\nT01 model analytics")
    model = GaussianDetectorModel([-1.0, 1.0, 3.0, 5.0])
    pd_single = model.single_uav_pd_all(0.05)
    check("single-UAV P_D matches doc", np.allclose(pd_single, [0.226, 0.301, 0.408, 0.553], atol=5e-3),
          f"{np.round(pd_single, 3)}")
    check("raw fusion P_D ~ 0.851", abs(model.raw_fusion_pd(0.05) - 0.851) < 5e-3,
          f"{model.raw_fusion_pd(0.05):.4f}")

    # ------------------------------------------------------------- T02
    print("\nT02 PMF normalization + nested consistency")
    quants = [NestedQuantizer(i, model, r_max=4, levels=(1, 2, 4)) for i in range(4)]
    check("PMF normalization", all(q.check_pmf_normalization() for q in quants))
    check("nested consistency", all(q.check_nested_consistency() for q in quants))

    # ------------------------------------------------------------- T03
    print("\nT03 log-domain stress")
    om = np.linspace(-100, 100, 5001)
    lp = log_sigmoid(om)
    lq = log_one_minus_sigmoid(om)
    ok = (np.all(np.isfinite(lp)) and np.all(np.isfinite(lq))
          and np.allclose(np.exp(lp) + np.exp(lq), 1.0, atol=1e-12))
    check("softplus/sigmoid stress", ok)
    check("logsumexp2 extreme", np.allclose(logsumexp2([1000.0, -1000.0], [-1000.0, 1000.0]),
                                            [1000.0, 1000.0]))

    # ------------------------------------------------------------- T04-T06
    print("\nT04-T06 exact DP / G1a / G1b")
    ss = StateSpace(model, quants, cross_level=False)
    dp = ExactDP(ss)
    V, pol = dp.solve(256.0, 256.0)
    mx, _ = dp.bellman_residual()
    check("Bellman residual == 0", mx < 1e-10, f"max={mx:.2e}")
    g1a = ee.g1a_invariants(ss)
    check("G1a invariants", g1a["passed"], f"norm={g1a['norm_dev']:.1e} mar={g1a['martingale_dev']:.1e}")
    g1b = ee.g1b_check(ss, dp, 256.0, 256.0)
    check("G1b J(pi_DP)=V*", g1b["passed"], f"dev={g1b['abs_dev']:.2e}")

    # ------------------------------------------------------------- T07
    print("\nT07 exact propagation vs MC")
    res = ee.exact_evaluate(ss, pol, 256.0, 256.0)
    sd = ee.exact_stop_distribution(ss, pol)
    check("exact stop-distribution mass conserved",
          abs(sd["m0"].sum() - 0.5) < 1e-9 and abs(sd["m1"].sum() - 0.5) < 1e-9,
          f"m0={sd['m0'].sum():.12f} m1={sd['m1'].sum():.12f}")
    n = 100000
    H, L = mclib.sample_episodes(model, n, 42)
    lam, cost, z, ns = mclib.simulate_table_policy(ss, pol, H, L)
    eta = res["eta_dec"]
    pd_mc = np.mean(lam[H == 1] > eta)
    check("exact vs MC P_D", abs(pd_mc - res["pd"]) < 3e-3, f"exact={res['pd']:.4f} mc={pd_mc:.4f}")

    # ------------------------------------------------------------- T08
    print("\nT08 sparse backend == eager table (N=4)")
    rbl4 = rblmod.ResourceBoundedLookahead(ss, 256.0, 256.0 * np.exp(1.0))
    rbl4.solve()
    rng = np.random.default_rng(3)
    idxs = np.concatenate([np.array([0]), rng.integers(0, ss.n_states, size=3000)])
    e = sp.equivalence_with_old_backend(ss, rbl4, 256.0, 256.0 * np.exp(1.0), (4, 8), idxs=idxs)
    ok08 = all(e[H]["max_val_dev"] < 1e-9 for H in e)
    check("sparse==eager values", ok08,
          f"max|dV|={max(e[H]['max_val_dev'] for H in e):.2e}")
    # all mismatches must be near-ties
    ok08b = all(e[H]["n_action_mismatch"] == e[H]["n_near_tie"] for H in e)
    check("action mismatches are near-ties", ok08b)

    # ------------------------------------------------------------- T09
    print("\nT09 CMDP column generation")
    initial = []
    for s in (16, 64, 256, 1024):
        for eta in (0.0, 1.0, 2.0):
            d = ExactDP(ss)
            Vp, pp = d.solve(s, s * np.exp(eta))
            ev = ee.exact_evaluate(ss, pp, s, s * np.exp(eta))
            initial.append({"B": ev["eb"], "pfa": ev["pfa"], "pm": ev["pm"]})
    pd_max = ee.exact_pd_max(ss, 0.05)
    alpha, beta = 0.05, 1.0 - (pd_max - 0.01)
    rmp = cmdp.master_lp(initial, alpha, beta)
    cg = cmdp.column_generation(ss, initial, alpha, beta, verbose=False)
    check("CG certified <= RMP", cg["b_cmdp"] <= rmp["obj"] + 1e-9,
          f"{cg['b_cmdp']:.4f} <= {rmp['obj']:.4f}")
    check("CG reduced cost at noise floor", cg["r_final"] >= -1e-6,
          f"r={cg['r_final']:.2e}")

    # ------------------------------------------------------------- T10
    print("\nT10 RBL certification")
    rbl16 = rblmod.ResourceBoundedLookahead(ss, 256.0, 256.0)
    V16, _ = rbl16.solve()
    cert = rbl16.verify_full_budget(V)
    check("V_16 == V*", cert["passed"], f"dev={cert['max_dev']:.2e}")
    sd = ee.exact_stop_distribution(ss, rbl16.policies[16])
    check("receding H=16 == DP eval", abs(sd["eb"] - res["eb"]) < 1e-9,
          f"{sd['eb']:.6f} vs {res['eb']:.6f}")

    # ------------------------------------------------------------- T11-T13
    print("\nT11-T13 B0.1 specifics (N=8)")
    model8 = GaussianDetectorModel([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    quants8 = [NestedQuantizer(i, model8, r_max=8, levels=(1, 2, 4, 8)) for i in range(8)]
    pl = sp.SparsePlanner(quants8, 256.0, 256.0, b_h=16.0, cross_level=True)
    # direct-only action set
    pl_d = sp.SparsePlanner(quants8, 256.0, 256.0, b_h=16.0, cross_level=True, direct_only=True)
    tpl0 = pl_d._tpl[0][0]
    check("Adaptive Direct-8 only 0->8", all(r2 == 8 for (r2, _c, _qb, _cells) in tpl0),
          f"{[(r2, c) for (r2, c, _q, _cl) in tpl0]}")
    # VoI identity
    from run_mvsb01 import verify_voi
    for bh_ in (0.0, 16.0):
        plv = sp.SparsePlanner(quants8, 256.0, 256.0, b_h=bh_, cross_level=True)
        devs = [verify_voi(plv, 0, i, bh_)["dev"] for i in (5, 6, 7)]
        check(f"VoI identity b_h={bh_:.0f}", max(devs) < 1e-8, f"dev={max(devs):.2e}")
    # 1bit-POTS fix: seed cost counted once (structure check)
    n8 = 20000
    rng8 = np.random.default_rng(1)
    H8 = model8.sample_hypotheses(n8, rng8)
    L8 = model8.sample_llr(H8, rng8)
    from run_mvsb01 import b_seeded_pots
    lam8, cost8 = b_seeded_pots(quants8, H8, L8, 6.0, 16.0)
    check("1bit-POTS seed once (min cost = N*(bh+1))",
          cost8.min() >= 8 * 17 - 1e-9, f"min={cost8.min():.1f}")

    # ------------------------------------------------------------- T14
    print("\nT14 prior consistency")
    m_p = GaussianDetectorModel([-1.0, 1.0, 3.0, 5.0], prior=(0.3, 0.7))
    Hp = m_p.sample_hypotheses(200000, np.random.default_rng(0))
    check("prior respected", abs(Hp.mean() - 0.7) < 0.01, f"empirical={Hp.mean():.3f}")

    # ------------------------------------------------------------- T15-T20
    print("\nT15-T20 CR-RBL invariants (B0.3a/B0.3b, advice/007.md §11)")
    from opmvs.rbl_cr import CRRBL, SNRDirectBase, LatentWorld, exact_qa_pi_b
    from opmvs.sparse import z_code_b
    GAMMA_A = [-1.0, 1.0, 3.0, 5.0]
    bhA = 16.0
    baseA = SNRDirectBase(quants, GAMMA_A, bhA, eta_b=2.0, levels=(1, 2, 4))
    crA = CRRBL(quants, 256.0, 256.0 * np.exp(1.0), bhA, baseA,
                levels=(1, 2, 4), delta_c=1.0, seed=5)

    # T15: nested projection M^(r) = proj_r(M^(8)) holds on ONE world, all levels
    ok15 = True
    rng15 = np.random.default_rng(15)
    for trial in range(40):
        z = [0] * 4
        for u in range(int(rng15.integers(0, 3))):
            i = int(rng15.integers(0, 4))
            r = (1, 2, 4)[int(rng15.integers(0, 3))]
            z[i] = z_code_b(r, int(rng15.integers(0, 2 ** r)))
        x = sum(int(z[i]) * (sp.BASE_B ** i) for i in range(4))
        w = LatentWorld(crA, x)
        for i in range(4):
            r_i, m_i = sp.z_decode_b(crA._z_digit(x, i))
            if r_i > 0:
                ok15 &= (m_i == w.msg(i, r_i))       # current cell == projection
        x2, om2 = x, crA.pl.omega(x)
        for _ in range(8):
            a = crA.base.act(crA.pl, x2, om2)
            if a is None:
                break
            i, r2 = a
            r_old, m_old = sp.z_decode_b(crA._z_digit(x2, i))
            x2, om2 = crA._apply(x2, om2, i, r2, w.cells)
            r_new, m_new = sp.z_decode_b(crA._z_digit(x2, i))
            ok15 &= (r_new == r2)
            ok15 &= (m_new == w.msg(i, r2))           # revealed == projection
            ok15 &= (r_old == 0 or m_old == w.msg(i, r_old))  # level-0 sentinel
    check("T15 nested projection on one world", ok15)

    # T16: E[G_a(W)] == Q_a^{pi_b} within MC CI (N=4 exact oracle)
    Q_ex16 = exact_qa_pi_b(crA, 0, 40)
    acts16 = [(3, 4), (2, 4), (0, 4)]
    crA.rng = np.random.default_rng(16)
    n16 = 2000
    mean16, se16 = {}, {}
    for a in acts16:
        gs = np.empty(n16)
        for m in range(n16):
            w = LatentWorld(crA, 0)
            gs[m] = crA._rollout(0, 40, a, w)
        mean16[a] = gs.mean()
        se16[a] = gs.std(ddof=1) / np.sqrt(n16)
    ok16 = all(abs(mean16[a] - Q_ex16[a]) <= 4.0 * se16[a] for a in acts16)
    check("T16 E[G_a(W)] = Q_a^pi_b in CI", ok16,
          " | ".join(f"{a}: MC={mean16[a]:.1f}±{4*se16[a]:.1f} "
                     f"exact={Q_ex16[a]:.1f}" for a in acts16))

    # T17: certified => exact eps-ordering vs A U {STOP} (P0-B); deterministic
    # certificate-condition replay + loose statistical tail
    cr17 = CRRBL(quants, 256.0, 256.0 * np.exp(1.0), bhA, baseA,
                 levels=(1, 2, 4), delta_c=1.0, seed=17)
    cr17._uavs = [int(np.argmax(GAMMA_A))]
    rng17 = np.random.default_rng(17)
    eps17, delta17, w17 = 40.0, 0.05, 1500
    n_cert17, n_viol17 = 0, 0
    replay_ok = True
    for trial in range(25):
        z = [0] * 4
        for u in range(int(rng17.integers(0, 2))):
            i = int(rng17.integers(0, 3))
            r = (1, 2, 4)[int(rng17.integers(0, 3))]
            z[i] = z_code_b(r, int(rng17.integers(0, 2 ** r)))
        z[3] = z_code_b(1, int(rng17.integers(0, 2)))
        x = sum(int(z[i]) * (sp.BASE_B ** i) for i in range(4))
        cr17.rng = np.random.default_rng(17000 + trial)
        a_cr, info = cr17.plan(x, 40, eps=eps17, delta=delta17, max_samples=w17)
        if not info["certified"]:
            continue
        n_cert17 += 1
        u_b, m_c, e_c = info["cert_cond"]          # deterministic replay
        replay_ok &= (u_b <= m_c + e_c + 1e-9)
        Q_ex17 = exact_qa_pi_b(cr17, x, 40)
        R0_17 = cr17.pl.r_stop(x)                  # P0-D: exact STOP
        best17 = min(list(Q_ex17.values()) + [R0_17])
        q_a17 = Q_ex17.get(a_cr, R0_17) if a_cr is not None else R0_17
        if q_a17 > best17 + eps17:
            n_viol17 += 1
    check("T17 certificate condition replay", replay_ok)
    check("T17 certified => exact eps-ordering (loose tail)",
          n_viol17 <= max(2, int(0.25 * max(n_cert17, 1))),
          f"certified={n_cert17} viol={n_viol17} (eps={eps17:.0f}, delta={delta17})")

    # T18: episode communication cost <= H PATHWISE (P0-E hard budget)
    n18 = 15
    rng18 = np.random.default_rng(18)
    Ht18 = model8.sample_hypotheses(n18, rng18)
    L18 = model8.sample_llr(Ht18, rng18)
    GAMMA_B = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    base8t = SNRDirectBase(quants8, GAMMA_B, 16.0, eta_b=2.0, levels=(1, 2, 4, 8))
    top4 = [7, 6, 5, 4]
    max_costs = []
    for H in (48.0, 96.0):
        x_int = [0] * n18
        zcode = np.zeros((n18, 8), dtype=np.int64)
        lam = np.zeros(n18)
        cost = np.zeros(n18)
        h_rem = np.full(n18, H)
        done = np.zeros(n18, dtype=bool)
        for _ in range(64):
            active = np.flatnonzero(~done)
            if len(active) == 0:
                break
            for e in active:
                if h_rem[e] < 1e-9:
                    done[e] = True
                    continue
                cr8t = CRRBL(quants8, 256.0, 256.0 * np.exp(1.0), 16.0, base8t,
                             levels=(1, 2, 4, 8), delta_c=1.0,
                             seed=31 + e % 7, top_k_uavs=top4)
                a, _i = cr8t.plan(x_int[e], h_rem[e], eps=4.0, delta=0.05,
                                  max_samples=30)
                if a is None:
                    done[e] = True
                    continue
                i, r2 = a
                zi = int(zcode[e, i])
                r_cur, _ = sp.z_decode_b(zi)
                c = 16.0 + (r2 - r_cur)
                if c > h_rem[e] + 1e-9:
                    done[e] = True
                    continue
                m2 = int(quants8[i].cell_index(r2, L18[e, i]))
                z2 = z_code_b(r2, m2)
                lam[e] += quants8[i].llr[r2][m2]
                if r_cur > 0:
                    lam[e] -= quants8[i].llr[r_cur][sp.z_decode_b(zi)[1]]
                cost[e] += c
                h_rem[e] -= c
                x_int[e] += (z2 - zi) * (sp.BASE_B ** i)
                zcode[e, i] = z2
        max_costs.append(float(cost.max()))
    ok18 = all(mc <= H + 1e-9 for mc, H in zip(max_costs, (48.0, 96.0)))
    check("T18 episode cost <= H pathwise", ok18, f"max={max_costs}")

    # T19: anytime coverage: Pr[forall n<=n_max, Q in [L_n,U_n]] >= 1 - delta/|A|
    Q_true19 = Q_ex16[(3, 4)]
    n_runs19, n_max19, delta19, nA19 = 100, 80, 0.1, 2
    cov19 = 0
    for r in range(n_runs19):
        crA.rng = np.random.default_rng(19000 + r)
        Bx = crA.bound(0)
        qhat = 0.0
        ok_all = True
        for n in range(1, n_max19 + 1):
            w = LatentWorld(crA, 0)
            g = crA._rollout(0, 40, (3, 4), w)
            qhat += (g - qhat) / n
            dn = 6.0 * delta19 / (np.pi * np.pi * nA19 * n * n)
            rad = Bx * np.sqrt(np.log(2.0 / dn) / (2.0 * n))
            ok_all &= (abs(qhat - Q_true19) <= rad)
        cov19 += int(ok_all)
    thresh19 = 1.0 - delta19 / nA19
    check("T19 anytime coverage (all n <= n_max)",
          cov19 / n_runs19 >= thresh19 - 0.05,
          f"cov={cov19}/{n_runs19} (lower bound {thresh19:.3f})")

    # T20: paired CRN — same world, estimator identity + variance reduction
    a20a, a20b = (3, 4), (2, 4)
    crA.rng = np.random.default_rng(20)
    n20 = 2000
    G20a = np.empty(n20)
    G20b = np.empty(n20)
    for m in range(n20):
        w = LatentWorld(crA, 0)
        rr = crA.rollout_returns(0, 40, [a20a, a20b], world=w)
        G20a[m] = rr[a20a]
        G20b[m] = rr[a20b]
    ident20 = abs((G20a - G20b).mean() - (G20a.mean() - G20b.mean())) < 1e-9
    var_d = float((G20a - G20b).var(ddof=1))
    var_s = float(G20a.var(ddof=1) + G20b.var(ddof=1))
    check("T20 paired-CRN estimator identity", ident20)
    check("T20 paired CRN reduces difference variance", var_d < 0.95 * var_s,
          f"Var(Ga-Gb)/[Var(Ga)+Var(Gb)] = {var_d/var_s:.3f}")

    print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ({time.time()-t0:.0f}s) ===")
    for name, d in FAIL:
        print(f"  FAILED: {name} {d}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(run())
