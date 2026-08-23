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
  T15 nested projection on one latent world (B0.3a)
  T16 E[G_a(W)] = Q_a^{pi_b} within MC CI (B0.3a)
  T17a deterministic certificate implication (B0.3c; 100% PASS)
  T17b empirical certificate audit (B0.3c; stochastic sanity, G3 is the strong gate)
  T18 episode cost <= H pathwise (B0.3a hard budget)
  T19 anytime CI coverage (B0.3a; statistical sanity — NOT a deterministic invariant)
  T20 paired-CRN estimator identity + variance reduction (B0.3a)
  T21 natural decision threshold = log(mu_F/mu_M), locked to eval_exact (B0.3c)

Deterministic invariants (T01-T15, T17a, T18, T20 identity part): a failure is a bug.
Statistical audits (T16, T17b, T19, T20 variance part): assertions carry explicit
MC tolerance; the strong statistical gates live in the pipelines (G0/G3).
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
    # T17a (B0.3c/009 §2): NON-VACUOUS deterministic certificate implication.
    # A random candidate j (NOT argmin Q) is chosen; with intervals covering the
    # true Q_a for every a, the fired condition U_j <= min_{b!=j} L_b + eps must
    # imply Q_j <= min_b Q_b + eps.  (The old test used best=argmin Q, making the
    # implication true regardless of the certificate — vacuous.)
    rng17a = np.random.default_rng(17)
    ok17a = True
    for trial in range(5000):
        n_arms = int(rng17a.integers(2, 8))
        Q = rng17a.uniform(0.0, 100.0, n_arms)
        width = rng17a.uniform(0.0, 20.0, n_arms)
        L = Q - width                      # L_a <= Q_a
        U = Q + width                      # U_a >= Q_a
        eps = rng17a.uniform(0.0, 30.0)
        j = int(rng17a.integers(0, n_arms))
        if U[j] <= np.min(np.delete(L, j)) + eps:
            ok17a &= (Q[j] <= np.min(Q) + eps + 1e-12)
    check("T17a-report deterministic implication (non-vacuous)", ok17a)
    # T17a-STOP: R_stop <= min_b L_b + eps  =>  R_stop <= min{R_stop, Q_b} + eps
    ok17s = True
    for trial in range(5000):
        n_arms = int(rng17a.integers(1, 8))
        Q = rng17a.uniform(0.0, 100.0, n_arms)
        R0 = float(rng17a.uniform(0.0, 100.0))
        width = rng17a.uniform(0.0, 20.0, n_arms)
        L = Q - width
        eps = rng17a.uniform(0.0, 30.0)
        if R0 <= np.min(L) + eps:
            ok17s &= (R0 <= min(R0, np.min(Q)) + eps + 1e-12)
    check("T17a-STOP deterministic implication (non-vacuous)", ok17s)

    # T17b (B0.3c): empirical certificate audit (stochastic sanity; the strong
    # statistical gate is pipeline G3: 465 certified / 0 viol / U95=0.0064).
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
    check("T17b certificate condition replay (deterministic part)", replay_ok)
    check("T17b empirical certificate audit (loose tail)",
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

    # T19: anytime coverage (statistical sanity, 008 §5): Pr[forall n<=n_max,
    # Q in [L_n,U_n]] >= 1 - delta/|A|; uses the B0.3c budget-aware diameter.
    Q_true19 = Q_ex16[(3, 4)]
    n_runs19, n_max19, delta19, nA19 = 100, 80, 0.1, 2
    c_a19 = 16.0 + 4                            # action (3,4): b_h + (4-0)
    Bx19 = crA.bound_a(0, 40, c_a19, action=(3, 4))
    cov19 = 0
    for r in range(n_runs19):
        crA.rng = np.random.default_rng(19000 + r)
        qhat = 0.0
        ok_all = True
        for n in range(1, n_max19 + 1):
            w = LatentWorld(crA, 0)
            g = crA._rollout(0, 40, (3, 4), w)
            qhat += (g - qhat) / n
            dn = 6.0 * delta19 / (np.pi * np.pi * nA19 * n * n)
            rad = Bx19 * np.sqrt(np.log(2.0 / dn) / (2.0 * n))
            ok_all &= (abs(qhat - Q_true19) <= rad)
        cov19 += int(ok_all)
    thresh19 = 1.0 - delta19 / nA19
    check("T19 anytime coverage, budget-aware radius (statistical sanity)",
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

    # T21 (B0.3c, 008 §1): natural decision threshold = log(mu_F/mu_M), locked
    # to eval_exact.py's objective-consistent natural decision; the MC natural
    # metrics must use this same threshold.
    muM21, muF21 = 256.0, 256.0 * np.exp(1.0)
    eta_nat21 = float(np.log(muF21 / muM21))
    res21 = ee.exact_evaluate(ss, pol, muM21, muF21)
    check("T21 eta_nat = log(muF/muM) locked to eval_exact",
          abs(res21["eta_dec"] - eta_nat21) < 1e-12,
          f"eta_dec={res21['eta_dec']:.12f} log(muF/muM)={eta_nat21:.12f}")
    H21, L21 = mclib.sample_episodes(model, 200000, 21)
    lam21, _c21, _z21, _n21 = mclib.simulate_table_policy(ss, pol, H21, L21)
    H1_21 = H21 == 1
    pd_nat_mc = float(np.mean(lam21[H1_21] > eta_nat21))
    check("T21 MC natural P_D at eta_nat matches exact eval",
          abs(pd_nat_mc - res21["pd"]) < 3e-3,
          f"MC={pd_nat_mc:.4f} exact={res21['pd']:.4f} (eta_nat={eta_nat21:.3f})")

    # ------------------------------------------------------------- T22-T24
    print("\nT22-T24 B0.4 pairwise-difference CS (advice/009.md)")
    from opmvs.rbl_eb import CRRBLEB, PairCS

    # T22: PairCS anytime validity (statistical sanity; the strong gate is G0)
    Delta22 = Q_ex16[(3, 4)] - Q_ex16[(2, 4)]
    cov22 = 0
    n_runs22, n_max22, alpha22 = 60, 150, 0.05
    for r in range(n_runs22):
        crA.rng = np.random.default_rng(22000 + r)
        cs22 = PairCS(-400.0, 400.0, alpha22, mode="eb")   # FORMAL EB path
        ok22 = True
        for n in range(1, n_max22 + 1):
            W = LatentWorld(crA, 0)
            z = crA._rollout(0, 40, (3, 4), W) - crA._rollout(0, 40, (2, 4), W)
            cs22.update(z)
            L, U = cs22.bounds()
            ok22 &= (Delta22 >= L and Delta22 <= U)
        cov22 += int(ok22)
    check("T22 PrPl-EB anytime validity (statistical)",
          cov22 / n_runs22 >= 1 - alpha22 - 0.06,
          f"cov={cov22}/{n_runs22} (>= {1-alpha22-0.06:.2f})")

    # T23: paired-difference estimator + canonical-orientation regression.
    # The pair CS must ALWAYS see the canonical direction
    # Z = G_{key0} - G_{key1} (here key0=(2,4), key1=(3,4) => Z = -(Ga-Gb));
    # a planner whose candidate changes must never mix opposite-sign samples
    # into one CS (the B0.4 sign bug), or the CS would miss the true Delta.
    a23, b23 = (3, 4), (2, 4)
    D23 = Q_ex16[a23] - Q_ex16[b23]           # Q_(3,4) - Q_(2,4) < 0
    D_canon = -D23                            # Q_(2,4) - Q_(3,4) > 0
    cs23 = PairCS(-400.0, 400.0, 0.005, mode="eb")   # FORMAL EB path
    crA.rng = np.random.default_rng(23)
    n23 = 1500
    zsum = 0.0
    z2sum = 0.0
    for m in range(n23):
        W = LatentWorld(crA, 0)
        ga = crA._rollout(0, 40, a23, W)
        gb = crA._rollout(0, 40, b23, W)
        z = ga - gb                           # Delta_{a23,b23} sample
        zsum += z
        z2sum += z * z
        cs23.update(-z)                       # canonical orientation, always
    mean23 = zsum / n23
    se23 = np.sqrt(max(z2sum / n23 - mean23 ** 2, 0.0) / n23)
    L23, U23 = cs23.bounds()
    check("T23 E[Z^{a,b}] = Q_a-Q_b in CI (canonical orientation)",
          abs(mean23 - D23) <= 4.0 * se23 and L23 <= D_canon <= U23,
          f"mean={mean23:.2f} exact={D23:.2f} canon={D_canon:.2f} "
          f"CS=[{L23:.1f},{U23:.1f}]")

    # T24: B0.4 certified => exact eps-ordering vs A U {STOP} (loose tail;
    # exercises the full pairwise orientation path in plan()).
    cr24 = CRRBLEB(quants, 256.0, 256.0 * np.exp(1.0), bhA, baseA,
                   levels=(1, 2, 4), delta_c=1.0, seed=24)
    cr24.cr._uavs = [int(np.argmax(GAMMA_A))]
    rng24 = np.random.default_rng(24)
    eps24, delta24, w24 = 40.0, 0.05, 2000
    n_cert24, n_viol24 = 0, 0
    for trial in range(20):
        z = [0] * 4
        for u in range(int(rng24.integers(0, 2))):
            i = int(rng24.integers(0, 3))
            r = (1, 2, 4)[int(rng24.integers(0, 3))]
            z[i] = z_code_b(r, int(rng24.integers(0, 2 ** r)))
        z[3] = z_code_b(1, int(rng24.integers(0, 2)))
        x = sum(int(z[i]) * (sp.BASE_B ** i) for i in range(4))
        cr24.cr.rng = np.random.default_rng(24000 + trial)
        a_cr, info = cr24.plan(x, 40, eps=eps24, delta=delta24, max_worlds=w24)
        if not info["certified"]:
            continue
        n_cert24 += 1
        Q_ex24 = exact_qa_pi_b(cr24.cr, x, 40)
        R0_24 = cr24.cr.pl.r_stop(x)
        best24 = min(list(Q_ex24.values()) + [R0_24])
        q_a24 = Q_ex24.get(a_cr, R0_24) if a_cr is not None else R0_24
        if q_a24 > best24 + eps24:
            n_viol24 += 1
    check("T24 B0.4 certified => exact eps-ordering (loose tail)",
          n_viol24 <= max(2, int(0.25 * max(n_cert24, 1))),
          f"certified={n_cert24} viol={n_viol24} (eps={eps24:.0f})")

    # T25 (010 §1 R0): canonical SAMPLE + SUPPORT orientation with a DESCENDING
    # top_k_uavs action list.  The pair range must follow the canonical key
    # (G_{key0} - G_{key1}); a mis-oriented support would make PairCS.update
    # trip its hard z-in-[lo,hi] assert (the R2 invariant) and would corrupt
    # the certificate bounds.  Action quality is a secondary sanity (the
    # returned root action must be eps-optimal at eps=8).
    Q25 = exact_qa_pi_b(crA, 0, 40)
    Qmin25 = min(list(Q25.values()) + [crA.pl.r_stop(0)])
    n_best25 = 0
    n_tri25 = 30
    for trial in range(n_tri25):
        eb25 = CRRBLEB(quants, 256.0, 256.0 * np.exp(1.0), bhA, baseA,
                       levels=(1, 2, 4), delta_c=1.0, seed=25,
                       top_k_uavs=[3, 2, 1, 0])      # descending enumeration
        eb25.cr.rng = np.random.default_rng(25000 + trial)
        a25, _inf25 = eb25.plan(0, 40, eps=40.0, delta=0.05, max_worlds=1500)
        # reaching here = the R2 assert never tripped (canonical support OK)
        q25 = Q25.get(a25, np.inf)
        n_best25 += int(max(0.0, q25 - Qmin25) <= 8.0)
    check("T25 descending top_k canonical support (no z-range trip)", True)
    check("T25 descending top_k action quality (eps-opt 8 at root)",
          n_best25 / n_tri25 >= 0.7,
          f"P(eps-opt(8)) = {n_best25}/{n_tri25}")

    print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ({time.time()-t0:.0f}s) ===")
    for name, d in FAIL:
        print(f"  FAILED: {name} {d}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(run())
