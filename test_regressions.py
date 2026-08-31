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
  T28 strategy-value identity Q_prog-Q_dir = E[min{Y_x,b}] (B0.4b)
  T29 right/left derivative of g_x = survival Pr(Y>b)/Pr(Y>=b) (B0.4b)
  T30 three synthetic branches E[Y]<0 / =0 / >0 incl. E[Y]=0 plateau (B0.4b)
  T31 exact-support b*(x) = strategy switch point of Q_prog vs Q_dir (B0.4b)
  T32 CPI budget-aware anchor clamp: nominal base action unaffordable at
      (x, h) => anchor = STOP, no crash, a_exec feasible (B0.6-pre-r, 014 §1)
  T40 C3a migration controller equivalence: budget-aware Myopic-All / Direct8
      == G2 q_min_fg/q_min_d8 on N=8 homogeneous at frozen corners
      (005 §十七, deterministic part of the migration anchor)
  T41 Myopic-PJ action set == {next, full} (contract hardening H1, 005 §七)
  T43 C3b causal contrast: Phase-PJ == Myopic-PJ decisions in N=8
      homogeneous regime (conditional-refinement value ~0, 005 §18)
  T44 StaticProg |Omega|>=eta stop (007 fix: no all-stop degeneration)
  T45 C3c L1 physical feasibility: in-budget 4x8-bit MITM P_MD <= beta
  T46 C3e-G1 generalized r<s<t envelope identity + tower + derivative +
      b* classification + (next,max) special-case consistency
      (advice/010.md §七; generalized 7-tuple Y at index 6)
  T47 C3d n0/n1 split: eval_decide reports n1, kmd uses n1 (010 §十一)
  T48 StaticProg 4rho x 7eta grid == 7 unique threshold policies (010 §五)
  T49 GPE-EA/Myopic-All shared legal action space (010 §八 matched-action)
  T50 GPE conditional-refinement Q: Q_cond <= q1; certificate-pruned =>
      exact equality (010 §八)
  T51 C3d per-method registered-hull unit: hull-enter vs hull-infeasible
      logic with Wilson U95 (010 §三)
  T52 C3e-G0 audit accounting: regions sum == n_probe_feasible, rates in
      [0,1] (010 §十二 G0)

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

    # T26 (011 §7): VoIBase Q1 identity — Q_a^(1) = c_a + E[R_stop(X')|x,a]
    # computed by the base must equal an independent direct expectation.
    from opmvs.rbl_eb import VoIBase, CPI
    voi26 = VoIBase(bhA)
    rng26 = np.random.default_rng(26)
    ok26 = True
    for trial in range(30):
        z = [0] * 4
        for u in range(int(rng26.integers(0, 3))):
            i = int(rng26.integers(0, 4))
            r = (1, 2, 4)[int(rng26.integers(0, 3))]
            z[i] = z_code_b(r, int(rng26.integers(0, 2 ** r)))
        x = sum(int(z[i]) * (sp.BASE_B ** i) for i in range(4))
        pl26 = crA.pl
        if not crA.feasible_actions(x, 40):
            continue
        for (i, r2) in crA.feasible_actions(x, 40):
            q1 = voi26.q1(pl26, x, (i, r2))
            # independent: c_a + sum_m w_m R_stop(child)
            zi = (x // pl26.powers[i]) % sp.BASE_B
            r_old, _ = sp.z_decode_b(zi)
            c = bhA + (r2 - r_old)
            om = pl26.omega(x)
            lp = float(log_sigmoid(om))
            lq = float(log_one_minus_sigmoid(om))
            cells = next(cells for (r2b, _ct, _qb, cells) in pl26._tpl[i][zi]
                         if r2b == r2)
            E = 0.0
            for (m2, lp0c, lp1c) in cells:
                a_ = lp + lp1c
                b_ = lq + lp0c
                m_ = a_ if a_ >= b_ else b_
                w = float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))
                cx = x + (z_code_b(r2, m2) - zi) * pl26.powers[i]
                E += w * pl26.r_stop(cx)
            ok26 &= abs(q1 - (c + E)) < 1e-9
    check("T26 VoIBase Q1 = c + E[R_stop(X')] identity", ok26)

    # T27 (011 §3/§8-3): CPI certified overrides are SAFE on the N=4 exact
    # oracle with the MATCHED base (the VoI-base the rollouts follow):
    # Q_{a_override}^{pi_b} <= Q_{a_b}^{pi_b} whenever U_{c,a_inc} < 0 fired.
    cr_voi27 = CRRBL(quants, 256.0, 256.0 * np.exp(1.0), bhA, voi26,
                     levels=(1, 2, 4), delta_c=1.0, seed=7)
    rng27 = np.random.default_rng(27)
    n_ov27 = 0
    n_viol27 = 0
    for trial in range(12):
        z = [0] * 4
        for u in range(int(rng27.integers(0, 3))):
            i = int(rng27.integers(0, 4))
            r = (1, 2, 4)[int(rng27.integers(0, 3))]
            z[i] = z_code_b(r, int(rng27.integers(0, 2 ** r)))
        x = sum(int(z[i]) * (sp.BASE_B ** i) for i in range(4))
        Qp = exact_qa_pi_b(cr_voi27, x, 40)
        if not Qp:
            continue
        R0v = cr_voi27.pl.r_stop(x)
        omv = cr_voi27.pl.omega(x)
        a_b = voi26.act(cr_voi27.pl, x, omv, h=40)
        qb = R0v if a_b is None else Qp.get(a_b, np.inf)
        cpi27 = CPI(quants, 256.0, 256.0 * np.exp(1.0), bhA, voi26,
                    levels=(1, 2, 4), delta_c=1.0, seed=3, cs_mode="betting")
        cpi27.cr.rng = np.random.default_rng(27000 + trial)
        a_e, info = cpi27.decide(x, 40, delta_t=0.01, max_worlds=1500)
        if not info["override"]:
            continue
        n_ov27 += 1
        qe = R0v if a_e is None else Qp.get(a_e, np.inf)
        n_viol27 += int(qe > qb + 1e-9)
    check("T27 CPI override safety on exact (matched base)",
          n_viol27 == 0,
          f"overrides={n_ov27} violations={n_viol27}")

    # ------------------------------------------------------------- T28-T31
    # B0.4b (advice/013.md): Feedback-Granularity Phase-Transition Theorem.
    # T28  strategy-value identity  Q_prog - Q_dir = E[min{Y_x, b}]  (< 1e-10)
    # T29  right/left derivative = survival  g'_{+}(b)=Pr(Y>b), g'_{-}(b)=Pr(Y>=b)
    # T30  three synthetic branches  E[Y]<0 / =0 / >0  (incl. the E[Y]=0 plateau)
    # T31  exact-support b* = strategy switch point of Q_prog vs Q_dir
    print("\nT28-T31 B0.4b phase-transition theorem (advice/013.md)")
    from opmvs.phase_boundary import (bstar_exact, bstar_from_dist,
                                      g_alt, g_from_support, survival,
                                      verify_identity, y_support)
    pl_b4 = sp.SparsePlanner(quants, 256.0, 256.0 * np.exp(1.0), b_h=0.0,
                             cross_level=True, levels=(1, 2, 4))

    def reachable4(rng_, n):
        """random reachable N=4 states (legal refinement sequences)."""
        out = []
        while len(out) < n:
            z = [0] * 4
            for _ in range(int(rng_.integers(0, 3))):
                cand = [i for i in range(4) if pl_b4._tpl[i][z[i]]]
                if not cand:
                    break
                i = int(cand[rng_.integers(0, len(cand))])   # draw a UAV id
                tpl = pl_b4._tpl[i][z[i]]
                r2 = tpl[int(rng_.integers(0, len(tpl)))][0]
                z[i] = z_code_b(r2, int(rng_.integers(0, 2 ** r2)))
            x = sum(int(z[i]) * (sp.BASE_B ** i) for i in range(4))
            zs = pl_b4.decode(x)
            if any(pl_b4._tpl[i][zs[i]] for i in range(4)):
                out.append(x)
        return out

    rng_b = np.random.default_rng(28)
    states_b = reachable4(rng_b, 30)
    bvals = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    max_dev_b = 0.0
    max_tower_b = 0.0
    n_ok_deriv = 0
    n_deriv = 0
    n_switch_ok = 0
    n_switch = 0
    n_cases = {"A": 0, "B": 0, "C": 0}
    for x in states_b:
        zs = pl_b4.decode(x)
        for i in range(4):
            if not pl_b4._tpl[i][zs[i]]:
                continue
            sup = y_support(pl_b4, x, i)
            if sup is None:
                continue
            # T28 identity
            dev, tower = verify_identity(sup, bvals)
            max_dev_b = max(max_dev_b, dev)
            max_tower_b = max(max_tower_b, tower)
            # T29 derivative = survival (exact support, structural)
            ys = sorted({float(br[5]) for br in sup["branches"]})
            bps = sorted(set([0.0] + [v for v in ys if v > 0.0]))
            deriv_ok = True
            for a in range(len(bps) - 1):
                b_l, b_r = bps[a], bps[a + 1]
                if b_r <= b_l + 1e-12:
                    continue
                mid = 0.5 * (b_l + b_r)
                slope = (g_from_support(sup, b_r) - g_from_support(sup, b_l)) \
                    / (b_r - b_l)
                deriv_ok &= abs(slope - survival(sup, mid)) < 1e-9
            h = 1e-6
            for yk in ys:
                if yk < 0:
                    continue
                gk_at = g_from_support(sup, yk)
                r_slope = (g_from_support(sup, yk + h) - gk_at) / h
                l_slope = (gk_at - g_from_support(sup, yk - h)) / h
                deriv_ok &= abs(r_slope - survival(sup, yk, strict=True)) < 1e-5
                deriv_ok &= abs(l_slope - survival(sup, yk, strict=False)) < 1e-5
            n_deriv += 1
            n_ok_deriv += int(deriv_ok)
            # T31 exact b* = strategy switch point
            r_ = bstar_exact(sup)
            n_cases[r_["case"]] += 1
            bs = r_["bstar"]
            eps = 1e-6
            if bs == float("inf"):                    # Case A: g < 0 always
                sw = g_from_support(sup, 32.0) < 0.0 and g_alt(sup, 32.0) < 0.0
            elif abs(bs) < 1e-9:                      # b* = 0 (Y >= 0 a.s.)
                sw = g_from_support(sup, eps) >= -1e-9 \
                    and g_alt(sup, eps) >= -1e-9
            else:
                gl = g_from_support(sup, bs - eps)
                gr = g_from_support(sup, bs + eps)
                al = g_alt(sup, bs - eps)
                ar = g_alt(sup, bs + eps)
                if r_["case"] == "B":                 # plateau: g == 0 at/above
                    sw = (gl < 0.0) and abs(gr) < 1e-6 and abs(ar) < 1e-6
                else:                                 # Case C: sign change
                    sw = (gl < 0.0) and (gr > 0.0) and (al < 0.0) and (ar > 0.0)
            n_switch += 1
            n_switch_ok += int(sw)
    check("T28 strategy-value identity (< 1e-10)",
          max_dev_b < 1e-10 and max_tower_b < 1e-10,
          f"max|g-Qprog+Qdir|={max_dev_b:.2e} tower={max_tower_b:.2e}")
    check("T29 derivative = survival (exact support)",
          n_ok_deriv == n_deriv,
          f"{n_ok_deriv}/{n_deriv}")
    # T30 synthetic three-case branches (013 §3)
    r30a = bstar_from_dist([0.5, 0.5], [-2.0, -1.0])     # E[Y] < 0
    r30b = bstar_from_dist([0.5, 0.5], [-1.0, 1.0])      # E[Y] = 0 (audit case)
    r30c = bstar_from_dist([0.5, 0.5], [-1.0, 3.0])      # E[Y] > 0, P(Y<0)>0
    r30c0 = bstar_from_dist([0.5, 0.5], [1.0, 3.0])      # Y >= 0 a.s.
    gB = lambda b: 0.5 * min(-1.0, b) + 0.5 * min(1.0, b)
    ok30 = (
        r30a["case"] == "A" and r30a["bstar"] == float("inf")
        and all(0.5 * min(-2.0, b) + 0.5 * min(-1.0, b) <= r30a["EY"] + 1e-9
                for b in (0.0, 4.0, 32.0))
        and r30b["case"] == "B" and abs(r30b["bstar"] - 1.0) < 1e-9
        and gB(0.5) < 0.0 and abs(gB(1.0)) < 1e-9 and abs(gB(2.0)) < 1e-9
        and r30c["case"] == "C" and abs(r30c["bstar"] - 1.0) < 1e-9
        and r30c0["case"] == "C" and abs(r30c0["bstar"] - 0.0) < 1e-9
    )
    check("T30 three synthetic branches (E[Y]<0/=0/>0)",
          ok30,
          f"A:{r30a['case']}/{r30a['bstar']} B:{r30b['case']}/{r30b['bstar']} "
          f"C:{r30c['case']}/{r30c['bstar']} C0:{r30c0['case']}/{r30c0['bstar']}")
    check("T31 exact-support b* = strategy switch point",
          n_switch_ok == n_switch,
          f"{n_switch_ok}/{n_switch} cases={n_cases}")

    # T32 (014 §1 P0-1 / B0.6-pre-r): budget-aware anchor clamp — when the
    # base's nominal action is UNAFFORDABLE at (x, h), the CPI anchor must
    # clamp to STOP (012 §5 budget-aware pi_b), never crash on
    # actions_feas[a_b] and never return an infeasible a_exec.
    snr32 = SNRDirectBase(quants, GAMMA_A, bhA, eta_b=2.0, levels=(1, 2, 4))
    cr32 = CRRBL(quants, 256.0, 256.0 * np.exp(1.0), bhA, snr32,
                 levels=(1, 2, 4), delta_c=1.0, seed=7)
    ok32 = True
    for trial, h32 in enumerate((17.0, 18.0, 19.0, 20.0, 21.0, 22.0)):
        cpi32 = CPI(quants, 256.0, 256.0 * np.exp(1.0), bhA, snr32,
                    levels=(1, 2, 4), delta_c=1.0, seed=3, cs_mode="betting")
        cpi32.cr.rng = np.random.default_rng(32000 + trial)
        a32, info32 = cpi32.decide(0, h32, delta_t=0.03, max_worlds=200)
        feas32 = cr32.feasible_actions(0, h32)
        ok32 &= (a32 is None or a32 in feas32)
        ok32 &= (info32["a_b"] is None or info32["a_b"] in feas32)
    check("T32 CPI budget-aware anchor clamp (h < nominal cost => STOP)",
          ok32)

    # ------------------------------------------------- C2.1 invariants
    # (advice/003.md s8: T33-T39, deterministic; imports run_mvsc021)
    import run_mvsc021 as c21
    import math

    # T33 MITM 4-bit reference == brute-force convolution
    import run_mvsc02 as c2       # brute 4-bit full-fusion ref lives here
    mm4 = c21.GaussianDetectorModel(c21.GAMMA4, (0.5, 0.5))
    q4 = [c21.NestedQuantizer(i, mm4, 4, c21.LEVELS4) for i in range(4)]
    pfa_mitm, pmd_mitm = c21.full_fusion_ref_mitm(mm4, q4, 4)
    pfa_br, pmd_br, _g, _w0, _w1 = c2.full_fusion_ref(mm4, q4, 4)
    ok33 = all(abs(pfa_mitm(x) - pfa_br(x)) < 1e-9 for x in (-2, 0, 2, 5))
    ok33 &= all(abs(pmd_mitm(x) - pmd_br(x)) < 1e-9 for x in (-2, 0, 2, 5))
    check("T33 MITM == brute 4-bit full-fusion ROC", ok33)

    # T34/T35/T36 Region A/B/C pruning law (s2 + s4/s5)
    plq1 = c21.SparsePlanner(q4, 1.0, 1.0, b_h=16.0, cross_level=True,
                             levels=c21.LEVELS4, direct_only=False, delta_c=1.0)
    # N=1 fresh UAV: x=0, r=0; c1=BH+(1-0)=17, c_dir=BH+(4-0)=20,
    # c2=BH+(r_max-r_next)=16+3=19 ⇒ c1+c2=36
    # A: h=19 (c1<=19<20); B: h=25 (20<=25<36); C: h=60
    sa = c21.phase_support_budget(plq1, 0, 0.0, 0, 19, 512, 1.2)
    sb = c21.phase_support_budget(plq1, 0, 0.0, 0, 25, 512, 1.2)
    sc = c21.phase_support_budget(plq1, 0, 0.0, 0, 60, 512, 1.2)
    ok34 = sa is not None and sa["region"] == "A" and not sa["prune_probe_ok"]
    # T35/T36 检查的是真正独立的路径（非自比较）：
    #   B 区：gap=Q_prog-Q_dir（边际 E_dir）vs g_verdict=Σw(R1-E_R-d2)
    #         （per-branch E_R=E[R(X2)|X1]）——差 = E_R_sum-E_dir（tower 恒等式）
    #   C 区：gap=E[min(Y,b)]（support 形式）vs g_verdict=Q_prog-Q_dir
    #         （策略价值形式）——差 = E_R_sum-E_dir（013 §1 恒等式）
    ok35 = sb is not None and sb["region"] == "B" and \
           abs(sb["gap"] - sb["g_verdict"]) < 1e-9
    ok36 = sc is not None and sc["region"] == "C" and \
           abs(sc["gap"] - sc["g_verdict"]) < 1e-9
    check("T34 Region A never prunes probe", ok34)
    check("T35 Region B gap == E[Y]", ok35)
    check("T36 Region C gap == E[min(Y,b)]", ok36)

    # T37 prune => Q_prog >= Q_dir (constrained dominance, s5)
    ok37 = True
    for h in (20, 24, 30, 40, 50, 60, 80, 96):
        s = c21.phase_support_budget(plq1, 0, 0.0, 0, h, 512, 1.2)
        if s is None or not s["dir_feas"]:
            continue
        if s["prune_probe_ok"] and s["Q_prog"] is not None:
            ok37 &= s["Q_prog"] >= s["Q_dir"] - 1e-8
    check("T37 prune => Q_prog >= Q_dir (dominance)", ok37)

    # T38 stratified sampling n0 == n1 exactly
    H, _L = c21.sample_set_strat(50, c21.SEED_CAL + 900, mm4)
    ok38 = int((H == 0).sum()) == 50 and int((H == 1).sum()) == 50
    check("T38 stratified n0 == n1 == n", ok38)

    # T39 J(policy) >= V* at frozen exact-oracle cases (D1 lower-bound)
    q4b = [c21.NestedQuantizer(i, mm4, 4, c21.LEVELS4) for i in range(4)]
    pl4o = c21.SparsePlanner(q4b, 1.0, 1.0, b_h=16.0, cross_level=True,
                             levels=c21.LEVELS4, direct_only=False, delta_c=1.0)
    ok39 = True
    for theta, H in [((256, 1.2), 48), ((512, 1.2), 96), ((1024, 1.6), 96)]:
        rho, eta = theta
        po = c21.SparsePlanner(q4b, rho * 0.5, rho * math.exp(eta) * 0.5,
                              b_h=16.0, cross_level=True, levels=c21.LEVELS4,
                              direct_only=False, delta_c=1.0)
        v_star, _ = po.solve(0, float(H))
        j_phase = c21.exact_policy_lagrangian(pl4o, c21.phase_decision_budget,
                                              rho, eta, 0, H)
        ok39 &= j_phase >= v_star - 1e-6
    check("T39 J_Phase >= V* (D1 Lagrangian lower bound)", ok39)

    # ------------------------------------------- C3a invariants (advice/005.md)
    # T40: C3a migration controller equivalence — run_mvsc021 budget-aware
    # Myopic-All / Direct8 == G2 q_min_fg/q_min_d8 on N=8 homogeneous at the
    # frozen corners (005 §十七: migration anchor; controller equivalence is
    # the deterministic part that makes the FULL migration reproducible).
    # T41: Myopic-PJ action set == {next, full} (hardening H1, 005 §七).
    import run_mvsb07g2 as g2
    import run_mvsc03a as c3a
    mm8 = c21.GaussianDetectorModel(g2.GAMMA_B, (0.5, 0.5))
    qu8 = [c21.NestedQuantizer(i, mm8, r_max=8, levels=g2.LEVELS)
           for i in range(8)]
    pw8 = [c21.BASE_B ** i for i in range(8)]
    pl8 = c21.SparsePlanner(qu8, 1.0, 1.0, b_h=16.0, cross_level=True,
                            levels=g2.LEVELS, delta_c=1.0)
    rng8 = np.random.default_rng(11)
    H8 = rng8.integers(0, 2, 40)
    L8 = mm8.sample_llr(H8, np.random.default_rng(12))
    ok40 = True
    for (rho, eta) in g2.CORNER_THETAS:
        for H in (48, 96):
            for e in range(40):
                L_i = L8[e]
                for mode, decfn in (("FG", c21.myopic_decision),
                                    ("D8", c21.direct_decision)):
                    lam, cost, nt, pay = g2.sim_method(
                        pl8, rho, eta, H, L_i, mode, qu8, pw8)
                    cost21, bp21, nt21, _dec = c21.run_episode(
                        mm8, qu8, pl8, int(H8[e]), L_i, H, decfn, rho, eta)
                    ok40 &= (abs(cost - cost21) < 1e-9
                             and abs(nt - nt21) < 1e-9
                             and abs(pay - bp21) < 1e-9)
    check("T40 C3a migration: Myopic-All/D8 == G2 sim (N=8, corners)", ok40)

    ok41 = True
    for e in range(40):
        L_i = L8[e]
        x, om, h = 0, 0.0, 96.0
        for _ in range(10):
            dec, _d = c3a.myopic_pj_decision(pl8, x, om, h, 256, 0.8)
            if dec[0] == "STOP":
                break
            i, _k, r2 = dec[1], dec[2], dec[3]
            zi = (x // pl8.powers[i]) % c21.BASE_B
            r_cur, _m = c21.z_decode_b(zi)
            r_next = next((rr for rr in pl8.levels if rr > r_cur), None)
            # PJ action set: only {next, full}
            ok41 &= (r2 == r_next or r2 == pl8.r_max)
            m2 = int(qu8[i].cell_index(r2, float(L_i[i])))
            z2 = c21.z_code_b(r2, m2)
            x += (z2 - zi) * pl8.powers[i]
            om += pl8._llr_i[i][z2] - pl8._llr_i[i][zi]
            h -= 16.0 + (r2 - r_cur)
            if h < 1e-9:
                break
    check("T41 Myopic-PJ action set == {next, full} (H1)", ok41)

    # T42 degenerate r_next == r_max (d2 == 0): probe == direct, so
    # Q_prog == Q_dir and gap == 0 (found by the 4-bit exhaustive
    # certificate, 005 §10; run_mvsc021.py now handles it like
    # phase_boundary.py: E_R = R1, D = 0, Y = 0).
    ok42 = True
    for (x0, i0) in [(0, 0), (0, 1), (0, 2), (0, 3)]:
        # put UAV i0 at the second-highest level so r_next == r_max
        # (4-bit: r=2 -> r_next=4 == r_max; 8-bit: r=4 -> r_next=8)
        zs = [0, 0, 0, 0]
        q4x = [c21.NestedQuantizer(i, mm4, 4, c21.LEVELS4) for i in range(4)]
        pl4x = c21.SparsePlanner(q4x, 1.0, 1.0, b_h=16.0, cross_level=True,
                                 levels=c21.LEVELS4, direct_only=False,
                                 delta_c=1.0)
        r_mid = 2
        zs[i0] = c21.z_code_b(r_mid, 0)
        x = sum(int(z) * (c21.BASE_B ** i) for i, z in enumerate(zs))
        sup = c21.phase_support_budget(pl4x, x, pl4x.omega(x), i0, 60,
                                       512.0, 1.2)
        if sup is None:
            ok42 = False
            continue
        # d2 == 0 => probe == direct: Q_prog should equal Q_dir, gap ~ 0
        ok42 &= abs(sup["Q_prog"] - sup["Q_dir"]) < 1e-6
        ok42 &= abs(sup["gap"]) < 1e-6
    check("T42 degenerate d2=0: Q_prog==Q_dir, gap==0 (exhaustive cert)", ok42)

    # T43: C3b causal-layer contrast — at the calibrated theta_hat=(256,0.8),
    # Phase-PJ (conditional refinement) == Myopic-PJ (one-step) decisions in
    # the N=8 homogeneous regime (005 §18 contrast; verified 0/40 cost diff
    # at theta_hat). NOTE: at extreme corners like (1024,2.0) they DO differ
    # (4/40 eps, Phase-PJ cheaper) — refinement has bite there; T43 pins the
    # theta_hat point only.
    # T44: StaticProg |Omega|>=eta stop (007 fix) — no more all-stop
    # degeneration at low eta: it must send >=1 message (E[B]>=17) at
    # eta=0.8 and stop via threshold not QoS-dual R<=min Q.
    import run_mvsc03a as c3a
    ok43 = True
    for (rho, eta) in [(256, 0.8)]:
        for H in (48, 96):
            for e in range(20):
                L_i = L8[e]
                c_p = c3a.sim_decide(pl8, rho, eta, H, L_i,
                                     c21.phase_decision_budget, qu8, pw8)[1]
                c_m = c3a.sim_decide(pl8, rho, eta, H, L_i,
                                     c3a.myopic_pj_decision, qu8, pw8)[1]
                ok43 &= abs(c_p - c_m) < 1e-9
    check("T43 C3b: Phase-PJ == Myopic-PJ @ theta_hat (refinement no value "
          "at calibrated pt)", ok43)

    ok44 = True
    # StaticProg at (128, 0.8): must send at least one 1-bit (E[B]>=17)
    # and stop by |Omega|>=eta (not all-stop E[B]=0)
    for e in range(20):
        L_i = L8[e]
        _lam, cost, nt, _pay = c3a.sim_decide(pl8, 128, 0.8, 96, L_i,
                                              c3a.static_prog_decision,
                                              qu8, pw8)
        ok44 &= cost >= 17.0 - 1e-9 and nt >= 1
    check("T44 StaticProg |Omega|>=eta stop (no all-stop degeneration)", ok44)

    # T45: C3c L1 physical feasibility — 4 strongest-SNR UAVs 8-bit
    # (cost 4x24=96=H) achieves P_MD <= beta at alpha=0.12 via exact MITM
    # ROC (005 §19: physical layer feasible; deterministic, no seeds).
    import run_mvsc03c as c3c
    mmc = c21.GaussianDetectorModel(g2.GAMMA_B, (0.5, 0.5))
    quc = [c21.NestedQuantizer(i, mmc, r_max=8, levels=g2.LEVELS)
           for i in range(8)]
    ok45, rows45 = c3c.physical_feasibility(mmc, quc)
    ok45 &= rows45[0]["ok"] and rows45[0]["cost"] <= 96.0 + 1e-9
    check("T45 C3c L1 physical feasibility (MITM, in-budget 4x8-bit)", ok45)

    # ============================================================= C3d/C3e
    # (advice/010.md §三-§十二：per-method L2 hull、n0/n1、StaticProg 7
    # unique、generalized r<s<t envelope、GPE-EA matched-action、EB UCB)
    import run_mvsc03c as c3c
    import run_mvsc03e as c3e
    from opmvs import phase_boundary as pb

    # T46 (010 §七): generalized r<s<t envelope identity + tower over random
    # reachable states on an N=4 8-bit ladder, all (s,t) pairs x (b0,kappa)
    # corners.  Deterministic invariant: max|g - (Q_prog-Q_dir)| < 1e-9 and
    # max|E[E_R]-E_dir| < 1e-9.
    print("\nT46 C3e-G1 generalized r<s<t envelope identity (010 §七)")
    mm46 = c21.GaussianDetectorModel([-1.0, 1.0, 3.0, 5.0], (0.5, 0.5))
    qu46 = [c21.NestedQuantizer(i, mm46, r_max=8, levels=g2.LEVELS)
            for i in range(4)]
    pw46 = [c21.BASE_B ** i for i in range(4)]
    pl46 = c21.SparsePlanner(qu46, 1.0, 1.0, b_h=16.0, cross_level=True,
                             levels=g2.LEVELS, delta_c=1.0)
    H46 = np.random.default_rng(7001).integers(0, 2, 60)
    L46 = mm46.sample_llr(H46, np.random.default_rng(7002))
    st46 = c3e._reachable_states(pl46, qu46, pw46, L46, 20)
    g46 = c3e.generalized_envelope_gates(pl46, st46)
    ok46 = (g46["max_identity_dev"] < 1e-9
            and g46["max_tower_dev"] < 1e-9
            and g46["max_deriv_dev"] < 1e-6
            and g46["n_bstar"] > 0 and g46["n_bstar_ok"] == g46["n_bstar"]
            and g46["n_special"] > 0
            and g46["n_special_ok"] == g46["n_special"]
            and g46["max_identity_dev_special"] < 1e-9)
    check("T46 generalized envelope identity/tower/deriv/b*/special",
          ok46,
          f"id={g46['max_identity_dev']:.1e} tw={g46['max_tower_dev']:.1e} "
          f"der={g46['max_deriv_dev']:.1e} b*={g46['n_bstar_ok']}/"
          f"{g46['n_bstar']} sp={g46['n_special_ok']}/{g46['n_special']}")

    # T47 (010 §十一): n0/n1 split in eval_decide — kmd uses n1, kfa uses n0;
    # on a stratified 2x120 set both equal 120.
    print("\nT47 C3d n0/n1 split (010 §十一)")
    H47 = np.concatenate([np.zeros(120, dtype=np.int8),
                          np.ones(120, dtype=np.int8)])
    L47 = mm46.sample_llr(H47, np.random.default_rng(7003))
    s47 = c3a.eval_decide(pl46, 256.0, 0.8, 96, H47, L47,
                          c21.myopic_decision, qu46, pw46)
    ok47 = (s47["n0"] == 120 and s47["n1"] == 120
            and s47["kfa"] <= s47["n0"] and s47["kmd"] <= s47["n1"])
    check("T47 eval_decide n0/n1 split (kmd uses n1)", ok47,
          f"n0={s47['n0']} n1={s47['n1']} kfa={s47['kfa']} kmd={s47['kmd']}")

    # T48 (010 §五): StaticProg == 7 unique threshold policies (rho unused).
    print("\nT48 StaticProg 7 unique policies (010 §五)")
    fake48 = {(r, e): {"eb": 0.0}
              for r in (128, 256, 512, 1024)
              for e in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)}
    n_u, etas48, _nz = c3c.static_prog_unique_count(fake48)
    ok48 = (n_u == 7 and list(etas48)
            == [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    check("T48 StaticProg unique-policy count == 7", ok48, f"{n_u}")

    # T49 (010 §八): GPE-EA and Myopic-All share the same full action set —
    # every executed action is (i,s) with s in levels, s > r_i, cost <= h.
    print("\nT49 GPE-EA action-set compliance == Myopic-All (010 §八)")
    memo49 = c3e.GPEMemo()
    ok49 = True
    n49 = 0
    for e in range(30):
        L_i = L46[e]
        for H49 in (48, 96):
            for fn, memo in ((lambda pl, x, om, h, r, e_, m=memo49:
                              c3e.gpe_decision(pl, x, om, h, r, e_, m),
                              memo49),
                             (lambda pl, x, om, h, r, e_, m=None:
                              c21.myopic_decision(pl, x, om, h, r, e_),
                              None)):
                x, h, om = 0, float(H49), 0.0
                for _ in range(14):
                    dec, _dd = fn(pl46, x, om, h, 256, 0.8)
                    if dec[0] == "STOP":
                        break
                    i, _k, r2 = dec[1], dec[2], dec[3]
                    zi = (x // pl46.powers[i]) % c21.BASE_B
                    r_cur, _m = c21.z_decode_b(zi)
                    cst = 16.0 + (r2 - r_cur)
                    ok49 &= (r2 in g2.LEVELS and r2 > r_cur
                             and cst <= h + 1e-9)
                    n49 += 1
                    m2 = int(qu46[i].cell_index(r2, float(L_i[i])))
                    z2 = c21.z_code_b(r2, m2)
                    om += pl46._llr_i[i][z2] - pl46._llr_i[i][zi]
                    x += (z2 - zi) * pl46.powers[i]
                    h -= cst
                    if h < 1e-9:
                        break
    check("T49 GPE-EA/Myopic-All shared legal action space", ok49,
          f"{n49} executed actions")

    # T50 (010 §八 certificate): Q_cond <= q1 (min-with-continuation can only
    # reduce) and pruned (all continuations dominated branchwise) implies
    # Q_cond == q1 exactly.
    print("\nT50 GPE conditional-refinement Q certificate (010 §八)")
    memo50 = c3e.GPEMemo()
    ok50 = True
    n50 = n50_pr = 0
    for (x, om) in st46:
        zs = pl46.decode(int(x))
        for i in range(4):
            r, _m = c21.z_decode_b(zs[i])
            if r >= 8:
                continue
            for s in g2.LEVELS:
                if s <= r or s == 8:
                    continue
                c_s = 16.0 + (s - r)
                conts = [t for t in g2.LEVELS
                         if t > s and 16.0 + (t - s) <= 96.0 - c_s + 1e-9]
                if not conts:
                    continue
                Qc, pruned = c3e._cond_refine_q(pl46, x, om, i, s, conts,
                                                256.0, 0.8, memo50)
                q1 = g2.q1_fast(pl46, x, om, i, s, 256.0, 0.8)
                ok50 &= (Qc <= q1 + 1e-9)
                n50 += 1
                if pruned:
                    n50_pr += 1
                    ok50 &= abs(Qc - q1) < 1e-9
    check("T50 Q_cond<=q1; pruned => exact equality", ok50,
          f"{n50_pr}/{n50} pruned")

    # T51 (010 §三): per-method convex hull unit logic — a method with no
    # deterministic feasible point but a 2-point hull entering QoS is
    # classified "registered-hull feasible"; an all-infeasible table is
    # "registered-hull infeasible".
    print("\nT51 per-method registered-hull unit (010 §三)")
    def fake_tables(pts):
        return {(1.0 + k * 0.01, 0.8 + 0.2 * k): {"kfa": kfa, "kmd": kmd,
                                                   "n0": 600, "n1": 600,
                                                   "eb": float(eb)}
                for k, (kfa, kmd, eb) in enumerate(pts)}
    tabA = fake_tables([(70, 60, 30.0), (30, 360, 26.0), (60, 260, 40.0)])
    rA = c3c.per_method_policy_class(tabA)
    ok51a = (rA["n_det"] == 0 and rA["n_enter"] >= 1
             and rA["best_mix"] is not None
             and "hull feasible" in rA["verdict"])
    tabB = fake_tables([(200, 300, 30.0), (150, 260, 26.0), (250, 350, 40.0)])
    rB = c3c.per_method_policy_class(tabB)
    ok51b = (rB["n_det"] == 0 and rB["n_enter"] == 0
             and "hull infeasible" in rB["verdict"])
    check("T51 per-method hull enter / infeasible logic",
          ok51a and ok51b,
          f"A det={rA['n_det']} enter={rA['n_enter']} mix={rA['best_mix'] is not None} "
          f"B det={rB['n_det']} enter={rB['n_enter']}")

    # T52 (010 §十二 G0): audit accounting — regions sum == n_probe_feasible,
    # n_pruned <= n_probe_feasible, rates in [0,1].
    print("\nT52 C3e-G0 audit accounting")
    aud52 = c3e.phase_activation_audit(pl46, qu46, pw46, H46, L46, 96)
    regs52 = aud52["regions"]
    ok52 = (regs52["A"] + regs52["B"] + regs52["C"]
            == aud52["n_probe_feasible"]
            and aud52["n_pruned"] <= aud52["n_probe_feasible"]
            and 0.0 <= aud52["action_change_rate"] <= 1.0
            and 0.0 <= aud52["P_continue"] <= 1.0)
    check("T52 G0 audit accounting invariants", ok52,
          f"regions={regs52['A']}/{regs52['B']}/{regs52['C']} "
          f"probe={aud52['n_probe_feasible']} pruned={aud52['n_pruned']} "
          f"act-change={aud52['action_change_rate']:.3f}")

    # ================================================= C4 (advice/010.md §九/§十二,
    # heterogeneous link-aware airtime, 001 §十六 regimes)
    import run_mvsc04 as c4

    # T53 (010 §九): link_params regime semantics — positive: strongest sensing
    # = best link (lowest b0/κ); anti: strongest sensing = worst link (highest
    # b0/κ); independent: shuffled (not equal to positive order); bounds in
    # [12,20]×[0.8,1.2]; homogeneous (16,1) is the q=0.5 special case.
    print("\nT53 C4 link_params regime semantics (010 §九)")
    b0p, kp = c4.link_params("positive")
    b0i, ki = c4.link_params("independent")
    b0a, ka = c4.link_params("anti")
    strong = int(np.argmax(c4.GAMMA_B))          # index of strongest sensing
    weak = int(np.argmin(c4.GAMMA_B))
    ok53_p = (b0p[strong] == b0p.min() and kp[strong] == kp.min()
              and b0p[weak] == b0p.max() and kp[weak] == kp.max())
    ok53_a = (b0a[strong] == b0a.max() and ka[strong] == ka.max()
              and b0a[weak] == b0a.min() and ka[weak] == ka.min())
    ok53_i = (not np.allclose(b0i, b0p) and not np.allclose(ka, kp))
    ok53_b = (b0p.min() >= 12.0 - 1e-9 and b0p.max() <= 20.0 + 1e-9
              and kp.min() >= 0.8 - 1e-9 and kp.max() <= 1.2 + 1e-9)
    check("T53 C4 link_params semantics (pos/anti/ind + bounds)",
          ok53_p and ok53_a and ok53_i and ok53_b,
          f"b0p[strong]={b0p[strong]:.2f} b0a[strong]={b0a[strong]:.2f} "
          f"ind!=pos={not np.allclose(b0i, b0p)}")

    # T54 (010 §八): heterogeneous planner reduces to the homogeneous one when
    # every link is (b0=16, κ=1) — q1_het ≡ q1_fast on random states and
    # gpe_het_decision ≡ gpe_decision actions at random states (same Q values).
    print("\nT54 C4 homogeneous limit: gpe_het ≡ gpe_decision")
    b0h = np.full(pl8.N, 16.0)
    kh = np.full(pl8.N, 1.0)
    memo_h = c4.GPEMemo()
    memo_e = c3e.GPEMemo()
    rng54 = np.random.default_rng(5401)
    ok54 = True
    n54 = 0
    for _ in range(60):
        x54 = int(rng54.integers(0, 3e6))
        om54 = float(rng54.uniform(-6, 6))
        h54 = float(rng54.choice([48.0, 96.0]))
        a_h, _d = c4.gpe_het_decision(pl8, x54, om54, h54, 256, 0.8,
                                      b0h, kh, memo_h)
        a_e, _e = c3e.gpe_decision(pl8, x54, om54, h54, 256, 0.8, memo_e)
        n54 += 1
        ok54 &= (a_h == a_e)
    check("T54 gpe_het(b0=16,κ=1) ≡ gpe_decision actions", ok54,
          f"{n54} random states; memo_h_q={len(memo_h.q)} "
          f"struct={len(memo_h.struct)}")

    # T55 (010 §九): heterogeneous budget bookkeeping — every episode satisfies
    # B = Σ_i(b0_i·N_tx,i + κ_i·pay_i) ≤ H pathwise (deterministic identity,
    # same as T18 but with per-UAV airtime costs).
    print("\nT55 C4 het budget identity B=Σ(b0_i N_tx,i + κ_i pay_i) ≤ H")
    H55, L55 = c4.sample_set(120, 5402, mm46)
    ok55 = True
    max_b55 = 0.0
    viol55 = 0
    for reg in c4.REGIMES:
        # 4-UAV sensing strengths（与 T46 的 mm46 一致；link_params 默认 8-UAV）
        b0r, kr = c4.link_params(reg, [-1.0, 1.0, 3.0, 5.0])
        memo_r = c4.GPEMemo()
        for e in range(len(H55)):
            om, b, nt, pay, nt_i, pay_i = c4.sim_decide_het(
                pl46, 256, 0.8, 96, L55[e],
                (lambda pl, x, om, h, rho, eta, b0=b0r, k=kr, m=memo_r:
                 c4.gpe_het_decision(pl, x, om, h, rho, eta, b0, k, m)),
                qu46, pw46, b0r, kr)
            chk = float(np.sum(b0r * nt_i + kr * pay_i))
            ok55 &= (abs(chk - b) < 1e-6 and b <= 96.0 + 1e-9)
            viol55 += int(not (abs(chk - b) < 1e-6 and b <= 96.0 + 1e-9))
            max_b55 = max(max_b55, b)
    check("T55 C4 het budget identity ≤ H (3 regimes)", ok55,
          f"viol={viol55} maxB={max_b55:.1f}")

    # T56 (P0 regression, user-audit): the Q_cond memo key must include
    # (rho, eta, b0, kappa) — otherwise a shared memo across the 28-combo
    # calibration loop leaks the FIRST (rho,eta)'s stale Q_cond into every
    # later combo (reproduced: Q(128,0.8)=81.0 leaked to Q(1024,2.0), fresh
    # value = 529.0), systematically understating probe Q and making GPE-EA
    # over-continue (H=96 E[N_tx] 2.50 vs 1.66 direction).  T50/T54 use a
    # fixed (256,0.8), so they cannot catch this; T56 pins the cross-(rho,eta)
    # distinctness AND the fresh-value equality with the identical memo
    # (structure layer must stay shared — om(x) is rho/eta-free, so the
    # (w,om) propagation is cached once and reused, only the r_rho/stopping
    # sum is recomputed per combo).
    print("\nT56 P0: _cond_refine_q memo key includes (rho,eta,b0,kappa)")
    x56, om56, i56, s56 = 0, 0.0, 0, 1
    conts56 = [t for t in g2.LEVELS
               if t > s56 and 16.0 + (t - s56) <= 96.0 - 16.0 + 1e-9]
    memo56 = c3e.GPEMemo()
    q56a = c3e._cond_refine_q(pl46, x56, om56, i56, s56, conts56,
                              128.0, 0.8, memo=memo56)[0]
    q56b = c3e._cond_refine_q(pl46, x56, om56, i56, s56, conts56,
                              1024.0, 2.0, memo=memo56)[0]
    q56c = c3e._cond_refine_q(pl46, x56, om56, i56, s56, conts56,
                              1024.0, 2.0, memo=memo56)[0]
    q56d = c3e._cond_refine_q(pl46, x56, om56, i56, s56, conts56,
                              1024.0, 2.0, memo=c3e.GPEMemo())[0]
    # distinct across (rho,eta) with the SAME shared memo (P0 fix)
    ok56 = (abs(q56a - q56b) > 1e-6
            # same combo: memo returns identical value (hit)
            and abs(q56b - q56c) < 1e-9
            # shared-memo value == fresh-memo value (no stale leak)
            and abs(q56b - q56d) < 1e-9
            # structure layer shared across combos (om(x) rho-free)
            and len(memo56.struct) >= 1
            and len(memo56.q) >= 2)
    check("T56 _cond_refine_q cross-(rho,eta) memo correctness",
          ok56,
          f"Q(128,0.8)={q56a:.1f} Q(1024,2.0)shared={q56b:.1f} "
          f"fresh={q56d:.1f} |D|={abs(q56a-q56b):.1f} "
          f"struct={len(memo56.struct)} q={len(memo56.q)}")

    # ================================= C5 protocol robustness (001 §二十六.1)
    # T57: ARQ-collapsed expectation identity — the extended cost
    # b0'=(b0+b_ctrl)/p_succ, kappa'=kappa/p_succ keeps c̄(Δr)=(b0+b_ctrl+
    # kappa Δr)/p_succ affine (010 §七 envelope holds exactly); and
    # collapsed E[B] ≈ explicit E[B] (geometric retries E[retries]=1/p_succ,
    # SystemModel §41 B1 vs B2; budget truncation/stop timing makes the small
    # residual difference).  Deterministic parts: affine identity + no budget
    # violations in either sim mode.
    print("\nT57 C5 ARQ collapsed affine + B1/B2 equivalence (001 §26.1, SM §41)")
    import run_mvsc05 as c5
    # B1/B2 验证用强 sensing 4-UA（anti 下坏链路但 sensing 有价值 → root 有
    # 行动；弱 sensing 组 root 即 STOP 正确反映 refinement 价值为负，但会
    # 使 B1/B2 记账为空，无法验证等价性）
    mm57 = c21.GaussianDetectorModel(g2.GAMMA_B[-4:], (0.5, 0.5))
    qu57 = [c21.NestedQuantizer(i, mm57, r_max=8, levels=g2.LEVELS)
            for i in range(4)]
    pwc57 = [c21.BASE_B ** i for i in range(4)]
    pl57 = c21.SparsePlanner(qu57, 1.0, 1.0, b_h=16.0, cross_level=True,
                             levels=g2.LEVELS, delta_c=1.0)
    b057, k57 = c4.link_params("anti", g2.GAMMA_B[-4:])
    # affine identity: b0e + ke*Δr == (b0 + b_ctrl + k*Δr)/p_succ
    ok57aff = True
    for psu in (1.0, 0.95, 0.9, 0.8):
        for bct in (0.0, 4.0, 8.0):
            b0e, ke = c5.extended_params(b057, k57, psu, bct)
            for dr in (1.0, 4.0, 8.0):
                lhs = b0e[0] + ke[0] * dr
                rhs = (b057[0] + bct + k57[0] * dr) / psu
                ok57aff &= abs(lhs - rhs) < 1e-9
    H57, L57 = c5.sample_set(120, 5701, mm57)
    ok57b = True
    for psu in (0.95, 0.9, 0.8):
        r = c5.arq_equivalence_check(
            pl57, list(b057), list(k57), psu, 0.0, 96, L57, qu57, pwc57,
            5705, (256.0, 0.8),
            (lambda pl, x, om, h, rho, eta, p=psu:
             c4.myopic_all_het(pl, x, om, h, rho, eta,
                               c5.extended_params(b057, k57, p, 0.0)[0],
                               c5.extended_params(b057, k57, p, 0.0)[1])))
        ok57b &= r["viol"] == 0
        # expectation identity: E_explicit ≈ E_collapsed (geometric retries)
        ok57b &= abs(r["E_collapsed"] - r["E_explicit"]) < 6.0
    check("T57 ARQ collapsed affine + B1/B2 E[B] equivalence",
          ok57aff and ok57b,
          f"affine={ok57aff} viol/equiv={ok57b}")

    # T58: correlation sampler — empirical corr(L_a,L_b|H) tracks rho
    # (common factor), and rho=0 reduces to the independent sampler.
    print("\nT58 C5 evidence-correlation sampler (common factor)")
    mm58 = c21.GaussianDetectorModel(g2.GAMMA_B, (0.5, 0.5))
    rho58 = 0.6
    H58, L58 = c5.sample_set_corr(40000, 5801, mm58, rho58)
    iy = H58 == 1
    cL = np.corrcoef(L58[iy].T)
    emp = float(np.mean([cL[0, j] for j in range(1, 8)]))
    # (true pairwise LL correlation of the common factor is rho, since
    # corr(L_i,L_j)=rho after scaling by the per-UAV gain)
    H0, L0 = c5.sample_set_corr(20000, 5802, mm58, 0.0)
    iy0 = H0 == 1
    cL0 = np.corrcoef(L0[iy0].T)
    emp0 = float(np.mean([cL0[0, j] for j in range(1, 8)]))
    ok58 = (abs(emp - rho58) < 0.05 and abs(emp0) < 0.05)
    check("T58 corr sampler: empirical rho tracking", ok58,
          f"rho=0.6 emp={emp:.3f} rho=0 emp0={emp0:.3f}")

    # T59: calibration mismatch semantics — planner(model) message-LLR ≠
    # true-LLR when Δγ≠0 (the model quantizer sees a shifted mixture), and
    # Δγ=0 must recover the true quantizer (identity).
    print("\nT59 C5 calibration-mismatch semantics")
    dg59 = 3.0
    mm_mod = c21.GaussianDetectorModel(np.asarray(g2.GAMMA_B) + dg59)
    qu_mod = [c21.NestedQuantizer(i, mm_mod, r_max=8, levels=g2.LEVELS)
              for i in range(8)]
    lrr_mod0 = qu_mod[0].llr[1]            # model 1-bit message LLRs (Δγ=3)
    mm_tr = c21.GaussianDetectorModel(g2.GAMMA_B)
    qu_tr = [c21.NestedQuantizer(i, mm_tr, r_max=8, levels=g2.LEVELS)
             for i in range(8)]
    lrr_tr0 = qu_tr[0].llr[1]              # true 1-bit message LLRs
    # Δγ≠0: model quantizer sees a shifted mixture ⇒ message LLRs differ;
    # the 1-bit cell structure (2 cells) is invariant.
    ok59 = (not np.allclose(lrr_mod0, lrr_tr0, atol=1e-4)
            and len(qu_tr[0].llr[1]) == 2
            and len(qu_mod[0].llr[1]) == 2)
    check("T59 mismatch: model LLR ≠ true LLR (Δγ≠0)", ok59,
          f"model max|Δ-LLR| vs true = {np.max(np.abs(lrr_mod0-lrr_tr0)):.3f}")

    # T60: decision-side params (p_succ, b_ctrl) enter the memo Q-key —
    # same (x,i,s,conts) but different (b0',kappa') give different Q_cond
    # with the SAME memo (extends T56 to the C5 extended params).
    print("\nT60 C5 decision params (p_succ/b_ctrl) in memo Q-key")
    x60, om60, i60, s60 = 0, 0.0, 0, 1
    conts60 = [t for t in g2.LEVELS
               if t > s60 and 16.0 + (t - s60) <= 96.0 - 16.0 + 1e-9]
    memo60 = c3e.GPEMemo()
    b0e0, ke0 = c5.extended_params(b057, k57, 1.0, 0.0)
    b0e1, ke1 = c5.extended_params(b057, k57, 0.8, 8.0)
    q60a = c3e._cond_refine_q(pl46, x60, om60, i60, s60, conts60, 256.0, 0.8,
                              memo=memo60, b0=b0e0[i60], kappa=ke0[i60])[0]
    q60b = c3e._cond_refine_q(pl46, x60, om60, i60, s60, conts60, 256.0, 0.8,
                              memo=memo60, b0=b0e1[i60], kappa=ke1[i60])[0]
    q60c = c3e._cond_refine_q(pl46, x60, om60, i60, s60, conts60, 256.0, 0.8,
                              memo=memo60, b0=b0e1[i60], kappa=ke1[i60])[0]
    ok60 = (abs(q60a - q60b) > 1e-6          # different (b0',kappa') -> diff Q
            and abs(q60b - q60c) < 1e-9      # same params -> memo hit
            and len(memo60.q) >= 2)
    check("T60 C5 (p_succ,b_ctrl) → (b0',κ') in memo key",
          ok60,
          f"Q(base)={q60a:.1f} Q(p=0.8,b8)={q60b:.1f} memo_q={len(memo60.q)}")

    # ================================================= 011 §十一 四类新回归
    # T61: global Depth-2 精确恒等式 Q_2 = Q_1 − E[Δ_all]（011 §三）——
    #      对随机可达状态×第一动作 (i,s)：`_cond_refine_q_global`(return_stats)
    #      的 Q2_dev<1e-9；且 E[Δ_all]≥E[Δ_self]（011 §四）、G_switch≥0、
    #      P_jstar_ne_i∈[0,1]。
    print("\nT61 global D2 identity Q2=Q1−E[Δ_all] (011 §三)")
    rng61 = np.random.default_rng(61001)
    ok61 = True
    n61 = 0
    max_dev61 = 0.0
    for _ in range(30):
        x61 = int(rng61.integers(0, 3_000_000))
        om61 = float(rng61.uniform(-8, 8))
        cand61 = None
        for i in range(4):
            zi = (x61 // pl46.powers[i]) % c21.BASE_B
            r, _ = c21.z_decode_b(zi)
            for s in pl46.levels:
                if s <= r:
                    continue
                if 16.0 + (s - r) > 96.0 + 1e-9:
                    continue
                cand61 = (i, s)
                break
            if cand61:
                break
        if cand61 is None:
            continue
        i61, s61 = cand61
        _q61, st61 = c3e._cond_refine_q_global(
            pl46, x61, om61, i61, s61, 96.0, 256.0, 0.8,
            memo=c3e.GPEMemo(), return_stats=True)
        n61 += 1
        max_dev61 = max(max_dev61, st61["Q2_dev"])
        ok61 &= (st61["Q2_dev"] < 1e-9
                 and st61["E_Δ_all"] + 1e-9 >= st61["E_Δ_self"]
                 and st61["G_switch"] >= -1e-9
                 and 0.0 <= st61["P_jstar_ne_i"] <= 1.0)
    check("T61 global D2 identity Q2=Q1−E[Δ_all] (+G_switch≥0)",
          n61 > 0 and ok61, f"n={n61} max_dev={max_dev61:.1e}")

    # T62: 011 §十 gate 统计自洽——global_depth2_mechanism_stats 的机制量
    #      形成一致不变式（如实报告，不硬编码激活）：G_switch≥0、
    #      P_Δ_all_gt0∈[0,1]、P_jstar_ne_i∈[0,1]、action_change_rate∈[0,1]。
    print("\nT62 global D2 gate stats self-consistency (011 §十)")
    ag62 = c3e.global_depth2_mechanism_stats(pl46, qu46, pw46,
                                             H46[:40], L46[:40], 96.0,
                                             theta=(256.0, 0.8))
    ok62 = (ag62["G_switch"] >= -1e-9
            and 0.0 <= ag62["P_Δ_all_gt0"] <= 1.0
            and 0.0 <= ag62["P_jstar_ne_i"] <= 1.0
            and 0.0 <= ag62["action_change_rate"] <= 1.0)
    check("T62 gate stats self-consistent (G_switch≥0, P∈[0,1])",
          ok62, f"G={ag62['G_switch']:.2e} Pj*={ag62['P_jstar_ne_i']:.2f}")

    # T63: 011 §六 exact-prune B&B 不减 min、不改策略——相同状态下对比
    #      full-enumeration min Q_1 与 "exact-prune(c_b≥R1 跳过) + B&B(c_b≥
    #      V_best 后跳过)" 的 min Q_1：两者相等（差<1e-9）。
    print("\nT63 exact-prune B&B does not change min Q_1 (011 §六)")
    rng63 = np.random.default_rng(63001)
    ok63 = True
    n63 = 0
    maxd63 = 0.0
    for _ in range(25):
        x63 = int(rng63.integers(0, 3_000_000))
        om63 = float(rng63.uniform(-8, 8))
        for i in range(4):
            zi = (x63 // pl46.powers[i]) % c21.BASE_B
            r, _ = c21.z_decode_b(zi)
            for s in pl46.levels:
                if s <= r:
                    continue
                if 16.0 + (s - r) > 96.0 + 1e-9:
                    continue
                h1 = 96.0 - (16.0 + (s - r))
                min_full = float("inf")
                acts = []
                for j in range(4):
                    zj = (x63 // pl46.powers[j]) % c21.BASE_B
                    rj, _ = c21.z_decode_b(zj)
                    for u in pl46.levels:
                        if u <= rj:
                            continue
                        cb = 16.0 + (u - rj)
                        if cb > h1 + 1e-9:
                            continue
                        q1 = c3e.q1_fast(pl46, x63, om63, j, u, 256.0, 0.8)
                        min_full = min(min_full, q1)
                        acts.append((cb, j, u, q1))
                if min_full == float("inf"):
                    continue
                acts.sort(key=lambda t: t[0])
                min_bb = float("inf")
                for (cb, j, u, _q) in acts:
                    if cb >= min_bb - 1e-12:    # B&B：c_b≥V_best 后跳过
                        break
                    min_bb = min(min_bb, _q)
                n63 += 1
                d = abs(min_full - min_bb)
                maxd63 = max(maxd63, d)
                ok63 &= (d < 1e-9)
                break
            if n63:
                break
    check("T63 B&B vs full-enum min Q_1 identical (<1e-9)",
          n63 > 0 and ok63, f"n={n63} max_d={maxd63:.1e}")

    # T64: 011 §七 same-message mismatch——冻结 model partition（bounds_override）
    #      后重算 true PMF/LLR：同 cells（逐 level bounds 相同）但 Δγ≠0 ⇒
    #      LLR 不同（同一 message index 的 true-PMF/LLR），PMF 仍归一。
    print("\nT64 mismatch frozen-partition true PMF/LLR (011 §七)")
    import run_mvsc05 as c5
    mm_t64 = c21.GaussianDetectorModel(g2.GAMMA_B[:4], (0.5, 0.5))
    qu_m64 = [c21.NestedQuantizer(i, mm_t64, r_max=8, levels=g2.LEVELS)
              for i in range(4)]
    mm_tru64 = c21.GaussianDetectorModel(np.asarray(g2.GAMMA_B[:4]) + 3.0)
    qu_f64 = [c21.NestedQuantizer(i, mm_tru64, r_max=8, levels=g2.LEVELS,
                                  bounds_override=qu_m64[i].bounds)
              for i in range(4)]
    ok64 = True
    for i in range(4):
        for lv in g2.LEVELS:
            ok64 &= np.allclose(qu_f64[i].bounds[lv], qu_m64[i].bounds[lv])
        ok64 &= qu_f64[i].check_pmf_normalization()
        # Δγ≠0 ⇒ 同 cells（同 message index）上 true LLR ≠ model LLR
        ok64 &= not np.allclose(qu_f64[i].llr[1], qu_m64[i].llr[1], atol=1e-6)
    check("T64 frozen-partition true LLR≠model LLR (same cells, PMF ok)",
          ok64, f"cells=2 per level; PMF-norm={ok64}")

    print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ({time.time()-t0:.0f}s) ===")
    for name, d in FAIL:
        print(f"  FAILED: {name} {d}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(run())
