"""Quick smoke test for the MVS-A core pipeline."""
import time

import numpy as np

from opmvs import ExactDP, GaussianDetectorModel, NestedQuantizer, OPEF1, OPEF2, StateSpace
from opmvs import baselines as bl
from opmvs import mc as mclib

model = GaussianDetectorModel([-1.0, 1.0, 3.0, 5.0])
quants = [NestedQuantizer(i, model) for i in range(4)]
for i, q in enumerate(quants):
    print(f"UAV{i}: pmf_norm={q.check_pmf_normalization()}, nested={q.check_nested_consistency()}")

ss = StateSpace(model, quants)
dp = ExactDP(ss)
V, pol = dp.solve(16.0, 16.0)
mx, mn = dp.bellman_residual()
print(f"DP Bellman residual: max={mx:.3e} mean={mn:.3e}")
o1 = OPEF1(ss)
V1, pol1 = o1.solve(16.0, 16.0)
o2 = OPEF2(ss)
V2, pol2 = o2.solve(16.0, 16.0, V1)

n = 50000
H, L = mclib.sample_episodes(model, n, 2026)
for name, p in (("DP", pol), ("OPEF1", pol1), ("OPEF2", pol2)):
    lam, cost, z, ns = mclib.simulate_table_policy(ss, p, H, L)
    m = mclib.evaluate(lam, cost, H, 0.05, ns)
    print(f"{name}: E[B]={cost.mean():.2f} E[Nq]={ns.mean():.2f} P_D={m['pd']:.4f} P_FA={m['pfa']:.4f}")

for name, lam, cost in [
    ("B1", *bl.baseline_all_neighbor(ss, H, L)),
    ("POTS", *bl.baseline_pots(ss, H, L, 2.0)),
    ("OTSF", *bl.baseline_ots_f(ss, H, L, 2.0)),
    ("Cens", *bl.baseline_censoring(ss, H, L, 1.5)),
    ("StaticProg", *bl.baseline_static_progressive(ss, H, L, 2.0)),
    ("GlobFixed", *bl.baseline_global_fixed(ss, H, L, 2.0)),
    ("SNRTopK2", *bl.baseline_snr_topk(ss, H, L, 2)),
    ("Raw", *bl.baseline_raw(ss, H, L)),
]:
    m = mclib.evaluate(lam, cost, H, 0.05)
    print(f"{name}: E[B]={cost.mean():.2f} P_D={m['pd']:.4f} P_FA={m['pfa']:.4f}")

# random-K sanity
rng = np.random.default_rng(7)
lam, cost = bl.baseline_random_k(ss, H, L, 2, rng)
m = mclib.evaluate(lam, cost, H, 0.05)
print(f"B2_RandomK2: E[B]={cost.mean():.2f} P_D={m['pd']:.4f} P_FA={m['pfa']:.4f}")

# precision audit
lam, cost, z, ns = mclib.simulate_table_policy(ss, pol2, H, L)
for row in mclib.precision_audit(z, ss):
    print("audit", row)
