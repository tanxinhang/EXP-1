# -*- coding: utf-8 -*-
"""C5 ARQ-equivalence debug: why E[B]=0 in B-table."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import run_mvsb07g2 as g2
import run_mvsc021 as c21
import run_mvsc04 as c4
import run_mvsc05 as c5
from opmvs.sparse import SparsePlanner

mm4 = c21.GaussianDetectorModel(g2.GAMMA_B[-4:])
qu4 = [c21.NestedQuantizer(i, mm4, r_max=8, levels=g2.LEVELS) for i in range(4)]
pw4 = [c21.BASE_B ** i for i in range(4)]
pl4 = SparsePlanner(qu4, 1.0, 1.0, b_h=16.0, cross_level=True,
                    levels=g2.LEVELS, delta_c=1.0)
b04, k4 = c4.link_params("anti", g2.GAMMA_B[-4:])
b0e, ke = c5.extended_params(b04, k4, 0.95, 0.0)
print("b04", list(np.round(b04, 2)), "k4", list(np.round(k4, 2)))
print("b0e", list(np.round(b0e, 2)), "ke", list(np.round(ke, 2)))

for th in ((256.0, 0.8), (256.0, 1.0), (128.0, 1.2), (512.0, 1.2)):
    dec, diag = c4.myopic_all_het(pl4, 0, 0.0, 96, th[0], th[1], b0e, ke)
    print(f"root @ {th}:", dec, diag)

L4_test = mm4.sample_llr(np.zeros(20, dtype=np.int8),
                         np.random.default_rng(1))
for e in range(3):
    r = c5.sim_dep_collapsed(
        pl4, 256.0, 0.8, 96, L4_test[e],
        (lambda pl, x, om, h, rho, eta: c4.myopic_all_het(
            pl, x, om, h, rho, eta, b0e, ke)),
        b0e, ke, 0.95, 0.0, qu4, pw4)
    print("collapsed ep", e, "cost=", r[1], "nt=", r[2])

# check the equivalent sim used by the runner (raw b0/k4 inside the sim)
r2 = c5.sim_dep_collapsed(
    pl4, 256.0, 0.8, 96, L4_test[0],
    (lambda pl, x, om, h, rho, eta: c4.myopic_all_het(
        pl, x, om, h, rho, eta, b0e, ke)),
    list(b04), list(k4), 0.95, 0.0, qu4, pw4)
print("collapsed(runner-style raw params) cost=", r2[1], "nt=", r2[2])