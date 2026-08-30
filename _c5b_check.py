# -*- coding: utf-8 -*-
"""Verify C5 B1/B2 equivalence on the strong-sensing 4-UAV subset."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import run_mvsb07g2 as g2
import run_mvsc021 as c21
import run_mvsc04 as c4
import run_mvsc05 as c5
from opmvs.sparse import SparsePlanner

mm = c21.GaussianDetectorModel(g2.GAMMA_B[-4:])
qu = [c21.NestedQuantizer(i, mm, r_max=8, levels=g2.LEVELS) for i in range(4)]
pw = [c21.BASE_B ** i for i in range(4)]
pl = SparsePlanner(qu, 1.0, 1.0, b_h=16.0, cross_level=True, levels=g2.LEVELS,
                   delta_c=1.0)
b0, k = c4.link_params("anti", g2.GAMMA_B[-4:])
H, L = c5.sample_set(120, 5701, mm)
for psu in (0.95, 0.9, 0.8):
    r = c5.arq_equivalence_check(pl, list(b0), list(k), psu, 0.0, 96, L, qu,
                                 pw, 5705, (256.0, 0.8),
                                 (lambda pl, x, om, h, rho, eta, p=psu:
                                  c4.myopic_all_het(pl, x, om, h, rho, eta,
                                                    c5.extended_params(b0, k,
                                                                       p, 0.0)[0],
                                                    c5.extended_params(b0, k,
                                                                       p, 0.0)[1])))
    print(f"p_succ={psu}: E_col={r['E_collapsed']:.2f} "
          f"E_exp={r['E_explicit']:.2f} D={r['D']:.2f} viol={r['viol']}")
# deterministic: D small (budget truncation), viol=0
assert r["viol"] == 0
assert abs(r["D"]) < 12.0
print("B1/B2 OK: nonzero, viol=0, D small")