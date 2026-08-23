"""MVS-B0.4b: Feedback-Granularity Phase-Transition Theorem (advice/013.md).

Pure-theory封板 for the state-dependent packetization phase boundary.  No
planner changes.  The exact identity (013 §1, verified to <1e-10):

        g_x(b) = Q_prog(x; b) - Q_dir(x; b) = E[ min{ Y_x, b } ],

    Y_x      = D_x(X_1) - Delta_2,
    D_x(X_1) = R(X_1) - E[ R(X_2) | X_1 ],        (second-stage info gain)
    Delta_2  = r_max - r_next,                    (full minus next packet)

where progressive = r -> r_next -> r_max (two transactions) and direct =
r -> r_max (one full packet), each transaction paying the fixed setup/header
cost b >= 0.

Because the X_1 support of the finite quantizer DAG is discrete, g_x is
EXACTLY piecewise linear with breakpoints at the support values of Y_x and
(013 §2):

        g'_{x,+}(b) = Pr(Y_x > b),    g'_{x,-}(b) = Pr(Y_x >= b),

i.e. the marginal effect of the setup cost on the packetization preference
equals the probability that the second feedback transaction is triggered.
The crossing b*(x) = inf{ b >= 0 : g_x(b) >= 0 } has a closed form on the
crossing interval (013 §5):

        b*(x) = b_0 - g_x(b_0) / Pr(Y_x > b_0),   b_0 = left breakpoint.

Three-case classification (013 §3):
  A  E[Y_x] < 0   =>  g(b) <= E[Y] < 0 for ALL b  =>  b* = +inf
     (progressive strictly dominates direct for every b_h >= 0)
  B  E[Y_x] = 0   =>  b* = max{ 0, ess sup Y_x };  g(b) = 0 for b >= b*
     (progressive == direct beyond b* — NOT direct-dominates)
  C  E[Y_x] > 0   =>  unique finite crossing;  b* = 0 iff Y_x >= 0 a.s.,
     otherwise 0 < b* < ess sup Y_x.
Hence (013 §4)  b*(x) < inf  <=>  E[Y_x] >= 0  <=>  E[D_x] >= Delta_2:
the packetization phase boundary exists iff the expected Bayes-risk
reduction of the second stage covers the extra payload of the full packet.
"""
from __future__ import annotations

import numpy as np

from opmvs.fusion import log_sigmoid, log_one_minus_sigmoid
from opmvs.sparse import BASE_B, z_code_b, z_decode_b

_EPS = 1e-12


def _weight(lp, lq, lp1c, lp0c):
    """Pr(message) from the log-conditional pmf (numerically stable)."""
    a_ = lp + lp1c
    b_ = lq + lp0c
    m_ = a_ if a_ >= b_ else b_
    return float(np.exp(m_ + np.log1p(np.exp(-abs(a_ - b_)))))


def y_support(pl, x_int, i):
    """Exact discrete support of Y_x = D_x(X_1) - Delta_2 for the
    packetization choice on UAV i at its current level r, given the FULL
    state x_int (other UAVs' messages stay fixed — reachable-state
    generalization of 013 §1; the single-UAV empty-others case is
    x_int = encode((0, ..., z_i, ..., 0))).

    Returns a dict, or None when UAV i is at a leaf (no r_next / r_max):
      branches : [(w1, x1, R1, E_R, D, Y)] over the X_1 support of r->r_next
                 w1 = Pr(X_1 branch),  R1 = R(x1) = r_stop(x1),
                 E_R = E[R(X_2) | x1]  (tower target),
                 D = R1 - E_R,  Y = D - Delta_2
      Delta2   : r_max - r_next
      r, r_next, r_max
      EY, Ymax, Ymin
      E_dir    : E[R(X_2)] via the direct r->r_max step (tower property)
      E_R_sum  : sum_k w1_k * E_R_k   (== E_dir in exact arithmetic)
      tower_dev: |E_R_sum - E_dir|
    """
    zs = pl.decode(int(x_int))
    zi = zs[i]
    r, _m = z_decode_b(zi)
    levels = pl.levels
    r_next = next((r2 for r2 in levels if r2 > r), None)
    r_max = pl.r_max
    if r_next is None or r_max not in levels:
        return None
    om, _p, lp, lq = pl.posterior(int(x_int))
    pw = pl.powers[i]
    llr_i = pl._llr_i[i]
    tpl_i = pl._tpl[i][zi]
    dir_tpl = next((a for a in tpl_i if a[0] == r_max), None)
    prog_tpl = next((a for a in tpl_i if a[0] == r_next), None)
    if dir_tpl is None or prog_tpl is None:
        return None
    d2 = r_max - r_next
    # direct: r -> r_max  (expected stopping risk after the full packet)
    E_dir = 0.0
    for (m2, lp0c, lp1c) in dir_tpl[3]:
        w = _weight(lp, lq, lp1c, lp0c)
        cx = int(x_int) + (z_code_b(r_max, m2) - zi) * pw
        E_dir += w * pl.r_stop(cx)
    # progressive: r -> r_next, then min{ R(x1), b + d2 + E[R(x2)|x1] }
    branches = []
    for (m1, lp0c1, lp1c1) in prog_tpl[3]:
        w1 = _weight(lp, lq, lp1c1, lp0c1)
        z1 = z_code_b(r_next, m1)
        x1 = int(x_int) + (z1 - zi) * pw
        R1 = pl.r_stop(x1)
        if d2 == 0.0:
            # DEGENERATE progressive (r_next == r_max): the second
            # transaction r_next -> r_max is a no-op, so the continuation at
            # x1 is STOP and E_R = R1, giving D = 0, Y = 0 — progressive and
            # direct coincide (g_x = 0, b* = 0), and the tower property
            # holds exactly (sum w1 R1 = E_dir by nestedness).
            E_R = R1
            D = 0.0
            Y = 0.0
        else:
            om1 = om + llr_i[z1] - llr_i[zi]
            p1 = 1.0 / (1.0 + np.exp(-om1))
            lp1 = float(np.log(p1))
            lq1 = float(np.log1p(-p1))
            ref_tpl = next((a for a in pl._tpl[i][z1] if a[0] == r_max), None)
            E_R = 0.0
            if ref_tpl is not None:
                for (m2, lp0c, lp1c) in ref_tpl[3]:
                    w = _weight(lp1, lq1, lp1c, lp0c)
                    cx = x1 + (z_code_b(r_max, m2) - z1) * pw
                    E_R += w * pl.r_stop(cx)
            D = R1 - E_R
            Y = D - d2
        branches.append((w1, x1, R1, E_R, D, Y))
    wsum = sum(br[0] for br in branches)
    E_R_sum = sum(br[0] * br[3] for br in branches)
    EY = sum(br[0] * br[5] for br in branches) / wsum
    Ymax = max(br[5] for br in branches)
    Ymin = min(br[5] for br in branches)
    return {"branches": branches, "Delta2": d2, "r": r, "r_next": r_next,
            "r_max": r_max, "EY": EY, "Ymax": Ymax, "Ymin": Ymin,
            "E_dir": E_dir, "E_R_sum": E_R_sum,
            "tower_dev": abs(E_R_sum - E_dir), "wsum": wsum}


# --------------------------------------------------------- exact g from support
def g_from_support(sup, b):
    """g_x(b) = E[min{ Y_x, b }] from the exact support (piecewise linear)."""
    b = float(b)
    return sum(br[0] * min(br[5], b) for br in sup["branches"]) / sup["wsum"]


def survival(sup, b, strict=True):
    """Pr(Y_x > b) (strict=True) or Pr(Y_x >= b) (strict=False)."""
    b = float(b)
    if strict:
        return sum(br[0] * (1.0 if br[5] > b else 0.0)
                   for br in sup["branches"]) / sup["wsum"]
    return sum(br[0] * (1.0 if br[5] >= b else 0.0)
               for br in sup["branches"]) / sup["wsum"]


def g_alt(sup, b):
    """Strategy-value form Q_prog(x;b) - Q_dir(x;b) from the SAME support
    (013 §1).  Algebra: with min{R1, b+d2+E_R} - (d2+E_R) = min{D-d2, b},
    Q_prog - Q_dir = E[min{Y,b}] + (E_R_sum - E_dir)  =>  identity exact
    up to float round-off (the tower property kills the E_R terms)."""
    b = float(b)
    d2 = sup["Delta2"]
    Q_prog = (b + (sup["r_next"] - sup["r"])) + sum(
        br[0] * min(br[2], b + d2 + br[3]) for br in sup["branches"]) / sup["wsum"]
    Q_dir = (b + (sup["r_max"] - sup["r"])) + sup["E_dir"]
    return Q_prog - Q_dir


# -------------------------------------------------------- exact b* (013 §3/§5)
def bstar_from_dist(w, y):
    """Three-case exact b*(x) = inf{ b >= 0 : E[min(Y,b)] >= 0 } from a plain
    discrete distribution (w_k, y_k) — used by T30's synthetic branches and by
    bstar_exact for planner states.

    Returns {"case": "A"|"B"|"C", "bstar": float|inf, "EY", "Ymax", "Ymin",
             "g0": g(0), "g_inf": E[Y]}.
    """
    w = np.asarray(w, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = w / w.sum()
    EY = float(np.dot(w, y))
    Ymax = float(y.max())
    Ymin = float(y.min())

    def g(b):
        return float(np.dot(w, np.minimum(y, b)))

    def surv(b):
        return float(np.dot(w, y > b))

    if EY < -_EPS:                                   # Case A (013 §3)
        return {"case": "A", "bstar": float("inf"), "EY": EY,
                "Ymax": Ymax, "Ymin": Ymin, "g0": g(0.0), "g_inf": EY}
    if g(0.0) >= -_EPS:                              # Y >= 0 a.s. => b* = 0
        case = "B" if abs(EY) <= _EPS else "C"
        return {"case": case, "bstar": 0.0, "EY": EY,
                "Ymax": Ymax, "Ymin": Ymin, "g0": g(0.0), "g_inf": EY}
    # g(0) < 0 <= E[Y]: unique crossing on (0, Ymax]; walk the exact support
    # breakpoints (positive support values), slope = Pr(Y > b_left) constant
    # on each open interval (013 §5: b* = b_left - g(b_left)/Pr(Y > b_left)).
    bps = sorted({float(v) for v in y if v > 0.0})
    b_left, gL = 0.0, g(0.0)
    for b_right in bps:
        gR = g(b_right)
        if gR >= -_EPS:
            s = surv(b_left)
            assert s > 0.0, "zero slope on a crossing interval"
            bstar = b_left - gL / s
            bstar = float(min(max(bstar, b_left), b_right))
            case = "B" if abs(EY) <= _EPS else "C"
            return {"case": case, "bstar": bstar, "EY": EY,
                    "Ymax": Ymax, "Ymin": Ymin, "g0": g(0.0), "g_inf": EY}
        b_left, gL = b_right, gR
    raise AssertionError("no crossing found although E[Y] >= 0")


def bstar_exact(sup):
    """bstar_exact(sup) — exact b*(x) for a y_support dict."""
    w = [br[0] for br in sup["branches"]]
    y = [br[5] for br in sup["branches"]]
    return bstar_from_dist(w, y)


def verify_identity(sup, b_values):
    """G0: max |g(b) - (Q_prog - Q_dir)| and tower deviation over b_values."""
    max_dev = 0.0
    for b in b_values:
        max_dev = max(max_dev, abs(g_from_support(sup, b) - g_alt(sup, b)))
    return max_dev, sup["tower_dev"]
