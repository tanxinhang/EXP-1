"""Nested multi-resolution evidence quantizer (SystemModel §7, §14).

Design density:  p_mix(L) = 0.5 p(L|H0) + 0.5 p(L|H1)   (per UAV).

A binary nested partition tree of depth r_max is built by recursive
*conditional median* splitting of the mixture: every level-r cell carries
mixture probability 2^-r, and cells of level r+1 partition cells of level r,
so the message families are nested:

    Q_i^(1) ≺ Q_i^(2) ≺ Q_i^(4)   (≺ = refinement / nested partition)

Outer cells are allowed to be (-inf, tau1] and [tau_K, +inf) — no artificial
finite truncation / tail gating of the Gaussian LLR (Section 14).

Message PMF:  theta_{i,h}^{(r)}(m) = P(M_i^(r)=m | H_h) computed from the
Gaussian CDFs of L_i under each hypothesis (log domain).
Message-LLR:  ell_i^{(r)}(m) = log P(M=m|H1) - log P(M=m|H0).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

# levels at which messages exist (MVS-A); level 0 = "no message" (empty)
MESSAGE_LEVELS = (1, 2, 4)
RMAX = 4

# finite stand-ins for +/-inf used only as brentq brackets (mass loss ~0)
_BRACKET = 60.0


class NestedQuantizer:
    def __init__(self, i, model, r_max=RMAX, levels=MESSAGE_LEVELS,
                 bounds_override=None):
        self.i = int(i)
        self.model = model
        self.r_max = int(r_max)
        self.levels = tuple(int(r) for r in levels)
        self.mu0, self.mu1, self.sd = model.llr_params(self.i)
        if bounds_override is not None:
            # 011 §七/§十一 P0.5：固定 deploy partition（B^deploy=B^model 的
            # 边界集），本 quantizer 只在该固定 cells 上重算 PMF/LLR ——
            # C5 mismatch fidelity 的"same-cells, true-PMF" 语义。
            self.bounds = dict(bounds_override)
        else:
            self.bounds = self._build_tree()      # level -> (2^r+1,) boundaries
        self.logP1 = {}                           # level -> (2^r,) log P(..|H1)
        self.logP0 = {}                           # level -> (2^r,) log P(..|H0)
        self.llr = {}                             # level -> (2^r,) message LLR
        self._precompute_pmf()

    # ------------------------------------------------------------ tree build
    def _f_mix(self, x):
        return self.model.mixture_cdf(x, self.i)

    def _truncated_median(self, lo, hi):
        """Median of the mixture restricted to cell (lo, hi)."""
        Flo = 0.0 if not np.isfinite(lo) else float(self._f_mix(lo))
        Fhi = 1.0 if not np.isfinite(hi) else float(self._f_mix(hi))
        target = 0.5 * (Flo + Fhi)
        lo_b = lo if np.isfinite(lo) else -_BRACKET
        hi_b = hi if np.isfinite(hi) else +_BRACKET
        f = lambda x: float(self._f_mix(x)) - target
        fa, fb = f(lo_b), f(hi_b)
        if not (fa < 0.0 < fb):
            raise RuntimeError(
                f"UAV {self.i}: truncated median bracket failure on ({lo}, {hi})")
        return float(brentq(f, lo_b, hi_b, xtol=1e-14, rtol=1e-13))

    def _build_tree(self):
        bounds = {0: np.array([-np.inf, np.inf])}
        for r in range(1, self.r_max + 1):
            prev = bounds[r - 1]
            new = np.empty(2 ** r + 1)
            new[::2] = prev                       # shared boundaries (nested)
            for m in range(2 ** (r - 1)):
                lo, hi = prev[m], prev[m + 1]
                new[2 * m + 1] = self._truncated_median(lo, hi)
            bounds[r] = new
        return bounds

    # ------------------------------------------------------------- cell ops
    def cell_index(self, level, L):
        """Cell index at `level` for LLR value(s) L (numpy array or scalar)."""
        b = self.bounds[int(level)]
        idx = np.searchsorted(b, np.asarray(L, dtype=float), side="right") - 1
        return np.clip(idx, 0, 2 ** int(level) - 1)

    def desc_cells(self, r, m, r2):
        """Cell indices at level r2 contained in cell (r, m); r=0 -> root."""
        r2 = int(r2)
        if r == 0:
            return np.arange(2 ** r2)
        k = 2 ** (r2 - int(r))
        return np.arange(int(m) * k, (int(m) + 1) * k)

    def cell_logprob(self, r, m, h):
        """log P(M_i^(r)=m | H_h) via Gaussian CDF (log domain).

        NOTE: interior cells must use the *linear* difference
        Phi(hi) - Phi(lo) and then take the log — never logcdf(hi)-logcdf(lo),
        which equals log(Phi(hi)/Phi(lo)), not log(Phi(hi)-Phi(lo)).
        """
        r, m = int(r), int(m)
        b = self.bounds[r]
        lo, hi = b[m], b[m + 1]
        mu = self.mu1 if h else self.mu0
        dist = stats.norm(loc=mu, scale=self.sd)
        if not np.isfinite(lo) and not np.isfinite(hi):
            return 0.0
        if not np.isfinite(lo):
            return float(dist.logcdf(hi))
        if not np.isfinite(hi):
            return float(dist.logsf(lo))
        p = float(dist.cdf(hi) - dist.cdf(lo))
        if p <= 0.0:
            return -np.inf
        return float(np.log(p))

    # ---------------------------------------------------------- message PMFs
    def _precompute_pmf(self):
        for r in self.levels:
            ncell = 2 ** r
            lp1 = np.empty(ncell)
            lp0 = np.empty(ncell)
            for m in range(ncell):
                lp1[m] = self.cell_logprob(r, m, 1)
                lp0[m] = self.cell_logprob(r, m, 0)
            self.logP1[r] = lp1
            self.logP0[r] = lp0
            self.llr[r] = lp1 - lp0

    # ------------------------------------------------------------- messages
    def message(self, level, L):
        """Message index m at `level` for LLR value(s) L."""
        return self.cell_index(level, L)

    def message_llr(self, level, m):
        return self.llr[int(level)][np.asarray(m, dtype=np.int64)]

    # ---------------------------------------------------------- sanity aids
    def check_pmf_normalization(self, rtol=1e-10):
        ok = True
        for r in self.levels:
            for h in (0, 1):
                lp = self.logP1[r] if h else self.logP0[r]
                s = float(np.exp(lp).sum())
                ok &= abs(s - 1.0) <= rtol
        return ok

    def check_nested_consistency(self, rtol=1e-9):
        """P(m|H_h) == sum over descendants at level 4 (and level 2 from 1)."""
        ok = True
        for r in self.levels:
            if r >= self.r_max:
                continue
            for m in range(2 ** r):
                for r2 in self.levels:
                    if r2 <= r:
                        continue
                    for h in (0, 1):
                        lp = self.logP1[r] if h else self.logP0[r]
                        lp2 = self.logP1[r2] if h else self.logP0[r2]
                        children = self.desc_cells(r, m, r2)
                        s = float(np.exp(lp2[children]).sum())
                        ok &= abs(s - np.exp(lp[m])) <= rtol
        return ok

    def summarize(self):
        """Return a compact per-level description for reporting."""
        rows = []
        for r in self.levels:
            rows.append({
                "level": r,
                "cells": 2 ** r,
                "llr_min": float(self.llr[r].min()),
                "llr_max": float(self.llr[r].max()),
                "llr_absmax": float(np.abs(self.llr[r]).max()),
            })
        return rows
