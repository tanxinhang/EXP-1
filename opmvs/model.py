"""MVS-A Gaussian local detector model (SystemModel §30-§32).

H0: X_i ~ N(0, 1)
H1: X_i ~ N(a_i, 1)
local LLR:  L_i = a_i X_i - a_i^2/2
so:         L_i | H0 ~ N(-a_i^2/2, a_i^2),  L_i | H1 ~ N(+a_i^2/2, a_i^2)
with        a_i^2 = 10^(gamma_i^s / 10)      (sensing strength in dB)

Default MVS-A sensing strength: gamma^s = [-1, 1, 3, 5] dB.
Single-UAV P_D at P_FA=0.05 ~ [0.226, 0.301, 0.408, 0.553];
continuous full fusion P_D,raw ~ 0.851  (weak evidence -> strong cooperation).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class GaussianDetectorModel:
    def __init__(self, gamma_db, prior=(0.5, 0.5)):
        self.gamma_db = np.asarray(gamma_db, dtype=float)
        self.N = int(self.gamma_db.size)
        self.a2 = 10.0 ** (self.gamma_db / 10.0)
        self.a = np.sqrt(self.a2)
        self.prior = prior
        self.prior_log_odds = float(np.log(prior[1] / prior[0]))

    # ------------------------------------------------------------------ LLR
    def llr_params(self, i):
        """(mu0, mu1, sigma) of L_i under H0/H1."""
        a2 = float(self.a2[i])
        return -0.5 * a2, +0.5 * a2, float(np.sqrt(a2))

    def llr_mean_std(self, i, h):
        mu0, mu1, sd = self.llr_params(i)
        return (mu1 if h else mu0), sd

    def llr_pdf(self, x, i, h=1):
        mu, sd = self.llr_mean_std(i, h)
        return stats.norm.pdf(x, loc=mu, scale=sd)

    def llr_cdf(self, x, i, h=1):
        mu, sd = self.llr_mean_std(i, h)
        return stats.norm.cdf(x, loc=mu, scale=sd)

    # --------------------------------------------------------------- sampling
    def sample_hypotheses(self, n, rng):
        """Sample hypotheses H ~ Bernoulli(pi_1) respecting the configured
        prior (fix B0.1 of adcice/005.md: the prior was stored but unused)."""
        return (rng.random(n) < self.prior[1]).astype(np.int8)

    def sample_llr(self, H, rng):
        """Sample local LLRs: shape (n, N). H: (n,) hypotheses {0,1}."""
        H = np.asarray(H)
        n = H.shape[0]
        L = np.empty((n, self.N), dtype=float)
        for i in range(self.N):
            mu0, mu1, sd = self.llr_params(i)
            L[:, i] = rng.normal(np.where(H == 1, mu1, mu0), sd)
        return L

    def sample_episodes(self, n, rng):
        """Sample (H, L) jointly with a shared RNG."""
        H = self.sample_hypotheses(n, rng)
        L = self.sample_llr(H, rng)
        return H, L

    # ------------------------------------------------------------- analytics
    def single_uav_pd(self, i, pfa):
        """Single-UAV P_D of UAV i at target P_FA (threshold on L_i)."""
        mu0, mu1, sd = self.llr_params(i)
        eta = stats.norm.ppf(1.0 - pfa, loc=mu0, scale=sd)
        return float(stats.norm.sf(eta, loc=mu1, scale=sd))

    def single_uav_pd_all(self, pfa):
        return np.array([self.single_uav_pd(i, pfa) for i in range(self.N)])

    def raw_fusion_pd(self, pfa):
        """Analytical P_D of continuous full fusion (sum of L_i) at P_FA."""
        S = float(self.a2.sum())
        sd = float(np.sqrt(S))
        eta = stats.norm.ppf(1.0 - pfa, loc=-0.5 * S, scale=sd)
        return float(stats.norm.sf(eta, loc=+0.5 * S, scale=sd))

    def raw_fusion_pfa_pd(self, eta):
        S = float(self.a2.sum())
        sd = float(np.sqrt(S))
        pfa = float(stats.norm.sf(eta, loc=-0.5 * S, scale=sd))
        pd = float(stats.norm.sf(eta, loc=+0.5 * S, scale=sd))
        return pfa, pd

    def mixture_pdf(self, x, i):
        """p_mix(L) = 0.5 p(L|H0) + 0.5 p(L|H1)  (quantizer design density)."""
        mu0, mu1, sd = self.llr_params(i)
        return 0.5 * stats.norm.pdf(x, mu0, sd) + 0.5 * stats.norm.pdf(x, mu1, sd)

    def mixture_cdf(self, x, i):
        mu0, mu1, sd = self.llr_params(i)
        return 0.5 * stats.norm.cdf(x, mu0, sd) + 0.5 * stats.norm.cdf(x, mu1, sd)
