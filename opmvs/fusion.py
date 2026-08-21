"""Log-domain numerical utilities (SystemModel §11).

All likelihoods / posteriors / message-probability normalizations are computed
in the log domain to avoid underflow / saturation for |Omega| >> 0.
"""
from __future__ import annotations

import numpy as np


def softplus(x):
    """log(1 + exp(x)), numerically stable for large |x|."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def log_sigmoid(omega):
    """log sigma(omega) = log(1/(1+exp(-omega))) = -softplus(-omega)."""
    return -softplus(-np.asarray(omega, dtype=float))


def log_one_minus_sigmoid(omega):
    """log(1 - sigma(omega)) = -softplus(omega)."""
    return -softplus(np.asarray(omega, dtype=float))


def logsumexp2(a, b):
    """log(exp(a) + exp(b)) elementwise, numerically stable."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.maximum(a, b)
    return m + np.log1p(np.exp(-np.abs(a - b)))


def log_mix2(log_p, log_q, log_l1, log_l0):
    """log( p*exp(log_l1) + (1-p)*exp(log_l0) ) given log p and log(1-p)."""
    return logsumexp2(log_p + log_l1, log_q + log_l0)


def posterior_from_odds(omega, prior_log_odds=0.0):
    """Posterior P(H1 | evidence) from log posterior odds, in log domain.

    Returns (log p, log(1-p), p) with p = P(H1|*), 1-p = P(H0|*).
    """
    om = np.asarray(omega, dtype=float) + prior_log_odds
    logp = log_sigmoid(om)
    logq = log_one_minus_sigmoid(om)
    p = np.exp(logp)
    return logp, logq, p
