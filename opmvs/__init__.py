"""O-PEF MVS-A minimal implementable system.

Optimization-Driven Progressive Evidence Fusion — Minimal Validation System A
(SystemModel.md §29-§36: N=4 Gaussian detector, nested 0/1/2/4-bit evidence,
exact DAG-DP oracle, O-PEF-1 / O-PEF-2E solvers, Monte Carlo evaluation).

Submodules
----------
model      : Gaussian local detector (H0/H1 LLR statistics)   [§30-§32]
quantizer  : per-UAV nested binary partition tree + message PMF/LLR  [§7, §14]
state      : evidence state z, mixed-radix encoding, state space   [§10, §20]
fusion     : log-domain posterior utilities (softplus/logsumexp)   [§11]
dp         : exact acyclic backward DAG-DP                        [§18, §21]
opef       : O-PEF-1 (depth-1) and O-PEF-2E (depth-2 exact)       [§23-§27]
mc         : vectorized Monte Carlo episode simulation + metrics  [§33, §57]
baselines  : B0..B11 reference and mechanism baselines            [§49-§53]
gates      : G0/G1/G2 gate checks                                 [§34-§36, §66]
"""

from . import model, quantizer, state, fusion, dp, opef, mc, baselines, gates, eval_exact, rbl, cmdp, sparse
from .model import GaussianDetectorModel
from .quantizer import NestedQuantizer
from .state import StateSpace, R_LEVELS, RMAX, z_code, z_decode, action_code, action_decode
from .dp import ExactDP, SolverBase
from .opef import OPEF1, OPEF2, OPEF3
from .rbl import ResourceBoundedLookahead, OnlinePlanner
from .sparse import SparsePlanner, z_code_b, z_decode_b, BASE_B

__version__ = "0.6.0"
__all__ = [
    "model", "quantizer", "state", "fusion", "dp", "opef", "mc", "baselines", "gates",
    "eval_exact", "rbl", "cmdp", "sparse",
    "GaussianDetectorModel", "NestedQuantizer", "StateSpace",
    "R_LEVELS", "RMAX", "z_code", "z_decode", "action_code", "action_decode",
    "ExactDP", "SolverBase", "OPEF1", "OPEF2", "OPEF3", "ResourceBoundedLookahead",
    "OnlinePlanner", "SparsePlanner", "z_code_b", "z_decode_b", "BASE_B",
]
