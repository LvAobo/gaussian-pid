"""
experiment1_benchmark.py
========================

Construction of the noise-cancellation Gaussian benchmark used by
Experiments 1 and 3 of the paper.

The benchmark
-------------
A 7-dimensional jointly Gaussian system with target T = (T_2, T_3) and
N = 5 scalar sources S_1, ..., S_5 organised in two "noise-cancelling"
groups:

    Pair group (synergy of order 2 around T_2)
        S_1 = a * T_2 + U     + eps_1
        S_2 = a * T_2 - U     + eps_2

    Triple group (synergy of order 3 around T_3)
        S_3 = b * T_3 + V_1   + eps_3
        S_4 = b * T_3 + V_2   + eps_4
        S_5 = b * T_3 - V_1 - V_2 + eps_5

All latents (T_2, T_3, U, V_1, V_2, eps_1, ..., eps_5) are mutually
independent zero-mean Gaussians. Looking at any single S_i the target
component is contaminated by U or V_j; pooling the right pair (S_1+S_2)
or the right triple (S_3+S_4+S_5) cancels the nuisance latents
exactly, so the synergistic content lives at orders 2 and 3 only.

The columns of the population covariance matrix are laid out as
[T_2, T_3, S_1, S_2, S_3, S_4, S_5]; the target indices are (0, 1) and
the source indices are (2, 3, 4, 5, 6).

Public surface
--------------
* `BenchmarkParams`         : dataclass holding the construction parameters.
* `build_covariance(p)`     : returns the 7x7 population covariance.
* `sample_data(p, n, rng)`  : draw n samples from the benchmark.
* `ground_truth_spectrum(S)`: closed-form SE_K spectrum, all 10 pair
                              narrow synergies, all 10 triple narrow
                              synergies, TSE, and per-source Un.

This module depends only on `numpy` and on `gaussian_pid.py`.
All quantities are returned in nats.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import gaussian_pid as gp


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# Indices into the 7x7 population covariance matrix.
TARGET_INDICES: Tuple[int, ...] = (0, 1)            # T_2, T_3
SOURCE_INDICES: Tuple[int, ...] = (2, 3, 4, 5, 6)   # S_1, S_2, S_3, S_4, S_5
N_SOURCES: int = len(SOURCE_INDICES)


def _source_groups() -> List[Tuple[int, ...]]:
    """All five sources, each as its own scalar block."""
    return [(c,) for c in SOURCE_INDICES]


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkParams:
    """Container for the noise-cancellation construction parameters.

    Conventions:
      * `sigma_T2`, `sigma_T3` are the standard deviations of the two
        target components.
      * `sigma_eps_pair` is the noise std on S_1 and S_2 (held common to
        keep the pair symmetry that makes (S_1 + S_2)/2 a clean
        average).
      * `sigma_eps_triple` is the noise std on S_3, S_4, S_5 (likewise
        held common).
    """

    a: float                        # T_2 loading on S_1, S_2
    b: float                        # T_3 loading on S_3, S_4, S_5
    sigma_U: float                  # std of the pair-cancellation latent
    sigma_V: float                  # common std of V_1, V_2
    sigma_eps_pair: float           # noise std on S_1, S_2
    sigma_eps_triple: float         # noise std on S_3, S_4, S_5
    sigma_T2: float = 1.0           # std of T_2
    sigma_T3: float = 1.0           # std of T_3

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Covariance construction
# ---------------------------------------------------------------------------

# Latent layout, column index in the latent vector L:
#   0 : T_2
#   1 : T_3
#   2 : U
#   3 : V_1
#   4 : V_2
#   5 : eps_1
#   6 : eps_2
#   7 : eps_3
#   8 : eps_4
#   9 : eps_5
_LATENT_DIM = 10


def _loading_matrix(p: BenchmarkParams) -> np.ndarray:
    """Return A in the linear model [T; S] = A @ L, where L is the
    latent vector laid out as in the comment above.
    """
    A = np.zeros((7, _LATENT_DIM), dtype=float)
    # Targets
    A[0, 0] = 1.0                                   # T_2
    A[1, 1] = 1.0                                   # T_3
    # Pair group
    A[2, 0] = p.a;  A[2, 2] = +1.0;  A[2, 5] = 1.0  # S_1 = a T_2 + U + eps_1
    A[3, 0] = p.a;  A[3, 2] = -1.0;  A[3, 6] = 1.0  # S_2 = a T_2 - U + eps_2
    # Triple group
    A[4, 1] = p.b;  A[4, 3] = +1.0;  A[4, 7] = 1.0  # S_3 = b T_3 + V_1 + eps_3
    A[5, 1] = p.b;  A[5, 4] = +1.0;  A[5, 8] = 1.0  # S_4 = b T_3 + V_2 + eps_4
    A[6, 1] = p.b;  A[6, 3] = -1.0   # S_5: -V_1 ...
    A[6, 4] = -1.0                  #          ... -V_2 ...
    A[6, 9] = 1.0                   #              ... + eps_5
    return A


def _latent_variances(p: BenchmarkParams) -> np.ndarray:
    """Return the diagonal of Cov(L), the 10-dim latent vector."""
    return np.array([
        p.sigma_T2 ** 2,
        p.sigma_T3 ** 2,
        p.sigma_U ** 2,
        p.sigma_V ** 2,
        p.sigma_V ** 2,
        p.sigma_eps_pair ** 2,
        p.sigma_eps_pair ** 2,
        p.sigma_eps_triple ** 2,
        p.sigma_eps_triple ** 2,
        p.sigma_eps_triple ** 2,
    ], dtype=float)


def build_covariance(p: BenchmarkParams) -> np.ndarray:
    """Build the 7x7 population covariance matrix Cov(T_2, T_3, S_1..S_5).

    Since every observable is a linear combination of independent
    Gaussians the closed-form covariance is

        Sigma = A @ diag(Var(L)) @ A^T,

    which we explicitly symmetrise to defang any micro-asymmetry
    introduced by floating-point arithmetic.
    """
    A = _loading_matrix(p)
    var = _latent_variances(p)
    Sigma = A @ np.diag(var) @ A.T
    return (Sigma + Sigma.T) / 2.0


def sample_data(p: BenchmarkParams, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_samples` rows from the benchmark distribution.

    Returns an array of shape (n_samples, 7) with columns laid out as
    (T_2, T_3, S_1, S_2, S_3, S_4, S_5). Sampling is performed in the
    latent space and pushed through the loading matrix so that
    Cov(samples) is unbiased for `build_covariance(p)` (no Cholesky of a
    nearly-singular Sigma is ever required).
    """
    A = _loading_matrix(p)
    sd = np.sqrt(_latent_variances(p))
    L = rng.standard_normal((n_samples, _LATENT_DIM)) * sd
    return L @ A.T


# ---------------------------------------------------------------------------
# Closed-form ground-truth quantities
# ---------------------------------------------------------------------------

def _all_pairs() -> List[Tuple[int, int]]:
    """All (i, j) with 1 <= i < j <= 5 (1-indexed for human-friendly output)."""
    return [(i, j) for i, j in itertools.combinations(range(1, N_SOURCES + 1), 2)]


def _all_triples() -> List[Tuple[int, int, int]]:
    """All (i, j, k) with 1 <= i < j < k <= 5."""
    return [(i, j, k) for i, j, k in itertools.combinations(range(1, N_SOURCES + 1), 3)]


def _subset_to_columns(subset: Sequence[int]) -> List[Tuple[int, ...]]:
    """Map a 1-indexed source subset (e.g. (1, 2)) to the corresponding
    list of source-block tuples in covariance-column space.
    """
    return [(SOURCE_INDICES[i - 1],) for i in subset]


def ground_truth_spectrum(Sigma: np.ndarray, ridge: float = 0.0) -> Dict[str, Any]:
    """Compute every population hierarchy quantity needed by Figure 1.

    Returned dictionary
    -------------------
    spectrum        : {K -> SE_K} for K = 2, ..., 5  (closed-form, nats)
    pair_synergy    : {(i, j) -> narrow_synergy(S_i, S_j)} for all 10 pairs
    triple_synergy  : {(i, j, k) -> narrow_synergy(S_i, S_j, S_k)} for all 10 triples
    tse             : TSE^G of the full N=5 system
    general_unique  : {i -> Un^G(S_i -> T | rest)}
    tse_telescoping : sum_K SE_K (sanity check; equals tse for population)
    """
    target = list(TARGET_INDICES)
    sources = _source_groups()

    spec_pkg = gp.gaussian_synergy_spectrum(
        Sigma, target=target, sources=sources, return_components=True, ridge=ridge,
    )
    spectrum: Dict[int, float] = dict(spec_pkg["spectrum"])  # type: ignore[arg-type]
    tse_telescoping: float = float(spec_pkg["tse_from_telescoping"])  # type: ignore[index]

    pair_synergy: Dict[Tuple[int, int], float] = {}
    for pair in _all_pairs():
        pair_synergy[pair] = gp.gaussian_narrow_synergy(
            Sigma, target=target, sources=_subset_to_columns(pair), ridge=ridge,
        )

    triple_synergy: Dict[Tuple[int, int, int], float] = {}
    for triple in _all_triples():
        triple_synergy[triple] = gp.gaussian_narrow_synergy(
            Sigma, target=target, sources=_subset_to_columns(triple), ridge=ridge,
        )

    tse = gp.gaussian_tse(Sigma, target=target, sources=sources, ridge=ridge)

    general_unique = {
        i + 1: gp.gaussian_general_unique(
            Sigma, target=target, sources=sources, source_index=i, ridge=ridge,
        )
        for i in range(N_SOURCES)
    }

    return {
        "spectrum": spectrum,
        "pair_synergy": pair_synergy,
        "triple_synergy": triple_synergy,
        "tse": float(tse),
        "general_unique": general_unique,
        "tse_telescoping": tse_telescoping,
    }


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def format_spectrum(spec: Dict[int, float]) -> str:
    return "  ".join(f"SE_{k}={v: .4f}" for k, v in sorted(spec.items()))


def format_pair_synergies(d: Dict[Tuple[int, int], float]) -> str:
    return "\n".join(
        f"  pair {k} : Syn={v: .4f}" for k, v in sorted(d.items())
    )


def format_triple_synergies(d: Dict[Tuple[int, int, int], float]) -> str:
    return "\n".join(
        f"  triple {k} : Syn={v: .4f}" for k, v in sorted(d.items())
    )
