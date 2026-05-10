r"""
Closed-form Gaussian estimators for the conditional-copy multivariate
information decomposition framework of Lyu et al. (2026).

Implemented quantities
----------------------
- Gaussian entropy, mutual information, conditional mutual information.
- Total correlation, dual total correlation, O-information.
- Two-source PID (paper Theorem 1):
      Red, Un_1, Un_2, Syn  via the conditional-copy construction.
- General unique information (paper Definition 2 / Theorem 3):
      Un^G(S_i -> T | S_{\i}) = (1/2) log det(Psi^orig_{[N]\{i}}) / det(Psi_{U_i}).
- K-th order synergistic effect (paper Definition 4 / Theorem 5):
      SE_K^G = (1/2) log det(Psi_{C_{K-1}}) / det(Psi_{C_K}),
  where C_K is the family of all K-subsets and Psi_{C_K} is the conditional
  covariance of T given the corresponding K-subset copy family.
- Narrow synergy = SE_N (paper Corollary 1).
- Total synergistic effect TSE (paper Theorem 6 / Proposition 1):
      TSE^G = (1/2) log det(Psi_{C_1}) / det(Psi^orig_{[N]}).

Conventions
-----------
- Inputs are column indices of a covariance matrix.
- `target` may be scalar or vector-valued.
- `sources` is a list of source groups; each group may be scalar or
  vector-valued. Examples:
      target = [0]
      sources = [1, 2, 3]                  # three scalar sources
      sources = [[1, 2], [3, 4], [5, 6]]   # three vector-valued sources
- Logs are natural by default (nats). Set base=2 for bits.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from math import e, log, pi
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

IndexLike = Union[int, np.integer, Sequence[int], np.ndarray]
SourceGroups = Sequence[IndexLike]


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SPDInfo:
    """Numerical diagnostics for a covariance submatrix."""

    dim: int
    ridge: float
    min_eig: float
    max_eig: float
    condition_number: float


@dataclass(frozen=True)
class TwoSourcePID:
    """Two-source PID values induced by the Gaussian conditional-copy synergy."""

    mi_s1: float
    mi_s2: float
    mi_joint: float
    redundancy: float
    unique_s1: float
    unique_s2: float
    synergy: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Input normalization and covariance utilities
# -----------------------------------------------------------------------------

def _as_tuple(group: IndexLike) -> Tuple[int, ...]:
    """Normalize an index or index collection to a tuple of ints."""
    if isinstance(group, (int, np.integer)):
        return (int(group),)
    arr = np.asarray(group).astype(int).ravel()
    if arr.size == 0:
        raise ValueError("Index groups cannot be empty.")
    return tuple(int(x) for x in arr)


def normalize_target(target: IndexLike) -> Tuple[int, ...]:
    return _as_tuple(target)


def normalize_sources(sources: SourceGroups) -> Tuple[Tuple[int, ...], ...]:
    if len(sources) == 0:
        raise ValueError("At least one source group is required.")
    groups = tuple(_as_tuple(s) for s in sources)
    flat = [idx for g in groups for idx in g]
    if len(flat) != len(set(flat)):
        raise ValueError("Source groups must be disjoint; duplicate indices found.")
    return groups


def _check_disjoint(target: Tuple[int, ...], sources: Tuple[Tuple[int, ...], ...]) -> None:
    tset = set(target)
    sflat = [idx for g in sources for idx in g]
    overlap = tset.intersection(sflat)
    if overlap:
        raise ValueError(f"Target and sources must be disjoint; overlap={sorted(overlap)}")


def flatten(groups: Iterable[Sequence[int]]) -> Tuple[int, ...]:
    return tuple(idx for g in groups for idx in g)


def _validate_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square 2D array.")
    if not np.allclose(cov, cov.T, atol=1e-8, rtol=1e-6):
        raise ValueError("cov must be symmetric within numerical tolerance.")
    return cov


def subcov(cov: np.ndarray, indices: Sequence[int], ridge: float = 0.0) -> np.ndarray:
    """Return covariance submatrix with optional diagonal ridge."""
    cov = _validate_cov(cov)
    idx = tuple(int(i) for i in indices)
    if len(idx) == 0:
        raise ValueError("Cannot form covariance for an empty index set.")
    m = cov[np.ix_(idx, idx)].copy()
    if ridge > 0:
        m += ridge * np.eye(len(idx))
    return m


def to_correlation(cov: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Convert covariance matrix to Pearson correlation matrix."""
    cov = _validate_cov(cov)
    d = np.sqrt(np.maximum(np.diag(cov), eps))
    corr = cov / np.outer(d, d)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def logdet_spd(mat: np.ndarray) -> float:
    """Stable log determinant for a symmetric positive definite matrix."""
    mat = np.asarray(mat, dtype=float)
    mat = (mat + mat.T) / 2.0
    try:
        chol = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError as exc:
        eigvals = np.linalg.eigvalsh(mat)
        raise np.linalg.LinAlgError(
            "Matrix is not positive definite. "
            f"min_eig={eigvals.min():.3e}, max_eig={eigvals.max():.3e}. "
            "Increase ridge or inspect duplicated / deterministic variables."
        ) from exc
    return 2.0 * float(np.sum(np.log(np.diag(chol))))


def spd_diagnostics(mat: np.ndarray, ridge: float = 0.0) -> SPDInfo:
    """Return eigenvalue-based SPD diagnostics."""
    mat = np.asarray(mat, dtype=float)
    mat = (mat + mat.T) / 2.0
    if ridge > 0:
        mat = mat + ridge * np.eye(mat.shape[0])
    eig = np.linalg.eigvalsh(mat)
    min_eig = float(eig.min())
    max_eig = float(eig.max())
    cond = float(np.inf if min_eig <= 0 else max_eig / min_eig)
    return SPDInfo(
        dim=int(mat.shape[0]),
        ridge=float(ridge),
        min_eig=min_eig,
        max_eig=max_eig,
        condition_number=cond,
    )


def _convert_base(value_nats: float, base: float) -> float:
    if base <= 0 or np.isclose(base, 1.0):
        raise ValueError("base must be positive and not equal to 1.")
    return float(value_nats / log(base))


# -----------------------------------------------------------------------------
# Gaussian entropy, mutual information, conditional mutual information
# -----------------------------------------------------------------------------

def gaussian_entropy_cov(cov_block: np.ndarray, base: float = e, ridge: float = 0.0) -> float:
    """Differential entropy of a Gaussian vector from its covariance block."""
    cov_block = _validate_cov(cov_block)
    dim = cov_block.shape[0]
    if ridge > 0:
        cov_block = cov_block + ridge * np.eye(dim)
    h_nats = 0.5 * (dim * log(2.0 * pi * e) + logdet_spd(cov_block))
    return _convert_base(h_nats, base)


def gaussian_entropy(cov: np.ndarray, variables: IndexLike, base: float = e, ridge: float = 0.0) -> float:
    """Differential entropy H(X_variables)."""
    idx = _as_tuple(variables)
    return gaussian_entropy_cov(subcov(cov, idx, ridge=0.0), base=base, ridge=ridge)


def gaussian_mi(
    cov: np.ndarray,
    x: IndexLike,
    y: IndexLike,
    base: float = e,
    ridge: float = 0.0,
) -> float:
    """Gaussian mutual information I(X;Y)."""
    x_idx = _as_tuple(x)
    y_idx = _as_tuple(y)
    xy_idx = x_idx + y_idx
    h_x = gaussian_entropy(cov, x_idx, base=base, ridge=ridge)
    h_y = gaussian_entropy(cov, y_idx, base=base, ridge=ridge)
    h_xy = gaussian_entropy(cov, xy_idx, base=base, ridge=ridge)
    return float(h_x + h_y - h_xy)


def gaussian_cmi(
    cov: np.ndarray,
    x: IndexLike,
    y: IndexLike,
    z: Optional[IndexLike] = None,
    base: float = e,
    ridge: float = 0.0,
) -> float:
    """Gaussian conditional mutual information I(X;Y|Z)."""
    if z is None:
        return gaussian_mi(cov, x, y, base=base, ridge=ridge)
    x_idx = _as_tuple(x)
    y_idx = _as_tuple(y)
    z_idx = _as_tuple(z)
    h_xz = gaussian_entropy(cov, x_idx + z_idx, base=base, ridge=ridge)
    h_yz = gaussian_entropy(cov, y_idx + z_idx, base=base, ridge=ridge)
    h_z = gaussian_entropy(cov, z_idx, base=base, ridge=ridge)
    h_xyz = gaussian_entropy(cov, x_idx + y_idx + z_idx, base=base, ridge=ridge)
    return float(h_xz + h_yz - h_z - h_xyz)


# -----------------------------------------------------------------------------
# TC, DTC, O-information
# -----------------------------------------------------------------------------

def total_correlation(
    cov: np.ndarray,
    variables: Optional[IndexLike] = None,
    groups: Optional[SourceGroups] = None,
    base: float = e,
    ridge: float = 0.0,
) -> float:
    """
    Gaussian total correlation for groups X_1,...,X_N.

    If `groups` is omitted, each variable in `variables` is treated as a scalar group.
    If `groups` is provided, each group may be vector-valued.
    """
    if groups is None:
        if variables is None:
            raise ValueError("Either variables or groups must be provided.")
        idx = _as_tuple(variables)
        groups_norm = tuple((i,) for i in idx)
    else:
        groups_norm = normalize_sources(groups)

    all_idx = flatten(groups_norm)
    h_sum = sum(gaussian_entropy(cov, g, base=base, ridge=ridge) for g in groups_norm)
    h_joint = gaussian_entropy(cov, all_idx, base=base, ridge=ridge)
    return float(h_sum - h_joint)


def dual_total_correlation(
    cov: np.ndarray,
    variables: Optional[IndexLike] = None,
    groups: Optional[SourceGroups] = None,
    base: float = e,
    ridge: float = 0.0,
) -> float:
    """
    Gaussian dual total correlation for groups X_1,...,X_N.

    DTC(X_1,...,X_N) = H(X_all) - sum_i H(X_i | X_-i).
    """
    if groups is None:
        if variables is None:
            raise ValueError("Either variables or groups must be provided.")
        idx = _as_tuple(variables)
        groups_norm = tuple((i,) for i in idx)
    else:
        groups_norm = normalize_sources(groups)

    n = len(groups_norm)
    if n < 2:
        return 0.0
    all_idx = flatten(groups_norm)
    h_all = gaussian_entropy(cov, all_idx, base=base, ridge=ridge)
    residual_sum = 0.0
    for i in range(n):
        rest_groups = [g for j, g in enumerate(groups_norm) if j != i]
        rest_idx = flatten(rest_groups)
        h_rest = gaussian_entropy(cov, rest_idx, base=base, ridge=ridge)
        residual_sum += h_all - h_rest
    return float(h_all - residual_sum)


def o_information(
    cov: np.ndarray,
    variables: Optional[IndexLike] = None,
    groups: Optional[SourceGroups] = None,
    base: float = e,
    ridge: float = 0.0,
) -> float:
    """
    Gaussian O-information Omega = TC - DTC (Rosas et al. 2019).

    Positive values indicate redundancy dominance; negative values indicate
    synergy dominance in the O-information sense.
    """
    tc = total_correlation(cov, variables=variables, groups=groups, base=base, ridge=ridge)
    dtc = dual_total_correlation(cov, variables=variables, groups=groups, base=base, ridge=ridge)
    return float(tc - dtc)


# -----------------------------------------------------------------------------
# Conditional-copy surrogate covariance (paper Lemma 1)
# -----------------------------------------------------------------------------

def conditional_independent_surrogate_cov(
    cov: np.ndarray,
    target: IndexLike,
    family_blocks: Sequence[IndexLike],
    ridge_target: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    """
    Build the Gaussian surrogate covariance for (T, S'_{A_1}, ..., S'_{A_m}),
    where {A_1,...,A_m} is a family of source-index blocks (paper Lemma 1).

    Each surrogate block S'_{A_a}:
      - has the same conditional distribution given T as S_{A_a}
        (so Cov(S'_{A_a}, T) = Sigma_{A_a T} and Cov(S'_{A_a}) = Sigma_{A_a A_a}),
      - is conditionally independent of S'_{A_b} (a != b) given T,
      hence Cov(S'_{A_a}, S'_{A_b}) = Sigma_{A_a T} Sigma_T^{-1} Sigma_{T A_b}.

    The same column may appear in multiple blocks (e.g. when family_blocks are
    overlapping K-subsets); each occurrence becomes its own independent copy.

    Parameters
    ----------
    cov:
        Original covariance matrix.
    target:
        Indices of the target.
    family_blocks:
        Sequence of blocks; each block is a sequence of column indices into `cov`.
    ridge_target:
        Optional ridge added to Sigma_{TT} before solving (numerical stability).

    Returns
    -------
    surrogate_cov:
        Covariance ordered as [T, S'_{A_1}, S'_{A_2}, ...].
    target_local:
        Local indices of T within surrogate_cov.
    block_locals:
        Tuple of local index tuples for each surrogate block, in the same order
        as family_blocks.
    """
    cov = _validate_cov(cov)
    t = normalize_target(target)
    blocks = tuple(_as_tuple(b) for b in family_blocks)
    if len(blocks) == 0:
        raise ValueError("family_blocks must be non-empty.")

    # Validate target/blocks disjointness against T (blocks themselves may overlap).
    tset = set(t)
    for b in blocks:
        if tset.intersection(b):
            raise ValueError("Target indices must not appear in any source block.")

    dt = len(t)
    block_dims = [len(b) for b in blocks]
    total = dt + sum(block_dims)

    M = np.zeros((total, total))
    # Top-left: Sigma_TT
    M[:dt, :dt] = cov[np.ix_(t, t)]

    # Layout local index ranges for each block.
    local_t = tuple(range(dt))
    block_locals: List[Tuple[int, ...]] = []
    cursor = dt
    for d in block_dims:
        block_locals.append(tuple(range(cursor, cursor + d)))
        cursor += d

    # Cov(T, S'_{A_a}) = Sigma_{T A_a} (Lemma 1, eq. (33))
    # and diagonal block Cov(S'_{A_a}) = Sigma_{A_a A_a}.
    for a, (b_cols, b_loc) in enumerate(zip(blocks, block_locals)):
        Sigma_T_a = cov[np.ix_(t, b_cols)]
        M[np.ix_(local_t, b_loc)] = Sigma_T_a
        M[np.ix_(b_loc, local_t)] = Sigma_T_a.T
        M[np.ix_(b_loc, b_loc)] = cov[np.ix_(b_cols, b_cols)]

    # Off-diagonal cross-copy blocks (Lemma 1, eq. (34), a != b).
    Sigma_TT = cov[np.ix_(t, t)].copy()
    if ridge_target > 0:
        Sigma_TT = Sigma_TT + ridge_target * np.eye(dt)

    for a, (ba_cols, ba_loc) in enumerate(zip(blocks, block_locals)):
        Sigma_a_T = cov[np.ix_(ba_cols, t)]
        for b, (bb_cols, bb_loc) in enumerate(zip(blocks, block_locals)):
            if a == b:
                continue
            Sigma_T_b = cov[np.ix_(t, bb_cols)]
            # Sigma_{A_a T} Sigma_TT^{-1} Sigma_{T A_b}, computed via solve.
            middle = np.linalg.solve(Sigma_TT, Sigma_T_b)
            M[np.ix_(ba_loc, bb_loc)] = Sigma_a_T @ middle

    M = (M + M.T) / 2.0
    return M, local_t, tuple(block_locals)


def _conditional_entropy_T_given_blocks(
    surrogate_cov: np.ndarray,
    target_local: Tuple[int, ...],
    block_locals: Sequence[Tuple[int, ...]],
    base: float = e,
    ridge: float = 0.0,
) -> float:
    """H(T | Y) = H(T,Y) - H(Y) for a given (surrogate) covariance and layout."""
    flat_blocks = flatten(block_locals)
    h_joint = gaussian_entropy(surrogate_cov, target_local + flat_blocks, base=base, ridge=ridge)
    h_blocks = gaussian_entropy(surrogate_cov, flat_blocks, base=base, ridge=ridge)
    return float(h_joint - h_blocks)


def _conditional_entropy_T_given_original(
    cov: np.ndarray,
    target: Tuple[int, ...],
    cond_indices: Sequence[int],
    base: float = e,
    ridge: float = 0.0,
) -> float:
    """H(T | X) computed on the ORIGINAL covariance (no surrogate)."""
    cond = tuple(int(i) for i in cond_indices)
    if len(cond) == 0:
        return gaussian_entropy(cov, target, base=base, ridge=ridge)
    h_joint = gaussian_entropy(cov, tuple(target) + cond, base=base, ridge=ridge)
    h_cond = gaussian_entropy(cov, cond, base=base, ridge=ridge)
    return float(h_joint - h_cond)


# -----------------------------------------------------------------------------
# Total Synergistic Effect (paper Theorem 6 / Proposition 1)
# -----------------------------------------------------------------------------

def gaussian_tse(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
) -> float:
    """
    Total Synergistic Effect (Theorem 6).

    TSE^G = (1/2) log det(Psi_{C_1}) / det(Psi^orig_{[N]})
          = H(T | S'_1, ..., S'_N) - H(T | S_1, ..., S_N),

    where the singleton family C_1 = {{1},...,{N}} is the conditional-copy
    family of size-1 blocks; the surrogate replaces every cross-source block
    by its conditional-independence value Sigma_{S_i T} Sigma_T^{-1} Sigma_{T S_j}.
    """
    t = normalize_target(target)
    s_groups = normalize_sources(sources)
    _check_disjoint(t, s_groups)

    # Family C_1: each source group is its own block.
    surrogate, t_loc, block_locs = conditional_independent_surrogate_cov(
        cov, target=t, family_blocks=s_groups, ridge_target=ridge_target
    )
    h_T_given_singletons = _conditional_entropy_T_given_blocks(
        surrogate, t_loc, block_locs, base=base, ridge=ridge
    )

    source_flat = flatten(s_groups)
    h_T_given_all = _conditional_entropy_T_given_original(
        cov, t, source_flat, base=base, ridge=ridge
    )
    return float(h_T_given_singletons - h_T_given_all)


# -----------------------------------------------------------------------------
# K-th order synergistic effect (paper Definition 4 / Theorem 5)
# -----------------------------------------------------------------------------

def _k_subset_blocks(s_groups: Sequence[Tuple[int, ...]], k: int) -> List[Tuple[int, ...]]:
    """Build the block list for family C_k = all K-subsets of [N].

    Each block is the concatenation of the source columns of one K-subset.
    For k == n the family contains the single full block (and gives the original
    joint distribution after the copy construction).
    """
    n = len(s_groups)
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= N (got k={k}, N={n}).")
    blocks: List[Tuple[int, ...]] = []
    for subset in combinations(range(n), k):
        cols: List[int] = []
        for i in subset:
            cols.extend(s_groups[i])
        blocks.append(tuple(cols))
    return blocks


def _conditional_entropy_T_given_family(
    cov: np.ndarray,
    target: Tuple[int, ...],
    s_groups: Sequence[Tuple[int, ...]],
    k: int,
    base: float,
    ridge: float,
    ridge_target: float,
) -> float:
    """H(T | Y_{C_k}), with the K-subset copy family.

    For k == N, the family C_N = {[N]} is a single block, so the surrogate
    distribution coincides with the original joint distribution and
    H(T | Y_{C_N}) = H(T | S_1,...,S_N) computed directly on the original cov.
    """
    n = len(s_groups)
    if k == n:
        source_flat = flatten(s_groups)
        return _conditional_entropy_T_given_original(
            cov, target, source_flat, base=base, ridge=ridge
        )
    blocks = _k_subset_blocks(s_groups, k)
    surrogate, t_loc, block_locs = conditional_independent_surrogate_cov(
        cov, target=target, family_blocks=blocks, ridge_target=ridge_target
    )
    return _conditional_entropy_T_given_blocks(
        surrogate, t_loc, block_locs, base=base, ridge=ridge
    )


def gaussian_se_k(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    k: int,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
) -> float:
    """
    K-th order synergistic effect SE_K (paper Definition 4 / Theorem 5).

        SE_K^G = H(T | Y_{C_{K-1}}) - H(T | Y_{C_K})
               = (1/2) log det(Psi_{C_{K-1}}) / det(Psi_{C_K}),

    where C_K is the family of all K-subsets and Y_{C_K} stacks the
    corresponding K-subset conditional copies (each pair of distinct copies is
    conditionally independent given T; each copy preserves the within-subset
    joint distribution given T).
    """
    t = normalize_target(target)
    s_groups = normalize_sources(sources)
    _check_disjoint(t, s_groups)
    n = len(s_groups)
    if k < 2 or k > n:
        raise ValueError(f"k must satisfy 2 <= k <= N (got k={k}, N={n}).")

    h_km1 = _conditional_entropy_T_given_family(
        cov, t, s_groups, k - 1, base=base, ridge=ridge, ridge_target=ridge_target
    )
    h_k = _conditional_entropy_T_given_family(
        cov, t, s_groups, k, base=base, ridge=ridge, ridge_target=ridge_target
    )
    return float(h_km1 - h_k)


def gaussian_narrow_synergy(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
) -> float:
    """Narrow synergy = SE_N (paper Corollary 1)."""
    n = len(normalize_sources(sources))
    return gaussian_se_k(
        cov, target, sources, k=n, base=base, ridge=ridge, ridge_target=ridge_target
    )


def gaussian_synergy_spectrum(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
    return_components: bool = False,
) -> Union[Dict[int, float], Dict[str, object]]:
    """
    Order-resolved synergistic spectrum (SE_2, SE_3, ..., SE_N).

    Reuses the conditional entropies H(T | Y_{C_k}) across orders so each
    surrogate covariance is built once.

    Parameters
    ----------
    return_components:
        If True, also returns the cached H(T | Y_{C_k}) values and the
        telescoping consistency check sum_K SE_K vs TSE.
    """
    t = normalize_target(target)
    s_groups = normalize_sources(sources)
    _check_disjoint(t, s_groups)
    n = len(s_groups)

    cond_entropies: Dict[int, float] = {}
    for k in range(1, n + 1):
        cond_entropies[k] = _conditional_entropy_T_given_family(
            cov, t, s_groups, k, base=base, ridge=ridge, ridge_target=ridge_target
        )

    spectrum: Dict[int, float] = {}
    for k in range(2, n + 1):
        spectrum[k] = float(cond_entropies[k - 1] - cond_entropies[k])

    if not return_components:
        return spectrum

    # Telescoping check: sum_K SE_K should equal H(T|Y_{C_1}) - H(T|Y_{C_N}) = TSE.
    tse_from_spectrum = float(cond_entropies[1] - cond_entropies[n])
    return {
        "spectrum": spectrum,
        "conditional_entropies": cond_entropies,
        "tse_from_telescoping": tse_from_spectrum,
    }


# -----------------------------------------------------------------------------
# General unique information (paper Definition 2 / Theorem 3)
# -----------------------------------------------------------------------------

def gaussian_general_unique(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    source_index: int,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
) -> float:
    r"""
    General unique information (paper Theorem 3):

        Un^G(S_i -> T | S_{\i}) = (1/2) log det(Psi^orig_{[N]\{i}}) / det(Psi_{U_i}),

    where U_i = {{i}, [N]\{i}} is a two-block family in which S'_i is the
    conditional copy of S_i (independent of S_{\i} given T) while S'_{[N]\{i}}
    is the conditional copy of the rest (preserving its within-block joint
    distribution given T).

    Equivalently:
        Un^G = H(T | S_{\i})_orig  -  H(T | S'_i, S'_{\i})_surrogate.
    """
    t = normalize_target(target)
    s_groups = normalize_sources(sources)
    _check_disjoint(t, s_groups)
    i = int(source_index)
    if i < 0 or i >= len(s_groups):
        raise IndexError("source_index out of range.")

    si = s_groups[i]
    rest_groups = [g for j, g in enumerate(s_groups) if j != i]

    # Family U_i = {{i}, [N]\{i}} -- two blocks, the second being the full rest block.
    if len(rest_groups) == 0:
        # Single source: there is no "rest"; Un degenerates to I(T;S_i).
        return gaussian_mi(cov, t, si, base=base, ridge=ridge)

    rest_flat = flatten(rest_groups)
    family_blocks = [si, rest_flat]
    surrogate, t_loc, block_locs = conditional_independent_surrogate_cov(
        cov, target=t, family_blocks=family_blocks, ridge_target=ridge_target
    )
    h_T_given_surrogate = _conditional_entropy_T_given_blocks(
        surrogate, t_loc, block_locs, base=base, ridge=ridge
    )
    h_T_given_rest_orig = _conditional_entropy_T_given_original(
        cov, t, rest_flat, base=base, ridge=ridge
    )
    return float(h_T_given_rest_orig - h_T_given_surrogate)


# -----------------------------------------------------------------------------
# Two-source PID (paper Theorem 1)
# -----------------------------------------------------------------------------

def gaussian_two_source_pid(
    cov: np.ndarray,
    target: IndexLike,
    source1: IndexLike,
    source2: IndexLike,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
) -> TwoSourcePID:
    """
    Two-source PID induced by the conditional-copy construction (Theorem 1).

    Atoms (closed forms):
      Un_1 = Un^G(S_1 -> T | S_2)        -- general unique with N=2
      Un_2 = Un^G(S_2 -> T | S_1)
      Syn  = SE_2 = TSE  (when N=2, narrow synergy equals TSE)
      Red  = I(T;S_1) - Un_1
            = I(T;S_2) - Un_2
            = I(T;S_1) + I(T;S_2) - I(T;S_1,S_2) + Syn

    The three identities for Red are algebraically equivalent under the PID
    consistency constraints (eqs. (8)-(11) in the paper). We use the first form
    via Un_1 to avoid relying solely on the algebraic identity.
    """
    t = normalize_target(target)
    s1 = _as_tuple(source1)
    s2 = _as_tuple(source2)
    _check_disjoint(t, normalize_sources([s1, s2]))

    mi1 = gaussian_mi(cov, t, s1, base=base, ridge=ridge)
    mi2 = gaussian_mi(cov, t, s2, base=base, ridge=ridge)
    mi12 = gaussian_mi(cov, t, s1 + s2, base=base, ridge=ridge)

    u1 = gaussian_general_unique(
        cov, t, [s1, s2], source_index=0,
        base=base, ridge=ridge, ridge_target=ridge_target,
    )
    u2 = gaussian_general_unique(
        cov, t, [s1, s2], source_index=1,
        base=base, ridge=ridge, ridge_target=ridge_target,
    )
    syn = gaussian_tse(
        cov, t, [s1, s2],
        base=base, ridge=ridge, ridge_target=ridge_target,
    )
    red = mi1 - u1
    return TwoSourcePID(
        mi_s1=float(mi1),
        mi_s2=float(mi2),
        mi_joint=float(mi12),
        redundancy=float(red),
        unique_s1=float(u1),
        unique_s2=float(u2),
        synergy=float(syn),
    )


# -----------------------------------------------------------------------------
# Diagnostics (NOT paper PID atoms; useful auxiliary quantities)
# -----------------------------------------------------------------------------

def cmi_source_given_rest(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    source_index: int,
    base: float = e,
    ridge: float = 0.0,
) -> float:
    r"""
    Diagnostic: conditional mutual information I(T; S_i | S_{\i}) on the
    ORIGINAL distribution (no copy construction).

    NOTE: This is NOT the paper's general unique information Un^G; for that
    use `gaussian_general_unique`. This quantity is provided as a
    source-indispensability diagnostic.
    """
    t = normalize_target(target)
    s_groups = normalize_sources(sources)
    _check_disjoint(t, s_groups)
    i = int(source_index)
    if i < 0 or i >= len(s_groups):
        raise IndexError("source_index out of range.")
    si = s_groups[i]
    rest = flatten(g for j, g in enumerate(s_groups) if j != i)
    if len(rest) == 0:
        return gaussian_mi(cov, t, si, base=base, ridge=ridge)
    return gaussian_cmi(cov, t, si, rest, base=base, ridge=ridge)


def leave_one_out_delta_tse(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    source_index: int,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
) -> float:
    """
    Diagnostic: TSE(T; all sources) - TSE(T; sources without source_index).
    """
    s_groups = list(normalize_sources(sources))
    i = int(source_index)
    if i < 0 or i >= len(s_groups):
        raise IndexError("source_index out of range.")
    full = gaussian_tse(cov, target, s_groups, base=base, ridge=ridge, ridge_target=ridge_target)
    reduced = [g for j, g in enumerate(s_groups) if j != i]
    if len(reduced) < 2:
        red_val = 0.0
    else:
        red_val = gaussian_tse(
            cov, target, reduced, base=base, ridge=ridge, ridge_target=ridge_target
        )
    return float(full - red_val)


# -----------------------------------------------------------------------------
# High-level summary
# -----------------------------------------------------------------------------

def gaussian_multisource_summary(
    cov: np.ndarray,
    target: IndexLike,
    sources: SourceGroups,
    base: float = e,
    ridge: float = 0.0,
    ridge_target: float = 0.0,
) -> Dict[str, object]:
    """High-level summary: joint MI, full SE_K spectrum, TSE, general unique
    informations, plus a couple of diagnostics."""
    t = normalize_target(target)
    s_groups = normalize_sources(sources)
    _check_disjoint(t, s_groups)
    n = len(s_groups)
    source_flat = flatten(s_groups)

    mi_joint = gaussian_mi(cov, t, source_flat, base=base, ridge=ridge)

    spectrum_pkg = gaussian_synergy_spectrum(
        cov, t, s_groups, base=base, ridge=ridge,
        ridge_target=ridge_target, return_components=True,
    )
    spectrum = spectrum_pkg["spectrum"]                    # type: ignore[index]
    cond_ent = spectrum_pkg["conditional_entropies"]       # type: ignore[index]
    tse_from_spec = spectrum_pkg["tse_from_telescoping"]   # type: ignore[index]

    tse_direct = gaussian_tse(
        cov, t, s_groups, base=base, ridge=ridge, ridge_target=ridge_target
    )
    narrow_syn = spectrum.get(n, 0.0) if n >= 2 else 0.0   # type: ignore[union-attr]

    general_unique = {
        i: gaussian_general_unique(
            cov, t, s_groups, i, base=base, ridge=ridge, ridge_target=ridge_target
        )
        for i in range(n)
    }
    cmi_diag = {
        i: cmi_source_given_rest(cov, t, s_groups, i, base=base, ridge=ridge)
        for i in range(n)
    }
    loo_tse = {
        i: leave_one_out_delta_tse(
            cov, t, s_groups, i, base=base, ridge=ridge, ridge_target=ridge_target
        )
        for i in range(n)
    }

    return {
        "target": t,
        "sources": s_groups,
        "base": base,
        "units": "bits" if np.isclose(base, 2.0) else "nats",
        "joint_mi": float(mi_joint),
        "tse": float(tse_direct),
        "tse_from_spectrum_telescoping": float(tse_from_spec),
        "narrow_synergy_SE_N": float(narrow_syn),
        "synergy_spectrum_SE_K": spectrum,
        "conditional_entropies_T_given_C_k": cond_ent,
        "general_unique": general_unique,
        "diagnostics": {
            "cmi_source_given_rest": cmi_diag,
            "leave_one_out_delta_tse": loo_tse,
            "o_information_sources": o_information(
                cov, groups=s_groups, base=base, ridge=ridge
            ),
            "tc_sources": total_correlation(
                cov, groups=s_groups, base=base, ridge=ridge
            ),
            "dtc_sources": dual_total_correlation(
                cov, groups=s_groups, base=base, ridge=ridge
            ),
            "spd_all": asdict(spd_diagnostics(subcov(cov, t + source_flat), ridge=ridge)),
        },
    }