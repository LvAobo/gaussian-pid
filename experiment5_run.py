"""
experiment5_run.py
==================

Experiment 5 ("N=2 sanity check") -- arXiv full-version appendix.

Goal
----
Numerically compare our closed-form Gaussian two-source PID estimator
(Red, Un_1, Un_2, Syn) against Barrett's mutual-information minimum
(MMI) baseline on three controlled Gaussian benchmarks that span the
Red / Un / Syn corner of the PID triangle. The output is a single
LaTeX table that anchors our estimator within the existing two-source
Gaussian PID literature.

Configurations (all jointly Gaussian, T scalar, S_i scalar)
-----------------------------------------------------------
1. Pure redundancy:   S_1 = T + eps_1,   S_2 = T + eps_2
                      T ~ N(0, 1),  eps_i ~ N(0, 1)  (eps's independent)
                      Both sources are independent noisy observations
                      of the same target signal -- the textbook
                      "redundancy" benchmark.
2. Pure unique:       T = S_1 + eps_T,   S_2 ~ N(0, 1) independent
                      sigma_eps_T = 1
                      Only S_1 informs T; S_2 is decoupled.
3. Pure synergy:      T = S_1 + S_2 + eps_T,   S_i ~ N(0, 1) independent
                      sigma_eps_T = 1
                      Gaussian XOR: each source individually carries
                      only partial info; only their joint observation
                      pins down T.
4. Mixed (correlated): T = S_1 + S_2 + eps_T, with Cov(S_1, S_2) = 0.3
                       and S_i marginal N(0, 1)
                       Sources are correlated, blurring the unique vs
                       redundant boundary.
5. Mixed (asymmetric): T = 2*S_1 + S_2 + eps_T, S_i ~ N(0, 1) independent
                       S_1 and S_2 are independent but enter T with
                       different gains, producing asymmetric Un_1, Un_2.

Estimators compared
-------------------
A. Ours (closed-form, this paper).
   PID atoms via the conditional-copy decomposition of
   `gaussian_pid.gaussian_two_source_pid`. For jointly Gaussian
   2-source systems the population values coincide numerically with
   the BROJA decomposition of Bertschinger et al. (2014).

B. Barrett MMI [Barrett, 2015]:
       Red_MMI = min(I(S_1; T), I(S_2; T))
       Un_i^MMI = I(S_i; T) - Red_MMI
       Syn_MMI  = I(S_1, S_2; T) - I(S_1; T) - I(S_2; T) + Red_MMI

E. Venkatesh & Schamberg (2022 ISIT) delta-PID.
   Implementation: `gpid.estimate.approx_pid_from_cov`, the original
   author release.

G. Venkatesh et al. (2023 NeurIPS) tilde G-PID.
   Implementation: `gpid.tilde_pid.exact_gauss_tilde_pid`, the
   original author release at github.com/praveenv253/gpid.

Empirically, for jointly Gaussian scalar inputs MMI, delta-PID and
tilde G-PID coincide numerically across all five configurations
(see Section A.X of the paper). Kay-Ince I_dep and Kay I_ig
(github.com/JWKay/PID) are only released as R code and are left to
future work; they are NOT included in this script.

Outputs
-------
- `experiment5_estimates.csv`     long-format trial-level estimates
- `experiment5_summary.csv`       per-(config, method, atom) mean / std / pop
- `table_n2_sanity.tex`           LaTeX table (multirow, booktabs, three configs)
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import gaussian_pid as gp


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_M = 1000
DEFAULT_N_TRIALS = 50
DEFAULT_SEED = 20260503

ESTIMATES_CSV = "experiment5_estimates.csv"
SUMMARY_CSV = "experiment5_summary.csv"
TABLE_PATH = "table_n2_sanity.tex"


# ---------------------------------------------------------------------------
# Three benchmark configurations (Gaussian, scalar T, scalar S_1, S_2)
# ---------------------------------------------------------------------------
# Variable order in every Sigma is [T, S_1, S_2]:
#     T   -> column 0
#     S_1 -> column 1
#     S_2 -> column 2

@dataclass(frozen=True)
class Config:
    name: str          # short key
    label: str         # LaTeX-ready human label
    Sigma: np.ndarray  # 3x3 population covariance


def _config_pure_redundancy() -> Config:
    """T ~ N(0, 1);  S_1 = T + eps_1;  S_2 = T + eps_2;  eps_i indep N(0, 1).

    Each source is an independent noisy observation of T -- the canonical
    redundancy benchmark in the PID literature.
    """
    sigma_T = 1.0
    sigma_eps = 1.0
    var_T = sigma_T ** 2
    var_eps = sigma_eps ** 2
    var_S = var_T + var_eps                          # = 2
    cov_T_S = var_T                                  # = 1
    cov_S1_S2 = var_T                                # eps_i independent
    Sigma = np.array([
        [var_T,    cov_T_S,   cov_T_S],
        [cov_T_S,  var_S,     cov_S1_S2],
        [cov_T_S,  cov_S1_S2, var_S],
    ])
    return Config(name="pure_red",
                  label=r"Pure redundancy",
                  Sigma=Sigma)


def _config_pure_unique() -> Config:
    """T = S_1 + eps_T;  S_2 i.i.d. N(0,1) independent of (T, S_1)."""
    sigma_S1 = 1.0
    sigma_eps_T = 1.0
    sigma_S2 = 1.0
    var_T = sigma_S1 ** 2 + sigma_eps_T ** 2
    Sigma = np.array([
        [var_T,        sigma_S1 ** 2, 0.0],
        [sigma_S1 ** 2, sigma_S1 ** 2, 0.0],
        [0.0,          0.0,          sigma_S2 ** 2],
    ])
    return Config(name="pure_un",
                  label=r"Pure unique ($S_2 \perp T$)",
                  Sigma=Sigma)


def _config_pure_synergy() -> Config:
    """T = S_1 + S_2 + eps_T;  S_1, S_2 ~ N(0,1) independent."""
    sigma_S1 = 1.0
    sigma_S2 = 1.0
    sigma_eps_T = 1.0
    var_T = sigma_S1 ** 2 + sigma_S2 ** 2 + sigma_eps_T ** 2
    Sigma = np.array([
        [var_T,        sigma_S1 ** 2, sigma_S2 ** 2],
        [sigma_S1 ** 2, sigma_S1 ** 2, 0.0],
        [sigma_S2 ** 2, 0.0,           sigma_S2 ** 2],
    ])
    return Config(name="pure_syn",
                  label=r"Pure synergy (Gaussian XOR)",
                  Sigma=Sigma)


def _config_mixed_correlated() -> Config:
    """T = S_1 + S_2 + eps_T with Cov(S_1, S_2) = 0.3; eps_T indep."""
    sigma_eps_T = 1.0
    rho = 0.3
    Sigma_S = np.array([[1.0, rho], [rho, 1.0]])
    var_T = (Sigma_S[0, 0] + Sigma_S[1, 1]
             + 2.0 * Sigma_S[0, 1] + sigma_eps_T ** 2)
    cov_T_S1 = Sigma_S[0, 0] + Sigma_S[0, 1]   # = 1 + rho
    cov_T_S2 = Sigma_S[1, 1] + Sigma_S[0, 1]   # = 1 + rho
    Sigma = np.array([
        [var_T,    cov_T_S1,    cov_T_S2],
        [cov_T_S1, Sigma_S[0,0], Sigma_S[0,1]],
        [cov_T_S2, Sigma_S[1,0], Sigma_S[1,1]],
    ])
    return Config(name="mixed_corr",
                  label=r"Mixed (correlated $S$, $\rho{=}0.3$)",
                  Sigma=Sigma)


def _config_mixed_asymmetric() -> Config:
    """T = 2*S_1 + S_2 + eps_T; S_1, S_2 ~ N(0,1) independent."""
    a, b = 2.0, 1.0
    sigma_eps_T = 1.0
    var_T = a ** 2 + b ** 2 + sigma_eps_T ** 2     # = 6
    cov_T_S1 = a                                    # = 2
    cov_T_S2 = b                                    # = 1
    Sigma = np.array([
        [var_T,    cov_T_S1, cov_T_S2],
        [cov_T_S1, 1.0,      0.0],
        [cov_T_S2, 0.0,      1.0],
    ])
    return Config(name="mixed_asym",
                  label=r"Mixed (asymmetric, $T{=}2S_1{+}S_2{+}\epsilon$)",
                  Sigma=Sigma)


CONFIGS: List[Config] = [
    _config_pure_redundancy(),
    _config_pure_unique(),
    _config_pure_synergy(),
    _config_mixed_correlated(),
    _config_mixed_asymmetric(),
]


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

ATOMS = ["Red", "Un_1", "Un_2", "Syn"]


def estimator_ours(Sigma: np.ndarray) -> Dict[str, float]:
    """Closed-form Gaussian two-source PID via the conditional-copy
    decomposition (this paper's contribution).
    """
    pid = gp.gaussian_two_source_pid(
        Sigma, target=[0], source1=[1], source2=[2],
    )
    return {
        "Red":  float(pid.redundancy),
        "Un_1": float(pid.unique_s1),
        "Un_2": float(pid.unique_s2),
        "Syn":  float(pid.synergy),
    }


def estimator_mmi(Sigma: np.ndarray) -> Dict[str, float]:
    """Barrett (2015) Minimum Mutual Information PID baseline."""
    I_T_S1 = gp.gaussian_mi(Sigma, x=[0], y=[1])
    I_T_S2 = gp.gaussian_mi(Sigma, x=[0], y=[2])
    I_T_joint = gp.gaussian_mi(Sigma, x=[0], y=[1, 2])
    Red = float(min(I_T_S1, I_T_S2))
    Un_1 = float(I_T_S1 - Red)
    Un_2 = float(I_T_S2 - Red)
    Syn = float(I_T_joint - I_T_S1 - I_T_S2 + Red)
    return {"Red": Red, "Un_1": Un_1, "Un_2": Un_2, "Syn": Syn}


# `gpid` (https://github.com/praveenv253/gpid) implements both
# delta-PID (Venkatesh & Schamberg 2022, ISIT) and ~G-PID (Venkatesh
# et al. 2023, NeurIPS) for multivariate Gaussians. Its return order
# is (I(M;X), I(M;Y), I(M;X,Y), aux1, aux2, Un_X, Un_Y, Red, Syn) in
# BITS; we convert to nats by multiplying by ln(2).
_BITS_TO_NATS = float(np.log(2.0))


def _gpid_unpack(ret) -> Dict[str, float]:
    """Convert a (imx, imy, imxy, *aux*, uix, uiy, ri, si) return tuple
    in bits to a {Red, Un_1, Un_2, Syn} dict in nats.
    """
    uix, uiy, ri, si = ret[-4], ret[-3], ret[-2], ret[-1]
    return {
        "Red":  float(ri)  * _BITS_TO_NATS,
        "Un_1": float(uix) * _BITS_TO_NATS,
        "Un_2": float(uiy) * _BITS_TO_NATS,
        "Syn":  float(si)  * _BITS_TO_NATS,
    }


def estimator_delta_pid(Sigma: np.ndarray) -> Dict[str, float]:
    """Venkatesh & Schamberg (2022 ISIT) deficiency-based PID.

    Implementation: `gpid.estimate.approx_pid_from_cov` -- the same
    code released by the authors at github.com/gabeschamberg/mvar-
    gauss-pid (and re-distributed inside the `gpid` package). The
    estimator solves a small CVXPY program; `verbose=False` suppresses
    the iteration log.
    """
    from gpid.estimate import approx_pid_from_cov
    ret = approx_pid_from_cov(Sigma, dm=1, dx=1, dy=1, verbose=False)
    return _gpid_unpack(ret)


def estimator_g_pid(Sigma: np.ndarray) -> Dict[str, float]:
    """Venkatesh et al. (2023 NeurIPS) ~G-PID.

    Implementation: `gpid.tilde_pid.exact_gauss_tilde_pid` from the
    authors' release at github.com/praveenv253/gpid. Internally a
    projected-gradient (RProp) scheme on a constrained convex program.
    """
    from gpid.tilde_pid import exact_gauss_tilde_pid
    ret = exact_gauss_tilde_pid(Sigma, dm=1, dx=1, dy=1, verbose=False)
    return _gpid_unpack(ret)


@dataclass(frozen=True)
class Estimator:
    key: str
    label: str                                          # LaTeX-ready
    fn: Callable[[np.ndarray], Dict[str, float]]


ESTIMATORS: List[Estimator] = [
    Estimator(key="ours",
              label=r"Ours / BROJA",
              fn=estimator_ours),
    Estimator(key="mmi",
              label=r"Barrett MMI",
              fn=estimator_mmi),
    Estimator(key="delta_pid",
              label=r"Venkatesh--Schamberg $\delta$-PID",
              fn=estimator_delta_pid),
    Estimator(key="g_pid",
              label=r"Venkatesh $\widetilde G$-PID",
              fn=estimator_g_pid),
]


# ---------------------------------------------------------------------------
# Sampling + plug-in pipeline
# ---------------------------------------------------------------------------

def _empirical_cov(samples: np.ndarray) -> np.ndarray:
    Sigma_hat = np.cov(samples, rowvar=False, bias=False)
    return (Sigma_hat + Sigma_hat.T) / 2.0


def _sample_from(Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from N(0, Sigma) via Cholesky."""
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, Sigma.shape[0]))
    return Z @ L.T


def run_trials(
    M: int = DEFAULT_M,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> List[Dict]:
    """For every (config, estimator, trial) cell, draw fresh samples
    and record the four PID atoms.

    Same-trial RNG state is shared across estimators within a config so
    every method sees the IDENTICAL Sigma_hat -- any disagreement is
    purely a function of estimator definition, not sampling noise.
    """
    parent = np.random.default_rng(seed)
    streams = list(parent.spawn(n_trials))

    records: List[Dict] = []
    for cfg in CONFIGS:
        if verbose:
            print(f"[experiment5] config = {cfg.name}")
        for trial_idx, rng_t in enumerate(streams):
            samples = _sample_from(cfg.Sigma, M, rng_t)
            Sigma_hat = _empirical_cov(samples)
            for est in ESTIMATORS:
                try:
                    atoms = est.fn(Sigma_hat)
                except np.linalg.LinAlgError:
                    atoms = {a: float("nan") for a in ATOMS}
                for atom_name, val in atoms.items():
                    records.append({
                        "config": cfg.name,
                        "method": est.key,
                        "trial": trial_idx,
                        "atom": atom_name,
                        "value": float(val),
                    })
        if verbose:
            print(f"  done {n_trials} trials on {cfg.name}")
    return records


# ---------------------------------------------------------------------------
# Population (closed-form) reference values
# ---------------------------------------------------------------------------

def population_atoms() -> Dict[Tuple[str, str], Dict[str, float]]:
    """For every (config, estimator) compute the closed-form atoms on
    the population covariance (no sampling).
    """
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for cfg in CONFIGS:
        for est in ESTIMATORS:
            out[(cfg.name, est.key)] = est.fn(cfg.Sigma)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_summary(
    records: List[Dict],
    pop: Dict[Tuple[str, str], Dict[str, float]],
) -> List[Dict]:
    """Group raw trial records by (config, method, atom) and compute
    mean, std, n_trials, plus the analytic population value.
    """
    by_cell: Dict[Tuple[str, str, str], List[float]] = {}
    for r in records:
        key = (r["config"], r["method"], r["atom"])
        by_cell.setdefault(key, []).append(r["value"])
    out: List[Dict] = []
    for (cfg_name, method, atom), vals in sorted(by_cell.items()):
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        n_ok = int(finite.size)
        out.append({
            "config": cfg_name,
            "method": method,
            "atom":   atom,
            "n_trials": n_ok,
            "mean": float(finite.mean()) if n_ok else float("nan"),
            "std":  float(finite.std(ddof=1)) if n_ok > 1 else 0.0,
            "population": float(pop[(cfg_name, method)][atom]),
        })
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_estimates_csv(records: List[Dict], path: str) -> None:
    fields = ["config", "method", "trial", "atom", "value"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({**r, "value": f"{r['value']:.10g}"})
    print(f"  wrote {len(records)} rows -> {path}")


def save_summary_csv(summary: List[Dict], path: str) -> None:
    fields = ["config", "method", "atom", "n_trials",
              "mean", "std", "population"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow({
                "config": r["config"], "method": r["method"], "atom": r["atom"],
                "n_trials": r["n_trials"],
                "mean":  f"{r['mean']:.6f}",
                "std":   f"{r['std']:.6f}",
                "population": f"{r['population']:.6f}",
            })
    print(f"  wrote {len(summary)} rows -> {path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _fmt_cell(mean: float, std: float, pop: float, decimals: int = 2,
              std_eps: float = 5e-3) -> str:
    """Format one cell as ``$mean{\\pm}std\\,(pop)$`` in LaTeX.

    When the standard deviation is below `std_eps` (i.e. would round
    to 0.00 at the chosen precision), the ``$\\pm$`` term is omitted
    and the cell shrinks to ``$mean\\,(pop)$``, matching the paper's
    typographic convention for noise-free entries. Mean and population
    values below half the display tolerance are snapped to +0 so the
    formatter never emits "-0.00".
    """
    half_tol = 0.5 * (10 ** (-decimals))
    if abs(mean) < half_tol:
        mean = 0.0
    if abs(pop) < half_tol:
        pop = 0.0
    fmt = f"{{:.{decimals}f}}"
    if std < std_eps:
        return f"${fmt.format(mean)}\\,({fmt.format(pop)})$"
    return (f"${fmt.format(mean)}{{\\pm}}{fmt.format(std)}\\,"
            f"({fmt.format(pop)})$")


# Display label for a group of estimators with numerically identical
# atoms within one configuration. Looked up by the FROZENSET of
# estimator `key` strings present in the group.
_GROUP_LABEL: Dict[frozenset, str] = {
    frozenset({"ours"}):                       r"Ours / BROJA",
    frozenset({"mmi"}):                        r"Barrett MMI",
    frozenset({"delta_pid"}):                  r"Venkatesh--Schamberg $\delta$-PID",
    frozenset({"g_pid"}):                      r"Venkatesh $\widetilde G$-PID",
    frozenset({"mmi", "delta_pid"}):           r"MMI / $\delta$-PID",
    frozenset({"mmi", "g_pid"}):               r"MMI / $\widetilde G$-PID",
    frozenset({"delta_pid", "g_pid"}):         r"$\delta$-PID / $\widetilde G$-PID",
    frozenset({"mmi", "delta_pid", "g_pid"}):  r"MMI / $\delta$-PID / $\widetilde G$-PID",
    frozenset({"ours", "mmi", "delta_pid", "g_pid"}): r"All four",
}


def _group_label(keys: List[str]) -> str:
    """Render the displayed label for a group of methods that gave
    numerically identical PID atoms within one configuration."""
    fs = frozenset(keys)
    if fs in _GROUP_LABEL:
        return _GROUP_LABEL[fs]
    # Fall back: join the per-method labels with " / ".
    by_key = {e.key: e.label for e in ESTIMATORS}
    return " / ".join(by_key[k] for k in keys)


def _merge_methods(
    summary: List[Dict],
    cfg_name: str,
    merge_tol: float = 5e-3,
) -> List[Tuple[List[str], Dict[str, Dict]]]:
    """Within one config, group methods whose (mean, std, population)
    match within `merge_tol` on every PID atom. Returns a list of
    (method_keys, atom_to_record) pairs in the registry order of
    ESTIMATORS so the table rows stay deterministic.
    """
    by_method: Dict[str, Dict[str, Dict]] = {}
    for r in summary:
        if r["config"] != cfg_name:
            continue
        by_method.setdefault(r["method"], {})[r["atom"]] = r

    groups: List[Tuple[List[str], Dict[str, Dict]]] = []
    for est in ESTIMATORS:
        rec = by_method.get(est.key)
        if rec is None:
            continue
        placed = False
        for group_keys, group_rec in groups:
            if all(
                abs(rec[a]["mean"] - group_rec[a]["mean"]) < merge_tol
                and abs(rec[a]["std"] - group_rec[a]["std"]) < merge_tol
                and abs(rec[a]["population"] - group_rec[a]["population"]) < merge_tol
                for a in ATOMS
            ):
                group_keys.append(est.key)
                placed = True
                break
        if not placed:
            groups.append(([est.key], rec))
    return groups


def save_latex_table(summary: List[Dict], path: str) -> None:
    """LaTeX table; rows are (config, method-group) pairs, columns are
    the four PID atoms (Red, Un_1, Un_2, Syn). Each cell shows the
    plug-in mean$\\pm$std and, in parentheses, the closed-form
    population value. Estimators whose four atoms agree within
    `merge_tol` are collapsed into a single row labelled with the
    join of their names (e.g. "MMI / $\\delta$-PID / $\\widetilde G$-PID");
    when all four estimators agree the row is labelled "All four".
    """
    lines: List[str] = []
    lines.append(r"% Auto-generated by experiment5_run.py")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Two-source Gaussian PID on five controlled "
                 r"configurations. Cells show plug-in mean~$\pm$~standard "
                 r"deviation over $50$ trials at $M = 1000$ samples; "
                 r"analytical population values are in parentheses. "
                 r"\emph{Ours / BROJA}: our population values numerically "
                 r"coincide with the BROJA decomposition across all five "
                 r"configurations to the displayed precision. "
                 r"MMI~\cite{barrett2015exploration}, "
                 r"$\delta$-PID~\cite{venkatesh2022partial} and "
                 r"$\widetilde G$-PID~\cite{venkatesh2023gaussian}, "
                 r"computed via the authors' \texttt{gpid} package, also "
                 r"coincide numerically and are collapsed into a single "
                 r"row whenever they do.}")
    lines.append(r"  \label{tab:n2_comparison}")
    lines.append(r"  \footnotesize")
    lines.append(r"  \renewcommand{\arraystretch}{1.15}")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \begin{tabular}"
                 r"{p{0.13\textwidth}p{0.20\textwidth}cccc}")
    lines.append(r"    \hline")
    lines.append(r"    Configuration & Estimator & "
                 r"$\operatorname{Red}$ & $\operatorname{Un}_1$ "
                 r"& $\operatorname{Un}_2$ & $\operatorname{Syn}$ \\")
    lines.append(r"    \hline")
    for cfg in CONFIGS:
        groups = _merge_methods(summary, cfg.name)
        for i, (keys, rec) in enumerate(groups):
            row_cells = [
                _fmt_cell(rec[a]["mean"], rec[a]["std"], rec[a]["population"])
                for a in ATOMS
            ]
            # First row of each config carries the config label; later
            # rows leave the first column empty (no source-level blank
            # line between rows so the .tex stays tidy).
            if i == 0:
                lines.append(f"    {cfg.label}")
            lines.append(f"      & {_group_label(keys)}")
            lines.append("      & " + " & ".join(row_cells) + r" \\")
        lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-M", "--samples", type=int, default=DEFAULT_M,
                   help="samples per trial (default %(default)s)")
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                   help="trials per (config, estimator) cell "
                        "(default %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="base RNG seed (default %(default)s)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="directory for CSV/TEX outputs (default %(default)s)")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    out_dir = args.output_dir
    if out_dir and out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)
    est_path = os.path.join(out_dir, ESTIMATES_CSV)
    sum_path = os.path.join(out_dir, SUMMARY_CSV)
    tex_path = os.path.join(out_dir, TABLE_PATH)

    # Closed-form population values.
    pop = population_atoms()
    print("[experiment5] population values (closed-form):")
    for cfg in CONFIGS:
        print(f"  {cfg.name}:")
        for est in ESTIMATORS:
            row = pop[(cfg.name, est.key)]
            print(f"    {est.key:>5s}: " + ",  ".join(
                f"{a}={row[a]:+.4f}" for a in ATOMS))

    # Trials.
    print(f"[experiment5] running {args.n_trials} trials at M={args.samples}")
    records = run_trials(M=args.samples, n_trials=args.n_trials, seed=args.seed)
    save_estimates_csv(records, est_path)

    summary = aggregate_summary(records, pop)
    save_summary_csv(summary, sum_path)
    save_latex_table(summary, tex_path)


if __name__ == "__main__":
    main()
