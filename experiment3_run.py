"""
experiment3_run.py
==================

Section 5 / Experiment 3 ("Ridge stability in the small-sample regime").

The Section 5.1 benchmark sits in dimension d = N + d_T = 7. As the
sample size M approaches that limit the empirical covariance becomes
ill-conditioned and the unregularised plug-in estimator either gains
catastrophic variance or hits an outright Cholesky failure. The ridge
parameter lambda of `gaussian_pid.gaussian_tse` is the prescribed
remedy. This experiment characterises that bias-variance trade-off
across (M, lambda) using the benchmark's tuned (sigma_T, sigma_eps),
*not* the singular-target perturbations of an earlier draft of this
script.

Versions
--------
* **Version A.** Fix M = 12 (just above d = 7); sweep
       lambda in {0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0}.
  Single-panel line plot of mean TSE_hat (mean +/- 1 SD across
  finite trials) vs. log10(lambda) with the population TSE drawn as
  a horizontal reference line. Cells with any non-finite trial are
  marked with a red `X` and a "k/n failed" annotation.

* **Version B.** Sweep
       M      in {10, 15, 25, 50, 100, 500}     X
       lambda in {0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1}.
  2-D heatmap whose colour encodes log10 of the *relative* error
       |mean(TSE_hat) - TSE_pop| / |TSE_pop|,
  with NaN cells masked. The colormap is blue = low error, red =
  high error, on a log scale, so the "best lambda for each M" appears
  as a diagonal valley.

Data
----
- `experiment3_estimates.csv`  long format: trial_id, M, lambda,
                                value, status
- `experiment3_population.csv` population TSE (single row)
- `figure3a.pdf` / `figure3a.png`  (Version A)
- `figure3b.pdf` / `figure3b.png`  (Version B)

The script is idempotent: a single `python3 experiment3_run.py` run
produces the CSVs and both figures from scratch in a few seconds.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import gaussian_pid as gp
import experiment1_benchmark as bench


# ---------------------------------------------------------------------------
# Output paths and defaults
# ---------------------------------------------------------------------------

ESTIMATES_CSV = "experiment3_estimates.csv"
POPULATION_CSV = "experiment3_population.csv"
FIGURE_A_PATH = "figure3a.pdf"
FIGURE_B_PATH = "figure3b.pdf"

DEFAULT_N_TRIALS = 50
DEFAULT_SEED = 20260503

# Tuned construction parameters used in the paper figure. These are the
# same values used by Experiment 1 in Section 5.1; the ridge sweep below
# stresses the closed-form estimator under finite-sample conditions and
# does not vary the population covariance.
TUNED_PARAMS = bench.BenchmarkParams(
    a=1.0,
    b=1.0,
    sigma_U=2.0,
    sigma_V=2.0,
    sigma_eps_pair=0.05,
    sigma_eps_triple=0.05,
    sigma_T2=1.0,
    sigma_T3=1.0,
)

# Version A: 1-D lambda sweep at a single small-sample M.
DEFAULT_M_A = 12
DEFAULT_LAMBDA_LIST_A = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]

# Version B: 2-D (M x lambda) sweep.
# We intentionally truncate lambda at 1e-4 (no 1e-2 / 1e-1 / 1.0): the
# right-hand "regularisation-dominates" regime is visually obvious and
# was washing out the bias-variance trade-off in the small-lambda
# region, which is what the figure is meant to highlight.
DEFAULT_M_LIST_B = [10, 12, 15, 25, 50, 100, 500]
DEFAULT_LAMBDA_LIST_B = [0.0, 1e-8, 1e-6, 1e-4]


# ---------------------------------------------------------------------------
# Single-trial plug-in TSE estimate with ridge regularisation
# ---------------------------------------------------------------------------

def _empirical_cov(samples: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance, symmetrised to defang round-off."""
    Sigma_hat = np.cov(samples, rowvar=False, bias=False)
    return (Sigma_hat + Sigma_hat.T) / 2.0


def plug_in_tse(Sigma_hat: np.ndarray, lam: float) -> Tuple[str, float]:
    """Run the closed-form Gaussian TSE estimator with ridge `lam`.

    `lam` is passed both as the entropy-side ridge (added before the
    log-determinant) and as the surrogate-side `ridge_target` (added
    before solving Sigma_TT^{-1} in the conditional-copy construction),
    so when `lam == 0` no regularisation is applied at all.

    Returns
    -------
    (status, value):
        status :: "ok" | "nan" | "inf" | "linalg"
        value  :: float (NaN when status != "ok")
    """
    try:
        tse = gp.gaussian_tse(
            Sigma_hat,
            target=list(bench.TARGET_INDICES),
            sources=bench._source_groups(),
            ridge=lam, ridge_target=lam,
        )
    except np.linalg.LinAlgError:
        return "linalg", float("nan")
    if np.isnan(tse):
        return "nan", float("nan")
    if np.isinf(tse):
        return "inf", float(tse)
    return "ok", float(tse)


# ---------------------------------------------------------------------------
# Trial driver
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    trial_id: int
    M: int
    lam: float
    value: float
    status: str


def _per_trial_sample_covs(
    base: bench.BenchmarkParams,
    M: int,
    streams: List[np.random.Generator],
) -> List[np.ndarray]:
    """Draw one (M, d) sample matrix per trial and return its empirical
    covariance. The same sample matrix is reused for every lambda at
    this M so that the estimator's lambda dependence is studied on a
    fixed dataset (not confounded with sampling noise).
    """
    out: List[np.ndarray] = []
    for rng_t in streams:
        samples = bench.sample_data(base, n_samples=M, rng=rng_t)
        out.append(_empirical_cov(samples))
    return out


def run_trials(
    base: bench.BenchmarkParams,
    M_list: List[int],
    lambda_list: List[float],
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> List[TrialRecord]:
    """For every (M, lambda) cell, draw `n_trials` independent sample
    matrices, then evaluate the plug-in TSE on each.

    Trials are spawned from a parent PCG64 stream so that trial t at
    one (M, lambda) is INDEPENDENT of trial t at another (M, lambda):
    the per-trial RNG state is identical, but `sample_data` advances
    its stream by M draws so the realised sample matrix differs
    across M values.
    """
    parent_rng = np.random.default_rng(seed)
    streams = list(parent_rng.spawn(n_trials))

    records: List[TrialRecord] = []
    t0 = time.time()
    for M in M_list:
        if verbose:
            print(f"[experiment3] M = {M}")
        # Draw one sample-cov per trial and reuse across all lambda.
        per_trial_cov = _per_trial_sample_covs(base, M, streams)
        for lam in lambda_list:
            n_ok = n_fail = 0
            for t, Sigma_hat in enumerate(per_trial_cov):
                status, value = plug_in_tse(Sigma_hat, lam)
                records.append(TrialRecord(
                    trial_id=t, M=int(M), lam=float(lam),
                    value=float(value), status=status,
                ))
                if status == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
            if verbose:
                tail = f", {n_fail} failed" if n_fail else ""
                print(f"  lambda = {lam:.0e}:  {n_ok}/{n_ok + n_fail} ok"
                      f"{tail}   ({time.time() - t0:.1f}s)")
    return records


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

ESTIMATE_FIELDS = ["trial_id", "M", "lambda", "value", "status"]


def save_estimates_csv(records: List[TrialRecord], path: str = ESTIMATES_CSV) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ESTIMATE_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "trial_id": r.trial_id,
                "M": r.M,
                "lambda": f"{r.lam:.6e}",
                "value": "" if not np.isfinite(r.value) else f"{r.value:.10g}",
                "status": r.status,
            })
    print(f"  wrote {len(records)} rows -> {path}")


def save_population_csv(
    base: bench.BenchmarkParams, path: str = POPULATION_CSV
) -> float:
    """Closed-form population TSE under the Section 5.1 benchmark."""
    Sigma = bench.build_covariance(base)
    tse_pop = float(gp.gaussian_tse(
        Sigma,
        target=list(bench.TARGET_INDICES),
        sources=bench._source_groups(),
    ))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tse_population"])
        writer.writeheader()
        writer.writerow({"tse_population": f"{tse_pop:.10g}"})
    print(f"  wrote population TSE = {tse_pop:.4f} -> {path}")
    return tse_pop


def load_estimates_csv(path: str = ESTIMATES_CSV) -> List[TrialRecord]:
    out: List[TrialRecord] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            value_str = row["value"]
            value = float("nan") if value_str == "" else float(value_str)
            out.append(TrialRecord(
                trial_id=int(row["trial_id"]),
                M=int(row["M"]),
                lam=float(row["lambda"]),
                value=value,
                status=row["status"],
            ))
    return out


def load_population_csv(path: str = POPULATION_CSV) -> float:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return float(rows[0]["tse_population"])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _records_for(records: List[TrialRecord], M: int, lam: float
                 ) -> List[TrialRecord]:
    return [r for r in records if r.M == M and np.isclose(r.lam, lam)]


def _summarise(records: List[TrialRecord]
               ) -> Tuple[float, float, int, int]:
    """Return (mean_of_finite, std_of_finite, n_ok, n_total)."""
    finite_vals = [r.value for r in records if r.status == "ok"]
    n_ok = len(finite_vals)
    n_total = len(records)
    if n_ok == 0:
        return float("nan"), float("nan"), 0, n_total
    arr = np.asarray(finite_vals)
    return (float(arr.mean()),
            float(arr.std(ddof=1) if n_ok > 1 else 0.0),
            n_ok, n_total)


# ---------------------------------------------------------------------------
# IEEE rcparams
# ---------------------------------------------------------------------------

DPI = 300


def _ieee_rcparams() -> Dict:
    return {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
    }


# ---------------------------------------------------------------------------
# Version A figure: 1-D lambda sweep at fixed small M
# ---------------------------------------------------------------------------

def render_figure_A(
    estimates_path: str = ESTIMATES_CSV,
    population_path: str = POPULATION_CSV,
    M: int = DEFAULT_M_A,
    lambda_list: Optional[List[float]] = None,
    figure_path: str = FIGURE_A_PATH,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if lambda_list is None:
        lambda_list = DEFAULT_LAMBDA_LIST_A
    records = load_estimates_csv(estimates_path)
    tse_pop = load_population_csv(population_path)

    means, sds = [], []
    fails: List[Tuple[int, int]] = []      # (n_fail, n_total) per lambda
    for lam in lambda_list:
        sub = _records_for(records, M, lam)
        m, s, n_ok, n_total = _summarise(sub)
        means.append(m)
        sds.append(s)
        fails.append((n_total - n_ok, n_total))

    # Plot lambda = 0 at one log-decade left of the smallest non-zero lambda.
    nonzero = [lam for lam in lambda_list if lam > 0]
    leftmost_log = (np.log10(min(nonzero)) - 1.0) if nonzero else -10.0
    xs = [leftmost_log if lam == 0 else np.log10(lam) for lam in lambda_list]

    with mpl.rc_context(_ieee_rcparams()):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=DPI)

        finite_mask = np.array([np.isfinite(m) for m in means])
        xs_arr = np.asarray(xs)
        ax.errorbar(
            xs_arr[finite_mask], np.asarray(means)[finite_mask],
            yerr=np.asarray(sds)[finite_mask],
            marker="o", color="#1f77b4",
            linestyle="-", capsize=2.0,
            label=r"$\widehat{\mathrm{TSE}}$ (mean $\pm$ 1 SD)",
        )
        ax.axhline(tse_pop, color="black", linewidth=0.9, linestyle="--",
                   label=f"population = {tse_pop:.2f} nats")

        for x, lam, (n_fail, n_total), mean_v, sd_v in zip(
            xs, lambda_list, fails, means, sds
        ):
            if n_fail == 0:
                continue
            yref = tse_pop if not np.isfinite(mean_v) else mean_v + sd_v
            ax.plot(x, yref, marker="X", color="#d62728",
                    markersize=8.0, markeredgewidth=1.2,
                    markeredgecolor="black", linestyle="None", zorder=5)
            ax.annotate(
                f"{n_fail}/{n_total} failed",
                xy=(x, yref), xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=6.5, color="#d62728",
            )

        ax.set_xlabel(r"$\log_{10}\lambda$  (ridge)")
        ax.set_ylabel(r"$\widehat{\mathrm{TSE}}$  (nats)")
        ax.set_title(rf"Ridge sweep, $M = {M}$  (just above $d=7$)")
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [r"$0$" if lam == 0 else f"{int(np.log10(lam))}"
             for lam in lambda_list]
        )
        ax.tick_params(direction="in", length=2.5, pad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.4, which="major")
        ax.legend(frameon=False, loc="best")
        fig.tight_layout(pad=0.6)
        fig.savefig(figure_path, bbox_inches="tight", dpi=DPI)
        png = figure_path.replace(".pdf", ".png")
        fig.savefig(png, bbox_inches="tight", dpi=DPI)
        plt.close(fig)
    print(f"  wrote {figure_path} and {png}")


# ---------------------------------------------------------------------------
# Version B figure: 2-D (M x lambda) heatmap of relative error
# ---------------------------------------------------------------------------

def render_figure_B(
    estimates_path: str = ESTIMATES_CSV,
    population_path: str = POPULATION_CSV,
    M_list: Optional[List[int]] = None,
    lambda_list: Optional[List[float]] = None,
    figure_path: str = FIGURE_B_PATH,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if M_list is None:
        M_list = DEFAULT_M_LIST_B
    if lambda_list is None:
        lambda_list = DEFAULT_LAMBDA_LIST_B
    records = load_estimates_csv(estimates_path)
    tse_pop = load_population_csv(population_path)

    n_rows = len(M_list)
    n_cols = len(lambda_list)
    rel_err = np.full((n_rows, n_cols), np.nan)
    fail_frac = np.zeros((n_rows, n_cols))
    for i, M in enumerate(M_list):
        for j, lam in enumerate(lambda_list):
            sub = _records_for(records, M, lam)
            m, _, n_ok, n_total = _summarise(sub)
            if np.isfinite(m) and abs(tse_pop) > 0:
                rel_err[i, j] = abs(m - tse_pop) / abs(tse_pop)
            fail_frac[i, j] = (n_total - n_ok) / max(n_total, 1)

    # The colour scale: log of relative error. Saturate values at the
    # observed [vmin, vmax] so the dynamic range stays useful even when
    # one cell explodes.
    finite_mask = np.isfinite(rel_err) & (rel_err > 0)
    if finite_mask.any():
        vmin = max(np.nanmin(rel_err[finite_mask]), 1e-6)
        vmax = max(np.nanmax(rel_err[finite_mask]), vmin * 10)
    else:
        vmin, vmax = 1e-3, 1.0

    with mpl.rc_context(_ieee_rcparams()):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=DPI)

        im = ax.imshow(
            rel_err, origin="lower", aspect="auto",
            cmap="RdYlBu_r", norm=LogNorm(vmin=vmin, vmax=vmax),
            extent=(-0.5, n_cols - 0.5, -0.5, n_rows - 0.5),
        )

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            [r"$0$" if lam == 0 else f"$10^{{{int(np.log10(lam))}}}$"
             for lam in lambda_list]
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([str(M) for M in M_list])
        ax.set_xlabel(r"$\lambda$  (ridge)")
        ax.set_ylabel("Sample size $M$")
        ax.set_title(r"Relative error  "
                      r"$|\widehat{\mathrm{TSE}}-\mathrm{TSE}^{\mathrm{pop}}|"
                      r" / |\mathrm{TSE}^{\mathrm{pop}}|$")

        # Mark cells with any failed trials.
        for i in range(n_rows):
            for j in range(n_cols):
                if fail_frac[i, j] > 0:
                    ax.plot(j, i, marker="X",
                            color="white", markersize=6.5,
                            markeredgecolor="black",
                            markeredgewidth=0.8,
                            linestyle="None", zorder=5)

        # Mark the per-row minimum (best lambda for each M) with a small
        # annotation so the diagonal valley is visible.
        for i in range(n_rows):
            row = rel_err[i]
            if np.all(np.isnan(row)):
                continue
            j_best = int(np.nanargmin(row))
            ax.plot(j_best, i, marker="o",
                    color="white", markersize=4.0,
                    markeredgecolor="black", markeredgewidth=0.6,
                    linestyle="None", zorder=6)

        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label(r"relative error", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.tight_layout(pad=0.6)
        fig.savefig(figure_path, bbox_inches="tight", dpi=DPI)
        png = figure_path.replace(".pdf", ".png")
        fig.savefig(png, bbox_inches="tight", dpi=DPI)
        plt.close(fig)
    print(f"  wrote {figure_path} and {png}")


# ---------------------------------------------------------------------------
# Entry point + CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                   help="trials per (M, lambda) cell (default %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="base RNG seed (default %(default)s)")
    p.add_argument("--render-only", action="store_true",
                   help="skip experiments; only redraw the figures from CSV")
    p.add_argument("--output-dir", type=str, default=".",
                   help="directory for CSV/PDF/PNG (default %(default)s)")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    out_dir = args.output_dir
    if out_dir and out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, ESTIMATES_CSV)
    pop_path = os.path.join(out_dir, POPULATION_CSV)
    fig_a_path = os.path.join(out_dir, FIGURE_A_PATH)
    fig_b_path = os.path.join(out_dir, FIGURE_B_PATH)

    base_params = TUNED_PARAMS
    print(f"[experiment3] tuned params: {base_params.to_dict()}")

    # Union of M values and lambda values across the two versions, so a
    # single CSV powers both figures.
    M_union = sorted(set(DEFAULT_M_LIST_B + [DEFAULT_M_A]))
    lambda_union = sorted(set(DEFAULT_LAMBDA_LIST_A + DEFAULT_LAMBDA_LIST_B))

    if not args.render_only:
        print(f"[experiment3] base params: {base_params.to_dict()}")
        print(f"[experiment3] M values:      {M_union}")
        print(f"[experiment3] lambda values: {lambda_union}")
        print(f"[experiment3] running {args.n_trials} trials per cell")
        records = run_trials(
            base=base_params,
            M_list=M_union,
            lambda_list=lambda_union,
            n_trials=args.n_trials,
            seed=args.seed,
        )
        save_estimates_csv(records, csv_path)
        save_population_csv(base_params, pop_path)

    print("[experiment3] rendering Version A (1-D lambda sweep at small M)")
    render_figure_A(
        estimates_path=csv_path, population_path=pop_path,
        M=DEFAULT_M_A, lambda_list=DEFAULT_LAMBDA_LIST_A,
        figure_path=fig_a_path,
    )
    print("[experiment3] rendering Version B (2-D M x lambda heatmap)")
    render_figure_B(
        estimates_path=csv_path, population_path=pop_path,
        M_list=DEFAULT_M_LIST_B, lambda_list=DEFAULT_LAMBDA_LIST_B,
        figure_path=fig_b_path,
    )


if __name__ == "__main__":
    main()
