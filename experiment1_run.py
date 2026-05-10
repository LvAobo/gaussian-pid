"""
experiment1_run.py
==================

Section 5 / Experiment 1 ("Order-spectrum recovery and subset
localisation"): plug-in estimation pipeline and 3-panel figure
generator.

Pipeline
--------
1. Use the tuned construction parameters in `TUNED_PARAMS` below.
2. Compute the population (closed-form) hierarchy quantities under the
   benchmark distribution; these are the horizontal-line overlays in
   the figure.
3. Run `N_TRIALS` independent Monte Carlo trials. In each trial:
     a. Draw `M_SAMPLES` samples from the benchmark distribution.
     b. Form the unbiased sample covariance matrix.
     c. Plug the sample covariance into the closed-form Gaussian
        estimators of `gaussian_pid.py` to obtain estimates of
        SE_2..SE_5, the 10 pair narrow synergies, and the 10 triple
        narrow synergies.
4. Persist the trial-level estimates to a long-format CSV
   (`experiment1_estimates.csv`) so that plotting can be re-run without
   re-sampling, and so that individual rows can be removed without
   re-running anything else.
5. Render the 3-panel Figure 1 (`figure1.pdf`).

CSV schema (long format)
-------------------------
trial_id : int
kind     : one of {"SE_K", "pair_syn", "triple_syn"}
subset   : human-readable subset label (e.g. "2", "(1,2)", "(3,4,5)")
value    : plug-in estimate, in nats

Population values are saved separately to `experiment1_population.csv`
with columns (kind, subset, value).

The script is invoked as `python3 experiment1_run.py`.
"""

from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import gaussian_pid as gp
import experiment1_benchmark as bench


# ---------------------------------------------------------------------------
# Tuned construction parameters used in the paper figure
# ---------------------------------------------------------------------------
# These were obtained by an offline grid search over (a, b, sigma_U,
# sigma_V, sigma_eps_pair, sigma_eps_triple) maximising a clean SE_2 +
# SE_3 spike pattern with strong pair/triple subset localisation. The
# resulting closed-form (population) hierarchy is reported in Section 5.1
# of the paper.
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


# ---------------------------------------------------------------------------
# Experiment configuration (kept at module level for transparency)
# ---------------------------------------------------------------------------

ESTIMATES_CSV = "experiment1_estimates.csv"
POPULATION_CSV = "experiment1_population.csv"
FIGURE_PATH = "figure1.pdf"

N_TRIALS = 50
M_SAMPLES = 1000
RNG_SEED = 20260503  # YYYYMMDD of the ITW deadline week


# ---------------------------------------------------------------------------
# Plug-in estimation
# ---------------------------------------------------------------------------

def _empirical_cov(samples: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance, symmetrised to defang round-off."""
    Sigma_hat = np.cov(samples, rowvar=False, bias=False)
    return (Sigma_hat + Sigma_hat.T) / 2.0


def plug_in_estimates(Sigma_hat: np.ndarray) -> Dict[str, Dict]:
    """Compute every quantity needed by the figure from one sample
    covariance matrix.

    Returns a dictionary with keys
        spectrum        : {K -> SE_K_hat}, K=2..5
        pair_synergy    : {(i, j) -> Syn_hat(S_i, S_j)} for all 10 pairs
        triple_synergy  : {(i, j, k) -> Syn_hat(S_i, S_j, S_k)} for all 10 triples
    """
    target = list(bench.TARGET_INDICES)
    sources = bench._source_groups()
    spec_pkg = gp.gaussian_synergy_spectrum(
        Sigma_hat, target=target, sources=sources, return_components=False,
    )
    pair_syn = {
        pair: gp.gaussian_narrow_synergy(
            Sigma_hat, target=target, sources=bench._subset_to_columns(pair),
        )
        for pair in bench._all_pairs()
    }
    triple_syn = {
        triple: gp.gaussian_narrow_synergy(
            Sigma_hat, target=target, sources=bench._subset_to_columns(triple),
        )
        for triple in bench._all_triples()
    }
    return {
        "spectrum": dict(spec_pkg),  # type: ignore[arg-type]
        "pair_synergy": pair_syn,
        "triple_synergy": triple_syn,
    }


# ---------------------------------------------------------------------------
# Trial driver
# ---------------------------------------------------------------------------

def run_trials(
    params: bench.BenchmarkParams,
    n_trials: int = N_TRIALS,
    n_samples: int = M_SAMPLES,
    seed: int = RNG_SEED,
    verbose: bool = True,
) -> List[Dict]:
    """Run `n_trials` independent samples and plug-in estimates.

    A single PCG64 stream is used and split per trial via
    `default_rng(seed).spawn(n_trials)` so that each trial's RNG state is
    deterministic and independent of the trial scheduling.
    """
    parent_rng = np.random.default_rng(seed)
    streams = parent_rng.spawn(n_trials)

    records: List[Dict] = []
    t0 = time.time()
    for t, rng in enumerate(streams):
        samples = bench.sample_data(params, n_samples=n_samples, rng=rng)
        Sigma_hat = _empirical_cov(samples)
        try:
            est = plug_in_estimates(Sigma_hat)
        except np.linalg.LinAlgError as e:
            if verbose:
                print(f"  trial {t}: linalg failure ({e}), skipping")
            continue
        records.append({"trial_id": t, **est})
        if verbose and (t + 1) % 10 == 0:
            print(f"  trial {t + 1}/{n_trials}  ({time.time() - t0:.1f}s)")
    return records


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _flatten_records(records: List[Dict]) -> List[Dict]:
    """Convert list-of-trial-dicts into long-format rows."""
    rows: List[Dict] = []
    for rec in records:
        tid = rec["trial_id"]
        for k, v in rec["spectrum"].items():
            rows.append({"trial_id": tid, "kind": "SE_K",
                         "subset": str(k), "value": float(v)})
        for pair, v in rec["pair_synergy"].items():
            label = f"({pair[0]},{pair[1]})"
            rows.append({"trial_id": tid, "kind": "pair_syn",
                         "subset": label, "value": float(v)})
        for triple, v in rec["triple_synergy"].items():
            label = f"({triple[0]},{triple[1]},{triple[2]})"
            rows.append({"trial_id": tid, "kind": "triple_syn",
                         "subset": label, "value": float(v)})
    return rows


def save_estimates_csv(records: List[Dict], path: str = ESTIMATES_CSV) -> None:
    rows = _flatten_records(records)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "kind", "subset", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def save_population_csv(gt: Dict, path: str = POPULATION_CSV) -> None:
    """Persist the closed-form population values used as horizontal-line
    overlays in the figure.
    """
    rows: List[Dict] = []
    for k, v in gt["spectrum"].items():
        rows.append({"kind": "SE_K", "subset": str(k), "value": float(v)})
    for pair, v in gt["pair_synergy"].items():
        rows.append({"kind": "pair_syn",
                     "subset": f"({pair[0]},{pair[1]})", "value": float(v)})
    for triple, v in gt["triple_synergy"].items():
        rows.append({"kind": "triple_syn",
                     "subset": f"({triple[0]},{triple[1]},{triple[2]})",
                     "value": float(v)})
    rows.append({"kind": "tse", "subset": "all", "value": float(gt["tse"])})
    rows.append({"kind": "tse_telescoping", "subset": "all",
                 "value": float(gt["tse_telescoping"])})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "subset", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def load_estimates_csv(path: str = ESTIMATES_CSV) -> Dict[str, Dict[str, np.ndarray]]:
    """Reload the long-format CSV into a dict of arrays for plotting.

    Returned structure
        {
          "SE_K":       {subset_label -> ndarray of size <= n_trials},
          "pair_syn":   {subset_label -> ndarray},
          "triple_syn": {subset_label -> ndarray},
        }
    """
    by_kind: Dict[str, Dict[str, List[float]]] = {
        "SE_K": {}, "pair_syn": {}, "triple_syn": {},
    }
    with open(path) as f:
        for row in csv.DictReader(f):
            kind, subset = row["kind"], row["subset"]
            by_kind[kind].setdefault(subset, []).append(float(row["value"]))
    return {
        kind: {sub: np.asarray(vals) for sub, vals in d.items()}
        for kind, d in by_kind.items()
    }


def load_population_csv(path: str = POPULATION_CSV) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {
        "SE_K": {}, "pair_syn": {}, "triple_syn": {},
        "tse": {}, "tse_telescoping": {},
    }
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(row["kind"], {})[row["subset"]] = float(row["value"])
    return out


# ---------------------------------------------------------------------------
# Figure 1 rendering
# ---------------------------------------------------------------------------

# IEEE-style sizing. ITW is a 4-page short paper; figures usually span the
# double column (7.16 inches) when they have 3 horizontal panels. Tweak
# `FIGURE_WIDTH_IN` to `3.5` for a single-column variant (3 panels stacked
# vertically).
FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 2.4
DPI = 300


def _ieee_rcparams() -> Dict:
    """Conservative matplotlib rc settings that look like IEEE figures."""
    return {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
    }


def _ordered_pair_labels() -> List[str]:
    return [f"({i},{j})" for i, j in bench._all_pairs()]


def _ordered_triple_labels() -> List[str]:
    return [f"({i},{j},{k})" for i, j, k in bench._all_triples()]


def render_figure1(
    estimates_path: str = ESTIMATES_CSV,
    population_path: str = POPULATION_CSV,
    figure_path: str = FIGURE_PATH,
) -> None:
    """Draw the 3-panel Figure 1 from the persisted CSV files.

    Panel layout
      (a) Bar plot of SE_K vs K (K = 2..5). Mean +/- 1 SD across trials.
          Population value overlaid as a black horizontal tick on each bar.
      (b) Bar plot of all 10 pair narrow synergies, ordered (1,2)...(4,5).
          Pair (1,2) (the structural pair-only signal) is highlighted.
      (c) Bar plot of all 10 triple narrow synergies. Triple (3,4,5)
          (the structural triple-only signal) is highlighted.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    estimates = load_estimates_csv(estimates_path)
    population = load_population_csv(population_path)

    with mpl.rc_context(_ieee_rcparams()):
        fig, axes = plt.subplots(
            1, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=DPI,
        )

        # Panel (a): SE_K
        ax = axes[0]
        ks = sorted(int(k) for k in estimates["SE_K"].keys())
        means = [estimates["SE_K"][str(k)].mean() for k in ks]
        sds = [estimates["SE_K"][str(k)].std(ddof=1) for k in ks]
        pop_vals = [population["SE_K"][str(k)] for k in ks]

        x = np.arange(len(ks))
        ax.bar(x, means, yerr=sds, color="#4C72B0",
               edgecolor="black", capsize=2.0, error_kw={"elinewidth": 0.6})
        # Population overlay: black horizontal segment on each bar
        for xi, pv in zip(x, pop_vals):
            ax.hlines(pv, xi - 0.4, xi + 0.4,
                      colors="black", linestyles="-", linewidth=1.2,
                      zorder=5)
        ax.axhline(0, color="0.4", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$K{{=}}{k}$" for k in ks])
        ax.set_ylabel(r"$\widehat{\mathrm{SE}}_K$ (nats)")
        ax.set_title("(a) Order spectrum")

        # Panel (b): pair narrow synergies
        ax = axes[1]
        labels_pair = _ordered_pair_labels()
        means_p = [estimates["pair_syn"][lbl].mean() for lbl in labels_pair]
        sds_p = [estimates["pair_syn"][lbl].std(ddof=1) for lbl in labels_pair]
        pop_p = [population["pair_syn"][lbl] for lbl in labels_pair]
        target_idx = labels_pair.index("(1,2)")
        colors = ["#C44E52" if i == target_idx else "#4C72B0"
                  for i in range(len(labels_pair))]
        x = np.arange(len(labels_pair))
        ax.bar(x, means_p, yerr=sds_p, color=colors,
               edgecolor="black", capsize=1.5, error_kw={"elinewidth": 0.4})
        for xi, pv in zip(x, pop_p):
            ax.hlines(pv, xi - 0.4, xi + 0.4,
                      colors="black", linestyles="-", linewidth=1.0,
                      zorder=5)
        ax.axhline(0, color="0.4", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_pair, rotation=45, ha="right")
        ax.set_ylabel(r"$\widehat{\mathrm{Syn}}^{(2)}$ (nats)")
        ax.set_title("(b) Pair narrow synergy")

        # Panel (c): triple narrow synergies
        ax = axes[2]
        labels_triple = _ordered_triple_labels()
        means_t = [estimates["triple_syn"][lbl].mean() for lbl in labels_triple]
        sds_t = [estimates["triple_syn"][lbl].std(ddof=1) for lbl in labels_triple]
        pop_t = [population["triple_syn"][lbl] for lbl in labels_triple]
        target_idx = labels_triple.index("(3,4,5)")
        colors = ["#C44E52" if i == target_idx else "#4C72B0"
                  for i in range(len(labels_triple))]
        x = np.arange(len(labels_triple))
        ax.bar(x, means_t, yerr=sds_t, color=colors,
               edgecolor="black", capsize=1.5, error_kw={"elinewidth": 0.4})
        for xi, pv in zip(x, pop_t):
            ax.hlines(pv, xi - 0.4, xi + 0.4,
                      colors="black", linestyles="-", linewidth=1.0,
                      zorder=5)
        ax.axhline(0, color="0.4", linewidth=0.4, linestyle="--", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_triple, rotation=45, ha="right")
        ax.set_ylabel(r"$\widehat{\mathrm{Syn}}^{(3)}$ (nats)")
        ax.set_title("(c) Triple narrow synergy")

        for ax in axes:
            ax.tick_params(direction="in", length=2.0, pad=2)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        fig.tight_layout(pad=0.6)
        fig.savefig(figure_path, bbox_inches="tight", dpi=DPI)
        png = figure_path.replace(".pdf", ".png")
        fig.savefig(png, bbox_inches="tight", dpi=DPI)
        plt.close(fig)
    print(f"  wrote {figure_path}  and  {png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params = TUNED_PARAMS
    print(f"[experiment1_run] tuned params: {params.to_dict()}")

    print(f"[experiment1_run] computing population (closed-form) reference")
    Sigma = bench.build_covariance(params)
    gt = bench.ground_truth_spectrum(Sigma)
    save_population_csv(gt, POPULATION_CSV)
    print(f"  population SE_K: {bench.format_spectrum(gt['spectrum'])}")
    print(f"  population TSE = {gt['tse']:.4f}")

    print(f"[experiment1_run] running {N_TRIALS} trials of M={M_SAMPLES} samples")
    records = run_trials(params, n_trials=N_TRIALS, n_samples=M_SAMPLES, seed=RNG_SEED)
    save_estimates_csv(records, ESTIMATES_CSV)

    print(f"[experiment1_run] rendering Figure 1")
    render_figure1(ESTIMATES_CSV, POPULATION_CSV, FIGURE_PATH)

    # A small textual summary so the run is self-documenting in stdout.
    est = load_estimates_csv(ESTIMATES_CSV)
    print()
    print("Plug-in mean +/- 1 SD across trials:")
    print("  Order spectrum:")
    for k in sorted(int(k) for k in est["SE_K"].keys()):
        arr = est["SE_K"][str(k)]
        pop = gt["spectrum"][k]
        print(f"    SE_{k}:  hat = {arr.mean():+.4f} +/- {arr.std(ddof=1):.4f}   "
              f"pop = {pop:+.4f}")
    print("  Pair (1,2) Syn:    hat = "
          f"{est['pair_syn']['(1,2)'].mean():+.4f} +/- "
          f"{est['pair_syn']['(1,2)'].std(ddof=1):.4f}   "
          f"pop = {gt['pair_synergy'][(1, 2)]:+.4f}")
    print("  Triple (3,4,5) Syn: hat = "
          f"{est['triple_syn']['(3,4,5)'].mean():+.4f} +/- "
          f"{est['triple_syn']['(3,4,5)'].std(ddof=1):.4f}   "
          f"pop = {gt['triple_synergy'][(3, 4, 5)]:+.4f}")


if __name__ == "__main__":
    main()
