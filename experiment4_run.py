"""
experiment4_run.py
==================

Experiment 4 ("Finite-sample convergence") -- arXiv full-version
appendix, with a brief mention in the main text.

Goal
----
Show that the plug-in Gaussian PID estimators converge to their
population values as M grows, for the same noise-cancellation
benchmark as Experiment 1 (Section V-A).

Setup
-----
Reuse Section V-A exactly:
    T_2, T_3 ~ N(0, 1)              (independent)
    U, V_1, V_2 ~ N(0, 2)           (sigma = 2)
    eps_i ~ N(0, 0.05)              i.i.d., i = 1..5
    S_1 = T_2 + U + eps_1
    S_2 = T_2 - U + eps_2
    S_3 = T_3 + V_1 + eps_3
    S_4 = T_3 + V_2 + eps_4
    S_5 = T_3 - V_1 - V_2 + eps_5

The construction parameters live in `TUNED_PARAMS` below.

Quantities computed (population + plug-in)
------------------------------------------
SE_2, SE_3                     -- two-source and three-source SE
TSE                            -- total synergistic effect
Syn(S_1, S_2 -> T)             -- pair narrow synergy
Syn(S_3, S_4, S_5 -> T)        -- triple narrow synergy

Outputs
-------
- `experiment4_estimates.csv`     long format: M, trial, quantity, value
- `experiment4_summary.csv`       per-(M, quantity) mean / std / bias
- `table_population.tex`          LaTeX table of population values
- `figure_finite_sample.png/.pdf` 2-panel convergence figure
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import gaussian_pid as gp
import experiment1_benchmark as bench


# ---------------------------------------------------------------------------
# Tuned construction parameters (same values as Experiment 1 / Section V-A)
# ---------------------------------------------------------------------------
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
# Experiment configuration
# ---------------------------------------------------------------------------
DEFAULT_M_LIST: List[int] = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
DEFAULT_N_TRIALS: int = 100
DEFAULT_SEED: int = 20260503

ESTIMATES_CSV = "experiment4_estimates.csv"
SUMMARY_CSV = "experiment4_summary.csv"
FIGURE_PATH = "figure_finite_sample.png"
TABLE_PATH = "table_population.tex"


# ---------------------------------------------------------------------------
# Quantity metadata
# ---------------------------------------------------------------------------
# Each entry is (key, latex_label, plot_color). All five quantities live
# on a comparable magnitude scale (~2-7 nats) and share one set of axes.
QUANTITIES: List[Tuple[str, str, str]] = [
    ("SE_2",       r"$\mathrm{SE}_2$",                       "#1f77b4"),
    ("SE_3",       r"$\mathrm{SE}_3$",                       "#ff7f0e"),
    ("TSE",        r"$\mathrm{TSE}$",                        "#2ca02c"),
    ("Syn_pair",   r"$\mathrm{Syn}(S_1,S_2{\to}T)$",         "#d62728"),
    ("Syn_triple", r"$\mathrm{Syn}(S_3,S_4,S_5{\to}T)$",     "#9467bd"),
]
QUANTITY_KEYS = [q[0] for q in QUANTITIES]
LATEX_LABEL = {k: latex for k, latex, _ in QUANTITIES}
COLOR_MAP = {k: c for k, _, c in QUANTITIES}


# ---------------------------------------------------------------------------
# Compute all seven quantities from one covariance matrix
# ---------------------------------------------------------------------------

def compute_quantities(Sigma: np.ndarray, ridge: float = 0.0) -> Dict[str, float]:
    """Return the five population / plug-in estimands as a dict."""
    target = list(bench.TARGET_INDICES)
    sources = bench._source_groups()

    spec = gp.gaussian_synergy_spectrum(
        Sigma, target=target, sources=sources, ridge=ridge,
    )
    tse = gp.gaussian_tse(Sigma, target=target, sources=sources, ridge=ridge)
    pair_syn = gp.gaussian_narrow_synergy(
        Sigma, target=target, sources=bench._subset_to_columns([1, 2]),
        ridge=ridge,
    )
    triple_syn = gp.gaussian_narrow_synergy(
        Sigma, target=target, sources=bench._subset_to_columns([3, 4, 5]),
        ridge=ridge,
    )
    return {
        "SE_2": float(spec[2]),
        "SE_3": float(spec[3]),
        "TSE": float(tse),
        "Syn_pair": float(pair_syn),
        "Syn_triple": float(triple_syn),
    }


# ---------------------------------------------------------------------------
# Trial driver
# ---------------------------------------------------------------------------

def run_trials(
    M_list: List[int],
    n_trials: int,
    seed: int,
    verbose: bool = True,
) -> List[Dict]:
    """Sample one (M, n_trials) sweep; return long-format records.

    Each trial gets its own independent RNG via `parent.spawn`, so trials
    are reproducible regardless of scheduling order. The same trial
    seed is used at every M -- only the sample size differs -- so the
    convergence curves can be inspected per-trial as well as in
    aggregate.
    """
    parent = np.random.default_rng(seed)
    streams = list(parent.spawn(n_trials))
    records: List[Dict] = []
    for M in M_list:
        n_fail = 0
        for trial_idx, rng_t in enumerate(streams):
            samples = bench.sample_data(TUNED_PARAMS, n_samples=M, rng=rng_t)
            Sigma_hat = np.cov(samples, rowvar=False, bias=False)
            Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0
            try:
                q = compute_quantities(Sigma_hat)
            except np.linalg.LinAlgError:
                n_fail += 1
                continue
            for name, val in q.items():
                records.append({
                    "M": M, "trial": trial_idx,
                    "quantity": name, "value": float(val),
                })
        if verbose:
            print(f"  M = {M:>5d} : {n_trials - n_fail}/{n_trials} trials OK"
                  + (f", {n_fail} failed" if n_fail else ""))
    return records


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_estimates_csv(records: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["M", "trial", "quantity", "value"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "M": r["M"], "trial": r["trial"],
                "quantity": r["quantity"],
                "value": f"{r['value']:.10g}",
            })
    print(f"  wrote {len(records)} rows -> {path}")


def aggregate_summary(records: List[Dict], pop_values: Dict[str, float]
                      ) -> List[Dict]:
    """Group by (M, quantity); compute mean, std, n, bias."""
    by_cell: Dict[Tuple[int, str], List[float]] = {}
    for r in records:
        by_cell.setdefault((r["M"], r["quantity"]), []).append(r["value"])
    out: List[Dict] = []
    for (M, q), values in sorted(by_cell.items()):
        arr = np.asarray(values, dtype=float)
        pop = float(pop_values[q])
        out.append({
            "M": M, "quantity": q,
            "n_trials": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "bias": float(arr.mean() - pop),
            "population": pop,
        })
    return out


def save_summary_csv(summary: List[Dict], path: str) -> None:
    fields = ["M", "quantity", "n_trials", "mean", "std", "bias", "population"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary:
            writer.writerow({
                "M": r["M"], "quantity": r["quantity"],
                "n_trials": r["n_trials"],
                "mean":  f"{r['mean']:.10g}",
                "std":   f"{r['std']:.10g}",
                "bias":  f"{r['bias']:.10g}",
                "population": f"{r['population']:.10g}",
            })
    print(f"  wrote {len(summary)} rows -> {path}")


def save_latex_table(pop_values: Dict[str, float], path: str) -> None:
    """LaTeX table of the seven population values, ready to \\input."""
    lines = []
    lines.append(r"% Auto-generated by experiment4_run.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Closed-form (population) values of the seven "
                 r"Gaussian hierarchy quantities computed on the "
                 r"Section~V-A noise-cancellation benchmark. These are the "
                 r"asymptotic targets of the finite-sample plug-in "
                 r"estimators (Fig.~\ref{fig:finite_sample}).}")
    lines.append(r"  \label{tab:pop_values}")
    lines.append(r"  \begin{tabular}{lc}")
    lines.append(r"    \toprule")
    lines.append(r"    Quantity & Population value (nats) \\")
    lines.append(r"    \midrule")
    for key, latex, _ in QUANTITIES:
        lines.append(f"    {latex} & {pop_values[key]:.4f} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure: 2-panel mean-+/-std vs M with population overlays
# ---------------------------------------------------------------------------

def render_figure(
    summary: List[Dict],
    pop_values: Dict[str, float],
    fig_path: str,
) -> None:
    """Single-panel convergence figure showing all five estimands
    (SE_2, SE_3, TSE, Syn(pair), Syn(triple)) on one set of axes."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.6,
    }

    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)

        for q in QUANTITY_KEYS:
            rows = sorted([r for r in summary if r["quantity"] == q],
                          key=lambda r: r["M"])
            if not rows:
                continue
            xs = np.array([r["M"] for r in rows], dtype=float)
            mean = np.array([r["mean"] for r in rows], dtype=float)
            std = np.array([r["std"] for r in rows], dtype=float)
            color = COLOR_MAP[q]
            ax.fill_between(xs, mean - std, mean + std,
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, mean, color=color, marker="o", markersize=3.0,
                    label=LATEX_LABEL[q])
            ax.axhline(pop_values[q], color=color, linestyle="--",
                       linewidth=0.7, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlabel("Sample size $M$")
        ax.set_ylabel("Estimator value (nats)")
        ax.set_title("Finite-sample convergence")
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction="in", length=2.5, pad=2)
        ax.legend(loc="best", frameon=True, framealpha=0.85,
                  edgecolor="0.6", handlelength=2.0)

        fig.tight_layout(pad=0.6)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        pdf = fig_path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {fig_path}  and  {pdf}")


# ---------------------------------------------------------------------------
# Entry point + CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--m-list", type=str,
                   default=",".join(str(m) for m in DEFAULT_M_LIST),
                   help="comma-separated sample sizes (default %(default)s)")
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                   help="trials per M (default %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="base RNG seed (default %(default)s)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="directory for CSV / PDF / PNG / TEX (default %(default)s)")
    p.add_argument("--render-only", action="store_true",
                   help="skip experiments; redraw figure + tables from CSV")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    out_dir = args.output_dir
    if out_dir and out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)
    est_path = os.path.join(out_dir, ESTIMATES_CSV)
    sum_path = os.path.join(out_dir, SUMMARY_CSV)
    fig_path = os.path.join(out_dir, FIGURE_PATH)
    tex_path = os.path.join(out_dir, TABLE_PATH)

    M_list = [int(m.strip()) for m in args.m_list.split(",") if m.strip()]

    # 1. Population values (closed form).
    Sigma_pop = bench.build_covariance(TUNED_PARAMS)
    pop_values = compute_quantities(Sigma_pop)
    print("[experiment4] population values (nats):")
    for k in QUANTITY_KEYS:
        print(f"  {k:<12s} = {pop_values[k]:+.4f}")
    save_latex_table(pop_values, tex_path)

    # 2. Trials (skipped in --render-only).
    if not args.render_only:
        print(f"[experiment4] running {args.n_trials} trials per M in {M_list}")
        records = run_trials(M_list, args.n_trials, args.seed)
        save_estimates_csv(records, est_path)
    else:
        # Reload existing CSV.
        if not os.path.exists(est_path):
            print(f"[experiment4] {est_path} missing; nothing to render")
            return
        records = []
        with open(est_path) as f:
            for row in csv.DictReader(f):
                records.append({
                    "M": int(row["M"]), "trial": int(row["trial"]),
                    "quantity": row["quantity"],
                    "value": float(row["value"]),
                })

    summary = aggregate_summary(records, pop_values)
    save_summary_csv(summary, sum_path)
    render_figure(summary, pop_values, fig_path)


if __name__ == "__main__":
    main()
