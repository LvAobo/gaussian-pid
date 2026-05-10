"""
experiment2_run.py
==================

Section 5 / Experiment 2 ("Computational scalability"): wall-clock time
of every estimator vs. the number of sources N, measured under
identical data and identical compute budgets.

Methods compared
----------------
Closed-form Gaussian (ours, solid lines):
    1. Full SE_K spectrum         gp.gaussian_synergy_spectrum
    2. Total synergistic effect   gp.gaussian_tse
    3. Per-source unique info     gp.gaussian_general_unique  (over i)
    4. Narrow synergy             gp.gaussian_narrow_synergy

Discrete plug-in baselines (dashed lines):
    5. PRE framework TSE          pidtools.total_syn_effect
    6. PRE framework narrow Syn   pidtools.multi_source_syn
    7. dit I_ccs narrow Syn       dit.pid.PID_CCS  (Ince 2017)

Aggregate baseline (dot-dash):
    8. Gaussian O-information     gp.o_information   (Rosas et al. 2019)

Optional (deferred, registered through `register_neural_baseline()`):
    9. Neural high-order synergy  TBD

Fairness conditions
-------------------
- Every method receives the SAME data: a single random PSD covariance
  Sigma_N is drawn for each N (deterministic seed), M = 1000 samples
  are drawn from it, and the empirical covariance + the binned
  DataFrame + the dit.Distribution are derived once and reused.
- Every method gets the SAME wall-clock budget per N.
- A method that exceeds the budget at some N is blacklisted: it does
  not run for any larger N, and its curve simply ends at the last N
  that completed within budget.

Outputs
-------
- experiment2_timings.csv (long format, appended incrementally)
- figure2.pdf / figure2.png (the multi-curve scaling figure)

CLI
---
    python3 experiment2_run.py
        [--n-list "2,3,4,...,200"]      # explicit, non-uniform N list
        [--n-min N0] [--n-max N1]       # filters applied on top of --n-list
        [--budget SECONDS]              # per-method per-N wall-clock budget
        [--budget-line SECONDS]         # horizontal budget line on figure
        [--samples M]                   # samples per N (default 1000)
        [--bins B]                      # bins per dim for discrete (default 3)
        [--seed S]                      # base RNG seed
        [--output-dir DIR]              # directory for CSV / PDF / PNG
        [--csv NAME]    [--figure NAME] # filenames within --output-dir
        [--reset]                       # truncate CSV before running
        [--render-only]                 # skip experiments, redraw from CSV

Default reproduction command (paper-version timing):

    python3 experiment2_run.py --reset --budget 600
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Compatibility patch for `dit.PID_CCS` under modern scipy
# ---------------------------------------------------------------------------
# `dit.algorithms.optimization` calls scipy.optimize.minimize with x0 having
# shape != 1D in some code paths. Modern scipy raises ValueError. We monkey-
# patch scipy.minimize so any incoming x0 is flattened.
import numpy as _np
import scipy.optimize as _so

_orig_minimize = _so.minimize


def _patched_minimize(*args, **kwargs):
    if "x0" in kwargs:
        kwargs["x0"] = _np.asarray(kwargs["x0"]).ravel()
    elif len(args) >= 2:
        args = list(args)
        args[1] = _np.asarray(args[1]).ravel()
        args = tuple(args)
    return _orig_minimize(*args, **kwargs)


_so.minimize = _patched_minimize


# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import argparse
import csv
import itertools
import os
import platform
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import gaussian_pid as gp
import pidtools

import dit
from dit.pid import PID_CCS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TIMINGS_CSV = "experiment2_timings.csv"
FIGURE_PATH = "exp2/figure2.pdf"
DEFAULT_M = 1000
DEFAULT_BINS = 3
DEFAULT_BUDGET_S = 1000.0      # paper-version per-method per-N budget (~16.7 min)
RNG_SEED_BASE = 20260503       # ITW deadline week

# Paper-version N list: dense low-N (2..15) captures the explosion points
# of exponential methods (dit I_ccs at N=3-4, PRE narrow_syn at N=5-6,
# the closed-form full SE_K spectrum at N=11-12, PRE TSE at N=12-13).
# Sparse high-N (18..200) verifies that polynomial closed-form methods
# stay tractable up to "large-scale neuroscience / network" sizes.
DEFAULT_N_LIST: List = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    18, 20, 25, 30, 35, 40, 50, 75, 100, 125, 150, 175, 200,
]


# ---------------------------------------------------------------------------
# Soft per-call timeout
# ---------------------------------------------------------------------------

class _MethodTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _MethodTimeout()


def _set_alarm(seconds: float) -> None:
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-3))


def _clear_alarm() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0.0)


def time_with_budget(fn: Callable[[], Any], budget_s: float) -> Tuple[float, str, Any]:
    """Run fn() once and return (elapsed_seconds, status, result_or_None).

    `status` is one of:
        "ok"      method completed within budget,
        "timeout" SIGALRM fired or method took longer than budget,
        "error"   method raised an unhandled exception (other than timeout).

    Important caveats:
    - SIGALRM only interrupts Python at byte-code boundaries, so methods
      stuck in a long C call may run past `budget_s`. The actual elapsed
      time is recorded regardless, so the CSV is always honest. The
      orchestrator still flags the method as `timeout` if elapsed >= budget.
    - The signal handler is reinstalled on every call.
    """
    _set_alarm(budget_s)
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        _clear_alarm()
        if elapsed >= budget_s:
            return elapsed, "timeout", result
        return elapsed, "ok", result
    except _MethodTimeout:
        elapsed = time.perf_counter() - t0
        _clear_alarm()
        return elapsed, "timeout", None
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        _clear_alarm()
        return elapsed, f"error:{type(e).__name__}:{e}", None


# ---------------------------------------------------------------------------
# Dataset for one value of N
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Everything every method might need at one value of N."""
    N: int                          # number of sources (target is one extra dim)
    M: int                          # sample size
    n_bins: int                     # bins per dim for discrete methods
    Sigma: np.ndarray               # population covariance (N+1, N+1)
    samples: np.ndarray             # (M, N+1)  column 0 is target
    Sigma_hat: np.ndarray           # empirical covariance
    df_binned: pd.DataFrame         # PRE-style joint pmf table (long format)
    dit_dist: dit.Distribution      # dit Distribution from binned data
    target_index: int = 0           # column index of T in `samples` and `Sigma_hat`
    source_indices: Tuple[int, ...] = field(default_factory=tuple)
    target_label: str = "T"
    source_labels: List[str] = field(default_factory=list)


def _random_psd_covariance(N_total: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random N_total x N_total positive-definite covariance."""
    A = rng.standard_normal((N_total, N_total))
    Sigma = A @ A.T + 0.5 * np.eye(N_total)
    return (Sigma + Sigma.T) / 2.0


def _bin_samples(samples: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency (quantile) binning per column.

    Returns an int array of the same shape as `samples` with values in
    [0, n_bins). Equal-frequency binning gives every bin equal marginal
    probability, which is the standard convention for discrete plug-in
    estimators on continuous data.
    """
    M, D = samples.shape
    binned = np.zeros((M, D), dtype=int)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for d in range(D):
        cuts = np.quantile(samples[:, d], qs)
        binned[:, d] = np.digitize(samples[:, d], cuts)
    return binned


def _build_pre_dataframe(
    binned: np.ndarray,
    source_labels: List[str],
    target_label: str,
) -> pd.DataFrame:
    """Convert binned (M, N+1) integer matrix into the (Pr, sources..., target)
    DataFrame format expected by `pidtools`.

    The first column of `binned` is the target T; columns 1..N are the
    sources. We aggregate identical rows into one row with probability
    equal to its empirical frequency.
    """
    df = pd.DataFrame(binned, columns=[target_label] + source_labels)
    counts = df.groupby([target_label] + source_labels, sort=False).size()
    counts = counts.reset_index(name="count")
    counts["Pr"] = counts["count"] / counts["count"].sum()
    counts = counts.drop(columns="count")
    # PRE pidtools expects sources first, target last is also fine (it
    # passes labels to the underlying Pandas operations).
    return counts[["Pr"] + source_labels + [target_label]]


def _build_dit_dist(binned: np.ndarray) -> dit.Distribution:
    """Build a dit.Distribution from binned (M, N+1) data, with the target
    in column 0 and sources in columns 1..N. Each variable becomes one
    random variable in the dit distribution; outcomes are tuples of ints
    (taken as strings by dit's default constructor).
    """
    M, D = binned.shape
    # Use string outcomes for compatibility with dit's default alphabet
    # detection. With n_bins <= 9 each variable maps to a single digit.
    outcomes_strs: Dict[str, float] = {}
    for row in binned:
        # Order: source_1, source_2, ..., source_N, target
        # (dit's PID_CCS will be called with target = (N,) in this order)
        s = "".join(str(int(v)) for v in row[1:]) + str(int(row[0]))
        outcomes_strs[s] = outcomes_strs.get(s, 0) + 1
    total = sum(outcomes_strs.values())
    outcomes = list(outcomes_strs.keys())
    pmf = [outcomes_strs[o] / total for o in outcomes]
    return dit.Distribution(outcomes, pmf)


def make_dataset(
    N: int,
    M: int = DEFAULT_M,
    n_bins: int = DEFAULT_BINS,
    seed: int = RNG_SEED_BASE,
    need_discrete_df: bool = True,
    need_dit_dist: bool = True,
) -> Dataset:
    """Generate one Dataset for the requested N.

    When `need_discrete_df` is False we skip the (very expensive at
    large N) pandas group-by that produces the PRE-format DataFrame;
    likewise `need_dit_dist=False` skips the dit.Distribution build.
    The orchestrator passes False whenever every consumer of those
    structures has been blacklisted, which is the usual case at large N
    once the discrete baselines have run out of budget.
    """
    n_total = N + 1
    rng = np.random.default_rng(seed + N)
    Sigma = _random_psd_covariance(n_total, rng)
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((M, n_total))
    samples = Z @ L.T  # column 0 = target, columns 1..N = sources
    Sigma_hat = np.cov(samples, rowvar=False, bias=False)
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0

    source_labels = [f"S{i + 1}" for i in range(N)]
    target_label = "T"

    df_binned: pd.DataFrame
    dit_dist: Any
    if need_discrete_df or need_dit_dist:
        binned = _bin_samples(samples, n_bins=n_bins)
        df_binned = (
            _build_pre_dataframe(binned, source_labels, target_label)
            if need_discrete_df else pd.DataFrame()
        )
        dit_dist = _build_dit_dist(binned) if need_dit_dist else None
    else:
        df_binned = pd.DataFrame()
        dit_dist = None

    return Dataset(
        N=N, M=M, n_bins=n_bins,
        Sigma=Sigma, samples=samples, Sigma_hat=Sigma_hat,
        df_binned=df_binned, dit_dist=dit_dist,
        target_index=0, source_indices=tuple(range(1, n_total)),
        target_label=target_label, source_labels=source_labels,
    )


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

@dataclass
class Method:
    name: str                           # human-readable label
    kind: str                           # "ours" | "discrete" | "aggregate" | "neural"
    linestyle: str                      # matplotlib linestyle
    color: str                          # matplotlib color
    fn: Callable[[Dataset], Any]        # callable that does ALL the work for one Dataset


def _ours_full_spectrum(ds: Dataset) -> Any:
    return gp.gaussian_synergy_spectrum(
        ds.Sigma_hat,
        target=[ds.target_index],
        sources=[(c,) for c in ds.source_indices],
        return_components=False,
    )


def _ours_tse(ds: Dataset) -> float:
    return gp.gaussian_tse(
        ds.Sigma_hat,
        target=[ds.target_index],
        sources=[(c,) for c in ds.source_indices],
    )


def _ours_per_source_un(ds: Dataset) -> List[float]:
    sources = [(c,) for c in ds.source_indices]
    return [
        gp.gaussian_general_unique(
            ds.Sigma_hat,
            target=[ds.target_index],
            sources=sources,
            source_index=i,
        )
        for i in range(len(sources))
    ]


def _ours_narrow_synergy(ds: Dataset) -> float:
    return gp.gaussian_narrow_synergy(
        ds.Sigma_hat,
        target=[ds.target_index],
        sources=[(c,) for c in ds.source_indices],
    )


def _discrete_pre_tse(ds: Dataset) -> float:
    return pidtools.total_syn_effect(
        ds.df_binned, src=list(ds.source_labels), tgt=ds.target_label
    )


def _discrete_pre_narrow_synergy(ds: Dataset) -> Any:
    # `multi_source_syn` returns a Series with key `order_{N}_syn`; we
    # only time the call (the value isn't used in the figure), but we
    # still return it so the orchestrator can save it for diagnostics.
    # `pidtools` prints intermediate state to stdout; suppress so the
    # CSV-driven log stays clean.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return pidtools.multi_source_syn(
            ds.df_binned, src=list(ds.source_labels), tgt=ds.target_label
        )


def _dit_iccs_narrow_synergy(ds: Dataset) -> Any:
    """Compute the full I_ccs PID lattice on the binned distribution.

    The ‘narrow synergy’ atom is the top atom ((0, 1, ..., N-1),) but
    PID_CCS internally builds and evaluates the WHOLE redundancy lattice
    plus a maxent fit — that full work is the cost we want to time.
    """
    sources = tuple((i,) for i in range(ds.N))   # variables 0..N-1 are sources
    target = (ds.N,)                              # variable N is target
    pid = PID_CCS(ds.dit_dist, sources, target)
    # Touch the top atom so the lattice values get evaluated.
    top = (tuple(range(ds.N)),)
    try:
        return float(pid.get_pi(top))
    except Exception:
        return float("nan")


def _gaussian_o_information(ds: Dataset) -> float:
    return gp.o_information(
        ds.Sigma_hat,
        groups=[(c,) for c in ds.source_indices],
    )


METHODS: List[Method] = [
    Method("Ours: SE_K spectrum",      "ours",      "-",  "#1f77b4", _ours_full_spectrum),
    Method("Ours: TSE",                 "ours",      "-",  "#ff7f0e", _ours_tse),
    Method("Ours: per-source Un (all)", "ours",      "-",  "#2ca02c", _ours_per_source_un),
    Method("Ours: narrow synergy",      "ours",      "-",  "#d62728", _ours_narrow_synergy),
    Method("PRE discrete: TSE",         "discrete",  "--", "#9467bd", _discrete_pre_tse),
    Method("PRE discrete: narrow syn",  "discrete",  "--", "#8c564b", _discrete_pre_narrow_synergy),
    Method("dit I_ccs: narrow syn",     "discrete",  "--", "#e377c2", _dit_iccs_narrow_synergy),
    Method("Gaussian O-information",    "aggregate", "-.", "#7f7f7f", _gaussian_o_information),
]


def register_neural_baseline(name: str, fn: Callable[[Dataset], Any],
                              color: str = "#bcbd22") -> None:
    """Hook for plugging a neural high-order estimator into the panel.

    Once a reference is chosen, call this function before `run_experiment`
    (or before importing this module from another script):

        register_neural_baseline("MINE-style high-order syn", my_neural_fn)

    The CSV will then carry a row for the new method at every successful
    N, and the figure will pick it up automatically.
    """
    METHODS.append(
        Method(name=name, kind="neural", linestyle=":", color=color, fn=fn)
    )


# ---------------------------------------------------------------------------
# Hardware / environment metadata for paper reproducibility
# ---------------------------------------------------------------------------

def _environment_metadata() -> Dict[str, str]:
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "n/a"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
        "scipy": scipy_v,
        "pandas": pd.__version__,
        "dit": dit.__version__,
    }


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "n", "method", "kind", "linestyle", "color",
    "elapsed_s", "status", "M", "n_bins", "seed",
    "python", "platform", "machine", "processor",
    "numpy", "scipy", "pandas", "dit",
]


def _ensure_csv(path: str) -> None:
    """Create a header-only CSV if the file is missing OR empty.

    The empty-file case matters because `--reset` truncates rather than
    unlinks (FUSE mounts may forbid unlink), so a re-run on a truncated
    file would otherwise produce header-less rows.
    """
    needs_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    if needs_header:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def _load_existing(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _append_csv(path: str, row: Dict[str, str]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_experiment(
    n_list: List[int],
    budget_s: float = DEFAULT_BUDGET_S,
    M: int = DEFAULT_M,
    n_bins: int = DEFAULT_BINS,
    seed: int = RNG_SEED_BASE,
    csv_path: str = TIMINGS_CSV,
    verbose: bool = True,
) -> None:
    """Iterate over the explicit `n_list`, time every active method at
    each N, and append rows to `csv_path`.

    The orchestrator is resumable: existing rows are loaded and any
    method that previously timed-out (or errored) at some N is treated
    as blacklisted for any N >= that value. This means a long run can
    be split into multiple smaller invocations (e.g. when running
    inside a sandbox with its own wall-clock cap).
    """
    _ensure_csv(csv_path)
    n_list = sorted(set(int(n) for n in n_list))
    if not n_list:
        print("[experiment2] empty n_list; nothing to do")
        return

    # Pull existing rows so we can resume without redoing finished work
    # and so we know which methods are already blacklisted at which N.
    existing = _load_existing(csv_path)
    blacklisted: Dict[str, int] = {}    # method_name -> first N where it stopped
    completed: set = set()              # (n, method_name) already in CSV
    for row in existing:
        try:
            completed.add((int(row["n"]), row["method"]))
        except (KeyError, ValueError):
            continue
        if row["status"] == "timeout" and row["method"] not in blacklisted:
            blacklisted[row["method"]] = int(row["n"])
        elif row["status"].startswith("error") and row["method"] not in blacklisted:
            # Errors are not budget-related but still indicate the method
            # cannot continue beyond this N for this code path.
            blacklisted[row["method"]] = int(row["n"])

    env = _environment_metadata()
    if verbose:
        print(f"[experiment2] platform: {env['platform']}")
        print(f"[experiment2] processor: {env['processor']}")
        print(f"[experiment2] python {env['python']}, "
              f"numpy {env['numpy']}, scipy {env['scipy']}, dit {env['dit']}")
        print(f"[experiment2] N list ({len(n_list)} values): {n_list}")
        print(f"[experiment2] per-call budget = {budget_s}s,  "
              f"M = {M},  n_bins = {n_bins},  seed = {seed}")
        if blacklisted:
            print(f"[experiment2] resuming with already blacklisted: "
                  f"{', '.join(blacklisted.keys())}")

    for N in n_list:
        active = [m for m in METHODS
                  if blacklisted.get(m.name, max(n_list) + 1) > N
                  and (N, m.name) not in completed]
        if not active:
            if verbose:
                print(f"  N={N}: nothing to do (all completed or blacklisted)")
            continue
        if verbose:
            names = ", ".join(m.name for m in active)
            print(f"  N={N}: building dataset (M={M}); active: {names}")

        # Decide whether to build the discrete-DataFrame and dit
        # Distribution. This pandas pipeline is O(N) per cell at large
        # N and produces a fragmented frame; skipping it once both
        # discrete-only consumers are blacklisted saves a lot of time
        # at sparse high-N points (N >= 25 in the default list).
        active_names = {m.name for m in active}
        need_df = any(name in active_names for name in (
            "PRE discrete: TSE", "PRE discrete: narrow syn"
        ))
        need_dit = "dit I_ccs: narrow syn" in active_names

        try:
            ds = make_dataset(
                N=N, M=M, n_bins=n_bins, seed=seed,
                need_discrete_df=need_df, need_dit_dist=need_dit,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  N={N}: dataset build failed: {e}")
            continue

        for m in active:
            elapsed, status, _ = time_with_budget(lambda m=m: m.fn(ds), budget_s)
            row = {
                "n": N,
                "method": m.name,
                "kind": m.kind,
                "linestyle": m.linestyle,
                "color": m.color,
                "elapsed_s": f"{elapsed:.6f}",
                "status": status,
                "M": M,
                "n_bins": n_bins,
                "seed": seed,
                **env,
            }
            _append_csv(csv_path, row)
            completed.add((N, m.name))
            if status != "ok":
                blacklisted.setdefault(m.name, N)
            if verbose:
                tag = status if status == "ok" else f"[{status}]"
                print(f"    {m.name:<32}  {elapsed:7.3f}s   {tag}")


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

# IEEE single-column = 3.5", double-column = 7.16". Experiment 2 figure
# is information-dense (8+ curves) so a double-column figure reads better.
FIGURE_WIDTH_IN = 7.16
FIGURE_HEIGHT_IN = 3.4
DPI = 300


def _ieee_rcparams() -> Dict:
    return {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
    }


def _budget_label(budget_s: float) -> str:
    """Produce a human-readable label for the budget horizontal line.

    Whole-minute budgets render as e.g. "10-minute wall-clock budget";
    otherwise we fall back to seconds. The default 1000 s is close to but
    not exactly an integer-minute count, so it renders as "1000 s
    wall-clock budget".
    """
    if budget_s >= 60.0 and abs(budget_s - 60.0 * round(budget_s / 60.0)) < 0.5:
        m = int(round(budget_s / 60.0))
        if m == 1:
            return "1-minute wall-clock budget"
        return f"{m}-minute wall-clock budget"
    return f"{budget_s:g} s wall-clock budget"


def render_figure2(
    csv_path: str = TIMINGS_CSV,
    figure_path: str = FIGURE_PATH,
    budget_horizontal_s: float = DEFAULT_BUDGET_S,
) -> None:
    """Plot wall-clock time (log10 sec) vs. N for every method."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rows = _load_existing(csv_path)
    if not rows:
        print(f"  no rows in {csv_path}; nothing to plot.")
        return

    df = pd.DataFrame(rows)
    df["n"] = df["n"].astype(int)
    df["elapsed_s"] = df["elapsed_s"].astype(float)
    # Methods are plotted in the registry order so the legend is grouped
    # by category (ours / discrete / aggregate / neural).
    method_order = [m.name for m in METHODS]

    with mpl.rc_context(_ieee_rcparams()):
        # Reserve right-hand strip for the legend so it does not overlap
        # the curves. We render with constrained_layout=False and fit the
        # legend via bbox_to_anchor.
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=DPI)

        for m in METHODS:
            sub = df[df["method"] == m.name].sort_values("n")
            if sub.empty:
                continue
            xs = sub["n"].to_numpy()
            ys = sub["elapsed_s"].to_numpy()
            statuses = sub["status"].to_numpy()
            ok_mask = statuses == "ok"
            # Main line: only the points that completed within budget.
            ax.plot(
                xs[ok_mask], np.log10(ys[ok_mask]),
                linestyle=m.linestyle, color=m.color, marker="o",
                markersize=3.0, label=m.name, zorder=3,
            )
            # Mark the first non-ok point with a large 'X' in the same
            # color, so the figure tells the reader exactly where the
            # method ran out of budget (or otherwise stopped).
            non_ok_idx = np.where(~ok_mask)[0]
            if non_ok_idx.size > 0:
                first = non_ok_idx[0]
                ax.plot(
                    [xs[first]], [np.log10(ys[first])],
                    marker="X", color=m.color, markersize=8.5,
                    markeredgewidth=1.3, markeredgecolor="black",
                    linestyle="None", zorder=5,
                )

        # Horizontal budget line.
        if budget_horizontal_s and budget_horizontal_s > 0:
            ax.axhline(np.log10(budget_horizontal_s),
                       color="0.3", linewidth=0.7, linestyle=":",
                       zorder=1)
            x_left = df["n"].min()
            ax.text(x_left + 0.1, np.log10(budget_horizontal_s) + 0.10,
                    _budget_label(budget_horizontal_s),
                    fontsize=6.5, color="0.25")

        ax.set_xlabel("Number of sources $N$")
        ax.set_ylabel("Wall-clock time  [$\\log_{10}$ s]")
        ax.set_title("Estimator scalability with $N$")

        # Log-scale x-axis. We label a curated subset of N values so
        # nothing collides on the log scale, where the dense range
        # [2, 15] occupies less than one decade. Tick marks are kept
        # at every data point (as minor ticks without labels) so the
        # full sweep is visible.
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter
        ax.set_xscale("log")
        labeled_ticks = [2, 3, 5, 8, 15, 30, 75, 200]
        all_data_ticks = sorted(set(
            list(range(2, 16)) + [18, 20, 25, 30, 35, 40, 50,
                                    75, 100, 125, 150, 175, 200]
        ))
        unlabeled_ticks = [n for n in all_data_ticks if n not in labeled_ticks]
        ax.xaxis.set_major_locator(FixedLocator(labeled_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in labeled_ticks]))
        ax.xaxis.set_minor_locator(FixedLocator(unlabeled_ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="major", length=3.0, labelsize=7.0,
                        rotation=0)
        ax.tick_params(axis="x", which="minor", length=1.5)
        ax.tick_params(direction="in", pad=3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.4, which="major")
        ax.set_xlim(left=1.8, right=230)

        # Place legend OUTSIDE the axes so it cannot cover the polynomial
        # curves clustered near the bottom. With ITW two-column figures
        # this fits comfortably on the right.
        leg = ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=False, ncol=1, handlelength=2.4,
            borderaxespad=0.0,
        )
        # Add a tiny "X = budget exceeded" annotation in the legend.
        from matplotlib.lines import Line2D
        x_marker = Line2D([0], [0], marker="X", color="0.3",
                           markeredgecolor="black", markeredgewidth=1.2,
                           markersize=7, linestyle="None",
                           label="budget exceeded")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(x_marker); labels.append("budget exceeded")
        ax.legend(
            handles, labels,
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=False, ncol=1, handlelength=2.4,
            borderaxespad=0.0,
        )

        fig.tight_layout(pad=0.6, rect=(0.0, 0.0, 0.78, 1.0))
        fig.savefig(figure_path, bbox_inches="tight", dpi=DPI)
        png = figure_path.replace(".pdf", ".png")
        fig.savefig(png, bbox_inches="tight", dpi=DPI)
        plt.close(fig)
    print(f"  wrote {figure_path} and {png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_n_list(s: str) -> List[int]:
    """Parse a comma-separated list of N values, supporting hyphen-ranges.

    Examples:
        "2,3,5"       -> [2, 3, 5]
        "2-5,10"      -> [2, 3, 4, 5, 10]
        "2-4,6,8-10"  -> [2, 3, 4, 6, 8, 9, 10]
    Whitespace is ignored. The result is sorted and deduplicated.
    """
    out: List[int] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_str, hi_str = piece.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(piece))
    return sorted(set(out))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n-list", type=str,
        default=",".join(str(n) for n in DEFAULT_N_LIST),
        help="comma-separated N values (hyphen ranges OK, e.g. '2-5,10'). "
             "Default = the 27-point list used for the paper figure.",
    )
    p.add_argument("--n-min", type=int, default=None,
                   help="restrict --n-list to N >= n_min (optional filter)")
    p.add_argument("--n-max", type=int, default=None,
                   help="restrict --n-list to N <= n_max (optional filter)")
    p.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET_S,
        help="per-method per-N wall-clock budget in seconds "
             "(default %(default)s = 10 minutes; the paper uses this value)",
    )
    p.add_argument(
        "--budget-line", type=float, default=None,
        help="horizontal-line budget marker on the figure (default = same "
             "as --budget; set <=0 to disable)",
    )
    p.add_argument("--samples", "-M", type=int, default=DEFAULT_M,
                   help="number of samples per N (default %(default)s)")
    p.add_argument("--bins", "-b", type=int, default=DEFAULT_BINS,
                   help="bins per dim for discrete plug-in methods "
                        "(default %(default)s)")
    p.add_argument("--seed", type=int, default=RNG_SEED_BASE,
                   help="base RNG seed; per-N seed = base + N "
                        "(default %(default)s)")
    p.add_argument("--output-dir", type=str, default=".",
                   help="directory for CSV / PDF / PNG (default %(default)s)")
    p.add_argument("--csv", type=str, default=TIMINGS_CSV,
                   help="filename of incremental timings CSV inside output-dir "
                        "(default %(default)s)")
    p.add_argument("--figure", type=str, default=FIGURE_PATH,
                   help="filename of the figure PDF inside output-dir "
                        "(default %(default)s)")
    p.add_argument("--reset", action="store_true",
                   help="truncate the CSV before running (start fresh)")
    p.add_argument("--render-only", action="store_true",
                   help="skip experiments; only redraw the figure from the CSV")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    # --- Resolve paths --------------------------------------------------
    out_dir = args.output_dir
    if out_dir and out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv)
    fig_path = os.path.join(out_dir, args.figure)

    # --- Resolve N list -------------------------------------------------
    n_list = _parse_n_list(args.n_list)
    if args.n_min is not None:
        n_list = [n for n in n_list if n >= args.n_min]
    if args.n_max is not None:
        n_list = [n for n in n_list if n <= args.n_max]

    # --- Resolve budget line --------------------------------------------
    budget_line = args.budget_line if args.budget_line is not None else args.budget

    # --- Reset --------------------------------------------------------
    if args.reset and os.path.exists(csv_path):
        # Use truncation rather than os.remove because the file may live
        # on a FUSE mount where unlink is forbidden but in-place
        # truncation works.
        try:
            open(csv_path, "w").close()
            print(f"[experiment2] truncated {csv_path}")
        except OSError as e:
            print(f"[experiment2] could not reset {csv_path}: {e}")

    # --- Run experiments + render --------------------------------------
    if not args.render_only:
        if not n_list:
            print("[experiment2] empty N list after filters; nothing to run")
        else:
            run_experiment(
                n_list=n_list,
                budget_s=args.budget,
                M=args.samples,
                n_bins=args.bins,
                seed=args.seed,
                csv_path=csv_path,
            )
    render_figure2(
        csv_path=csv_path,
        figure_path=fig_path,
        budget_horizontal_s=budget_line,
    )


if __name__ == "__main__":
    main()
