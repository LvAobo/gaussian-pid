# Closed-Form Gaussian Estimators for Multi-Source Partial Information Decomposition

Code release accompanying the submission to the **2026 IEEE
Information Theory Workshop**:

> *Closed-Form Gaussian Estimators for Multi-Source Partial Information
> Decomposition*.

This repository contains the reference NumPy implementation of every
closed-form estimator introduced in the paper, the discrete plug-in
baseline used in Experiment 2, and five experiment scripts that produce
the figures and tables of Section 5 and the arXiv appendices.

A single `pip install -r requirements.txt` and one `python3` command
per script reproduces the entire experimental section bit-for-bit; all
random seeds are fixed, i.e., `seed = 20260503` or `seed = 20260510` (ITW 2026 ddl).

---

## Files

### Library code

| File | Role |
|------|------|
| `gaussian_pid.py` | **Main contribution.** Closed-form Gaussian estimators: full SE spectrum (`gaussian_synergy_spectrum`), narrow synergy (`gaussian_narrow_synergy`), total synergistic effect (`gaussian_tse`), general unique information (`gaussian_general_unique`), two-source PID (`gaussian_two_source_pid`), total / dual / O-information utilities. |
| `experiment1_benchmark.py` | Construction of the Section 5.1 noise-cancellation Gaussian benchmark, closed-form ground-truth spectrum, and helpers reused by Experiments 1, 3 and 4. |
| `pidtools.py` | Reference discrete-PID implementation, used as the *PRE discrete TSE / N-order Syn* baseline in Experiment 2. |

### Experiment scripts

| File | Section / figure | Purpose |
|------|------------------|---------|
| `experiment1_run.py` | §5.1 / Fig. 1 | Plug-in sampling pipeline ($M{=}1000$, 50 trials), three-panel bar figure of the SE spectrum, all 10 pair narrow synergies, and all 10 triple narrow synergies. |
| `experiment2_run.py` | §5.2 / Fig. 2 | Wall-clock scalability from $N = 2$ to $N = 500$ across eight estimators; raw + summary CSVs with per-platform metadata. |
| `experiment3_run.py` | §5.3 / Fig. 3 | Ridge stability in the small-sample regime ($M = 10\dots 500$, $\lambda = 0\dots 10^{-1}$). Produces a 1-D $\lambda$ sweep and a 2-D $M{\times}\lambda$ heatmap. |
| `experiment4_run.py` | arXiv appendix / Fig. 4 | Finite-sample convergence of the five plug-in estimators $\mathrm{SE}_2$, $\mathrm{SE}_3$, TSE, Syn(pair), Syn(triple) at $M \in \{50, \dots, 10000\}$, 100 trials per cell. |
| `experiment5_run.py` | arXiv appendix / Table 1 | Two-source ($N{=}2$) comparison against Barrett MMI, Venkatesh--Schamberg $\delta$-PID and Venkatesh~$\widetilde G$-PID on five canonical configurations. |

### Reproducibility configuration

| File | Role |
|------|------|
| `requirements.txt` | Pinned Python dependencies. |
| `README.md` | This file. |

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `numpy<2` and `scipy<1.15` because the `dit`
library used as the I_ccs baseline in Experiment 2 is incompatible
with newer releases. The `gpid` package used by Experiment 5 is
fetched directly from the authors' GitHub.

---

## Reproducing the figures and tables

### Figure 1 — order spectrum and subset localisation (Section 5.1)

```bash
python3 experiment1_run.py
```

Runtime: ≈ 15 s on M1 Pro. Outputs:
- `experiment1_estimates.csv` (long-format trial × quantity table)
- `experiment1_population.csv` (closed-form reference values)
- `figure1.pdf`, `figure1.png`

### Figure 2 — computational scalability (Section 5.2)

```bash
python3 experiment2_run.py --reset --budget 1000 --trials 10
```

Runtime: ≈ 1.5 hours on M1 Pro. The runtime is dominated by the
five "death" cells (four exponential baselines plus the closed-form
full SE spectrum) that each consume one wall-clock budget once and
are then blacklisted. Polynomial methods finish the entire
$N = 2 \dots 500$ sweep in well under a minute combined.

Outputs:
- `experiment2_timings_raw.csv` (one row per `(N, method, trial)`)
- `experiment2_timings_summary.csv` (per-cell median / mean / min / max)
- `figure2.pdf`, `figure2.png`

The CSV header records the platform, processor, NumPy / SciPy / `dit`
versions, so reviewers can read the exact compute environment off the
file.

To redraw the figure without re-timing:

```bash
python3 experiment2_run.py --render-only
```

### Figure 3 — ridge stability (Section 5.3)

```bash
python3 experiment3_run.py
```

Runtime: ≈ 5 s on M1 Pro. Outputs:
- `experiment3_estimates.csv` (trial × $(M, \lambda)$ × value)
- `experiment3_population.csv` (closed-form TSE reference)
- `figure3a.pdf` (1-D $\lambda$ sweep)
- `figure3b.pdf` (2-D $M{\times}\lambda$ heatmap)

### Figure 4 — finite-sample convergence (arXiv appendix)

```bash
python3 experiment4_run.py
```

Runtime: ≈ 10 s on M1 Pro. 100 trials per $M$ for
$M \in \{50, 100, 200, 500, 1000, 2000, 5000, 10000\}$ on the
Section 5.1 benchmark. Tracks $\mathrm{SE}_2$, $\mathrm{SE}_3$,
$\mathrm{TSE}$, $\mathrm{Syn}(\{S_1, S_2\}{\to}T)$ and
$\mathrm{Syn}(\{S_3, S_4, S_5\}{\to}T)$.

Outputs:
- `experiment4_estimates.csv` (long-format)
- `experiment4_summary.csv` (per-$M$ mean / std / bias)
- `table_population.tex` (LaTeX table of the five population values)
- `figure_finite_sample.pdf`, `figure_finite_sample.png`

Re-draw the figure without re-sampling:
```bash
python3 experiment4_run.py --render-only
```

### Table 1 — two-source ($N{=}2$) estimator comparison (arXiv appendix)

```bash
python3 experiment5_run.py
```

Runtime: ≈ 30 s on M1 Pro. Compares our closed-form estimator
against Barrett MMI~\cite{barrett2015exploration},
Venkatesh--Schamberg $\delta$-PID~\cite{venkatesh2022partial} and
Venkatesh~$\widetilde G$-PID~\cite{venkatesh2023gaussian} on five
controlled jointly Gaussian configurations. The latter three are
invoked via the authors' released
[`gpid`](https://github.com/praveenv253/gpid) Python package; our
implementation reads from `gaussian_pid.gaussian_two_source_pid`.

Outputs:
- `experiment5_estimates.csv` (per-trial, per-method, per-atom)
- `experiment5_summary.csv` (per-(config, method, atom) mean / std / pop)
- `table_n2_sanity.tex` (LaTeX table, auto-merging estimators that
  give numerically identical decompositions in a config)

---

## Anonymity

The repository was prepared for double-blind review:

- No author names, ORCID numbers, GitHub handles, or institutional
  paths are embedded anywhere in the code.
- All random seeds are fixed (`seed = 20260503`) so a fresh clone
  produces bit-identical figures.
- The `pidtools.py` discrete reference implementation is included so
  Experiment 2 is self-contained; it was originally developed for a
  companion paper that is also under review.
- The `gpid` package fetched by `requirements.txt` is third-party
  code released by Venkatesh et al.; we use it only as a black box
  baseline in Experiment 5.

---

## Quick API tour

```python
import numpy as np
import gaussian_pid as gp

# 5 sources S_1..S_5, target T (column 0).
Sigma = ...                                   # any (N+1)x(N+1) PSD matrix

target  = [0]
sources = [(1,), (2,), (3,), (4,), (5,)]

# K-th order synergistic effect spectrum (SE_2..SE_5)
spectrum = gp.gaussian_synergy_spectrum(Sigma, target, sources)

# Narrow synergy (= SE_N)
syn = gp.gaussian_narrow_synergy(Sigma, target, sources)

# Total synergistic effect
tse = gp.gaussian_tse(Sigma, target, sources)

# General unique information of source i
un_i = gp.gaussian_general_unique(Sigma, target, sources, source_index=0)

# Two-source PID atoms (used by Experiment 5)
pid = gp.gaussian_two_source_pid(Sigma, target=[0], source1=[1], source2=[2])
print(pid.redundancy, pid.unique_s1, pid.unique_s2, pid.synergy)
```

All quantities are returned in **nats** by default; pass `base=2`
for bits. The `ridge` and `ridge_target` keyword arguments allow
Tikhonov regularisation when the empirical conditional covariance is
ill-conditioned (used in Experiment 3).
