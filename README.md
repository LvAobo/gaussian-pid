# Closed-Form Gaussian Estimators for Multi-Source Partial Information Decomposition

Code release accompanying the submission to the **2026 IEEE
Information Theory Workshop**:

> *Closed-Form Gaussian Estimators for Multi-Source Partial Information
> Decomposition*.

This repository contains the reference NumPy implementation of every
closed-form estimator introduced in the paper, the discrete plug-in
baseline used in Section&nbsp;5.2, and the three experiment scripts that
produce Figures&nbsp;1–3.

A single `pip install` and one `python3` command per figure are enough
to reproduce the entire experimental section bit-for-bit; all random
seeds are fixed, i.e., `seed = 20260503` or `seed = 20260510`(ITW 2026 ddl).

---

## Files

| File | Role |
|------|------|
| `gaussian_pid.py` | **Main contribution.** Closed-form Gaussian estimators: full SE spectrum (`gaussian_synergy_spectrum`), narrow synergy (`gaussian_narrow_synergy`), total synergistic effect (`gaussian_tse`), general unique information (`gaussian_general_unique`), two-source PID (`gaussian_two_source_pid`), plus total / dual / O-information utilities. |
| `experiment1_benchmark.py` | Construction of the noise-cancellation Gaussian benchmark of Section&nbsp;5.1, closed-form ground-truth spectrum, and the parameter tuner used to obtain the configuration in `experiment1_params.json`. |
| `experiment1_params.json` | Tuned construction parameters (`a`, `b`, `σ_U`, `σ_V`, `σ_ε`, `σ_T`) used by the paper figure. |
| `experiment1_run.py` | **Figure&nbsp;1.** Plug-in sampling pipeline (M = 1000, 50 trials), three-panel bar figure of the SE spectrum, all 10 pair narrow synergies, and all 10 triple narrow synergies. |
| `experiment2_run.py` | **Figure&nbsp;2.** Wall-clock scalability study from N = 2 to N = 500 across eight estimators; per-trial raw CSV plus per-cell summary CSV. |
| `experiment3_run.py` | **Figure&nbsp;3.** Ridge-stability study in the small-sample regime (M = 10–500, λ = 0–10⁻¹). Produces both the 1-D λ sweep (Version&nbsp;A) and the 2-D M × λ heatmap (Version&nbsp;B). |
| `pidtools.py` | Reference discrete-PID implementation used as the *PRE discrete TSE / narrow synergy* baseline in Experiment&nbsp;2. |
| `requirements.txt` | Pinned Python dependencies. |

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `numpy<2` and `scipy<1.15`. Both bounds are
needed exclusively by the `dit` library used as the I_ccs baseline in
Experiment&nbsp;2; `gaussian_pid.py` itself runs on any modern NumPy.

---

## Reproducing the figures

### Figure 1 — order-spectrum recovery (Section 5.1)

```bash
python3 experiment1_run.py
```

≈ 15 s on M1 Pro. Outputs `experiment1_estimates.csv`,
`experiment1_population.csv`, `figure1.pdf`, `figure1.png`.

### Figure 2 — computational scalability (Section 5.2)

```bash
python3 experiment2_run.py --reset --budget 1000 --trials 10
```

≈ 1.5 hours on M1 Pro. The runtime is dominated by five "death" cells
(the four exponential baselines plus the closed-form full SE spectrum)
that each consume one wall-clock budget once and are then blacklisted.
Polynomial methods finish the entire N = 2 .. 500 sweep in well under
a minute combined.

The script writes `experiment2_timings_raw.csv` (one row per
`(N, method, trial)`) and `experiment2_timings_summary.csv` (per-cell
median / mean / min / max), then renders `figure2.pdf` / `figure2.png`.
The CSV header records the platform, processor, NumPy / SciPy / `dit`
versions, so reviewers can read the exact compute environment off the
file.

To redraw the figure without re-timing:

```bash
python3 experiment2_run.py --render-only
```

### Figure 3 — ridge stability in the small-sample regime (Section 5.3)

```bash
python3 experiment3_run.py
```

≈ 5 s on M1 Pro. Outputs `experiment3_estimates.csv`,
`experiment3_population.csv`, `figure3a.pdf` (Version A: 1-D λ sweep)
and `figure3b.pdf` (Version B: M × λ heatmap).

---

## Anonymity

The repository was prepared for double-blind review:

- No author names, ORCID numbers, GitHub handles, or institutional
  paths are embedded anywhere in the code.
- All random seeds are fixed (`seed = 20260503`) so a fresh clone
  produces bit-identical figures.
- The `pidtools.py` discrete reference implementation is included so
  Experiment&nbsp;2 is self-contained; it was originally developed for
  a companion paper that is also under review.

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

# Two-source PID atoms (Theorem 1)
pid = gp.gaussian_two_source_pid(Sigma, target=[0], source1=[1], source2=[2])
print(pid.redundancy, pid.unique_s1, pid.unique_s2, pid.synergy)
```

All quantities are returned in **nats** by default; pass `base=2` for bits.
The `ridge` and `ridge_target` keyword arguments allow Tikhonov
regularisation when the empirical conditional covariance is
ill-conditioned (Section&nbsp;5.3).
