# IVIMBench

**Extensible quantitative-MRI uncertainty calibration, acquisition-design, scan–rescan repeatability, model-misspecification, and clinical decision-utility benchmark suite for IVIM diffusion MRI.**

[![CI](https://github.com/akarlin3/IVIMBench/actions/workflows/ci.yml/badge.svg)](https://github.com/akarlin3/IVIMBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Typed: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](src/ivimbench/py.typed)
[![Version 1.1.0](https://img.shields.io/badge/version-1.1.0-green.svg)](pyproject.toml)

`IVIMBench` provides a unified, zero-PHI, purely NumPy/SciPy/Pandas benchmark platform for intravoxel incoherent motion (IVIM) diffusion MRI:

$$S(b) = S_0 \left[ f \exp(-b D^*) + (1 - f) \exp(-b D) \right]$$

All parameters use canonical physical MRI units across the entire library: diffusion weighting $b$ in $\text{s/mm}^2$, slow tissue diffusivity $D$ and microvascular pseudo-diffusion $D^*$ in $\text{mm}^2/\text{s}$, and perfusion fraction $f \in [0, 1]$ (ordered `("D", "Dstar", "f")`, with `D_star` accepted interchangeably). Every stochastic phantom, noise realization, bootstrap CI, and permutation test is deterministically seeded with `numpy.random.PCG64` and 100% synthetic/PHI-free.

---

## Architecture Overview (v1.1.0)

1. **Estimator Registry & Multi-Task Benchmark Runner (`ivimbench.registry`, `ivimbench.runner`)** — `@register_estimator` protocol accepting any callable or class fitting `(signals, bvalues) -> {D, D_star, f, intervals?}`, shipped with pure NumPy/SciPy deterministic baselines (`"segmented"`, `"nlls"`, `"crlb"`) and `run_benchmark_suite(...)` exporting reproducible Markdown, JSON, LaTeX, and CSV tables across 5 forward families, SNR regimes, and clinical schedules.
2. **Acquisition Design & NEX-Weighted Fisher Information (`ivimbench.acquisition`)** — Closed-form IVIM Fisher Information Matrix (`ivim_fisher_matrix`), Cramér–Rao Lower Bounds (`compute_crlb`) with explicit scan-time / NEX weighting, `SchemeEvaluator`, and `optimize_b_schedule(...)` reporting sharpness (`delta_sharp`), marginal coverage, and high-$D^*$ conditional coverage (`delta_cond`).
3. **Misspecification & Scan–Rescan Repeatability Harnesses (`ivimbench.harnesses`)** —
   - `MisspecificationHarness`: Evaluates bi-exponential estimators under controlled non-bi-exponential forward physics (`dispersion_gamma`, `dispersion_lognormal`, `stretched`, `triexp`, `kurtosis`, `heavy_tail`), explicitly separating global marginal coverage from slow-diffusion $D$ vs. pseudo-diffusion $D^*$ conditional degradation.
   - `RepeatabilityHarness`: Simulates paired same-session acquisitions (`scan_A`, `scan_B`) under exact Rician or multi-coil non-central $\chi$ noise with optional SNR/gain/coil drift, scoring ground-truth-free retest capture alongside Spearman width-vs-scatter rank correlation.
4. **Clinical Decision Value & Domain-Shift Safeguards (`ivimbench.decision`)** — Threshold utility pricing (`evaluate_decision_voi`, `compare_decision_calibration`) computing expected utility regret, the decision–calibration gap, and second-order Value of Information (`second_order_voi`) across clinical action cutoffs $\tau$, paired with a two-sample observable Mahalanobis signal-space + residual-space staleness detector (`check_domain_shift`).
5. **Enriched 2D/3D Anatomical DROs & CLI (`ivimbench.phantoms`, `ivimbench.cli`)** — Structured `liver`, `pancreas` (desmoplastic fibrotic/hypovascular core + steep hypervascular perfusion rim), and `breast` (ACRIN-6698 aligned) phantoms with optional spatial $B_1$ and surface-coil SNR inhomogeneity maps, plus the `ivimbench` console entry point.

---

## Installation

Install from GitHub or PyPI:

```bash
pip install git+https://github.com/akarlin3/IVIMBench.git
# Or via PyPI:
pip install ivimbench

# Optional matplotlib/seaborn diagnostic plotting extras:
pip install "ivimbench[vis]"
```

For local development, linting, static type-checking, and testing:

```bash
git clone https://github.com/akarlin3/IVIMBench.git
cd IVIMBench
pip install -e ".[dev]"
pytest -q
```

---

## Command-Line Interface (`ivimbench`)

`IVIMBench` installs the `ivimbench` console script for immediate reproducible benchmarking:

```bash
# 1. Fast (<10s) end-to-end synthetic DRO + multi-family calibration, repeatability & VoI smoke benchmark
ivimbench demo

# 2. Multi-task benchmark across all 5 forward families, exporting Markdown, JSON, LaTeX & CSV tables
ivimbench benchmark --protocol osipi --snr 30 --out results/

# 3. Compare fixed clinical schedules (and optional CRLB/calibration-optimized designs)
ivimbench compare-schemes --schemes osipi,acrin_6698 --include-optimized
```

---

## Quickstart 1: Benchmark Any Custom Estimator in 5 Lines of Python

Register any NLLS, segmented, Bayesian, conformal, or deep-learning (`IVIMnet`) model via `@register_estimator` and evaluate it across protocols, SNR regimes, and forward physics:

```python
import numpy as np
from ivimbench import register_estimator, run_benchmark_suite

@register_estimator("my_ivimnet")
def fit_my_ivimnet(signals, bvalues):
    d_hat = np.clip(-np.log(np.clip(signals[:, -1] / signals[:, 0], 1e-4, 1.0)) / bvalues[-1], 1e-4, 4e-3)
    pt = np.column_stack([d_hat, np.full_like(d_hat, 35e-3), np.full_like(d_hat, 0.18)])
    return {"D": pt[:, 0], "D_star": pt[:, 1], "f": pt[:, 2], "intervals": (0.8 * pt, 1.2 * pt)}

res = run_benchmark_suite(
    estimators=["my_ivimnet", "segmented", "nlls", "crlb"],
    protocols=["osipi", "tcga_lihc", "acrin_6698"],
    snr_regimes=[20.0, 40.0],
    out_dir="results/",
)
print(res.to_markdown(aggregate=True))
```

---

## Quickstart 2: Acquisition CRLB Design, Repeatability & Decision VoI

```python
from ivimbench import (
    SchemeEvaluator,
    check_domain_shift,
    compute_crlb,
    evaluate_decision_voi,
    evaluate_misspecification,
    evaluate_repeatability,
    generate_cohort,
    generate_dro,
    optimize_b_schedule,
    run_estimator,
    simulate_scan_rescan,
)

# 1. Enriched 2D Pancreas DRO (fibrotic core + steep perfusion rim + B1/SNR inhomogeneity)
dro_pancreas = generate_dro(
    shape=(64, 64),
    organ="pancreas",
    protocol="osipi",
    snr=30.0,
    b1_inhomogeneity=0.15,
    snr_inhomogeneity=0.20,
    seed=42,
)

# 2. NEX-weighted Fisher CRLB & calibration-aware b-value schedule optimization
opt_sched = optimize_b_schedule(n_b=8, organ="liver", snr=30.0, objective="calibration_aware", total_nex=12)
scheme_report = SchemeEvaluator(organ="liver", snr=30.0).compare(["osipi", "acrin_6698", opt_sched])
print(scheme_report.to_markdown())

# 3. Misspecification stress harness (separating slow D vs. pseudo-diffusion D* degradation)
misspec = evaluate_misspecification(estimators=["segmented", "crlb"], protocol="osipi", snr=30.0)
print(misspec.to_markdown())

# 4. Ground-truth-free Rician scan-rescan repeatability & Spearman width-vs-scatter sizing
pair = simulate_scan_rescan(protocol="osipi", organ="liver", snr=30.0, noise_type="rician", seed=42)
rep = evaluate_repeatability(pair, estimators=["segmented", "crlb"], alpha=0.10)
print(rep.to_markdown())

# 5. Clinical threshold decision VoI & observable domain-shift check
ref = generate_cohort(protocol="osipi", organ="pancreas", n_voxels=400, snr=25.0, seed=42)
tgt = generate_cohort(protocol="osipi", organ="pancreas", n_voxels=400, snr=12.0, family="triexp", seed=99)
est = run_estimator("crlb", ref.signals, ref.bvalues, alpha=0.10, snr=25.0)
voi = evaluate_decision_voi(ref.params, est.lower, est.upper, point=est.point, parameter="f", cost_ratio=0.15)
shift = check_domain_shift(ref.signals, tgt.signals, bvalues=ref.bvalues)
print(voi.to_markdown())
print("Domain shift stale?", shift.is_stale, shift.staleness_reasons)
```

---

## Standardized Acquisition Protocols

`ivimbench.protocols` ships validated open-benchmark diffusion $b$-value schedules ($\text{s/mm}^2$):

| Protocol Key | Title | $b$-values ($\text{s/mm}^2$) | $N_b$ | Reference / Provenance |
| :--- | :--- | :--- | :---: | :--- |
| `osipi` | OSIPI TF2.4 Abdominal IVIM | `0, 10, 20, 30, 40, 50, 75, 100, 150, 250, 400, 600` | 12 | Zenodo [`10.5281/zenodo.14605039`](https://doi.org/10.5281/zenodo.14605039) |
| `osipi_dro` | OSIPI TF2.4 Synthetic DRO | `0, 10, 10, 25, 50, 700, 950` | 7 | Zenodo [`10.5281/zenodo.14605039`](https://doi.org/10.5281/zenodo.14605039) |
| `tcga_lihc` | TCGA-LIHC Clinical Liver 4-$b$ | `0, 50, 500, 800` | 4 | TCIA [`10.7937/K9/TCIA.2016.IMMQW8UQ`](https://doi.org/10.7937/K9/TCIA.2016.IMMQW8UQ) |
| `tcga_lihc_3b` | TCGA-LIHC Sparse Liver 3-$b$ | `50, 400, 800` | 3 | TCIA [`10.7937/K9/TCIA.2016.IMMQW8UQ`](https://doi.org/10.7937/K9/TCIA.2016.IMMQW8UQ) |
| `acrin_6698` | ACRIN 6698 Breast Trial 4-$b$ | `0, 100, 600, 800` | 4 | TCIA [`10.7937/TCIA.2019.CWVLPD26`](https://doi.org/10.7937/TCIA.2019.CWVLPD26) |
| `lattice_22b` | Lattice Dense Research 22-$b$ | `0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000` | 22 | Lattice DRO reference schedule |

---

## Mathematical Formulation

### 1. NEX-Weighted Fisher Information & Cramér–Rao Lower Bounds (`ivimbench.acquisition`)
For voxel parameters $\theta = (D, D^*, f)$ (and optional nuisance amplitude $S_0$), baseline noise $\sigma_0 = S_0 / \text{SNR}$, and per-$b$ signal averages (NEX) $n_k$:

$$I(\theta) = \frac{1}{\sigma_0^2} \sum_{k=1}^{N_b} n_k \, \nabla_\theta S(b_k; \theta) \nabla_\theta S(b_k; \theta)^\top, \qquad \sigma_{\text{CRLB}, j}(\theta) = \sqrt{\left[I(\theta)^{-1}\right]_{jj}}$$

Consequently, $\sigma_{\text{CRLB}} \propto \text{SNR}^{-1} \cdot (\text{NEX})^{-1/2}$. When comparing acquisition schemes $s$ against a reference protocol $s_{\text{ref}}$, `SchemeEvaluator` reports relative interval sharpness $\Delta_{\text{sharp}}(s) = (\bar{W}_{\text{norm}}(s) - \bar{W}_{\text{norm}}(s_{\text{ref}})) / \bar{W}_{\text{norm}}(s_{\text{ref}})$ alongside high-$D^*$ tercile conditional coverage $\Delta_{\text{cond}}(s) = \hat{C}_{\text{high-}D^*}(s) - (1 - \alpha)$.

### 2. Forward Signal Families & Continuity Limits (`ivimbench.phantoms`, `ivimbench.harnesses.misspecification`)
All non-bi-exponential forward generators converge smoothly to the canonical bi-exponential signal as their heterogeneity parameter vanishes:
- **`dispersion_gamma`**: Gamma microvascular velocity spectrum $f (1 + b D^* / k)^{-k} \to f \exp(-b D^*)$ as $k \to \infty$.
- **`dispersion_lognormal`**: Gauss–Hermite log-normal $D^*$ distribution ($\sigma_{\ln D^*} \to 0$).
- **`stretched`**: Kohlrausch–Williams–Watts tissue anomalous diffusion $(1 - f)\exp(-(b D)^\alpha)$ ($\alpha \to 1$).
- **`triexp`**: Three-compartment slow + intermediate + fast vascular flow ($f_{\text{interm}} \to 0$).
- **`kurtosis`**: Non-Gaussian restricted tissue kurtosis $(1 - f)\exp(-b D + \frac{1}{6} b^2 D^2 K_{\text{eff}})$ ($K \to 0$).
- **`heavy_tail`**: Student-$t$ pulsatile/outlier innovations ($\nu \to \infty$, `outlier_scale` $\to 0$).

### 3. Scan–Rescan Repeatability & Spearman Precision Sizing (`ivimbench.harnesses.repeatability`)
Given independent paired scans $(S^{(A)}, S^{(B)})$ without ground-truth labels, `RepeatabilityHarness` computes directional retest capture $\hat{C}_{A \to B} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\{\hat{\theta}_{i,B} \in [\hat{L}_{i,A}, \hat{U}_{i,A}]\}$, symmetric retest capture $\frac{1}{2}(\hat{C}_{A \to B} + \hat{C}_{B \to A})$ (whose exact Gaussian expectation for unbiased $(1-\alpha)$ intervals is $2\Phi(z_{1-\alpha/2}/\sqrt{2}) - 1 \approx 0.755$ at $1-\alpha = 0.90$), within-voxel standard deviation $\text{wSD} = \sqrt{\frac{1}{2N}\sum_i (\hat{\theta}_{i,B} - \hat{\theta}_{i,A})^2}$, repeatability coefficient $\text{RC} = 1.96\sqrt{2}\,\text{wSD}$, and Spearman rank correlation $r_s(\bar{W}_i, |\hat{\theta}_{i,B} - \hat{\theta}_{i,A}|)$ to separate adaptive heteroscedastic sizing from systematic bias.

### 4. Threshold Decision Regret, Second-Order VoI & Domain Shift (`ivimbench.decision`)
For a clinical intervention triggered when $\theta \le \tau$ (or $\theta \ge \tau$) with relative false-positive cost $c \in (0, 1)$ versus false-negative cost $1 - c$, the Bayes-optimal action is $\hat{a}_i(\tau) = \mathbf{1}\{\hat{p}_i(\tau) \ge c\}$ where $\hat{p}_i(\tau) = \hat{\mathbb{P}}(\theta_i \le \tau \mid \hat{L}_i, \hat{U}_i)$. Realized utility regret against the oracle $E_i(\tau) = \mathbf{1}\{\theta_i \le \tau\}$ is:

$$R(\hat{a}; \tau) = \frac{1}{N}\sum_{i=1}^N \Big[ c \, \hat{a}_i (1 - E_i) + (1 - c)(1 - \hat{a}_i) E_i \Big]$$

`evaluate_decision_voi` quantifies:
- **Decision–Calibration Gap**: $|\hat{R}_{\text{perceived}}(\tau) - R(\hat{a}; \tau)|$, measuring how badly an estimator misprices the clinical risk of its own recommended actions.
- **Second-Order Value of Information (`second_order_voi`)**: $R(\hat{a}_{\text{raw}}; \tau) - R(\hat{a}_{\text{cal}}; \tau)$, pricing the expected clinical utility gained purely by calibrating predictive uncertainty.
- **Observable Domain-Shift Check (`check_domain_shift`)**: Computes two-sample regularized Mahalanobis distances, reference-quantile outlier rates, and deterministic `PCG64` permutation $p$-values across normalized signal-decay space and bi-exponential fit residual space.

---

## Repository Structure

```text
IVIMBench/
├── .github/workflows/
│   ├── ci.yml                  # Ruff, Mypy, Pytest & CLI smoke matrix (Python 3.10–3.12)
│   └── publish.yml             # Tagged wheel/sdist build, Twine check & GitHub Release upload
├── pyproject.toml              # PEP 621 metadata, console script & tool configs
├── README.md
├── LICENSE
├── CITATION.cff
├── src/ivimbench/
│   ├── __init__.py
│   ├── _version.py
│   ├── py.typed                # PEP 561 typed package marker
│   ├── cli.py                  # `ivimbench` CLI (`demo`, `benchmark`, `compare-schemes`)
│   ├── registry.py             # `@register_estimator` protocol & built-in registrations
│   ├── runner.py               # Multi-task runner + Markdown/JSON/LaTeX/CSV exporters
│   ├── protocols.py            # Clinical b-schedules (OSIPI TF2.4, TCGA-LIHC, ACRIN 6698)
│   ├── evaluate.py             # Core BenchmarkSuite & BenchmarkReport
│   ├── vis.py                  # Calibration curves & D*-tercile bar plots
│   ├── acquisition/
│   │   ├── crlb.py             # Analytical NEX/scan-time Fisher matrix & CRLB calculator
│   │   └── design.py           # `SchemeEvaluator` & `optimize_b_schedule`
│   ├── harnesses/
│   │   ├── misspecification.py # Non-bi-exponential stress harness (D vs D* separation)
│   │   └── repeatability.py    # Rician / nc-chi scan-rescan harness & Spearman sizing
│   ├── decision/
│   │   └── voi.py              # Threshold utility regret, 2nd-order VoI & domain-shift check
│   ├── phantoms/
│   │   ├── synthetic.py        # Organ priors & multi-compartment forward models
│   │   ├── dro.py              # 2D/3D liver, pancreas & breast DROs + B1/SNR maps
│   │   └── noise.py            # Exact Rician & multi-coil non-central chi injectors
│   ├── metrics/
│   │   ├── coverage.py         # Marginal, SNR- & D*-tercile coverage + conformal wrappers
│   │   ├── sharpness.py        # Absolute/normalized width & CRLB efficiency ratios
│   │   └── railing.py          # Boundary railing fractions & bootstrap CIs
│   └── baselines/              # Segmented, simultaneous NLLS, CRLBNormal & MLP baselines
└── tests/
    ├── test_acquisition.py
    ├── test_decision.py
    ├── test_dro_and_cli.py
    ├── test_harnesses.py
    ├── test_metrics.py
    ├── test_phantoms.py
    ├── test_protocols.py
    └── test_registry_and_runner.py
```

---

## Citation

If you use `IVIMBench`, its digital reference objects, acquisition-design tools, or calibration/decision harnesses in your research, please cite the software and open IVIM standardization benchmarks documented in [`CITATION.cff`](CITATION.cff).
