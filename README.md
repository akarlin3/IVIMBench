# IVIMBench

**Standardized digital reference phantoms, clinical acquisition schedules, and uncertainty calibration benchmarks for quantitative IVIM MRI.**

[![CI](https://github.com/akarlin3/IVIMBench/actions/workflows/ci.yml/badge.svg)](https://github.com/akarlin3/IVIMBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Typed: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](src/ivimbench/py.typed)
[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

`IVIMBench` consolidates synthetic phantom generation (`Lattice`), benchmark protocols and submission harnesses (`Datum`), and distribution-free uncertainty quantification (UQ) calibration metrics (`Caliper`) into a unified, lightweight Python evaluation suite for intravoxel incoherent motion (IVIM) diffusion MRI:

$$S(b) = S_0 \left[ f \exp(-b D^*) + (1 - f) \exp(-b D) \right]$$

All parameters use canonical physical MRI units across the entire library: diffusion weighting $b$ in $\text{s/mm}^2$, tissue diffusivity $D$ and pseudo-diffusion $D^*$ in $\text{mm}^2/\text{s}$, and microvascular perfusion fraction $f \in [0, 1]$ (ordered `("D", "Dstar", "f")`). All generated cohorts and digital reference objects (DROs) are 100% synthetic and PHI-free.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/akarlin3/IVIMBench.git

# With optional visualization dependencies (matplotlib, seaborn)
pip install "ivimbench[vis] @ git+https://github.com/akarlin3/IVIMBench.git"
```

Or via PyPI:

```bash
pip install ivimbench

# Optional visualization dependencies (matplotlib, seaborn)
pip install "ivimbench[vis]"
```

For local development and testing:

```bash
git clone https://github.com/akarlin3/IVIMBench.git
cd IVIMBench
pip install -e ".[dev]"
pytest
```

---

## Quickstart: Benchmarking Custom Neural Network Outputs on OSIPI TF2.4

Generate a reproducible 10,000-voxel hepatic cohort on the **OSIPI Taskforce 2.4** clinical protocol and score any custom neural network's point estimates and $(1 - \alpha) = 90\%$ prediction intervals in a few lines:

```python
import numpy as np
from ivimbench import (
    BenchmarkSuite,
    generate_cohort,
    plot_calibration_curves,
    plot_tercile_coverage,
)

# 1. One-line synthetic cohort on the OSIPI TF2.4 12-b schedule (SNR = 30, Rician noise)
cohort = generate_cohort(
    protocol="osipi",
    organ="liver",
    n_voxels=10000,
    snr=30,
    seed=42,
)
signals, ground_truth = cohort  # (10000, 12) DWI signals and (10000, 3) [D, D*, f]

# 2. Plug in predictions from your custom neural network (point estimates + 90% intervals)
#    Replace with your PyTorch / JAX / TensorFlow model's inference outputs:
rng = np.random.default_rng(123)
nn_point = np.clip(
    ground_truth + rng.normal(0.0, [1.2e-4, 1.4e-2, 0.03], size=ground_truth.shape),
    a_min=[1.0e-4, 3.0e-3, 0.0],
    a_max=[4.0e-3, 0.20, 0.70],
)
nn_std = np.broadcast_to([1.3e-4, 1.5e-2, 0.032], ground_truth.shape)
z_90 = 1.6448536
nn_lower, nn_upper = nn_point - z_90 * nn_std, nn_point + z_90 * nn_std

# 3. Run the unified UQ calibration, CRLB sharpness, and boundary-railing harness
suite = BenchmarkSuite(protocol="osipi", snr=30, alpha=0.10, seed=42)
report = suite.evaluate(
    cohort,
    point_pred=nn_point,
    intervals=(nn_lower, nn_upper),
    method="CustomIVIMNet",
)

print(report.format_table())
df = report.to_dataframe()  # Tidy pandas DataFrame across ("D", "Dstar", "f")

# 4. Compare directly against built-in reference baselines (Segmented, NLLS, StandardMLP)
baseline_df = suite.run_baselines(cohort, include=("segmented", "nlls", "mlp"))

# 5. Optional diagnostic plots (requires `ivimbench[vis]`)
fig_cal, _ = plot_calibration_curves(cohort, {"CustomIVIMNet": (nn_point, nn_std)})
fig_ter, _ = plot_tercile_coverage(report, parameter="Dstar")
```

`BenchmarkSuite.evaluate` natively accepts:
1. **Explicit central intervals**: `point_pred` + `intervals=(lower, upper)` (or `lower=..., upper=...`),
2. **Gaussian error bars**: `point_pred` + `std_pred`,
3. **Quantile grids / posterior percentiles**: `q_pred` of shape `(n_voxels, 3, n_quantiles)` + `q_levels`, or
4. **Estimator objects**: `estimator=...` implementing `predict_quantiles(signals, q_levels)` or `predict_intervals(signals, alpha)`.

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

Custom 1D NumPy arrays or Python sequences of $b$-values can also be passed anywhere `protocol` is accepted.

---

## Organ Priors, Forward Families, and Spatial DROs

### Organ Parameter Priors (`ivimbench.phantoms.synthetic`)
Physiologically grounded parameter ranges with Gaussian copula inter-parameter correlations (`prior="realistic"`, log-uniform $D^*$ marginals) or independent uniform/log-uniform priors:

| Organ | $D$ ($\times 10^{-3}\text{ mm}^2/\text{s}$) | $D^*$ ($\times 10^{-3}\text{ mm}^2/\text{s}$) | $f$ |
| :--- | :---: | :---: | :---: |
| `liver` | $[0.6, 1.8]$ | $[25.0, 120.0]$ | $[0.12, 0.40]$ |
| `breast` | $[0.6, 2.4]$ | $[10.0, 60.0]$ | $[0.03, 0.22]$ |
| `brain` | $[0.5, 1.5]$ | $[8.0, 45.0]$ | $[0.02, 0.12]$ |
| `pancreas` | $[0.8, 2.2]$ | $[15.0, 90.0]$ | $[0.10, 0.38]$ |
| `generic` | $[0.5, 3.0]$ | $[10.0, 100.0]$ | $[0.05, 0.40]$ |

Five forward signal generators (`family="biexp"`, `"dispersion_gamma"`, `"dispersion_lognormal"`, `"stretched"`, `"triexp"`) enable controlled model-misspecification stress testing, each reducing to `biexp` at its analytical continuity limit.

### Spatial Digital Reference Objects (`ivimbench.phantoms.dro`)
`generate_dro(shape=(64, 64), organ="liver", protocol="osipi", snr=30, seed=42)` synthesizes 2D or 3D structured anatomical phantoms with spatially correlated parenchymal fields and embedded hypervascular, hypovascular, and necrotic focal lesions.

### Exact Magnitude Noise Models (`ivimbench.phantoms.noise`)
- **Rician (`noise_type="rician"`)**: Single-coil quadrature magnitude $\sqrt{(S + \varepsilon_{\text{re}})^2 + \varepsilon_{\text{im}}^2}$ with $\sigma = S_0 / \text{SNR}$.
- **Non-central $\chi$ (`noise_type="noncentral_chi"`, `n_coils=L`)**: Multi-coil root-sum-of-squares (SoS) magnitude across $L$ channels ($2L$ degrees of freedom), reducing bit-for-bit to Rician noise at `n_coils=1`.

---

## Evaluation Metrics & CRLB Efficiency Ratio

### 1. Marginal & Conditional Coverage (`ivimbench.metrics.coverage`)
- **Marginal Coverage & Gap**: $\hat{C}(1-\alpha) = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\{y_i \in [\hat{L}_i, \hat{U}_i]\}$ and $\Delta C = \hat{C}(1-\alpha) - (1-\alpha)$ with non-parametric bootstrap CIs.
- **$D^*$-Tercile Conditional Coverage**: Stratifies empirical coverage across low, mid, and high ground-truth $D^*$ terciles (`tercile_cov_low`, `tercile_cov_mid`, `tercile_cov_high`) to expose local under-coverage above the pseudo-diffusion identifiability wall even when marginal coverage appears nominal.
- **SNR-Stratified Coverage & Proper Scoring Rules**: Quantile Expected Calibration Error (`ece_quantile`), pinball loss, Winkler interval score, and post-hoc conformal wrappers (`SplitConformalResidual`, `SplitConformalQuantile`, `MondrianConformalQuantile`).

### 2. Sharpness & Cramér-Rao Lower Bound Efficiency (`ivimbench.metrics.sharpness`)
At ground-truth voxel parameters $\theta_i = (D_i, D^*_i, f_i)$ and noise scale $\sigma_i = S_0 / \text{SNR}_i$, the Gaussian Fisher information matrix is:

$$I(\theta_i) = \frac{1}{\sigma_i^2} J(\theta_i)^\top W J(\theta_i), \qquad \sigma_{\text{CRLB}, i, j} = \sqrt{\left[I(\theta_i)^{-1}\right]_{jj}}$$

where $J(\theta_i) = \partial S(b; \theta_i) / \partial \theta_i$ is the analytical $N_b \times 3$ bi-exponential Jacobian and $W = \text{diag}(n_{\text{avg}})$ weights repeated excitations. The minimum asymptotic $(1 - \alpha)$ Gaussian interval width attainable by any locally unbiased estimator is:

$$W_{\text{CRLB}, i, j}(1 - \alpha) = 2 \, z_{1 - \alpha/2} \, \sigma_{\text{CRLB}, i, j}$$

Comparing predicted interval widths $\hat{W}_{i,j} = \hat{U}_{i,j} - \hat{L}_{i,j}$ against $W_{\text{CRLB}, i, j}$ across identifiable voxels defines two reciprocal summary metrics:

1. **Sharpness-to-CRLB Ratio (`sharpness_to_crlb_ratio`)**:
   $$\rho_{\text{CRLB}, j} = \operatorname{median}_i \left( \frac{\hat{W}_{i,j}}{W_{\text{CRLB}, i, j}(1 - \alpha)} \right)$$
   Equals `1.0` for an asymptotically efficient unbiased Gaussian interval. Values `> 1.0` quantify interval inflation relative to the information-theoretic floor; values `< 1.0` indicate informative Bayesian/learned prior shrinkage (if calibrated) or over-confident under-dispersion (if under-covering).
2. **Classical CRLB Efficiency (`crlb_efficiency`)**:
   $$\eta_{\text{CRLB}, j} = \operatorname{median}_i \left( \frac{W_{\text{CRLB}, i, j}(1 - \alpha)}{\hat{W}_{i,j}} \right)$$
   Bounded in $(0, 1]$ for unbiased calibrated estimators (`1.0` = optimal Cramér-Rao efficiency).

### 3. Boundary Railing Diagnostics (`ivimbench.metrics.railing`)
Flags point estimates pinned within `rail_tol * (upper_bound - lower_bound)` (default `1%` of box span) of optimization or physiological boundaries (`DEFAULT_PARAM_BOUNDS`), reporting lower, upper, and total railing percentages alongside chunked percentile bootstrap 95% CIs (`railing_pct`, `railing_ci_lo_pct`, `railing_ci_hi_pct`).

---

## Reproducible Random Seeds

Every stochastic operation in `IVIMBench` uses explicit `numpy.random.Generator` instances (`PCG64`) seeded from integer seeds (default `seed=42`) with zero reliance on global `np.random` state:

- **`generate_cohort(..., seed=s)`** decouples RNG streams via fixed integer offsets:
  - `s + 0`: Base physiological parameter sampling $(D, D^*, f)$ (`sample_organ_params`).
  - `s + 4040`: Auxiliary voxel-level SNR grid assignment when `snr` is passed as a multi-SNR schedule.
  - `s + 7000`: Thermal Rician / non-central $\chi$ / Gaussian noise draws (`inject_noise`).
  - Changing `snr` or `noise_type` while keeping `seed=42` preserves the exact underlying ground-truth parameter cohort $(D, D^*, f)$ byte-for-bit.
- **`generate_dro(..., seed=s)`** uses `s + 101` for smooth spatial Gaussian random fields and `s + 7000` for magnitude noise injection.
- **`BenchmarkSuite(..., seed=s)`** seeds parameter-wise bootstrap confidence intervals deterministically at `s + j` (railing CIs) and `s + 100 + j` (coverage CIs).

---

## Repository Structure

```text
IVIMBench/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/
│   └── ivimbench/
│       ├── __init__.py
│       ├── phantoms/
│       │   ├── synthetic.py    # Organ parameter priors & 5 forward signal families
│       │   ├── dro.py          # 2D/3D spatial Digital Reference Objects with lesions
│       │   └── noise.py        # Exact Rician & multi-coil non-central chi injectors
│       ├── protocols.py        # Clinical b-value schedules (OSIPI TF2.4, TCGA-LIHC, ACRIN 6698)
│       ├── metrics/
│       │   ├── coverage.py     # Marginal, SNR- & D*-tercile coverage + conformal wrappers
│       │   ├── sharpness.py    # Absolute/normalized width & CRLB efficiency ratios
│       │   └── railing.py      # Boundary railing fractions & percentile bootstrap CIs
│       ├── baselines/          # Reference estimators (NLLS, Segmented, StandardMLP)
│       ├── vis.py              # Calibration curves & D*-tercile coverage bar plots
│       └── evaluate.py         # Unified BenchmarkSuite harness
└── tests/
    ├── test_phantoms.py
    ├── test_protocols.py
    └── test_metrics.py
```

---

## Citation

If you use `IVIMBench`, its digital reference objects, or its calibration benchmark suite in your research, please cite the software and underlying open IVIM standardization benchmarks documented in [`CITATION.cff`](CITATION.cff).
