# IVIMBench

**Extensible quantitative-MRI uncertainty calibration, anomalous fractional-order diffusion identifiability, acquisition-design, scan–rescan repeatability, model-misspecification, and closed-loop adaptive radiotherapy benchmark suite for IVIM diffusion MRI.**

[![CI](https://github.com/akarlin3/IVIMBench/actions/workflows/ci.yml/badge.svg)](https://github.com/akarlin3/IVIMBench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Typed: PEP 561](https://img.shields.io/badge/typing-PEP%20561-informational.svg)](src/ivimbench/py.typed)
[![Version 1.5.0](https://img.shields.io/badge/version-1.5.0-green.svg)](pyproject.toml)

`IVIMBench` provides a unified, zero-PHI, purely NumPy/SciPy/Pandas benchmark platform for intravoxel incoherent motion (IVIM) and anomalous fractional-order diffusion MRI:

$$S(b) = S_0 \left[ f \exp(-b D^*) + (1 - f) \exp(-b D) \right], \qquad S_{\text{anom}}(b; S_0, D, \alpha) = S_0 \exp\!\left(-(b D)^\alpha\right)$$

All parameters use canonical physical MRI units across the entire library: diffusion weighting $b$ in $\text{s/mm}^2$, slow tissue diffusivity $D$ and microvascular pseudo-diffusion $D^*$ in $\text{mm}^2/\text{s}$, perfusion fraction $f \in [0, 1]$ (ordered `("D", "Dstar", "f")`, with `D_star` accepted interchangeably), and fractional anomalous stretching exponent $\alpha \in (0, 1]$. Every stochastic phantom, noise realization, bootstrap CI, and permutation test is deterministically seeded with `numpy.random.PCG64` and 100% synthetic/PHI-free.

---

## AveryCorp Scientific Suites Architecture

`IVIMBench` is **Stage 3** of the **AveryCorp Open Scientific Software Ecosystem**, sharing a standardized `AuditReport` diagnostic schema (`ivimbench audit --json` / `ivimbench pipeline --json` / `run_audit()`) across two non-overlapping suites:

- **Suite I: Clinical Quantitative-MRI Trust Spine (`Diagnose -> Calibrate -> Benchmark`)**
  - [`sextantIVIM`](https://github.com/akarlin3/sextantIVIM) (Stage 1): Forward NLLS fitting, Fisher CRLB & assumption-free $D^*$ boundary-railing gate ->
  - [`conformalIVIM`](https://github.com/akarlin3/conformalIVIM) (Stage 2): Split/CQR/Mondrian/ACI finite-sample conformal calibration, cohort shift & voxel OOD gate ->
  - **[`IVIMBench`](https://github.com/akarlin3/IVIMBench) (Stage 3 — *This Package*)**: Canonical DRO phantoms, clean-room Gnomon/Caliper rulers & closed-loop adaptive RT benchmark.
- **Suite II: Scientific Identifiability & Null-Audit Suite**
  - [`weakSINDY`](https://github.com/akarlin3/weakSINDY) (Dynamical & Fractional PDEs) · [`chimeraAnneal`](https://github.com/akarlin3/chimeraAnneal) (Metastable Oscillator Survival) · [`proteusMine`](https://github.com/akarlin3/proteusMine) (Structural Bioinformatics TOST Auditing).

---

## Architecture Overview (v1.5.0)

1. **Publication-Grade Diagnostic Visualization Suite (`ivimbench.vis`)** — Optional `ivimbench[vis]` (`matplotlib>=3.7`) figure generators (`plot_reliability_and_terciles`, `plot_misspecification_aliasing`, `plot_vernier_pareto_frontier`, `plot_dro_slice_montage`, plus legacy `plot_calibration_curves` and `plot_tercile_coverage`).
2. **BIDS-Derivatives (`qMRI`) Exporter & Self-Contained HTML Leaderboard (`ivimbench.reporting`)** — Zero-dependency BIDS-qMRI NIfTI-1 derivatives writer (`export_bids_derivatives`: `sub-<id>_param-{D,Dstar,f}_dwimap.nii.gz`, `sub-<id>_desc-railed_mask.nii.gz`, JSON sidecars, and SVG/HTML QC summary) plus `generate_html_report` emitting a standalone interactive HTML report with sortable leaderboard tables, inline SVG reliability/Pareto plots, and `Gnomon` audit badges.
3. **License-Safe `Trust -> Value -> Action` Ecosystem Bridge (`ivimbench.bridges`)** — End-to-end 4-stage quantitative spine (`run_trust_value_action_pipeline`) dynamically bridging MIT `conformal-ivim` (`ConformalIVIMEstimator` across `split`, `cqr`, and railed/interior `mondrian` modes + `DeploymentValidityMonitor`) and runtime-discovered `sextant-core` (`SextantNLLSEstimator` via `importlib.util.find_spec`, never statically linked in core MIT metadata) with deterministic pure-NumPy internal fallbacks.
4. **Volumetric 3D/4D NIfTI & DICOM Clinical Image I/O (`ivimbench.io`)** — Affine-preserving 4D DWI & FSL `.bval` loader (`load_dwi_volume`), 3D RT-planning & viewer bundle writer (`save_parameter_maps` exporting `D_map`, `Dstar_map`, `f_map`, CI widths, `uint8` `Dstar_railed_mask`, and `exchangeability_report.json`), and `export_dro_to_nifti` with lazy `nibabel`/`pydicom` support (`ivimbench[io]`) plus a built-in pure-NumPy/gzip NIfTI-1.1 reader/writer.
5. **`Gnomon` Clean-Room Verification Harness (`ivimbench.gnomon`)** — Publication-independent reproduce-or-refute harness (`GNOMON_TARGETS`, `run_gnomon_cleanroom_audit`) auditing open-data NLLS $D^*$ boundary-railing rates (`54.7%` target; `[52.0%, 56.4%]` clean-room window), high-$D^*$ tercile symmetric Gaussian (`Laplace-SD`, `MCMC-SD`) under-coverage vs. quantile-interval (`MCMC-quantile`, conformal) recovery, and explicit separation of `honest_crlb` (`~0.80–0.90` marginal) from `floored_railed` (`~0.30–0.67` collapsed-SD artifact).
6. **`Forge` 1.5T MR-Linac 7 MV FFF & Electron Return Effect Physics (`ivimbench.closed_loop.ere_forge`)** — Analytical Elekta Unity-class 7 MV FFF photon spectrum (`generate_unity_7mv_fff_spectrum` with strict zero-proton contamination QA), transverse $1.5\text{ T}$ Lorentz Electron Return Effect boundary perturbation (`compute_ere_boundary_perturbation`), Monte Carlo transport floor estimation (`estimate_mc_simulation_floor`), and `apply_mr_linac_ere=True` integration inside `SyntheticTwinLoop` and `ParameterizedDoseGeometryTwin`.
7. **Anomalous Fractional-Order Diffusion Identifiability & Recovery Wall (`ivimbench.anomalous`)** — Continuous-time random walk (CTRW) / fractional Bloch–Torrey stretched-exponential forward physics (`anomalous_signal`, `fractional_ivim_signal`, `simulate_anomalous_observations`) with exact Rician and multi-coil non-central $\chi$ magnitude noise, analytical $3 \times 3$ Fisher Information Matrix and Cramér–Rao Lower Bounds (`compute_anomalous_crlb`), and `map_recovery_collapse_wall(...)`.
8. **Synthetic-Twin Closed-Loop Adaptive Radiotherapy Harness (`ivimbench.closed_loop`)** — Zero-scanner, 100% synthetic multi-fraction adaptive dosing simulation (`SyntheticTwinLoop`, `compare_closed_loop_policies`) modeling the 6-stage feedback cycle across `SyntheticOrganTwin` and `ParameterizedDoseGeometryTwin`.
9. **Estimator Registry, Acquisition Design, Harnesses & Decision VoI (`ivimbench.registry`, `ivimbench.acquisition`, `ivimbench.harnesses`, `ivimbench.decision`)** — `@register_estimator` protocol, NEX-weighted Fisher CRLB schedule optimization (`optimize_b_schedule`), `MisspecificationHarness`, `RepeatabilityHarness`, threshold decision VoI (`evaluate_decision_voi`), and observable domain-shift checks (`check_domain_shift`).

---

## Installation

Install from GitHub or PyPI (core dependencies remain strictly `numpy`, `scipy`, and `pandas` — zero PyTorch or AGPL dependencies required):

```bash
pip install git+https://github.com/akarlin3/IVIMBench.git
# Or via PyPI:
pip install ivimbench

# Optional 3D/4D clinical NIfTI & DICOM readers (pure-NumPy NIfTI-1 fallback included in core):
pip install "ivimbench[io]"

# Optional MIT conformal-ivim ecosystem bridge:
pip install "ivimbench[ecosystem]"

# Optional matplotlib>=3.7/seaborn publication figure suite:
pip install "ivimbench[vis]"

# Optional PyTorch baselines (IVIMNetBaseline, HeteroscedasticQuantileMLP, MAFPosteriorEstimator):
pip install "ivimbench[torch]"
```

For local development, linting, static type-checking, and testing:

```bash
git clone https://github.com/akarlin3/IVIMBench.git
cd IVIMBench
pip install -e ".[dev]"
pytest -q
ivimbench selfcheck
```

---

## Command-Line Interface (`ivimbench`)

`IVIMBench` installs the `ivimbench` console script for immediate reproducible benchmarking:

```bash
# 1. Generate standalone interactive HTML leaderboard report with inline SVGs & Gnomon badges
ivimbench report --out ./benchmark_report.html

# 2. Export BIDS-Derivatives (qMRI) NIfTI parameter maps, JSON sidecars & QC summary
ivimbench export-bids --subject 01 --out ./bids_derivatives/

# 3. Unified 4-stage `Trust -> Value -> Action` pipeline (DRO -> Fit/Railing -> CQR/Shift -> VoI -> RT Loop)
ivimbench pipeline --organ pancreas --protocol osipi --conformal cqr --fractions 5

# 4. Volumetric 3D/4D NIfTI/DICOM fitting & RT-planning parameter/uncertainty/railed-mask export
ivimbench fit-volume --demo-volume --out results/vol_demo/
ivimbench fit-volume --dwi scan.nii.gz --bval scan.bval --mask roi.nii.gz --out results/maps/

# 5. Clean-room `Gnomon` reproduce-or-refute audit across pinned IVIM uncertainty targets
ivimbench reproduce-gnomon

# 6. `Forge` 1.5T MR-Linac 7 MV FFF photon purity, Lorentz ERE hotspot & MC floor QA check
ivimbench forge-ere-check

# 7. Clean-room continuity-limit residual & noiseless bi-exponential NLLS round-trip self-check
ivimbench selfcheck

# 8. Fast (<10s) end-to-end synthetic DRO + multi-family calibration, repeatability & VoI smoke benchmark
ivimbench demo

# 9. Multi-task benchmark across all 5 forward families, exporting Markdown, JSON, LaTeX & CSV tables
ivimbench benchmark --protocol osipi --snr 30 --out results/

# 10. Compare fixed clinical schedules (and optional CRLB/calibration-optimized designs)
ivimbench compare-schemes --schemes osipi,acrin_6698 --include-optimized

# 11. Print terminal summary table of the fractional anomalous diffusion alpha recovery wall
ivimbench anomalous-wall --protocol osipi --snr-min 10 --snr-max 80

# 12. Run a 5-fraction synthetic-twin closed-loop adaptive RT harness (with optional `--ere` 1.5T mode)
ivimbench closed-loop --fractions 5 --snr 30 --ere
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

## Quickstart 2: Anomalous Fractional Diffusion Wall & Closed-Loop Adaptive RT Twin

Quantify the exact $\text{SNR} \times b_{\max}$ collapse boundary for CTRW fractional-order diffusion $\alpha \in (0, 1]$, and simulate multi-fraction closed-loop adaptive radiotherapy across procedural organ DROs or parameterized 3D dose-volume substrates:

```python
import numpy as np
from ivimbench import (
    ParameterizedDoseGeometryTwin,
    SyntheticOrganTwin,
    SyntheticTwinLoop,
    compare_closed_loop_policies,
    compute_anomalous_crlb,
    map_recovery_collapse_wall,
)

# 1. Analytical 3x3 Fisher CRLB & Monte Carlo recovery collapse wall for S0 * exp(-(b * D)^alpha)
crlb = compute_anomalous_crlb(bvalues=[0, 50, 150, 400, 800, 1200], D=1.2e-3, alpha=0.80, snr=35.0)
wall = map_recovery_collapse_wall(
    b_schedules={"osipi": "osipi", "high_b": [0, 50, 150, 400, 800, 1500]},
    snr_grid=[15.0, 30.0, 60.0],
    alpha_grid=[0.65, 0.85, 1.0],
    seed=42,
)
print(wall.to_markdown())

# 2. Multi-fraction closed-loop RT simulation comparing uncalibrated NLLS vs. conformalized intervals
comparison = compare_closed_loop_policies(
    substrate=SyntheticOrganTwin(shape=(12, 12), seed=42),
    n_fractions=5,
    snr=30.0,
    seed=42,
)
print(comparison.to_markdown())

# 3. Interchangeable 3D ParameterizedDoseGeometryTwin (no anatomical patient blobs required)
dose_grid = np.linspace(2.4, 0.8, 6 * 6 * 6).reshape(6, 6, 6)
organ_mask = np.ones((6, 6, 6), dtype=bool)
twin_3d = ParameterizedDoseGeometryTwin(dose_grid=dose_grid, organ_mask=organ_mask, seed=42)
report_3d = SyntheticTwinLoop(
    substrate=twin_3d, n_fractions=5, snr=30.0, interval_mode="conformal", seed=42
).run()
print(report_3d.to_markdown())
```

---

## Quickstart 3: Acquisition CRLB Design, Repeatability & Decision VoI

```python
from ivimbench import (
    SchemeEvaluator,
    check_domain_shift,
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

### 1. Anomalous Fractional-Order Diffusion Fisher Matrix & Collapse Wall (`ivimbench.anomalous`)
Under the continuous-time random walk (CTRW) / fractional Bloch–Torrey stretched-exponential forward model $S(b; \psi) = S_0 \exp(-(b D)^\alpha)$ with $\psi = (S_0, D, \alpha)$ and $\alpha \in (0, 1]$ (reducing identically to Gaussian mono-exponential ADC at $\alpha = 1$), the closed-form signal sensitivities for $b > 0$ are:

$$\frac{\partial S}{\partial S_0} = \exp\!\left(-(b D)^\alpha\right), \qquad \frac{\partial S}{\partial D} = -S(b) \, \alpha \, b^\alpha D^{\alpha - 1}, \qquad \frac{\partial S}{\partial \alpha} = -S(b) \, (b D)^\alpha \ln(b D)$$

with $\lim_{b \to 0^+} (b D)^\alpha \ln(b D) = 0$. `compute_anomalous_crlb` evaluates the exact $3 \times 3$ Fisher Information Matrix $I(\psi) = \sigma_0^{-2} \sum_k n_k \nabla_\psi S(b_k) \nabla_\psi S(b_k)^\top$, while `map_recovery_collapse_wall` pairs relative CRLB CVs with Monte Carlo bounded-NLLS recovery under Rician or non-central $\chi$ noise and bootstrap 95% CIs to identify the exact $\text{SNR} \times b_{\max}$ regime where $\alpha$ identifiability collapses.

### 2. Synthetic-Twin Closed-Loop Adaptive Radiotherapy (`ivimbench.closed_loop`)
Across fractions $t = 1, \dots, K$, `SyntheticTwinLoop` executes a 6-stage feedback loop on either `SyntheticOrganTwin` or `ParameterizedDoseGeometryTwin`:
1. **Scan**: Acquire multi-$b$ Rician / non-central $\chi$ magnitude signals from current latent tissue parameters $(D_t(\mathbf{x}), D^*_t(\mathbf{x}), f_t(\mathbf{x}))$.
2. **Posterior / Interval Estimation**: Fit voxel-wise parameters and $(1 - \alpha)$ intervals (comparing uncalibrated Fisher/NLLS intervals against studentized split-conformal intervals).
3. **Trust Gate**: Suppress dose escalation when fit degeneracy/railing is detected or when `check_domain_shift` flags longitudinal signal/residual drift beyond tolerance.
4. **Action Gate**: Trigger voxel boost only when the Bayes posterior probability of hypoxic low perfusion $\hat{\mathbb{P}}(f_t(\mathbf{x}) \le \tau_f \mid \hat{L}_t, \hat{U}_t) \ge c$ (while holding OAR voxels at safe sparing multipliers).
5. **Dose Replan & Longitudinal Response**: Apply bounded fractional dose multipliers, evolve microvascular perfusion drop and cellularity clearance, and re-scan at fraction $t + 1$, tracking tumor control proxy, OAR dose sparing, false-escalation rate, and delivered-dose drift on held voxels.

### 3. NEX-Weighted Fisher Information & Cramér–Rao Lower Bounds (`ivimbench.acquisition`)
For voxel parameters $\theta = (D, D^*, f)$ (and optional nuisance amplitude $S_0$), baseline noise $\sigma_0 = S_0 / \text{SNR}$, and per-$b$ signal averages (NEX) $n_k$:

$$I(\theta) = \frac{1}{\sigma_0^2} \sum_{k=1}^{N_b} n_k \, \nabla_\theta S(b_k; \theta) \nabla_\theta S(b_k; \theta)^\top, \qquad \sigma_{\text{CRLB}, j}(\theta) = \sqrt{\left[I(\theta)^{-1}\right]_{jj}}$$

Consequently, $\sigma_{\text{CRLB}} \propto \text{SNR}^{-1} \cdot (\text{NEX})^{-1/2}$. When comparing acquisition schemes $s$ against a reference protocol $s_{\text{ref}}$, `SchemeEvaluator` reports relative interval sharpness $\Delta_{\text{sharp}}(s) = (\bar{W}_{\text{norm}}(s) - \bar{W}_{\text{norm}}(s_{\text{ref}})) / \bar{W}_{\text{norm}}(s_{\text{ref}})$ alongside high-$D^*$ tercile conditional coverage $\Delta_{\text{cond}}(s) = \hat{C}_{\text{high-}D^*}(s) - (1 - \alpha)$.

### 4. Forward Signal Families & Continuity Limits (`ivimbench.phantoms`, `ivimbench.harnesses.misspecification`)
All non-bi-exponential forward generators converge smoothly to the canonical bi-exponential signal as their heterogeneity parameter vanishes:
- **`dispersion_gamma`**: Gamma microvascular velocity spectrum $f (1 + b D^* / k)^{-k} \to f \exp(-b D^*)$ as $k \to \infty$.
- **`dispersion_lognormal`**: Gauss–Hermite log-normal $D^*$ distribution ($\sigma_{\ln D^*} \to 0$).
- **`stretched`**: Kohlrausch–Williams–Watts tissue anomalous diffusion $(1 - f)\exp(-(b D)^\alpha)$ ($\alpha \to 1$).
- **`triexp`**: Three-compartment slow + intermediate + fast vascular flow ($f_{\text{interm}} \to 0$).
- **`kurtosis`**: Non-Gaussian restricted tissue kurtosis $(1 - f)\exp(-b D + \frac{1}{6} b^2 D^2 K_{\text{eff}})$ ($K \to 0$).
- **`heavy_tail`**: Student-$t$ pulsatile/outlier innovations ($\nu \to \infty$, `outlier_scale` $\to 0$).

### 5. Scan–Rescan Repeatability & Spearman Precision Sizing (`ivimbench.harnesses.repeatability`)
Given independent paired scans $(S^{(A)}, S^{(B)})$ without ground-truth labels, `RepeatabilityHarness` computes directional retest capture $\hat{C}_{A \to B} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\{\hat{\theta}_{i,B} \in [\hat{L}_{i,A}, \hat{U}_{i,A}]\}$, symmetric retest capture $\frac{1}{2}(\hat{C}_{A \to B} + \hat{C}_{B \to A})$ (whose exact Gaussian expectation for unbiased $(1-\alpha)$ intervals is $2\Phi(z_{1-\alpha/2}/\sqrt{2}) - 1 \approx 0.755$ at $1-\alpha = 0.90$), within-voxel standard deviation $\text{wSD} = \sqrt{\frac{1}{2N}\sum_i (\hat{\theta}_{i,B} - \hat{\theta}_{i,A})^2}$, repeatability coefficient $\text{RC} = 1.96\sqrt{2}\,\text{wSD}$, and Spearman rank correlation $r_s(\bar{W}_i, |\hat{\theta}_{i,B} - \hat{\theta}_{i,A}|)$ to separate adaptive heteroscedastic sizing from systematic bias.

### 6. Threshold Decision Regret, Second-Order VoI & Domain Shift (`ivimbench.decision`)
For a clinical intervention triggered when $\theta \le \tau$ (or $\theta \ge \tau$) with relative false-positive cost $c \in (0, 1)$ versus false-negative cost $1 - c$, the Bayes-optimal action is $\hat{a}_i(\tau) = \mathbf{1}\{\hat{p}_i(\tau) \ge c\}$ where $\hat{p}_i(\tau) = \hat{\mathbb{P}}(\theta_i \le \tau \mid \hat{L}_i, \hat{U}_i)$. Realized utility regret against the oracle $E_i(\tau) = \mathbf{1}\{\theta_i \le \tau\}$ is:

$$R(\hat{a}; \tau) = \frac{1}{N}\sum_{i=1}^N \Big[ c \, \hat{a}_i (1 - E_i) + (1 - c)(1 - \hat{a}_i) E_i \Big]$$

`evaluate_decision_voi` quantifies:
- **Decision–Calibration Gap**: $|\hat{R}_{\text{perceived}}(\tau) - R(\hat{a}; \tau)|$, measuring how badly an estimator misprices the clinical risk of its own recommended actions.
- **Second-Order Value of Information (`second_order_voi`)**: $R(\hat{a}_{\text{raw}}; \tau) - R(\hat{a}_{\text{cal}}; \tau)$, pricing the expected clinical utility gained purely by calibrating predictive uncertainty.
- **Observable Domain-Shift Check (`check_domain_shift`)**: Computes two-sample regularized Mahalanobis distances, reference-quantile outlier rates, and deterministic `PCG64` permutation $p$-values across normalized signal-decay space and bi-exponential fit residual space.

### 7. Plug-and-Play Neural/PINN Adapters, Dual-SD Conventions & Self-Checks (`ivimbench.baselines`, `ivimbench.metrics`, `ivimbench.phantoms`)
Optional PyTorch neural baselines (`pip install "ivimbench[torch]"`) are registered under `@register_estimator` while keeping core `ivimbench` strictly pure NumPy/SciPy:
- **`IVIMNetBaseline` (`"ivimnet"`)**: Physics-informed self-supervised bi-exponential network minimizing Gaussian or first-moment Rician signal reconstruction loss $\mathcal{L}(\hat{S}(b; \hat{D}, \hat{D}^*, \hat{f}), y(b))$ with sigmoid box-constrained parameter heads and combined MC-Dropout + heteroscedastic predictive quantiles.
- **`HeteroscedasticQuantileMLP` (`"quantile_mlp"`)**: Direct pinball-loss quantile regression network predicting ordered $(\hat{q}_{\alpha/2}, \hat{q}_{0.5}, \hat{q}_{1-\alpha/2})$ for $(D, D^*, f)$, outputting full `(n_voxels, 3, 3)` quantile tensors directly compatible with `SplitConformalQuantile` and `conformalIVIM`.
- **`MAFPosteriorEstimator`**: Conditional Masked Autoregressive Flow posterior density estimator.

Bounded IVIM solvers (`NLLSEstimator`, `CRLBNormalEstimator`, `crlb_normal_intervals`, `BenchmarkSuite`, and `ivimbench.metrics.coverage`) also expose `sd_convention: Literal["honest_crlb", "floored_railed"] = "honest_crlb"`, and clean-room physics verification helpers (`verify_continuity_limits()`, `run_nlls_roundtrip_selfcheck()`, and `ivimbench selfcheck`) verify machine-precision continuity limits and `<= 1e-9` noiseless NLLS parameter recovery.

```python
from ivimbench import (
    CRLBNormalEstimator,
    HeteroscedasticQuantileMLP,  # requires `pip install "ivimbench[torch]"`
    IVIMNetBaseline,             # requires `pip install "ivimbench[torch]"`
    MAFPosteriorEstimator,       # requires `pip install "ivimbench[torch]"`
    run_nlls_roundtrip_selfcheck,
    verify_continuity_limits,
)

# Verify analytical continuity limits and noiseless NLLS round-trip accuracy
limits = verify_continuity_limits()
roundtrip = run_nlls_roundtrip_selfcheck()

# Compare honest Fisher/CRLB SDs vs. legacy floored boundary-railed SDs
est_honest = CRLBNormalEstimator(sd_convention="honest_crlb")
est_floored = CRLBNormalEstimator(sd_convention="floored_railed")
```

---

## Quickstart 4: Unified `Trust -> Value -> Action` Spine, NIfTI/DICOM Volume I/O, `Gnomon` & `Forge` ERE

```python
import numpy as np
from ivimbench import (
    compute_ere_boundary_perturbation,
    export_dro_to_nifti,
    generate_dro,
    generate_unity_7mv_fff_spectrum,
    load_dwi_volume,
    run_estimator,
    run_gnomon_cleanroom_audit,
    run_trust_value_action_pipeline,
    save_parameter_maps,
)

# 1. Unified 4-stage `Trust -> Value -> Action` pipeline (MIT-safe across Split / CQR / Mondrian)
report = run_trust_value_action_pipeline(
    protocol="osipi", organ="pancreas", snr=30, conformal_method="cqr", alpha=0.10, fractions=5, seed=42
)
print(report.format_summary_table())

# 2. Affine-preserving 3D/4D NIfTI volume export -> load -> vectorized fit -> RT-planning map bundle
dro_3d = generate_dro(shape=(12, 12, 8), organ="pancreas", protocol="osipi", snr=30.0, seed=42)
exported = export_dro_to_nifti(dro_3d, "results/dro_nifti/")
vol = load_dwi_volume(exported["dwi"], exported["bval"], mask_path=exported["roi_mask"])
fit_out = run_estimator("conformal_ivim", vol.signals, vol.bvalues, alpha=0.10)
written_maps = save_parameter_maps("results/fitted_maps/", fit_out, vol)

# 3. Publication-independent `Gnomon` clean-room reproduce-or-refute audit
gnomon_verdict = run_gnomon_cleanroom_audit(n_voxels=2000, seed=42)
print(gnomon_verdict.format_summary_table())

# 4. `Forge` 1.5T MR-Linac 7 MV FFF photon spectrum & transverse Lorentz ERE boundary perturbation
spectrum = generate_unity_7mv_fff_spectrum(n_bins=64)
dose_3d = np.full((20, 16, 16), 2.0)
density_3d = np.ones((20, 16, 16))
density_3d[8:13, :, :] = 0.0012  # Internal air cavity
ere = compute_ere_boundary_perturbation(dose_3d, density_3d, b_field_tesla=1.5, beam_axis=0)
print(f"Peak 1.5T water->air exit hotspot: +{ere.peak_exit_enhancement_pct:.2f}%")
```

---

## Quickstart 5: Publication Figures (`[vis]`), BIDS-Derivatives (`qMRI`) & HTML Leaderboard

```python
from ivimbench import (
    SchemeEvaluator,
    evaluate_misspecification,
    export_bids_derivatives,
    generate_cohort,
    generate_dro,
    generate_html_report,
    plot_dro_slice_montage,
    plot_misspecification_aliasing,
    plot_reliability_and_terciles,
    plot_vernier_pareto_frontier,
    run_benchmark_suite,
    run_estimator,
)

# 1. Publication diagnostic figures (pip install "ivimbench[vis]")
cohort = generate_cohort(protocol="osipi", organ="liver", n_voxels=300, snr=30.0, seed=42)
plot_reliability_and_terciles(cohort, out_path="figures/reliability_terciles.png")

misspec = evaluate_misspecification(estimators=["segmented", "crlb"], protocol="osipi", snr=30.0)
plot_misspecification_aliasing(misspec, out_path="figures/misspec_aliasing.png")

schemes = SchemeEvaluator(organ="liver", snr=30.0).compare(["osipi", "acrin_6698"], include_optimized=True)
plot_vernier_pareto_frontier(schemes, out_path="figures/vernier_pareto.png")

dro = generate_dro(shape=(32, 32, 8), organ="pancreas", protocol="osipi", snr=30.0, seed=42)
plot_dro_slice_montage(dro, out_path="figures/dro_montage.png")

# 2. BIDS-Derivatives (qMRI) NIfTI-1 + JSON sidecars + SVG/HTML QC summary (zero external deps)
sig_fg, _ = dro.to_voxels(foreground_only=True)
fit_res = run_estimator("segmented", sig_fg, dro.bvalues, snr=30.0)
export_bids_derivatives(
    output_dir="./bids_derivatives",
    subject_id="01",
    fit_result=fit_res,
    volume_meta={"mask": dro.foreground_mask, "shape": dro.spatial_shape, "bvalues": dro.bvalues},
    protocol_name="osipi",
)

# 3. Standalone interactive HTML leaderboard with Gnomon audit badges & inline SVGs
bench = run_benchmark_suite(estimators=["segmented", "crlb"], protocols=["osipi", "acrin_6698"])
generate_html_report(bench, out_path="ivimbench_report.html")
```

---

## Repository Structure

```text
IVIMBench/
├── .github/workflows/
│   ├── ci.yml                              # Ruff, Mypy, Pytest & CLI smoke matrix (Python 3.10–3.12)
│   └── publish.yml                         # Tagged wheel/sdist build, Twine check & GitHub Release upload
├── pyproject.toml                          # PEP 621 metadata, optional `[io]`, `[ecosystem]`, `[vis]`, `[torch]` extras
├── README.md
├── LICENSE
├── CITATION.cff
├── src/ivimbench/
│   ├── __init__.py
│   ├── _version.py
│   ├── py.typed                            # PEP 561 typed package marker
│   ├── cli.py                              # `ivimbench` CLI (`report`, `export-bids`, `pipeline`, `fit-volume`, `reproduce-gnomon`, ...)
│   ├── osipi.py                            # Network-isolated MD5-verified OSIPI TF2.4 Zenodo DRO fetcher
│   ├── registry.py                         # `@register_estimator` protocol & built-in registrations
│   ├── runner.py                           # Multi-task runner + Markdown/JSON/LaTeX/CSV exporters
│   ├── protocols.py                        # Clinical b-schedules (OSIPI TF2.4, TCGA-LIHC, ACRIN 6698)
│   ├── evaluate.py                         # Core BenchmarkSuite & BenchmarkReport (`sd_convention` aware)
│   ├── bridges/
│   │   ├── __init__.py
│   │   └── avery_stack.py                  # License-safe `ConformalIVIMEstimator`, `SextantNLLSEstimator` & 4-stage pipeline
│   ├── io/
│   │   ├── __init__.py
│   │   └── volumes.py                      # Affine-preserving 3D/4D NIfTI-1 & DICOM clinical volume I/O
│   ├── gnomon/
│   │   ├── __init__.py
│   │   ├── manifest.py                     # Frozen `GNOMON_TARGETS` verification manifest
│   │   └── reproduce.py                    # Clean-room NLLS/CRLB/Laplace/MCMC/conformal audit harness
│   ├── vis/                                # Publication figure suite (`ivimbench[vis]`, `matplotlib>=3.7`)
│   │   ├── __init__.py
│   │   ├── calibration.py                  # Reliability curves & `plot_reliability_and_terciles`
│   │   └── publication.py                  # Misspec aliasing, Vernier Pareto frontier & DRO slice montage
│   ├── reporting/                          # Zero-dependency BIDS-qMRI NIfTI exporter & HTML leaderboard
│   │   ├── __init__.py
│   │   ├── bids.py                         # `export_bids_derivatives` + pure-NumPy NIfTI-1 I/O
│   │   └── html_report.py                  # `generate_html_report` + Gnomon audit badges & inline SVGs
│   ├── anomalous/
│   │   ├── __init__.py
│   │   └── wall.py                         # CTRW stretched-exp physics, 3x3 Fisher CRLB & collapse wall
│   ├── closed_loop/
│   │   ├── __init__.py
│   │   ├── ere_forge.py                    # `Forge` 1.5T 7 MV FFF photon spectrum, Lorentz ERE & MC floor
│   │   ├── substrates.py                   # `SyntheticOrganTwin` & `ParameterizedDoseGeometryTwin`
│   │   └── loop.py                         # 6-stage `SyntheticTwinLoop` & policy comparison harness
│   ├── acquisition/
│   │   ├── crlb.py                         # Analytical NEX/scan-time Fisher matrix & CRLB calculator
│   │   └── design.py                       # `SchemeEvaluator` & `optimize_b_schedule`
│   ├── harnesses/
│   │   ├── misspecification.py             # Non-bi-exponential stress harness (D vs D* separation)
│   │   └── repeatability.py                # Rician / nc-chi scan-rescan harness & Spearman sizing
│   ├── decision/
│   │   └── voi.py                          # Threshold utility regret, 2nd-order VoI & domain-shift check
│   ├── phantoms/
│   │   ├── synthetic.py                    # Organ priors, `Cohort`/`to_scorer_inputs()` & forward models
│   │   ├── selfcheck.py                    # Continuity-limit & noiseless NLLS round-trip diagnostics
│   │   ├── dro.py                          # 2D/3D liver, pancreas & breast DROs + B1/SNR maps
│   │   └── noise.py                        # Exact Rician & multi-coil non-central chi injectors
│   ├── metrics/
│   │   ├── coverage.py                     # Marginal/tercile coverage, `sd_convention` & conformal wrappers
│   │   ├── sharpness.py                    # Absolute/normalized width & CRLB efficiency ratios
│   │   └── railing.py                      # Boundary railing fractions & bootstrap CIs
│   └── baselines/                          # Segmented, NLLS, CRLBNormal, MLP, `neural.py` (`IVIMNet`, QuantileMLP) & MAF
└── tests/
    ├── test_acquisition.py
    ├── test_anomalous_and_closed_loop.py
    ├── test_decision.py
    ├── test_dro_and_cli.py
    ├── test_harnesses.py
    ├── test_metrics.py
    ├── test_option2_ecosystem_io_gnomon_forge.py
    ├── test_phantoms.py
    ├── test_protocols.py
    ├── test_registry_and_runner.py
    ├── test_selfcheck_and_conventions.py
    └── test_vis_bids_html.py
```

---

## Citation

If you use `IVIMBench`, its digital reference objects, anomalous diffusion identifiability walls, closed-loop synthetic twins, or calibration/decision harnesses in your research, please cite the software and open IVIM standardization benchmarks documented in [`CITATION.cff`](CITATION.cff).
