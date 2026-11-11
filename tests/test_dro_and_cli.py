"""Tests for enriched anatomical DRO presets (liver, pancreas, breast), inhomogeneity maps, and CLI."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from ivimbench import ORGAN_DRO_PRESETS, generate_dro
from ivimbench.cli import main as cli_main


def test_enriched_pancreas_and_breast_dro_presets_and_inhomogeneity() -> None:
    assert {"liver", "pancreas", "breast"}.issubset(set(ORGAN_DRO_PRESETS))

    # 1. Pancreas DRO: concentric steep perfusion rim (label 2) around fibrotic/hypovascular core (label 3)
    panc = generate_dro(
        shape=(48, 48),
        organ="pancreas",
        protocol="organ_default",
        snr=30.0,
        b1_inhomogeneity=0.20,
        snr_inhomogeneity=0.25,
        seed=42,
    )
    assert panc.protocol == "osipi"
    assert panc.lesion_names[2] == "steep_perfusion_rim"
    assert panc.lesion_names[3] == "fibrotic_hypovascular_core"
    assert panc.lesion_names[4] == "pancreatic_pseudocyst"

    rim_mask = panc.region_mask == 2
    core_mask = panc.region_mask == 3
    assert int(np.sum(rim_mask)) > 10 and int(np.sum(core_mask)) > 10
    # Rim perfusion fraction f and D* strictly exceed desmoplastic fibrotic core
    assert float(np.mean(panc.f_map[rim_mask])) > 3.0 * float(np.mean(panc.f_map[core_mask]))
    assert float(np.mean(panc.Dstar_map[rim_mask])) > 2.0 * float(np.mean(panc.Dstar_map[core_mask]))

    # Spatial B1 and SNR inhomogeneity maps are smooth, positive, and non-constant on foreground
    fg = panc.foreground_mask
    assert panc.b1_map.shape == (48, 48)
    assert panc.snr_map.shape == (48, 48)
    assert float(np.std(panc.b1_map[fg])) > 0.02
    assert float(np.std(panc.snr_map[fg])) > 1.0
    assert np.all(panc.snr_map[~fg] == 0.0)

    # Exact PCG64 determinism
    panc_repeat = generate_dro(
        shape=(48, 48),
        organ="pancreas",
        protocol="organ_default",
        snr=30.0,
        b1_inhomogeneity=0.20,
        snr_inhomogeneity=0.25,
        seed=42,
    )
    np.testing.assert_array_equal(panc.signals, panc_repeat.signals)
    np.testing.assert_array_equal(panc.b1_map, panc_repeat.b1_map)
    np.testing.assert_array_equal(panc.snr_map, panc_repeat.snr_map)

    # 2. Breast DRO: organ_default automatically resolves to ACRIN-6698 4-b protocol
    breast = generate_dro(shape=(40, 40), organ="breast", protocol="organ_default", snr=35.0, seed=42)
    assert breast.protocol == "acrin_6698"
    np.testing.assert_array_equal(breast.bvalues, [0.0, 100.0, 600.0, 800.0])
    assert breast.lesion_names[2] == "invasive_ductal_carcinoma"
    assert breast.lesion_names[3] == "benign_fibroadenoma"
    # Malignant IDC exhibits restricted diffusion (lower D) and elevated f vs benign fibroadenoma
    assert float(np.mean(breast.D_map[breast.region_mask == 2])) < float(
        np.mean(breast.D_map[breast.region_mask == 3])
    )
    assert float(np.mean(breast.f_map[breast.region_mask == 2])) > float(
        np.mean(breast.f_map[breast.region_mask == 3])
    )

    # 3. 3D breast DRO with kurtosis forward truth
    breast_3d = generate_dro(
        shape=(20, 20, 10),
        organ="breast",
        protocol="osipi",
        family="kurtosis",
        snr=30.0,
        b1_inhomogeneity=0.15,
        seed=9,
    )
    assert breast_3d.signals.shape == (20, 20, 10, 12)
    assert np.all(np.isfinite(breast_3d.signals))


def test_cli_demo_benchmark_and_compare_schemes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. ivimbench demo completes well under 10s and prints clean terminal tables
    t0 = time.perf_counter()
    rc_demo = cli_main(["demo", "--n-voxels", "60", "--seed", "42"])
    elapsed = time.perf_counter() - t0
    out_demo = capsys.readouterr().out
    assert rc_demo == 0
    assert elapsed < 10.0
    assert "IVIMBench v1.2.0" in out_demo
    assert "pancreas" in out_demo and "breast" in out_demo

    # JSON output mode for ivimbench demo
    rc_json = cli_main(["demo", "--n-voxels", "40", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc_json == 0
    assert payload["version"] == "1.2.0"
    assert payload["domain_shift"]["is_stale"] is True

    # 2. ivimbench benchmark --protocol osipi --snr 30 --out results/
    out_dir = tmp_path / "results"
    rc_bench = cli_main(
        [
            "benchmark",
            "--protocol",
            "osipi",
            "--snr",
            "30",
            "--estimators",
            "segmented,crlb",
            "--n-voxels",
            "40",
            "--out",
            str(out_dir),
        ]
    )
    out_bench = capsys.readouterr().out
    assert rc_bench == 0
    assert "| segmented |" in out_bench
    assert (out_dir / "benchmark_summary.md").exists()
    assert (out_dir / "benchmark_summary.json").exists()
    assert (out_dir / "benchmark_summary.tex").exists()

    # 3. ivimbench compare-schemes --schemes osipi,acrin_6698
    scheme_dir = tmp_path / "schemes"
    rc_comp = cli_main(
        [
            "compare-schemes",
            "--schemes",
            "osipi,acrin_6698",
            "--n-voxels",
            "50",
            "--out",
            str(scheme_dir),
        ]
    )
    out_comp = capsys.readouterr().out
    assert rc_comp == 0
    assert "osipi" in out_comp and "acrin_6698" in out_comp
    assert (scheme_dir / "scheme_comparison.md").exists()
    assert (scheme_dir / "scheme_comparison.json").exists()
    assert (scheme_dir / "scheme_comparison.tex").exists()
