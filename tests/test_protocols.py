"""Unit tests for clinical diffusion acquisition protocols and b-value resolution."""

from __future__ import annotations

import numpy as np
import pytest

from ivimbench.protocols import (
    PROTOCOLS,
    AcquisitionProtocol,
    get_protocol,
    list_protocols,
    resolve_bvalues,
)


def test_registered_clinical_protocols_and_exact_schedules() -> None:
    keys = list_protocols()
    for expected in ("osipi", "osipi_dro", "tcga_lihc", "tcga_lihc_3b", "acrin_6698", "lattice_22b"):
        assert expected in keys

    osipi = get_protocol("osipi")
    assert osipi.bvalues == (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 250.0, 400.0, 600.0)
    assert osipi.n_b == 12
    assert osipi.has_b0 is True
    assert osipi.supports_segmented_fit() is True
    assert osipi.doi == "10.5281/zenodo.14605039"

    lihc = get_protocol("tcga_lihc")
    assert lihc.bvalues == (0.0, 50.0, 500.0, 800.0)
    assert lihc.has_b0 is True
    assert lihc.supports_segmented_fit() is False  # Only one non-zero low-b <= 50

    lihc_3b = get_protocol("tcga_lihc_3b")
    assert lihc_3b.bvalues == (50.0, 400.0, 800.0)
    assert lihc_3b.has_b0 is False

    acrin = get_protocol("acrin_6698")
    assert acrin.bvalues == (0.0, 100.0, 600.0, 800.0)
    assert acrin.organ == "breast"
    assert acrin.doi == "10.7937/TCIA.2019.CWVLPD26"

    dro_sched = get_protocol("osipi_dro")
    assert dro_sched.n_b == 7
    assert dro_sched.n_unique_b == 6  # Repeated b = 10 s/mm^2


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("OSIPI TF2.4", "osipi"),
        ("osipi-tf24", "osipi"),
        ("TCGA-LIHC", "tcga_lihc"),
        ("lihc_3b", "tcga_lihc_3b"),
        ("ACRIN 6698", "acrin_6698"),
        ("acrin", "acrin_6698"),
        ("dense-22b", "lattice_22b"),
    ],
)
def test_protocol_alias_normalization(alias: str, canonical: str) -> None:
    resolved = get_protocol(alias)
    assert resolved.key == canonical
    assert resolved == PROTOCOLS[canonical]


def test_resolve_bvalues_accepts_protocol_names_objects_and_custom_arrays() -> None:
    b_str, name_str = resolve_bvalues("OSIPI TF2.4")
    assert name_str == "osipi"
    assert b_str.shape == (12,)

    proto_obj = get_protocol("acrin_6698")
    b_obj, name_obj = resolve_bvalues(proto_obj)
    assert name_obj == "acrin_6698"
    np.testing.assert_array_equal(b_obj, [0.0, 100.0, 600.0, 800.0])

    custom = [0, 25, 75, 200, 500, 900]
    b_cust, name_cust = resolve_bvalues(custom)
    assert name_cust == "custom"
    np.testing.assert_array_equal(b_cust, np.asarray(custom, dtype=np.float64))


def test_protocol_validation_and_scan_duration() -> None:
    proto = AcquisitionProtocol(
        key="custom_nex",
        title="Custom NEX",
        bvalues=(0.0, 50.0, 200.0, 800.0),
        averages=(1, 2, 2, 5),
    )
    assert proto.n_acquisitions == 10
    assert proto.scan_minutes(seconds_per_volume=12.0) == pytest.approx(2.0)

    with pytest.raises(ValueError, match="at least 2"):
        AcquisitionProtocol(key="bad1", title="Bad", bvalues=(0.0,))
    with pytest.raises(ValueError, match="negative"):
        AcquisitionProtocol(key="bad2", title="Bad", bvalues=(-10.0, 100.0))
    with pytest.raises(ValueError, match="non-decreasing"):
        AcquisitionProtocol(key="bad3", title="Bad", bvalues=(0.0, 500.0, 100.0))
    with pytest.raises(ValueError, match="averages length"):
        AcquisitionProtocol(key="bad4", title="Bad", bvalues=(0.0, 100.0), averages=(1, 2, 3))
    with pytest.raises(KeyError, match="Unknown acquisition protocol"):
        get_protocol("nonexistent_protocol")
