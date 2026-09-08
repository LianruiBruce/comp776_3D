from __future__ import annotations

import numpy as np
import pytest

from id_layers.likeness_power import _effects, run_power_grid


def tiny_config():
    return {
        "identity_counts": [12, 24],
        "mean_gaps": [0.0, 1.0],
        "identity_gap_sds": [0.5],
        "missing_rates": [0.0, 0.25],
        "expression_extra_gaps": [0.0],
        "replicates": 40,
        "population_reference_identities": 2000,
        "seed": 11,
    }


def test_power_is_deterministic_marked_synthetic_and_tracks_missingness():
    rows = run_power_grid(tiny_config())
    assert rows == run_power_grid(tiny_config())
    assert len(rows) == 16
    assert {r["endpoint"] for r in rows} == {"D", "E"}
    assert all(r["evidence_type"] == "SIMULATION_ONLY" for r in rows)
    for row in rows:
        assert row["realized_missing_rate"] == pytest.approx(
            row["requested_missing_rate"], abs=0.025
        )
        assert 0 <= row["two_sided_zero_test_power"] <= 1
        assert 0 <= row["ci95_population_coverage_given_estimable"] <= 1
        if row["requested_missing_rate"] == 0:
            assert row["mean_analyzable_identity_count"] == row["identity_count"]


def test_power_effect_directions_and_rounding_are_reported():
    config = tiny_config()
    config.update(mean_gaps=[1.0], expression_extra_gaps=[0.5])
    rows = run_power_grid(config)
    d = next(r for r in rows if r["endpoint"] == "D")
    e = next(r for r in rows if r["endpoint"] == "E")
    assert d["population_observed_scale_effect"] > 0.8
    assert e["population_observed_scale_effect"] > 0.3
    assert "Student-t" in d["interval_method"]


def test_effects_do_not_convert_missing_cells_to_zero():
    scores = np.asarray(
        [[[6, 6], [5, 5], [6, 6], [4, 4]], [[6, 6], [np.nan, np.nan], [6, 6], [4, 4]]]
    )
    d, e = _effects(scores)
    assert d[0] == 1.5 and e[0] == 1.0
    assert np.isnan(d[1]) and np.isnan(e[1])


def test_mnar_and_invalid_configs_are_explicit():
    config = tiny_config()
    config["missing_mechanisms"] = ["low_score_mnar"]
    rows = run_power_grid(config)
    assert all(r["missing_mechanism"] == "low_score_mnar" for r in rows)
    with pytest.raises(ValueError, match="Unknown power"):
        run_power_grid({"surprise": 3})
    with pytest.raises(ValueError, match="Missing rates"):
        run_power_grid({"missing_rates": [1.2]})
