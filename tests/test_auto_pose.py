import numpy as np
import pandas as pd
import pytest

from id_layers.auto_pose import circular_identity_interval, circular_mean, wrap_degrees


def test_angle_changes_wrap_at_boundary():
    assert float(wrap_degrees(-179 - 179)) == 2
    assert float(wrap_degrees(179 - (-179))) == -2
    mean, concentration = circular_mean([179, -179])
    assert abs(mean) == pytest.approx(180)
    assert concentration > 0.99


def test_circular_bootstrap_stays_near_direction_not_zero():
    values = pd.Series([179, -179, 178, -178], index=["a", "b", "c", "d"])
    result = circular_identity_interval(values, resamples=50, seed=1)
    assert abs(result["estimate_degrees"]) == pytest.approx(180)
    assert result["ci95_upper_centered_degrees"] - result["ci95_lower_centered_degrees"] < 5
    assert result == circular_identity_interval(values.iloc[::-1], resamples=50, seed=1)


def test_ambiguous_or_missing_angles_are_not_zero_pose():
    assert circular_mean([0, 180])[0] is None
    assert circular_mean([np.nan])[0] is None
    with pytest.raises(ValueError, match="Infinite"):
        circular_mean([np.inf])
