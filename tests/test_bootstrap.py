import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.utils.bootstrap import (
    benjamini_hochberg,
    bootstrap_metric_significance,
)


def seed_metric(seed_ids, seed_values, start_months=None):
    """Build a metric whose only variability is across training seeds."""
    if start_months is None:
        start_months = pd.to_datetime(["2000-01-01", "2001-01-01"])
    seed_values = np.asarray(seed_values, dtype=float)
    values = np.broadcast_to(
        seed_values[None, None, :],
        (len(start_months), 1, len(seed_values)),
    ).copy()
    return xr.DataArray(
        values,
        dims=("start_prediction_month", "lead_time", "nn_member_id"),
        coords={
            "start_prediction_month": start_months,
            "lead_time": [1],
            "nn_member_id": seed_ids,
        },
    )


def test_bootstrap_resamples_training_seeds_independently():
    metric_a = seed_metric([0, 1], [0.0, 2.0])
    metric_b = seed_metric([0, 1], [1.0, 1.0])

    result = bootstrap_metric_significance(
        metric_a,
        metric_b,
        n_bootstrap=2000,
        random_seed=42,
    ).sel(month=1, lead_time=1)

    assert result["delta"].item() == pytest.approx(0.0)
    assert result["ci_low"].item() < 0
    assert result["ci_high"].item() > 0


def test_equal_seed_labels_are_not_required_or_aligned():
    metric_a = seed_metric([0, 1], [0.0, 2.0])
    metric_b = seed_metric([10, 11], [2.0, 2.0])

    result = bootstrap_metric_significance(
        metric_a,
        metric_b,
        n_bootstrap=100,
        random_seed=42,
    ).sel(month=1, lead_time=1)

    assert result["delta"].item() == pytest.approx(-1.0)


def test_bootstrap_requires_exact_verification_coordinates():
    metric_a = seed_metric([0, 1], [0.0, 2.0])
    metric_b = seed_metric(
        [10, 11],
        [0.0, 2.0],
        start_months=pd.to_datetime(["2000-01-01", "2002-01-01"]),
    )

    with pytest.raises(ValueError, match="cannot align"):
        bootstrap_metric_significance(
            metric_a,
            metric_b,
            n_bootstrap=100,
            random_seed=42,
        )


def test_benjamini_hochberg_known_values_and_nans():
    p_values = np.array([[0.01, 0.04], [0.03, np.nan], [0.002, np.nan]])
    expected = np.array([[0.02, 0.04], [0.04, np.nan], [0.008, np.nan]])

    np.testing.assert_allclose(
        benjamini_hochberg(p_values),
        expected,
        equal_nan=True,
    )


def test_bootstrap_adds_within_comparison_q_values():
    metric_a = seed_metric([0, 1], [0.0, 2.0])
    metric_b = seed_metric([10, 11], [2.0, 2.0])

    result = bootstrap_metric_significance(
        metric_a,
        metric_b,
        n_bootstrap=100,
        random_seed=42,
    )

    assert "q_value" in result
    finite = np.isfinite(result["p_value"])
    assert np.all(result["q_value"].where(finite, drop=True) >=
                  result["p_value"].where(finite, drop=True))


@pytest.mark.parametrize(
    ("n_bootstrap", "alpha"), [(0, 0.05), (100, 0), (100, 1)]
)
def test_bootstrap_rejects_invalid_sampling_settings(n_bootstrap, alpha):
    values = xr.DataArray(
        np.ones((1, 1)),
        dims=("start_prediction_month", "lead_time"),
        coords={
            "start_prediction_month": [np.datetime64("2000-01-01", "ns")],
            "lead_time": [1],
        },
    )
    with pytest.raises(ValueError):
        bootstrap_metric_significance(
            values,
            values,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )
