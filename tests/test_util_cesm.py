"""Focused regression tests for CESM preprocessing utilities."""

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.util_cesm import detrend_quadratic


def _synthetic_data():
    times = pd.date_range("2000-01", "2009-12", freq="MS")
    year = times.year.to_numpy() - 2000
    trend = (0.2 * year + 0.01 * year**2).astype(np.float32)
    values = np.stack([trend, trend + 0.5, trend - 0.25], axis=0)
    return xr.DataArray(
        values[:, :, None, None],
        dims=("member_id", "time", "y", "x"),
        coords={
            "member_id": ["train_a", "train_b", "held_out"],
            "time": times,
            "y": [0],
            "x": [0],
        },
        name="synthetic",
    )


def test_member_fit_is_unchanged_by_held_out_values():
    baseline = _synthetic_data()
    contaminated = baseline.copy(deep=True)
    contaminated.loc[{"member_id": "held_out"}] += 1e10
    train_members = ["train_a", "train_b"]

    baseline_result, baseline_coefficients = detrend_quadratic(
        baseline, fit_da=baseline.sel(member_id=train_members)
    )
    contaminated_result, contaminated_coefficients = detrend_quadratic(
        contaminated, fit_da=contaminated.sel(member_id=train_members)
    )

    xr.testing.assert_allclose(baseline_coefficients, contaminated_coefficients)
    xr.testing.assert_allclose(
        baseline_result.sel(member_id=train_members),
        contaminated_result.sel(member_id=train_members),
    )


def test_time_fit_is_unchanged_by_held_out_values():
    baseline = _synthetic_data().sel(member_id=["train_a"])
    train_times = baseline.time[:72]
    held_out_times = baseline.time[72:]
    contaminated = baseline.copy(deep=True)
    contaminated.loc[{"time": held_out_times}] += 1e10

    baseline_result, baseline_coefficients = detrend_quadratic(
        baseline, fit_da=baseline.sel(time=train_times)
    )
    contaminated_result, contaminated_coefficients = detrend_quadratic(
        contaminated, fit_da=contaminated.sel(time=train_times)
    )

    xr.testing.assert_allclose(baseline_coefficients, contaminated_coefficients)
    xr.testing.assert_allclose(
        baseline_result.sel(time=train_times),
        contaminated_result.sel(time=train_times),
    )


def test_default_fit_preserves_legacy_full_data_behavior():
    data = _synthetic_data()

    default_result, default_coefficients = detrend_quadratic(data)
    explicit_result, explicit_coefficients = detrend_quadratic(data, fit_da=data)

    xr.testing.assert_equal(default_result, explicit_result)
    xr.testing.assert_equal(default_coefficients, explicit_coefficients)
