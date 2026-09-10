"""Focused synthetic checks for the isolated detrending triage helper."""

import numpy as np
import pandas as pd
import xarray as xr

from generate_and_compare import detrend_quadratic_train_only
from src.utils.util_cesm import detrend_quadratic


def test_held_out_extreme_does_not_change_training_fit():
    times = pd.date_range("2000-01", "2009-12", freq="MS")
    members = ["train_a", "train_b", "held_out"]
    year = times.year.to_numpy() - 2000
    seasonal_trend = (0.2 * year + 0.01 * year**2).astype(np.float32)
    values = np.stack(
        [seasonal_trend, seasonal_trend + 0.5, seasonal_trend.copy()],
        axis=0,
    )[:, :, None, None]
    baseline = xr.DataArray(
        values,
        dims=("member_id", "time", "y", "x"),
        coords={"member_id": members, "time": times, "y": [0], "x": [0]},
        name="synthetic",
    )
    contaminated = baseline.copy(deep=True)
    contaminated.loc[{"member_id": "held_out"}] += 1e10

    clean_result, clean_coefficients = detrend_quadratic_train_only(
        baseline,
        ["train_a", "train_b"],
    )
    contaminated_result, contaminated_coefficients = detrend_quadratic_train_only(
        contaminated,
        ["train_a", "train_b"],
    )

    xr.testing.assert_allclose(clean_coefficients, contaminated_coefficients)
    xr.testing.assert_allclose(
        clean_result.sel(member_id=["train_a", "train_b"]),
        contaminated_result.sel(member_id=["train_a", "train_b"]),
    )


def test_output_preserves_input_layout():
    times = pd.date_range("2000-01", "2003-12", freq="MS")
    data = xr.DataArray(
        np.arange(2 * len(times) * 2 * 3, dtype=np.float32).reshape(2, len(times), 2, 3),
        dims=("member_id", "time", "y", "x"),
        coords={"member_id": ["train", "test"], "time": times, "y": [0, 1], "x": [0, 1, 2]},
        name="synthetic",
    )
    result, coefficients = detrend_quadratic_train_only(data, ["train"])

    assert result.dims == data.dims
    assert result.shape == data.shape
    assert coefficients.dims == ("month", "coeff", "y", "x")
    assert coefficients.shape == (12, 3, 2, 3)


def test_all_members_fit_matches_legacy_implementation():
    rng = np.random.default_rng(7)
    times = pd.date_range("2000-01", "2005-12", freq="MS")
    members = ["member_a", "member_b", "member_c"]
    data = xr.DataArray(
        rng.normal(size=(3, len(times), 2, 2)).astype(np.float32),
        dims=("member_id", "time", "y", "x"),
        coords={"member_id": members, "time": times, "y": [0, 1], "x": [0, 1]},
        name="synthetic",
    )

    legacy_result, legacy_coefficients = detrend_quadratic(data)
    isolated_result, isolated_coefficients = detrend_quadratic_train_only(
        data,
        members,
    )

    xr.testing.assert_equal(legacy_result, isolated_result)
    xr.testing.assert_equal(legacy_coefficients, isolated_coefficients)
