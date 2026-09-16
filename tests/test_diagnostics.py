import os

import numpy as np
import pandas as pd
import xarray as xr

from src.models import diagnostics


def legacy_reconstruct(anomaly, data_split_settings, processed_root):
    norm_dir = os.path.join(
        processed_root, "normalized_inputs", data_split_settings["name"]
    )
    monthly_mean = xr.load_dataarray(os.path.join(norm_dir, "icefrac_mean.nc"))
    detrend_coeffs = xr.load_dataarray(
        os.path.join(norm_dir, "icefrac_detrend_coeffs.nc")
    )

    def add_months(dt64, months):
        timestamp = pd.Timestamp(dt64)
        return (timestamp + pd.DateOffset(months=int(months))).to_datetime64()

    valid_time = xr.apply_ufunc(
        add_months,
        anomaly.start_prediction_month,
        anomaly.lead_time - 1,
        vectorize=True,
    )
    valid_month = valid_time.dt.month
    time_value = valid_time.dt.year + valid_time.dt.month / 12.0
    mean_by_valid_month = monthly_mean.sel(month=valid_month)
    coeffs_by_valid_month = detrend_coeffs.sel(month=valid_month)
    constant = coeffs_by_valid_month.sel(coeff="const")
    linear = coeffs_by_valid_month.sel(coeff="linear")
    quadratic = coeffs_by_valid_month.sel(coeff="quadratic")
    time_broadcast = time_value.broadcast_like(mean_by_valid_month)
    trend = constant + linear * time_broadcast + quadratic * time_broadcast ** 2
    return anomaly + mean_by_valid_month + trend


def test_chunked_diagnostics_match_eager_reference(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    x = np.arange(4)
    y = np.arange(3)
    months = np.arange(1, 13)
    norm_dir = tmp_path / "normalized_inputs" / "diagnostic_equivalence"
    norm_dir.mkdir(parents=True)
    monthly_mean = xr.DataArray(
        rng.normal(scale=0.1, size=(12, 3, 4)).astype(np.float32),
        dims=("month", "y", "x"),
        coords={"month": months, "y": y, "x": x},
        name="icefrac",
    )
    detrend_coeffs = xr.DataArray(
        rng.normal(scale=1e-4, size=(12, 3, 3, 4)).astype(np.float32),
        dims=("month", "coeff", "y", "x"),
        coords={
            "month": months,
            "coeff": ["const", "linear", "quadratic"],
            "y": y,
            "x": x,
        },
        name="icefrac",
    )
    monthly_mean.to_netcdf(norm_dir / "icefrac_mean.nc")
    detrend_coeffs.to_netcdf(norm_dir / "icefrac_detrend_coeffs.nc")

    start_months = pd.date_range("2000-01", periods=4, freq="MS")
    prediction_dims = (
        "start_prediction_month",
        "member_id",
        "nn_member_id",
        "lead_time",
        "y",
        "x",
    )
    prediction_coords = {
        "start_prediction_month": start_months,
        "member_id": ["a", "b"],
        "nn_member_id": [0, 1],
        "lead_time": [1, 2],
        "y": y,
        "x": x,
    }
    prediction_shape = tuple(
        len(prediction_coords[dim]) for dim in prediction_dims
    )
    predictions_eager = xr.DataArray(
        rng.normal(scale=0.2, size=prediction_shape).astype(np.float32),
        dims=prediction_dims,
        coords=prediction_coords,
    )
    target_coords = {
        key: value
        for key, value in prediction_coords.items()
        if key != "nn_member_id"
    }
    targets_eager = xr.DataArray(
        rng.normal(scale=0.2, size=(4, 2, 2, 3, 4)).astype(np.float32),
        dims=("start_prediction_month", "member_id", "lead_time", "y", "x"),
        coords=target_coords,
    )
    predictions = predictions_eager.chunk(
        {
            "start_prediction_month": 2,
            "member_id": 1,
            "nn_member_id": -1,
            "lead_time": -1,
            "y": -1,
            "x": -1,
        }
    )
    targets = targets_eager.chunk(
        {
            "start_prediction_month": 2,
            "member_id": 1,
            "lead_time": -1,
            "y": -1,
            "x": -1,
        }
    )
    area = xr.DataArray(
        rng.uniform(1, 2, size=(3, 4)),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    weights = area / area.mean()
    monkeypatch.setattr(
        diagnostics.config_cesm, "PROCESSED_DATA_DIRECTORY", str(tmp_path)
    )
    monkeypatch.setattr(
        diagnostics, "REFERENCE_GRID", xr.Dataset({"area": area})
    )
    monkeypatch.setattr(diagnostics, "AREA_WEIGHTS", weights)

    expected_acc = xr.corr(predictions_eager, targets_eager, dim=("x", "y"))
    expected_rmse = np.sqrt(
        (((predictions_eager - targets_eager) ** 2) * weights).sum(("x", "y"))
        / weights.sum()
    )
    split = {"name": "diagnostic_equivalence"}
    expected_pred_sic = legacy_reconstruct(
        predictions_eager, split, str(tmp_path)
    )
    expected_truth_sic = legacy_reconstruct(targets_eager, split, str(tmp_path))
    expected_iiee = (
        ((expected_pred_sic > 0.15) ^ (expected_truth_sic > 0.15)) * area
    ).sum(("x", "y"))

    actual_acc = diagnostics.calculate_acc(predictions, targets).compute()
    actual_rmse = diagnostics.calculate_rmse(predictions, targets).compute()
    actual_iiee = diagnostics.calculate_iiee(predictions, targets, split).compute()

    xr.testing.assert_allclose(actual_acc, expected_acc, rtol=2e-6, atol=1e-7)
    xr.testing.assert_allclose(actual_rmse, expected_rmse, rtol=1e-6, atol=1e-7)
    xr.testing.assert_identical(actual_iiee, expected_iiee)
