"""Tests for dynamic CESM model-sample construction."""

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src import config_cesm
from src.experiment_configs import ExperimentConfig, ensemble_member_split
from src.models.models_util import CESM_Dataset
from src.utils import util_cesm


def _write_normalized_field(path, name, values, members, times):
    da = xr.DataArray(
        values,
        dims=("member_id", "time", "y", "x"),
        coords={
            "member_id": members,
            "time": times,
            "month": ("time", times.month),
            "y": np.arange(values.shape[-2]),
            "x": np.arange(values.shape[-1]),
        },
        name=name,
    )
    da.to_dataset().to_netcdf(path)


def test_dataset_builds_lagged_inputs_and_targets_at_runtime(
    monkeypatch, tmp_path
):
    processed_dir = tmp_path / "processed"
    normalized_dir = processed_dir / "normalized_inputs" / "synthetic"
    normalized_dir.mkdir(parents=True)
    land_mask_path = tmp_path / "land_mask.nc"

    members = ["member1", "member2", "member3"]
    times = pd.date_range("1999-11", "2000-03", freq="MS")
    shape = (len(members), len(times), 2, 3)
    time_values = np.arange(len(times), dtype=np.float32)[None, :, None, None]
    member_values = 100 * np.arange(len(members), dtype=np.float32)[:, None, None, None]
    icefrac = np.broadcast_to(time_values + member_values, shape).copy()
    z500 = icefrac + 10

    _write_normalized_field(
        normalized_dir / "icefrac_norm.nc",
        "icefrac",
        icefrac,
        members,
        times,
    )
    _write_normalized_field(
        normalized_dir / "z500_norm.nc",
        "z500",
        z500,
        members,
        times,
    )
    land_mask = np.arange(6, dtype=np.float32).reshape(2, 3)
    xr.Dataset(
        {"mask": (("y", "x"), land_mask)},
        coords={"y": np.arange(2), "x": np.arange(3)},
    ).to_netcdf(land_mask_path)

    monkeypatch.setattr(
        config_cesm, "PROCESSED_DATA_DIRECTORY", str(processed_dir)
    )
    monkeypatch.setattr(util_cesm, "LAND_MASK_PATH", str(land_mask_path))

    input_config = {
        "icefrac": {
            "include": True,
            "auxiliary": False,
            "lag": 2,
        },
        "z500": {
            "include": True,
            "auxiliary": False,
            "lag": 1,
        },
        "cosine_of_init_month": {
            "include": True,
            "auxiliary": True,
        },
        "sine_of_init_month": {
            "include": True,
            "auxiliary": True,
        },
        "land_mask": {
            "include": True,
            "auxiliary": True,
        },
    }
    config = ExperimentConfig(
        experiment_name="synthetic",
        notes="",
        data_name="synthetic",
        data_split=ensemble_member_split(
            "synthetic",
            train=["member1"],
            val=["member2"],
            test=["member3"],
            time_range=pd.date_range("2000-01", "2000-02", freq="MS"),
        ),
        input_config=input_config,
        target_config={"predict_anom": True, "predict_classes": False},
        max_lead_months=2,
    )

    dataset = CESM_Dataset("train", config)
    sample = dataset[0]
    input_da = dataset.input_data_array("member1", pd.Timestamp("2000-01"))

    assert len(dataset) == 2
    assert input_da.channel.values.tolist() == [
        "icefrac_lag2",
        "icefrac_lag1",
        "z500_lag1",
        "cosine_of_init_month",
        "sine_of_init_month",
        "land_mask",
    ]
    assert sample["input"].shape == torch.Size([6, 2, 3])
    assert sample["target"].shape == torch.Size([2, 2, 3])
    np.testing.assert_allclose(sample["input"][0], 0)
    np.testing.assert_allclose(sample["input"][1], 1)
    np.testing.assert_allclose(sample["input"][2], 11)
    np.testing.assert_allclose(
        sample["input"][3], np.cos(2 * np.pi / 12)
    )
    np.testing.assert_allclose(
        sample["input"][4], np.sin(2 * np.pi / 12)
    )
    np.testing.assert_allclose(sample["input"][5], land_mask)
    np.testing.assert_allclose(sample["target"][0], 2)
    np.testing.assert_allclose(sample["target"][1], 3)
    np.testing.assert_array_equal(
        sample["start_prediction_month"],
        np.array([[2000, 1], [2000, 2]]),
    )
    assert sample["member_id"] == "member1"
    assert not (processed_dir / "data_pairs").exists()

    dataset._close_cache()

    no_sic_input_config = {
        **input_config,
        "icefrac": {"include": False, "auxiliary": False, "lag": 2},
    }
    no_sic_dataset = CESM_Dataset(
        "train", replace(config, input_config=no_sic_input_config)
    )
    no_sic_sample = no_sic_dataset[0]

    assert no_sic_sample["input"].shape == torch.Size([4, 2, 3])
    np.testing.assert_allclose(no_sic_sample["target"][0], 2)
    no_sic_dataset._close_cache()


def test_input_sample_can_exclude_icefrac():
    times = pd.date_range("1999-12", "2000-01", freq="MS")
    z500 = xr.DataArray(
        np.ones((1, 2, 2, 3), dtype=np.float32),
        dims=("member_id", "time", "y", "x"),
        coords={
            "member_id": ["member1"],
            "time": times,
            "y": np.arange(2),
            "x": np.arange(3),
        },
    )
    input_config = {
        "icefrac": {"include": False, "auxiliary": False, "lag": 12},
        "z500": {"include": True, "auxiliary": False, "lag": 1},
        "land_mask": {"include": True, "auxiliary": True},
    }

    sample = util_cesm.build_input_sample(
        {"z500": z500},
        input_config,
        "member1",
        pd.Timestamp("2000-01"),
        land_mask=np.zeros((2, 3), dtype=np.float32),
    )

    assert sample.channel.values.tolist() == ["z500_lag1", "land_mask"]
    assert sample.shape == (2, 2, 3)
