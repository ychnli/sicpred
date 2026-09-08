"""Tests for precomputed CESM model-sample loading and pair construction."""

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src import config_cesm
from src.experiment_configs import ExperimentConfig, ensemble_member_split
from src.models.models_util import CESM_Dataset
from src.utils import util_cesm


def _config(input_config):
    return ExperimentConfig(
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


def test_dataset_loads_precomputed_pairs(monkeypatch, tmp_path):
    pair_dir = tmp_path / "data_pairs" / "synthetic"
    pair_dir.mkdir(parents=True)
    months = pd.date_range("2000-01", "2000-02", freq="MS")

    for member_index, member in enumerate(("member1", "member2", "member3")):
        inputs = np.full((2, 4, 2, 3), member_index, dtype=np.float32)
        targets = np.full((2, 2, 2, 3), member_index + 10, dtype=np.float32)
        xr.Dataset(
            {"data": (("start_prediction_month", "channel", "y", "x"), inputs)},
            coords={"start_prediction_month": months},
        ).to_netcdf(pair_dir / f"inputs_member_{member}.nc")
        xr.Dataset(
            {"data": (("start_prediction_month", "lead_time", "y", "x"), targets)},
            coords={"start_prediction_month": months, "lead_time": [1, 2]},
        ).to_netcdf(pair_dir / f"targets_member_{member}.nc")

    monkeypatch.setattr(
        config_cesm, "PROCESSED_DATA_DIRECTORY", str(tmp_path)
    )
    input_config = {
        "icefrac": {"include": True, "auxiliary": False, "lag": 2},
        "cosine_of_init_month": {"include": True, "auxiliary": True},
        "land_mask": {"include": True, "auxiliary": True},
    }
    dataset = CESM_Dataset("train", _config(input_config))
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["input"].shape == torch.Size([4, 2, 3])
    assert sample["target"].shape == torch.Size([2, 2, 3])
    np.testing.assert_allclose(sample["input"], 0)
    np.testing.assert_allclose(sample["target"], 10)
    np.testing.assert_array_equal(
        sample["start_prediction_month"],
        np.array([[2000, 1], [2000, 2]]),
    )
    assert sample["member_id"] == "member1"
    assert dataset.target_data_array("member1", months[1]).shape == (2, 2, 3)


def test_precomputed_pairs_support_no_sic_inputs(monkeypatch, tmp_path):
    processed_dir = tmp_path / "processed"
    normalized_dir = processed_dir / "normalized_inputs" / "synthetic"
    normalized_dir.mkdir(parents=True)
    pair_dir = processed_dir / "data_pairs" / "synthetic"
    land_mask_path = tmp_path / "land_mask.nc"

    members = ["member1", "member2", "member3"]
    times = pd.date_range("1999-12", "2000-03", freq="MS")
    shape = (len(members), len(times), 2, 3)
    time_values = np.arange(len(times), dtype=np.float32)[None, :, None, None]
    values = np.broadcast_to(time_values, shape).copy()
    for name, offset in (("icefrac", 0), ("z500", 10)):
        xr.DataArray(
            values + offset,
            dims=("member_id", "time", "y", "x"),
            coords={
                "member_id": members,
                "time": times,
                "month": ("time", times.month),
                "y": np.arange(2),
                "x": np.arange(3),
            },
            name=name,
        ).to_dataset().to_netcdf(normalized_dir / f"{name}_norm.nc")
    xr.Dataset(
        {"mask": (("y", "x"), np.zeros((2, 3), dtype=np.float32))}
    ).to_netcdf(land_mask_path)

    monkeypatch.setattr(
        config_cesm, "PROCESSED_DATA_DIRECTORY", str(processed_dir)
    )
    monkeypatch.setattr(
        util_cesm.config, "PROCESSED_DATA_DIRECTORY", str(processed_dir)
    )
    monkeypatch.setattr(util_cesm, "LAND_MASK_PATH", str(land_mask_path))
    input_config = {
        "icefrac": {"include": False, "auxiliary": False, "lag": 12},
        "z500": {"include": True, "auxiliary": False, "lag": 1},
        "land_mask": {"include": True, "auxiliary": True},
    }
    config = _config(input_config)

    util_cesm.save_inputs_files(input_config, str(pair_dir), config.data_split)
    util_cesm.save_targets_files(
        config.target_config,
        str(pair_dir),
        config.max_lead_months,
        config.data_split,
    )
    sample = CESM_Dataset("train", config)[0]

    assert sample["input"].shape == torch.Size([2, 2, 3])
    assert sample["target"].shape == torch.Size([2, 2, 3])
    np.testing.assert_allclose(sample["input"][0], 10)
    np.testing.assert_allclose(sample["target"][0], 1)
    np.testing.assert_allclose(sample["target"][1], 2)
