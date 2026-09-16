"""Equivalence and access-speed tests for eager dynamic CESM samples."""

from time import perf_counter

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src import config_cesm
from src.experiment_configs import ExperimentConfig, ensemble_member_split
from src.models.models_util import (
    CESM_Dataset,
    EagerCESMDataStore,
    EagerDynamicCESMDataset,
    build_cesm_dataloader,
    load_cesm_targets_data_array,
)
from src.utils import util_cesm


def _synthetic_config():
    input_config = {
        name: {
            "include": True,
            "auxiliary": False,
            "lag": 3 if name == "icefrac" else 2,
        }
        for name in ("icefrac", "z500", "z50", "psl", "t2m")
    }
    input_config.update(
        {
            "cosine_of_init_month": {"include": True, "auxiliary": True},
            "sine_of_init_month": {"include": True, "auxiliary": True},
            "land_mask": {"include": True, "auxiliary": True},
        }
    )
    return ExperimentConfig(
        experiment_name="dynamic_synthetic",
        notes="",
        data_name="dynamic_synthetic",
        data_split=ensemble_member_split(
            "dynamic_synthetic",
            train=["member1"],
            val=["member2"],
            test=["member3"],
            time_range=pd.date_range("2000-01", "2000-06", freq="MS"),
        ),
        input_config=input_config,
        target_config={"predict_anom": True, "predict_classes": False},
        max_lead_months=2,
        batch_size=2,
    )


def _write_normalized_fields(root, config):
    normalized_dir = (
        root / "normalized_inputs" / config.data_split["name"]
    )
    normalized_dir.mkdir(parents=True)
    members = ["member1", "member2", "member3"]
    times = pd.date_range("1999-10", "2000-07", freq="MS")
    y_coords = np.arange(6)
    x_coords = np.arange(7)
    shape = (len(members), len(times), len(y_coords), len(x_coords))

    member_component = np.arange(len(members), dtype=np.float32)[:, None, None, None]
    time_component = np.arange(len(times), dtype=np.float32)[None, :, None, None]
    y_component = y_coords.astype(np.float32)[None, None, :, None]
    x_component = x_coords.astype(np.float32)[None, None, None, :]
    base = member_component * 100 + time_component * 10 + y_component + x_component / 10

    for variable_index, name in enumerate(("icefrac", "z500", "z50", "psl", "t2m")):
        values = np.broadcast_to(base + variable_index * 1000, shape).copy()
        values[0, 2, 1, 1] = np.nan
        xr.DataArray(
            values,
            dims=("member_id", "time", "y", "x"),
            coords={
                "member_id": members,
                "time": times,
                "month": ("time", times.month),
                "y": y_coords,
                "x": x_coords,
            },
            name=name,
        ).to_dataset().to_netcdf(normalized_dir / f"{name}_norm.nc")

    land_mask = np.indices((len(y_coords), len(x_coords))).sum(axis=0) % 2
    land_mask_path = root / "land_mask.nc"
    xr.Dataset(
        {"mask": (("y", "x"), land_mask.astype(np.float32))},
        coords={"y": y_coords, "x": x_coords},
    ).to_netcdf(land_mask_path)
    return land_mask_path


def test_dynamic_dataset_matches_pairs_prefetches_and_is_faster(
    monkeypatch, tmp_path
):
    config = _synthetic_config()
    processed_dir = tmp_path / "processed"
    land_mask_path = _write_normalized_fields(processed_dir, config)
    pair_dir = processed_dir / "data_pairs" / config.data_name

    monkeypatch.setattr(
        config_cesm, "PROCESSED_DATA_DIRECTORY", str(processed_dir)
    )
    monkeypatch.setattr(
        util_cesm.config, "PROCESSED_DATA_DIRECTORY", str(processed_dir)
    )
    monkeypatch.setattr(util_cesm, "LAND_MASK_PATH", str(land_mask_path))

    # Keep exercising the existing preprocessing interface: pair artifacts are
    # optional for training, but remain available to notebooks and debugging.
    util_cesm.save_inputs_files(
        config.input_config, str(pair_dir), config.data_split
    )
    util_cesm.save_targets_files(
        config.target_config,
        str(pair_dir),
        config.max_lead_months,
        config.data_split,
    )

    legacy = CESM_Dataset("train", config)
    store = EagerCESMDataStore(config, splits=("train", "val"))
    dynamic = EagerDynamicCESMDataset("train", config, store=store)
    assert len(dynamic) == len(legacy)

    for index in range(len(legacy)):
        expected = legacy[index]
        actual = dynamic[index]
        torch.testing.assert_close(actual["input"], expected["input"], rtol=0, atol=0)
        torch.testing.assert_close(actual["target"], expected["target"], rtol=0, atol=0)
        np.testing.assert_array_equal(
            actual["start_prediction_month"],
            expected["start_prediction_month"],
        )
        assert actual["member_id"] == expected["member_id"]

    start_month = config.data_split["time_range"][2]
    xr.testing.assert_allclose(
        dynamic.input_data_array("member1", start_month),
        legacy.input_data_array("member1", start_month),
    )
    xr.testing.assert_allclose(
        dynamic.target_data_array("member1", start_month),
        legacy.target_data_array("member1", start_month),
    )

    dynamic_targets = load_cesm_targets_data_array("train", config)
    legacy_targets = load_cesm_targets_data_array(
        "train", config, data_source="precomputed"
    )
    xr.testing.assert_allclose(dynamic_targets, legacy_targets)
    lazy_targets = load_cesm_targets_data_array(
        "train",
        config,
        chunks={
            "start_prediction_month": 2,
            "member_id": 1,
            "lead_time": -1,
            "y": -1,
            "x": -1,
        },
    )
    assert lazy_targets.chunks is not None
    xr.testing.assert_allclose(lazy_targets.compute(), legacy_targets)
    lazy_targets.close()

    loader = build_cesm_dataloader(
        dynamic,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=2,
    )
    iterator = iter(loader)
    try:
        batch = next(iterator)
        torch.testing.assert_close(
            batch["input"],
            torch.stack([dynamic[0]["input"], dynamic[1]["input"]]),
            rtol=0,
            atol=0,
        )
    finally:
        iterator._shutdown_workers()

    input_only_store = EagerCESMDataStore(
        config, splits=("test",), include_targets=False
    )
    input_only = EagerDynamicCESMDataset(
        "test", config, store=input_only_store, include_targets=False
    )
    assert input_only_store.target_array is None
    assert "target" not in input_only[0]

    access_indices = list(range(len(dynamic))) * 20
    legacy[0]
    dynamic[0]
    started = perf_counter()
    for index in access_indices:
        legacy[index]
    legacy_seconds = perf_counter() - started

    started = perf_counter()
    for index in access_indices:
        dynamic[index]
    dynamic_seconds = perf_counter() - started
    speedup = legacy_seconds / dynamic_seconds
    print(
        f"legacy={legacy_seconds:.4f}s dynamic={dynamic_seconds:.4f}s "
        f"speedup={speedup:.1f}x"
    )
    assert speedup > 2
