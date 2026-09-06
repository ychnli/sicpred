"""Tests for named experiment configuration resolution."""

from dataclasses import replace

import pandas as pd
import pytest

from src.experiment_configs import (
    ExperimentConfig,
    load_config,
    time_split,
    validate_config,
)


@pytest.mark.parametrize(
    ("selector", "experiment_name", "data_name"),
    [
        ("exp1_inputs:input2", "exp1_input2", "seaice_plus_auxiliary"),
        ("exp1_inputs:input3a", "exp1_input3a", "seaice_plus_sst"),
        ("exp1_inputs:input3b", "exp1_input3b", "seaice_plus_psl"),
        ("exp1_inputs:input3c", "exp1_input3c", "seaice_plus_z500"),
        ("exp1_inputs:input3d", "exp1_input3d", "seaice_plus_t2m"),
        ("exp1_inputs:input4", "exp1_input4", "seaice_plus_all"),
        ("exp2_data_volume:vol1", "exp2_vol1", "seaice_plus_auxiliary_vol1"),
        ("exp2_data_volume:vol4", "exp2_vol4", "seaice_plus_auxiliary_vol4"),
        ("exp3_obs:obs_input2", "obs_input2_ensemble", "seaice_plus_auxiliary_obs"),
        ("exp3_obs:obs_input2_finetune", "obs_input2_finetune", "seaice_plus_auxiliary_obs"),
    ],
)
def test_named_configs_preserve_output_identifiers(selector, experiment_name, data_name):
    config = load_config(selector)

    assert config.experiment_name == experiment_name
    assert config.data_name == data_name
    assert config.data_split["name"] == data_name


def test_input_variants_only_enable_their_named_predictors():
    expected = {
        "input2": set(),
        "input3a": {"sst"},
        "input3b": {"psl"},
        "input3c": {"geopotential"},
        "input3d": {"t2m"},
        "input4": {"sst", "psl", "geopotential", "t2m"},
    }

    for variant, extra_inputs in expected.items():
        config = load_config(f"exp1_inputs:{variant}")
        physical_inputs = {
            name
            for name, settings in config.input_config.items()
            if settings["include"] and not settings["auxiliary"]
        }
        assert physical_inputs == {"icefrac", *extra_inputs}


def test_data_volume_variants_only_change_training_volume_and_identity():
    expected_volumes = {"vol1": 1, "vol2": 4, "vol3": 16, "vol4": 64}
    configs = {
        variant: load_config(f"exp2_data_volume:{variant}")
        for variant in expected_volumes
    }

    for variant, expected_volume in expected_volumes.items():
        assert len(configs[variant].data_split["train"]) == expected_volume
        assert configs[variant].num_epochs == 20
    assert all(
        config.data_split["val"] == configs["vol1"].data_split["val"]
        and config.data_split["test"] == configs["vol1"].data_split["test"]
        for config in configs.values()
    )


def test_obs_finetune_only_changes_optimization_settings_and_identity():
    baseline = load_config("exp3_obs:obs_input2")
    finetune = load_config("exp3_obs:obs_input2_finetune")

    for key in ("train", "val", "test"):
        assert baseline.data_split[key].equals(finetune.data_split[key])
    assert baseline.data_split["member_ids"] == finetune.data_split["member_ids"]
    assert baseline.input_config == finetune.input_config
    assert baseline.learning_rate == 1e-3
    assert finetune.learning_rate == 1e-4
    assert baseline.weight_decay == 5e-2
    assert finetune.weight_decay == 1e-3


def test_loading_returns_independent_mutable_values():
    first = load_config("exp1_inputs:input2")
    first.input_config["icefrac"]["lag"] = 99

    second = load_config("exp1_inputs:input2")
    assert second.input_config["icefrac"]["lag"] == 12


def test_time_split_accepts_a_resolved_rolling_origin_fold():
    split = time_split(
        "obs_fold_1",
        train=pd.date_range("1979-01", "1994-12", freq="MS"),
        val=pd.date_range("1995-01", "2000-06", freq="MS"),
        test=pd.date_range("2001-01", "2006-12", freq="MS"),
        member_ids=["obs"],
    )
    config = ExperimentConfig(
        experiment_name="obs_fold_1",
        notes="Synthetic fold used to verify the configuration shape",
        data_name="obs_fold_1",
        data_split=split,
        input_config=load_config("exp3_obs:obs_input2").input_config,
        target_config={"predict_anom": True, "predict_classes": False},
    )

    validate_config(config)
    assert config.data_split["split_by"] == "time"


def test_validation_rejects_overlapping_partitions():
    config = load_config("exp3_obs:obs_input2")
    invalid_split = dict(config.data_split)
    invalid_split["val"] = invalid_split["train"][-1:].union(invalid_split["val"])

    with pytest.raises(ValueError, match="must be disjoint"):
        validate_config(replace(config, data_split=invalid_split))


@pytest.mark.parametrize("selector", ["exp1_inputs", "unknown:variant", "exp1_inputs:unknown"])
def test_invalid_selectors_have_clear_errors(selector):
    with pytest.raises(ValueError):
        load_config(selector)
