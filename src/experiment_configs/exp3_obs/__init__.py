"""Observational-data experiment variants."""

from copy import deepcopy

import pandas as pd

from src.experiment_configs.config import ExperimentConfig, time_split
from src.experiment_configs.exp1_inputs import CONFIGS as INPUT_CONFIGS

_TARGET = {"predict_anom": True, "predict_classes": False}
_SPECIAL_TEST_YEARS = pd.date_range("2014-01", "2014-12", freq="MS").union(
    pd.date_range("2017-01", "2017-12", freq="MS")
)


def _obs_config(*, experiment_name: str, data_name: str,
                train: pd.DatetimeIndex, val: pd.DatetimeIndex,
                test: pd.DatetimeIndex, finetune: bool) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=experiment_name,
        notes="Inputs: same as input2. ERA5 data",
        data_name=data_name,
        data_split=time_split(
            data_name, train=train, val=val, test=test, member_ids=["obs"]),
        input_config=deepcopy(INPUT_CONFIGS["input2"].input_config),
        target_config=deepcopy(_TARGET),
        learning_rate=1e-4 if finetune else 1e-3,
        weight_decay=1e-3 if finetune else 5e-2,
        batch_size=32,
        num_epochs=50,
        checkpoint_interval=10,
        patience=10,
        lr_scheduler="cosine",
        lr_scheduler_args={"t_max": 50, "eta_min": 0 if finetune else 5e-5},
    )


_CURRENT_SPLIT = {
    "train": pd.date_range("1979-01", "2011-12", freq="MS"),
    "val": pd.date_range("2012-01", "2019-12", freq="MS").difference(
        _SPECIAL_TEST_YEARS),
    "test": pd.date_range("2020-01", "2024-12", freq="MS").union(
        _SPECIAL_TEST_YEARS),
}
_OLD_SPLIT = {
    "train": pd.date_range("1979-01", "2011-12", freq="MS"),
    "val": pd.date_range("2012-01", "2015-12", freq="MS"),
    "test": pd.date_range("2016-01", "2024-01", freq="MS"),
}

CONFIGS = {
    "obs_input2": _obs_config(
        experiment_name="obs_input2_ensemble",
        data_name="seaice_plus_auxiliary_obs", finetune=False, **_CURRENT_SPLIT),
    "obs_input2_finetune": _obs_config(
        experiment_name="obs_input2_finetune",
        data_name="seaice_plus_auxiliary_obs", finetune=True, **_CURRENT_SPLIT),
    "obs_input2_oldsplit": _obs_config(
        experiment_name="obs_input2_old_split",
        data_name="seaice_plus_auxiliary_obs_oldsplit", finetune=False, **_OLD_SPLIT),
    "obs_input2_oldsplit_ft": _obs_config(
        experiment_name="obs_input2_oldsplit_ft",
        data_name="seaice_plus_auxiliary_obs_oldsplit", finetune=True, **_OLD_SPLIT),
}

__all__ = ["CONFIGS"]
