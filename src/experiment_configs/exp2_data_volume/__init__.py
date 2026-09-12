"""Training-data-volume experiment variants."""

from copy import deepcopy

import pandas as pd

from src.config_cesm import AVAILABLE_CESM_MEMBERS
from src.experiment_configs.config import (
    ExperimentConfig,
    ensemble_member_split,
    time_split,
)
from src.experiment_configs.exp1_inputs import CONFIGS as INPUT_CONFIGS

_TARGET = {"predict_anom": True, "predict_classes": False}
_TRAINING_VOLUMES = {"vol1": 1, "vol2": 4, "vol3": 16, "vol4": 64}
_REANALYSIS_VOLUME_MEMBERS = (
    "r2i1251p1f1",
    "r2i1281p1f1",
    "r2i1301p1f1",
    "r3i1041p1f1",
)
_REANALYSIS_VOLUME_SPLIT = {
    "train": pd.date_range("1968-01", "2000-12", freq="MS"),
    "val": pd.date_range("2001-01", "2006-12", freq="MS"),
    "test": pd.date_range("2007-01", "2013-12", freq="MS"),
}


def _volume_config(variant: str, training_members: int) -> ExperimentConfig:
    data_name = f"seaice_plus_auxiliary_{variant}"
    split = ensemble_member_split(
        data_name,
        train=AVAILABLE_CESM_MEMBERS[:training_members],
        val=AVAILABLE_CESM_MEMBERS[64:66],
        test=AVAILABLE_CESM_MEMBERS[66:70],
        time_range=pd.date_range("1851-01", "2013-12", freq="MS"),
    )
    return ExperimentConfig(
        experiment_name=f"exp2_{variant}",
        notes=f"Inputs: same as input2. Data volume: {training_members} ens member. Detrended data",
        data_name=data_name,
        data_split=split,
        input_config=deepcopy(INPUT_CONFIGS["input2"].input_config),
        target_config=deepcopy(_TARGET),
        weight_decay=1e-3,
        num_epochs=20,
    )


def _reanalysis_volume_config(member_id: str) -> ExperimentConfig:
    """Build a reanalysis-sized temporal split for one CESM realization."""
    variant = f"reanalysis_volume_{member_id}"
    data_name = f"seaice_plus_auxiliary_{variant}"
    return ExperimentConfig(
        experiment_name=f"exp2_{variant}",
        notes=(
            "Inputs: same as input2. Reanalysis-volume contiguous temporal "
            f"split for CESM2-LE member {member_id}. Detrended data."
        ),
        data_name=data_name,
        data_split=time_split(
            data_name,
            member_ids=[member_id],
            **_REANALYSIS_VOLUME_SPLIT,
        ),
        input_config=deepcopy(INPUT_CONFIGS["input2"].input_config),
        target_config=deepcopy(_TARGET),
        learning_rate=1e-3,
        weight_decay=5e-2,
        batch_size=32,
        num_epochs=50,
        checkpoint_interval=10,
        patience=10,
        lr_scheduler="cosine",
        lr_scheduler_args={"t_max": 50, "eta_min": 5e-5},
    )


CONFIGS = {
    variant: _volume_config(variant, member_count)
    for variant, member_count in _TRAINING_VOLUMES.items()
}
CONFIGS.update({
    f"reanalysis_volume_{member_id}": _reanalysis_volume_config(member_id)
    for member_id in _REANALYSIS_VOLUME_MEMBERS
})

__all__ = ["CONFIGS"]
