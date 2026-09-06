"""Training-data-volume experiment variants."""

from copy import deepcopy

import pandas as pd

from src.config_cesm import AVAILABLE_CESM_MEMBERS
from src.experiment_configs.config import ExperimentConfig, ensemble_member_split
from src.experiment_configs.exp1_inputs import CONFIGS as INPUT_CONFIGS

_TARGET = {"predict_anom": True, "predict_classes": False}
_TRAINING_VOLUMES = {"vol1": 1, "vol2": 4, "vol3": 16, "vol4": 64}


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


CONFIGS = {
    variant: _volume_config(variant, member_count)
    for variant, member_count in _TRAINING_VOLUMES.items()
}

__all__ = ["CONFIGS"]
