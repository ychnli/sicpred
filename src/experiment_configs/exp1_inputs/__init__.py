"""Input-ablation experiment variants."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from src.config_cesm import AVAILABLE_CESM_MEMBERS
from src.experiment_configs.config import ExperimentConfig, ensemble_member_split

_TARGET = {"predict_anom": True, "predict_classes": False}
_TIME_RANGE = pd.date_range("1851-01", "2013-12", freq="MS")


def _inputs(*additional_inputs: str) -> dict[str, dict]:
    """Build the common input recipe with selected predictors enabled."""
    enabled = {"icefrac", *additional_inputs}
    inputs = {
        "icefrac": {"include": True, "norm": True, "land_mask": True, "lag": 12,
                    "divide_by_stdev": False, "auxiliary": False, "use_min_max": False},
        "icethick": {"include": "icethick" in enabled, "norm": True,
                     "land_mask": True, "lag": 6, "divide_by_stdev": False,
                     "auxiliary": False, "use_min_max": True},
        "sst": {"include": "sst" in enabled, "norm": True, "land_mask": True, "lag": 6,
                "divide_by_stdev": False, "auxiliary": False, "use_min_max": True},
        "ohc200": {"include": "ohc200" in enabled, "norm": True,
                   "land_mask": True, "lag": 6, "divide_by_stdev": False,
                   "auxiliary": False, "use_min_max": True},
        "z500": {"include": "z500" in enabled, "norm": True,
                 "land_mask": False, "lag": 6, "divide_by_stdev": False,
                 "auxiliary": False, "use_min_max": True},
        "z50": {"include": "z50" in enabled, "norm": True,
                "land_mask": False, "lag": 6, "divide_by_stdev": False,
                "auxiliary": False, "use_min_max": True},
        "psl": {"include": "psl" in enabled, "norm": True, "land_mask": False, "lag": 6,
                "divide_by_stdev": False, "auxiliary": False, "use_min_max": True},
        "t2m": {"include": "t2m" in enabled, "norm": True, "land_mask": False, "lag": 6,
                "divide_by_stdev": False, "auxiliary": False, "use_min_max": True},
        "cosine_of_init_month": {"include": True, "norm": False, "auxiliary": True},
        "sine_of_init_month": {"include": True, "norm": False, "auxiliary": True},
        "land_mask": {"include": True, "norm": False, "auxiliary": True},
    }
    unknown = set(additional_inputs) - set(inputs)
    if unknown:
        raise ValueError(f"Unknown input variables: {sorted(unknown)}")
    return inputs


def _active_config(variant: str, data_name: str, notes: str,
                   *additional_inputs: str) -> ExperimentConfig:
    split = ensemble_member_split(
        data_name,
        train=AVAILABLE_CESM_MEMBERS[0:8],
        val=AVAILABLE_CESM_MEMBERS[8:10],
        test=AVAILABLE_CESM_MEMBERS[10:14],
        time_range=_TIME_RANGE,
    )
    return ExperimentConfig(
        experiment_name=f"exp1_{variant}", notes=notes, data_name=data_name,
        data_split=split, input_config=_inputs(*additional_inputs),
        target_config=deepcopy(_TARGET),
        weight_decay=1e-3 if variant == "input2" else 5e-3,
    )


# Only these members were used by the inactive development configs. Keeping the
# local ordering preserves their historical splits after the global list changed.
_LEGACY_MEMBERS = [
    "r10i1181p1f1", "r10i1231p1f1", "r10i1251p1f1", "r10i1281p1f1",
    "r10i1301p1f1", "r1i1001p1f1", "r1i1231p1f1", "r1i1251p1f1",
    "r1i1281p1f1", "r1i1301p1f1", "r2i1021p1f1", "r2i1231p1f1",
    "r2i1251p1f1", "r2i1281p1f1", "r2i1301p1f1", "r3i1041p1f1",
]


def _legacy_inputs(*, divide_by_stdev: bool) -> dict[str, dict]:
    """Preserve the schema used by the inactive development variants."""
    inputs = _inputs()
    inputs.pop("sst")
    inputs.pop("t2m")
    inputs["icethick"] = {
        "include": False, "norm": True, "land_mask": True, "lag": 12,
        "divide_by_stdev": False, "auxiliary": False, "use_min_max": True,
    }
    inputs["temp"] = {
        "include": True, "norm": True, "land_mask": True, "lag": 6,
        "divide_by_stdev": divide_by_stdev, "auxiliary": False, "use_min_max": False,
    }
    for name in ("lw_flux", "sw_flux", "ua"):
        inputs[name] = {
            "include": False, "norm": True, "land_mask": False, "lag": 3,
            "divide_by_stdev": False, "auxiliary": False, "use_min_max": True,
        }
    return inputs


def _legacy_config(variant: str, data_name: str, notes: str,
                   *, divide_by_stdev: bool) -> ExperimentConfig:
    split = ensemble_member_split(
        data_name,
        train=_LEGACY_MEMBERS[0:8], val=_LEGACY_MEMBERS[8:10],
        test=_LEGACY_MEMBERS[12:16], time_range=_TIME_RANGE,
    )
    return ExperimentConfig(
        experiment_name=f"exp1_{variant}", notes=notes, data_name=data_name,
        data_split=split,
        input_config=_legacy_inputs(divide_by_stdev=divide_by_stdev),
        target_config=deepcopy(_TARGET),
        loss_function_args={
            "apply_month_weights": True,
            "monthly_weights": {"data_split_settings": split, "use_softmax": True, "T": 2},
            "apply_area_weights": True,
            "l2_lambda": 0,
        },
        checkpoint_to_evaluate="epoch_10",
    )


CONFIGS = {
    "input2": _active_config(
        "input2", "seaice_plus_auxiliary",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month"),
    "input3a": _active_config(
        "input3a", "seaice_plus_sst",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of SST",
        "sst"),
    "input3b": _active_config(
        "input3b", "seaice_plus_psl",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of psl",
        "psl"),
    "input3c": _active_config(
        "input3c", "seaice_plus_z500",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of z500",
        "z500"),
    "input3d": _active_config(
        "input3d", "seaice_plus_t2m",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of t2m",
        "t2m"),
    "input3e": _active_config(
        "input3e", "seaice_plus_z50",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of z50",
        "z50"),
    "input3f": _active_config(
        "input3f", "seaice_plus_ohc200",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of top-200-m ocean heat content",
        "ohc200"),
    "input3g": _active_config(
        "input3g", "seaice_plus_icethick",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of sea ice thickness",
        "icethick"),
    "input4": _active_config(
        "input4", "seaice_plus_all",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month      + 6 months of SST + atmospheric vars",
        "sst", "z500", "psl", "t2m"),
    "input3a_dev": _legacy_config(
        "input3a_dev", "seaice_plus_temp_dev",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of SST",
        divide_by_stdev=False),
    "input3a_std": _legacy_config(
        "input3a_std", "seaice_plus_temp_std",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month + 6 months of SST",
        divide_by_stdev=True),
    "input_noise": _legacy_config(
        "noise", "seaice_plus_noise",
        "Previous 12 months of sea ice + land mask and sin() and cos() of month +     6 channels of unit gaussian noise",
        divide_by_stdev=True),
}

__all__ = ["CONFIGS"]
