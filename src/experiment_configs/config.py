"""Shared experiment configuration types and loading utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    """A fully resolved configuration for one experiment run."""

    experiment_name: str
    notes: str
    data_name: str
    data_split: dict[str, Any]
    input_config: dict[str, dict[str, Any]]
    target_config: dict[str, Any]
    max_lead_months: int = 6
    model: str = "UNetRes3"
    model_args: dict[str, Any] = field(
        default_factory=lambda: {"n_channels_factor": 0.5}
    )
    loss_function: str = "MSE"
    loss_function_args: dict[str, Any] = field(default_factory=dict)
    optimizer: str = "adamw"
    optimizer_args: dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    num_epochs: int = 10
    checkpoint_interval: int = 1
    patience: int = 3
    lr_scheduler: str | None = None
    lr_scheduler_args: dict[str, Any] = field(default_factory=dict)
    checkpoint_to_evaluate: str = "best"
    date: str = ""


def ensemble_member_split(
    data_name: str,
    *,
    train: Sequence[str],
    val: Sequence[str],
    test: Sequence[str],
    time_range: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Build a split whose train/validation/test partitions are member IDs."""

    return {
        "name": data_name,
        "split_by": "ensemble_member",
        "train": list(train),
        "val": list(val),
        "test": list(test),
        "time_range": time_range.copy(),
        "member_ids": None,
    }


def time_split(
    data_name: str,
    *,
    train: pd.DatetimeIndex,
    val: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    member_ids: Sequence[str],
) -> dict[str, Any]:
    """Build a split whose partitions are prediction initialization dates.

    The date ranges may be contiguous or disjoint. This makes a resolved fold from
    a future cross-validation split generator usable without changing consumers.
    """

    return {
        "name": data_name,
        "split_by": "time",
        "train": train.copy(),
        "val": val.copy(),
        "test": test.copy(),
        "time_range": None,
        "member_ids": list(member_ids),
    }


def validate_config(config: ExperimentConfig) -> None:
    """Validate invariants shared by preprocessing, training, and evaluation."""

    if not config.experiment_name:
        raise ValueError("experiment_name must not be empty")
    if not config.data_name:
        raise ValueError("data_name must not be empty")
    if config.data_split.get("name") != config.data_name:
        raise ValueError("data_split['name'] must match data_name")

    split_by = config.data_split.get("split_by")
    if split_by not in {"time", "ensemble_member"}:
        raise ValueError(f"Unsupported split_by={split_by!r}")

    partitions = [config.data_split.get(key) for key in ("train", "val", "test")]
    if any(partition is None or len(partition) == 0 for partition in partitions):
        raise ValueError("train, val, and test partitions must all be non-empty")

    partition_sets = [set(partition) for partition in partitions]
    if any(
        partition_sets[left] & partition_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("train, val, and test partitions must be disjoint")

    if split_by == "time":
        if not config.data_split.get("member_ids"):
            raise ValueError("time splits require member_ids")
        if config.data_split.get("time_range") is not None:
            raise ValueError("time splits must not define time_range")
    else:
        if config.data_split.get("member_ids") is not None:
            raise ValueError("ensemble-member splits must not define member_ids")
        if config.data_split.get("time_range") is None:
            raise ValueError("ensemble-member splits require time_range")

    if not config.input_config:
        raise ValueError("input_config must not be empty")
    for variable, settings in config.input_config.items():
        if "include" not in settings or "auxiliary" not in settings:
            raise ValueError(
                f"Input {variable!r} must define include and auxiliary settings"
            )


def load_config(selector: str) -> ExperimentConfig:
    """Load a named config using a ``family:variant`` selector."""

    try:
        family, variant = selector.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid config selector {selector!r}; expected 'family:variant'"
        ) from exc

    if not family or not variant:
        raise ValueError(
            f"Invalid config selector {selector!r}; expected 'family:variant'"
        )

    try:
        module = import_module(f"src.experiment_configs.{family}")
    except ModuleNotFoundError as exc:
        expected_module = f"src.experiment_configs.{family}"
        if exc.name != expected_module:
            raise
        raise ValueError(f"Unknown experiment family {family!r}") from exc

    configs: Mapping[str, ExperimentConfig] | None = getattr(module, "CONFIGS", None)
    if configs is None:
        raise ValueError(f"Experiment family {family!r} does not define CONFIGS")
    if variant not in configs:
        available = ", ".join(sorted(configs))
        raise ValueError(
            f"Unknown variant {variant!r} for {family!r}; available: {available}"
        )

    config = deepcopy(configs[variant])
    validate_config(config)
    return config
