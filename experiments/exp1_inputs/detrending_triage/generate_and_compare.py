#!/usr/bin/env python3
"""Generate training-only-detrended fields and compare with legacy outputs.

This is an isolated experiment helper. It intentionally does not call the full
preprocessing pipeline or write model-ready input/target pairs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir()
)
sys.path.insert(0, str(REPO_ROOT))

from src import config_cesm  # noqa: E402
from src.experiment_configs import load_config  # noqa: E402
from src.utils.util_shared import write_nc_file  # noqa: E402


LEGACY_VARIANTS = ("input2", "input3a", "input3b", "input3c", "input3d", "input4")
EXPECTED_VARIABLES = ("icefrac", "sst", "psl", "z500", "t2m")
SPLITS = ("train", "val", "test")
DIFFERENCE_THRESHOLDS = (1e-6, 1e-3, 1e-1, 1.0, 1e3, 1e6)
DEFAULT_OUTPUT_DIR = (
    Path(config_cesm.PROCESSED_DATA_DIRECTORY)
    / "normalization_triage"
    / "exp1_legacy_train_only_detrend"
)


def included_physical_inputs(config) -> dict[str, dict]:
    """Return enabled, normalized, non-auxiliary input recipes."""
    return {
        name: settings
        for name, settings in config.input_config.items()
        if settings["include"] and settings["norm"] and not settings["auxiliary"]
    }


def resolve_recipe(variable: str):
    """Resolve and validate the shared recipe for one legacy variable."""
    configs = {variant: load_config(f"exp1_inputs:{variant}") for variant in LEGACY_VARIANTS}
    users = [
        variant
        for variant, config in configs.items()
        if variable in included_physical_inputs(config)
    ]
    if not users:
        raise ValueError(f"Variable {variable!r} is not used by any legacy configuration")

    reference = configs[users[0]]
    reference_settings = included_physical_inputs(reference)[variable]
    reference_split = reference.data_split
    for variant in users[1:]:
        config = configs[variant]
        settings = included_physical_inputs(config)[variable]
        if settings != reference_settings:
            raise ValueError(f"Normalization recipe differs for {variable}: {users[0]} vs {variant}")
        for split in SPLITS:
            if config.data_split[split] != reference_split[split]:
                raise ValueError(f"Member split differs for {variable}: {users[0]} vs {variant}")
        if not config.data_split["time_range"].equals(reference_split["time_range"]):
            raise ValueError(f"Time range differs for {variable}: {users[0]} vs {variant}")

    return configs, users, reference, reference_settings


def detrend_quadratic_train_only(
    data: xr.DataArray,
    train_members: list[str],
    time_dim: str = "time",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Fit monthly quadratic trends on training members and apply to all members.

    This preserves the legacy algorithm (a quadratic fit to the ensemble mean
    for each calendar month) while changing only the members used for fitting.
    """
    if "member_id" not in data.dims:
        raise ValueError("Training-member detrending requires a member_id dimension")
    missing = sorted(set(train_members) - set(data.member_id.values.astype(str)))
    if missing:
        raise ValueError(f"Training members missing from data: {missing}")

    time_values = data[time_dim].dt.year + data[time_dim].dt.month / 12.0
    time_numeric = xr.DataArray(
        time_values,
        coords={time_dim: data[time_dim]},
        dims=time_dim,
    )
    spatial_dims = [dim for dim in data.dims if dim not in {time_dim, "member_id"}]
    all_stacked = data.stack(space=spatial_dims)
    train_stacked = all_stacked.sel(member_id=train_members)
    detrended = all_stacked.copy()
    coefficients = np.full(
        (12, 3, all_stacked.sizes["space"]),
        np.nan,
        dtype=data.dtype,
    )

    for month in range(1, 13):
        selected = all_stacked[time_dim].dt.month == month
        if not bool(selected.any()):
            continue
        fit_time = time_numeric.sel({time_dim: selected})
        fit_values = train_stacked.sel({time_dim: selected}).mean("member_id")
        design = np.stack(
            [np.ones_like(fit_time), fit_time, fit_time**2],
            axis=1,
        )
        beta = np.linalg.solve(design.T @ design, design.T @ fit_values.values)
        coefficients[month - 1] = beta
        trend = xr.DataArray(
            (design @ beta).astype(fit_values.dtype),
            coords=fit_values.coords,
            dims=fit_values.dims,
        )
        detrended.loc[{time_dim: selected}] = all_stacked.sel(
            {time_dim: selected}
        ) - trend

    coefficient_array = xr.DataArray(
        coefficients,
        dims=("month", "coeff", "space"),
        coords={
            "month": np.arange(1, 13),
            "coeff": ["const", "linear", "quadratic"],
            "space": all_stacked.coords["space"],
        },
        name=data.name,
    ).unstack("space")
    return detrended.unstack("space"), coefficient_array


def normalize_and_detrend(variable: str, reference, settings):
    """Reproduce legacy scaling, then fit detrending on training members only."""
    split = reference.data_split
    member_ids = split["train"] + split["val"] + split["test"]
    prediction_times = split["time_range"]
    all_times = pd.date_range(
        prediction_times[0] - pd.DateOffset(months=settings["lag"]),
        prediction_times[-1] + pd.DateOffset(months=reference.max_lead_months - 1),
        freq="MS",
    )
    raw_path = (
        Path(config_cesm.DATA_DIRECTORY)
        / "cesm_data"
        / variable
        / f"{variable}_combined.nc"
    )
    with xr.open_dataset(raw_path) as raw_dataset:
        data = raw_dataset[variable].sel(member_id=member_ids, time=all_times)
        train_data = data.sel(member_id=split["train"])
        reduction_dims = [dim for dim in ("time", "member_id") if dim in data.dims]
        months = data.time.dt.month

        if settings["divide_by_stdev"]:
            monthly_center = train_data.groupby("time.month").mean(reduction_dims).load()
            monthly_scale = train_data.groupby("time.month").std(reduction_dims).load()
            with np.errstate(divide="ignore", invalid="ignore"):
                scaled = (data - monthly_center.sel(month=months)) / monthly_scale.sel(month=months)
                scaled = scaled.where(monthly_scale.sel(month=months) != 0, 0)
        elif settings["use_min_max"]:
            monthly_min = train_data.groupby("time.month").min(reduction_dims).load()
            monthly_max = train_data.groupby("time.month").max(reduction_dims).load()
            with np.errstate(divide="ignore", invalid="ignore"):
                scaled = (data - monthly_min.sel(month=months)) / (
                    monthly_max.sel(month=months) - monthly_min.sel(month=months)
                )
        else:
            monthly_center = train_data.groupby("time.month").mean(reduction_dims).load()
            scaled = data - monthly_center.sel(month=months)

        scaled = scaled.load()

    detrended, coefficients = detrend_quadratic_train_only(
        scaled,
        list(split["train"]),
    )
    detrended.name = variable
    coefficients.name = variable
    return detrended.transpose(*scaled.dims), coefficients


def finite_max_abs(values: np.ndarray) -> float:
    """Return maximum absolute finite value, or NaN when none are finite."""
    finite = np.isfinite(values)
    return float(np.max(np.abs(values[finite]))) if finite.any() else float("nan")


def summarize_split(old_values: np.ndarray, new_values: np.ndarray) -> dict[str, float | int]:
    """Calculate exact scalar comparison statistics for one data split."""
    old_finite = np.isfinite(old_values)
    new_finite = np.isfinite(new_values)
    paired = old_finite & new_finite
    paired_difference = new_values[paired].astype(np.float64) - old_values[paired]
    absolute_difference = np.abs(paired_difference)
    old_absolute = np.abs(old_values[old_finite].astype(np.float64))
    new_absolute = np.abs(new_values[new_finite].astype(np.float64))
    old_rms = float(np.sqrt(np.mean(old_absolute**2))) if old_absolute.size else float("nan")
    new_rms = float(np.sqrt(np.mean(new_absolute**2))) if new_absolute.size else float("nan")
    difference_rmse = (
        float(np.sqrt(np.mean(paired_difference**2)))
        if paired_difference.size
        else float("nan")
    )
    result: dict[str, float | int] = {
        "total_values": int(old_values.size),
        "old_finite_values": int(old_finite.sum()),
        "new_finite_values": int(new_finite.sum()),
        "finite_status_mismatches": int(np.logical_xor(old_finite, new_finite).sum()),
        "old_max_abs": finite_max_abs(old_values),
        "new_max_abs": finite_max_abs(new_values),
        "old_mean_abs": float(old_absolute.mean()) if old_absolute.size else float("nan"),
        "new_mean_abs": float(new_absolute.mean()) if new_absolute.size else float("nan"),
        "old_rms": old_rms,
        "new_rms": new_rms,
        "difference_mae": float(absolute_difference.mean()) if paired_difference.size else float("nan"),
        "difference_rmse": difference_rmse,
        "difference_rmse_fraction_of_old_rms": difference_rmse / old_rms if old_rms else float("nan"),
        "difference_max_abs": float(absolute_difference.max()) if paired_difference.size else float("nan"),
    }
    for threshold in DIFFERENCE_THRESHOLDS:
        label = f"difference_abs_gt_{threshold:g}"
        result[label] = int((absolute_difference > threshold).sum())
    for threshold in (1.0, 2.0, 5.0, 10.0, 1e2, 1e6, 1e10):
        label = f"old_abs_gt_{threshold:g}"
        result[label] = int((np.abs(old_values[old_finite]) > threshold).sum())
        label = f"new_abs_gt_{threshold:g}"
        result[label] = int((np.abs(new_values[new_finite]) > threshold).sum())
    return result


def spatial_maps(old_values: np.ndarray, new_values: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce member/time axes to compact spatial comparison maps."""
    paired = np.isfinite(old_values) & np.isfinite(new_values)
    difference = np.where(paired, new_values - old_values, np.nan)
    paired_count = paired.sum(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        max_abs_difference = np.nanmax(np.abs(difference), axis=(0, 1))
        rmse_difference = np.sqrt(np.nansum(difference.astype(np.float64) ** 2, axis=(0, 1)) / paired_count)
        old_max_abs = np.nanmax(np.where(np.isfinite(old_values), np.abs(old_values), np.nan), axis=(0, 1))
        new_max_abs = np.nanmax(np.where(np.isfinite(new_values), np.abs(new_values), np.nan), axis=(0, 1))
    max_abs_difference[paired_count == 0] = np.nan
    rmse_difference[paired_count == 0] = np.nan
    return {
        "max_abs_difference": max_abs_difference.astype(np.float32),
        "rmse_difference": rmse_difference.astype(np.float32),
        "old_max_abs": old_max_abs.astype(np.float32),
        "new_max_abs": new_max_abs.astype(np.float32),
        "paired_finite_count": paired_count.astype(np.int32),
    }


def compare_legacy_outputs(variable, new_data, configs, users, output_dir):
    """Compare one canonical corrected field with every legacy config using it."""
    rows = []
    map_datasets = []
    for variant in users:
        config = configs[variant]
        legacy_path = (
            Path(config_cesm.PROCESSED_DATA_DIRECTORY)
            / "normalized_inputs"
            / config.data_name
            / f"{variable}_norm.nc"
        )
        print(f"Comparing {variable} with {variant}: {legacy_path}", flush=True)
        with xr.open_dataset(legacy_path) as legacy_dataset:
            legacy_data = legacy_dataset[variable].load()
        legacy_data, aligned_new = xr.align(legacy_data, new_data, join="exact")

        split_map_datasets = []
        for split in SPLITS:
            member_ids = config.data_split[split]
            old_values = legacy_data.sel(member_id=member_ids).values
            new_values = aligned_new.sel(member_id=member_ids).values
            row = {"configuration": variant, "variable": variable, "split": split}
            row.update(summarize_split(old_values, new_values))
            rows.append(row)

            coordinates = {"y": legacy_data.y, "x": legacy_data.x}
            split_maps = xr.Dataset(
                {
                    name: (("y", "x"), values)
                    for name, values in spatial_maps(old_values, new_values).items()
                },
                coords=coordinates,
            ).expand_dims(split=[split])
            split_map_datasets.append(split_maps)

        map_datasets.append(
            xr.concat(split_map_datasets, dim="split").expand_dims(configuration=[variant])
        )
        del legacy_data
        gc.collect()

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"{variable}_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    maps = xr.concat(map_datasets, dim="configuration")
    maps.attrs.update(
        variable=variable,
        comparison="legacy all-member detrending minus training-member-only detrending",
    )
    write_nc_file(
        maps,
        str(output_dir / f"{variable}_comparison_maps.nc"),
        overwrite=True,
    )
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", choices=EXPECTED_VARIABLES, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs, users, reference, settings = resolve_recipe(args.variable)
    output_path = output_dir / f"{args.variable}_norm_train_only_detrend.nc"
    coefficient_path = output_dir / f"{args.variable}_detrend_coeffs_train_only.nc"

    if output_path.exists() and coefficient_path.exists() and not args.overwrite:
        print(f"Reusing {output_path}", flush=True)
        with xr.open_dataset(output_path) as dataset:
            corrected = dataset[args.variable].load()
    else:
        print(
            f"Generating {args.variable} for configurations {users}; "
            f"detrending fit members={reference.data_split['train']}",
            flush=True,
        )
        corrected, coefficients = normalize_and_detrend(args.variable, reference, settings)
        common_attributes = {
            "triage_only": "true",
            "scaling_fit_partition": "train",
            "detrending_fit_partition": "train",
            "detrending_fit_members": ",".join(reference.data_split["train"]),
            "covered_legacy_configurations": ",".join(users),
        }
        corrected.attrs.update(common_attributes)
        coefficients.attrs.update(common_attributes)
        write_nc_file(
            corrected.to_dataset(name=args.variable),
            str(output_path),
            overwrite=args.overwrite,
            verbose=2,
        )
        write_nc_file(
            coefficients.to_dataset(name=args.variable),
            str(coefficient_path),
            overwrite=args.overwrite,
            verbose=2,
        )

    summary_path = compare_legacy_outputs(
        args.variable,
        corrected,
        configs,
        users,
        output_dir,
    )
    metadata = {
        "variable": args.variable,
        "covered_legacy_configurations": users,
        "training_members": reference.data_split["train"],
        "validation_members": reference.data_split["val"],
        "test_members": reference.data_split["test"],
        "normalization_settings": settings,
        "normalized_output": str(output_path),
        "coefficient_output": str(coefficient_path),
        "comparison_summary": str(summary_path),
    }
    metadata_path = output_dir / f"{args.variable}_metadata.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_metadata, metadata_path)
    print(f"Completed {args.variable}; artifacts in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
