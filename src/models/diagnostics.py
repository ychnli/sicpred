"""Compute ACC, RMSE, and IIEE diagnostics for a named experiment's predictions.

Command-line usage:
    --config SELECTOR          Required experiment configuration selector.
    --overwrite                Recompute and replace existing diagnostic files.
    --baselines                Also evaluate persistence and climatology baselines.
    --ensemble-mean            Also evaluate the ensemble-mean prediction.
    --predictions-path PATH    Use predictions from PATH instead of the configured checkpoint output.
    --label LABEL              Add LABEL to diagnostic output filenames.
    --permute-var CHANNEL      Read the corresponding permuted-prediction file and label outputs accordingly.

Examples:
    python -m src.models.diagnostics --config exp1_inputs:input3e
    python -m src.models.diagnostics --config exp1_inputs:input4 --baselines --ensemble-mean --overwrite
"""

import argparse
import os

import dask
import numpy as np
import pandas as pd
import xarray as xr

from src.utils import util_cesm
from src.utils import util_shared
from src.experiment_configs import load_config
from src import config_cesm
from src.models import baselines
from src.models.models_util import load_cesm_targets_data_array

REFERENCE_GRID = util_cesm.generate_sps_grid()
AREA_WEIGHTS = util_cesm.calculate_area_weights()
PREDICTION_CHUNKS = {
    "start_prediction_month": 12,
    "member_id": 1,
    "nn_member_id": -1,
    "lead_time": -1,
    "y": -1,
    "x": -1,
}
TARGET_CHUNKS = {
    "start_prediction_month": 12,
    "member_id": 1,
    "lead_time": -1,
    "y": -1,
    "x": -1,
}


def get_ensemble_members_and_time_coords(data_split_settings, split):
    if data_split_settings["split_by"] == "ensemble_member":
        ensemble_members = data_split_settings[split]
        time_coords = data_split_settings["time_range"]
    elif data_split_settings["split_by"] == "time":
        ensemble_members = data_split_settings["member_ids"]
        time_coords = data_split_settings[split]

    return ensemble_members, time_coords


def open_predictions(path):
    """Open a prediction artifact with memory-bounded logical chunks."""
    dataset = xr.open_dataset(path, chunks=PREDICTION_CHUNKS)
    predictions = dataset["predictions"]
    predictions.set_close(dataset.close)
    return predictions


def load_model_predictions(config):
    """Open predictions for the configured checkpoint."""
    output_dir = os.path.join(
        config_cesm.PREDICTIONS_DIRECTORY, config.experiment_name
    )
    output_path = os.path.join(
        output_dir,
        f"{config.model}_{config.checkpoint_to_evaluate}_predictions.nc",
    )
    return open_predictions(output_path)


def load_targets(config, split, *, data_source="dynamic"):
    """Open labeled targets using the diagnostic chunk layout."""
    return load_cesm_targets_data_array(
        split,
        config,
        data_source=data_source,
        chunks=TARGET_CHUNKS,
    )


def calculate_acc(pred_anom, truth_anom, aggregate=False, dim=("x","y")):
    """
    Calculate the Anomaly Correlation Coefficient (ACC) between predictions and truth.

    Parameters:
    - pred_anom  (xr.DataArray): Predicted anomalies
    - truth_anom (xr.DataArray): True anomalies
    - aggregate          (bool): if True, mean-aggregate down to dims (month, lead_time) 
    - dim                 (str): The dimension over which to calculate the ACC.
                                 Default are the spatial dimensions ("x","y").
    
    Returns:
    - xr.DataArray: ACC values with dimensions remaining after collapsing `dim`.
    """

    acc = xr.corr(pred_anom, truth_anom, dim=dim)

    if aggregate:
        acc = aggregate_metric(acc, dim)

    return acc


def calculate_rmse(pred_anom, truth_anom, aggregate=False):
    """
    Calculate the Root Mean Square Error (RMSE) between predictions and truth,
    weighted by area.

    Parameters:
    - pred_anom  (xr.DataArray): Predicted anomalies
    - truth_anom (xr.DataArray): True anomalies
    - aggregate          (bool): if True, mean-aggregate down to dims (month, lead_time)

    Returns:
    - xr.DataArray: RMSE values with dimensions remaining after collapsing `dim`.
    """
    rmse = np.sqrt((((pred_anom - truth_anom) ** 2) * AREA_WEIGHTS).sum(dim=("x","y")) / AREA_WEIGHTS.sum())

    if aggregate:
        rmse = aggregate_metric(rmse, dim=("x","y"))
    
    return rmse

def _valid_time_coordinates(anom):
    """Return valid datetimes for every initialization and lead."""
    def _add_months(dt64, months):
        timestamp = pd.Timestamp(dt64)
        return (
            timestamp + pd.DateOffset(months=int(months))
        ).to_datetime64()

    return xr.apply_ufunc(
        _add_months,
        anom["start_prediction_month"],
        anom["lead_time"] - 1,
        vectorize=True,
    )


def reconstruction_components(anom, data_split_settings):
    """Build monthly-mean and trend components used to recover SIC."""
    norm_dir = os.path.join(
        config_cesm.PROCESSED_DATA_DIRECTORY,
        "normalized_inputs",
        data_split_settings["name"],
    )
    with xr.open_dataset(
        os.path.join(norm_dir, "icefrac_mean.nc")
    ) as mean_ds:
        monthly_mean = mean_ds["icefrac"].load()
    with xr.open_dataset(
        os.path.join(norm_dir, "icefrac_detrend_coeffs.nc")
    ) as coeff_ds:
        detrend_coeffs = coeff_ds["icefrac"].load()

    monthly_mean = monthly_mean.chunk({"month": -1, "y": -1, "x": -1})
    detrend_coeffs = detrend_coeffs.chunk(
        {"month": -1, "coeff": -1, "y": -1, "x": -1}
    )
    valid_time = _valid_time_coordinates(anom)
    valid_month = valid_time.dt.month
    time_value = (
        valid_time.dt.year + valid_time.dt.month / 12.0
    ).chunk({"start_prediction_month": 12, "lead_time": -1})

    mean_by_valid_month = monthly_mean.sel(month=valid_month)
    coeffs_by_valid_month = detrend_coeffs.sel(month=valid_month)
    constant = coeffs_by_valid_month.sel(coeff="const")
    linear = coeffs_by_valid_month.sel(coeff="linear")
    quadratic = coeffs_by_valid_month.sel(coeff="quadratic")
    trend = constant + linear * time_value + quadratic * time_value ** 2
    chunk_layout = {
        "start_prediction_month": 12,
        "lead_time": -1,
        "y": -1,
        "x": -1,
    }
    return (
        mean_by_valid_month.chunk(chunk_layout),
        trend.chunk(chunk_layout),
    )


def reconstruct_sic_from_anomaly(anom, data_split_settings, save_path=None):
    """Reconstruct sea ice concentration from detrended anomalies."""
    monthly_mean, trend = reconstruction_components(
        anom, data_split_settings
    )
    reconstructed = anom + monthly_mean + trend
    if save_path is not None:
        reconstructed.to_dataset(name="icefrac").to_netcdf(save_path)
    return reconstructed


def calculate_iiee(
    pred_anom, truth_anom, data_split_settings, sic_threshold=0.15
):
    """Calculate area where predicted and observed binary ice masks differ."""
    monthly_mean, trend = reconstruction_components(
        pred_anom, data_split_settings
    )
    pred_ice_mask = pred_anom + monthly_mean + trend > sic_threshold
    truth_ice_mask = truth_anom + monthly_mean + trend > sic_threshold
    disagreement = pred_ice_mask ^ truth_ice_mask
    return (disagreement * REFERENCE_GRID.area).sum(dim=("x", "y"))


def roll_metric(metric):
    return xr.concat(
        [metric.roll(month=int(lt - 1), roll_coords=False).sel(lead_time=lt) for lt in metric.lead_time.values],
        dim="lead_time"
    )

def aggregate_metric(metric, dim):
    if dim == ("x", "y"):
        metric = metric.mean('member_id').groupby("start_prediction_month.month").mean("start_prediction_month")
    else: 
        raise ValueError()
        
    # roll in lead time so that it gets lined up
    return roll_metric(metric)


def compute_ice_mask(data_source):
    """Load or compute the mask of grid cells with nonzero ice occurrence."""
    if data_source == "cesm":
        source_path = os.path.join(
            config_cesm.DATA_DIRECTORY,
            "cesm_data/icefrac/icefrac_combined.nc",
        )
        save_name = os.path.join(
            config_cesm.DATA_DIRECTORY,
            "cesm_data/grids/ice_occurrence_mask.nc",
        )
    elif data_source == "obs":
        source_path = os.path.join(
            config_cesm.DATA_DIRECTORY, "obs_data/icefrac_obs.nc"
        )
        save_name = os.path.join(
            config_cesm.DATA_DIRECTORY, "obs_data/ice_occurrence_mask.nc"
        )
    else:
        raise ValueError(
            f"data_source should be one of 'cesm', 'obs', but was {data_source}"
        )

    if os.path.exists(save_name):
        with xr.open_dataset(save_name) as mask_ds:
            return mask_ds["mask"].load()

    with xr.open_dataset(source_path, chunks={"member_id": 1}) as icefrac_ds:
        icefrac_mean = icefrac_ds["icefrac"].mean(("member_id", "time"))
        ice_occurrence_mask = (icefrac_mean > 0).astype(np.float32).compute()
    ice_occurrence_mask.to_dataset(name="mask").to_netcdf(save_name)
    return ice_occurrence_mask


def _metric_output_paths(save_dir, metric_name, label, suffix=""):
    raw_path = os.path.join(save_dir, f"{metric_name}{label}{suffix}.nc")
    aggregate_path = os.path.join(
        save_dir, f"{metric_name}{label}_agg{suffix}.nc"
    )
    return raw_path, aggregate_path


def _write_metric_outputs(
    metric, metric_name, raw_path, aggregate_path, *, overwrite
):
    """Write an already-computed metric and any required aggregate."""
    if overwrite or not os.path.exists(raw_path):
        util_shared.write_nc_file(
            metric.to_dataset(name=metric_name), raw_path, overwrite=overwrite
        )
    if overwrite or not os.path.exists(aggregate_path):
        aggregate = aggregate_metric(metric, dim=("x", "y"))
        util_shared.write_nc_file(
            aggregate.to_dataset(name=metric_name),
            aggregate_path,
            overwrite=overwrite,
        )


def _repair_aggregate(metric_name, raw_path, aggregate_path):
    """Build a missing small aggregate without reopening prediction fields."""
    with xr.open_dataset(raw_path) as dataset:
        metric = dataset[metric_name].load()
    aggregate = aggregate_metric(metric, dim=("x", "y"))
    util_shared.write_nc_file(
        aggregate.to_dataset(name=metric_name),
        aggregate_path,
        overwrite=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute diagnostics for model predictions.")
    parser.add_argument("--config", type=str, required=True, help="Named configuration selector (e.g., exp1_inputs:input2)")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing output files.")
    parser.add_argument("--baselines", action="store_true", help="If set, calculates persistence and climatology baselines too.")
    parser.add_argument("--ensemble-mean", action="store_true", help="If set, computes the ensemble mean prediction and calculates the diagnostics for it.")
    parser.add_argument("--predictions-path", type=str, default=None, help="Set this to evaluate predictions outside the configured checkpoint output")
    parser.add_argument("--label", type=str, default=None, help="Diagnostics will be saved as {diag}{label}_...")
    parser.add_argument("--permute-var", type=str, default=None, help="If set, use predictions with the specified permuted variable.")
    parser.add_argument(
        "--data-source",
        choices=("dynamic", "precomputed"),
        default="dynamic",
        help="Load targets from normalized fields (default) or legacy pair files.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = os.path.join(
        config_cesm.PREDICTIONS_DIRECTORY, config.experiment_name
    )
    save_dir = os.path.join(base_dir, "diagnostics")
    os.makedirs(save_dir, exist_ok=True)

    if args.permute_var is not None:
        predictions_path = os.path.join(
            base_dir,
            "permute",
            f"permute_{args.permute_var}_predictions.nc",
        )
        label = f"_permute_{args.permute_var}"
    else:
        predictions_path = args.predictions_path
        label = args.label or ""

    metric_paths = {
        name: _metric_output_paths(save_dir, name, label)
        for name in ("acc", "rmse", "iiee")
    }
    metrics_to_compute = []
    for metric_name, (raw_path, aggregate_path) in metric_paths.items():
        if args.overwrite or not os.path.exists(raw_path):
            metrics_to_compute.append(metric_name)
        elif not os.path.exists(aggregate_path):
            print(f"Repairing missing aggregate {aggregate_path}", flush=True)
            _repair_aggregate(metric_name, raw_path, aggregate_path)

    if not metrics_to_compute and not args.baselines:
        print(f"All diagnostics already exist for {config.experiment_name}")
        return

    targets_source = load_targets(
        config,
        split="test",
        data_source=args.data_source,
    )
    if predictions_path is not None:
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(predictions_path)
        predictions_source = open_predictions(predictions_path)
    else:
        predictions_source = load_model_predictions(config)

    try:
        predictions = predictions_source
        if args.ensemble_mean:
            predictions = xr.concat(
                [
                    predictions,
                    predictions.mean("nn_member_id").expand_dims(
                        {"nn_member_id": [-1]}
                    ),
                ],
                dim="nn_member_id",
            )

        data_source = (
            "obs"
            if config.data_split["member_ids"] == ["obs"]
            else "cesm"
        )
        ice_mask = compute_ice_mask(data_source=data_source)
        predictions = predictions.where(ice_mask == 1)
        targets = targets_source.where(ice_mask == 1)

        print(
            f"Computing {', '.join(name.upper() for name in metrics_to_compute)} "
            f"for {config.experiment_name}",
            flush=True,
        )
        metrics = {}
        if "acc" in metrics_to_compute:
            metrics["acc"] = calculate_acc(predictions, targets)
        if "rmse" in metrics_to_compute:
            metrics["rmse"] = calculate_rmse(predictions, targets)
        if "iiee" in metrics_to_compute:
            metrics["iiee"] = calculate_iiee(
                predictions, targets, config.data_split
            )

        computed_values = dask.compute(*metrics.values())
        computed = dict(zip(metrics, computed_values))
        for metric_name in metrics_to_compute:
            raw_path, aggregate_path = metric_paths[metric_name]
            _write_metric_outputs(
                computed[metric_name],
                metric_name,
                raw_path,
                aggregate_path,
                overwrite=args.overwrite,
            )
            print(f"Finished {metric_name.upper()}", flush=True)

        if args.baselines:
            print(
                "Computing persistence and climatology diagnostics...",
                flush=True,
            )
            persistence_pred = baselines.anomaly_persistence(
                config.data_split,
                os.path.join(base_dir, "baselines"),
                overwrite=args.overwrite,
            )["predictions"].where(ice_mask == 1)
            persistence_metrics = {
                "acc": calculate_acc(persistence_pred, targets),
                "rmse": calculate_rmse(persistence_pred, targets),
                "iiee": calculate_iiee(
                    persistence_pred, targets, config.data_split
                ),
            }
            persistence_values = dask.compute(*persistence_metrics.values())
            persistence_metrics = dict(
                zip(persistence_metrics, persistence_values)
            )
            for metric_name in persistence_metrics:
                raw_path, aggregate_path = _metric_output_paths(
                    save_dir, metric_name, label, suffix="_persist"
                )
                _write_metric_outputs(
                    persistence_metrics[metric_name],
                    metric_name,
                    raw_path,
                    aggregate_path,
                    overwrite=args.overwrite,
                )

            climatology_pred = xr.zeros_like(targets)
            climatology_metrics = {
                "rmse": calculate_rmse(climatology_pred, targets),
                "iiee": calculate_iiee(
                    climatology_pred, targets, config.data_split
                ),
            }
            climatology_values = dask.compute(*climatology_metrics.values())
            climatology_metrics = dict(
                zip(climatology_metrics, climatology_values)
            )
            for metric_name in climatology_metrics:
                raw_path, aggregate_path = _metric_output_paths(
                    save_dir, metric_name, label, suffix="_climatology"
                )
                _write_metric_outputs(
                    climatology_metrics[metric_name],
                    metric_name,
                    raw_path,
                    aggregate_path,
                    overwrite=args.overwrite,
                )
            print("Finished baselines", flush=True)
    finally:
        predictions_source.close()
        targets_source.close()


if __name__ == "__main__":
    main()
