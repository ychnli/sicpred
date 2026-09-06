from operator import lt
import xarray as xr
import numpy as np 
import os
import argparse
import pandas as pd

from src.utils import util_cesm
from src.utils import util_shared
from src.experiment_configs import load_config
from src import config_cesm
from src.models import baselines
from src.models.models_util import CESM_Dataset

REFERENCE_GRID = util_cesm.generate_sps_grid()
AREA_WEIGHTS = util_cesm.calculate_area_weights()

def get_ensemble_members_and_time_coords(data_split_settings, split):
    if data_split_settings["split_by"] == "ensemble_member":
        ensemble_members = data_split_settings[split]
        time_coords = data_split_settings["time_range"]
    elif data_split_settings["split_by"] == "time":
        ensemble_members = data_split_settings["member_ids"]
        time_coords = data_split_settings[split]

    return ensemble_members, time_coords


def load_model_predictions(config):
    """
    """
    output_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config.experiment_name)
    output_path = os.path.join(output_dir, f"{config.model}_{config.checkpoint_to_evaluate}_predictions.nc")
    predictions = xr.open_dataset(output_path).predictions 
    return predictions 


def load_targets(config, split):
    ensemble_members, time_coords = get_ensemble_members_and_time_coords(
        config.data_split, split
    )
    dataset = CESM_Dataset(split, config)
    member_arrays = []
    for member_id in ensemble_members:
        time_arrays = [
            dataset.target_data_array(member_id, start_prediction_month)
            for start_prediction_month in time_coords
        ]
        member_arrays.append(
            xr.concat(time_arrays, dim="start_prediction_month")
        )

    targets = xr.concat(member_arrays, dim="member_id").assign_coords(
        member_id=ensemble_members
    )
    return targets.transpose(
        "start_prediction_month", "member_id", "lead_time", "y", "x"
    ).load()


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

def reconstruct_sic_from_anomaly(anom, data_split_settings, save_path=None):
    """
    Reconstruct true sea ice concentration from quadratically detrended anomalies.
    
    The anomaly is computed as: anom = sic - monthly_mean - quadratic_trend
    So we reconstruct: sic = anom + quadratic_trend + monthly_mean
    
    Parameters:
    - anom (xr.DataArray): Sea ice concentration anomalies with dimensions 
                           (start_prediction_month, member_id, lead_time, y, x) or similar
    - data_split_settings (dict): Data split settings containing the name for finding saved files
    
    Returns:
    - xr.DataArray: Reconstructed sea ice concentration values
    """
    if save_path is not None and os.path.exists(save_path):
        return xr.open_dataset(save_path)["icefrac"]

    norm_dir = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", data_split_settings["name"])
    
    # Load monthly means
    monthly_mean = xr.open_dataset(os.path.join(norm_dir, "icefrac_mean.nc"))["icefrac"]
    detrend_coeffs = xr.open_dataset(os.path.join(norm_dir, "icefrac_detrend_coeffs.nc"))["icefrac"]
    
    def _add_months(dt64, months):
        ts = pd.Timestamp(dt64)  # dt64 is numpy datetime64[ns]
        return (ts + pd.DateOffset(months=int(months))).to_datetime64()

    valid_time = xr.apply_ufunc(_add_months, anom["start_prediction_month"], (anom["lead_time"] - 1), vectorize=True)
    valid_month = valid_time.dt.month
    
    # time coordinate used in trend calculation
    t = valid_time.dt.year + valid_time.dt.month / 12.0        
  
    mean_lt = monthly_mean.sel(month=valid_month)
    coeffs_lt = detrend_coeffs.sel(month=valid_month)
    const = coeffs_lt.sel(coeff="const")
    linear = coeffs_lt.sel(coeff="linear")
    quadratic = coeffs_lt.sel(coeff="quadratic")

    # Broadcast t to (start_prediction_month, lead_time, y, x)
    t_broadcast = t.broadcast_like(mean_lt)

    trend = const + linear * t_broadcast + quadratic * (t_broadcast ** 2)
    reconstructed = anom + mean_lt + trend
    if save_path is not None:
        reconstructed.to_dataset(name='icefrac').to_netcdf(save_path)
    return reconstructed


def calculate_iiee(pred_anom, truth_anom, data_split_settings, experiment_name, sic_threshold=0.15, save_abs=True):
    """
    Calculate the Integrated Ice Edge Error (IIEE) between predictions and truth.
    
    IIEE is defined as the total area of disagreement between binary ice masks,
    where ice is defined as SIC > threshold.
    
    Parameters:
    - pred_anom (xr.DataArray): Predicted sea ice concentration anomalies
    - truth_anom (xr.DataArray): True sea ice concentration anomalies
    - data_split_settings (dict): Data split settings for reconstructing true SIC values
    - sic_threshold (float): Threshold for defining ice presence (default 0.15)
    - aggregate (bool): If True, aggregate to (month, lead_time) dimensions
    
    Returns:
    - xr.DataArray: IIEE values (total area of disagreement in m^2)
    """

    save_path = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, experiment_name, "diagnostics/")

    # Reconstruct true SIC values from anomalies
    if save_abs:
        save_name_pred = os.path.join(save_path, "pred_abs.nc")
    else:
        save_name_pred = None
    pred_sic = reconstruct_sic_from_anomaly(pred_anom, data_split_settings, save_name_pred)
    truth_sic = reconstruct_sic_from_anomaly(truth_anom, data_split_settings, os.path.join(save_path, "truth_abs.nc"))
    
    # Create binary ice masks (ice where SIC > threshold)
    pred_ice_mask = pred_sic > sic_threshold
    truth_ice_mask = truth_sic > sic_threshold
    
    # Calculate disagreement (XOR of the two masks)
    disagreement = pred_ice_mask ^ truth_ice_mask
    
    # Get grid cell areas
    area = REFERENCE_GRID.area
    
    # Calculate IIEE as total area of disagreement
    iiee = (disagreement * area).sum(dim=("x", "y"))
        
    return iiee


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
    """
    Compute and save ice occurrence mask based on mean ice fraction. Any grid cell that
    has ice fraction > 0 at any time is marked as 1 in the mask, else 0. We will use this
    mask to mask out points that are trivially 0 when calculating metrics.

    Parameters:
    - data_source (str): "cesm" or "obs"

    Returns:
    - xr.DataArray: ice occurrence mask
    """
    if data_source not in ["cesm", "obs"]:
        raise ValueError(f"data_source should be one of 'cesm', 'obs', but was {data_source}")

    if data_source == "cesm":
        icefrac_ds = xr.open_dataset(os.path.join(config_cesm.DATA_DIRECTORY, "cesm_data/icefrac/icefrac_combined.nc"))
        save_name = os.path.join(config_cesm.DATA_DIRECTORY, "cesm_data/grids/ice_occurrence_mask.nc")
    if data_source == "obs":
        icefrac_ds = xr.open_dataset(os.path.join(config_cesm.DATA_DIRECTORY, "obs_data/icefrac_obs.nc"))
        save_name = os.path.join(config_cesm.DATA_DIRECTORY, "obs_data/ice_occurrence_mask.nc")

    if os.path.exists(save_name):
        return xr.open_dataset(save_name)["mask"]
    
    icefrac_mean = icefrac_ds["icefrac"].mean(("member_id", "time"))
    ice_occurrence_mask = (icefrac_mean > 0).astype(np.float32)
    ice_occurrence_mask.to_dataset(name="mask").to_netcdf(save_name)
    return ice_occurrence_mask


def main():
    parser = argparse.ArgumentParser(description="Compute diagnostics for model predictions.")
    parser.add_argument("--config", type=str, required=True, help="Named configuration selector (e.g., exp1_inputs:input2)")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing output files.")
    parser.add_argument("--baselines", action="store_true", help="If set, calculates persistence and climatology baselines too.")
    parser.add_argument("--ensemble-mean", action="store_true", help="If set, computes the ensemble mean prediction and calculates the diagnostics for it.")
    parser.add_argument("--predictions-path", type=str, default=None, help='Set this if you want to evaluate a checkpoint that is not the one specified in the config file')
    parser.add_argument("--label", type=str, default=None, help='Diagnostics will be saved as {diag}{label}_...')
    parser.add_argument("--permute-var", type=str, default=None, help="If set, permute the specified variable before computing diagnostics.")
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config.experiment_name)
    save_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config.experiment_name, "diagnostics")
    predictions_path = args.predictions_path
    os.makedirs(save_dir, exist_ok=True)

    targets = load_targets(config, split="test")

    if args.permute_var is not None:
        # TODO: remove the permute-var flag and merge this logic into using just predictions-path and label
        predictions_fp = os.path.join(
            config_cesm.PREDICTIONS_DIRECTORY, 
            config.experiment_name,
            "permute",
            f"permute_{args.permute_var}_predictions.nc"
        )
        print(predictions_fp)
        predictions = xr.open_dataset(predictions_fp)["predictions"]

        label = f"_permute_{args.permute_var}"
    else:
        if predictions_path is not None:
            assert os.path.exists(predictions_path)
            predictions = xr.open_dataset(predictions_path)["predictions"]
        else:
            predictions = load_model_predictions(config)
        
        if args.label is not None:
            label = args.label
        else:
            label = ""

    if args.ensemble_mean:
        # add an ensemble mean prediction with nn_member_id label -1
        # so that the diagnostics will be calculated for this forecast as well
        predictions = xr.concat(
            [predictions, predictions.mean("nn_member_id").expand_dims({"nn_member_id": [-1]})],
            dim='nn_member_id'
        )

    # mask out points that are always ice-free in the dataset
    if config.data_split["member_ids"] == ["obs"]:
        data_source = "obs"
    else:
        data_source = "cesm"
    ice_mask = compute_ice_mask(data_source=data_source)
    predictions = predictions.where(ice_mask == 1)
    targets = targets.where(ice_mask == 1)


    print(f"Computing diagnostics for {config.experiment_name}")

    if args.overwrite or not os.path.exists(os.path.join(save_dir, f"acc{label}.nc")):
        print("Computing ACC...")
        acc = calculate_acc(predictions, targets)
        acc_agg = aggregate_metric(acc, dim=("x","y"))
        util_shared.write_nc_file(acc.to_dataset(name="acc"), os.path.join(save_dir, f"acc{label}.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(acc_agg.to_dataset(name="acc"), os.path.join(save_dir, f"acc{label}_agg.nc"), overwrite=args.overwrite)
        print("done!\n")
    
    if args.overwrite or not os.path.exists(os.path.join(save_dir, f"rmse{label}.nc")):
        print("Computing RMSE...")
        rmse = calculate_rmse(predictions, targets)
        rmse_agg = aggregate_metric(rmse, dim=("x","y"))
        util_shared.write_nc_file(rmse.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(rmse_agg.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_agg.nc"), overwrite=args.overwrite)
        print("done!\n")

    if args.overwrite or not os.path.exists(os.path.join(save_dir, f"iiee{label}.nc")):
        print("Computing IIEE...")
        iiee = calculate_iiee(predictions, targets, config.data_split, config.experiment_name)
        iiee_agg = aggregate_metric(iiee, dim=("x","y"))
        util_shared.write_nc_file(iiee.to_dataset(name="iiee"), os.path.join(save_dir, f"iiee{label}.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(iiee_agg.to_dataset(name="iiee"), os.path.join(save_dir, f"iiee{label}_agg.nc"), overwrite=args.overwrite)
        print("done!\n")

    if args.baselines:
        print(f"Computing ACC, RMSE, and IIEE for persistence and climatology forecasts...")
        persistence_pred = baselines.anomaly_persistence(config.data_split, os.path.join(base_dir, "baselines"), overwrite=args.overwrite)
        persistence_pred = persistence_pred.where(ice_mask == 1)
        acc = calculate_acc(persistence_pred["predictions"], targets)
        acc_agg = aggregate_metric(acc, dim=("x","y"))
        util_shared.write_nc_file(acc.to_dataset(name="acc"), os.path.join(save_dir, f"acc{label}_persist.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(acc_agg.to_dataset(name="acc"), os.path.join(save_dir, f"acc{label}_agg_persist.nc"), overwrite=args.overwrite)
        
        rmse = calculate_rmse(persistence_pred["predictions"], targets)
        rmse_agg = aggregate_metric(rmse, dim=("x","y"))
        util_shared.write_nc_file(rmse.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_persist.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(rmse_agg.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_agg_persist.nc"), overwrite=args.overwrite)

        iiee = calculate_iiee(persistence_pred["predictions"], targets, config.data_split, config.experiment_name, save_abs=False)
        iiee_agg = aggregate_metric(iiee, dim=("x","y"))
        util_shared.write_nc_file(iiee.to_dataset(name="iiee"), os.path.join(save_dir, f"iiee{label}_persist.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(iiee_agg.to_dataset(name="iiee"), os.path.join(save_dir, f"iiee{label}_agg_persist.nc"), overwrite=args.overwrite)

        climatology_pred = xr.zeros_like(targets)
        climatology_pred = climatology_pred.where(ice_mask == 1)
        rmse = calculate_rmse(climatology_pred, targets)
        rmse_agg = aggregate_metric(rmse, dim=("x","y"))
        util_shared.write_nc_file(rmse.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_climatology.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(rmse_agg.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_agg_climatology.nc"), overwrite=args.overwrite)

        iiee = calculate_iiee(climatology_pred, targets, config.data_split, config.experiment_name, save_abs=False)
        iiee_agg = aggregate_metric(iiee, dim=("x","y"))
        util_shared.write_nc_file(iiee.to_dataset(name="iiee"), os.path.join(save_dir, f"iiee{label}_climatology.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(iiee_agg.to_dataset(name="iiee"), os.path.join(save_dir, f"iiee{label}_agg_climatology.nc"), overwrite=args.overwrite)

        print("done!\n")


if __name__ == "__main__":
    main()