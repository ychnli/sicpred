import xarray as xr
import numpy as np 
import os
import argparse

from src.utils import util_cesm
from src.utils import util_shared
from src import config_cesm
from src.models import baselines

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
    output_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config["EXPERIMENT_NAME"])
    output_path = os.path.join(output_dir, f"{config['MODEL']}_{config['CHECKPOINT_TO_EVALUATE']}_predictions.nc")
    predictions = xr.open_dataset(output_path).predictions 
    return predictions 


def load_targets(config, split):
    ensemble_members, time_coords = get_ensemble_members_and_time_coords(config["DATA_SPLIT_SETTINGS"], split)
    data_dir = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "data_pairs", config["DATA_SPLIT_SETTINGS"]["name"])
    ds_list = []
    for member_id in ensemble_members:
        ds = xr.open_dataset(os.path.join(data_dir, f"targets_member_{member_id}.nc")).data.load()
        ds_list.append(ds)

    targets = xr.concat(ds_list, dim="member_id").sel(start_prediction_month=time_coords)
    targets = targets.transpose("start_prediction_month", "member_id", "lead_time", "y", "x")

    return targets


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
    # # Ensure dimensions match
    # if pred_anom.dims != truth_anom.dims:
    #     raise ValueError("Predictions and truth must have the same dimensions.")

    acc = xr.cov(pred_anom, truth_anom, dim=dim) / (pred_anom.std(dim=dim) * truth_anom.std(dim=dim))

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

def calculate_iiee(pred_anom, truth_anom):
    # TODO 
    return 


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


def main():
    parser = argparse.ArgumentParser(description="Compute diagnostics for model predictions.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., config.py)")
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrite existing output files.")
    parser.add_argument("--baselines", action="store_true", help="If set, calculates persistence and climatology baselines too.")
    parser.add_argument("--permute-var", type=str, default=None, help="If set, permute the specified variable before computing diagnostics.")
    args = parser.parse_args()

    config = util_shared.load_config(args.config)
    config_dict = util_shared.load_globals(config)
    base_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config_dict["EXPERIMENT_NAME"])
    save_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config_dict["EXPERIMENT_NAME"], "diagnostics")
    os.makedirs(save_dir, exist_ok=True)

    targets = load_targets(config_dict, split="test")

    if args.permute_var is not None:
        predictions_fp = os.path.join(
            config_cesm.PREDICTIONS_DIRECTORY, 
            config_dict["EXPERIMENT_NAME"], 
            "permute",
            f"permute_{args.permute_var}_predictions.nc"
        )
        print(predictions_fp)
        predictions = xr.open_dataset(predictions_fp)["predictions"]

        label = f"_permute_{args.permute_var}"
    else:
        predictions = load_model_predictions(config_dict)
        label = ""

    print(f"Computing diagnostics for {config_dict['EXPERIMENT_NAME']}")

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

    if args.baselines:
        print(f"Computing ACC and RMSE for persistence and climatology forecasts...")
        persistence_pred = baselines.anomaly_persistence(config_dict["DATA_SPLIT_SETTINGS"], os.path.join(base_dir, "baselines"))
        acc = calculate_acc(persistence_pred, targets)
        acc_agg = aggregate_metric(acc, dim=("x","y"))
        util_shared.write_nc_file(acc.to_dataset(name="acc"), os.path.join(save_dir, f"acc{label}_persist.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(acc_agg.to_dataset(name="acc"), os.path.join(save_dir, f"acc{label}_agg_persist.nc"), overwrite=args.overwrite)
        
        rmse = calculate_rmse(persistence_pred, targets)
        rmse_agg = aggregate_metric(rmse, dim=("x","y"))
        util_shared.write_nc_file(rmse.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_persist.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(rmse_agg.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_agg_persist.nc"), overwrite=args.overwrite)

        climatology_pred = xr.zeros_like(targets)
        rmse = calculate_rmse(climatology_pred, targets)
        rmse_agg = aggregate_metric(rmse, dim=("x","y"))
        util_shared.write_nc_file(rmse.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_climatology.nc"), overwrite=args.overwrite)
        util_shared.write_nc_file(rmse_agg.to_dataset(name="rmse"), os.path.join(save_dir, f"rmse{label}_agg_climatology.nc"), overwrite=args.overwrite)


if __name__ == "__main__":
    main()