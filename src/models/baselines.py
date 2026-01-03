import xarray as xr
import pandas as pd
import numpy as np
import os

from src import config_cesm
from src.utils import util_cesm
from src.utils import util_shared

def anomaly_persistence(data_split_settings, save_dir, max_lead_time=6, overwrite=False):
    """
    Computes the anomaly persistence baseline (carries forward the anomaly from some init month)
    on the test dataset. 

    This returns a dataset of the same format as the ML prediction models which makes comparing
    predictions a little easier. Note that start_prediction_month corresponds to the first month
    of the prediction, so the anomaly used for each prediction is taken from the month preceding
    the start_prediction_month. 

    Param:
        (dict)          data_split_settings: dictionary containing the data split settings 
        (str)           save_dir
    
    Returns:
        (xr.Dataset)    predictions: the anomaly persistence predictions
    """

    # Check if the file already exists
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, "persistence_predictions.nc")
        if os.path.exists(save_name) and not overwrite:
            print(f"Info: found pre-existing {save_name}")
            ds = xr.open_dataset(save_name)
            return ds

    # Load the data
    ds = xr.open_dataset(os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", data_split_settings["name"], "icefrac_norm.nc"))
    if data_split_settings["split_by"] == "ensemble_member":
        ensemble_members = data_split_settings["test"]
        time_coords_test = data_split_settings["time_range"]
    elif data_split_settings["split_by"] == "time":
        ensemble_members = data_split_settings["member_ids"]
        time_coords_test = data_split_settings["test"]

    # these are all times needed to compute the persistence forecast over the test period
    # the months are shifted back by 1 because each prediction requires the previous
    # month's anomaly
    time_coords = time_coords_test - pd.DateOffset(months=1)
    anom_da = ds["icefrac"].sel(time=time_coords, member_id=ensemble_members)
    
    # Initialize an empty xarray Dataset
    reference_grid = anom_da # this just needs to have x and y
    ds = util_cesm.generate_empty_predictions_ds(time_coords_test, ensemble_members, num_nn_members=1)

    for i, start_month in enumerate(time_coords_test):
        for j in range(max_lead_time):
            # init_month is the month before the prediction and is the anomaly we want to carry forward
            init_month = start_month - pd.DateOffset(months=1)
            pred = anom_da.sel(time=init_month).values
            ds["predictions"][i, :, 0, j, :, :] = pred
    
    # save 
    if save_dir is not None:
        util_shared.write_nc_file(ds, save_name, overwrite)
        
    return ds


def climatology(data_split_settings, save_dir, max_lead_time=6):
    """
    The climatology baseline model. 

    Param:
        (dict)          data_split_settings: dictionary containing the data split settings 
        (str)           save_dir
    
    Returns:
        (xr.Dataset)    predictions
    """
    
    # Check if the file already exists
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, "climatology_predictions.nc")
        if os.path.exists(save_name):
            print(f"Found pre-existing file with path {save_name}. Skipping...")
            return 

    # Load the data
    ds = xr.open_dataset(os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "normalized_inputs", data_split_settings["name"], "icefrac_mean.nc"))
    da_means = ds["icefrac"]
    reference_grid = da_means # this just needs to have x and y 
    ds = models_util.generate_empty_predictions_ds(reference_grid, time_coords, ensemble_members, max_lead_time, 80, 80)

    for i, start_month in enumerate(time_coords):
        for j in range(max_lead_time):
            pred_month = (start_month + pd.DateOffset(months=j))
            if data_split_settings["split_by"] == "ensemble_member":
                pred = da_means.sel(time=pred_month).values
            elif data_split_settings["split_by"] == "time":
                pred = da_means.sel(month=pred_month.month).values
                
            ds["predictions"][i, :, j, :, :] = pred

    # save 
    if save_dir is not None:
        ds.to_netcdf(save_name) 

    return ds