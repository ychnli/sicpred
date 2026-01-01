"""
Regrid era5 data so that it looks like the 80x80 polar projection
that is used for cesm data. Then normalizes the data according to
the data split and constructs the inputs. 
"""

import xarray as xr
import xesmf as xe
import os
import numpy as np 
import pandas as pd 
import argparse 

from src.utils import util_cesm
from src import config_cesm
import src.config as config_era5


def normalize_obs():
    data_da_dict = {}

    for var_name in config.INPUT_CONFIG.keys(): 
        if config.INPUT_CONFIG[var_name]['include'] and config.INPUT_CONFIG[var_name]['norm']:
            max_lag_months = config.INPUT_CONFIG[var_name]['lag']
            divide_by_stdev = config.INPUT_CONFIG[var_name]['divide_by_stdev']

            # Load the observational data
            ds = xr.open_dataset(os.path.join(config_cesm.DATA_DIRECTORY, f"ERA5/cesm_format/{var_name}_obs.nc"))
            obs_da = ds[var_name] 

            da_train_subset = obs_da[train_subset]
            monthly_means = da_train_subset.groupby('time.month').mean(("time"))
            monthly_stdev = da_train_subset.groupby('time.month').std(("time"))

            # normalize
            months = obs_da['time'].dt.month

            if divide_by_stdev: 
                normalized_da = (obs_da - monthly_means.sel(month=months)) / monthly_stdev.sel(month=months)
            else: 
                normalized_da = obs_da - monthly_means.sel(month=months)
            
            data_da_dict[var_name] = normalized_da.drop_vars("month")
    return data_da_dict 


def get_obs_start_prediction_months(data_da_dict, data_split_settings):
    max_lag = 12
    first_month = data_da_dict["icefrac"].time[max_lag].item()
    last_month = data_da_dict["icefrac"].time[-config.MAX_LEAD_MONTHS].item()
    return pd.date_range(first_month, last_month, freq="MS")


def save_obs_inputs(input_config, save_path, data_da_dict, data_split_settings):
    """
    Writes a model-ready input file (.nc) for each ensemble member to save_path

    Param:
        (dict)      input_config
        (string)    save_path
    """

    # get some auxiliary data
    x_coords = data_da_dict["icefrac"].x.data
    y_coords = data_da_dict["icefrac"].y.data
    land_mask = np.isnan(data_da_dict["icefrac"].isel(time=0)).data
    land_mask = np.transpose(land_mask.reshape(1, 80, 80), [0, 2, 1]) # for some reason, x and y get switched

    start_prediction_months = get_obs_start_prediction_months(data_da_dict, data_split_settings)
    save_name = os.path.join(save_path, f"inputs_obs.nc")
    if os.path.exists(save_name):
       return

    da_list = []
    for start_prediction_month in start_prediction_months:
        time_da_list = []
        for input_var, input_var_params in input_config.items():
            if not input_var_params["include"]: 
                continue 
            
            if not input_var_params["auxiliary"]:
                prediction_input_months = pd.date_range(start_prediction_month - pd.DateOffset(months=input_var_params["lag"]), 
                                                        start_prediction_month - pd.DateOffset(months=1), freq="MS")

                input_data = data_da_dict[input_var].sel(time=prediction_input_months)

                # mask out NaN values
                input_data = input_data.fillna(0)

                # rename the time coordinate to channel 
                lag = input_var_params["lag"]
                input_data = input_data.assign_coords(time=[f"{input_var}_lag{lag+1-i}" for i in range(1, lag+1)])
                input_data = input_data.rename({"time": "channel"})
            else:
                if input_var == "cosine_of_init_month":
                    input_data = xr.DataArray(
                        np.full((1, 80, 80), np.cos(2 * np.pi * start_prediction_month.month / 12)),
                        dims=["channel", "x", "y"],
                        coords={"channel": [input_var], "x": x_coords, "y": y_coords},
                    )
                elif input_var == "sine_of_init_month":
                    input_data = xr.DataArray(
                        np.full((1, 80, 80), np.sin(2 * np.pi * start_prediction_month.month / 12)),
                        dims=["channel", "x", "y"],
                        coords={"channel": [input_var], "x": x_coords, "y": y_coords},
                    )
                elif input_var == "land_mask": 
                    input_data = xr.DataArray(
                        land_mask, 
                        dims=["channel", "x", "y"],
                        coords={"channel": [input_var], "x": x_coords, "y": y_coords},
                    )
                else: 
                    raise NotImplementedError()

            # add a coordinate to denote the start prediction month (time origin)
            input_data = input_data.assign_coords(start_prediction_month=start_prediction_month)

            time_da_list.append(input_data)

        da_list.append(xr.concat(time_da_list, dim="channel", coords='minimal', compat='override'))

    da_merged = xr.concat(da_list, dim="start_prediction_month", coords="minimal", compat='override')

    # rechunk
    da_merged = da_merged.chunk(chunks={"start_prediction_month":12, "channel":-1})

    # clean up singleton dimensions
    if "z_t" in da_merged.dims: 
        da_merged = da_merged.drop_vars("z_t")
    if "lev" in da_merged.dims:
        da_merged = da_merged.drop_vars("lev")


    print("done! Saving...")
    da_merged.to_dataset(name="data").to_netcdf(save_name)
    da_merged.close()


def save_obs_targets(input_da_dict, input_config, target_config, 
                    save_path, max_lead_months, data_split_settings):
    """
    Writes a model-ready targets file (.nc) for each ensemble member to save_path
    
    Param:
        (dict)      input_config
        (dict)      target_config
        (string)    save_path
        (int)       max_lead_months
        (dict)      data_split_settings
    """

    if not target_config["predict_anom"]:
        raise NotImplementedError()
    else:
        da = input_da_dict["icefrac"]

    start_prediction_months = get_obs_start_prediction_months(input_da_dict, data_split_settings)
    save_name = os.path.join(save_path, f"targets_obs.nc")
    if os.path.exists(save_name):
        return

    time_da_list = []

    for start_prediction_month in start_prediction_months:
        prediction_target_months = pd.date_range(start_prediction_month, 
                                                start_prediction_month + pd.DateOffset(months=max_lead_months-1), 
                                                freq="MS")
        
        target_data = da.sel(time=prediction_target_months)

        # mask out nans
        target_data = target_data.fillna(0)

        target_data = target_data.assign_coords(time=np.arange(1,7))
        target_data = target_data.rename({"time": "lead_time"}) 

        # add a coordinate to denote the start prediction month (time origin)
        target_data = target_data.assign_coords(start_prediction_month=start_prediction_month)

        time_da_list.append(target_data)

    da_merged = xr.concat(time_da_list, dim="start_prediction_month", coords='minimal', compat='override')

    da_merged = da_merged.chunk(chunks={"start_prediction_month":12, "lead_time":-1})
    
    da_merged.to_dataset(name="data").to_netcdf(save_name)
    da_merged.close()


