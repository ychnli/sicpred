import os 
import xarray as xr
import numpy as np 
import pandas as pd
import time

from src import config_cesm as config
from src.utils.util_shared import write_nc_file

ALL_VAR_NAMES = config.ALL_VAR_NAMES

def check_valid_data_split_settings(data_split_settings):
    ### TODO: check that data_split_settings is valid 

    return None
    

def find_downloaded_vars():
    """
    Finds and verifies the downloaded variables and their ensemble members.
        
    Raises:
        ValueError: If member IDs do not match across variables at any index.

    """

    member_ids = np.empty((len(ALL_VAR_NAMES), 100), dtype='object')
    n_members = []

    for i,variable in enumerate(ALL_VAR_NAMES):
        directory = os.path.join(config.RAW_DATA_DIRECTORY, variable)
        
        if os.path.exists(directory):
            files = sorted(os.listdir(directory))

            if len(files) == 0: continue

            for j,file_name in enumerate(files):
                file_path = os.path.join(directory, file_name)
                ds = xr.open_dataset(file_path)
                member_ids[i,j] = ds["member_id"].values

            print(f"Found {len(files)} ensemble members for {variable}")
            n_members.append(len(files))
        
    # check if member_ids match across variables
    min_members = np.min(n_members)
    for j in range(min_members):
        if not np.all(np.logical_or(member_ids[:, j] == member_ids[0, j], member_ids[:, j] == None)):
            print(member_ids[:, j])
            raise ValueError(f"Member IDs do not match across variables at index {j}")

    print("All member IDs match across variables")
    
    return min_members


def normalize(x, m, s, var_name=None):
    # Avoid divide by zero by setting normalized value to zero where std deviation is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (x - m) / s
        normalized = np.where(s == 0, 0, normalized)  # Set to zero where std dev is zero

    # For SST below sea ice, the stdev is very low. Normalized values are set to 0 
    # if the stdev is below threshold value
    if var_name == "temp":
        threshold = 0.001
        normalized = np.where(s <= threshold, 0, normalized)

    return normalized


def get_start_prediction_months(data_split_settings):
    """
    Get the start prediction months for the data split settings. 
    """

    if data_split_settings["split_by"] == "time":
        start_prediction_months = data_split_settings["train"].union(data_split_settings["val"]).union(data_split_settings["test"])
    elif data_split_settings["split_by"] == "ensemble_member":
        start_prediction_months = data_split_settings["time_range"]
    else:
        raise ValueError("data_split_settings split_by must be 'time' or 'ensemble_member'")

    return start_prediction_months


def normalize_data(var_name, data_split_settings, max_lead_months=6,
                    overwrite=False, verbose=1, divide_by_stdev=True):
    """ 
    Normalize inputs based on statistics of the training data and save. 

    Param:
        (string)    var_name: the standard name of the variable
        (dict)      normaliation_scheme: a dict specifying how the data is split 
        (bool)      overwrite 
        (int)       verbose  
        (bool)      divide_by_stdev: if True, computes (x - mu)/(sigma)
                                     if False, computes (x - mu)  
    """

    save_dir = os.path.join(config.PROCESSED_DATA_DIRECTORY, "normalized_inputs", data_split_settings["name"])
    save_path = os.path.join(save_dir, f"{var_name}_norm.nc")
        
    if os.path.exists(save_path) and not overwrite:
        if verbose >= 1: print(f"Already found normalized file for {var_name}. Skipping...")
        return None

    print(f"Normalizing {var_name}, divide_by_stdev = {divide_by_stdev}...", end=" ")

    # First make a merged dataset from the separate ones 
    file_list = sorted(os.listdir(f"{config.RAW_DATA_DIRECTORY}/{var_name}"))
    if f"{var_name}_combined.nc" in file_list: 
        if overwrite: 
            file_list.remove(f"{var_name}_combined.nc")
        else: 
            merged_ds = xr.open_dataset(os.path.join(config.RAW_DATA_DIRECTORY, f"{var_name}/{var_name}_combined.nc"))
    else: 
        ds_list = []

        for file in file_list:
            ds = xr.open_dataset(os.path.join(f"{config.RAW_DATA_DIRECTORY}/{var_name}", file), chunks={'time': 120})
            # change the time index to pandas instead of cftime
            ds = ds.assign_coords(time=pd.date_range("1850-01", "2100-12", freq="MS"))
            ds_list.append(ds)

        merged_ds = xr.concat(ds_list, dim="member_id")
    
        # save the merged ds before normalizing 
        write_nc_file(merged_ds, f"{config.RAW_DATA_DIRECTORY}/{var_name}/{var_name}_combined.nc", overwrite)
    
    # create a subsetted DataArray that contains the data requested by data_split_settings
    da = merged_ds[var_name]

    if data_split_settings["split_by"] == "time": 
        all_times = get_start_prediction_months(data_split_settings)

        # this adds an extension of max_lead_months since we potentially need the normalized sea ice
        # for constructing the labels 
        all_times = all_times.union(pd.date_range(all_times[-1], all_times[-1] + pd.DateOffset(months=max_lead_months), freq="MS"))
        da = da.sel(time=all_times) 
        da_train_subset = da.sel(time=data_split_settings["train"], member_id=data_split_settings["member_ids"])

    elif data_split_settings["split_by"] == "ensemble_member": 
        all_member_ids = data_split_settings["train"] + data_split_settings["val"] + data_split_settings["test"]
        da = da.sel(member_id=all_member_ids) 
        da_train_subset = da.sel(member_id=data_split_settings["train"], time=data_split_settings["time_range"])
        
    else:
        raise ValueError("data_split_settings split_by must be 'time' or 'ensemble_member'")

    if divide_by_stdev:
        print("calculating means and stdev...", end=" ")

        monthly_means = da_train_subset.groupby("time.month").mean(dim=("time", "member_id")).load()
        monthly_stdevs = da_train_subset.groupby("time.month").std(dim=("time", "member_id")).load()
        print("done!")

        months = da['time'].dt.month
        normalized_da = xr.apply_ufunc(
            normalize,
            da,
            monthly_means.sel(month=months),
            monthly_stdevs.sel(month=months),
            var_name,
            output_dtypes=[da.dtype],
            dask="allowed"
        )
        
        normalized_ds = normalized_da.to_dataset(name=var_name)
        monthly_means_ds = monthly_means.to_dataset(name=var_name)
        monthly_stdevs_ds = monthly_stdevs.to_dataset(name=var_name)
    else: 
        print("calculating means...", end=" ")
        monthly_means = da_train_subset.groupby("time.month").mean(dim=("time", "member_id")).load()
        print("done!")

        months = da['time'].dt.month
        anom_da = da - monthly_means.sel(month=months)
        
        anom_ds = anom_da.to_dataset(name=var_name)
        monthly_means_ds = monthly_means.to_dataset(name=var_name)


    print("Saving...", end="")
    write_nc_file(monthly_means_ds, os.path.join(save_dir, f"{var_name}_mean.nc"), overwrite)
    if divide_by_stdev: 
        write_nc_file(monthly_stdevs_ds, os.path.join(save_dir, f"{var_name}_stdev.nc"), overwrite)
        write_nc_file(normalized_ds, save_path, overwrite)
    else:
        write_nc_file(anom_ds, save_path, overwrite)
    print("done!")



def load_inputs_data_da_dict(input_config, data_split_settings):
    """
    Param:
        (dict)      input_config
        (dict)      data_split_settings 

    Returns:
        (dict)      dictionary of xr.DataArray of normalized inputs, with 
                    variable names as keys. 
    """

    data_da_dict = {}

    dir = os.path.join(config.PROCESSED_DATA_DIRECTORY, "normalized_inputs", data_split_settings["name"])

    for var in ALL_VAR_NAMES: 
        if var not in input_config.keys() or not input_config[var]["include"]:
            continue

        ds = xr.open_dataset(os.path.join(dir, f"{var}_norm.nc"), chunks={"member_id": 1})

        data_da_dict[var] = ds[var]

    # make sure all the data arrays have the same member_ids 
    common_member_ids = set()
    for i,da in enumerate(data_da_dict.values()): 
        if i == 0: 
            common_member_ids = set(da.member_id.data)
        else:
            common_member_ids = common_member_ids & set(da.member_id.data)

    for var, da in data_da_dict.items():
        da = da.sel(member_id = list(common_member_ids))
        
        # drop this auxiliary variable that is leftover from normalization 
        da = da.drop_vars("month")
        data_da_dict[var] = da

    return data_da_dict



def save_inputs_files(input_config, save_path, data_split_settings):
    """
    Writes a model-ready input file (.nc) for each ensemble member to save_path

    Param:
        (dict)      input_config
        (string)    save_path
    """
    
    data_da_dict = load_inputs_data_da_dict(input_config, data_split_settings)

    # get some auxiliary data
    x_coords = data_da_dict["icefrac"].x.data
    y_coords = data_da_dict["icefrac"].y.data
    land_mask = np.isnan(data_da_dict["temp"].isel(time=0, member_id=0)).data 
    land_mask = np.transpose(land_mask.reshape(1, 80, 80), [0, 2, 1]) # for some reason, x and y get switched
    
    # save each ensemble member separately so the files don't get too big 
    member_ids = data_da_dict["icefrac"].member_id.data
    start_prediction_months = get_start_prediction_months(data_split_settings)
    for member_id in member_ids:
        save_name = os.path.join(save_path, f"inputs_member_{member_id}.nc")
        if os.path.exists(save_name):
            continue

        print(f"Concatenating data into model input format for member {member_id}...")
        start_time = time.time()
        member_da_list = [] # we will concat this later 
        
        for start_prediction_month in start_prediction_months:

            time_da_list = []
            for input_var, input_var_params in input_config.items():
                if not input_var_params["include"]: 
                    continue 
                
                if not input_var_params["auxiliary"]:
                    prediction_input_months = pd.date_range(start_prediction_month - pd.DateOffset(months=input_var_params["lag"]), 
                                                            start_prediction_month - pd.DateOffset(months=1), freq="MS")

                    input_data = data_da_dict[input_var].sel(time=prediction_input_months, member_id=member_id)

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

            time_da_merged = xr.concat(time_da_list, dim="channel", coords='minimal', compat='override')
            member_da_list.append(time_da_merged)
        
        da_merged = xr.concat(member_da_list, dim="start_prediction_month", coords="minimal", compat='override')

        # rechunk
        da_merged = da_merged.chunk(chunks={"start_prediction_month":12, "channel":-1})

        # clean up singleton dimensions
        da_merged = da_merged.drop_vars(("z_t", "lev"))
        
        
        print("done! Saving...")
        da_merged.to_dataset(name="data").to_netcdf(save_name)
        da_merged.close()

        end_time = time.time()
        print(f"done! Elapsed time: {end_time - start_time:.2f} seconds")


def save_targets_files(input_config, target_config, save_path, max_lead_months, data_split_settings):
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
        ds = xr.open_dataset(os.path.join(config.RAW_DATA_DIRECTORY, "icefrac/icefrac_combined.nc"))
        da = ds["icefrac"] 
    else:
        input_da_dict = load_inputs_data_da_dict(input_config, data_split_settings)
        da = input_da_dict["icefrac"]

    member_ids = da.member_id.data

    start_prediction_months = get_start_prediction_months(data_split_settings)
    for member_id in member_ids:
        save_name = os.path.join(save_path, f"targets_member_{member_id}.nc")
        if os.path.exists(save_name):
            continue

        print(f"Concatenating ground-truth data into model output format for member {member_id}...")
        start_time = time.time()
        time_da_list = []

        for start_prediction_month in start_prediction_months:
            prediction_target_months = pd.date_range(start_prediction_month, 
                                                    start_prediction_month + pd.DateOffset(months=max_lead_months-1), 
                                                    freq="MS")
            
            target_data = da.sel(time=prediction_target_months, member_id=member_id)

            # mask out nans
            target_data = target_data.fillna(0)

            target_data = target_data.assign_coords(time=np.arange(1,7))
            target_data = target_data.rename({"time": "lead_time"}) 

            # add a coordinate to denote the start prediction month (time origin)
            target_data = target_data.assign_coords(start_prediction_month=start_prediction_month)

            time_da_list.append(target_data)

        da_merged = xr.concat(time_da_list, dim="start_prediction_month", coords='minimal', compat='override')

        da_merged = da_merged.chunk(chunks={"start_prediction_month":12, "lead_time":-1})
        
        print("done! Saving...")
        da_merged.to_dataset(name="data").to_netcdf(save_name)
        da_merged.close()

        end_time = time.time()
        print(f"done! Elapsed time: {end_time - start_time:.2f} seconds")


