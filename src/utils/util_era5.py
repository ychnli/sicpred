import xarray as xr
import numpy as np
import pandas as pd
from time import time
from src import config
import os
import pickle
from netCDF4 import Dataset
import h5py

from src.models.models import linear_trend
from src.utils.util_shared import write_nc_file

##########################################################################################
# File I/O wrappers
##########################################################################################





def write_hdf5_file(data, save_path, dataset_name, verbose=1):
    """
    Saves hdf5 file to save_path. 

    Params:
        data:           numpy ndarray 
        save_path:      (str) valid file path ending in .h5
        dataset_name:   (str) name of dataset 
    """

    with h5py.File(save_path, 'w') as f:
        f.create_dataset(dataset_name, data=data)

    if verbose == 2: print(f'Saved to {save_path}')


def save_dict_to_pickle(data_dict, save_path):
    """
    Saves a dictionary to a specified path using pickle.
    
    Params:
        data_dict:  (dict) The dictionary to save.
        save_path:  (str) The file path where the dictionary will be saved.
    """
    with open(save_path, 'wb') as file:
        pickle.dump(data_dict, file)

def load_pickle(load_path):
    with open(load_path, 'rb') as file:
        data = pickle.load(file)
    return data


##########################################################################################
# Data preprocessing 
##########################################################################################


def normalize(x, m, s, var_name=None):
    # Avoid divide by zero by setting normalized value to zero where std deviation is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (x - m) / s
        normalized = np.where(s == 0, 0, normalized)  # Set to zero where std dev is zero

    # For SST below sea ice, the stdev is very low. Normalized values are set to 0 
    # if the stdev is below threshold value
    if var_name == "sea_surface_temperature":
        threshold = 0.001
        normalized = np.where(s <= threshold, 0, normalized)

    return normalized


def normalize_data(overwrite=False, verbose=1, vars_to_normalize="all"):
    """ 
    Normalize inputs based on statistics of the training data and save. 
    """

    if vars_to_normalize == "all":
        vars_to_normalize = list(config.era5_variables_dict.keys())

        # This is the ERA5 boundary condition sea ice, which is not the same as the one we use
        vars_to_normalize.remove("sea_ice_cover")

        # Use this as the name instead
        vars_to_normalize.append("siconc")
    
    if verbose >= 1: print(f"Normalizing variables {vars_to_normalize}")

    save_dir = os.path.join(config.DATA_DIRECTORY, "sicpred/normalized_inputs")

    for var_name in vars_to_normalize:
        if os.path.exists(os.path.join(save_dir, f"{var_name}_norm.nc")) and not overwrite:
            if verbose >= 1: print(f"Already found normalized file for {var_name}. Skipping...")
            continue

        print(f"Normalizing {var_name}...", end=" ")
        if var_name in config.era5_variables_dict:
            plevel = config.era5_variables_dict[var_name]["plevel"]

        if var_name == "siconc":
            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc")
            da = ds[var_name]

        elif plevel != None:
            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_{plevel}hPa_SPS.nc")
            da = ds[config.era5_variables_dict[var_name]["short_name"]]

        else:
            if not os.path.exists(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_SPS.nc"):
                raise FileNotFoundError(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_SPS.nc does not exist!")

            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_SPS.nc")
            da = ds[config.era5_variables_dict[var_name]["short_name"]]

        print("calculating means and stdev...", end=" ")
        # calculate the normalization statistics over the training set only 
        time_subset = config.TRAIN_MONTHS.intersection(da.time)
        monthly_means = da.sel(time=time_subset).groupby("time.month").mean("time")
        monthly_stdevs = da.sel(time=time_subset).groupby("time.month").std("time")
        print("done!")

        months = da['time'].dt.month
        normalized_da = xr.apply_ufunc(
            normalize,
            da,
            monthly_means.sel(month=months),
            monthly_stdevs.sel(month=months),
            var_name,
            output_dtypes=[da.dtype]
        )

        if var_name != "siconc":
            ds_name = config.era5_variables_dict[var_name]["short_name"] 
        else: 
            ds_name = "siconc" 
        
        normalized_ds = normalized_da.to_dataset(name=ds_name)
        monthly_means_ds = monthly_means.to_dataset(name=ds_name)
        monthly_stdevs_ds = monthly_stdevs.to_dataset(name=ds_name)

        # Make a new folder to save the normalized variables 
        os.makedirs(save_dir, exist_ok=True)
        print("Saving...", end="")
        write_nc_file(monthly_means_ds, os.path.join(save_dir, f"{var_name}_mean.nc"), overwrite)
        write_nc_file(monthly_stdevs_ds, os.path.join(save_dir, f"{var_name}_stdev.nc"), overwrite)
        write_nc_file(normalized_ds, os.path.join(save_dir, f"{var_name}_norm.nc"), overwrite)
        print("done!")

    print("done! \n\n")


def calculate_siconc_anom(overwrite=False, verbose=1):
    print('Calculating siconc anomalies from mean without dividing by stdev')
    save_path = os.path.join(config.DATA_DIRECTORY, "sicpred/normalized_inputs/siconc_anom.nc")
    
    if os.path.exists(save_path) and not overwrite:
        if verbose >= 2: print(f"Already found {save_path}. Skipping... \n\n")
        return
    
    da = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc").siconc

    time_subset = config.TRAIN_MONTHS.intersection(da.time)
    monthly_means = da.sel(time=time_subset).groupby("time.month").mean("time")

    siconc_anom = xr.apply_ufunc(
        lambda x, m: x - m,
        da,
        monthly_means.sel(month=da['time'].dt.month),
        output_dtypes=[da.dtype]
    )

    siconc_anom_ds = siconc_anom.to_dataset(name="siconc")
    write_nc_file(siconc_anom_ds, save_path, overwrite) 
    print("done! \n\n")


def remove_expver_from_era5(verbose=1):
    if verbose >= 1: print("Removing expver dimension from ERA5 variables")
    for var_name in config.era5_variables_dict.keys():
        plevel = config.era5_variables_dict[var_name]["plevel"]
        if plevel != None:
            file_path = os.path.join(config.DATA_DIRECTORY, f"ERA5/{var_name}_{plevel}hPa_SPS.nc")
            ds = xr.open_dataset(file_path)
        else:
            file_path = os.path.join(config.DATA_DIRECTORY, f"ERA5/{var_name}_SPS.nc")
            ds = xr.open_dataset(file_path)
        
        if "expver" not in ds.dims:
            if verbose == 1:
                continue
            elif verbose > 1: 
                print(f"expver not found in {var_name}. Skipping... ")
                continue
            else: continue 

        print(f"Removing expver dimension from {var_name}...", end=' ')
        combined_ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
        
        write_nc_file(combined_ds, file_path, overwrite=True)
        print("saved!")

    print("done! \n\n")
        

def concatenate_nsidc(verbose=1, working_path=None, overwrite=False):
    """ 
    Merge the NSIDC data into one netCDF file called seaice_conc_monthly_all 
    """
    print("Merging monthly NSIDC files")
    if working_path == None:
        working_path = os.path.join(config.DATA_DIRECTORY, 'NSIDC')

    if not os.path.exists(working_path):
        raise FileNotFoundError(f"{working_path} does not exist. make sure the NSIDC \
             data is in its own directory")
    
    save_path = f'{working_path}/seaice_conc_monthly_all.nc'
    if os.path.exists(save_path) and not overwrite:
        print(f"Already found file {save_path}. Skipping... \n\n")
        return

    all_files = os.listdir(working_path)

    # Files that start with "icdr" correspond to the Near Real Time product,
    # which is updated past the last year
    icdr_files = [f for f in all_files if f.startswith('seaice_conc_monthly_icdr')] 
    icdr_files.sort()
    sh_files = [f for f in all_files if f.startswith('seaice_conc_monthly_sh')]
    sh_files.sort()

    sorted_files = sh_files + icdr_files
    sorted_files = [os.path.join(working_path, f) for f in sorted_files]

    print("Concatenating files...")
    ds_concat = xr.open_mfdataset(sorted_files, combine="nested", concat_dim='tdim')
    ds_concat = ds_concat.rename({'cdr_seaice_conc_monthly': 'siconc'})
    ds_concat.load()

    # Add time coordinate 
    print("Adding time index and renaming time coordinate...")
    ds_concat = ds_concat.swap_dims({'tdim': 'time'})
    ds_concat["time"] = pd.to_datetime(ds_concat["time"].values)

    print('Saving...', end=' ')
    write_nc_file(ds_concat, save_path, overwrite)
    print('done! \n\n')


def remove_missing_data_nsidc(verbose=1):
    """
    NSIDC data is missing for 1987 December and 1988 January. Need to remove it so that
    we don't get NaNs messing up averaging or trend computations 
    """

    if verbose >= 1: print("Checking for missing data in NSIDC sea ice")
    nsidc_sic = xr.open_dataset(f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc')

    # The missing data lives as NaN. Check if we need to get rid of it by finding years where
    # there are NaN values 
    needs_fixing = len(nsidc_sic.time[np.isnan(nsidc_sic.siconc).sum(dim=('x', 'y')) != 0]) != 0 

    if not needs_fixing:
        if verbose >= 2: print("Didn't find any NaN values in NSIDC dataset. Skipping... \n\n")
        return 
    
    if verbose >= 1: print("Removing missing ice data in 1987 Dec and 1988 Jan...")
    nsidc_sic = nsidc_sic.sel(time=nsidc_sic.time[np.isnan(nsidc_sic.siconc).sum(dim=('x', 'y')) == 0])

    temp_path = f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc.tmp'
    nsidc_sic.to_netcdf(temp_path)
    os.replace(temp_path, f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc')


def apply_land_mask_to_nsidc_siconc(verbose=1):
    if verbose >= 1: print("Applying land mask to NSIDC sea ice data")

    file_path = f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc'
    nsidc_sic = xr.open_dataset(file_path)

    # This condition checks for the presence of NSIDC land flag values (2.54)
    # in the first time step. Not a foolproof check, but fast and should work
    needs_fixing = np.any(nsidc_sic.siconc.isel(time=0) > 2)

    if not needs_fixing:
        if verbose >= 2: print("Land mask has already been applied. Skipping... \n\n")
        return 
    
    land_mask = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/land_mask.nc").mask.values
    nsidc_sic_masked = nsidc_sic.siconc * ~land_mask
    write_nc_file(nsidc_sic_masked.to_dataset(name="siconc"), file_path, overwrite=True) 
    print("done! \n\n")


def compute_linear_forecast(overwrite=False, parallelize=True, verbose=1):
    months_to_calculate_linear_forecast = pd.date_range(start='1981-01-01', end='2024-06-01', freq='MS')

    if not overwrite and os.path.exists(f"{config.DATA_DIRECTORY}/sicpred/linear_forecasts/linear_forecast_all_years.nc"):
        print("Already found computed linear trend. Skipping... \n\n")
        return 

    if parallelize:
        from joblib import Parallel, delayed

        Parallel(n_jobs=-1)(delayed(linear_trend)(month, f"{config.DATA_DIRECTORY}/sicpred/linear_forecasts/") \
            for month in months_to_calculate_linear_forecast)
    else:
        for month in months_to_calculate_linear_forecast:
            linear_trend(month, f"{config.DATA_DIRECTORY}/sicpred/linear_forecasts/")


def concatenate_linear_trend(overwrite=False, verbose=1):
    """
    Concatenates the linear trend files
    """
    save_path = os.path.join(config.DATA_DIRECTORY, 'sicpred/linear_forecasts/linear_forecast_all_years.nc')

    if verbose >= 1: print("Concatenating linear trend data into one file")

    if not overwrite and os.path.exists(save_path):
        if verbose >= 2: print(f"Already found file at {save_path}. Skipping... \n\n")
        return 
    
    working_path = os.path.join(config.DATA_DIRECTORY, 'sicpred/linear_forecasts/')
    files_to_concatenate = os.listdir(working_path)
    files_to_concatenate = [os.path.join(working_path, f) for f in files_to_concatenate]
    files_to_concatenate = sorted(files_to_concatenate)

    if len(files_to_concatenate) == 0:
        print(f"found no files to concatenate in {working_path}")
        return

    print("Concatenating linear trend files...", end='')
    ds_concat = xr.open_mfdataset(files_to_concatenate, combine="nested", concat_dim='time')
    
    print("saving...", end= '')
    write_nc_file(ds_concat, save_path, overwrite)
    print("done! \n\n")


def load_inputs_data_da_dict(input_config):
    """
    Loads a dict containing DataArray objects as specified by input_config (dict). 
    See config.py for an example of the specified input configurations. 
    """

    data_da_dict = {}

    for input_var, input_var_params in input_config.items():
        if not input_var_params["include"]: 
            continue

        if input_var == "siconc":
            if input_var_params['anom'] and input_var_params['div_stdev']:
                data_da_dict[input_var] = xr.open_dataset(f"{config.DATA_DIRECTORY}/sicpred/normalized_inputs/{input_var}_norm.nc").siconc
            elif input_var_params['anom'] and not input_var_params['div_stdev']:
                data_da_dict[input_var] = xr.open_dataset(f"{config.DATA_DIRECTORY}/sicpred/normalized_inputs/{input_var}_anom.nc").siconc
            else: 
                data_da_dict[input_var] = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc").siconc

        elif input_var == 'siconc_linear_forecast':
            data_da_dict[input_var] = xr.open_dataset(f"{config.DATA_DIRECTORY}/sicpred/linear_forecasts/linear_forecast_all_years.nc").siconc
        
        elif input_var in ['cosine_of_init_month', 'sine_of_init_month']:
            # These will get generated within this function 
            continue 
                
        else:
            if input_var_params['anom']:
                data_da_dict[input_var] = xr.open_dataset(f"{config.DATA_DIRECTORY}/sicpred/normalized_inputs/{input_var}_norm.nc")[input_var_params['short_name']]
            else:
                data_da_dict[input_var] = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/{input_var}_SPS.nc")[input_var_params['short_name']]
    
    return data_da_dict 


def save_inputs_file(input_config_name, input_config, inputs_save_path, start_prediction_months, 
                    clip_PDF_tails=True, clip_constant=6, verbose=1):

    # Retrieve land mask 
    land_mask = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/land_mask.nc").mask.values
    land_mask = land_mask[np.newaxis, :, :]

    data_da_dict = load_inputs_data_da_dict(input_config)
    
    all_inputs = []
    for start_prediction_month in start_prediction_months:
        if verbose >= 2: print(f"Concatenating inputs for init month {start_prediction_month}")
        prediction_target_months = pd.date_range(start_prediction_month, \
            start_prediction_month + pd.DateOffset(months=config.max_month_lead_time-1), freq="MS")
        
        # For each target, generate data pairs
        input_list = []
        for input_var, input_var_params in input_config.items():
            if not input_var_params["include"]: 
                continue 

            if input_var == 'siconc_linear_forecast':
                input_data_npy = data_da_dict[input_var].sel(time=prediction_target_months).data
                
            elif input_var == 'cosine_of_init_month':
                input_data_npy = np.ones((1, 332, 316))
                input_data_npy *= np.cos(2 * np.pi * start_prediction_month.month / 12)

            elif input_var == 'sine_of_init_month':
                input_data_npy = np.ones((1, 332, 316))
                input_data_npy *= np.sin(2 * np.pi * start_prediction_month.month / 12)

            else:
                prediction_input_months = pd.date_range(start_prediction_month - pd.DateOffset(months=input_var_params["lag"]), \
                    start_prediction_month - pd.DateOffset(months=1), freq="MS")
                input_data_npy = data_da_dict[input_var].sel(time=prediction_input_months).data

                # to prevent super out-of-distribution SST and ssr data points
                if input_var in ["sea_surface_temperature", "surface_net_solar_radiation"]:
                    if clip_PDF_tails: 
                        input_data_npy = np.where(input_data_npy > clip_constant, clip_constant, input_data_npy)
                        input_data_npy = np.where(input_data_npy < -clip_constant, -clip_constant, input_data_npy)
            
            # Apply land mask. Land values go to 0 
            if input_var_params['land_mask']:
                land_mask_broadcast = np.repeat(land_mask, input_data_npy.shape[0], axis=0)
                input_data_npy[land_mask_broadcast] = 0

            input_list.append(input_data_npy)
        
        input_all_vars_npy = np.concatenate(input_list, axis=0)

        # add a new axis at the beginning for n_samples
        input_all_vars_npy = input_all_vars_npy[np.newaxis,:,:,:]
        all_inputs.append(input_all_vars_npy)

    all_inputs = np.concatenate(all_inputs, axis=0)

    print(f"Saving to {inputs_save_path}")
    write_hdf5_file(all_inputs, inputs_save_path, f"inputs_{input_config_name}")



def save_targets_file(input_config, target_config, targets_save_path, start_prediction_months, verbose=1):
    # Retrieve land mask 
    land_mask = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/land_mask.nc").mask.values
    land_mask = land_mask[np.newaxis, :, :]

    # Retrieve targets (ground truth) dataarray 
    if not target_config["predict_siconc_anom"]:
        target_da = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc").siconc
    else:
        if input_config["siconc"]["div_stdev"]:
            target_da = xr.open_dataset(f"{config.DATA_DIRECTORY}/sicpred/normalized_inputs/siconc_norm.nc").siconc 
        else:
            target_da = xr.open_dataset(f"{config.DATA_DIRECTORY}/sicpred/normalized_inputs/siconc_anom.nc").siconc
    
    all_targets = []

    for start_prediction_month in start_prediction_months:
        if verbose >= 2: print(f"Concatenating targets for init month {start_prediction_month}")
        prediction_target_months = pd.date_range(start_prediction_month, \
            start_prediction_month + pd.DateOffset(months=config.max_month_lead_time-1), freq="MS")
    
        target_npy = target_da.sel(time=prediction_target_months).data
        land_mask_broadcast = np.repeat(land_mask, config.max_month_lead_time, axis=0)
        target_npy[land_mask_broadcast] = 0

        # add a new axis at the beginning for n_samples
        target_npy = target_npy[np.newaxis,:,:,:]
        all_targets.append(target_npy)

    all_targets = np.concatenate(all_targets, axis=0)

    print(f"Saving to {targets_save_path}")
    write_hdf5_file(all_targets, targets_save_path, f"targets_sea_ice_only")



def prep_prediction_samples(input_config_name, target_config_name, overwrite=False, verbose=1): 
    """
    Collects the input variables and target (ground truth) outputs and saves to HDF5 file 
    """
    
    save_directory = os.path.join(config.DATA_DIRECTORY, "sicpred/data_pairs_npy")
    os.makedirs(save_directory, exist_ok=True)

    # Get input and targets data configs
    if input_config_name in config.input_configs.keys():
        print(f"Prepping and saving data pairs for input config {input_config_name}")
        input_config = config.input_configs[input_config_name]
    else:
        raise KeyError(f"Input config not found in {config.input_configs.keys()}")

    if target_config_name in config.target_configs.keys():
        target_config = config.target_configs[target_config_name]
    else:
        raise KeyError(f"Target config not found in {config.target_configs.keys()}")

    inputs_save_path = os.path.join(save_directory, f"inputs_{input_config_name}.h5")
    targets_save_path = os.path.join(save_directory, f"targets_{target_config_name}.h5")
    
    # Create a list of the starting month of each 6-month prediction target
    # Note missing data for 1987 Dec and 1988 Jan. So remove those from the prediction months
    first_range = pd.date_range('1981-01', pd.Timestamp('1987-12') - pd.DateOffset(months=config.max_month_lead_time+1), freq='MS')
    second_range = pd.date_range(pd.Timestamp('1988-01') + pd.DateOffset(months=input_config['siconc']['lag']+1), '2024-01', freq='MS')
    start_prediction_months = first_range.append(second_range)

    if os.path.exists(inputs_save_path) and not overwrite:
        print(f"Already found saved inputs for {input_config_name}")
    else: 
        save_inputs_file(input_config_name, input_config, inputs_save_path, start_prediction_months, verbose)
    
    if os.path.exists(targets_save_path) and not overwrite:
        print(f"Already found saved targets for {target_config_name}")
    else: 
        save_targets_file(input_config, target_config, targets_save_path, start_prediction_months, verbose)
        
    print("done! \n\n")
    


def generate_masks(overwrite=False, verbose=1):
    """
    Generates a mask that is True if there is ever nonzero sea ice concentration at 
    that grid point and False if always zero. Exists for each month in the year 
    """

    if verbose >= 1: print("Generating land and ice masks")

    ice_mask_save_path = os.path.join(config.DATA_DIRECTORY, "NSIDC/monthly_ice_mask.nc")
    land_mask_save_path = os.path.join(config.DATA_DIRECTORY, "NSIDC/land_mask.nc")

    if os.path.exists(ice_mask_save_path) and os.path.exists(land_mask_save_path) and not overwrite:
        if verbose >= 2: print("Already found mask files. Skipping... \n\n")
        return 
    
    # find land mask
    print("Generating and saving land mask")
    nsidc_sic = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc")
    sst = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/sea_surface_temperature_SPS.nc").sst

    land_mask_from_sst = np.isnan(sst.isel(time=0)).values
    land_mask_from_sic = np.logical_or(nsidc_sic.siconc.isel(time=0) == 2.53, nsidc_sic.siconc.isel(time=0) == 2.54)
    land_mask = np.logical_or(land_mask_from_sst, land_mask_from_sic)
    land_mask_ds = land_mask.to_dataset(name="mask")

    write_nc_file(land_mask_ds, land_mask_save_path, overwrite)
    print(f"done!")

    print("Generating and saving ice cover mask for custom loss function")

    monthly_means = nsidc_sic.siconc.groupby('time.month').mean('time')

    # find ice mask for each month
    ice_mask = monthly_means != 0
    ice_mask = ice_mask.where(~land_mask.data, 0)

    ice_mask_ds = ice_mask.to_dataset(name="mask")

    write_nc_file(ice_mask_ds, ice_mask_save_path, overwrite)

    print(f"done! \n\n")


def calculate_climatological_siconc_over_train(overwrite=False, verbose=1):
    if verbose >= 1: print("Calculating sea ice conc climatology...")

    if os.path.exists(f"{config.DATA_DIRECTORY}/NSIDC/siconc_clim.nc") and not overwrite:
        if verbose >= 2: print("Already found climatology file. Skipping... \n\n")
        return 

    siconc = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc").siconc
    train_months = config.TRAIN_MONTHS.intersection(siconc.time) # Need to do this because of missing data
    siconc_clim = siconc.sel(time=train_months).groupby('time.month').mean('time')

    siconc_clim_ds = siconc_clim.to_dataset(name="siconc")
    write_nc_file(siconc_clim_ds, f"{config.DATA_DIRECTORY}/NSIDC/siconc_clim.nc", overwrite)
    print("done! \n\n")

##########################################################################################
# Statistical methods 
##########################################################################################


def linear_regress_array(x_arr, y_arr, axis=0, mask=None):
    """
    Fit the linear model y = Ax + b using least squares. Works element-wise along axis.

    Params:
        x_arr:  either an array of same shape as y_arr or a 1-dimensional array with
                the same size as y_arr along the specified axis
        y_arr:  ndarray of arbitrary size 
        axis:   axis along which to fit the linear model. By default, set to 0 (time axis)
        mask:   mask of same shape as y_arr (excluding the regression axis) specifying 
                which points to compute the regression and which ones to skip 
    """
    num_dimensions = y_arr.ndim
    
    if axis >= num_dimensions or axis < 0:
        raise ValueError(f"Axis {axis} is out of bounds for array of dimension {num_dimensions}")
    
    if x_arr.ndim == 1:
        if len(x_arr) != y_arr.shape[axis]:
            raise ValueError(f"Length of x_arr must match the size of y_arr along axis {axis}")
        
    # Get the shape of the input array and modify it for the output array
    original_shape = list(y_arr.shape)
    modified_shape = original_shape.copy()
    modified_shape[axis] = 2
    result_array = np.empty(modified_shape)
    
    # Iterate over all slices along the specified axis and perform linear regression
    start_time = time()
    it = np.nditer(np.zeros(original_shape), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index

        y_slice_index = list(idx)
        y_slice_index[axis] = slice(None)

        if mask is None:
            mask_slice = True
        else: 
            mask_index = idx[:axis] + idx[axis+1:]
            mask_slice = mask[mask_index]
        
        if np.any(mask_slice):
            # Select the current slices of x_arr and y_arr
            if x_arr.ndim == 1:
                x_slice = x_arr
            else:
                x_slice = x_arr[tuple(y_slice_index)]
            
            y_slice = y_arr[tuple(y_slice_index)]
            A = np.vstack([x_slice, np.ones_like(x_slice)]).T
            slope, intercept = np.linalg.lstsq(A, y_slice, rcond=None)[0]
        else:
            slope, intercept = 0.0, 0.0
        
        # Store the results in the result_array
        result_array[tuple(y_slice_index)] = [slope, intercept]
        
        it.iternext()

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    return result_array



##########################################################################################
# Sea ice diagnostics 
##########################################################################################

def calculate_SIA(sector="all"):
    ### TODO

    return None

def calculate_SIE(sector="all"):
    ### TODO 

    return None