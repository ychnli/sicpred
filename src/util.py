import xarray as xr
import numpy as np
import pandas as pd
from time import time
import config
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt


##########################################################################################
# File I/O wrappers
##########################################################################################


def write_nc_file(ds, save_path, overwrite, verbose=1):
    if type(overwrite) != bool:
        raise TypeError("overwrite needs to be bool")

    if os.path.exists(save_path):
        if overwrite:
            temp_path = save_path + '.tmp'
            ds.to_netcdf(temp_path)
            os.replace(temp_path, save_path)
            if verbose == 2: print(f"Overwrote {save_path}")

    else: 
        ds.to_netcdf(save_path)
        if verbose == 2: print(f"Saved to {save_path}")


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
        if var_name == "siconc":
            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc")
            da = ds[var_name]

        elif var_name == "geopotential":
            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_500hPa_SPS.nc")
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
        if var_name == "geopotential":
            file_path = os.path.join(config.DATA_DIRECTORY, "ERA5/geopotential_500hPa_SPS.nc")
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


def load_data_da_dict(input_config):
    """
    Loads a dict containing DataArray objects as specified by input_config (dict). 
    See config.py for an example of the specified input configurations. 
    """

    data_da_dict = {}

    for input_var, input_var_params in input_config.items():
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


def prep_prediction_samples(input_config_name, overwrite=False, verbose=1): 
    """
    Collects the input variables and target outputs and saves to file 
    """
    
    save_directory = os.path.join(config.DATA_DIRECTORY, "sicpred/data_pairs_npy")
    os.makedirs(save_directory, exist_ok=True)

    inputs_save_path = os.path.join(save_directory, f"inputs_{input_config_name}.h5")
    outputs_save_path = os.path.join(save_directory, f"targets.h5")

    if os.path.exists(inputs_save_path) and os.path.exists(outputs_save_path) and not overwrite:
        print(f"Already found saved {inputs_save_path} and {outputs_save_path}. Skipping...")
        return
    
    # Get input data config
    if input_config_name in config.input_configs.keys():
        print(f"Prepping and saving data pairs for input config {input_config_name}")
        input_config = config.input_configs[input_config_name]
    else:
        raise NameError(f"Input config not found in {config.input_configs.keys()}")
    
    # Note missing data for 1987 Dec and 1988 Jan. So remove those from the prediction months
    first_range = pd.date_range('1981-01', pd.Timestamp('1987-12') - pd.DateOffset(months=config.max_month_lead_time+1), freq='MS')
    second_range = pd.date_range(pd.Timestamp('1988-01') + pd.DateOffset(months=input_config['siconc']['lag']+1), '2024-01', freq='MS')
    start_prediction_months = first_range.append(second_range)

    # Define land mask using both SST and sea ice 
    sst = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/sea_surface_temperature_SPS.nc").sst
    land_mask_from_sst = np.isnan(sst.isel(time=0)).values

    nsidc_sic = xr.open_dataset(f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc')
    land_mask_from_sic = np.logical_or(nsidc_sic.siconc.isel(time=0) == 2.53, nsidc_sic.siconc.isel(time=0) == 2.54)

    land_mask = np.logical_or(land_mask_from_sst, land_mask_from_sic).data
    land_mask = land_mask[np.newaxis, :, :]

    data_da_dict = load_data_da_dict(input_config)

    all_inputs = []
    all_outputs = []

    for start_prediction_month in start_prediction_months:
        if verbose >= 2: print(f"Concatenating inputs and target for init month {start_prediction_month}")
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
            
            # Apply land mask. Land values go to 0 
            if input_var_params['land_mask']:
                land_mask_broadcast = np.repeat(land_mask, input_data_npy.shape[0], axis=0)
                input_data_npy[land_mask_broadcast] = 0

            input_list.append(input_data_npy)
        
        input_all_vars_npy = np.concatenate(input_list, axis=0)
        target_npy = data_da_dict["siconc"].sel(time=prediction_target_months).data
        land_mask_broadcast = np.repeat(land_mask, config.max_month_lead_time, axis=0)
        target_npy[land_mask_broadcast] = 0

        # add a new axis at the beginning for n_samples
        input_all_vars_npy = input_all_vars_npy[np.newaxis,:,:,:]
        target_npy = target_npy[np.newaxis,:,:,:]

        all_inputs.append(input_all_vars_npy)
        all_outputs.append(target_npy)

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    print(f"Saving to .h5 files to {save_directory}...", end='')
    write_hdf5_file(all_inputs, inputs_save_path, f"inputs_{input_config_name}")

    if not os.path.exists(outputs_save_path) or overwrite:
        write_hdf5_file(all_outputs, outputs_save_path, f"targets_sea_ice_only")

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
# Model utils
##########################################################################################


class SeaIceDataset(Dataset):
    def __init__(self, data_directory, configuration, split_array, start_prediction_months, \
                split_type='train', target_shape=(336, 320)):
        self.data_directory = data_directory
        self.configuration = configuration
        self.split_array = split_array
        self.start_prediction_months = start_prediction_months
        self.split_type = split_type
        self.target_shape = target_shape

        # Open the HDF5 files
        self.inputs_file = h5py.File(f"{config.DATA_DIRECTORY}/sicpred/data_pairs_npy/inputs_{configuration}.h5", 'r')
        self.targets_file = h5py.File(f"{config.DATA_DIRECTORY}/sicpred/data_pairs_npy/targets.h5", 'r')
        
        self.inputs = self.inputs_file[f"inputs_{configuration}"]
        self.targets = self.targets_file['targets_sea_ice_only']

        self.n_samples, self.n_channels, self.n_y, self.n_x = self.inputs.shape
        
        # Get indices for the specified split type
        self.indices = np.where(self.split_array == split_type)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        input_data = self.inputs[actual_idx]
        target_data = self.targets[actual_idx]
        start_prediction_month = self.start_prediction_months[actual_idx]

        # Pad input_data and target_data to the target shape
        pad_y = self.target_shape[0] - self.n_y
        pad_x = self.target_shape[1] - self.n_x
        input_data = np.pad(input_data, ((0, 0), (pad_y//2, pad_y//2), (pad_x//2, pad_x//2)), mode='constant', constant_values=0)
        target_data = np.pad(target_data, ((0, 0), (pad_y//2, pad_y//2), (pad_x//2, pad_x//2)), mode='constant', constant_values=0)

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        # Get the target months for this sample
        target_months = pd.date_range(start=start_prediction_month, end=start_prediction_month + pd.DateOffset(months=5), freq="MS")
        target_months = target_months.month.to_numpy()
        
        return input_tensor, target_tensor, target_months

    def __del__(self):
        self.inputs_file.close()
        self.targets_file.close()


class MaskedMSELoss(nn.Module):
    def __init__(self, use_weights, zero_class_weight=None):
        super(MaskedMSELoss, self).__init__()
        self.ice_mask = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/monthly_ice_mask.nc").mask
        self.use_weights = use_weights
        self.zero_class_weight = zero_class_weight

    def forward(self, outputs, targets, target_months):
        n_active_cells = 0

        for target_months_subset in target_months:
            n_active_cells += self.ice_mask.sel(month=target_months_subset.cpu()).sum().values
        
        # Punish predictions of sea ice in ice free zones 
        if self.use_weights:
            weights = torch.where(targets == 0, self.zero_class_weight, 1)
            loss = torch.sum(((targets - outputs) ** 2) * weights) / n_active_cells
        else:
            loss = torch.sum((targets - outputs) ** 2) / n_active_cells

        return loss


def print_split_stats(split_array):
    ntrain = sum(split_array == 'train')
    nval = sum(split_array == 'val')
    ntest = sum(split_array == 'test')
    
    print(f"train samples: {ntrain} ({round(ntrain / len(split_array), 2)})")
    print(f"val samples: {nval} ({round(nval / len(split_array), 2)})")
    print(f"test samples: {ntest} ({round(ntest / len(split_array), 2)})")


def generate_start_prediction_months(max_month_lead_time=6, max_input_lag_time=12):
    # Construct the date range for the data pairs 
    # Note that this is not continuous due to the missing data in 1987-1988 
    first_range = pd.date_range('1981-01', pd.Timestamp('1987-12') - pd.DateOffset(months=max_month_lead_time+1), freq='MS')
    second_range = pd.date_range(pd.Timestamp('1988-01') + pd.DateOffset(months=max_input_lag_time+1), '2024-01', freq='MS')

    return first_range.append(second_range)


def generate_split_array(verbose=1):
    start_prediction_months = generate_start_prediction_months()
    split_array = np.empty(np.shape(start_prediction_months), dtype=object)
    
    for i,month in enumerate(start_prediction_months):
        if month in config.TRAIN_MONTHS: split_array[i] = "train"
        if month in config.VAL_MONTHS: split_array[i] = "val"
        if month in config.TEST_MONTHS: split_array[i] = "test"

    if verbose == 2: print_split_stats(split_array)
    
    return split_array, start_prediction_months


def train_model(model, device, model_hyperparam_configs, optimizer, criterion,
                plot_training_curve=True, verbose=1, save_val_predictions=True):

    # read in some metadata
    configuration = model_hyperparam_configs["input_config"]
    batch_size = model_hyperparam_configs["batch_size"]
    early_stopping = model_hyperparam_configs["early_stopping"]
    patience = model_hyperparam_configs["patience"]
    model_name = model_hyperparam_configs["name"]
    num_epochs = model_hyperparam_configs["max_epochs"]

    # generate array labeling which months are train/val/test
    split_array, start_prediction_months = generate_split_array()

    # create dataset instances for training, validation, and testing
    data_pairs_directory = os.path.join(config.DATA_DIRECTORY, 'sicpred/data_pairs_npy')
    train_dataset = SeaIceDataset(data_pairs_directory, configuration, split_array, start_prediction_months, split_type='train', target_shape=(336, 320))
    val_dataset = SeaIceDataset(data_pairs_directory, configuration, split_array, start_prediction_months, split_type='val', target_shape=(336, 320))
    test_dataset = SeaIceDataset(data_pairs_directory, configuration, split_array, start_prediction_months, split_type='test', target_shape=(336, 320))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    train_losses = []
    val_losses = []
    val_predictions = []
    best_val_loss = float('inf')
    epoch_at_early_stopping = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, target_months in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets, target_months)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        if verbose >= 1: print(f"Epoch [{epoch+1}/{num_epochs}], train loss: {epoch_loss:.4f}", end=', ')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, target_months in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets, target_months)
                val_loss += loss.item() * inputs.size(0)        

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        if verbose >= 1: print(f"validation loss: {val_loss:.4f}")

        # Early stopping check
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model weights
                os.makedirs(f"{config.DATA_DIRECTORY}/sicpred/models", exist_ok=True)
                torch.save(model.state_dict(), f"{config.DATA_DIRECTORY}/sicpred/models/{model_name}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1: print("Early stopping triggered")
                    epoch_at_early_stopping = epoch
                    break
        
    # Plot training curves
    if plot_training_curve:
        if verbose >= 1: print("Plotting training curve... ", end="")
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(0.5, len(train_losses) + 0.5, 1), train_losses, label="train")
        plt.plot(np.arange(1, len(val_losses) + 1, 1), val_losses, label="val")
        plt.legend()
        plt.title(model_name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()

        os.makedirs(f"{config.DATA_DIRECTORY}/figures/training_curves", exist_ok=True)
        plt.savefig(f"{config.DATA_DIRECTORY}/figures/training_curves/{model_name}_training_curve.png", bbox_inches='tight', dpi=300)
        plt.close()
        if verbose >= 1: print("done!")

    # Save the validation predictions for hyperparam tuning 
    if save_val_predictions:
        model.eval()
        with torch.no_grad():
            for inputs, targets, target_months in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_predictions.append(outputs.cpu().numpy())
        os.makedirs(f"{config.DATA_DIRECTORY}/sicpred/val_predictions", exist_ok=True)
        np.save(f"{config.DATA_DIRECTORY}/sicpred/val_predictions/{model_name}_val_predictions.npy", np.concatenate(val_predictions, axis=0))

    # Save the trained model
    if verbose >= 1: print("Saving model weights...", end='')

    # record the best val loss and, if early stopping happened, the last epoch
    model_hyperparam_configs["best_val_loss"] = best_val_loss

    if epoch_at_early_stopping is not None: 
        model_hyperparam_configs["final_epoch"] = epoch_at_early_stopping
    else:
        model_hyperparam_configs["final_epoch"] = num_epochs
        
    os.makedirs(f"{config.DATA_DIRECTORY}/sicpred/val_predictions", exist_ok=True)
    save_dict_to_pickle(model_hyperparam_configs, f"{config.DATA_DIRECTORY}/sicpred/models/{model_name}.pkl")
    torch.save(model.state_dict(), f"{config.DATA_DIRECTORY}/sicpred/models/{model_name}.pth")
    if verbose >= 1: print("done! \n\n")

    return train_losses, val_losses


##########################################################################################
# Sea ice diagnostics 
##########################################################################################

def calculate_SIA(sector="all"):
    ### TODO

    return None

def calculate_SIE(sector="all"):
    ### TODO 

    return None