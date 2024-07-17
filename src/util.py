import xarray as xr
import numpy as np
import pandas as pd
from time import time
import config
import os

##########################################################################################
# Data preprocessing stuff 
##########################################################################################

def prep_prediction_samples(): 
    """
    Returns 
    input tuple[0]: ndarray (n_samples, n_lat, n_lon, n_vars)
    output tuple[1]: ndarray (n_samples, n_lat, n_lon)
    """
    input = None
    output = None

    return input, output 



def normalize_data(overwrite=False, verbose=1):
    """ 
    Normalize inputs and save. 
    """

    vars_to_normalize = config.era5_variables_dict.keys()
    save_dir = os.path.join(config.DATA_DIRECTORY, "sicpred/normalized_inputs")

    for var_name in vars_to_normalize:
        if os.path.exists(os.path.join(save_dir, f"{var_name}_norm.nc")):
            if verbose == 1: print(f"Already found normalized file for {var_name}. Skipping...")
            continue

        print(f"Normalizing {var_name}...")
        if var_name == "siconc":
            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc")
            da = ds[var_name].sel(time=config.TRAIN_MONTHS)

        elif var_name == "geopotential":
            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_500hPa_SPS.nc")
            da = ds[config.era5_variables_dict[var_name]["short_name"]].sel(time=config.TRAIN_MONTHS)

        else:
            if not os.path.exists(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_SPS.nc"):
                raise FileNotFoundError(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_SPS.nc does not exist!")

            ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/ERA5/{var_name}_SPS.nc")
            da = ds[config.era5_variables_dict[var_name]["short_name"]].sel(time=config.TRAIN_MONTHS)

        print("Calculating means and stdev...", end="")
        monthly_means = da.groupby("time.month").mean("time")
        monthly_stdevs = da.groupby("time.month").std("time")
        print("done!")
        
        months = da['time'].dt.month
        normalized_da = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s,
            da,
            monthly_means.sel(month=months),
            monthly_stdevs.sel(month=months),
            output_dtypes=[da.dtype]
        )

        # Make a new folder to save the normalized variables 
        os.makedirs(save_dir, exist_ok=True)
        print("Saving...", end="")
        normalized_da.to_netcdf(os.path.join(save_dir, f"{var_name}_norm.nc"))
        print("done!")



def remove_expver_from_era5(verbose=1):
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
                print(f"expver not found in {variable}. Skipping... ")
                continue
            else: continue 

        print(f"Removing expver dimension from {var_name}...", end=" ")
        combined_ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

        temp_file_path = file_path + '.tmp'
        combined_ds.to_netcdf(temp_file_path)
        combined_ds.close()
        os.replace(temp_file_path, file_path)

        print("done!")
        


def concatenate_nsidc(working_path=None, overwrite=False):
    """ 
    Merge the NSIDC data into one netCDF file called seaice_conc_monthly_all 
    """
    print("1) Merge monthly NSIDC files")
    if working_path == None:
        working_path = os.path.join(config.DATA_DIRECTORY, 'NSIDC')

    if not os.path.exists(working_path):
        raise FileNotFoundError(f"{working_path} does not exist. make sure the NSIDC data is in its own directory")
    
    save_path = f'{working_path}/seaice_conc_monthly_all.nc'
    if os.path.exists(save_path) and not overwrite:
        print(f"Already found file {save_path}. Skipping...")
        return

    all_files = os.listdir(working_path)

    # Files that start with "icdr" correspond to the Near Real Time product, which is updated past the last year
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
    ds_concat.to_netcdf(save_path)
    print('done!')



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