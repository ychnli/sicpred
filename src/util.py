import xarray as xr
import numpy as np
import pandas as pd
from time import time
import config
import os

##########################################################################################
# Data preprocessing 
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



def write_nc_file(ds, save_path, overwrite, verbose=1):
    if type(overwrite) != bool:
        raise TypeError("overwrite needs to be bool")

    if overwrite:
        if not os.path.exists(save_path):
            if verbose == 2: print(f"Nothing to overwrite: {save_path} doesn't exist yet!")
            return 

        temp_path = save_path + '.tmp'
        ds.to_netcdf(temp_path)
        os.replace(temp_path, save_path)
        if verbose == 2: print(f"Overwrote {save_path}")

    else: 
        ds.to_netcdf(save_path)
        if verbose == 2: print(f"Saved to {save_path}")


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


def normalize_train_data(overwrite=False, verbose=1, vars_to_normalize="all"):
    """ 
    Normalize inputs and save. 
    """

    if vars_to_normalize == "all":
        vars_to_normalize = list(config.era5_variables_dict.keys())
        vars_to_normalize.remove("sea_ice_cover")
    
    save_dir = os.path.join(config.DATA_DIRECTORY, "sicpred/normalized_inputs")

    for var_name in vars_to_normalize:
        if os.path.exists(os.path.join(save_dir, f"{var_name}_norm.nc")) and not overwrite:
            if verbose == 1: print(f"Already found normalized file for {var_name}. Skipping...")
            continue

        print(f"Normalizing {var_name}...", end=" ")
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

        print("calculating means and stdev...", end=" ")
        monthly_means = da.groupby("time.month").mean("time")
        monthly_stdevs = da.groupby("time.month").std("time")
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

        normalized_ds = normalized_da.to_dataset(name=config.era5_variables_dict[var_name]["short_name"])
        monthly_means_ds = monthly_means.to_dataset(name=config.era5_variables_dict[var_name]["short_name"])
        monthly_stdevs_ds = monthly_stdevs.to_dataset(name=config.era5_variables_dict[var_name]["short_name"])

        # Make a new folder to save the normalized variables 
        os.makedirs(save_dir, exist_ok=True)
        print("Saving...", end="")
        write_nc_file(monthly_means_ds, os.path.join(save_dir, f"{var_name}_mean.nc"), overwrite)
        write_nc_file(monthly_stdevs_ds, os.path.join(save_dir, f"{var_name}_stdev.nc"), overwrite)
        write_nc_file(normalized_ds, os.path.join(save_dir, f"{var_name}_norm.nc"), overwrite)
        print("done! \n\n")



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
        
        write_nc_file(combined_ds, file_path, overwrite=True)

        print("done! \n\n")
        


def concatenate_nsidc(working_path=None, overwrite=False):
    """ 
    Merge the NSIDC data into one netCDF file called seaice_conc_monthly_all 
    """
    print("Merging monthly NSIDC files... ", end="")
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

    nsidc_sic = xr.open_dataset(f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc')

    # The missing data lives as NaN. Check if we need to get rid of it by finding years where
    # there are NaN values 
    needs_fixing = len(nsidc_sic.time[np.isnan(nsidc_sic.siconc).sum(dim=('x', 'y')) != 0]) != 0 

    if not needs_fixing:
        if verbose == 2: print("Didn't find any NaN values in NSIDC dataset. Skipping...")
        return 
    
    if verbose == 1: print("Removing missing ice data in 1987 Dec and 1988 Jan...")
    nsidc_sic = nsidc_sic.sel(time=nsidc_sic.time[np.isnan(nsidc_sic.siconc).sum(dim=('x', 'y')) == 0])

    temp_path = f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc.tmp'
    nsidc_sic.to_netcdf(temp_path)
    os.replace(temp_path, f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc')


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