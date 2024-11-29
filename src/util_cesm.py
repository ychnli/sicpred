import os 
import xarray as xr
import numpy as np 

from src import util_era5
from src import config_cesm
from src.util_era5 import write_nc_file

VAR_NAMES = config_cesm.VAR_NAMES

def find_downloaded_vars():
    """
    Finds and verifies the downloaded variables and their ensemble members.
        
    Raises:
        ValueError: If member IDs do not match across variables at any index.

    """

    member_ids = np.empty((len(VAR_NAMES), 100), dtype='object')
    n_members = []

    for i,variable in enumerate(VAR_NAMES):
        directory = os.path.join(config_cesm.RAW_DATA_DIRECTORY, variable)
        
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


def normalize_data(overwrite=False, verbose=1, vars_to_normalize="all"):
    """ 
    Normalize inputs based on statistics of the training data and save. 
    """

    if vars_to_normalize == "all":
        vars_to_normalize = VAR_NAMES
    
    if verbose >= 1: print(f"Normalizing variables {vars_to_normalize}")

    save_dir = os.path.join(config.RAW_DATA_DIRECTORY, "/normalized_inputs")
    os.makedirs(save_dir, exist_ok=True)

    for var_name in vars_to_normalize:
        save_path = os.path.join(save_dir, f"{var_name}_norm.nc")
        if os.path.exists(save_path) and not overwrite:
            if verbose >= 1: print(f"Already found normalized file for {var_name}. Skipping...")
            continue

        print(f"Normalizing {var_name}...", end=" ")

        # First make a merged dataset from the separate ones 
        file_list = sorted(os.listdir(f"{config.RAW_DATA_DIRECTORY}/{var_name}"))
        ds_list = []

        for file in file_list:
            ds = xr.open_dataset(os.path.join(f"{config.RAW_DATA_DIRECTORY}/{var_name}", file), chunks={'time': 120})
            ds_list.append(ds)

        merged_ds = xr.concat(ds_list, dim="member_id")

        # change the time index to pandas instead of cftime 
        merged_ds[var_name].assign_coords(time=pd.date_range("1850-01", "2100-12", freq="MS"))

        # save the merged ds before normalizing 
        merged_ds.to_netcdf(f"{config.RAW_DATA_DIRECTORY}/{var_name}/{var_name}_combined.nc")
        
        # now calculate the climatology. We define this as the period from 1850 to 1980 
        # across all ensemble members. This means that the climate change signal, especially
        # for the ssp simulations, will be present. 
        da = ds[var_name]
        print("calculating means and stdev...", end=" ")

        time_subset = pd.date_range("1850-01", "1979-12", freq="MS")
        monthly_means = da.sel(time=time_subset).groupby("time.month").mean("time", "member_id").load()
        monthly_stdevs = da.sel(time=time_subset).groupby("time.month").std("time", "member_id").load()
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
        
        normalized_ds = normalized_da.to_dataset(name=var_name)
        monthly_means_ds = monthly_means.to_dataset(name=var_name)
        monthly_stdevs_ds = monthly_stdevs.to_dataset(name=var_name)

        print("Saving...", end="")
        write_nc_file(monthly_means_ds, os.path.join(save_dir, f"{var_name}_mean.nc"), overwrite)
        write_nc_file(monthly_stdevs_ds, os.path.join(save_dir, f"{var_name}_stdev.nc"), overwrite)
        write_nc_file(normalized_ds, os.path.join(save_dir, f"{var_name}_norm.nc"), overwrite)
        print("done!")

    print("done! \n\n")
