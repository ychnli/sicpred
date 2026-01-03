import os 
import xarray as xr
import numpy as np 
import pandas as pd
import time
import pyproj

from src import config_cesm as config
from src.utils.util_shared import write_nc_file

ALL_VAR_NAMES = config.ALL_VAR_NAMES
LAND_MASK_PATH = os.path.join(config.DATA_DIRECTORY, "cesm_data", "grids", "land_mask.nc")

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


def normalize(x, m, s):
    # Avoid divide by zero by setting normalized value to zero where std deviation is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (x - m) / s
        normalized = np.where(s == 0, 0, normalized)  # Set to zero where std dev is zero

    return normalized


def detrend_quadratic(da, time_dim='time'):
    time_vals = da[time_dim].dt.year + da[time_dim].dt.month / 12.0
    time_numeric = xr.DataArray(time_vals, coords={time_dim: da[time_dim]}, dims=time_dim)

    spatial_dims = [d for d in da.dims if d not in [time_dim, 'member_id']]
    da_stacked = da.stack(space=spatial_dims)
    detrended = da_stacked.copy()

    coeffs_all = []

    for month in range(1, 13):
        sel = da_stacked[time_dim].dt.month == month
        if not sel.any(): continue

        t = time_numeric.sel({time_dim: sel})
        Y = da_stacked.sel({time_dim: sel})
        if "member_id" in Y.dims:
            Y = Y.mean('member_id')
        X = np.stack([np.ones_like(t), t, t**2], axis=1)

        beta = np.linalg.solve(X.T @ X, X.T @ Y.values)  # (3, space)
        coeffs_all.append((month, beta))

        trend = xr.DataArray((X @ beta).astype(Y.dtype), coords=Y.coords, dims=Y.dims)
        detrended.loc[{time_dim: sel}] = da_stacked.sel({time_dim: sel}) - trend

    detrended_unstacked = detrended.unstack('space')

    # Build coeff DataArray
    coeffs = np.full((12, 3) + (da_stacked.sizes['space'],), np.nan, dtype=da.dtype)
    for month, beta in coeffs_all:
        coeffs[month - 1] = beta 

    coeff_da = xr.DataArray(
        coeffs,
        dims=('month', 'coeff', 'space'),
        coords={
            'month': np.arange(1, 13),
            'coeff': ['const', 'linear', 'quadratic'],
            'space': da_stacked.coords['space']
        }
    ).unstack('space')

    return detrended_unstacked, coeff_da



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


def merge_data_by_member(overwrite=False):
    """
    Merge CESM ensemble-member NetCDF files into a single dataset indexed
    by member_id, then delete the individual member files.
    """
    for var_name in ALL_VAR_NAMES:
        var_dir = os.path.join(config.DATA_DIRECTORY, "cesm_data", var_name)
        file_list = sorted(os.listdir(var_dir))
        combined_fname = f"{var_name}_combined.nc"
        combined_path = os.path.join(var_dir, combined_fname)

        if combined_fname in file_list and not overwrite:
            continue

        member_files = [
            f for f in file_list
            if f.endswith(".nc") and f != combined_fname
        ]

        ds_list = []
        try:
            for f in member_files:
                path = os.path.join(var_dir, f)
                ds = xr.open_dataset(path, chunks={"time": 120})
                # use pandas time index instead of cftime
                ds = ds.assign_coords(time=pd.date_range("1850-01", "2014-12", freq="MS"))
                ds_list.append(ds)

            merged_ds = xr.concat(ds_list, dim="member_id")
            write_nc_file(merged_ds, combined_path, overwrite)
        finally:
            for ds in ds_list:
                ds.close()

        # delete individual member files after successful write
        for f in member_files:
            os.remove(os.path.join(var_dir, f))


def normalize_data(var_name, data_split_settings, max_lag_months, max_lead_months=6,
                    overwrite=False, verbose=1, divide_by_stdev=False, use_min_max=True, 
                    detrend=True):
    """ 
    Normalize inputs based on statistics of the training data and save. 

    Param:
        (string)    var_name: the standard name of the variable
        (dict)      normaliation_scheme: a dict specifying how the data is split 
        (bool)      overwrite 
        (int)       verbose  
        (bool)      divide_by_stdev: if True, computes (x - mu)/(sigma)
                                     if False, computes (x - mu)  
        (bool)      use_min_max: if True, computes (x - min(x))/(max(x) - min(x))
        (bool)      detrend: if True, detrends using quadratic least squares fit
    
    Note that use_min_max and divide_by_stdev should not both be set to true
    """

    if divide_by_stdev and use_min_max:
        raise ValueError("divide_by_stdev and use_min_max should not both be set to true")

    save_dir = os.path.join(config.PROCESSED_DATA_DIRECTORY, "normalized_inputs", data_split_settings["name"])
    save_path = os.path.join(save_dir, f"{var_name}_norm.nc")
        
    if os.path.exists(save_path) and not overwrite:
        if verbose >= 1: print(f"Already found normalized file for {var_name}. Skipping...")
        return None

    print(f"Normalizing {var_name}, divide_by_stdev = {divide_by_stdev}...", end=" ")
    if data_split_settings["member_ids"] == "obs":
        merged_ds = xr.open_dataset(os.path.join(config.DATA_DIRECTORY, "obs_data", f"{var_name}_obs.nc"))
    else:
        merged_ds = xr.open_dataset(os.path.join(config.DATA_DIRECTORY, "cesm_data", var_name, f"{var_name}_combined.nc"))

    # create a subsetted DataArray that contains the data requested by data_split_settings
    da = merged_ds[var_name]

    if data_split_settings["split_by"] == "time": 
        all_times = get_start_prediction_months(data_split_settings)

        # all_times gives all time coordinates that are accessed, given variable leads and lags
        all_times = pd.date_range(all_times[0] - pd.DateOffset(months=max_lag_months), 
                                  all_times[-1] + pd.DateOffset(months=max_lead_months-1),
                                  freq="MS")        
        da = da.sel(time=all_times, member_id=data_split_settings["member_ids"]) 
        da_train_subset = da.sel(time=data_split_settings["train"])

    elif data_split_settings["split_by"] == "ensemble_member": 
        all_member_ids = data_split_settings["train"] + data_split_settings["val"] + data_split_settings["test"]
        
        all_times = data_split_settings["time_range"]
        all_times = pd.date_range(all_times[0] - pd.DateOffset(months=max_lag_months), 
                                  all_times[-1] + pd.DateOffset(months=max_lead_months-1),
                                  freq="MS") 
        da = da.sel(member_id=all_member_ids, time=all_times) 
        da_train_subset = da.sel(member_id=data_split_settings["train"])
        
    else:
        raise ValueError("data_split_settings split_by must be 'time' or 'ensemble_member'")

    dims_to_reduce = [d for d in ["time", "member_id"] if d in da.dims]

    if divide_by_stdev:
        print("calculating means and stdev...", end=" ")

        monthly_means = da_train_subset.groupby("time.month").mean(dim=dims_to_reduce).load()
        monthly_stdevs = da_train_subset.groupby("time.month").std(dim=dims_to_reduce).load()
        print("done!")

        months = da['time'].dt.month
        normalized_da = xr.apply_ufunc(
            normalize,
            da,
            monthly_means.sel(month=months),
            monthly_stdevs.sel(month=months),
            output_dtypes=[da.dtype],
            dask="allowed"
        )
        
        monthly_means_ds = monthly_means.to_dataset(name=var_name)
        monthly_stdevs_ds = monthly_stdevs.to_dataset(name=var_name)
    else: 
        if use_min_max:
            print("calculating min and max...", end=" ")
            monthly_mins = da_train_subset.groupby("time.month").min(dim=dims_to_reduce).load()
            monthly_maxs = da_train_subset.groupby("time.month").max(dim=dims_to_reduce).load()
            print("done!")

            months = da['time'].dt.month
            normalized_da = (da - monthly_mins.sel(month=months)) / (monthly_maxs.sel(month=months) - monthly_mins.sel(month=months))

            monthly_mins_ds = monthly_mins.to_dataset(name=var_name)
            monthly_maxs_ds = monthly_maxs.to_dataset(name=var_name)

        else:
            print("calculating means...", end=" ")
            monthly_means = da_train_subset.groupby("time.month").mean(dim=dims_to_reduce).load()
            print("done!")

            months = da['time'].dt.month
            normalized_da = da - monthly_means.sel(month=months)
            
            monthly_means_ds = monthly_means.to_dataset(name=var_name)

    if detrend:
        print("detrending data with quadratic fit...", end=" ")
        normalized_da, coeffs = detrend_quadratic(normalized_da)
        write_nc_file(coeffs.to_dataset(name=var_name), 
                      os.path.join(save_dir, f"{var_name}_detrend_coeffs.nc"), overwrite)
        print("done!")

    normalized_ds = normalized_da.to_dataset(name=var_name)
    if "member_id" not in normalized_ds.indexes:
        normalized_ds = normalized_ds.expand_dims("member_id")

    print("Saving...", end="")
    if divide_by_stdev: 
        write_nc_file(monthly_stdevs_ds, os.path.join(save_dir, f"{var_name}_stdev.nc"), overwrite)
        write_nc_file(normalized_ds, save_path, overwrite)
        write_nc_file(monthly_means_ds, os.path.join(save_dir, f"{var_name}_mean.nc"), overwrite)
    else:
        if use_min_max:
            write_nc_file(monthly_mins_ds, os.path.join(save_dir, f"{var_name}_min.nc"), overwrite)
            write_nc_file(monthly_maxs_ds, os.path.join(save_dir, f"{var_name}_max.nc"), overwrite)
            write_nc_file(normalized_ds, save_path, overwrite)
        else:
            write_nc_file(normalized_ds, save_path, overwrite)
            write_nc_file(monthly_means_ds, os.path.join(save_dir, f"{var_name}_mean.nc"), overwrite)
    print("done!")



def load_inputs_data_da_dict(input_config, data_split_settings):
    """
    This function loads in combined data (after normalization) into a dictionary.

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
    if data_split_settings["member_ids"] != ["obs"]:
        common_member_ids = set()
        for i,da in enumerate(data_da_dict.values()): 
            if i == 0: 
                common_member_ids = set(da.member_id.data)
            else:
                common_member_ids = common_member_ids & set(da.member_id.data)

    for var, da in data_da_dict.items():
        if data_split_settings["member_ids"] != ["obs"]:
            da = da.sel(member_id = list(common_member_ids))
        
        # drop this auxiliary variable that is leftover from normalization 
        da = da.drop_vars("month")
        data_da_dict[var] = da

    return data_da_dict



def save_inputs_files(input_config, save_path, data_split_settings, overwrite=False):
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
    land_mask = xr.open_dataset(LAND_MASK_PATH).mask.data
    land_mask = np.transpose(land_mask.reshape(1, 80, 80), [0, 2, 1]) # for some reason, x and y get switched
    
    # save each ensemble member separately so the files don't get too big 
    member_ids = data_da_dict["icefrac"].member_id.data
    start_prediction_months = get_start_prediction_months(data_split_settings)
    for member_id in member_ids:
        save_name = os.path.join(save_path, f"inputs_member_{member_id}.nc")
        if os.path.exists(save_name) and not overwrite:
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
        if "z_t" in da_merged.dims: 
            da_merged = da_merged.drop_vars("z_t")
        if "lev" in da_merged.dims:
            da_merged = da_merged.drop_vars("lev")
        
        
        print("done! Saving...")
        ds = da_merged.to_dataset(name="data")
        write_nc_file(ds, save_name, overwrite)

        end_time = time.time()
        print(f"done! Elapsed time: {end_time - start_time:.2f} seconds")


def save_targets_files(input_config, target_config, save_path, max_lead_months, data_split_settings, overwrite=False):
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
        if os.path.exists(save_name) and not overwrite:
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
        ds = da_merged.to_dataset(name="data")
        write_nc_file(ds, save_name, overwrite)

        end_time = time.time()
        print(f"done! Elapsed time: {end_time - start_time:.2f} seconds")


def get_num_input_channels(input_config):
    """
    Get the number of input channels from the input_config dict. 

    Param:
        (dict)      input_config

    Returns:
        (int)       num_channels
    """

    num_channels = 0
    for _, var_params in input_config.items():
        if var_params["include"]:
            if var_params["auxiliary"]:
                num_channels += 1
            else:
                num_channels += var_params["lag"]

    return num_channels


def get_num_output_channels(max_lead_months, target_config):
    if not target_config["predict_classes"]:
        return max_lead_months
    else:
        raise NotImplementedError()
    
def save_land_mask():
    """
    This saves the SST land mask (i.e., a boolean map equal to 1 where SST is NaN)

    This is the same land mask as icethick, but NOT the same as icefrac. See the function below
    for more info. 
    """
    save_dir = os.path.join(config.DATA_DIRECTORY, "cesm_data", "grids")
    os.makedirs(save_dir, exist_ok=True)
    save_path = LAND_MASK_PATH

    if os.path.exists(save_path): return 

    ds = xr.open_dataset(os.path.join(config.DATA_DIRECTORY, "cesm_data", "sst", "sst_combined.nc"))
    land_mask = np.isnan(ds.temp.isel(time=0, member_id=0))
    land_mask = land_mask.to_dataset(name="mask").drop_vars(("member_id","z_t","time"))

    land_mask.to_netcdf(save_path)

def save_icefrac_land_mask():
    """
    Note: this is DISTINCT from the SST land mask. The icefrac variable seems to account for
    more detailed coastlines and is therefore nonzero in coastline areas where SST is NaN. 
    
    Thus in general, the region where icefrac is always 0 is smaller than the SST & icethick 
    land mask. We will define the icefrac land mask as the intersection of the SST land mask 
    and the region where icefrac is always 0. 
    """
    save_dir = os.path.join(config.DATA_DIRECTORY, "cesm_data", "grids")
    save_path = os.path.join(save_dir, "icefrac_land_mask.nc")

    if os.path.exists(save_path): return 

    ds = xr.open_dataset(f"{config.DATA_DIRECTORY}/cesm_data/icefrac/icefrac_combined.nc")

    # we'll use 5 ensemble members to calculate the region where icefrac is always 0
    icefrac_zero_mask = ds["icefrac"].isel(member_id=slice(0,5)).mean(("member_id", "time")) == 0
    
    # get the SST land mask
    try:
        sst_land_mask = xr.open_dataset(LAND_MASK_PATH).mask
    except:
        raise Exception("Didn't find the sst land mask variable. You should call save_land_mask first")

    icefrac_land_mask = np.logical_and(sst_land_mask, icefrac_zero_mask)
    icefrac_land_mask.to_dataset(name="mask").to_netcdf(save_path)


def generate_sps_grid(grid_size=80, lat_boundary=-52.5):
    # Define the South Polar Stereographic projection (EPSG:3031)
    proj_south_pole = pyproj.Proj(proj='stere', lat_0=-90, lon_0=0, lat_ts=-70)

    # Define the geographic coordinate system (EPSG:4326)
    proj_geographic = pyproj.Proj(proj='latlong', datum='WGS84')

    # Create a transformer object for forward (Stereographic to Geographic) transformations
    transformer = pyproj.Transformer.from_proj(proj_south_pole, proj_geographic, always_xy=True)

    # Compute the maximum radius from the South Pole in stereographic coordinates
    _, max_radius = proj_south_pole(0, lat_boundary)

    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    X, Y = np.meshgrid(x, y)

    # Transform coordinates to geographic (lat, lon) for the cell centers
    lon, lat = transformer.transform(X, Y)

    # Compute the edges (boundaries) of the grid cells
    x_edges = np.linspace(-max_radius, max_radius, grid_size + 1)
    y_edges = np.linspace(-max_radius, max_radius, grid_size + 1)
    X_edges, Y_edges = np.meshgrid(x_edges, y_edges)

    # Transform the edges to geographic coordinates
    lon_edges, lat_edges = transformer.transform(X_edges, Y_edges)

    # Calculate grid cell areas
    geod = pyproj.Geod(ellps="WGS84")
    areas = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            # Define the four corners of the grid cell using edges
            lons = [lon_edges[i, j], lon_edges[i + 1, j], lon_edges[i + 1, j + 1], lon_edges[i, j + 1]]
            lats = [lat_edges[i, j], lat_edges[i + 1, j], lat_edges[i + 1, j + 1], lat_edges[i, j + 1]]
            
            # Compute the polygon area
            area, _ = geod.polygon_area_perimeter(lons, lats)
            areas[i, j] = abs(area)  # Area might be negative due to polygon orientation

    # Create the output dataset
    output_grid = xr.Dataset(
        {
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
            "area": (["y", "x"], areas),
        },
        coords={
            "x": (["x"], x),
            "y": (["y"], y),
        }
    )

    return output_grid


def calculate_area_weights():
    """
    Calculates a grid of weights according to grid cell area 

    Returns:
        (np.array)      weights (shape=(80,80))
    """

    grid = generate_sps_grid()
    
    weights = (grid.area / np.mean(grid.area)).values 

    return weights 


def calculate_monthly_weights(data_split_settings):
    """
    Calculates the monthly weights based on the seasonal cycle of mean squared sea ice concentration anomaly
    
    Param:
        (dict)      data_split_settings
    Returns:
        (np.array)  weights (12,)
    """

    fp = os.path.join(config.PROCESSED_DATA_DIRECTORY, "normalized_inputs", data_split_settings["name"], "icefrac_norm.nc")
    with xr.open_dataset(fp) as ds: 
        dims_to_reduce = [d for d in ["time", "member_id", "x", "y"] if d in ds.dims]
        weights = (ds["icefrac"] ** 2).groupby("time.month").mean(dims_to_reduce)
        weights = (1/weights)
        weights = weights / weights.mean()
        weights = weights.values
    
    return weights


def generate_empty_predictions_ds(
    time_coords, 
    ensemble_members, 
    num_nn_members, 
    max_lead_time=6, 
    reference_grid=generate_sps_grid()
):
    x_dim, y_dim = reference_grid.sizes["x"], reference_grid.sizes["y"]
    num_members = len(ensemble_members)

    return xr.Dataset(
        {
            "predictions": (
                ["start_prediction_month", "member_id", "nn_member_id", "lead_time", "y", "x"],
                np.full((len(time_coords), num_members, num_nn_members, max_lead_time, y_dim, x_dim), 
                        np.nan, dtype=np.float32)
            )
        },
        coords={
            "start_prediction_month": time_coords,
            "member_id": ensemble_members,
            "nn_member_id": np.arange(num_nn_members),
            "lead_time": np.arange(1, max_lead_time + 1),
            "y": reference_grid.y.values,
            "x": reference_grid.x.values,
        }
    )