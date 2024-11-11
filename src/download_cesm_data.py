import intake
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import time
import os
import pyproj
from src import config 

#################### configure these! ####################
download_settings = {
    # variables to download 
    "vars": ["TEMP"], #["ICEFRAC", "TEMP", "FLNS", "FSNS", "PSL", "Z3", "U", "V", "hi"],

    "chunk": "default",
    
    "member_id": "all", 

    "save_directory": "/scratch/users/yucli/"
}

var_args = {
    "ICEFRAC": {
        "p_index": None,
        "save_name": "icefrac",
        "long_name": "Sea ice fraction"
    },
    "TEMP": {
        "p_index": 0,
        "lat_slice": slice(0, 93),
        "save_name": "temp",
        "long_name": "Sea surface temperature"
    },
    "FLNS": {
        "p_index": None,
        "save_name": "lw_flux",
        "long_name": "Net longwave flux at surface"
    },
    "FSNS": {
        "p_index": None,
        "save_name": "sw_flux",
        "long_name": "Net shortwave flux at surface"
    },
    "PSL": {
        "p_index": None,
        "save_name": "psl",
        "long_name": "Sea level pressure"
    },
    "Z3": {
        "p_index": 20,
        "save_name": "geopotential",
        "long_name": "Geopotential height at 500 hPa"
    },
    "U": {
        "p_index": -1,
        "save_name": "ua",
        "long_name": "Surface zonal wind"
    },
    "V": {
        "p_index": -1,
        "save_name": "va",
        "long_name": "Surface meridional wind"
    },
    "hi": {
        "p_index": None,
        "lat_slice": slice(0, 93),
        "save_name": "icethick",
        "long_name": "Sea ice thickness"
    }
}

#################### constants ####################
DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'

CATALOG = intake.open_esm_datastore(
    'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
)

CESM_OCEAN_GRID = xr.open_dataset(f"{DATA_DIRECTORY}/cesm_lens/grids/ocean_grid.nc")


#################### functions ####################
def retrieve_variable_dataset(catalog, variable, verbose=1):
    """
    
    """

    if verbose >= 1: print(f"Finding {variable}...", end="")
    catalog_subset = CATALOG.search(variable=variable, frequency='monthly')

    if len(catalog_subset.df) == 0: 
        if verbose >= 1: print(f"did not find any saved data. Skipping...")
        return None 
        
    if len(catalog_subset.df) != 4: 
        if verbose >= 1: print(f"only found {len(catalog_subset.df)} saved experiments instead of expected (4)")

    dsets = catalog_subset.to_dataset_dict(storage_options={'anon':True})
    if verbose >= 1: print("done! merging datasets...", end="")

    # get the model component and save it to var_args dict 
    component = catalog_subset.df.component[0]
    var_args[variable]["component"] = component 
    
    cmip_hist_ds = dsets.get(f"{component}.historical.monthly.cmip6", None)
    smbb_hist_ds = dsets.get(f"{component}.historical.monthly.smbb", None)
    cmip_ssp_ds = dsets.get(f"{component}.ssp370.monthly.cmip6", None)
    smbb_ssp_ds = dsets.get(f"{component}.ssp370.monthly.smbb", None)

    cmip_merge_ds, smbb_merge_ds = None, None
    if cmip_hist_ds and cmip_ssp_ds:
        cmip_merge_ds = xr.concat([cmip_hist_ds, cmip_ssp_ds], dim='time') 
    if smbb_hist_ds and smbb_ssp_ds: 
        smbb_merge_ds = xr.concat([smbb_hist_ds, smbb_ssp_ds], dim='time')
    if cmip_merge_ds and smbb_merge_ds:
        merged_ds = xr.concat([cmip_merge_ds, smbb_merge_ds], dim='member_id')
    elif cmip_merge_ds:
        merged_ds = cmip_merge_ds
    else:
        raise Exception(f"{variable} seems to be missing CMIP protocol ensemble members")

    if verbose >= 1: print("done! \n")
    return merged_ds 

def subset_variable_dataset(merged_ds, variable, member_id, chunk="default", var_args=var_args):
    print(f"Subsetting ensemble member {member_id}... ", end="")
    
    component = var_args[variable]["component"]
    subset = merged_ds[variable].isel(member_id=member_id) 
    
    if chunk == "default":
        chunk_settings = {"time": 502}
        # chunk the depths individually so we don't download every depth?
        if var_args[variable]["p_index"] is not None: 
            if component == "ocn": chunk_settings["z_t"] = 1
            elif component == "atm": chunk_settings["lev"] = 1
        subset = subset.chunk(chunk_settings)
    else: 
        subset = subset.chunk(chunk)

    # all the atmospheric variables can be subsetted simply
    if component == "atm":
        subset = subset.sel(lat=slice(-90, -30))
    elif component == "ice":
        subset = subset.sel(nj=var_args[variable]["lat_slice"])
    else: 
        subset = subset.sel(nlat=var_args[variable]["lat_slice"])

    # subset z/p dimension, if it exists
    if var_args[variable]["p_index"] is not None: 
        if component == "ocn":
            subset = subset.isel(z_t=var_args[variable]["p_index"])
        else:
            subset = subset.isel(lev=var_args[variable]["p_index"])

    # add latitude/longitude to ocean and ice variables 
    if component != "atm": 
        if component == "ocn":
            lat_index_name, lon_index_name = "nlat", "nlon"
        elif component == "ice":
            lat_index_name, lon_index_name = "nj", "ni"

        lat = CESM_OCEAN_GRID.lat.sel(nlat=var_args[variable]["lat_slice"]).data
        lon = CESM_OCEAN_GRID.lon.sel(nlat=var_args[variable]["lat_slice"]).data

        subset = subset.assign_coords(lat=([lat_index_name, lon_index_name], lat), lon=([lat_index_name, lon_index_name], lon))
    
    print("done!")
    
    # turn subset into a dataset and rename it 
    subset = subset.to_dataset(name=var_args[variable]["save_name"])

    return subset

def generate_sps_grid(grid_size=80, lat_boundary=-52.5):
    # Define the South Polar Stereographic projection (EPSG:3031)
    proj_south_pole = pyproj.Proj(proj='stere', lat_0=-90, lon_0=0, lat_ts=-70)

    # Define the geographic coordinate system (EPSG:4326)
    proj_geographic = pyproj.Proj(proj='latlong', datum='WGS84')

    # Compute the maximum radius from the South Pole in stereographic coordinates
    _, max_radius = proj_south_pole(0, lat_boundary)

    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    X, Y = np.meshgrid(x, y)

    lon, lat = pyproj.transform(proj_south_pole, proj_geographic, X, Y)

    output_grid = xr.Dataset(
        {
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
        },
        coords={
            "x": (["x"], x),
            "y": (["y"], y),
        }
    )

    return output_grid


def regrid_variable(ds_to_regrid, output_grid):
    print("Downloading data and regridding (this step may take a while)... ", end="")
    start_time = time.time()
    weight_file = f'{config.DATA_DIRECTORY}/cesm_lens/grids/cesm_to_sps_bilinear_regridding_weights.nc'

    if os.path.exists(weight_file):
        regridder = xe.Regridder(ds_to_regrid, output_grid, 'bilinear', weights=weight_file, 
                                ignore_degenerate=True, reuse_weights=True, periodic=True)
    else:
        regridder = xe.Regridder(ds_to_regrid, output_grid, 'bilinear', filename=weight_file, 
                                ignore_degenerate=True, reuse_weights=False, periodic=True)

    ds_regridded = regridder(ds_to_regrid)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"done! (Time taken: {elapsed_time:.2f} seconds)")
    
    return ds_regridded

def make_save_directories(download_settings=download_settings, parent_dir="/scratch/users/yucli/"):
    """
    Constructs the fire directory structure for saving files

    Args:
    download_settings (dict): dictionary containing the download settings
    parent_dir (str): parent directory to save the data

    Returns:
    variable_dirs (dict): dictionary containing the paths to save the variables
    """
    variable_dirs = {}

    for variable in download_settings["vars"]:
        path_name = os.path.join(parent_dir, "cesm_data", var_args[variable]["save_name"])
        os.makedirs(path_name, exist_ok=True)
        variable_dirs[variable] = path_name

    return variable_dirs


def check_if_downloaded(variable_dirs, download_settings=download_settings, parent_dir="/scratch/users/yucli/"):
    """
    Given download_settings, find all files that have already been downloaded. 
    Return a modified download_settings only for files that have not been downloaded.

    Args:
    variable_dirs (dict): dictionary containing the paths to save the variables
    download_settings (dict): dictionary containing the download settings
    parent_dir (str): parent directory to save the data

    Returns:
    download_settings_updated (dict): dictionary containing the updated download settings
    """
    download_settings_updated = download_settings

    for variable in download_settings["vars"]:
        path_name = os.path.join(parent_dir, "cesm_data", var_args[variable]["save_name"])
        files = os.listdir(path_name)
        
        # get the specific member ids that have been downloaded
        downloaded_members = []
        for f in files:
            downloaded_members.append(int(f.split("_")[-1].split(".")[0]))

        # remove the downloaded member ids from the download_settings
        if download_settings["member_id"] == "all":
            download_settings_updated["member_id"] = [i for i in range(100) if i not in downloaded_members]
        elif type(download_settings["member_id"]) == list:
            download_settings_updated["member_id"] = [i for i in download_settings["member_id"] if i not in downloaded_members]
        else:
            raise TypeError("download_settings['member_id'] needs to be a list or 'all'")
        
        if len(download_settings_updated["member_id"]) == 0:
            download_settings_updated["vars"].remove(variable)
            print(f"Variable {variable} has already been downloaded for all ensemble members. Skipping...")
    return download_settings_updated
    
def main():
    variable_dirs = make_save_directories(download_settings=download_settings, parent_dir=download_settings["save_directory"])

    output_grid = generate_sps_grid()

    # check what has already been downloaded and update download settings 
    download_settings = check_if_downloaded(variable_dirs, download_settings=download_settings, parent_dir=download_settings["save_directory"])

    for variable in download_settings["vars"]:
        merged_ds = retrieve_variable_dataset(catalog=CATALOG, variable=variable)
        
        if merged_ds is None: continue

        # download one ensemble member at a time 
        if download_settings["member_id"] == "all": 
            members_to_download = range(merged_ds.member_id.size)
        elif isinstance(download_settings["member_id"], list): 
            members_to_download = download_settings["member_id"]
        else: raise TypeError("download_settings['member_id'] needs to be a list or 'all'")

        for i in members_to_download:
            subset = subset_variable_dataset(merged_ds, variable, member_id=i, chunk=download_settings["chunk"])

            # regrid variable 
            regridded_subset = regrid_variable(subset, output_grid)
            print("saving... ", end="")
            save_path = os.path.join(variable_dirs[variable], f"{var_args[variable]['save_name']}_member_{i}.nc")
            regridded_subset.to_netcdf(save_path)
            regridded_subset.close()
            print("done!")            

if __name__ == "__main__":
    main()