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
    
    "member_id": [0], 

    "save_directory": "/scratch/users/yucli/"
}

var_args = {
    "ICEFRAC":  {"p_index": None}, # saved in the atm component
    "TEMP":     {"p_index": 0, "lat_slice": slice(0, 93)}, # note, lat_slice is index 
    "FLNS":     {"p_index": None}, 
    "FSNS":     {"p_index": None}, 
    "PSL":      {"p_index": None}, 
    "Z3":       {"p_index": 20}, 
    "U":        {"p_index": -1}, 
    "V":        {"p_index": -1}, # apparently, there is no V! 
    "hi":       {"p_index": None, "lat_slice": slice(0, 93)} # for some reason thickness is in ice component, not atm
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


TEMP_SAVE_DIR = "/scratch/users/yucli/test/"

output_grid = generate_sps_grid()

for variable in download_settings["vars"]:
    merged_ds = retrieve_variable_dataset(catalog=CATALOG, variable=variable)
    
    if merged_ds is None: continue

    # download one ensemble member at a time 
    if download_settings["member_id"] == "all": 
        n_members = merged_ds.member_id.size
    elif type(download_settings["member_id"]) == list: 
        n_members = len(download_settings["member_id"])
    else: raise TypeError("download_settings['member_id'] needs to be a list or 'all'")

    for i in range(n_members):
        subset = subset_variable_dataset(merged_ds, variable, member_id=i, chunk=download_settings["chunk"])

        # TEST: load in subset
        subset = subset.load()

        print("loaded")
        print(subset)
        raise NotImplementedError()

        # regrid variable 
        regridded_subset = regrid_variable(subset, output_grid)
        print("saving... ", end="")
        regridded_subset.to_netcdf(os.path.join(TEMP_SAVE_DIR, f"{variable}_memberid{i}.nc"))
        print("done!")

        

        

