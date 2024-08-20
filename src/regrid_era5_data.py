"""
Regrids variables to the NSIDC SPS (South Polar Stereographic) grid. 
"""

import xarray as xr
import xesmf as xe 
import argparse
import os

from src import config

# Get the variable name to regrid 
parser = argparse.ArgumentParser()
parser.add_argument('--var')
args = parser.parse_args()

variable = args.var

def write_nc_file(ds, save_path, overwrite, verbose=1):
    """
    Saves xarray Dataset as a netCDF file to save_path. 

    Params:
        ds:         xarray Dataset
        save_path:  (str) valid file path ending in .nc 
        overwrite:  (bool) 
    """

    if type(overwrite) != bool:
        raise TypeError("overwrite needs to be a bool")

    if type(save_path) != str:
        raise TypeError("save_path needs to be a string")

    if save_path[-3:] != ".nc":
        print(f"Attempting to write netCDF file to save_path = {save_path} without .nc suffix; appending .nc to end of filename...")
        save_path += ".nc"

    if os.path.exists(save_path):
        if overwrite:
            temp_path = save_path + '.tmp'
            ds.to_netcdf(temp_path)
            os.replace(temp_path, save_path)
            if verbose == 2: print(f"Overwrote {save_path}")

    else: 
        ds.to_netcdf(save_path)
        if verbose == 2: print(f"Saved to {save_path}")


"""
Regrid using bilinear interpolation to the grid specified by the output grid

Params:
    var_name (str) name of variable 
    data_path (str) path where the variable 
"""
def regrid_var(var_name, output_grid=config.SPS_GRID, grid_name='SPS', overwrite=False):
    plevel = config.era5_variables_dict[var_name]["plevel"]
    if plevel != None:
        path = f'{config.DATA_DIRECTORY}/ERA5/{var_name}_{plevel}hPa.nc'
    else: 
        path = f'{config.DATA_DIRECTORY}/ERA5/{var_name}.nc'

    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_path} does not exist")
    
    if plevel != None:
        save_path = f'{config.DATA_DIRECTORY}/ERA5/{var_name}_{plevel}hPa_{grid_name}.nc'
    else: 
        save_path = f'{config.DATA_DIRECTORY}/ERA5/{var_name}_{grid_name}.nc'

    print(f'Regridding {var_name}')
    if os.path.exists(save_path):
        print(f'Found a pre-existing regridded file for {var_name}...', end=' ')
        if not overwrite:
            print('skipping') 
            return 
        else:
            print('overwriting')

    ds_to_regrid = xr.open_dataset(path)

    # rename lat/lon variables for consistency with xESMF conventions
    ds_to_regrid = ds_to_regrid.rename({"latitude": "lat", "longitude": "lon"})
    output_grid = output_grid.rename({"latitude": "lat", "longitude": "lon"})

    # file to save the weights
    weight_file = f'{config.DATA_DIRECTORY}/ERA5/ERA5_to_{grid_name}_bilinear_regridding_weights.nc'

    if os.path.exists(weight_file):
        regridder = xe.Regridder(ds_to_regrid, output_grid, 'bilinear', weights=weight_file, 
                                ignore_degenerate=True, reuse_weights=True, periodic=True)
    else:
        regridder = xe.Regridder(ds_to_regrid, output_grid, 'bilinear', filename=weight_file, 
                                ignore_degenerate=True, reuse_weights=False, periodic=True)

    ds_regridded = regridder(ds_to_regrid)
    print(f'Finished regridding {var_name}! saving...', end='')
    
    write_nc_file(ds_regridded, save_path, overwrite)

    print('Done!')

regrid_var(variable, overwrite=False)
