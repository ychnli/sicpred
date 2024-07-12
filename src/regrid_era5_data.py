"""
Regrids variables to the NSIDC SPS (South Polar Stereographic) grid. 
"""

import xarray as xr
import xesmf as xe 
import config 
import argparse
import os

# Get the variable name to regrid 
parser = argparse.ArgumentParser()
parser.add_argument('--var')
args = parser.parse_args()

variable = args.var

"""
Params:
    var_name (str) name of variable 
    data_path (str) path where the variable 
"""
def regrid_var(var_name, output_grid=config.SPS_GRID, grid_name='SPS', overwrite=False):
    if var_name == 'geopotential':
        path = f'{config.DATA_DIRECTORY}/ERA5/{var_name}_500hPa.nc'
    else: 
        path = f'{config.DATA_DIRECTORY}/ERA5/{var_name}.nc'

    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_path} does not exist")
    
    if var_name == 'geopotential':
        save_path = f'{config.DATA_DIRECTORY}/ERA5/{var_name}_500hPa_{grid_name}.nc'
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
    
    ds_regridded.to_netcdf(save_path)
    print('Done!')

regrid_var(variable, overwrite=False)
