import xarray as xr
import os

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