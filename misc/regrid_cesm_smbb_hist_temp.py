import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import time
import os
import pyproj
import sys

from src.download_cesm_data import generate_sps_grid 
from src.download_cesm_data import regrid_variable 

DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'
CESM_OCEAN_GRID = xr.open_dataset(f"{DATA_DIRECTORY}/cesm_lens/grids/ocean_grid.nc")
RAW_DATA_DIRECTORY = '/scratch/users/yucli/cesm_data'


def get_member_ids(dir="/scratch/users/yucli/cesm_temp_raw"):
    """
    Generates a list of unique member_ids for downloaded raw thetao files

    Returns: 
        (list)  list of netCDF files in dir 
        (list)  items of format ####-### (init year-realization number)
    """
    
    files = sorted(os.listdir(dir))

    # remove all non netcdf files. I'm not sure why the bash script doesn't get removed 
    # by the for loop..?
    files.remove('wget-ucar.cgd.cesm2le.ocn.proc.monthly_ave.TEMP.AllFiles-20241128T1755.sh')
    for f in files: 
        if f[-3:] != ".nc": files.remove(f)

    # get the member id tag from the filename 
    # the format is ####-### (init year - realization)
    member_ids = [] 

    for f in files:
        member_id = f.split("-")[1][0:8]
        member_ids.append(member_id)

    member_ids = np.unique(member_ids)

    return files, member_ids

def main():
    files, member_ids = get_member_ids()

    output_grid = generate_sps_grid()

    # get each member separately 
    for i,member_id in enumerate(member_ids):
        realization = int(member_id.split(".")[1])
        init_year = int(member_id.split(".")[0])
        member_id_save_name = f"r{realization}i{init_year}p1f2"
        save_path = f"/scratch/users/yucli/cesm_temp_hist_regridded/temp_member_{member_id_save_name}.nc"
        if os.path.exists(save_path):
            print(f"already found existing {save_path}, skipping")
            continue

        files_subset = []
        for f in files:
            if member_id in f: files_subset.append(os.path.join("/scratch/users/yucli/cesm_temp_raw", f))
        
        ds = xr.open_mfdataset(files_subset)

        subset = ds.TEMP.isel(z_t=0, nlat=slice(0, 93))

        lat = CESM_OCEAN_GRID.lat.sel(nlat=slice(0, 93)).data
        lon = CESM_OCEAN_GRID.lon.sel(nlat=slice(0, 93)).data

        subset = subset.assign_coords(lat=(["nlat", "nlon"], lat), lon=(["nlat", "nlon"], lon))
        subset = subset.to_dataset(name="temp")

        print(f"Regridding member_id: {member_id_save_name}...")
        regridded_subset = regrid_variable(subset, "ocn", output_grid)
        regridded_subset = regridded_subset.assign_coords(member_id=member_id_save_name)

        # save 
        regridded_subset.to_netcdf(save_path)

if __name__ == "__main__":
    main()
