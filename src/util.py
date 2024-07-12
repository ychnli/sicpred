import xarray as xr
import numpy as np
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

def normalize_data():
    """ 
    Normalize and save variables  
    """
    return None

def concatenate_nsidc(working_path=None):
    """ 
    Merge the NSIDC data into one netCDF file called seaice_conc_monthly_all 
    """

    if working_path == None:
        working_path = f'{config.DATA_DIRECTORY}/NSIDC'

    if not os.path.exists(working_path):
        raise FileNotFoundError(f"{working_path} does not exist. make sure the NSIDC data is in its own directory")
    
    save_path = f'{working_path}/seaice_conc_monthly_all.nc'
    if os.path.exists(save_path):
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

    ds_concat = xr.open_mfdataset(sorted_files, combine="nested", concat_dim='tdim')
    ds_concat.rename({'cdr_seaice_conc_monthly': 'siconc'})
    print('Concatenating files...', end=' ')
    ds_concat.to_netcdf(save_path)
    print('done!')

##########################################################################################
# Sea ice diagnostics 
##########################################################################################

def calculate_SIA(sector="all"):
    ### TODO

    return None

def calculate_SIE(sector="all"):
    ### TODO 

    return None