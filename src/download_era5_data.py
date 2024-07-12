"""
Download variable from ERA5. 

First, make sure a directory exists where the data is saved. Once you have created 
the directory, change the path below. 
"""

# Change as needed 
save_directory = '/scratch/groups/earlew/yuchen/ERA5'

import xarray as xr
import cdsapi
import os
import argparse

# Get the variable name to download 
parser = argparse.ArgumentParser()
parser.add_argument('--var')
args = parser.parse_args()

variable = args.var

#######################################################################

years = ['1978', '1979', '1980',
        '1981', '1982', '1983',
        '1984', '1985', '1986',
        '1987', '1988', '1989',
        '1990', '1991', '1992',
        '1993', '1994', '1995',
        '1996', '1997', '1998',
        '1999', '2000', '2001',
        '2002', '2003', '2004',
        '2005', '2006', '2007',
        '2008', '2009', '2010',
        '2011', '2012', '2013',
        '2014', '2015', '2016',
        '2017', '2018', '2019',
        '2020', '2021', '2022',
        '2023', '2024']

months = ['01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12']

variables_dict = {
    '10m_u_component_of_wind': {
        'plevel': None,
        'short_name': 'u10'
    }, 
    '10m_v_component_of_wind': {
        'plevel': None,
        'short_name': 'v10'
    }, 
    '2m_temperature': {
        'plevel': None,
        'short_name': 't2m'
    }, 
    'mean_sea_level_pressure': {
        'plevel': None,
        'short_name': 'msl'
    }, 
    'sea_surface_temperature': {
        'plevel': None,
        'short_name': 'sst'
    }, 
    'surface_net_solar_radiation': {
        'plevel': None,
        'short_name': 'ssr'
    }, 
    'surface_net_thermal_radiation': {
        'plevel': None,
        'short_name': 'str'
    }, 
    'sea_ice_cover': {
        'plevel': None
    }, 
    'land_sea_mask': {
        'plevel': None
    }, 
    'geopotential': {
        'plevel': '500'
    }
}

#######################################################################

def download_era5_variable(var_name, save_path, plevel=None, overwrite=False):
    if os.path.exists(save_path):
        if overwrite: 
            print(f'{save_path} already exists! Overwriting...\n')
        else:
            print(f'{save_path} already exists! Skipping...\n')
            return
    
    retrieval_info = {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': var_name,
        'year': years,
        'month': months,
        'time': '00:00',
        'format': 'netcdf'
    }

    if plevel is not None:
        dataset = 'reanalysis-era5-pressure-levels-monthly-means'
        retrieval_info['pressure_level'] = plevel
        print(f'Downloading {var_name} at pressure lev {plevel} and saving to {save_path}')
    else: 
        dataset = 'reanalysis-era5-single-levels-monthly-means'
        print(f'Downloading {var_name} and saving to {save_path}')

    cds_client.retrieve(dataset, retrieval_info, save_path)

cds_client = cdsapi.Client()

plevel = variables_dict[variable]["plevel"]
if plevel is not None:
    save_path = os.path.join(save_directory, f'{variable}_{plevel}hPa.nc')
else: 
    save_path = os.path.join(save_directory, f'{variable}.nc')

download_era5_variable(variable, save_path, plevel=variables_dict[variable]["plevel"])