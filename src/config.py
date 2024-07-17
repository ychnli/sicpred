import os
import xarray as xr
import pandas as pd

DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'

SPS_GRID = xr.open_dataset(os.path.join(DATA_DIRECTORY, 'NSIDC/sps_grid.nc'))

era5_variables_dict = {
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
    'geopotential': {
        'plevel': '500',
        'short_name': 'z'
    }
}

TRAIN_MONTHS = pd.date_range(start='1981-01-01', end='2014-12-01', freq='MS')

VAL_MONTHS = pd.date_range(start='2015-01-01', end='2018-12-01', freq='MS')

TEST_MONTHS = pd.date_range(start='2019-01-01', end='2024-06-01', freq='MS')