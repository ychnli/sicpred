import pandas as pd 

DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'

RAW_DATA_DIRECTORY = '/scratch/users/yucli/cesm_data'

MODEL_DATA_DIRECTORY = '/scratch/users/yucli/model-ready_cesm_data'

# Renamed variable names 
VAR_NAMES = ["icefrac", "icethick", "temp", "geopotential", "psl", "lw_flux", "sw_flux", "ua"]

MAX_LEAD_MONTHS = 6

input_config_all = {
    "name": "all",
    
    "start_prediction_months": pd.date_range("1851-01", "2013-12", freq="MS"),

    'icefrac': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 'divide_by_stdev': False, 'auxiliary': False
    }, 
    'icethick': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 'divide_by_stdev': False, 'auxiliary': False
    }, 
    'temp': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 'divide_by_stdev': True, 'auxiliary': False
    }, 
    'geopotential': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 6, 'divide_by_stdev': True, 'auxiliary': False
    }, 
    'psl': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 6, 'divide_by_stdev': True, 'auxiliary': False
    }, 
    'lw_flux': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 'divide_by_stdev': True, 'auxiliary': False
    }, 
    'sw_flux': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 'divide_by_stdev': True, 'auxiliary': False
    }, 
    'ua': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 'divide_by_stdev': True, 'auxiliary': False
    }, 
    'cosine_of_init_month': {
        'include': True, 'auxiliary': True
    }, 
    'sine_of_init_month': {
        'include': True, 'auxiliary': True
    },
    'land_mask': {
        'include': True, 'auxiliary': True
    }
}

target_config = {
    "predict_anom": True
}