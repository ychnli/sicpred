import pandas as pd

################################ description ################################
DATE = ""
EXPERIMENT_NAME = ""
NOTES = ""

################################ data configs ################################

MAX_LEAD_MONTHS = 6

DATA_CONFIG_NAME = "icefrac+temp_time"

"""
data_split_settings should be a dict with keys split_by, train, val, and test
valid values for split_by is "time" or "ensemble_member" 

If split_by = "time", the train/val/test values should be pd.date_range and 
you should specify the member_ids to use 

If split_by = "ensemble_member", the train/val/test values should be member_ids
and you should specify the time range to use 
"""

DATA_SPLIT_SETTINGS = {
    "name": "",
    "split_by": "time",
    "train": pd.date_range("1851-01", "1979-12", freq="MS"),
    "val": pd.date_range("1980-01", "1994-12", freq="MS"),
    "test": pd.date_range("1995-01", "2013-12", freq="MS"),
    "time_range": None,
    "member_ids": [""]
}


# Renamed variable names 
VAR_NAMES = ["icefrac", "icethick", "temp", "geopotential", "psl", "lw_flux", "sw_flux", "ua"]


INPUT_CONFIG = {
    'icefrac': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 
        'divide_by_stdev': False, 'auxiliary': False
    }, 
    'icethick': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 
        'divide_by_stdev': False, 'auxiliary': False
    }, 
    'temp': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'geopotential': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 6, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'psl': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 6, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'lw_flux': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'sw_flux': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'ua': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 
        'divide_by_stdev': True, 'auxiliary': False
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

TARGET_CONFIG = {
    "predict_anom": True
}


# model configurations
MODEL 

# training configurations 
LEARNING_RATE
BATCH_SIZE
NUM_EPOCHS