"""
This is a template for an experiment configuration file.
"""

import pandas as pd

################################ description ################################
EXPERIMENT_NAME = "obs_only"
NOTES = "Train "
DATE = "" # optional 

################################ data configs ################################

MAX_LEAD_MONTHS = 6

DATA_CONFIG_NAME = "simple_plus_psl"

"""
data_split_settings should be a dict with keys split_by, train, val, and test
valid values for split_by is "time" or "ensemble_member" 

If split_by = "time", the train/val/test values should be pd.date_range and 
you should specify the member_ids to use. These are the start prediction dates,
so the inputs will be lagged and the targets will go past the end of the date range
by MAX_LEAD_MONTHS.

If split_by = "ensemble_member", the train/val/test values should be member_ids
and you should specify the time range to use 
"""

DATA_SPLIT_SETTINGS = {
    "name": DATA_CONFIG_NAME, 
    "split_by": "time",
    "train": pd.date_range("1851-01", "2013-12", freq="MS"),
    "val": pd.date_range("1851-01", "2013-12", freq="MS"),
    "test": pd.date_range("1851-01", "2013-12", freq="MS"),
    "time_range": None,
    "member_ids": ["obs"]
}

INPUT_CONFIG = {
    'icefrac': {
        'include': True, 'norm': True, 'land_mask': True, 'lag': 12, 
        'divide_by_stdev': False, 'auxiliary': False
    }, 
    'icethick': {
        'include': False, 'norm': True, 'land_mask': True, 'lag': 6, 
        'divide_by_stdev': False, 'auxiliary': False
    }, 
    'temp': {
        'include': True, 'norm': True, 'land_mask': True, 'lag': 12, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'geopotential': {
        'include': False, 'norm': True, 'land_mask': False, 'lag': 6, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'psl': {
        'include': True, 'norm': True, 'land_mask': False, 'lag': 6, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'lw_flux': {
        'include': False, 'norm': True, 'land_mask': False, 'lag': 3, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'sw_flux': {
        'include': False, 'norm': True, 'land_mask': False, 'lag': 3, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'ua': {
        'include': False, 'norm': True, 'land_mask': False, 'lag': 3, 
        'divide_by_stdev': True, 'auxiliary': False
    }, 
    'cosine_of_init_month': {
        'include': True, 'norm': False, 'auxiliary': True
    }, 
    'sine_of_init_month': {
        'include': True, 'norm': False, 'auxiliary': True
    },
    'land_mask': {
        'include': True, 'norm': False, 'auxiliary': True
    }
}

TARGET_CONFIG = {
    "predict_anom": True,
    "predict_classes": False
}


############################### model configs ###############################
MODEL = "UNetRes3"
LOSS_FUNCTION = "MSE" 
LOSS_FUNCTION_ARGS = {
    "apply_month_weights": True,
    "monthly_weights": {"data_split_settings": DATA_SPLIT_SETTINGS,
                        "use_softmax": True,
                        "T": 2}, # a dict of params for util_cesm.calculate_monthly_weights
    "apply_area_weights": True
}

############################# training configs ##############################
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 8
CHECKPOINT_INTERVAL = 1

############################# evaluation configs #############################
CHECKPOINT_TO_EVALUATE = "epoch_8"


