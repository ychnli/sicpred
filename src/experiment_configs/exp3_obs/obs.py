"""
This is a template for an experiment configuration file.
"""

import pandas as pd
from src.config_cesm import AVAILABLE_CESM_MEMBERS

################################ description ################################
EXPERIMENT_NAME = "exp2_vol1"
NOTES = "Inputs: same as input2. Data volume: 1 ens member. Detrended data"
DATE = "" # optional 

################################ data configs ################################

MAX_LEAD_MONTHS = 6

DATA_CONFIG_NAME = "seaice_plus_auxiliary_obs"

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

special_test_yrs = (pd.date_range("2014-01", "2014-12", freq="MS")).union(pd.date_range("2017-01", "2017-12", freq="MS"))

DATA_SPLIT_SETTINGS = {
    "name": DATA_CONFIG_NAME, 
    "split_by": "ensemble_member",
    "train": pd.date_range("1979-01", "2011-12", freq="MS"), 
    "val": (pd.date_range("2012-01", "2019-12", freq="MS")).difference(special_test_yrs),
    "test": (pd.date_range("2020-01", "2023-12", freq="MS")).union(special_test_yrs),
    "time_range": None,
    "member_ids": "obs"
}


INPUT_CONFIG = {
    'icefrac': {
        'include': True, 'norm': True, 'land_mask': True, 'lag': 12, 
        'divide_by_stdev': False, 'auxiliary': False, 'use_min_max': False
    }, 
    'sst': {
        'include': False, 'norm': True, 'land_mask': True, 'lag': 6, 
        'divide_by_stdev': False, 'auxiliary': False, 'use_min_max': True
    }, 
    'geopotential': {
        'include': False, 'norm': True, 'land_mask': False, 'lag': 6, 
        'divide_by_stdev': False, 'auxiliary': False, 'use_min_max': True
    }, 
    'psl': {
        'include': False, 'norm': True, 'land_mask': False, 'lag': 6, 
        'divide_by_stdev': False, 'auxiliary': False, 'use_min_max': True
    }, 
    't2m': {
        'include': False, 'norm': True, 'land_mask': False, 'lag': 6, 
        'divide_by_stdev': False, 'auxiliary': False, 'use_min_max': True
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
MODEL_ARGS = {
    "n_channels_factor": 0.5,
}
LOSS_FUNCTION = "MSE" 

############################# training configs ##############################
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 20
CHECKPOINT_INTERVAL = 1
PATIENCE = 3

############################# evaluation configs #############################
CHECKPOINT_TO_EVALUATE = "best"