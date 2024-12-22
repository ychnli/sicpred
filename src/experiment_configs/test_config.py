"""
This is a template for an experiment configuration file.
"""

import pandas as pd

################################ description ################################
DATE = ""
EXPERIMENT_NAME = ""
NOTES = ""

################################ data configs ################################

MAX_LEAD_MONTHS = 6

DATA_CONFIG_NAME = "simple-inputs_small-dataset"

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
    "split_by": "ensemble_member",
    "train": ["r10i1181p1f1", "r10i1231p1f1", "r10i1251p1f1", "r10i1281p1f1", "r2i1231p1f1"], 
    "val": ["r2i1021p1f1"],
    "test": ["r2i1301p1f1"],
    "time_range": pd.date_range("1851-01", "2013-12", freq="MS"),
    "member_ids": None
}


INPUT_CONFIG = {
    'icefrac': {
        'include': True, 'norm': True, 'land_mask': True, 'lag': 12, 
        'divide_by_stdev': False, 'auxiliary': False
    }, 
    'icethick': {
        'include': False, 'norm': True, 'land_mask': True, 'lag': 12, 
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
        'include': False, 'norm': True, 'land_mask': False, 'lag': 6, 
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
    "predict_anom": True
}


############################### model configs ###############################
MODEL = None

############################# training configs ##############################
LEARNING_RATE = None
BATCH_SIZE = None
NUM_EPOCHS = None
