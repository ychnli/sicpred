"""
This is a template for an experiment configuration file.
"""

import pandas as pd

# hardcoded for now 
avail_ens_members = ['r10i1181p1f1', 'r10i1231p1f1', 'r10i1251p1f1', 'r10i1281p1f1',
       'r10i1301p1f1', 'r1i1001p1f1', 'r1i1231p1f1', 'r1i1251p1f1',
       'r1i1281p1f1', 'r1i1301p1f1', 'r2i1021p1f1', 'r2i1231p1f1',
       'r2i1251p1f1', 'r2i1281p1f1', 'r2i1301p1f1', 'r3i1041p1f1',
       'r3i1231p1f1', 'r3i1251p1f1', 'r3i1281p1f1', 'r3i1301p1f1',
       'r4i1061p1f1', 'r4i1231p1f1', 'r4i1251p1f1', 'r4i1281p1f1',
       'r4i1301p1f1', 'r5i1081p1f1', 'r5i1231p1f1', 'r5i1251p1f1',
       'r5i1281p1f1', 'r5i1301p1f1', 'r6i1101p1f1', 'r6i1231p1f1',
       'r6i1251p1f1', 'r6i1281p1f1', 'r6i1301p1f1', 'r7i1121p1f1',
       'r7i1231p1f1', 'r7i1251p1f1', 'r7i1281p1f1', 'r7i1301p1f1',
       'r8i1141p1f1', 'r8i1231p1f1', 'r8i1251p1f1', 'r8i1281p1f1',
       'r8i1301p1f1', 'r9i1161p1f1', 'r9i1231p1f1', 'r9i1251p1f1',
       'r9i1281p1f1', 'r9i1301p1f1', 'r11i1231p1f2', 'r11i1251p1f2',
       'r11i1281p1f2', 'r11i1301p1f2', 'r12i1231p1f2', 'r12i1251p1f2',
       'r12i1281p1f2', 'r12i1301p1f2', 'r13i1231p1f2', 'r13i1251p1f2',
       'r13i1281p1f2', 'r13i1301p1f2', 'r14i1231p1f2', 'r14i1251p1f2',
       'r14i1281p1f2', 'r14i1301p1f2', 'r15i1231p1f2', 'r15i1251p1f2',
       'r15i1281p1f2', 'r15i1301p1f2', 'r16i1231p1f2', 'r16i1251p1f2',
       'r16i1281p1f2', 'r16i1301p1f2', 'r17i1231p1f2', 'r17i1251p1f2',
       'r17i1281p1f2', 'r17i1301p1f2', 'r18i1231p1f2', 'r18i1251p1f2',
       'r18i1281p1f2', 'r18i1301p1f2', 'r19i1231p1f2', 'r19i1251p1f2',
       'r19i1281p1f2', 'r19i1301p1f2', 'r20i1231p1f2', 'r20i1251p1f2',
       'r20i1281p1f2', 'r20i1301p1f2']

################################ description ################################
EXPERIMENT_NAME = "seaice-only"
NOTES = "Inputs are previous 12 months of sea ice"
DATE = "" # optional 

################################ data configs ################################

MAX_LEAD_MONTHS = 6

DATA_CONFIG_NAME = "seaice-only"

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
    "train": avail_ens_members[0:8], 
    "val": avail_ens_members[8:10],
    "test": avail_ens_members[10:12],
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
        'include': False, 'norm': True, 'land_mask': True, 'lag': 12, 
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
        'include': False, 'norm': False, 'auxiliary': True
    }, 
    'sine_of_init_month': {
        'include': False, 'norm': False, 'auxiliary': True
    },
    'land_mask': {
        'include': False, 'norm': False, 'auxiliary': True
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
NUM_EPOCHS = 10
CHECKPOINT_INTERVAL = 1

############################# evaluation configs #############################
CHECKPOINT_TO_EVALUATE = "epoch_5"

