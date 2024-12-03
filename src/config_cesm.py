DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'

RAW_DATA_DIRECTORY = '/scratch/users/yucli/cesm_data'

# Renamed variable names 
VAR_NAMES = ["icefrac", "icethick", "temp", "geopotential", "psl", "lw_flux", "sw_flux", "ua"]

input_config_all = {
    "name": "all",
    
    "start_prediction_months": start_prediction_months,

    'icefrac': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 'divide_by_stdev': False
    }, 
    'icethick': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 'divide_by_stdev': False
    }, 
    'temp': {
        'include': True, 'anom': True, 'land_mask': True, 'lag': 12, 'divide_by_stdev': True
    }, 
    'geopotential': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 6, 'divide_by_stdev': True
    }, 
    'psl': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 6, 'divide_by_stdev': True
    }, 
    'lw_flux': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 'divide_by_stdev': True
    }, 
    'sw_flux': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 'divide_by_stdev': True
    }, 
    'ua': {
        'include': True, 'anom': True, 'land_mask': False, 'lag': 3, 'divide_by_stdev': True
    }, 
    'cosine_of_init_month': {
        'include': True
    }, 
    'sine_of_init_month': {
        'include': True
    }
}