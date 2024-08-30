import os
import xarray as xr
import pandas as pd

DATA_DIRECTORY = '/oak/stanford/groups/earlew/yuchen'

SPS_GRID = xr.open_dataset(os.path.join(DATA_DIRECTORY, 'NSIDC/sps_grid.nc'))

era5_variables_dict = {
    '10m_u_component_of_wind': {'plevel': None, 'short_name': 'u10'}, 
    '10m_v_component_of_wind': {'plevel': None, 'short_name': 'v10'}, 
    '2m_temperature': {'plevel': None, 'short_name': 't2m'}, 
    'mean_sea_level_pressure': {'plevel': None, 'short_name': 'msl'}, 
    'sea_surface_temperature': {'plevel': None, 'short_name': 'sst'}, 
    'surface_net_solar_radiation': {'plevel': None, 'short_name': 'ssr'}, 
    'surface_net_thermal_radiation': {'plevel': None, 'short_name': 'str'}, 
    'sea_ice_cover': {'plevel': None}, 
    'geopotential': {'plevel': '500', 'short_name': 'z'},
    'u_component_of_wind': {'plevel': '10', 'short_name': 'u'}
}


# Model configuration parrameters
TRAIN_MONTHS = pd.date_range(start='1981-01-01', end='2014-12-01', freq='MS')
VAL_MONTHS = pd.date_range(start='2015-01-01', end='2018-12-01', freq='MS')
TEST_MONTHS = pd.date_range(start='2019-01-01', end='2024-01-01', freq='MS')

# how many months to predict into the future
max_month_lead_time = 6 

######################################################################
# Input configurations 
######################################################################

input_config_siconly = {
    'siconc': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': False, 'land_mask': True, 'lag': 12
    },
    'siconc_linear_forecast': {
        'plevel': None, 'short_name': 'siconc', 'include': False,
        'anom': False, 'land_mask': True, 'lag': None
    },
    '10m_u_component_of_wind': {
        'plevel': None, 'short_name': 'u10', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '10m_v_component_of_wind': {
        'plevel': None, 'short_name': 'v10', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '2m_temperature': {
        'plevel': None, 'short_name': 't2m', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'mean_sea_level_pressure': {
        'plevel': None, 'short_name': 'msl', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'sea_surface_temperature': {
        'plevel': None, 'short_name': 'sst', 'include': False,
        'anom': True, 'land_mask': True, 'lag': 9
    }, 
    'surface_net_solar_radiation': {
        'plevel': None, 'short_name': 'ssr', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'surface_net_thermal_radiation': {
        'plevel': None, 'short_name': 'str', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'geopotential': {
        'plevel': '500', 'short_name': 'z', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'u_component_of_wind': {
        'plevel': '10', 'short_name': 'u', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'cosine_of_init_month': {
        'include': True, 'land_mask': False
    },
    'sine_of_init_month': {
        'include': True,'land_mask': False
    }
}

input_config_simple = {
    'siconc': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': False, 'land_mask': True, 'lag': 12
    },
    'siconc_linear_forecast': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': False, 'land_mask': True, 'lag': None
    },
    '10m_u_component_of_wind': {
        'plevel': None, 'short_name': 'u10', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '10m_v_component_of_wind': {
        'plevel': None, 'short_name': 'v10', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '2m_temperature': {
        'plevel': None, 'short_name': 't2m', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'mean_sea_level_pressure': {
        'plevel': None, 'short_name': 'msl', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'sea_surface_temperature': {
        'plevel': None, 'short_name': 'sst', 'include': True,
        'anom': True, 'land_mask': True, 'lag': 3
    }, 
    'surface_net_solar_radiation': {
        'plevel': None, 'short_name': 'ssr', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'surface_net_thermal_radiation': {
        'plevel': None, 'short_name': 'str', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'geopotential': {
        'plevel': '500', 'short_name': 'z', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'u_component_of_wind': {
        'plevel': '10', 'short_name': 'u', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'cosine_of_init_month': {
        'include': True, 'land_mask': False
    },
    'sine_of_init_month': {
        'include': True,'land_mask': False
    }
}

input_config_simple_sicanom = {
    'siconc': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': True, 'div_stdev': False, 'land_mask': True, 'lag': 12
    },
    'siconc_linear_forecast': {
        'plevel': None, 'short_name': 'siconc', 'include': False,
        'anom': False, 'land_mask': True, 'lag': None
    },
    '10m_u_component_of_wind': {
        'plevel': None, 'short_name': 'u10', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '10m_v_component_of_wind': {
        'plevel': None, 'short_name': 'v10', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '2m_temperature': {
        'plevel': None, 'short_name': 't2m', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'mean_sea_level_pressure': {
        'plevel': None, 'short_name': 'msl', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'sea_surface_temperature': {
        'plevel': None, 'short_name': 'sst', 'include': True,
        'anom': True, 'land_mask': True, 'lag': 3
    }, 
    'surface_net_solar_radiation': {
        'plevel': None, 'short_name': 'ssr', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'surface_net_thermal_radiation': {
        'plevel': None, 'short_name': 'str', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'geopotential': {
        'plevel': '500', 'short_name': 'z', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'u_component_of_wind': {
        'plevel': '10', 'short_name': 'u', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'cosine_of_init_month': {
        'include': True, 'land_mask': False
    },
    'sine_of_init_month': {
        'include': True,'land_mask': False
    }
}

input_config_all = {
    'siconc': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': False, 'land_mask': True, 'lag': 12
    },
    'siconc_linear_forecast': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': False, 'land_mask': True, 'lag': None
    },
    '10m_u_component_of_wind': {
        'plevel': None, 'short_name': 'u10', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '10m_v_component_of_wind': {
        'plevel': None, 'short_name': 'v10', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '2m_temperature': {
        'plevel': None, 'short_name': 't2m', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'mean_sea_level_pressure': {
        'plevel': None, 'short_name': 'msl', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'sea_surface_temperature': {
        'plevel': None, 'short_name': 'sst', 'include': True,
        'anom': True, 'land_mask': True, 'lag': 3
    }, 
    'surface_net_solar_radiation': {
        'plevel': None, 'short_name': 'ssr', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'surface_net_thermal_radiation': {
        'plevel': None, 'short_name': 'str', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'geopotential': {
        'plevel': '500', 'short_name': 'z', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'u_component_of_wind': {
        'plevel': '10', 'short_name': 'u', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'cosine_of_init_month': {
        'include': True, 'land_mask': False
    },
    'sine_of_init_month': {
        'include': True,'land_mask': False
    }
}

input_config_all_sicanom = {
    'siconc': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': True, 'div_stdev': False, 'land_mask': True, 'lag': 12
    },
    'siconc_linear_forecast': {
        'plevel': None, 'short_name': 'siconc', 'include': False,
        'anom': False, 'land_mask': True, 'lag': None
    },
    '10m_u_component_of_wind': {
        'plevel': None, 'short_name': 'u10', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '10m_v_component_of_wind': {
        'plevel': None, 'short_name': 'v10', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '2m_temperature': {
        'plevel': None, 'short_name': 't2m', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'mean_sea_level_pressure': {
        'plevel': None, 'short_name': 'msl', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'sea_surface_temperature': {
        'plevel': None, 'short_name': 'sst', 'include': True,
        'anom': True, 'land_mask': True, 'lag': 3
    }, 
    'surface_net_solar_radiation': {
        'plevel': None, 'short_name': 'ssr', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'surface_net_thermal_radiation': {
        'plevel': None, 'short_name': 'str', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'geopotential': {
        'plevel': '500', 'short_name': 'z', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'u_component_of_wind': {
        'plevel': '10', 'short_name': 'u', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'cosine_of_init_month': {
        'include': True, 'land_mask': False
    },
    'sine_of_init_month': {
        'include': True,'land_mask': False
    }
}

input_config_all_sicanom_modified = {
    'siconc': {
        'plevel': None, 'short_name': 'siconc', 'include': True,
        'anom': True, 'div_stdev': False, 'land_mask': True, 'lag': 12
    },
    'siconc_linear_forecast': {
        'plevel': None, 'short_name': 'siconc', 'include': False,
        'anom': False, 'land_mask': True, 'lag': None
    },
    '10m_u_component_of_wind': {
        'plevel': None, 'short_name': 'u10', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '10m_v_component_of_wind': {
        'plevel': None, 'short_name': 'v10', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 1
    }, 
    '2m_temperature': {
        'plevel': None, 'short_name': 't2m', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'mean_sea_level_pressure': {
        'plevel': None, 'short_name': 'msl', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'sea_surface_temperature': {
        'plevel': None, 'short_name': 'sst', 'include': False,
        'anom': True, 'land_mask': True, 'lag': 3
    }, 
    'surface_net_solar_radiation': {
        'plevel': None, 'short_name': 'ssr', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'surface_net_thermal_radiation': {
        'plevel': None, 'short_name': 'str', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    }, 
    'geopotential': {
        'plevel': '500', 'short_name': 'z', 'include': True,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'u_component_of_wind': {
        'plevel': '10', 'short_name': 'u', 'include': False,
        'anom': True, 'land_mask': False, 'lag': 3
    },
    'cosine_of_init_month': {
        'include': True, 'land_mask': False
    },
    'sine_of_init_month': {
        'include': True,'land_mask': False
    }
}



input_configs = {
    'sea_ice_only': input_config_siconly,
    'simple': input_config_simple,
    'simple_sicanom': input_config_simple_sicanom,
    'all': input_config_all,
    'all_sicanom': input_config_all_sicanom,
    'all_sicanom_modified': input_config_all_sicanom_modified
}

######################################################################
# Target configurations 
######################################################################

target_config_regression = {
    'predict_siconc_anom': False,
    'mode': 'regression'
}

target_config_anom_regression = {
    'predict_siconc_anom': True,
    'mode': 'regression'
}

target_configs = {
    'regression': target_config_regression,
    'anom_regression': target_config_anom_regression,
}