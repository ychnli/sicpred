import os
import xarray as xr

DATA_DIRECTORY = '/scratch/groups/earlew/yuchen/'

SPS_GRID = xr.open_dataset(os.path.join(DATA_DIRECTORY, 'NSIDC/sps_grid.nc'))

era5_dataarray_names = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "sea_surface_temperature": "sst",
    "surface_net_solar_radiation": "ssr",
    "surface_net_thermal_radiation": "str"
}