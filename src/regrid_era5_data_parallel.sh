mkdir -p logs/era5_regrid_logs/

python3 -m src.regrid_era5_data --var 10m_u_component_of_wind > logs/era5_regrid_logs/10m_u_component_of_wind.txt 2>&1 &
python3 -m src.regrid_era5_data --var 10m_v_component_of_wind > logs/era5_regrid_logs/10m_v_component_of_wind.txt 2>&1 &
python3 -m src.regrid_era5_data --var 2m_temperature > logs/era5_regrid_logs/2m_temperature.txt 2>&1 &
python3 -m src.regrid_era5_data --var mean_sea_level_pressure > logs/era5_regrid_logs/mean_sea_level_pressure.txt 2>&1 &
python3 -m src.regrid_era5_data --var sea_surface_temperature > logs/era5_regrid_logs/sea_surface_temperature.txt 2>&1 &
python3 -m src.regrid_era5_data --var surface_net_solar_radiation > logs/era5_regrid_logs/surface_net_solar_radiation.txt 2>&1 &
python3 -m src.regrid_era5_data --var surface_net_thermal_radiation > logs/era5_regrid_logs/surface_net_thermal_radiation.txt 2>&1 &
python3 -m src.regrid_era5_data --var sea_ice_cover > logs/era5_regrid_logs/sea_ice_cover.txt 2>&1 &
python3 -m src.regrid_era5_data --var land_sea_mask > logs/era5_regrid_logs/land_sea_mask.txt 2>&1 &
python3 -m src.regrid_era5_data --var geopotential > logs/era5_regrid_logs/geopotential.txt 2>&1 &
python3 -m src.regrid_era5_data --var u_component_of_wind > logs/era5_regrid_logs/u10_hPa.txt 2>&1 &