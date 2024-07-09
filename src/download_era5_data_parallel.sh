mkdir -p logs/era5_download_logs/

python3 download_era5_data.py --var 10m_u_component_of_wind > logs/era5_download_logs/10m_u_component_of_wind.txt 2>&1 &
python3 download_era5_data.py --var 10m_v_component_of_wind > logs/era5_download_logs/10m_v_component_of_wind.txt 2>&1 &
python3 download_era5_data.py --var 2m_temperature > logs/era5_download_logs/2m_temperature.txt 2>&1 &
python3 download_era5_data.py --var mean_sea_level_pressure > logs/era5_download_logs/mean_sea_level_pressure.txt 2>&1 &
python3 download_era5_data.py --var sea_surface_temperature > logs/era5_download_logs/sea_surface_temperature.txt 2>&1 &
python3 download_era5_data.py --var surface_net_solar_radiation > logs/era5_download_logs/surface_net_solar_radiation.txt 2>&1 &
python3 download_era5_data.py --var surface_net_thermal_radiation > logs/era5_download_logs/surface_net_thermal_radiation.txt 2>&1 &