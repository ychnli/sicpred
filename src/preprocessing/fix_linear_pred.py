import xarray as xr
import os 

directory = "/oak/stanford/groups/earlew/yuchen/sicpred/linear_forecasts"
linear_predictions = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nc')])

for file_path in linear_predictions:
    ds = xr.open_dataset(file_path)
    siconc = ds['siconc']
    siconc = siconc.where(siconc <= 1, 1)
    ds['siconc'] = siconc

    temp_file_path = file_path + '.tmp'
    ds.to_netcdf(temp_file_path)
    ds.close()

    os.replace(temp_file_path, file_path)
    print(f"Successfully processed {file_path}")