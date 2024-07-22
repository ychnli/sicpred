import xarray as xr
import numpy as np
import os
import config
import util
from models import linear_trend
#from joblib import Parallel, delayed
import time
import pandas as pd


# concatenate monthly sea ice concentration files 
util.concatenate_nsidc()

# remove missing data in 1987 and 1988 
util.remove_missing_data_nsidc() 

# calculate linear trend forecast. Execute in parallel for efficiency  
months_to_calculate_linear_forecast = pd.date_range(start='1981-01-01', end='2024-06-01', freq='MS')

# Parallel(n_jobs=-1)(delayed(linear_trend)(month, f"{config.DATA_DIRECTORY}/sicpred/linear_forecasts/") \
#     for month in months_to_calculate_linear_forecast)

# concatenate linear trend forecasts
util.concatenate_linear_trend()

# remove the expver dimension from the ERA5 data to make it easier to deal with
util.remove_expver_from_era5()

# normalize variables and save 
util.normalize_data()

# generate data pairs
util.prep_prediction_samples('sea_ice_only', verbose=2)
util.prep_prediction_samples('simple', verbose=2)
util.prep_prediction_samples('all', verbose=2)