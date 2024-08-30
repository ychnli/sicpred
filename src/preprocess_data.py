import xarray as xr
import numpy as np
import os
from src import config
from src import util
import time
import pandas as pd


# concatenate monthly sea ice concentration files 
util.concatenate_nsidc()

# remove missing data in 1987 and 1988 
util.remove_missing_data_nsidc(verbose=2) 

# generate land and ice masks
util.generate_masks(verbose=2)
util.apply_land_mask_to_nsidc_siconc(verbose=2) 

# calculate linear trend forecast. Execute in parallel for efficiency
util.compute_linear_forecast()

# concatenate linear trend forecasts
util.concatenate_linear_trend(verbose=2)

# remove the expver dimension from the ERA5 data to make it easier to deal with
util.remove_expver_from_era5(verbose=2)

# normalize variables and save 
util.normalize_data(verbose=2)

# calculate sea ice anomaly (no divide by stdev)
util.calculate_siconc_anom(verbose=2)

# calculate sea ice climatology over training dataset
util.calculate_climatological_siconc_over_train(overwrite=True)

# generate data pairs
util.prep_prediction_samples('all_sicanom', 'anom_regression', verbose=2, overwrite=False)
util.prep_prediction_samples('all_sicanom_modified', 'anom_regression', verbose=2, overwrite=False)