import xarray as xr
import pandas as pd
import numpy as np
import scipy
from time import time
import os
import config
import util 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def linear_trend(target_month, save_path, linear_years="all", verbose=1):
    """
    Computes the gridcell-wise linear trend forecast for target_month using linear_years of historical data.

    Params:
        target_month:   pd.datetime object for the month to predict
        save_path:      string 
        linear_years:   int or "all" for all years. If int, must be greater than 1 and less than
                        total years between the target year and the beginning of observations (1987)
    
    Returns:
        prediction:     output map as a xarray Dataset 

    """

    save_name = os.path.join(save_path, f"{target_month.year}-{target_month.month:02}_{linear_years}_years_linear_forecast.nc")

    # First check if it exists or not
    if os.path.exists(save_name):
        if verbose >= 1: print(f"Found pre-existing file with path {save_name}. Skipping...")
        return 

    INITIAL_YEAR = pd.to_datetime('1978-11')

    if verbose >= 1: print(f"Computing linear trend forecast for {target_month} using {linear_years} years")
    nsidc_sic = xr.open_dataset(f'{config.DATA_DIRECTORY}/NSIDC/seaice_conc_monthly_all.nc')

    # Select subset of dataset of only the target month 
    subset_target_months = nsidc_sic.siconc.sel(time=nsidc_sic.time.dt.month == target_month.month)

    if linear_years == "all":
        subset_target_months = subset_target_months.sel(time=slice(INITIAL_YEAR, target_month - pd.DateOffset(years=1)))
    else: 
        if type(linear_years) != int:
            raise TypeError("Expected linear_years to be an integer")
        if linear_years == 1: 
            raise ValueError("Cannot make linear trend prediction with only 1 year!")
        if target_month.year - INITIAL_YEAR.year <= linear_years:
            raise ValueError(f"{linear_years} exceeds the total number of years in the timeseries")

        subset_target_months = subset_target_months.sel(time=slice(INITIAL_YEAR - pd.DateOffset(years=linear_years), \
                                                        target_month - pd.DateOffset(years=1)))

    # Mask out land (2.54 and 2.53 flag values)
    land_mask = np.logical_or(nsidc_sic.siconc.isel(time=0) == 2.53, nsidc_sic.siconc.isel(time=0) == 2.54)

    # Mask out open ocean (defined as a gridcell that is always zero)
    all_zeros_mask = np.sum(nsidc_sic.siconc == 0, axis=0) == len(nsidc_sic.time)
    land_and_open_ocean_mask = ~np.logical_or(land_mask.values, all_zeros_mask.values)

    # Compute regression 
    years = subset_target_months.time.dt.year.values
    print(f"Computing linear regression using {linear_years} years...", end=" ")
    reg_result = util.linear_regress_array(years, subset_target_months.values, axis=0, mask=land_and_open_ocean_mask)
    print("done!")

    prediction_npy = reg_result[0] * target_month.year + reg_result[1]
    
    # Correct unphysical predictions 
    prediction_npy[prediction_npy < 0] = 0
    prediction_npy[prediction_npy > 1] = 1

    # Save it as a netCDF file 
    prediction = xr.Dataset(
        data_vars={
            "siconc": (("ygrid", "xgrid"), prediction_npy)
        }, 
        coords={
            "time": target_month,
            "xgrid": nsidc_sic.xgrid.values,
            "ygrid": nsidc_sic.ygrid.values
        },
        attrs={
            "years_used": linear_years
        }
    )

    prediction.to_netcdf(save_name)

    return prediction


def anomaly_persistence():
    return None


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters_factor=1, filter_size=3):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, int(64 * n_filters_factor), filter_size)
        self.encoder2 = self.conv_block(int(64 * n_filters_factor), int(128 * n_filters_factor), filter_size)
        self.encoder3 = self.conv_block(int(128 * n_filters_factor), int(256 * n_filters_factor), filter_size)
        self.bottleneck = self.conv_block(int(256 * n_filters_factor), int(512 * n_filters_factor), filter_size)
        
        self.decoder1 = self.conv_block(int(512 * n_filters_factor) + int(256 * n_filters_factor), int(256 * n_filters_factor), filter_size)
        self.decoder2 = self.conv_block(int(256 * n_filters_factor) + int(128 * n_filters_factor), int(128 * n_filters_factor), filter_size)
        self.decoder3 = self.conv_block(int(128 * n_filters_factor) + int(64 * n_filters_factor), int(64 * n_filters_factor), filter_size)
        
        self.final_conv = nn.Conv2d(int(64 * n_filters_factor), out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def conv_block(self, in_channels, out_channels, filter_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=filter_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=filter_size, padding=filter_size//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        
        dec1 = self.upsample(bottleneck)
        dec1 = torch.cat((enc3, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        dec2 = self.upsample(dec1)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec3 = self.upsample(dec2)
        dec3 = torch.cat((enc1, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        
        return torch.sigmoid(self.final_conv(dec3))
