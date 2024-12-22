import xarray as xr
import pandas as pd
import numpy as np
import scipy
from time import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src import config
from src.utils import util_era5 


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
    reg_result = util_era5.linear_regress_array(years, subset_target_months.values, axis=0, mask=land_and_open_ocean_mask)
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


def anomaly_persistence(start_prediction_month, predict_anomalies=False):
    """
    The anomaly persistence baseline model 
    """

    siconc_anom = xr.open_dataset(f"{DATA_DIRECTORY}/sicpred/normalized_inputs/siconc_anom.nc").siconc
    siconc_clim = xr.open_dataset(f"{DATA_DIRECTORY}/NSIDC/siconc_clim.nc").siconc
    land_mask = xr.open_dataset(f"{DATA_DIRECTORY}/NSIDC/land_mask.nc").mask
    siconc_anom *= ~land_mask

    anomaly_to_persist = siconc_anom.sel(time=start_prediction_month - pd.DateOffset(months=1)) 
    months_to_predict_clim = siconc_clim.sel(time=pd.date_range(start_prediction_month, \
        start_prediction_month+pd.DateOffset(months=5), freq='MS'))

    prediction = months_to_predict_clim + anomaly_to_persist 

    # clip unphysical values outside of [0, 1] 
    prediction = prediction.where(prediction > 1, 1)
    prediction = prediction.where(prediction < 0, 0)

    if predict_anomalies:
        prediction = prediction - months_to_predict_clim

    return prediction


class UNetRes3(nn.Module):
    """
    Builds a UNet of resolution 3 (nomenclature from Williams et al. 2023)
    The resolution is defined as the number of encoder/decoder blocks. The
    number at the end of encoder and decoder blocks denote their depth in 
    the network (thus we have, E1 -> E2 -> E3 -> B -> D3 -> D2 -> D1) where
    B is the bottleneck block
    """

    def __init__(self, in_channels, out_channels, predict_anomalies, 
                spatial_shape=(80, 80), 
                n_channels_factor=1, 
                filter_size=3, 
                clip_near_zero_values=True, 
                epsilon=0.01):

        super(UNetRes3, self).__init__()
        self.clip_near_zero_values = clip_near_zero_values
        self.epsilon = epsilon 

        self.encoder1 = self.conv_block(in_channels, int(64 * n_channels_factor), filter_size)
        self.encoder2 = self.conv_block(int(64 * n_channels_factor), int(128 * n_channels_factor), filter_size)
        self.encoder3 = self.conv_block(int(128 * n_channels_factor), int(256 * n_channels_factor), filter_size)

        self.bottleneck = self.conv_block(int(256 * n_channels_factor), int(512 * n_channels_factor), filter_size)
        
        self.decoder3_conv = self.conv(int(512 * n_channels_factor), int(256 * n_channels_factor), filter_size)
        self.decoder3_conv_block = self.conv_block(2 * int(256 * n_channels_factor), int(256 * n_channels_factor), filter_size)

        self.decoder2_conv = self.conv(int(256 * n_channels_factor), int(128 * n_channels_factor), filter_size)
        self.decoder2_conv_block = self.conv_block(2 * int(128 * n_channels_factor), int(128 * n_channels_factor), filter_size)

        self.decoder1_conv_1 = self.conv(int(128 * n_channels_factor), int(64 * n_channels_factor), filter_size)
        self.decoder1_conv_2 = self.conv(int(64 * n_channels_factor), int(64 * n_channels_factor), filter_size)
        
        self.final_conv_reg = nn.Conv2d(int(64 * n_channels_factor), out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Make a land mask tensor that is the same shape as the output tensor
        land_mask = self.create_land_mask(spatial_shape)
        self.register_buffer("land_mask", land_mask)
    
    def create_land_mask(self, spatial_shape):
        ds = xr.open_dataset("/scratch/users/yucli/cesm_data/temp/temp_member_00.nc").isel(time=0)
        land_mask_npy = ~np.isnan(ds.temp).values       

        return torch.from_numpy(land_mask_npy).unsqueeze(0).repeat(6, 1, 1)
        
    def conv_block(self, in_channels, out_channels, filter_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=filter_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=filter_size, padding=filter_size//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    
    def conv(self, in_channels, out_channels, filter_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=filter_size//2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        
        dec3 = self.upsample(bottleneck)
        dec3 = self.decoder3_conv(dec3)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3_conv_block(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = self.decoder2_conv(dec2)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2_conv_block(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = self.decoder1_conv_1(dec1)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1_conv_1(dec1)
        dec1 = self.decoder1_conv_2(dec1)
        dec1 = self.decoder1_conv_2(dec1)

        # Mapping to (-1, 1)
        output = torch.tanh(self.final_conv_reg(dec1))

        if self.clip_near_zero_values:
            output = output.where(output.abs() > self.epsilon, 0)

        output = output * self.land_mask

        return output


class UNetRes4(UNetRes3):
    """
    Builds a UNet of resolution 4 (nomenclature from Williams et al. 2023)
    The resolution is defined as the number of encoder/decoder blocks. The
    number at the end of encoder and decoder blocks denote their depth in 
    the network (E1 -> E2 -> E3 -> E4 -> B -> D4 -> D3 -> D2 -> D1) where 
    B is the bottleneck block
    """

    def __init__(self, in_channels, out_channels, mode, device, spatial_shape=(336, 320), 
                n_channels_factor=1, filter_size=3, T=1.0, n_classes=2, predict_anomalies=False,
                clip_near_zero_anomalies=True, epsilon=0.01):

        super(UNetRes4, self).__init__(in_channels, out_channels, mode, device, spatial_shape, \
                                        n_channels_factor, filter_size, T, n_classes, predict_anomalies, \
                                        clip_near_zero_anomalies, epsilon)

        self.encoder4 = self.conv_block(int(256 * n_channels_factor), int(512 * n_channels_factor), filter_size)
        self.bottleneck = self.conv_block(int(512 * n_channels_factor), int(1024 * n_channels_factor), filter_size)

        self.decoder4_conv = self.conv(int(1024 * n_channels_factor), int(512 * n_channels_factor), filter_size)
        self.decoder4_conv_block = self.conv_block(2 * int(512 * n_channels_factor), int(512 * n_channels_factor), filter_size)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upsample(bottleneck)
        dec4 = self.decoder4_conv(dec4)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4_conv_block(dec4)
        
        dec3 = self.upsample(dec4)
        dec3 = self.decoder3_conv(dec3)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3_conv_block(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = self.decoder2_conv(dec2)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2_conv_block(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = self.decoder1_conv_1(dec1)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1_conv_1(dec1)
        dec1 = self.decoder1_conv_2(dec1)
        dec1 = self.decoder1_conv_2(dec1)

        if self.mode == "regression":
            if self.predict_anomalies:
                # Mapping to (-1, 1)
                output = torch.tanh(self.final_conv_reg(dec1))

                if self.clip_near_zero_anomalies:
                    output = output.where(torch.abs(output) < self.epsilon, 0, output)
            else: 
                # Mapping to (0, 1)
                output = torch.sigmoid(self.final_conv_reg(dec1))
            
            # Apply the land mask
            output = output * self.land_mask

        elif self.mode == "classification": 
            final_logits = torch.stack([self.final_convs_class[i](dec1) for i in range(out_channels)], dim=2)
            final_logits = final_logits.view(-1, 6, 3, spatial_shape[0], spatial_shape[1])
            final_logits = final_logits / self.T  # Apply temperature scaling
            output = F.softmax(final_logits, dim=2)  

            land_mask = self.land_mask.unsqueeze(2)  # Add a class dimension
            output = output * land_mask

            # for the no sea ice class, the land should be automatically assigned probability 1  
            output[:, :, 0, :, :] += (~land_mask[:, :, 0, :, :])

        return output

