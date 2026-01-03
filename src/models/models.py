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
from torch.nn.utils import weight_norm

from src import config_cesm
from src.utils import util_cesm
from src.models import models_util


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
                epsilon=0.02):

        super(UNetRes3, self).__init__()
        self.clip_near_zero_values = clip_near_zero_values
        self.epsilon = epsilon 
        self.predict_anomalies = predict_anomalies

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
        land_mask = self.create_inverted_land_mask()
        self.register_buffer("land_mask", land_mask)

    def create_inverted_land_mask(self):
        """
        Creates an inverted land mask (0s on land and 1s on ocean) according to the icefrac land 
        mask. This is smaller than the SST land mask due to representation of coastlines (thus
        icefrac is nonzero on coastline cells, whereas SST is NaN). 
        """
        try: 
            ds = xr.open_dataset(os.path.join(config_cesm.DATA_DIRECTORY, "cesm_data/grids/icefrac_land_mask.nc"))
        except:
            raise Exception("Uh oh, seems like you still need to run the preprocess script to generate \
                an icefrac land mask. See src/util_cesm for the function")

        land_mask_npy = ~ds.mask.values # inverted  

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

        if self.predict_anomalies:
            # Mapping to (-1, 1)
            output = torch.tanh(self.final_conv_reg(dec1))
        else: 
            # Mapping to (0, 1)
            output = torch.sigmoid(self.final_conv_reg(dec1)) 

        # apply clipping to zeros and land mask 
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


class Convblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Convblock,self).__init__()
        self.out_c = out_channel
        self.in_c = in_channel
        self.conv2d = nn.Conv2d(self.in_c,self.out_c,3,stride=1,padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size != 0:
            x = x[:, :, :-self.chomp_size].contiguous()
        else: x 
        return x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TSAM(nn.Module):
    def __init__(self, channels):
        super(TSAM, self).__init__()
        self.channels = channels
        kernel_size_tcn = max(channels // 12, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_tcn1 = TCN(num_inputs = 1, num_channels = [8,8,8], kernel_size = channels//12)
        self.max_tcn2 = TCN(num_inputs = 8, num_channels = [8,8,8], kernel_size = channels//12)
        self.max_tcn3 = TCN(num_inputs = 8, num_channels = [1,1], kernel_size = 1)
        self.avg_tcn1 = TCN(num_inputs = 1, num_channels = [8,8,8], kernel_size = channels//12)
        self.avg_tcn2 = TCN(num_inputs = 8, num_channels = [8,8,8], kernel_size = channels//12)
        self.avg_tcn3 = TCN(num_inputs = 8, num_channels = [1,1], kernel_size= 1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pooled = self.max_pool(x)

        avg_pooled = self.avg_pool(x)

        max_out = self.max_tcn3(self.max_tcn2(self.max_tcn1(torch.squeeze(max_pooled,-1).permute(0,2,1)))).permute(0,2,1)
        avg_out = self.avg_tcn3(self.avg_tcn2(self.avg_tcn1(torch.squeeze(avg_pooled,-1).permute(0,2,1)))).permute(0,2,1)

        tcn_out = max_out + avg_out
        tcn_attention = self.sigmoid(tcn_out.unsqueeze(-1))
        x = tcn_attention * x

        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_cat = torch.cat([max_pool, avg_pool], dim=1)
        spatial_attention = self.spatial_conv(pool_cat)

        x = x * self.sigmoid(spatial_attention)

        return x

class CNNTSAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, T):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tsam = TSAM(T)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.tsam(x)

class ResNetTSAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, T):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.tsam = TSAM(T)
        self.need_proj = in_channels != out_channels
        if self.need_proj:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.tsam(out)
        if self.need_proj:
            identity = self.proj(identity)
        return F.relu(out + identity)

class SICNet(nn.Module):
    def __init__(self, T, T_pred, base_channels, clip_near_zero_values=True, epsilon=0.01):
        super().__init__()
        base = base_channels
        self.T, self.T_pred = T, T_pred
        self.clip_near_zero_values = clip_near_zero_values
        self.epsilon = epsilon

        self.enc1 = CNNTSAMBlock(T, base*2, T)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            ResNetTSAMBlock(base*2, base*4, T), ResNetTSAMBlock(base*4, base*4, T))

        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            ResNetTSAMBlock(base*4, base*6, T), ResNetTSAMBlock(base*6, base*6, T))

        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(
            ResNetTSAMBlock(base*6, base*8, T), ResNetTSAMBlock(base*8, base*8, T))

        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = nn.Sequential(
            ResNetTSAMBlock(base*8, base*10, T), ResNetTSAMBlock(base*10, base*10, T))

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec4 = ResNetTSAMBlock(base*10 + base*8, base*8, T)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = ResNetTSAMBlock(base*8 + base*6, base*6, T)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ResNetTSAMBlock(base*6 + base*4, base*4, T)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ResNetTSAMBlock(base*4 + base*2, base*2, T)

        self.out_conv = nn.Conv2d(base*2, T_pred, kernel_size=1)

        land_mask = self.create_inverted_land_mask()
        self.register_buffer("land_mask", land_mask)

    def create_inverted_land_mask(self):
        try: 
            ds = xr.open_dataset(os.path.join(config_cesm.DATA_DIRECTORY, "cesm_lens/grids/icefrac_land_mask.nc"))
        except:
            raise Exception("Uh oh, seems like you still need to run the preprocess script to generate \
                an icefrac land mask. See src/util_cesm for the function")

        mask = ~ds.mask.values.astype(bool)  # invert land mask
        return torch.from_numpy(mask).float().unsqueeze(0).repeat(self.T_pred, 1, 1)

    def forward(self, x):  # x: (N, T, H, W)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        e5 = self.enc5(p4)

        d4 = self.up4(e5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = torch.tanh(self.out_conv(d1))  # (N, T_pred, H, W)

        if self.clip_near_zero_values:
            out = out.where(out.abs() > self.epsilon, 0)

        out = out * self.land_mask.unsqueeze(0)  # (1, T_pred, H, W)
        return out
