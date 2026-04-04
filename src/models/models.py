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

