import xarray as xr
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

from src.utils import util_cesm

class WeightedMSELoss(nn.Module):
    """
    Modification of standard MSE loss that allows for weighting by month and area
    """

    def __init__(self, device, area_weights_np, month_weights_np):
        
        super(WeightedMSELoss, self).__init__()
        
        self.device = device
        self.monthly_weights = torch.from_numpy(month_weights_np).to(device=device, dtype=torch.float32)
        self.area_weights = torch.from_numpy(area_weights_np).to(device=device, dtype=torch.float32)    

    def forward(self, prediction, target, target_months):
        squared_diff = (target - prediction) ** 2 # shape = (batch, channel, x, y)
        
        month_w = self.monthly_weights[target_months.long() - 1]     # (B, C)
        month_w = month_w[..., None, None]                           # (B, C, X, Y)
        w = month_w * self.area_weights
        mse_loss = (w * squared_diff).sum() / w.sum()
        return mse_loss
