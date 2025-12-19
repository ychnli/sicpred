import xarray as xr
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

from src import config
from src.utils import util_cesm

class WeightedMSELoss(nn.Module):
    """
    Modification of standard MSE loss that allows for weighting by month and area
    """

    def __init__(self, device, model, 
                 monthly_weights=None, 
                 apply_area_weights=True, 
                 l2_lambda=0):
        
        super(WeightedMSELoss, self).__init__()
        
        self.device = device
        self.model = model
        self.monthly_weights = torch.from_numpy(monthly_weights).to(device=device, dtype=torch.float32)
        self.apply_area_weights = apply_area_weights
        if apply_area_weights:
            area_weights = util_cesm.calculate_area_weights()
            self.area_weights = torch.from_numpy(area_weights).to(device=device, dtype=torch.float32)
        self.scale_factor = scale_factor
        self.l2_lambda = l2_lambda
    

    def forward(self, prediction, target, target_months):
        squared_diff = (target - prediction) ** 2 # shape = (batch, channel, x, y)
        
        if self.apply_month_weights:
            month_w = self.monthly_weights[target_months.long() - 1]     # (B, C)
            month_w = month_w[..., None, None]                           # (B, C, X, Y)
        else:
            month_w = 1.0

        if self.apply_area_weights:
            area_w = self.area_weights
        else:
            area_w = 1.0

        w = month_w * area_w
        mse_loss = (w * squared_diff).sum() / w.sum()

        # L2 regularization term 
        if self.l2_lambda != 0: 
            reg_loss = self.l2_lambda * sum(torch.sum(param ** 2) for param in self.model.parameters() if param.requires_grad)
            return mse_loss + reg_loss 
        
        return mse_loss
        

class CategoricalFocalLoss(nn.Module):
    """
    A modification of cross-entropy loss with an extra factor of (1 - P)^gamma 
    to up-weight egregiously incorrect classifications (e.g., low predicted P
    for the correct class). 

    Params:
        device:         "cuda" or "cpu" 
        gamma:          strength of focal scaling parameter (default=2)
    """

    def __init__(self, device, gamma=2., weight_by_month=True): 
        super(CategoricalFocalLoss, self).__init__()
        self.ice_mask = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/monthly_ice_mask.nc").mask
        self.epsilon = 1e-10
        self.gamma = gamma 
        self.device = device
        self.weight_by_month = weight_by_month

        if weight_by_month: self.monthly_weights = self.calculate_monthly_weight()

    def calculate_monthly_weight(self):
        monthly_max_ice_cells = self.ice_mask.sum(dim=("x","y"))

        return monthly_max_ice_cells / np.min(monthly_max_ice_cells)
    
    def forward(self, outputs, targets, target_months):
        """
        Params: 
            outputs:        represent a probability distribution over n classes
            targets:        one-hot encoded vector of the same shape as outputs of the correct class
            target_months:  corresponding 6-month period of prediction 
        """
        
        # clip 0 values to prevent inf during log step 
        outputs = torch.where(outputs == 0, outputs + self.epsilon, outputs) 
        
        cross_entropy = -targets * torch.log(outputs)

        # N x n_classes x 6 x d1 x d2 
        focal_loss = ((1 - outputs) ** self.gamma) * cross_entropy 

        if self.weight_by_month:
            for i,batch_target_months in enumerate(target_months):
                # Make a tensor of the monthly weights
                weights = torch.tensor(self.monthly_weights.sel(month=batch_target_months.cpu()).values).to(self.device)
                weights = weights.unsqueeze(0).unsqueeze(2).unsqueeze(2)

                focal_loss[i,:,:,:,:] *= weights

        return torch.mean(focal_loss)