import xarray as xr
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src import config

class MaskedMSELoss(nn.Module):
    def __init__(self, device, use_weights=False, use_area_weighting=True, zero_class_weight=None, scale_factor=1):
        super(MaskedMSELoss, self).__init__()
        self.ice_mask = xr.open_dataset(f"{config.DATA_DIRECTORY}/NSIDC/monthly_ice_mask.nc").mask
        self.latitude_weights = np.cos(np.deg2rad(config.SPS_GRID.latitude.values))
        self.use_weights = use_weights
        self.use_area_weighting = use_area_weighting
        self.zero_class_weight = zero_class_weight
        self.scale_factor = scale_factor

        # pad the latitude weights and convert it to tensor
        self.latitude_weights = np.pad(self.latitude_weights, ((2,2), (2,2)), mode='constant', constant_values=0)
        self.latitude_weights = torch.tensor(self.latitude_weights).to(device)

    def forward(self, outputs, targets, target_months):
        n_active_cells = 0

        for target_months_subset in target_months:
            n_active_cells += self.ice_mask.sel(month=target_months_subset.cpu()).sum().values
        
        # Punish predictions of sea ice in ice free zones 
        diff = targets - outputs 
        if self.use_area_weighting: 
            # ignore padded regions
            # maybe this code is better to be generalized for arbitrary padding but I don't think
            # the padding will be much different 
            diff = diff * self.latitude_weights

        if self.use_weights:
            weights = torch.where(targets == 0, self.zero_class_weight, 1)
            loss = torch.sum(((diff) ** 2) * weights) / n_active_cells
        else:
            loss = torch.sum((diff) ** 2) / n_active_cells
        
        return loss * self.scale_factor


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