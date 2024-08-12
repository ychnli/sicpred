import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import pickle
import os
import pandas as pd
import h5py

# module imports
from src import models
from src import util
from src import config
from src import losses

model_hyperparam_configs = {
    "name": "UNetRes3_allinputs_pred_anom_test",
    "architecture": "UNetRes3",
    "input_config": "all_sicanom", 
    "batch_size": 4,
    "lr": 1e-4,
    "optimizer": "adam",
    "seed": 1,
    "use_zeros_weight": False,
    "zeros_weight": None,
    "early_stopping": True,
    "patience": 10,
    "max_epochs": 100,
    "notes": ""
}

torch.cuda.manual_seed(model_hyperparam_configs["seed"])
torch.cuda.manual_seed_all(model_hyperparam_configs["seed"])

print(f"Training UNetRes3 with lr={lr} and input_config={input_config} and batch_size={b}")

model = models.UNetRes3(in_channels=in_channels_config[model_hyperparam_configs["input_config"]], \
                        out_channels=6, mode="regression", device=device, n_channels_factor=1, filter_size=3).to(device)

criterion = util.MaskedMSELoss(use_weights=model_hyperparam_configs["use_zeros_weight"], \
                            zero_class_weight=model_hyperparam_configs["zeros_weight"])

if model_hyperparam_configs["optimizer"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=model_hyperparam_configs["lr"])
else: 
    raise ValueError("Haven't yet configured training procedure for other optimizers")

