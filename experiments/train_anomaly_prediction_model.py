"""
Experiment: comparing predicting absolute sea ice vs. predicting the anomalies
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import xarray as xr
import pickle
import os
import pandas as pd
from netCDF4 import Dataset
import h5py

# module imports
from src import models
from src import models_util
from src import util_era5
from src import config
from src import losses

model_hyperparam_configs = {
    "name": "",
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

device = models_util.get_device()

# train 5 instances of each model 
for input_config in ["all_sicanom", "all"]:
    for seed in range(5):
        # set hyperparams
        model_hyperparam_configs["name"] = f"UNetRes3_{input_config}_{seed}"
        model_hyperparam_configs["input_config"] = input_config
        model_hyperparam_configs["seed"] = seed
        models_util.set_initialization_seed(model_hyperparam_configs["seed"], verbose=2)
        in_channels = 34 if input_config == "all_sicanom" else 40 

        # get constant hyperparams
        lr = model_hyperparam_configs["lr"]
        b = model_hyperparam_configs["batch_size"]

        print(f"Training {model_hyperparam_configs['name']} with lr={lr} and input_config={input_config} and batch_size={b}")

        model = models.UNetRes3(in_channels=in_channels, out_channels=6, mode="regression", device=device, \
                                n_channels_factor=1, filter_size=3, predict_anomalies=True).to(device)

        criterion = losses.MaskedMSELoss(use_weights=model_hyperparam_configs["use_zeros_weight"], \
                                    zero_class_weight=model_hyperparam_configs["zeros_weight"])

        if model_hyperparam_configs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=model_hyperparam_configs["lr"])
        else: 
            raise ValueError("Haven't yet configured training procedure for other optimizers")

        models_util.train_model(model, device, model_hyperparam_configs, optimizer, criterion, \
                        plot_training_curve=True, save_val_predictions=True, \
                        save_dir=f"{config.DATA_DIRECTORY}/sicpred/models/experiments/anom_vs_abs")