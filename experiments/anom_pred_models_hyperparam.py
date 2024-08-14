"""
Experiment: generate a hyperparameter sweep over the anomaly prediction models 
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
from src import util
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

device = util.get_device()

# train 5 instances of each model 
for arch in ["UNetRes4", "UNetRes3"]:
    for lr in [1e-4, 1e-5, 1e-6]:
        for batch_size in [1, 4]:
            # set hyperparams
            if lr == 1e-4: lr_str = "1e-4"
            if lr == 1e-5: lr_str = "1e-5"
            if lr == 1e-6: lr_str = "1e-6"
            model_hyperparam_configs["name"] = f"{arch}_all_sicanom_lr={lr_str}b={batch_size}"
            model_hyperparam_configs["arch"] = arch
            model_hyperparam_configs["batch_size"] = batch_size
            model_hyperparam_configs["lr"] = lr

            util.set_initialization_seed(model_hyperparam_configs["seed"], verbose=2)
            in_channels = 34 

            # get constant hyperparams
            print(f"Training {model_hyperparam_configs['arch']} with lr={lr} and batch_size={batch_size}")

            model = models.UNetRes3(in_channels=in_channels, out_channels=6, mode="regression", device=device, \
                                    n_channels_factor=1, filter_size=3, predict_anomalies=True).to(device)

            criterion = losses.MaskedMSELoss(use_weights=model_hyperparam_configs["use_zeros_weight"], \
                                        zero_class_weight=model_hyperparam_configs["zeros_weight"])

            if model_hyperparam_configs["optimizer"] == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=model_hyperparam_configs["lr"])
            else: 
                raise ValueError("Haven't yet configured training procedure for other optimizers")

            util.train_model(model, device, model_hyperparam_configs, optimizer, criterion, \
                            plot_training_curve=True, save_val_predictions=True, \
                            save_dir=f"{config.DATA_DIRECTORY}/sicpred/models/anom_pred")