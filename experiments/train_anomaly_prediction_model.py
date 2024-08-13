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


cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# If available, print the name of the GPU
if cuda_available:
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set the seed for random weights initialization 
torch.cuda.manual_seed(model_hyperparam_configs["seed"])
torch.cuda.manual_seed_all(model_hyperparam_configs["seed"])

lr = model_hyperparam_configs["lr"]
input_config = model_hyperparam_configs["input_config"]
b = model_hyperparam_configs["batch_size"]

print(f"Training {model_hyperparam_configs['name']} with lr={lr} and input_config={input_config} and batch_size={b}")

model = models.UNetRes3(in_channels=34, out_channels=6, mode="regression", device=device, \
                        n_channels_factor=1, filter_size=3, predict_anomalies=True).to(device)

criterion = losses.MaskedMSELoss(use_weights=model_hyperparam_configs["use_zeros_weight"], \
                            zero_class_weight=model_hyperparam_configs["zeros_weight"])

if model_hyperparam_configs["optimizer"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=model_hyperparam_configs["lr"])
else: 
    raise ValueError("Haven't yet configured training procedure for other optimizers")

util.train_model(model, device, model_hyperparam_configs, optimizer, criterion, \
                plot_training_curve=True, save_val_predictions=True)