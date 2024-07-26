import models
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
import util
import config

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# If available, print the name of the GPU
if cuda_available:
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")

# "Default" values. These will get changed in the hyperparam sweep loop
model_hyperparam_configs = {
    "name": "UNetRes3_simpleinputs_b1lr1",
    "architecture": "UNetRes3",
    "input_config": "simple", 
    "batch_size": 4,
    "lr": 1e-4,
    "optimizer": "adam",
    "use_zeros_weight": False,
    "zeros_weight": None,
    "early_stopping": True,
    "patience": 10,
    "max_epochs": 100,
    "notes": ""
}

in_channels_config = {
    "sea_ice_only": 14,
    "simple": 23,
    "all": 40
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_sizes = [16, 32]
learning_rates = [1e-4, 1e-5]
input_configs = ["simple", "all"]
architectures = ["UNetRes3", "UNetRes4"]

for i,b in enumerate(batch_sizes):
    for j,lr in enumerate(learning_rates):
        for input_config in input_configs:
            for arch in architectures: 
                model_hyperparam_configs["name"] = f"{arch}_{input_config}inputs_b{i}lr{j}"
                model_hyperparam_configs["architecture"] = arch
                model_hyperparam_configs["lr"] = lr
                model_hyperparam_configs["input_config"] = input_config

                if arch == "UNetRes3":
                    model = models.UNetRes3(in_channels=in_channels_config[model_hyperparam_configs["input_config"]], \
                                            out_channels=6, device=device, n_channels_factor=1, filter_size=3).to(device)
                elif arch == "UNetRes4":
                    model = models.UNetRes4(in_channels=in_channels_config[model_hyperparam_configs["input_config"]], \
                                            out_channels=6, device=device, n_channels_factor=1, filter_size=3).to(device)

                criterion = util.MaskedMSELoss(use_weights=model_hyperparam_configs["use_zeros_weight"], \
                                            zero_class_weight=model_hyperparam_configs["zeros_weight"])

                if model_hyperparam_configs["optimizer"] == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=model_hyperparam_configs["lr"])
                else: 
                    raise ValueError("Haven't yet configured training procedure for other optimizers")

                util.train_model(model, device, model_hyperparam_configs, optimizer, criterion, plot_training_curve=True)