"""
Experiment: Train an ensemble of 15 sea ice anomaly prediction networks
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
input_config = "all_sicanom"

overwrite_model_training = False

if __name__ == "__main__":
    # training loop 
    for seed in range(0, 15, 1):
        # set hyperparams
        model_hyperparam_configs["name"] = f"UNetRes3_{input_config}_{seed}"
        model_hyperparam_configs["input_config"] = input_config
        model_hyperparam_configs["seed"] = seed
        models_util.set_initialization_seed(model_hyperparam_configs["seed"], verbose=2)
        in_channels = 37

        # check if model has already been trained 
        model_val_pred_path = f"{config.DATA_DIRECTORY}/sicpred/models/experiments/anom_ensemble_2/val_predictions/UNetRes3_{input_config}_{seed}_val_predictions.npy"
        if not overwrite_model_training and os.path.exists(model_val_pred_path): 
            print(f"Already found trained UNetRes3 for {input_config} and seed {seed}, skipping...")
            continue 

        # get constant hyperparams
        lr = model_hyperparam_configs["lr"]
        b = model_hyperparam_configs["batch_size"]

        print(f"Training {model_hyperparam_configs['name']} with lr={lr} and input_config={input_config} and batch_size={b}")

        model = models.UNetRes3(in_channels=in_channels, out_channels=6, mode="regression", device=device, \
                                n_channels_factor=1, filter_size=3, predict_anomalies=True, \
                                clip_near_zero_anomalies=False).to(device)

        criterion = losses.MaskedMSELoss(device, use_weights=model_hyperparam_configs["use_zeros_weight"], \
                                        zero_class_weight=model_hyperparam_configs["zeros_weight"], \
                                        use_area_weighting=True, scale_factor=100)

        if model_hyperparam_configs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=model_hyperparam_configs["lr"])
        else: 
            raise ValueError("Haven't yet configured training procedure for other optimizers")

        models_util.train_model(model, device, model_hyperparam_configs, optimizer, criterion, \
                        plot_training_curve=True, save_val_predictions=True, \
                        save_dir=f"{config.DATA_DIRECTORY}/sicpred/models/experiments/anom_ensemble_2")

    print("\n\n")

    # evaluate
    predictions_list = []
    for seed in range(0, 15, 1):
        print(f"Running inference using UNetRes3 anomaly prediction seed={seed} on val and test data")
        model = models.UNetRes3(in_channels=in_channels, out_channels=6, mode="regression", device=device, \
                                n_channels_factor=1, filter_size=3, predict_anomalies=True)

        model_weights_save_path = f"{config.DATA_DIRECTORY}/sicpred/models/experiments/anom_ensemble_2/UNetRes3_{input_config}_{seed}.pth"

        model.load_state_dict(torch.load(model_weights_save_path, weights_only=True))
        model.to(device)

        predictions, targets = models_util.evaluate_model(model, model_hyperparam_configs, device)
        
        predictions_list.append(predictions)
    
    print(f"\n\n Concatenating predictions...", end='')
    predictions_list = np.stack(predictions_list, axis=0)

    valtest_init_months = config.VAL_MONTHS.append(config.TEST_MONTHS) - pd.DateOffset(months=1)

    valtest_pred_ds = xr.Dataset(
        data_vars=dict(
            siconc=(["ens_member", "init_month", "lead", "y", "x"], predictions_list[:,:,:,2:-2,2:-2]),
        ),
        coords=dict(
            y=("y", config.SPS_GRID.ygrid.data),
            x=("x", config.SPS_GRID.xgrid.data),
            init_month=valtest_init_months, 
            lead=np.arange(1,7,1),
            ens_member=np.arange(15)
        )
    )
    print("saving...", end="")
    
    util_era5.write_nc_file(valtest_pred_ds, f"{config.DATA_DIRECTORY}/sicpred/models/experiments/anom_ensemble_2/valtest_pred.nc", overwrite=True)
    print("done! \n\n")