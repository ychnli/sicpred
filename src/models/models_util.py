import xarray as xr
import numpy as np
import pandas as pd
from time import time
from src import config
import os
import pickle
from netCDF4 import Dataset
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils import util_era5
from src import config
from src import config_cesm

##########################################################################################
# Model utils
##########################################################################################

class CESM_Dataset(torch.utils.data.Dataset):
    def __init__(self, split, data_split_settings):
        """
        Param:
            (string)        split ('test', 'val', or 'train')
            (dict)          data_split_settings

        """

        self.data_dir = os.path.join(config_cesm.PROCESSED_DATA_DIRECTORY, "data_pairs", data_split_settings["name"])
        self.split = split

        # Extract split settings
        self.split_by = data_split_settings["split_by"]
        self.split_range = data_split_settings[split]

        # Build a global index of samples
        self.samples = []
        if self.split_by == "time": 
            for member_id in data_split_settings["member_ids"]:
                input_file = os.path.join(self.data_dir, f"inputs_member_{member_id}.nc")
                ds = xr.open_dataset(input_file) 

                time_values = ds["start_prediction_month"].values
                for start_idx, time_val in enumerate(time_values):
                    if time_val in self.split_range:
                        self.samples.append((member_id, time_val, start_idx))

        elif self.split_by == "ensemble_member":
            for member_id in self.split_range: 
                input_file = os.path.join(self.data_dir, f"inputs_member_{member_id}.nc")
                ds = xr.open_dataset(input_file) 

                time_values = ds["start_prediction_month"].values
                for start_idx, time_val in enumerate(time_values):
                    if time_val in data_split_settings["time_range"]:
                        self.samples.append((member_id, time_val, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        member_id, start_prediction_month, start_idx = self.samples[idx]
        input_file = os.path.join(self.data_dir, f"inputs_member_{member_id}.nc")
        target_file = os.path.join(self.data_dir, f"targets_member_{member_id}.nc")

        # Load the specific sample lazily
        with xr.open_dataset(input_file) as input_ds:
            input_sample = input_ds["data"].isel(start_prediction_month=start_idx)

        with xr.open_dataset(target_file) as target_ds:
            target_sample = target_ds["data"].isel(start_prediction_month=start_idx)
        
        start_prediction_month = input_sample.start_prediction_month.values

        max_lead_months = target_sample.shape[0]
        start_prediction_months = pd.date_range(start_prediction_month, 
                                                start_prediction_month + pd.DateOffset(months=max_lead_months-1),
                                                freq="MS")

        time_npy = np.column_stack((start_prediction_months.year, start_prediction_months.month))

        sample = {
            "input": torch.tensor(input_sample.values, dtype=torch.float32),
            "target": torch.tensor(target_sample.values, dtype=torch.float32),
            "start_prediction_month": time_npy,
            "member_id": member_id
        }

        return sample


class Obs_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_directory, configuration, split_array, start_prediction_months, \
                split_type='train', target_shape=(336, 320), mode="regression", class_splits=None):
        self.data_directory = data_directory
        self.configuration = configuration
        self.split_array = split_array
        self.start_prediction_months = start_prediction_months
        self.split_type = split_type
        self.target_shape = target_shape
        self.class_splits = class_splits
        self.mode = mode

        # Open the HDF5 files
        self.inputs_file = h5py.File(f"{data_directory}/inputs_{configuration}.h5", 'r')

        if "sicanom" in configuration: 
            targets_configuration = "anom_regression" 
        else: 
            targets_configuration = "regression"

        self.targets_file = h5py.File(f"{data_directory}/targets_{targets_configuration}.h5", 'r')
        
        self.inputs = self.inputs_file[f"inputs_{configuration}"]
        self.targets = self.targets_file['targets_sea_ice_only']

        self.n_samples, self.n_channels, self.n_y, self.n_x = self.inputs.shape
        
        # Get indices for the specified split type
        if isinstance(split_type, str): 
            self.indices = np.where(split_array == split_type)[0]
        elif isinstance(split_type, list):
            self.indices = np.where(np.isin(split_array, split_type))[0]
        else:
            raise TypeError("split_type needs to be one of str or list")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        input_data = self.inputs[actual_idx]
        target_data = self.targets[actual_idx]
        start_prediction_month = self.start_prediction_months[actual_idx]

        # Pad input_data and target_data to the target shape
        pad_y = self.target_shape[0] - self.n_y
        pad_x = self.target_shape[1] - self.n_x
        input_data = np.pad(input_data, ((0, 0), (pad_y//2, pad_y//2), (pad_x//2, pad_x//2)), mode='constant', constant_values=0)
        target_data = np.pad(target_data, ((0, 0), (pad_y//2, pad_y//2), (pad_x//2, pad_x//2)), mode='constant', constant_values=0)

        # If we are doing classification, then discretise the target data
        if self.mode == "classification":
            if self.class_splits is None:
                raise ValueError("need to specify a monotonically increasing list class_splits denoting class boundaries")

            # check if class_split is monotonically increasing
            if len(self.class_splits) > 1 and np.any(np.diff(self.class_splits) < 0): 
                raise ValueError("class_splits needs to be monotonically increasing")

            bounds = [] # bounds for classes
            for i,class_split in enumerate(self.class_splits): 
                if i == 0: 
                    bounds.append([0, class_split])
                if i == len(self.class_splits) - 1: 
                    bounds.append([class_split, 1])
                else: 
                    bounds.append([class_split, self.class_splits[i+1]])
            
            target_classes_data = np.zeros_like(target_data) 
            target_classes_data = target_classes_data[np.newaxis,:,:,:]
            target_classes_data = np.repeat(target_classes_data, len(bounds), axis=0)
            for i,bound in enumerate(bounds): 
                if i == len(bounds) - 1: 
                    target_classes_data[i,:,:,:] = np.logical_and(target_data >= bound[0], target_data <= bound[1]).astype(int)
                else:
                    target_classes_data[i,:,:,:] = np.logical_and(target_data >= bound[0], target_data < bound[1]).astype(int)
            
            target_data = target_classes_data 

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        # Get the target months for this sample
        target_months = pd.date_range(start=start_prediction_month, end=start_prediction_month + pd.DateOffset(months=5), freq="MS")
        target_months = target_months.month.to_numpy()
        
        return input_tensor, target_tensor, target_months

    def __del__(self):
        self.inputs_file.close()
        self.targets_file.close()



def print_split_stats(split_array):
    ntrain = sum(split_array == 'train')
    nval = sum(split_array == 'val')
    ntest = sum(split_array == 'test')
    
    print(f"train samples: {ntrain} ({round(ntrain / len(split_array), 2)})")
    print(f"val samples: {nval} ({round(nval / len(split_array), 2)})")
    print(f"test samples: {ntest} ({round(ntest / len(split_array), 2)})")


def generate_start_prediction_months(max_month_lead_time=6, max_input_lag_time=12):
    # Construct the date range for the data pairs 
    # Note that this is not continuous due to the missing data in 1987-1988 
    first_range = pd.date_range('1981-01', pd.Timestamp('1987-12') - pd.DateOffset(months=max_month_lead_time+1), freq='MS')
    second_range = pd.date_range(pd.Timestamp('1988-01') + pd.DateOffset(months=max_input_lag_time+1), '2024-01', freq='MS')

    return first_range.append(second_range)


def generate_split_array(verbose=1):
    start_prediction_months = generate_start_prediction_months()
    split_array = np.empty(np.shape(start_prediction_months), dtype=object)
    
    for i,month in enumerate(start_prediction_months):
        if month in config.TRAIN_MONTHS: split_array[i] = "train"
        if month in config.VAL_MONTHS: split_array[i] = "val"
        if month in config.TEST_MONTHS: split_array[i] = "test"

    if verbose == 2: print_split_stats(split_array)
    
    return split_array, start_prediction_months


def get_device(verbose=1):
    cuda_available = torch.cuda.is_available()
    if verbose >= 1: print(f"CUDA available: {cuda_available}")

    # If available, print the name of the GPU
    if cuda_available and verbose >= 1:
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_initialization_seed(seed, verbose=1):
    if verbose >= 2: print(f"Setting random init seed to {seed}")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, device, model_hyperparam_configs, optimizer, criterion,
                plot_training_curve=True, verbose=1, save_val_predictions=True, 
                save_dir=f"{config.DATA_DIRECTORY}/sicpred/models"):

    # read in some metadata
    configuration = model_hyperparam_configs["input_config"]
    batch_size = model_hyperparam_configs["batch_size"]
    early_stopping = model_hyperparam_configs["early_stopping"]
    patience = model_hyperparam_configs["patience"]
    model_name = model_hyperparam_configs["name"]
    num_epochs = model_hyperparam_configs["max_epochs"]

    # generate array labeling which months are train/val/test
    split_array, start_prediction_months = generate_split_array()

    # create dataset instances for training, validation, and testing
    data_pairs_directory = os.path.join(config.DATA_DIRECTORY, 'sicpred/data_pairs_npy')
    train_dataset = SeaIceDataset(data_pairs_directory, configuration, split_array, start_prediction_months, split_type='train', target_shape=(336, 320))
    val_dataset = SeaIceDataset(data_pairs_directory, configuration, split_array, start_prediction_months, split_type='val', target_shape=(336, 320))
    test_dataset = SeaIceDataset(data_pairs_directory, configuration, split_array, start_prediction_months, split_type='test', target_shape=(336, 320))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    train_losses = []
    val_losses = []
    val_predictions = []
    best_val_loss = float('inf')
    epoch_at_early_stopping = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, target_months in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets, target_months)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        if verbose >= 1: print(f"Epoch [{epoch+1}/{num_epochs}], train loss: {epoch_loss:.4f}", end=', ')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, target_months in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets, target_months)
                val_loss += loss.item() * inputs.size(0)        

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        if verbose >= 1: print(f"validation loss: {val_loss:.4f}")

        # Early stopping check
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model weights
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1: print("Early stopping triggered")
                    epoch_at_early_stopping = epoch
                    break
        
    # Plot training curves
    if plot_training_curve:
        if verbose >= 1: print("Plotting training curve... ", end="")
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(0.5, len(train_losses) + 0.5, 1), train_losses, label="train")
        plt.plot(np.arange(1, len(val_losses) + 1, 1), val_losses, label="val")
        plt.legend()
        plt.title(model_name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()

        os.makedirs(f"{config.DATA_DIRECTORY}/figures/training_curves", exist_ok=True)
        plt.savefig(f"{config.DATA_DIRECTORY}/figures/training_curves/{model_name}_training_curve.png", bbox_inches='tight', dpi=300)
        plt.close()
        if verbose >= 1: print("done!")

    # Save the validation predictions for hyperparam tuning 
    if save_val_predictions:
        model.eval()
        with torch.no_grad():
            for inputs, targets, target_months in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_predictions.append(outputs.cpu().numpy())

        val_pred_save_path = os.path.join(save_dir, "val_predictions")
        os.makedirs(val_pred_save_path, exist_ok=True)
        np.save(f"{val_pred_save_path}/{model_name}_val_predictions.npy", np.concatenate(val_predictions, axis=0))

    # Save the trained model
    if verbose >= 1: print("Saving model weights...", end='')

    # record the best val loss and, if early stopping happened, the last epoch
    model_hyperparam_configs["best_val_loss"] = best_val_loss

    if epoch_at_early_stopping is not None: 
        model_hyperparam_configs["final_epoch"] = epoch_at_early_stopping
    else:
        model_hyperparam_configs["final_epoch"] = num_epochs
        
    os.makedirs(f"{config.DATA_DIRECTORY}/sicpred/val_predictions", exist_ok=True)
    util_era5.save_dict_to_pickle(model_hyperparam_configs, f"{config.DATA_DIRECTORY}/sicpred/models/{model_name}.pkl")
    torch.save(model.state_dict(), f"{config.DATA_DIRECTORY}/sicpred/models/{model_name}.pth")
    if verbose >= 1: print("done! \n\n")

    return train_losses, val_losses


def evaluate_model(model, model_hyperparam_configs, device):
    """
    Evaluates the model on val and test datasets and returns the predictions and 
    targets as numpy arrays. Note that the model should have weights loaded in 
    already and should be transferred to cuda if using GPU 
    """

    configuration = model_hyperparam_configs["input_config"]
    batch_size = model_hyperparam_configs["batch_size"]

    data_pairs_directory = os.path.join(config.DATA_DIRECTORY, 'sicpred/data_pairs_npy')
    split_array, start_prediction_months = generate_split_array()

    valtest_dataset = SeaIceDataset(data_pairs_directory, configuration, split_array, start_prediction_months, split_type=['val','test'], target_shape=(336, 320))
    valtest_loader = torch.utils.data.DataLoader(valtest_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
   
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets, _ in valtest_loader:
            inputs = inputs.to(device)
            predictions.append(model(inputs))
            targets_list.append(targets)
            
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    targets = torch.cat(targets_list, dim=0).cpu().numpy()

    return predictions, targets


def generate_empty_predictions_ds(reference_grid, time_coords, ensemble_members, max_lead_time, y_dim, x_dim):
    ds = xr.Dataset(
        {
            "predictions": (
                ["start_prediction_month", "member_id", "lead_time", "y", "x"],
                np.full((len(time_coords), len(ensemble_members), max_lead_time, y_dim, x_dim), np.nan, dtype=np.float32)
            )
        },
        coords={
            "start_prediction_month": time_coords,
            "member_id": ensemble_members,
            "lead_time": np.arange(1, max_lead_time+1),
            "y": reference_grid.y.values,
            "x": reference_grid.x.values,
        }
    )

    return ds 
