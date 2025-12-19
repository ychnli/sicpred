import xarray as xr
import numpy as np
import pandas as pd
from time import time
import os
import pickle
from netCDF4 import Dataset
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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