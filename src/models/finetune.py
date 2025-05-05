"""
Finetunes models pretrained on CESM data to the ~500 datapoints of ERA5. 
Alternatively, you can also use this script to train models on observations 
only (by simply specifying random weights initialization instead of using 
a pretrained checkpoint). 

Make sure you run preprocess_era5_data.py before running this script
"""

import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import torch 
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb
import argparse 
import importlib.util
import inspect
import random

from src.models.models_util import CESM_Dataset
from src.models.models import UNetRes3
from src.utils import util_cesm
from src import config_cesm
from src.models.losses import WeightedMSELoss
import src.config as config_era5

