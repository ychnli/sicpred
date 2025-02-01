import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import importlib.util
import xarray as xr
from tqdm import tqdm  

from src.models.models_util import CESM_Dataset
from src.models.models import UNetRes3
from src.utils import util_cesm
from src import config_cesm


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train a model with specified config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., config.py)")
    args = parser.parse_args()
    
    # Load configurations
    config = load_config(args.config)

    test_dataset = CESM_Dataset("test", config.DATA_SPLIT_SETTINGS)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load the model and checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = util_cesm.get_num_input_channels(config.INPUT_CONFIG)
    out_channels = util_cesm.get_num_output_channels(config.MAX_LEAD_MONTHS, config.TARGET_CONFIG)
    
    if config.MODEL == "UNetRes3": 
        model = UNetRes3(in_channels=in_channels, 
                        out_channels=out_channels, 
                        predict_anomalies=config.TARGET_CONFIG["predict_anom"]).to(device)
    else: 
        raise NotImplementedError(f"Model {config.MODEL} not implemented.")

    checkpoint_path = os.path.join(config_cesm.MODEL_DIRECTORY, config.EXPERIMENT_NAME,
                                 f"{config.MODEL}_{config.EXPERIMENT_NAME}_{config.CHECKPOINT_TO_EVALUATE}.pth")

    if not os.path.exists(checkpoint_path): 
        raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Known dimensions
    if config.DATA_SPLIT_SETTINGS["split_by"] == "ensemble_member":
        ensemble_members = config.DATA_SPLIT_SETTINGS["test"]
        time_coords = config.DATA_SPLIT_SETTINGS["time_range"]
    elif config.DATA_SPLIT_SETTINGS["split_by"] == "time":
        ensemble_members = config.DATA_SPLIT_SETTINGS["member_ids"]
        time_coords = config.DATA_SPLIT_SETTINGS["test"]
    num_members = len(ensemble_members)
    channels, x_dim, y_dim = config.MAX_LEAD_MONTHS, 80, 80
    reference_grid = util_cesm.generate_sps_grid()

    # Initialize an empty xarray Dataset
    ds = xr.Dataset(
        {
            "predictions": (
                ["start_prediction_month", "member_id", "lead_time", "y", "x"],
                np.full((len(time_coords), num_members, channels, y_dim, x_dim), np.nan, dtype=np.float32)
            )
        },
        coords={
            "start_prediction_month": time_coords,
            "member_id": ensemble_members,
            "lead_time": np.arange(1, channels+1),
            "y": reference_grid.y.values,
            "x": reference_grid.x.values,
        }
    )

    # Populate the Dataset with predictions
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Evaluating", unit="sample")): 
            inputs = batch["input"].to(device)
            predictions = model(inputs).cpu().numpy()  # Move predictions to CPU
            
            # since batch size = 1, get the only sample in the batch
            predictions = predictions[0]

            # Extract metadata
            start_year, start_month = batch["start_prediction_month"].cpu().numpy()[0,0]
            start_prediction_month = pd.Timestamp(year=start_year, month=start_month, day=1)
            member_id = batch["member_id"][0]

            # Find the appropriate indices
            time_idx = list(time_coords).index(start_prediction_month)
            member_idx = list(ensemble_members).index(member_id)

            ds["predictions"][time_idx, member_idx, :, :, :] = predictions

    # Save the Dataset as NetCDF
    output_dir = os.path.join(config_cesm.PREDICTIONS_DIRECTORY, config.EXPERIMENT_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{config.MODEL}_{config.CHECKPOINT_TO_EVALUATE}_predictions.nc")
    ds.to_netcdf(output_path)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()