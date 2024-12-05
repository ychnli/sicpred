import os
import torch
import xarray as xr
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import UNetRes3
from data_loader import CESM_SeaIceDataset


def main():
    # Paths and settings
    checkpoint_path = "/scratch/users/yucli/UNetRes3_mini_experiment_5_ens_members_checkpoint.pth"
    output_path = "/scratch/users/yucli/UNetRes3_mini_experiment_5_ens_members_test_predictions.nc"
    data_dir = "/scratch/users/yucli/model-ready_cesm_data/data_pairs_setting1"

    data_split_settings = {
        "split_by": "time",
        "train": pd.date_range("1851-01", "1979-12", freq="MS"),
        "val": pd.date_range("1980-01", "1994-12", freq="MS"),
        "test": pd.date_range("1995-01", "2013-12", freq="MS")
    }

    ensemble_members = np.unique([name.split("_")[2].split(".")[0] for name in os.listdir(data_dir)])[0:5]
    test_dataset = CESM_SeaIceDataset(data_dir, ensemble_members, "test", data_split_settings)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load the model and checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetRes3(in_channels=60, out_channels=6).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference and collect predictions
    all_predictions = []
    all_times = []
    all_ensemble_members = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch["input"].to(device)
            predictions = model(inputs)  # Shape: [batch_size, out_channels, x, y]
            predictions = predictions.cpu().numpy()  # Move to CPU for saving
            
            all_predictions.append(predictions)
            
            # Process additional metadata
            start_prediction_month = batch["start_prediction_month"].cpu().numpy()
            all_times.append(start_prediction_month)
            all_ensemble_members.append(batch["member_id"])

    # Concatenate predictions and metadata
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: [time_samples, channels, x, y]
    all_times = np.array(all_times).reshape(-1, 2)  # Shape: [time_samples, 2]
    all_ensemble_members = np.concatenate(all_ensemble_members)

    # Convert predictions to xarray Dataset
    years, months = all_times[:, 0], all_times[:, 1]
    time_coords = pd.date_range(
        start=f"{int(years[0])}-{int(months[0]):02d}",
        periods=len(years),
        freq="MS"
    )
    ds = xr.Dataset(
        {
            "predictions": (
                ["time", "ensemble_member", "channel", "x", "y"],
                all_predictions,
            )
        },
        coords={
            "time": time_coords,
            "ensemble_member": all_ensemble_members,
            "x": np.arange(all_predictions.shape[-2]),
            "y": np.arange(all_predictions.shape[-1]),
        },
    )

    # Save predictions as NetCDF
    ds.to_netcdf(output_path)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
