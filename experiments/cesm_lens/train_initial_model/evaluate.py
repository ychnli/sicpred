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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Known dimensions
    time_coords = data_split_settings["test"]
    num_members = len(ensemble_members)
    channels, x_dim, y_dim = 6, 80, 80
    reference_grid = xr.open_dataset(f"/scratch/users/yucli/cesm_data/icefrac/icefrac_member_00.nc")

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
        for i,batch in enumerate(test_dataloader):
            print(f"Evaluating for batch {i}")
            inputs = batch["input"].to(device)
            predictions = model(inputs).cpu().numpy()  # Move predictions to CPU

            # Extract metadata
            start_year, start_month = batch["start_prediction_month"].cpu().numpy()[0]
            member_id = batch["member_id"][0]

            # Find the appropriate indices
            time_idx = list(time_coords).index(pd.Timestamp(year=start_year, month=start_month, day=1))
            member_idx = list(ensemble_members).index(member_id)

            # Populate the dataset at the appropriate indices
            ds["predictions"][time_idx, member_idx, :, :, :] = predictions[0] 

    # Save the Dataset as NetCDF
    ds.to_netcdf(output_path)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
