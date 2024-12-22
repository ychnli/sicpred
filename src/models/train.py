import os
import torch
import wandb
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
import argparse 
import importlib.util

from src.models.models_util import CESM_Dataset
from src.models.models import UNetRes3


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, total_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs} [Train]")
    
    for batch_idx, batch in progress_bar:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        # Update progress bar with batch loss
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_fn, device, epoch, total_epochs):
    model.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs} [Val]")

    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            
            epoch_loss += loss.item()
            # Update progress bar with batch loss
            progress_bar.set_postfix({"Batch Loss": loss.item()})
    
    return epoch_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train a model with specified config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., config.py)")
    args = parser.parse_args()
    
    # Load configurations
    config = load_config(args.config)

    # Initialize WandB
    wandb.init(project="sea-ice-prediction", name=config.experiment_name, 
               notes=config.notes)

    # Data split settings
    data_dir = "/scratch/users/yucli/model-ready_cesm_data/data_pairs_setting1"
    ensemble_members = np.unique([name.split("_")[2].split(".")[0] for name in os.listdir(data_dir)])[0:5]

    train_dataset = CESM_Dataset(data_dir, ensemble_members, "train", data_split_settings)
    val_dataset = CESM_Dataset(data_dir, ensemble_members, "val", data_split_settings)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetRes3(in_channels=60, out_channels=6).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = MSELoss()
    
    checkpoint_path = f"UNetRes3_{experiment_name}_checkpoint.pth"
    start_epoch = 0

    # Check for an existing checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}.")

    total_epochs = 50

    for epoch in range(start_epoch, total_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device, epoch + 1, total_epochs)
        val_loss = validate_epoch(model, val_dataloader, loss_fn, device, epoch + 1, total_epochs)

        # Log metrics to WandB
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # Save a checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"UNetRes3_{experiment_name}.pth")
    wandb.finish()


if __name__ == "__main__":
    main()
