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
from src.utils import util_cesm
from src import config_cesm


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, total_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs} [Train]")
    
    for _, batch in progress_bar:
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
    wandb.init(project="sea-ice-prediction", name=config.EXPERIMENT_NAME, 
               notes=config.NOTES)

    # Data split settings
    train_dataset = CESM_Dataset("train", config.DATA_SPLIT_SETTINGS)
    val_dataset = CESM_Dataset("val", config.DATA_SPLIT_SETTINGS)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = util_cesm.get_num_input_channels(config.INPUT_CONFIG)
    out_channels = util_cesm.get_num_output_channels(config.MAX_LEAD_MONTHS, config.TARGET_CONFIG)
    
    if config.MODEL == "UNetRes3": 
        model = UNetRes3(in_channels=in_channels, out_channels=out_channels).to(device)
    else: 
        raise NotImplementedError(f"Model {config.MODEL} not implemented.")
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    if config.LOSS == "MSE": 
        loss_fn = MSELoss()
    
    # Load a checkpoint if it exists
    save_dir = os.path.join(config_cesm.MODEL_DIRECTORY, config.EXPERIMENT_NAME)
    start_epoch = 1
    total_epochs = config.NUM_EPOCHS

    # Check for an existing checkpoint
    latest_checkpoint_name = max([f for f in os.listdir(save_dir) if f.endswith(".pth")]) 
    latest_epoch = int(latest_checkpoint_name.split("_")[-1].split(".")[0])

    if latest_epoch < total_epochs:
        checkpoint_path = os.path.join(save_dir, latest_checkpoint_name)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}.")

    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device, epoch, total_epochs)
        val_loss = validate_epoch(model, val_dataloader, loss_fn, device, epoch, total_epochs)

        # Log metrics to WandB
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Save a checkpoint
        checkpoint_path = os.path.join(save_dir, f"{config.MODEL}_{config.EXPERIMENT_NAME}_epoch_{epoch}.pth")
        if epoch % config.CHECKPOINT_INTERVAL == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)

        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the final model
    final_checkpoint_path = os.path.join(save_dir, f"{config.MODEL}_{config.EXPERIMENT_NAME}_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    main()
