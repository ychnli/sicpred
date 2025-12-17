import os
import torch
import wandb
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import argparse 
import importlib.util
import inspect
import random

from src.models.models_util import CESM_Dataset
from src.models.models import UNetRes3
from src.models.models import SICNet
from src.utils import util_cesm
from src import config_cesm
from src.models.losses import WeightedMSELoss


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, total_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs} [Train]")
    loss_fn_params = inspect.signature(loss_fn.forward).parameters

    for _, batch in progress_bar:
        inputs, targets = batch["input"].to(device), batch["target"].to(device)
        
        optimizer.zero_grad()
        predictions = model(inputs)

        loss_kwargs = {"prediction": predictions, "target": targets}
        if "target_months" in loss_fn_params:
            # batch["start_prediction_month"] is of shape (batch_size, max_lead_months, 2)
            target_months = batch["start_prediction_month"][:, :, 1].to(device)
            loss_kwargs["target_months"] = target_months

        loss = loss_fn(**loss_kwargs)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        # Update progress bar with batch loss
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    return epoch_loss / len(dataloader)

def set_random_seed(seed):
    """Set random seeds for ensemble generation"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_epoch(model, dataloader, loss_fn, device, epoch, total_epochs):
    model.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{total_epochs} [Val]")
    loss_fn_params = inspect.signature(loss_fn.forward).parameters

    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            inputs, targets = batch["input"].to(device), batch["target"].to(device)
            predictions = model(inputs)

            loss_kwargs = {"prediction": predictions, "target": targets}
            if "target_months" in loss_fn_params:
                target_months = batch["start_prediction_month"][:, :, 1].to(device)
                loss_kwargs["target_months"] = target_months

            loss = loss_fn(**loss_kwargs)
            
            epoch_loss += loss.item()
            # Update progress bar with batch loss
            progress_bar.set_postfix({"Batch Loss": loss.item()})
    
    return epoch_loss / len(dataloader)

def get_best_checkpoint(checkpoint_dir, member_id=None):
    # Check for 'final' checkpoint first
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if (f.endswith(".pth") and "final" not in f)]

    # select the member_id
    if member_id is not None:
        checkpoint_files = [f for f in checkpoint_files if (f"member_{member_id}" in f)]
    
    if len(checkpoint_files) == 0:
        return None 
    
    latest_checkpoint_name = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])) 
    return os.path.join(checkpoint_dir, latest_checkpoint_name)


def main():
    parser = argparse.ArgumentParser(description="Train a model with specified config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., config.py)")
    parser.add_argument("--members", type=int, default=1, help="Number of ensemble members to train (default = 1)")
    parser.add_argument("--start_ens_id", type=int, default=0, help="Base seed (default = 0)")
    parser.add_argument("--resume", type=int, default=0, help="Number of additional epochs to resume training from latest checkpoint.")
    args = parser.parse_args()
    
    # Load configurations
    config = load_config(args.config)

    for ensemble_id in range(args.members):
        ensemble_id = args.start_ens_id + ensemble_id
        set_random_seed((ensemble_id) * 100) 

        # Initialize WandB
        wandb.init(project="sea-ice-prediction", 
                   name=f"{config.EXPERIMENT_NAME}_member_{ensemble_id}",
                   notes=config.NOTES, 
                   mode="online", 
                   config={"lr": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE})

        # Data split settings
        train_dataset = CESM_Dataset("train", config.DATA_SPLIT_SETTINGS)
        val_dataset = CESM_Dataset("val", config.DATA_SPLIT_SETTINGS)

        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Initialize model, optimizer, and loss function
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_channels = util_cesm.get_num_input_channels(config.INPUT_CONFIG)
        out_channels = util_cesm.get_num_output_channels(config.MAX_LEAD_MONTHS, config.TARGET_CONFIG)
        
        if config.MODEL == "UNetRes3": 
            model = UNetRes3(in_channels=in_channels, 
                            out_channels=out_channels, 
                            predict_anomalies=config.TARGET_CONFIG["predict_anom"]).to(device)
        elif config.MODEL == "SICNet":
            model = SICNet(T=in_channels, T_pred=out_channels, base_channels=32).to(device)
        else: 
            raise NotImplementedError(f"Model {config.MODEL} not implemented.")
        
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

        if config.LOSS_FUNCTION == "MSE": 
            loss_fn = WeightedMSELoss(device=device, model=model, **config.LOSS_FUNCTION_ARGS)
        
        # Load a checkpoint if it exists
        save_dir = os.path.join(config_cesm.MODEL_DIRECTORY, config.EXPERIMENT_NAME)
        start_epoch = 1
        total_epochs = config.NUM_EPOCHS
        if args.resume > 0:
            if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
                checkpoint_path = get_best_checkpoint(save_dir, ensemble_id)
                if checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_epoch = checkpoint["epoch"] + 1
                    total_epochs = start_epoch + args.resume - 1
                    print(f"Resuming training from epoch {start_epoch} for {args.resume} additional epochs.")
            else:
                print("No checkpoint found to resume from.")
                return


        # train model 
        os.makedirs(save_dir, exist_ok=True)
        for epoch in range(start_epoch, total_epochs + 1):
            train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device, epoch, total_epochs)
            val_loss = validate_epoch(model, val_dataloader, loss_fn, device, epoch, total_epochs)

            # Log metrics to WandB
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # Save a checkpoint
            file_name = f"{config.MODEL}_{config.EXPERIMENT_NAME}_member_{ensemble_id}_epoch_{epoch}.pth"
            checkpoint_path = os.path.join(save_dir, file_name)
            if epoch % config.CHECKPOINT_INTERVAL == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, checkpoint_path)

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the final model
        final_checkpoint_path = os.path.join(save_dir, 
                                f"{config.MODEL}_{config.EXPERIMENT_NAME}_member_{ensemble_id}_final.pth")
        torch.save(model.state_dict(), final_checkpoint_path)
        wandb.finish()


if __name__ == "__main__":
    main()
