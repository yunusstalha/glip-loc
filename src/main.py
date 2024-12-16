# src/main.py
import argparse
import torch
import random
import numpy as np
from utils.config_parser import load_config
from utils.wandb_logger import WandbLogger
from accelerate import Accelerator

from data.dataloaders import build_dataloaders
# from models.model_builder import build_model
# from training.trainer import Trainer

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Main entry point for training.")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load the full configuration
    cfg = load_config(args.config)
    
    # Set random seed
    set_random_seed(cfg.training.seed)

    # Initialize W&B if enabled
    logger = WandbLogger(cfg)

    # Initialize accelerator if enabled
    accelerator = None
    if cfg.accelerate.enabled:
        precision = 'fp16' if cfg.accelerate.fp16 else 'no'
        accelerator = Accelerator(mixed_precision=precision)

    # Build Dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

    # Build Model (placeholder)
    # model = build_model(cfg.model)

    # Prepare with accelerator if using multi-GPU / mixed precision
    # if accelerator is not None:
    #     model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)

    # Trainer (placeholder)
    # trainer = Trainer(cfg, model, train_loader, val_loader, logger, accelerator)
    # trainer.run()

    print("Dataloaders and dataset integrated successfully.")

if __name__ == '__main__':
    main()
