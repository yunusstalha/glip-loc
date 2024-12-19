# src/main.py
import argparse
import torch
import random
import numpy as np
from utils.config_parser import load_config
from utils.wandb_logger import WandbLogger

from data.dataloaders import build_dataloaders
from models.glip_loc import GLIPLocModel
from training.trainer import Trainer

import pickle

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
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(cfg)

    # Build Dataloaders (train and val)
    train_loader, val_loader = build_dataloaders(cfg)
    with open("./data/VIGOR/gps_dict_cross.pkl", "rb") as f:
        sim_dict = pickle.load(f)


    # Build Model
    model = GLIPLocModel(
    model_name=cfg.model.name, 
    pretrained=cfg.model.pretrained, 
    use_text=cfg.model.use_text
)

    # Initialize Trainer
    # Trainer will handle accelerate setup, optimizers, and schedulers internally.
    trainer = Trainer(cfg, model, train_loader, val_loader, logger=logger, sim_dict=sim_dict)

    # Run Training
    trainer.run()

if __name__ == '__main__':
    main()
