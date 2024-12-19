# src/main.py
import argparse
import torch
import random
import numpy as np
from utils.config_parser import load_config

from data.dataloaders import build_dataloaders
from models.glip_loc import GLIPLocModel
from training.trainer import Trainer

from accelerate import Accelerator, DistributedDataParallelKwargs

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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.accelerate.fp16 else "no",
        log_with="wandb" if cfg.wandb.enabled else None,
        project_dir=cfg.accelerate.logging_dir if hasattr(cfg.accelerate, 'logging_dir') else None,
        kwargs_handlers=[ddp_kwargs])

    if cfg.wandb.enabled and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=vars(cfg),
            init_kwargs={"entity": cfg.wandb.entity}
        )

    # Build Dataloaders (train and val)
    train_loader, val_loader = build_dataloaders(cfg)
    with open("/home/erzurumlu.1/yunus/research_drive/data/VIGOR/gps_dict_cross.pkl", "rb") as f:
        sim_dict = pickle.load(f)


    # Build Model
    model = GLIPLocModel(
    model_name=cfg.model.name, 
    pretrained=cfg.model.pretrained, 
    use_text=cfg.model.use_text
)

    # Initialize Trainer
    # Trainer will handle accelerate setup, optimizers, and schedulers internally.
    trainer = Trainer(cfg, model, train_loader, val_loader, sim_dict=sim_dict, accelerator=accelerator)

    # Run Training
    trainer.run()

if __name__ == '__main__':
    main()
