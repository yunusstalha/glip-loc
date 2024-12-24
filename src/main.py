# src/main.py
import argparse
import random
import pickle 
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator, DistributedDataParallelKwargs
from utils.config_parser import load_config
from data.dataloaders import build_dataloaders
from models.glip_loc import GLIPLocModel
from training.trainer import Trainer
# import torchvision.transforms as T
# from data.vigor_plus import VigorDataset

# def build_eval_dataloaders(cfg):
#     # Minimal eval transforms
#     sat_transforms = T.Compose([
#         T.Resize((384, 384)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225])
#     ])
#     ground_transforms = T.Compose([
#         T.Resize((384, 768)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225])
#     ])

#     reference_dataset = VigorDataset(
#         data_folder=cfg.dataset.data_folder,
#         split="test",
#         same_area=cfg.dataset.same_area,
#         dataset_mode="reference",
#         satellite_transforms=sat_transforms,
#         ground_transforms=None,
#         prob_flip=0.0,
#         prob_rotate=0.0,
#         use_captions=False
#     )
#     reference_loader = DataLoader(
#         reference_dataset,
#         batch_size=cfg.dataset.batch_size,
#         shuffle=False,
#         num_workers=cfg.dataset.num_workers,
#         pin_memory=True
#     )

#     query_dataset = VigorDataset(
#         data_folder=cfg.dataset.data_folder,
#         split="test",
#         same_area=cfg.dataset.same_area,
#         dataset_mode="query",
#         ground_transforms=ground_transforms,
#         satellite_transforms=None,
#         prob_flip=0.0,
#         prob_rotate=0.0,
#         use_captions=False
#     )
#     query_loader = DataLoader(
#         query_dataset,
#         batch_size=cfg.dataset.batch_size,
#         shuffle=False,
#         num_workers=cfg.dataset.num_workers,
#         pin_memory=True
#     )
#     return reference_loader, query_loader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Main entry point for training.")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    # 1) Load config
    cfg = load_config(args.config)

    # 2) Set seed for reproducibility
    set_seed(cfg.training.seed)

    # 3) Accelerator setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.accelerate.fp16 else "no",
        log_with="wandb" if cfg.wandb.enabled else None,
        project_dir=cfg.accelerate.logging_dir if hasattr(cfg.accelerate, 'logging_dir') else None,
        kwargs_handlers=[ddp_kwargs]
    )

    # (Optional) If you want Weights & Biases logging:
    # if cfg.wandb.enabled and accelerator.is_main_process:
    #     accelerator.init_trackers(
    #         project_name=cfg.wandb.project,
    #         config=vars(cfg),
    #         init_kwargs={"entity": cfg.wandb.entity}
    #     )

    # 4) Build train & val dataloaders from your config
    train_loader, val_loader = build_dataloaders(cfg)
    # reference_loader, query_loader = build_eval_dataloaders(cfg)
    # (Optional) Subset for quick debug
    # train_small = Subset(train_loader.dataset, range(50))
    # val_small = Subset(val_loader.dataset, range(50))
    # train_loader = DataLoader(train_small, batch_size=cfg.dataset.batch_size, shuffle=True)
    # val_loader = DataLoader(val_small, batch_size=cfg.dataset.batch_size, shuffle=False)

    # 5) Build model
    model = GLIPLocModel(
        model_name=cfg.model.name,
        pretrained=cfg.model.pretrained,
        use_text=cfg.model.use_text
    )
    sim_dict = None
    with open('/home/erzurumlu.1/yunus/research_drive/data/VIGOR/gps_dict_cross.pkl', "rb") as f:
        sim_dict = pickle.load(f)
        
    # 6) Initialize the Trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        sim_dict=sim_dict,       # or load your gps_dict if needed
        accelerator=accelerator
    )

    # 7) Run Training
    trainer.run()

if __name__ == "__main__":
    main()
