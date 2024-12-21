# src/main.py
import argparse
import torch
import random
import numpy as np
from utils.config_parser import load_config

from data.dataloaders import build_dataloaders
from models.glip_loc import GLIPLocModel
from training.trainer import Trainer
from torch.utils.data import Subset, Dataset

from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader 

import pickle
from PIL import Image
class DebugDataset(Dataset):
    """
    Creates N unique ground-satellite pairs.
    For each sample i:
      - ground_img is a solid color (r, g, b) = (i*30 mod 256, 100, 200).
      - sat_img is a slightly different color (r, g, b) = (i*30 mod 256, 200, 100).
    The difference ensures the diagonal can be recognized as "positive pair".
    """
    def __init__(self, size=8, transform=None):
        super().__init__()
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Ensure idx is within 0...255 range for a color channel.
        color_val = (idx * 30) % 256

        # Ground image: color_val changes with idx
        ground_img_pil = Image.new("RGB", (224, 224), (color_val, 100, 200))

        # Satellite image: different color scheme
        sat_img_pil = Image.new("RGB", (224, 224), (color_val, 200, 100))

        if self.transform:
            ground_img_pil = self.transform(ground_img_pil)
            sat_img_pil = self.transform(sat_img_pil)

        # We won’t use label or text here, but keep placeholders to match your Trainer’s signature
        label = torch.tensor(idx, dtype=torch.long)  # or 0, but idx is safer for debugging
        ground_txt = ""
        sat_txt = ""

        return ground_img_pil, sat_img_pil, label, ground_txt, sat_txt
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

    # # Build Dataloaders (train and val)
    train_loader, val_loader = build_dataloaders(cfg)

    ## For testing
    train_dataset_small = Subset(train_loader.dataset, range(24))
    val_dataset_small = Subset(val_loader.dataset, range(24))
    train_loader = DataLoader(
    train_dataset_small,
    batch_size=cfg.dataset.batch_size,
    shuffle=True,
    num_workers=cfg.dataset.num_workers
)
    val_loader = DataLoader(
    val_dataset_small,
    batch_size=cfg.dataset.batch_size,
    shuffle=False,
    num_workers=cfg.dataset.num_workers
)
    # from torchvision import transforms
    # train_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])

    # # Create small debug datasets with e.g. 8 samples each
    # train_ds = DebugDataset(size=8, transform=train_transform)
    # val_ds = DebugDataset(size=8, transform=val_transform)
    # # Build loaders
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg.dataset.batch_size,  # e.g. 2 or 4
    #     shuffle=True,
    #     num_workers=cfg.dataset.num_workers
    # )
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=cfg.dataset.batch_size,
    #     shuffle=False,
    #     num_workers=cfg.dataset.num_workers
    # )
    sim_dict = None 
    # with open("/home/erzurumlu.1/yunus/research_drive/data/VIGOR/gps_dict_cross.pkl", "rb") as f:
    #     sim_dict = pickle.load(f)
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self, embed_dim=32):
            super().__init__()
            # Very small 2-layer MLP
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(224*224*3, 128),
                nn.ReLU(),
                nn.Linear(128, embed_dim),
            )

            # Fixed temperature, say 0.07
            self.temp = 0.07

        def forward(self, ground_image=None, satellite_image=None, *args, **kwargs):
            g_emb = self.net(ground_image) if ground_image is not None else None
            s_emb = self.net(satellite_image) if satellite_image is not None else None
            return g_emb, s_emb, self.temp
    # Build Model
    model = GLIPLocModel(
    model_name=cfg.model.name, 
    pretrained=cfg.model.pretrained, 
    use_text=cfg.model.use_text
)
    # model = TinyModel(embed_dim=512)
    # # Check if model requires grad is true
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    #     break  # Just check the first param for a quick look
    print(model.vision_model)
    for name, param in model.vision_model.named_parameters():
        param.requires_grad = False
    for name, param in model.vision_model.visual_projection.named_parameters():
        param.requires_grad = True
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Initialize Trainer
    # Trainer will handle accelerate setup, optimizers, and schedulers internally.
    trainer = Trainer(cfg, model, val_loader, val_loader, sim_dict=sim_dict, accelerator=accelerator)

    # Run Training
    trainer.run()

if __name__ == '__main__':
    main()
