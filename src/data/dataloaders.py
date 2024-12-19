# src/data/dataloaders.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .vigor_plus import VigorDataset

def build_transforms(transform_cfg):
    """Build torchvision transforms from config."""
    if transform_cfg is None:
        return transforms.Compose([
            transforms.ToTensor()
        ])

    # This is a simple example. You might need to adjust according to your config structure.
    t_list = []
    for t in transform_cfg:
        t_type = t["type"]
        if t_type == "RandomHorizontalFlip":
            t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        elif t_type == "RandomCrop":
            t_list.append(transforms.RandomCrop((t["height"], t["width"])))
        elif t_type == "Resize":
            t_list.append(transforms.Resize((t["height"], t["width"])))
        elif t_type == "Normalize":
            t_list.append(transforms.Normalize(mean=t["mean"], std=t["std"]))
        elif t_type == "ToTensor":
            t_list.append(transforms.ToTensor())

    # Ensure we have a ToTensor at some point; if not already included, add at the end
    if not any(isinstance(x, transforms.ToTensor) for x in t_list):
        t_list.append(transforms.ToTensor())

    return transforms.Compose(t_list)

def build_dataloaders(cfg):
    # Build query and reference transforms from the config using torchvision transforms
    ground_transforms = build_transforms(cfg.dataset.ground_transforms)
    satellite_transforms = build_transforms(cfg.dataset.satellite_transforms)

    # Create the training dataset and dataloader
    train_dataset = VigorDataset(
        data_folder=cfg.dataset.data_folder,
        split='train',
        same_area=cfg.dataset.same_area,
        ground_transforms=ground_transforms,
        satellite_transforms=satellite_transforms,
        use_captions=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    # If you have a validation split defined in the config and want a val loader:
    if hasattr(cfg.dataset, 'val_split') and cfg.dataset.val_split:
        val_dataset = VigorDataset(
            data_folder=cfg.dataset.data_folder,
            split='val',
            same_area=cfg.dataset.same_area,
            ground_transforms=ground_transforms,
            satellite_transforms=satellite_transforms,
            use_captions=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
