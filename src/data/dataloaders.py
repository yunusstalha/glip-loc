# src/data/dataloaders.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from .vigor_plus import VigorDataset

# def build_transforms(transform_cfg):
#     """Build torchvision transforms from config."""
#     if transform_cfg is None:
#         return transforms.Compose([
#             transforms.ToTensor()
#         ])

#     # This is a simple example. You might need to adjust according to your config structure.
#     t_list = []
#     for t in transform_cfg:
#         t_type = t.type
#         if t_type == "RandomHorizontalFlip":
#             t_list.append(transforms.RandomHorizontalFlip(p=0.5))
#         elif t_type == "RandomCrop":
#             t_list.append(transforms.RandomCrop((t.height, t.width)))
#         elif t_type == "Resize":
#             t_list.append(transforms.Resize((t.height, t.width)))
#         elif t_type == "Normalize":
#             t_list.append(transforms.Normalize(mean=t.mean, std=t.std))
#         elif t_type == "ToTensor":
#             t_list.append(transforms.ToTensor())

#     # Ensure we have a ToTensor at some point; if not already included, add at the end
#     if not any(isinstance(x, transforms.ToTensor) for x in t_list):
#         t_list.append(transforms.ToTensor())

#     return transforms.Compose(t_list)

def build_dataloaders(cfg):
    # Build query and reference transforms from the config using torchvision transforms
    # For ground images (384 x 768)
    ground_transforms = T.Compose([
        T.Resize((384, 768)),   
        T.ToTensor(),             # from your config
        T.ColorJitter(brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1),              # approximate color changes
        T.RandomApply([                      # approximate "blur" or "sharpen"
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ], p=0.3),
        # There's no built-in sharpen in TorchVision, so GaussianBlur is a partial sub
        T.RandomErasing(p=0.3, scale=(0.02, 0.1)),

        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # For satellite images (384 x 384)
    satellite_transforms = T.Compose([
        T.Resize((384, 384)),               # from your config
        T.ToTensor(),
        T.ColorJitter(brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1),
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ], p=0.3),
        T.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_ground_transforms = T.Compose([
        T.Resize((384, 768)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_satellite_transforms = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Create the training dataset and dataloader
    train_dataset = VigorDataset(
        data_folder=cfg.dataset.data_folder,
        split='train',
        same_area=cfg.dataset.same_area,
        ground_transforms=ground_transforms,
        satellite_transforms=satellite_transforms,
        use_captions=cfg.model.use_text,
        prob_rotate=cfg.dataset.prob_rotate,
        prob_flip=cfg.dataset.prob_flip,
        shuffle_batch_size=cfg.dataset.batch_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        drop_last=True
    )



    val_loader = None
    # If you have a validation split defined in the config and want a val loader:
    # if hasattr(cfg.dataset, 'val_split') and cfg.dataset.val_split:
    val_dataset = VigorDataset(
        data_folder=cfg.dataset.data_folder,
        split='test',
        same_area=cfg.dataset.same_area,
        ground_transforms=val_ground_transforms,
        satellite_transforms=val_satellite_transforms,
        use_captions=cfg.model.use_text,
        prob_rotate=0,
        prob_flip=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
