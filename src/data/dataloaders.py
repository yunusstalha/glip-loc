# src/data/dataloaders.py
from torch.utils.data import DataLoader
from .datasets import VigorDatasetTrain, VigorDatasetEval
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(transform_cfg):
    """Build Albumentations transforms from config."""
    if transform_cfg is None:
        return None

    # A simple example: we expect a list of transforms in the config
    transforms_list = []
    for t in transform_cfg:
        t_type = t["type"]
        if t_type == "RandomHorizontalFlip":
            transforms_list.append(A.HorizontalFlip(p=0.5))
        elif t_type == "RandomCrop":
            transforms_list.append(A.RandomCrop(height=t["height"], width=t["width"]))
        elif t_type == "Resize":
            transforms_list.append(A.Resize(height=t["height"], width=t["width"]))
        elif t_type == "Normalize":
            transforms_list.append(A.Normalize(mean=t["mean"], std=t["std"]))
        elif t_type == "ToTensorV2":
            transforms_list.append(ToTensorV2())
        # Add other transforms as needed.

    return A.Compose(transforms_list)


def build_dataloaders(cfg):
    # Build query and reference transforms if needed
    query_transforms = build_transforms(cfg.dataset.query_transforms)
    reference_transforms = build_transforms(cfg.dataset.reference_transforms)

    if cfg.dataset.name == "VIGOR":
        if cfg.dataset.split == "train":
            # For training, we might only need a train dataset
            train_dataset = VigorDatasetTrain(
                data_folder=cfg.dataset.data_folder,
                same_area=cfg.dataset.same_area,
                transforms_query=query_transforms,
                transforms_reference=reference_transforms,
                prob_flip=0.0,          # could also come from config
                prob_rotate=0.0,        # could also come from config
                shuffle_batch_size=cfg.dataset.batch_size,
                use_cubemaps=cfg.dataset.use_cubemaps
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.dataset.batch_size,
                shuffle=True,           # shuffle training data
                num_workers=cfg.dataset.num_workers,
                pin_memory=True,
                drop_last=True
            )

            # If we have a validation dataset, we can create it similarly
            # by setting cfg.dataset.split = 'val' and img_type='query'/'reference'
            # For now, let's assume only train is needed.

            return train_loader, None

        else:
            # For evaluation, we might need a query dataset and a reference dataset
            query_dataset = VigorDatasetEval(
                data_folder=cfg.dataset.data_folder,
                split=cfg.dataset.split,
                img_type="query",             # from config if needed
                same_area=cfg.dataset.same_area,
                transforms=query_transforms,
                use_cubemaps=cfg.dataset.use_cubemaps
            )

            reference_dataset = VigorDatasetEval(
                data_folder=cfg.dataset.data_folder,
                split=cfg.dataset.split,
                img_type="reference",
                same_area=cfg.dataset.same_area,
                transforms=reference_transforms,
                use_cubemaps=cfg.dataset.use_cubemaps
            )

            query_loader = DataLoader(
                query_dataset,
                batch_size=cfg.dataset.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
                pin_memory=True
            )

            ref_loader = DataLoader(
                reference_dataset,
                batch_size=cfg.dataset.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
                pin_memory=True
            )

            return query_loader, ref_loader
    else:
        raise ValueError("Dataset not supported")
