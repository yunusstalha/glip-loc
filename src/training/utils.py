# src/training/utils.py
import random
import torch
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clip_gradients(accelerator, model, max_norm: float):
    # accelerator gives correct handling in distributed/mixed precision scenario.
    params = [p for p in model.parameters() if p.requires_grad]
    accelerator.clip_grad_norm_(params, max_norm)

def predict(accelerator, config, model, dataloader):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Depending on your VigorDatasetEval, it might return (img, label)
            # For "reference" type: img, label
            # For "query" type: img, label
            imgs, lbls = batch  # Ensure dataset returns this format

            imgs = imgs.to(accelerator.device)

            # Extract embeddings
            emb = model(ground_image=imgs) # if query set returns ground image
            # OR model(satellite_image=imgs) if reference set is sat images
            # For simplicity, let's assume query_dataloader_eval returns ground images and reference_dataloader_eval returns satellite images.
            
            emb = accelerator.gather(emb)   # gather embeddings from all GPUs
            lbls = accelerator.gather(lbls) # gather labels from all GPUs

            features_list.append(emb.cpu())
            labels_list.append(lbls.cpu())

    # Concatenate all
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return features, labels
