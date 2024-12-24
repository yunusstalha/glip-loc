# src/training/utils.py
import random
import torch
import numpy as np
import os 

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

def load_checkpoint(model, checkpoint_path, strict=True):
    """
    Loads model weights from a checkpoint.
    TODO: Add all other necessary components to the checkpoint for continuing the training from where we left.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state["model"], strict=strict)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return state