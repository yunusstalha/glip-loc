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


def predict_embeddings(
    model,
    dataloader,
    mode:str,
    accelerator=None,
    chunk_size:int=512
):
    """
    Extract embeddings for all samples in the dataloader in a memory-friendly manner.
    
    Args:
      model: Your GLIPLocModel or any model returning embeddings.
      dataloader: yields either (img, label, ...) depending on mode.
      mode: "reference" or "query" – we need to pass the right argument to model(...).
      accelerator: optional, for distributed inference.
      chunk_size: number of queries to process at once for matrix multiplication if needed.
    Returns:
      all_embeddings: [N, embed_dim]
      all_labels: [N, ...]
    """
    model.eval()
    device = accelerator.device if accelerator else "cuda"
    all_embeddings = []
    all_labels = []

    # No grad
    with torch.no_grad():
        for batch in dataloader:
            # batch could be (img, label, caption, _, _)
            if mode == "reference":
                images, labels = batch[0], batch[1]
                images = images.to(device)
                # get embedding
                emb = model(satellite_image=images)
                if isinstance(emb, (list, tuple)):
                    emb = emb[0]  # if model returns tuple
                if accelerator:
                    emb = accelerator.gather(emb)
                    labels = accelerator.gather(labels)
                else:
                    emb = emb
                    labels = labels
                all_embeddings.append(emb.cpu())
                all_labels.append(labels.cpu())

            elif mode == "query":
                # query => (ground_img, label[4], caption, _, _)
                images, labels = batch[0], batch[1]
                images = images.to(device)
                emb = model(ground_image=images)
                if isinstance(emb, (list, tuple)):
                    emb = emb[0]
                if accelerator:
                    emb = accelerator.gather(emb)
                    labels = accelerator.gather(labels)
                all_embeddings.append(emb.cpu())
                all_labels.append(labels.cpu())

            else:
                raise ValueError("predict_embeddings only supports 'reference' or 'query' mode now.")

    # Concat
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Optionally normalize
    # all_embeddings = F.normalize(all_embeddings, dim=-1)

    return all_embeddings, all_labels

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