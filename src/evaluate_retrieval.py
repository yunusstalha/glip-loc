#!/usr/bin/env python3
# src/eval_vigor.py

import argparse
import torch
from torch.utils.data import DataLoader
import time
import gc
import copy
import numpy as np
from tqdm import tqdm

from data.vigor_plus import VigorDataset
from models.glip_loc import GLIPLocModel  # Or whichever model you use
from training.utils import load_checkpoint  # e.g. from your trainer utils
from utils.config_parser import load_config
import torchvision.transforms as T

##########################################
# 1) The "predict" function you rely on
##########################################
# def predict(config, model, dataloader):
#     """
#     Extract embeddings & labels from a dataloader in inference mode.
#     Returns:
#       features: (N, D) Tensor
#       labels:   (N, ...) Tensor
#     """
#     model.eval()
#     all_feats = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Extracting embeddings"):
#             # Depending on dataset_mode, the batch might be:
#             #  - reference mode => (sat_img, label, ...)
#             #  - query mode => (ground_img, label, ...)
#             # We'll assume the first 2 items are (images, labels).
#             images, labels = batch[0], batch[1]

#             # Move images to GPU if available
#             images = images.to(config.device)

#             # In "reference" mode => model(satellite_image=images)
#             # In "query" mode => model(ground_image=images)
#             # We'll do a generic approach: you can pass a param or check dataset_mode
#             if getattr(config, "eval_mode", "reference") == "reference":
#                 feats = model(satellite_image=images)
#             else:
#                 feats = model(ground_image=images)

#             # If model(...) returns a tuple, grab the first item
#             if isinstance(feats, (tuple, list)):
#                 feats = feats[0]

#             # Move feats & labels to CPU memory
#             all_feats.append(feats.cpu())
#             all_labels.append(labels.cpu())

#     # Concatenate
#     features = torch.cat(all_feats, dim=0)
#     labels = torch.cat(all_labels, dim=0)
#     return features, labels

def predict(config, model, dataloader):
    model.eval()
    num_samples = len(dataloader.dataset)
    feature_dim = 1024  # Assuming this is the feature size from the model
    
    # Pre-allocate tensors
    all_feats = torch.zeros((num_samples, feature_dim), dtype=torch.float32)
    if getattr(config, "eval_mode", "reference") == "reference":
        all_labels = torch.zeros((num_samples,), dtype=torch.long)
    else:
        all_labels = torch.zeros((num_samples, 4), dtype=torch.long)
    
    start_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images, labels = batch[0], batch[1]
            images = images.to(config.device)

            # Compute features
            if getattr(config, "eval_mode", "reference") == "reference":
                feats = model(satellite_image=images)
            else:
                feats = model(ground_image=images)

            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            
            # Determine batch size and copy data to pre-allocated tensors
            batch_size = labels.size(0)
            all_feats[start_idx : start_idx + batch_size] = feats.cpu()
            all_labels[start_idx : start_idx + batch_size] = labels.cpu()
            start_idx += batch_size

    return all_feats, all_labels

##########################################
# 2) The same "evaluate" + "calc_sim" logic
##########################################
def evaluate(config, model, reference_dataloader, query_dataloader, 
             ranks=[1, 5, 10], step_size=1000, cleanup=True):
    """
    From your reference code:
      1) Extract embeddings for reference & query
      2) Calculate final scores
      3) Cleanup memory
    """
    print("\nExtract Features:")
    # We'll set config.eval_mode="reference" or "query" to control predict
    config.eval_mode = "reference"
    reference_features, reference_labels = predict(config, model, reference_dataloader)

    config.eval_mode = "query"
    query_features, query_labels = predict(config, model, query_dataloader)

    print("Compute Scores:")
    r1 = calculate_scores(query_features, reference_features,
                          query_labels, reference_labels,
                          step_size=step_size, ranks=ranks)

    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1


def calc_sim(config, model, reference_dataloader, query_dataloader,
             ranks=[1, 5, 10], step_size=1000, cleanup=True):
    """
    Similarly, from your reference code, but also returns nearest_dict.
    """
    print("\nExtract Features:")
    config.eval_mode = "reference"
    reference_features, reference_labels = predict(config, model, reference_dataloader)

    config.eval_mode = "query"
    query_features, query_labels = predict(config, model, query_dataloader)

    print("Compute Scores Train:")
    r1 = calculate_scores_train(query_features, reference_features,
                                query_labels, reference_labels,
                                step_size=step_size, ranks=ranks)

    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.get("neighbour_range", 64),
                                  step_size=step_size)

    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1, near_dict


##########################################
# 3) Exactly your "calculate_scores" logic
##########################################
def calculate_scores(query_features, reference_features, 
                     query_labels, reference_labels, 
                     step_size=1000, ranks=[1, 5, 10]):

    # We do small chunking in "step_size" for the query dimension, build up a
    # QxR matrix in CPU memory -> same approach as your snippet
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)

    steps = Q // step_size + 1

    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    # Build ref2index
    ref2index = { idx : i for i, idx in enumerate(reference_labels_np) }

    # Build the full similarity on CPU
    similarity = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T  # GPU matmul if small, or CPU if already on CPU
        similarity.append(sim_tmp.cpu())
    similarity = torch.cat(similarity, dim=0)  # shape [Q, R]

    # We'll do topk plus R//100
    topk.append(R // 100 if R >= 100 else 1)
    results = np.zeros(len(topk), dtype=float)
    hit_rate = 0.0

    bar = tqdm(range(Q), desc="Ranking")
    for i in bar:
        # gt reference
        gt_sat = query_labels_np[i][0]  # the first col is the GT sat
        gt_idx = ref2index[gt_sat]
        gt_sim = similarity[i, gt_idx]

        # how many references exceed that similarity
        row = similarity[i]
        higher_sim = row > gt_sim
        ranking = higher_sim.sum().item()

        # update recall
        for j, K in enumerate(topk):
            if ranking < K:
                results[j] += 1

        # near positives => ignoring them in hit_rate
        mask = torch.ones(R, dtype=torch.bool)
        for near_sat in query_labels_np[i][1:]:
            mask[ref2index[near_sat]] = False

        # how many outrank gt among non-near-positives
        outrank_np = (higher_sim & mask).sum().item()
        if outrank_np < 1:
            hit_rate += 1

    results = results / Q * 100
    hit_rate = hit_rate / Q * 100

    bar.close()
    time.sleep(0.1)

    # Print
    s = []
    for i in range(len(topk) - 1):
        s.append(f"Recall@{topk[i]}: {results[i]:.4f}")
    s.append(f"Recall@top1: {results[-1]:.4f}")
    s.append(f"Hit_Rate: {hit_rate:.4f}")
    print(" - ".join(s))

    # Return the first metric (Recall@1) if you want, or anything else
    return results[0]


def calculate_scores_train(query_features, reference_features,
                           query_labels, reference_labels,
                           step_size=1000, ranks=[1, 5, 10]):
    """
    Same as your code but ignoring near-positives, using the first col only.
    """
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)

    steps = Q // step_size + 1
    # train queries might have shape [Q,4], but we only want col 0
    # per your original snippet:
    query_labels_np = query_labels[:, 0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = { idx : i for i, idx in enumerate(reference_labels_np) }

    similarity = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    similarity = torch.cat(similarity, dim=0)

    topk.append(R // 100 if R >= 100 else 1)
    results = np.zeros(len(topk), dtype=float)

    bar = tqdm(range(Q), desc="RankingTrain")
    for i in bar:
        gt_idx = ref2index[ query_labels_np[i] ]
        gt_sim = similarity[i, gt_idx]

        row = similarity[i]
        higher_sim = row > gt_sim
        ranking = higher_sim.sum().item()
        for j, K in enumerate(topk):
            if ranking < K:
                results[j] += 1

    results = results / Q * 100
    bar.close()
    time.sleep(0.1)

    s = []
    for i in range(len(topk)-1):
        s.append(f"Recall@{topk[i]}: {results[i]:.4f}")
    s.append(f"Recall@top1: {results[-1]:.4f}")
    print(" - ".join(s))

    return results[0]


def calculate_nearest(query_features, reference_features,
                      query_labels, reference_labels,
                      neighbour_range=64, step_size=1000):
    """
    Builds a QxR similarity, then picks top K references for each query,
    ignoring the GT if found (like your snippet).
    """
    query_labels = query_labels[:,0]  # [Q]
    Q = len(query_features)
    steps = Q // step_size + 1

    similarity = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    similarity = torch.cat(similarity, dim=0)  # shape [Q, R]

    # get topk for each row
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)

    # Build final references
    near_dict = {}
    # We'll map each row's topk_ids to actual reference_labels
    reference_labels_cpu = reference_labels.cpu()

    for i in range(len(topk_ids)):
        row_ids = topk_ids[i]  # shape [neighbour_range+2]
        row_labels = reference_labels_cpu[row_ids]  # shape [neighbour_range+2]
        # Exclude the ground-truth ID from that row
        mask = (row_labels != query_labels[i].item())
        # keep up to 'neighbour_range' that are not the GT
        near_ids = row_labels[mask][:neighbour_range]
        near_dict[ query_labels[i].item() ] = near_ids.tolist()

    return near_dict


##########################################
# 4) A simple main function
##########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--same_area", action="store_true", help="Override config to same_area if set.")
    parser.add_argument("--step_size", type=int, default=1000, help="Chunk size for queries.")
    args = parser.parse_args()
    batch_size = 128
    num_workers = 12
    # 1) Load config
    cfg = load_config(args.config)
    # You might define: cfg.device = "cuda" if available, else "cpu"
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # # 2) Optionally override same_area
    # if args.same_area:
    #     cfg.dataset.same_area = True

    # 3) Build model
    model = GLIPLocModel(cfg.model.name, pretrained=False, use_text=False)
    model.to(cfg.device)

    # 4) Load checkpoint if provided
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, strict=True)


    # Minimal eval transforms
    sat_transforms = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    ground_transforms = T.Compose([
        T.Resize((384, 768)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    reference_dataset = VigorDataset(
        data_folder=cfg.dataset.data_folder,
        split="test",
        same_area=cfg.dataset.same_area,
        dataset_mode="reference",
        satellite_transforms=sat_transforms,
        prob_flip=0.0, prob_rotate=0.0,
        use_captions=False
    )
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 6) Build query dataloader
    query_dataset = VigorDataset(
        data_folder=cfg.dataset.data_folder,
        split="test",
        same_area=cfg.dataset.same_area,
        dataset_mode="query",
        ground_transforms=ground_transforms,
        prob_flip=0.0, prob_rotate=0.0,
        use_captions=False
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 7) Run evaluate (or calc_sim)
    print("Running evaluation:")
    _ = evaluate(config=cfg,
                 model=model,
                 reference_dataloader=reference_loader,
                 query_dataloader=query_loader,
                 ranks=[1,5,10],
                 step_size=args.step_size,
                 cleanup=True)

if __name__ == "__main__":
    main()