import torch
import time
import gc
import copy
import numpy as np
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, step_size=1000, ranks=[1, 5, 10], cleanup=True):
        self.model = model
        self.step_size = step_size
        self.ranks = ranks
        self.cleanup = cleanup

    # def predict(self, dataloader, mode='query'):
    #     self.model.eval()
    #     device = self.model.device
    #     num_samples = len(dataloader.dataset)
    #     feature_dim = 1024  # TODO: Automatically get this from model

    #     # Pre-allocate tensors
    #     all_feats = torch.zeros((num_samples, feature_dim), dtype=torch.float32)
    #     if mode == "reference":
    #         all_labels = torch.zeros((num_samples,), dtype=torch.long)
    #     else:
    #         all_labels = torch.zeros((num_samples, 4), dtype=torch.long)

    #     start_idx = 0
    #     with torch.no_grad():
    #         for batch in tqdm(dataloader, desc="Extracting embeddings"):
    #             images, labels = batch[0], batch[1]
    #             images = images.to(device)

    #             # Compute features
    #             if mode == "reference":
    #                 feats = self.model(satellite_image=images)
    #             else:
    #                 feats = self.model(ground_image=images)

    #             if isinstance(feats, (tuple, list)):
    #                 feats = feats[0]

    #             # Determine batch size and copy data to pre-allocated tensors
    #             batch_size = labels.size(0)
    #             all_feats[start_idx : start_idx + batch_size] = feats.cpu()
    #             all_labels[start_idx : start_idx + batch_size] = labels.cpu()
    #             start_idx += batch_size

    #     return all_feats, all_labels
    # In your evaluate.py or a new file
    def distributed_predict_embeddings(accelerator, model, dataloader, mode="query"):
        """
        Distributed extraction of embeddings.
        - Each GPU processes part of the dataloader.
        - We gather all embeddings & labels to the CPU (rank=0).
        - Returns (embeddings, labels) on rank=0, else (None, None) on other ranks.
        """
        model.eval()
        device = accelerator.device

        all_embs_list = []
        all_labels_list = []
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[0], batch[1]
                images = images.to(device)

                # Forward pass
                if mode == "reference":
                    feats = model(satellite_image=images)
                else:
                    feats = model(ground_image=images)

                if isinstance(feats, (list, tuple)):
                    feats = feats[0]  # If model returns a tuple

                # Gather for all ranks
                gathered_feats = accelerator.gather_for_metrics(feats)
                gathered_labels = accelerator.gather_for_metrics(labels)

                # On each rank we get the portion that belongs to that rank; 
                # but `gather_for_metrics()` accumulates it so that 
                # only rank=0 returns the full set after the loop ends.
                # We'll store them in a local list *only if we are rank=0*
                if accelerator.is_main_process:
                    all_embs_list.append(gathered_feats.cpu())
                    all_labels_list.append(gathered_labels.cpu())

        # rank=0 merges the list of all embeddings
        if accelerator.is_main_process:
            all_embs = torch.cat(all_embs_list, dim=0)
            all_labels = torch.cat(all_labels_list, dim=0)
            return all_embs, all_labels
        else:
            return None, None

    def calculate_scores(self, query_features, reference_features, query_labels, reference_labels):
        topk = copy.deepcopy(self.ranks)
        Q = len(query_features)
        R = len(reference_features)

        steps = Q // self.step_size + 1

        query_labels_np = query_labels.cpu().numpy()
        reference_labels_np = reference_labels.cpu().numpy()

        # Build ref2index
        ref2index = {idx: i for i, idx in enumerate(reference_labels_np)}

        # Build the full similarity on CPU
        similarity = []
        for i in range(steps):
            start = self.step_size * i
            end = start + self.step_size
            sim_tmp = query_features[start:end] @ reference_features.T
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

        return results, hit_rate

    def evaluate(self, reference_dataloader, query_dataloader):
        print("\nExtract Features:")
        reference_features, reference_labels = self.predict(reference_dataloader, mode="reference")
        query_features, query_labels = self.predict(query_dataloader, mode="query")

        print("Compute Scores:")
        results, hit_rate = self.calculate_scores(query_features, reference_features, query_labels, reference_labels)

        if self.cleanup:
            del reference_features, reference_labels, query_features, query_labels
            gc.collect()

        return results, hit_rate
