# src/training/trainer.py
import os
import time
import torch
import wandb
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from typing import Optional, Any

from .optimizers import create_optimizer
from .schedulers import create_scheduler
from .utils import clip_gradients

class Trainer:
    def __init__(self, cfg: Any, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, logger=None, sim_dict=None, accelerator=None):
        """
        Args:
            cfg: Configuration object.
            model (nn.Module): The model to train.
            train_loader (DataLoader): Training dataloader.
            val_loader (DataLoader): Validation dataloader for frequent metric checks.
            logger: A logger object (e.g. WandbLogger). If None and cfg.wandb.enabled, wandb is initialized here.
            sim_dict: A dictionary containing similarity scores for data mining.
            accelerator: The Accelerator instance for distributed/mixed-precision training.

        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.sim_dict = sim_dict
        self.accelerator = accelerator


        # Create optimizer & scheduler
        self.optimizer = create_optimizer(cfg, model)
        self.scheduler = create_scheduler(cfg, self.optimizer)


        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Initialize W&B if needed
        if cfg.wandb.enabled and self.logger is None:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=cfg.__dict__
            )
            self.logger = wandb

        # Checkpoint directory
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    def run(self):
        """Run the full training loop."""
        num_epochs = self.cfg.training.num_epochs
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            # Step scheduler if needed
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            if self.logger is not None:
                self.logger.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, step=self.global_step)

            # Checkpoint on improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")

            # Occasionally run full retrieval evaluation
            if (epoch + 1) % self.cfg.training.eval_recall_every == 0:
                recall = self.evaluate_recall()
                if self.logger is not None:
                    self.logger.log({"recall": recall}, step=self.global_step)

            # Save periodic checkpoint
            if (epoch + 1) % self.cfg.training.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Custom sampling for data mining
            if self.sim_dict is not None:    
                self.train_loader.dataset.shuffle(sim_dict=self.sim_dict, neighbour_select=64, neighbour_range=128)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Extract batch data
            ground_img, sat_img, label, ground_txt, sat_txt = batch

            # Forward pass with or without captions based on `use_text`
            if self.cfg.model.use_text:
                g_emb, s_emb, g_txt_emb, s_txt_emb = self.model(
                    ground_image=ground_img,
                    satellite_image=sat_img,
                    ground_captions=ground_txt,
                    satellite_captions=sat_txt
                )
            else:
                g_emb, s_emb = self.model(
                    ground_image=ground_img,
                    satellite_image=sat_img
                )


            # Compute loss
            loss = self.compute_loss(g_emb, s_emb, g_txt_emb, s_txt_emb, label)

            # Backward
            self.accelerator.backward(loss)

            # Gradient clipping
            if self.cfg.training.grad_clip.enabled:
                clip_gradients(self.accelerator, self.model, self.cfg.training.grad_clip.max_norm)

            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Intermediate logging
            if self.logger is not None and self.global_step % self.cfg.training.log_every == 0:
                self.logger.log({"train_loss": loss.item(), "step": self.global_step})

            # Warm-up logic if needed (depends on scheduler or separate step)
            # For example, if you have a warmup scheduler step per iteration:
            # if self.cfg.training.warmup.enabled and current_step < warmup_steps:
            #     warmup_scheduler.step()

        avg_loss = total_loss / (batch_idx + 1)
        elapsed = time.time() - start_time
        print(f"Epoch {self.epoch+1}/{self.cfg.training.num_epochs} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                ground_img, sat_img, label, ground_txt, sat_txt = batch

                if self.cfg.model.use_text:
                    g_emb, s_emb, g_txt_emb, s_txt_emb = self.model(
                        ground_image=ground_img,
                        satellite_image=sat_img,
                        ground_captions=ground_txt,
                        satellite_captions=sat_txt
                    )
                    loss = self.compute_loss(g_emb, s_emb, g_txt_emb, s_txt_emb, label)
                else:
                    g_emb, s_emb = self.model(
                        ground_image=ground_img,
                        satellite_image=sat_img
                    )
                    loss = self.compute_loss(g_emb, s_emb, None, None, label)

                total_loss += loss.item()

        avg_val_loss = total_loss / (batch_idx + 1)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def evaluate_recall(self):
        # Placeholder for a full retrieval metric
        # For a full dataset retrieval:
        # 1. Gather all ground embeddings and satellite embeddings for the test set.
        # 2. Compute pairwise similarities and determine recall@K.
        # This is expensive, so done infrequently.
        recall = 0.0
        # Implement logic or call separate function
        return recall

    def compute_loss(self, g_emb, s_emb, g_txt_emb, s_txt_emb, label):
        # If not using text, ignore g_txt_emb, s_txt_emb.
        # Normalize embeddings
        g_emb = torch.nn.functional.normalize(g_emb, dim=-1)
        s_emb = torch.nn.functional.normalize(s_emb, dim=-1)

        # Gather from all processes if distributed
        g_emb_all = self.accelerator.gather(g_emb)
        s_emb_all = self.accelerator.gather(s_emb)

        # Now g_emb_all and s_emb_all contain embeddings from all GPUs
        # Compute similarity
        # g_emb_all: [B_total, D], s_emb_all: [B_total, D]
        logits = g_emb_all @ s_emb_all.T  # [B_total, B_total]

        # Temperature
        temp = torch.exp(self.model.temperature)  # get actual temperature
        # Targets
        batch_size = g_emb_all.size(0)
        targets = torch.arange(batch_size, device=g_emb_all.device)

        # We can apply label smoothing if desired:
        # CrossEntropy with label smoothing defined once outside or here:
        # Example: loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)
        # If label_smoothing is defined in cfg:
        label_smoothing = getattr(self.cfg.training, 'label_smoothing', 0.1)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ground->sat
        loss1 = loss_fn(logits / temp, targets)
        # sat->ground (symmetric)
        loss2 = loss_fn(logits.T / temp, targets)

        loss = (loss1 + loss2) / 2.0
        return loss


    def save_checkpoint(self, filename):
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.cfg.training.checkpoint_dir, filename)
            state = {
                "model": self.accelerator.get_state_dict(self.model),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.cfg.__dict__
            }
            torch.save(state, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
