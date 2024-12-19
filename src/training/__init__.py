# src/training/trainer.py
import torch
import time
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from typing import Optional, Dict

from .optimizers import create_optimizer
from .schedulers import create_scheduler
from .utils import set_seed, clip_gradients


class Trainer:
    def __init__(self, cfg, model, train_dataset, val_dataset, logger=None):
        """
        Args:
            cfg: Configuration object (parsed from YAML or similar).
            model (nn.Module): The model to train.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset, for frequent validation metric.
            logger: An optional logger (e.g., a WandbLogger) for custom logging. If None and cfg.wandb.enabled, use wandb directly.
        """
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = logger

        # Set random seeds for reproducibility
        set_seed(cfg.training.seed)

        # Create DataLoaders
        self.train_loader = self._create_dataloader(train_dataset, cfg.dataset.batch_size, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, cfg.dataset.batch_size, shuffle=False)

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(cfg, model)
        self.scheduler = create_scheduler(cfg, self.optimizer)

        # Setup accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16" if cfg.accelerate.fp16 else "no",
            # Add additional parameters if needed, e.g. cpu, deepspeed_config
        )

        # Wrap model, optimizer, and dataloaders
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Initialize W&B if enabled and logger is None
        if cfg.wandb.enabled and logger is None:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=cfg.__dict__
            )
            self.logger = wandb

        # Hooks (optional)
        # self.hooks = []  # Add custom hooks if needed

    def _create_dataloader(self, dataset, batch_size, shuffle):
        # If you have a custom sampler, integrate it here. For example:
        # sampler = CustomSampler(dataset, cfg.dataset.shuffle_batch_size) if needed
        # return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=cfg.dataset.num_workers)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.cfg.dataset.num_workers)

    def run(self):
        """Run the full training loop."""
        for epoch in range(self.cfg.training.num_epochs):
            self.epoch = epoch
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            
            # Scheduler step after validation if that is your strategy
            if self.scheduler is not None:
                self.scheduler.step()

            # Log results
            if self.logger:
                self.logger.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, step=self.global_step)

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")

            # Occasionally run a full retrieval evaluation
            if epoch % self.cfg.training.eval_recall_every == 0:
                # This can be a very expensive operation, e.g. compute recall@K on full test set
                recall = self.evaluate_recall()
                if self.logger:
                    self.logger.log({"recall": recall}, step=self.global_step)

            # Maybe save epoch checkpoint
            if epoch % self.cfg.training.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Extract batch data
            # For example, if the dataset returns ground_img, sat_img, label, ground_txt, sat_txt
            ground_img, sat_img, label, ground_txt, sat_txt = batch
            # Forward pass
            g_emb, s_emb, g_txt_emb, s_txt_emb = self.model(
                ground_image=ground_img, 
                satellite_image=sat_img,
                ground_captions=ground_txt,
                satellite_captions=sat_txt
            )

            # Compute loss (define your loss function or pass through config)
            loss = self.compute_loss(g_emb, s_emb, g_txt_emb, s_txt_emb, label)
            
            # Backward pass with accelerate
            self.accelerator.backward(loss)

            # Gradient clipping
            if self.cfg.training.grad_clip.enabled:
                clip_gradients(self.accelerator, self.model, self.cfg.training.grad_clip.max_norm)

            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Logging intermediate steps
            if self.logger and self.global_step % self.cfg.training.log_every == 0:
                self.logger.log({"train_loss": loss.item()}, step=self.global_step)

            # Warm-up or other steps can be integrated here

        avg_loss = total_loss / (batch_idx + 1)
        end = time.time()
        print(f"Epoch {self.epoch} train loss: {avg_loss:.4f} - Time: {end - start:.2f}s")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                ground_img, sat_img, label, ground_txt, sat_txt = batch
                g_emb, s_emb, g_txt_emb, s_txt_emb = self.model(
                    ground_image=ground_img, 
                    satellite_image=sat_img,
                    ground_captions=ground_txt,
                    satellite_captions=sat_txt
                )
                loss = self.compute_loss(g_emb, s_emb, g_txt_emb, s_txt_emb, label)
                total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {self.epoch} val loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_recall(self):
        # Placeholder for full dataset retrieval evaluation
        # This might involve running inference on the entire test set,
        # extracting all embeddings, and computing recall@K.
        # Because this is expensive, done less frequently.
        recall = 0.0  # Implement the logic or call a separate function
        return recall

    def compute_loss(self, g_emb, s_emb, g_txt_emb, s_txt_emb, label):
        # Implement your loss. For example, contrastive loss between ground and satellite embeddings.
        # Or you can combine text embeddings as well.
        # This is a placeholder:
        loss = torch.tensor(0.0, device=g_emb.device)
        return loss

    def save_checkpoint(self, filename):
        if self.accelerator.is_main_process:
            checkpoint_path = f"{self.cfg.training.checkpoint_dir}/{filename}"
            state = {
                "model": self.accelerator.get_state_dict(self.model),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss
            }
            torch.save(state, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
