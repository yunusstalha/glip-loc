# src/training/trainer.py
import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Any

from .optimizers import create_optimizer
from .schedulers import create_scheduler
from .utils import clip_gradients

import torch.nn as nn
import torch.nn.functional as F

 
class Trainer:
    def __init__(self, cfg: Any, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, sim_dict=None, accelerator=None):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sim_dict = sim_dict
        self.accelerator = accelerator

        # 1) Create optimizer
        self.optimizer = create_optimizer(cfg, model)

        # 2) Conditionally create scheduler
        self.scheduler = None
        if hasattr(cfg.training, "scheduler") and cfg.training.scheduler and getattr(cfg.training.scheduler, "type", None):
            print("Creating scheduler")
            self.scheduler = create_scheduler(cfg, self.optimizer)

        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # 3) Ensure checkpoint dir exists
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)

    def run(self):
        num_epochs = self.cfg.training.num_epochs

        if self.sim_dict is not None:
            self.train_loader.dataset.shuffle(sim_dict=self.sim_dict, neighbour_select=32, neighbour_range=64)

        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            # 4) Step scheduler if it exists
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            self.accelerator.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=self.global_step)

            # Checkpoint if better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")

            # Save periodic checkpoint
            if (epoch + 1) % self.cfg.training.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            # Shuffle again if sim_dict
            if self.sim_dict is not None:
                self.train_loader.dataset.shuffle(sim_dict=self.sim_dict, neighbour_select=64, neighbour_range=128)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            ground_img, sat_img, label, ground_txt, sat_txt = batch

            if self.cfg.model.use_text:
                g_emb, s_emb, g_txt_emb, s_txt_emb, temp = self.model(
                    ground_image=ground_img,
                    satellite_image=sat_img,
                    ground_captions=ground_txt,
                    satellite_captions=sat_txt
                )
                loss = self.compute_loss(g_emb, s_emb, g_txt_emb, s_txt_emb, label, temp)
            else:
                g_emb, s_emb, temp = self.model(
                    ground_image=ground_img,
                    satellite_image=sat_img
                )
                loss = self.compute_loss(g_emb, s_emb, None, None, label, temp)

            print(f"Batch {batch_idx}, Loss = {loss.item()}")  # (You asked for loss printing)
            with torch.no_grad():
                print("Ground embedding mean:", g_emb.mean().item())
                print("Sat embedding mean:", s_emb.mean().item())

            # Backward
            self.accelerator.backward(loss)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(f"{name} grad mean: {param.grad.mean().item():.6f}")
                break  
            # 5) Conditionally clip gradients
            use_grad_clip = hasattr(self.cfg.training, "grad_clip") and self.cfg.training.grad_clip.enabled
            if use_grad_clip:
                clip_gradients(self.accelerator, self.model, self.cfg.training.grad_clip.max_norm)

            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Intermediate logging
            if self.global_step % self.cfg.training.log_every == 0:
                self.accelerator.log({"train_loss": loss.item()}, step=self.global_step)

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
                    g_emb, s_emb, g_txt_emb, s_txt_emb, temp = self.model(
                        ground_image=ground_img,
                        satellite_image=sat_img,
                        ground_captions=ground_txt,
                        satellite_captions=sat_txt
                    )
                    loss = self.compute_loss(g_emb, s_emb, g_txt_emb, s_txt_emb, label, temp)
                else:
                    g_emb, s_emb, temp = self.model(
                        ground_image=ground_img,
                        satellite_image=sat_img
                    )
                    loss = self.compute_loss(g_emb, s_emb, None, None, label, temp)

                total_loss += loss.item()

        avg_val_loss = total_loss / (batch_idx + 1)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def evaluate_recall(self):
        return 0.0  # Placeholder

    def compute_loss(self, g_emb, s_emb, g_txt_emb, s_txt_emb, label, temp):
        # # 6) Optional label smoothing
        # import torch.nn.functional as F

        # g_emb = F.normalize(g_emb, dim=-1)
        # s_emb = F.normalize(s_emb, dim=-1)
        # g_emb_all = self.accelerator.gather(g_emb)
        # s_emb_all = self.accelerator.gather(s_emb)

        # logits = g_emb_all @ s_emb_all.T
        # batch_size = g_emb_all.size(0)
        # targets = torch.arange(batch_size, device=g_emb_all.device)

        # label_smoothing = getattr(self.cfg.training, 'label_smoothing', 0.0)
        # print(f"Label Smoothing: {label_smoothing}")
        # if smooth:
        #     loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        g_emb = F.normalize(g_emb, dim=-1)
        s_emb = F.normalize(s_emb, dim=-1)
        
        logits = temp * g_emb @ s_emb.T        
        labels = torch.arange(len(logits), dtype=torch.long, device=g_emb.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels))/2


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
