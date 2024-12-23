# src/training/trainer.py
import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Any

from .optimizers import create_optimizer
from .schedulers import create_scheduler
from .utils import clip_gradients

import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function

from transformers import (
    get_cosine_schedule_with_warmup,
)

def generate_run_name(cfg):
    import time
    run_name = f"{cfg.model.name}_lr{cfg.training.optimizer.lr}_{time.strftime('%Y%m%d_%H%M%S')}"
    return run_name


class AllGatherWithGrad(Function):
    """
    All-gather a tensor from each rank to form a larger tensor, while preserving gradients.

    The input tensor must be [local_batch, ...].
    The returned tensor will be [global_batch, ...] where global_batch = local_batch * world_size.
    """

    @staticmethod
    def forward(ctx, tensor):
        """
        Expect `tensor` shape: [local_batch, ...].
        We'll create an output shape: [local_batch * world_size, ...].
        """
        if not dist.is_available() or not dist.is_initialized():
            # Single GPU case, or no distributed
            return tensor
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # The local batch size
        local_batch = tensor.shape[0]
        # The total batch across all ranks
        total_batch = local_batch * world_size

        # We'll store the slice range for backward
        start_idx = local_batch * rank
        ctx.start_idx = start_idx
        ctx.local_batch = local_batch
        ctx.world_size = world_size

        # Prepare an output buffer
        out_shape = list(tensor.shape)
        out_shape[0] = total_batch
        output = tensor.new_empty(out_shape)

        # Split that buffer into world_size chunks along dim=0
        chunks = list(output.split(local_batch, dim=0))  # each chunk is local_batch in size

        # All-gather across ranks
        dist.all_gather(chunks, tensor)
        # `chunks[rank]` is now the local portion; but we effectively filled the entire `output`

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output is [total_batch, ...], we want to slice out this rank's portion.
        """
        start = ctx.start_idx
        length = ctx.local_batch

        # The slice of grad_output that belongs to this rank
        grad_input = grad_output.narrow(0, start, length)

        return grad_input

class Trainer:
    def __init__(self, cfg: Any, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, sim_dict=None, accelerator=None):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sim_dict = sim_dict
        self.accelerator = accelerator

        self.run_name = generate_run_name(cfg)


        if self.cfg.wandb.enabled and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.cfg.wandb.project,
                config=vars(self.cfg),
                init_kwargs={
                    "entity": self.cfg.wandb.entity,
                    "name": self.run_name
                    
                }
            )
        # 1) Create optimizer
        self.optimizer = create_optimizer(cfg, model)

        # # 2) Conditionally create scheduler
        # self.scheduler = None
        # if hasattr(cfg.training, "scheduler") and cfg.training.scheduler and getattr(cfg.training.scheduler, "type", None):
        #     print("Creating scheduler")
        #     self.scheduler = create_scheduler(cfg, self.optimizer)

        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        num_training_steps = len(self.train_loader) * cfg.training.num_epochs
        warmup_steps = len(self.train_loader) * 1  # 1 epoch warmup
        
        # We'll override self.scheduler with huggingface schedule
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
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


            # Logging
            if self.accelerator.is_main_process:
                self.accelerator.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=self.global_step)

            if self.accelerator.is_main_process:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.detach().float().mean().item()
                        # Log the grad mean
                        self.accelerator.log({f"grad_mean/{name}": grad_mean}, step=self.global_step)

            # Checkpoint if better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")

            # Save periodic checkpoint
            if (epoch + 1) % self.cfg.training.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

            # Shuffle again if sim_dict
            if self.sim_dict is not None:
                self.train_loader.dataset.shuffle(sim_dict=self.sim_dict, neighbour_select=32, neighbour_range=64)

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

            # Backward
            self.accelerator.backward(loss)

            # 5) Conditionally clip gradients
            use_grad_clip = hasattr(self.cfg.training, "grad_clip") and self.cfg.training.grad_clip.enabled
            if use_grad_clip:
                clip_gradients(self.accelerator, self.model, self.cfg.training.grad_clip.max_norm)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            self.global_step += 1

            if self.accelerator.is_main_process:
                print(f'Loss: {loss.item()}, LR: {self.optimizer.param_groups[0]["lr"]}, Global Step: {self.global_step}, Epoch: {self.epoch+1}/{self.cfg.training.num_epochs}')
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

    # def compute_loss(self, g_emb, s_emb, g_txt_emb, s_txt_emb, label, temp):
    #     g_emb = F.normalize(g_emb, dim=-1)
    #     s_emb = F.normalize(s_emb, dim=-1)
        
    #     logits = temp * g_emb @ s_emb.T        
    #     labels = torch.arange(len(logits), dtype=torch.long, device=g_emb.device)         
    #     label_smoothing = getattr(self.cfg.training, 'label_smoothing', 0.0)  # default 0   
    #     loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    #     loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels))/2


    #     return loss
    def compute_loss(self, ground_embedding, satellite_embedding, g_txt_emb, s_txt_emb,label,  temperature):
        # Normalize features
        ground_embedding = F.normalize(ground_embedding, dim=-1)
        satellite_embedding = F.normalize(satellite_embedding, dim=-1)

        # Gather features from all devices
        all_ground_embedding = AllGatherWithGrad.apply(ground_embedding)
        all_satellite_embedding = AllGatherWithGrad.apply(satellite_embedding)
        global_batch_size = all_ground_embedding.size(0)
        # print('Requires Grad')
        # print(all_ground_embedding.requires_grad)
        # Compute logits
        logits_per_image = temperature * all_ground_embedding @ all_satellite_embedding.T
        logits_per_text = logits_per_image.T


        # labels = torch.arange(batch_size, device=g_emb.device)
        global_labels = torch.arange(global_batch_size, device=logits_per_image.device)

        # Compute loss
        label_smoothing = getattr(self.cfg.training, 'label_smoothing', 0.0)  # default 0   
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss_image_to_text = loss_fn(logits_per_image, global_labels)
        loss_text_to_image = loss_fn(logits_per_text, global_labels)
        loss = (loss_image_to_text + loss_text_to_image) / 2
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
