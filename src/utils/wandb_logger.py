# src/utils/wandb_logger.py
import wandb
from datetime import datetime

class WandbLogger:
    def __init__(self, cfg):
        self.enabled = cfg.wandb.enabled
        self.cfg = cfg
        if self.enabled:
            run_name = self.generate_run_name(cfg)
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                config=cfg.__dict__
            )

    def generate_run_name(self, cfg):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_part = cfg.model.name
        dataset_part = cfg.dataset.name
        lr_part = f"lr{cfg.training.optimizer.lr}"
        return f"{model_part}_{dataset_part}_{lr_part}_{timestamp}"

    def log_metrics(self, metrics_dict, step=None):
        if self.enabled:
            wandb.log(metrics_dict, step=step)

    def log_gradients(self, model, step=None):
        if self.enabled:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Convert gradient to numpy
                    grad_np = param.grad.detach().cpu().numpy()
                    # You might want to log just a histogram of gradients
                    wandb.log({f"gradients/{name}": wandb.Histogram(grad_np)}, step=step)

    def watch(self, model):
        if self.enabled:
            wandb.watch(model, log='all', log_freq=10)

