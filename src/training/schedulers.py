import torch

def create_scheduler(cfg, optimizer):
    """
    Create and return a learning rate scheduler based on config.
    """
    if not hasattr(cfg.training, 'scheduler') or cfg.training.scheduler.type is None:
        return None

    sched_type = cfg.training.scheduler.type.lower()

    if sched_type == "step":
        step_size = cfg.training.scheduler.step_size
        gamma = cfg.training.scheduler.gamma
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif sched_type == "cosine_annealing":
        T_max = cfg.training.scheduler.T_max
        eta_min = getattr(cfg.training.scheduler, "eta_min", 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif sched_type == "multistep":
        milestones = cfg.training.scheduler.milestones
        gamma = cfg.training.scheduler.gamma
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    else:
        # No known scheduler
        return None

    return scheduler
