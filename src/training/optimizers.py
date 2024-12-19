import torch

def create_optimizer(cfg, model):
    """
    Create and return an optimizer based on the config.
    """
    opt_type = cfg.training.optimizer.type.lower()
    lr = cfg.training.optimizer.lr
    weight_decay = cfg.training.optimizer.weight_decay

    # Collect parameters that need optimization
    parameters = model.parameters()

    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = getattr(cfg.training.optimizer, "momentum", 0.9)
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    return optimizer
