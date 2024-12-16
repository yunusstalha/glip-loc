# src/utils/config_parser.py
import yaml
from types import SimpleNamespace

def load_config(path: str):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    # Convert dict to a namespace-like object for easy access: cfg.dataset.name instead of cfg["dataset"]["name"]
    return dict_to_namespace(cfg_dict)

def dict_to_namespace(d):
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


if __name__ == "__main__":
    # Usage example
    cfg = load_config("config.yaml")
    print(cfg.dataset.name)
    print(cfg.dataset.path)
    print(cfg.dataset.batch_size)
    print(cfg.model.name)
    print(cfg.model.num_classes)
    print(cfg.model.num_layers)
    print(cfg.model.hidden_size)