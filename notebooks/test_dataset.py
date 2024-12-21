import sys
import os
from pathlib import Path

# If needed, add the project root to sys.path so we can import from src
project_root = Path(os.getcwd()).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root / 'src'))

from data.vigor_plus import VigorDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
ground_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

satellite_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
data_folder = "/home/erzurumlu.1/yunus/research_drive/data/VIGOR"  # Adjust this to your actual data directory
dataset = VigorDataset(
    data_folder=data_folder,
    split='train',         # or 'val', depending on what you want to test
    same_area=False,        # set to False if you want to test cross-area splits
    ground_transforms=ground_transforms,
    satellite_transforms=satellite_transforms,
    use_captions=True
)
import random
import numpy as np
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_random_seed(42)
import pickle
with open("/home/erzurumlu.1/yunus/research_drive/data/VIGOR/gps_dict_cross.pkl", "rb") as f:
    sim_dict = pickle.load(f)

dataset.shuffle(sim_dict=sim_dict, neighbour_select=64, neighbour_range=128)