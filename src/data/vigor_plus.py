# src/data/datasets.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class VigorDataset(Dataset):
    """
    A simplified version of VIGOR dataset class for training and validation.
    This version:
      - Uses torchvision transforms instead of Albumentations.
      - Omits cubemap code.
      - Handles train/val splits easily.
    """

    def __init__(self,
                 data_folder: str,
                 split: str = 'train',
                 same_area: bool = True,
                 transforms_query=None,
                 transforms_reference=None,
                 use_captions: bool = True):
        """
        Args:
            data_folder (str): Path to the VIGOR dataset directory.
            split (str): One of ['train', 'val']. Determines which split to load.
            same_area (bool): Whether to use the same_area splits for training/validation.
            transforms_query: Torchvision transforms for query (ground) images.
            transforms_reference: Torchvision transforms for reference (satellite) images.
            use_captions (bool): If True, load captions for ground and sat images.
        """
        super().__init__()
        self.data_folder = data_folder
        self.split = split
        self.same_area = same_area
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.use_captions = use_captions

        # Define cities based on same_area and split
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            if split == 'train':
                self.cities = ['NewYork', 'Seattle']
            else:  # val or test scenario
                self.cities = ['Chicago', 'SanFrancisco']

        # Load satellite list (global)
        self.df_sat = self._load_satellite_list()

        # Build mappings
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))

        # Load ground and satellite pairs
        self.df_ground = self._load_ground_split(sat2idx, self.split)

        # Create index mappings
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))

        # Prepare samples: (ground_idx, sat_idx)
        self.samples = list(zip(self.df_ground.index, self.df_ground.sat))

        # Load captions if needed
        if self.use_captions:
            self.ground_captions = self._load_captions('panorama_captions.csv', 'panorama')
            self.sat_captions = self._load_captions('satellite_captions.csv', 'satellite')
        else:
            self.ground_captions = {}
            self.sat_captions = {}

    def _load_satellite_list(self):
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/satellite_list.txt', 
                                 header=None, sep='\s+', names=['sat'])
            df_tmp["path"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat}', axis=1)
            df_tmp['city'] = city
            sat_list.append(df_tmp)
        return pd.concat(sat_list, axis=0).reset_index(drop=True)

    def _load_ground_split(self, sat2idx, split):
        # Choose the correct split file
        if self.same_area:
            split_file = f'same_area_balanced_{split}.txt'
        else:
            # Fallback if needed. In original code, if same_area=False and split=train, 
            # it uses 'pano_label_balanced.txt' but let's assume now we also have a val equivalent.
            # If not provided, we can revert to the original logic.
            if split == 'train':
                split_file = 'pano_label_balanced.txt'
            else:
                # If no val file is defined for non-same-area split, you must decide how to handle it.
                # Let's just reuse 'pano_label_balanced.txt' for val as well (not ideal, but placeholder).
                split_file = 'pano_label_balanced.txt'

        ground_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/{split_file}', 
                                 header=None, sep='\s+')
            # columns: ground, sat, sat_np1, sat_np2, sat_np3 at indices [0, 1, 4, 7, 10]
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={
                0: "ground",
                1: "sat",
                4: "sat_np1",
                7: "sat_np2",
                10: "sat_np3"
            })
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/panorama/{x.ground}', axis=1)

            # Map satellites to indices
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[sat_n] = df_tmp[sat_n].map(sat2idx)

            df_tmp['city'] = city
            ground_list.append(df_tmp)
        return pd.concat(ground_list, axis=0).reset_index(drop=True)

    def _load_captions(self, caption_file_name, img_type):
        # img_type is 'panorama' or 'satellite'
        # We'll load captions for all cities included.
        captions = {}
        for city in self.cities:
            caption_path = f'{self.data_folder}/{city}/{caption_file_name}'
            if os.path.exists(caption_path):
                df = pd.read_csv(caption_path)
                # df should have columns: filename, caption
                # filename matches either panorama or satellite images
                cap_dict = dict(zip(df['filename'], df['caption']))
                captions.update(cap_dict)
        return captions

    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]

        # Load images
        query_img = self._load_image(self.idx2ground_path[idx_ground])
        reference_img = self._load_image(self.idx2sat_path[idx_sat])

        # Apply transforms
        if self.transforms_query:
            query_img = self.transforms_query(query_img)
        if self.transforms_reference:
            reference_img = self.transforms_reference(reference_img)

        # Get captions
        ground_filename = self.idx2ground[idx_ground]
        sat_filename = self.idx2sat[idx_sat]
        ground_caption = self.ground_captions.get(ground_filename, "")
        sat_caption = self.sat_captions.get(sat_filename, "")

        # Label is the sat_idx for retrieval tasks
        label = torch.tensor(idx_sat, dtype=torch.long)

        return query_img, reference_img, label, ground_caption, sat_caption


    def _load_image(self, path):
        try:
            # Open the image using PIL
            img = Image.open(path)
            # Ensure the image is in RGB mode
            img = img.convert("RGB")
            return img
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading image from {path}: {e}")


    def __len__(self):
        return len(self.samples)
