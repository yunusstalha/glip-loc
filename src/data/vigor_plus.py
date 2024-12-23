# src/data/vigor_dataset.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

import copy
from collections import defaultdict
import random
from torchvision import transforms


class VigorDataset(Dataset):
    """
    A single class that can serve as:
      - Train dataset (ground+sat pairs),
      - Query dataset (only ground images + multi-label),
      - Reference dataset (only satellite images).
    """

    def __init__(self,
                 data_folder: str,
                 split: str = 'train',
                 same_area: bool = True,
                 dataset_mode: str = 'train',  
                 # ^-- new param: "train", "query", "reference"
                 ground_transforms=None,
                 satellite_transforms=None,
                 use_captions: bool = True,
                 prob_flip: float = 0.5,
                 prob_rotate: float = 0.5,
                 shuffle_batch_size: int = 64):
        """
        Args:
            data_folder (str): Path to the VIGOR dataset directory.
            split (str): 'train', 'val', or 'test'.
            same_area (bool): Whether to use the same-area splits or cross-area splits.
            dataset_mode (str): One of ["train", "query", "reference"].
            ground_transforms (callable): Torchvision transforms for ground images.
            satellite_transforms (callable): Torchvision transforms for satellite images.
            use_captions (bool): If True, load captions for ground and sat images.
            prob_flip (float): Probability of horizontal flip (only used in train mode).
            prob_rotate (float): Probability of random rotation (only used in train mode).
            shuffle_batch_size (int): For custom shuffle in train mode only.
        """
        super().__init__()
        self.data_folder = data_folder
        self.split = split
        self.same_area = same_area
        self.dataset_mode = dataset_mode.lower().strip()  # "train", "query", "reference"

        self.ground_transforms = ground_transforms
        self.satellite_transforms = satellite_transforms
        self.use_captions = use_captions

        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        # For simplicity, define the set of possible cities
        if self.same_area:
            # same-area scenario
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            # cross-area scenario
            if self.split == 'train':
                self.cities = ['NewYork', 'Seattle']
            else:
                self.cities = ['Chicago', 'SanFrancisco']

        # 1) Load the global satellite list
        self.df_sat = self._load_satellite_list()  # all satellites in these cities
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))

        # 2) Load the ground splits
        self.df_ground = self._load_ground_split(sat2idx, self.split)
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))

        # If we want near-positives, they are in columns sat_np1, sat_np2, sat_np3
        # We'll store that as well
        self.ground_np = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values
        # shape [N_ground, 4], the first col is the "main" sat ID, next are near-positives

        # 3) If training => build pairs
        #    If reference => build list of *unique satellite images*
        #    If query => build list of ground images
        self.samples = []
        self.idx2pairs = defaultdict(list)

        if self.dataset_mode == "train":
            # We'll do the existing logic: each row => (ground_idx, sat_idx)
            self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
            for pair in self.pairs:
                self.idx2pairs[pair[1]].append(pair)
            self.samples = copy.deepcopy(self.pairs)

        elif self.dataset_mode == "reference":
            # We'll gather *all satellite images* from df_sat
            # Because we want to evaluate retrieval among all satellites.
            # We'll store them as a list of (sat_idx).
            # The label = sat_idx (so we know which sat is which).
            self.samples = self.df_sat.index.tolist()  # all indices
            # or we could do: self.samples = list(range(len(self.df_sat)))

        elif self.dataset_mode == "query":
            # We'll gather *all ground images* from df_ground
            # The label = [sat, sat_np1, sat_np2, sat_np3] so we can compute hit-rate
            self.samples = self.df_ground.index.tolist()

        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}. Must be train/query/reference.")

        # 4) Load captions if needed
        if self.use_captions:
            self.ground_captions = self._load_captions('panorama_captions.csv', 'panorama')
            self.sat_captions = self._load_captions('satellite_captions.csv', 'satellite')
        else:
            self.ground_captions = {}
            self.sat_captions = {}

    # -------------------------------------------------------------------------
    # Internal loading functions
    # -------------------------------------------------------------------------
    def _load_satellite_list(self):
        sat_list = []
        for city in self.cities:
            file_path = f'{self.data_folder}/splits/{city}/satellite_list.txt'
            df_tmp = pd.read_csv(file_path, header=None, sep='\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(
                lambda x: f'{self.data_folder}/{city}/satellite/{x.sat}', axis=1
            )
            df_tmp['city'] = city
            sat_list.append(df_tmp)
        return pd.concat(sat_list, axis=0).reset_index(drop=True)

    def _load_ground_split(self, sat2idx, split):
        if self.same_area:
            split_file = f'same_area_balanced_{split}.txt'
        else:
            # cross-area fallback
            # if no separate "test" file is provided, we do:
            split_file = 'pano_label_balanced.txt'

        ground_list = []
        for city in self.cities:
            file_path = f'{self.data_folder}/splits/{city}/{split_file}'
            df_tmp = pd.read_csv(file_path, header=None, sep='\s+')
            # columns: ground, sat, sat_np1, sat_np2, sat_np3 => indices 0,1,4,7,10
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={
                0:  "ground",
                1:  "sat",
                4:  "sat_np1",
                7:  "sat_np2",
                10: "sat_np3"
            })
            df_tmp["path_ground"] = df_tmp.apply(
                lambda x: f'{self.data_folder}/{city}/panorama/{x.ground}', axis=1
            )
            # map satellites to indices
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[sat_n] = df_tmp[sat_n].map(sat2idx)

            df_tmp['city'] = city
            ground_list.append(df_tmp)
        return pd.concat(ground_list, axis=0).reset_index(drop=True)

    def _load_captions(self, caption_file_name, img_type):
        """
        img_type is 'panorama' or 'satellite'. 
        We'll attempt to load a CSV with columns [filename, caption].
        """
        captions = {}
        for city in self.cities:
            caption_path = f'{self.data_folder}/{city}/{caption_file_name}'
            if os.path.exists(caption_path):
                df = pd.read_csv(caption_path)
                cap_dict = dict(zip(df['filename'], df['caption']))
                captions.update(cap_dict)
        return captions

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return img
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading image from {path}: {e}")

    # -------------------------------------------------------------------------
    # __getitem__ logic depends on mode
    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        """
        Returns different structures based on dataset_mode.
        - train: (ground_img, sat_img, label, ground_caption, sat_caption)
        - query: (ground_img, label, None, ground_caption, None)
                 where label = [sat, sat_np1, sat_np2, sat_np3]
        - reference: (sat_img, label, None, None, None)
                     where label = sat_idx
        """
        if self.dataset_mode == "train":
            # existing logic
            return self._get_train_item(index)

        elif self.dataset_mode == "reference":
            return self._get_reference_item(index)

        elif self.dataset_mode == "query":
            return self._get_query_item(index)

        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")

    def _get_train_item(self, index):
        """
        Original train logic: self.samples[index] is a (ground_idx, sat_idx) pair
        with possible augmentations.
        """
        idx_ground, idx_sat = self.samples[index]
        ground_img = self._load_image(self.idx2ground_path[idx_ground])
        sat_img = self._load_image(self.idx2sat_path[idx_sat])

        # flips
        if random.random() < self.prob_flip:
            ground_img = ground_img.transpose(Image.FLIP_LEFT_RIGHT)
            sat_img = sat_img.transpose(Image.FLIP_LEFT_RIGHT)

        # transforms
        if self.ground_transforms:
            ground_img = self.ground_transforms(ground_img)
        if self.satellite_transforms:
            sat_img = self.satellite_transforms(sat_img)

        # random rotate
        if random.random() < self.prob_rotate:
            r = random.choice([1, 2, 3])
            # rotate sat
            sat_img = torch.rot90(sat_img, k=r, dims=(1, 2))
            # roll ground
            _, _, w = ground_img.shape
            shifts = - w // 4 * r
            ground_img = torch.roll(ground_img, shifts=shifts, dims=2)

        ground_filename = self.idx2ground[idx_ground]
        sat_filename = self.idx2sat[idx_sat]
        g_cap = self.ground_captions.get(ground_filename, "")
        s_cap = self.sat_captions.get(sat_filename, "")

        # label => sat_idx
        label = torch.tensor(idx_sat, dtype=torch.long)

        return (ground_img, sat_img, label, g_cap, s_cap)

    def _get_reference_item(self, index):
        """
        Return a single satellite image and label=sat_idx.
        self.samples is a list of sat_idxs in the DataFrame.
        """
        sat_idx = self.samples[index]
        sat_img = self._load_image(self.idx2sat_path[sat_idx])

        # Typically no random augmentations in reference mode
        if self.satellite_transforms:
            sat_img = self.satellite_transforms(sat_img)

        sat_filename = self.idx2sat[sat_idx]
        s_cap = self.sat_captions.get(sat_filename, "")

        label = torch.tensor(sat_idx, dtype=torch.long)

        # We'll return (sat_img, label, caption, None, None) or something,
        # but typically we only need (sat_img, label).
        return (sat_img, label)

    def _get_query_item(self, index):
        """
        Return a single ground image, with label = [sat, sat_np1, sat_np2, sat_np3].
        (i.e. the “4 columns” from ground_np).
        """
        ground_idx = self.samples[index]
        ground_img = self._load_image(self.idx2ground_path[ground_idx])

        if self.ground_transforms:
            ground_img = self.ground_transforms(ground_img)

        # label => array of 4 sat IDs
        label_np = self.ground_np[ground_idx]  # shape: [4]
        label_tensor = torch.tensor(label_np, dtype=torch.long)

        ground_filename = self.idx2ground[ground_idx]
        g_cap = self.ground_captions.get(ground_filename, "")

        # For a query, we typically only need ground image + label
        return (ground_img, label_tensor)

    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------------------
    # Shuffle method only relevant in train mode
    # -------------------------------------------------------------------------
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        """
        Custom shuffle for train pairs only.
        If dataset_mode != "train", it won't do anything.
        """
        if self.dataset_mode != "train":
            return

        import time
        from tqdm import tqdm
        print("\nShuffle Dataset:")

        # proceed with your existing shuffle logic
        pair_pool = copy.deepcopy(self.pairs)
        idx2pair_pool = copy.deepcopy(self.idx2pairs)

        neighbour_split = neighbour_select // 2
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # shuffle pair order
        random.shuffle(pair_pool)

        pairs_epoch = set()
        idx_batch = set()

        batches = []
        current_batch = []

        break_counter = 0
        pbar = tqdm()

        while True:
            pbar.update()
            if len(pair_pool) == 0:
                break

            pair = pair_pool.pop(0)
            _, idx = pair

            if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                idx_batch.add(idx)
                current_batch.append(pair)
                pairs_epoch.add(pair)

                idx2pair_pool[idx].remove(pair)

                # Hard negative mining
                if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                    near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                    near_always = near_similarity[:neighbour_split]
                    near_random = near_similarity[neighbour_split:]
                    random.shuffle(near_random)
                    near_random = near_random[:neighbour_split]
                    near_similarity_select = near_always + near_random

                    for idx_near in near_similarity_select:
                        if len(current_batch) >= self.shuffle_batch_size:
                            break
                        if idx_near not in idx_batch:
                            near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                            random.shuffle(near_pairs)
                            for near_pair in near_pairs:
                                idx_batch.add(idx_near)
                                current_batch.append(near_pair)
                                pairs_epoch.add(near_pair)
                                idx2pair_pool[idx_near].remove(near_pair)
                                similarity_pool[idx].remove(idx_near)
                                break
                break_counter = 0
            else:
                if pair not in pairs_epoch:
                    pair_pool.append(pair)
                break_counter += 1
                if break_counter >= 1024:
                    break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)

        self.samples = batches
        print("pair_pool:", len(pair_pool))
        print(f"Original Length: {len(self.pairs)} - Length after Shuffle: {len(self.samples)}")
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        if len(self.samples) > 0:
            print(f"First Element ID: {self.samples[0][1]} - Last Element ID: {self.samples[-1][1]}")
