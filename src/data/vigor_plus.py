# src/data/datasets.py

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
                 ground_transforms=None,
                 satellite_transforms=None,
                 use_captions: bool = True,
                 prob_flip = 0.5,
                 prob_rotate = 0.5,
                 shuffle_batch_size = 64):
        """
        Args:
            data_folder (str): Path to the VIGOR dataset directory.
            split (str): One of ['train', 'val']. Determines which split to load.
            same_area (bool): Whether to use the same_area splits for training/validation.
            ground_transforms: Torchvision transforms for ground images.
            satellite_transforms: Torchvision transforms for satellite images.
            use_captions (bool): If True, load captions for ground and sat images.
            prob_flip (float): Probability of horizontal flip in transforms.
            prob_rotate (float): Probability of random rotation in transforms.
            shuffle_batch_size (int): Number of samples in a batch for shuffling.
        """
        super().__init__()
        self.data_folder = data_folder
        self.split = split
        self.same_area = same_area
        self.ground_transforms = ground_transforms
        self.satellite_transforms = satellite_transforms
        self.use_captions = use_captions
        self.shuffle_batch_size =  64

        # Augmentation probabilities
        self.prob_flip = prob_flip        
        self.prob_rotate = prob_rotate
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

        # Prepare pairs: (ground_idx, sat_idx)
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        # for a unique sat_id we can have 1 or 2 ground views as gt
        for pair in self.pairs:      
            self.idx2pairs[pair[1]].append(pair)

        # Load captions if needed
        if self.use_captions:
            self.ground_captions = self._load_captions('panorama_captions.csv', 'panorama')
            self.sat_captions = self._load_captions('satellite_captions.csv', 'satellite')
        else:
            self.ground_captions = {}
            self.sat_captions = {}
        self.samples = copy.deepcopy(self.pairs)

    def _load_satellite_list(self):
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/satellite_list.txt', 
                                 header=None, sep='\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})  # Match original column naming
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
    
    # def _build_idx2pairs(self):
    #     from collections import defaultdict
    #     self.idx2pairs = defaultdict(list)
    #     # self.pairs is a list of (ground_idx, sat_idx)
    #     for pair in self.pairs:
    #         _, sat_idx = pair
    #         self.idx2pairs[sat_idx].append(pair)

    def __getitem__(self, index):
        idx_ground, idx_sat = self.samples[index]

        # Load images
        ground_img = self._load_image(self.idx2ground_path[idx_ground])
        satellite_img = self._load_image(self.idx2sat_path[idx_sat])
        if random.random() < self.prob_flip:
            ground_img = ground_img.transpose(Image.FLIP_LEFT_RIGHT)
            satellite_img = satellite_img.transpose(Image.FLIP_LEFT_RIGHT)
        # Apply transforms
        if self.ground_transforms:
            ground_img = self.ground_transforms(ground_img)
        if self.satellite_transforms:
            satellite_img = self.satellite_transforms(satellite_img)

        if random.random() < self.prob_rotate:
            r = random.choice([1, 2, 3])
            satellite_img = torch.rot90(satellite_img, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            _, _, w = ground_img.shape # c, h, w
            shifts = - w//4 * r
            ground_img = torch.roll(ground_img, shifts=shifts, dims=2)   

        # Get captions
        ground_filename = self.idx2ground[idx_ground]
        sat_filename = self.idx2sat[idx_sat]
        ground_caption = self.ground_captions.get(ground_filename, "")
        sat_caption = self.sat_captions.get(sat_filename, "")

        # Label is the sat_idx for retrieval tasks
        label = torch.tensor(idx_sat, dtype=torch.long)

        return ground_img, satellite_img, label, ground_caption, sat_caption


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

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''
        import time
        from tqdm import tqdm 
        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)
        idx2pair_pool = copy.deepcopy(self.idx2pairs)
        
        neighbour_split = neighbour_select // 2
                    
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)                
        
        # Shuffle pairs order
        random.shuffle(pair_pool)
        
        
        # Lookup if already used in epoch
        pairs_epoch = set()   
        idx_batch = set()
    
        
        # buckets
        batches = []
        current_batch = []
        
        
        # counter
        break_counter = 0
        
        # progressbar
        pbar = tqdm()

        while True:
            
            pbar.update()
            
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                
                _, idx = pair
                
                if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    
                    # remove from pool used for sim-sampling
                    idx2pair_pool[idx].remove(pair)

                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        
                        near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                        near_always = copy.deepcopy(near_similarity[:neighbour_split]) 
                        near_random = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(near_random)
                        near_random = near_random[:neighbour_split]
                        near_similarity_select = near_always + near_random

                        
                        for idx_near in near_similarity_select:
                        
                        
                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                        
                            # no check for pair in epoch necessary cause all we add is removed from pool
                            if idx_near not in idx_batch:
                        
                                near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                
                                # up to 2 for one sat view 
                                random.shuffle(near_pairs)
                            
                                for near_pair in near_pairs:
                                                                                
                                    idx_batch.add(idx_near)
                                    current_batch.append(near_pair)
                                    pairs_epoch.add(near_pair)
                                    
                                    idx2pair_pool[idx_near].remove(near_pair)
                                    similarity_pool[idx].remove(idx_near)
                                    
                                    # only select one view
                                    break
                            
                    break_counter = 0
                    
                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                        
                    break_counter += 1
                    
                if break_counter >= 1024:
                    break
                
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
            
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []
    
        pbar.close()
        
        # wait before closing progress bar
        time.sleep(0.3)
        
        self.samples = batches
        print("pair_pool:", len(pair_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  