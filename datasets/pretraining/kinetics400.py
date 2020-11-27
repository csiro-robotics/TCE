import os
import torch 
import pickle
import numpy as np 
import random 
from glob import glob
from tqdm import tqdm
import PIL.Image as Image
from multiprocessing import Process, Manager, Pool

class kineticsFolderInstance:
    """
    Dataset instance for a single kinetics video
    """
    def __init__(self, root, membank_idx, frame_padding):
        self.root = root
        self.membank_idx = membank_idx
        self.frame_padding              = frame_padding
        self.vid_length = len(glob(os.path.join(root,'*')))

    def __len__(self):
        return self.vid_length

    def __call__(self):
        anchor_idx = np.random.randint(0, self.vid_length)
        if anchor_idx >= self.vid_length - (2 * self.frame_padding) - 1:
            pair_idx = anchor_idx - self.frame_padding
        else:
            pair_idx = anchor_idx + self.frame_padding

        anchor_path = os.path.join(self.root, 'frame{}.jpg'.format(anchor_idx))
        pair_path = os.path.join(self.root, 'frame{}.jpg'.format(pair_idx))
        
        return anchor_path, pair_path

class Kinetics400:
    def __init__(self, cfg, transform):
        self.root = os.path.join(cfg.DATASET.KINETICS400.FRAMES_PATH, 'train')
        self.frame_padding = cfg.TRAIN.PRETRAINING.FRAME_PADDING
        self.transform = transform 
        self.K = cfg.LOSS.PRETRAINING.NCE.NEGATIVES
        assert self.root is not '', 'Please specify Kinetics400 Path in config'
        pickle_file = os.path.join(cfg.ASSETS_PATH, 'Kinetics400Dataset.pickle')        
        if not os.path.exists(pickle_file):     
            # ===== Makes Sample List =====
            self.samples = self.get_sample_list()

            # ===== Save Pickle File =====
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.samples,f)        
        else:
            # ===== Load Pickle File =====
            with open(pickle_file, 'rb') as f:
                self.samples = pickle.load(f)
            f.close()

        self.all_membank_negatives = list(range(len(self.samples)))

    
    def process_video(self, x):
        video, idx, L = x
        try:
            sample = kineticsFolderInstance(
                root = video,
                membank_idx = idx,
                frame_padding = self.frame_padding
            )
            if len(sample) != 0:
                L.append(sample)
        except:
            pass

    def get_sample_list(self):
        video_folders = glob(os.path.join(self.root,'*','*'))

        # ===== Set up multiprocessing =====
        L = Manager().list()
        pool = Pool(processes=32)

        # ===== Prepare inputs for multiprocessing =====
        inputs = [[video, idx, L] for idx, video in enumerate(video_folders)]

        # ===== Create sample list =====
        pbar = tqdm(total = len(inputs))
        print(len(inputs))
        for i in pool.imap_unordered(self.process_video, inputs):
            pbar.update(1)
        
        pool.close()
        pool.join()
        
        all_samples = list(L)

        # ===== Fix membank_idx due to failed samples =====
        for idx, sample in enumerate(all_samples):
            sample.membank_idx = idx 

        return all_samples



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        # Get sample from index
        sample = self.samples[index]
        membank_idx = sample.membank_idx
        anchor_path, pair_path = sample()

        anchor_frame = Image.open(anchor_path)
        pair_frame = Image.open(pair_path)
        anchor_tensor, pair_tensor, rotation_tensor, rotation_gt = self.transform(anchor_frame, pair_frame)

        # Get negatives 
        potential_negatives = self.all_membank_negatives[:membank_idx] + self.all_membank_negatives[membank_idx + 1:]
        negatives = torch.tensor(random.sample(potential_negatives, self.K + 1))

        inputs = {
            'anchor_tensor': anchor_tensor,
            'pair_tensor': pair_tensor,
            'membank_idx': membank_idx,
            'rotation_tensor': rotation_tensor,
            'rotation_gt': rotation_gt,
            'negatives': negatives,
        }

        return inputs
