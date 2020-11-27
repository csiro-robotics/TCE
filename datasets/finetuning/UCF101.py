import os
import pickle 
import random 
from tqdm import tqdm 
import PIL.Image as Image 
import multiprocessing as mp 

from utils import UCF101_splitter
from torchvision import transforms 
from torch.utils.data import DataLoader

class UCF_dataset:
    def __init__(self, root, file_dict, mode, transforms):
        self.root = root
        self.keys = list(file_dict.keys())
        self.gts = list(file_dict.values())
        self.mode = mode
        self.transform = transforms

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self, video_name, index):

        frame_path = os.path.join(self.root, '{}/frame{:06d}.jpg'.format(
             video_name, index))

        frame = Image.open(frame_path)
        frame_tensor = self.transform(frame)

        return frame_tensor

    def __getitem__(self, idx):
        if self.mode == 'train':
            # ===== Get training sample =====
            video_name, nb_clips = self.keys[idx].split(' ')

            nb_clips = int(nb_clips)
            label = self.gts[idx]
            clips = []
            clips.append(random.randint(1, int(nb_clips / 3)))
            clips.append(random.randint(int(nb_clips / 3), int(nb_clips * 2 / 3)))
            clips.append(random.randint(int(nb_clips * 2 / 3), nb_clips + 1))

        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index = abs(int(index))
        else:
            raise ValueError('There are only train and val modes')

        label = self.gts[idx]
        label = int(label) - 1

        if self.mode == 'train':
            data = []
            for i in range(len(clips)):
                index = clips[i]
                data.append(self.load_ucf_image(video_name, index))
            sample = (data, label)
        elif self.mode == 'val':
            data = self.load_ucf_image(video_name, index)
            sample = (video_name.split('/')[0], data, label)
        
        return sample 


class UCF101:
    def __init__(self, cfg, logger):
        self.batch_size = cfg.TRAIN.FINETUNING.BATCH_SIZE 
        self.num_workers = cfg.WORKERS 
        self.root = cfg.DATASET.UCF101.FRAMES_PATH
        self.splits_path = cfg.DATASET.UCF101.SPLITS_PATH
        self.split = cfg.DATASET.UCF101.SPLIT 
        self.cfg = cfg
        self.logger = logger 

        # ===== Split the dataset =====
        splitter = UCF101_splitter(self.splits_path, self.split)
        self.train_videos, self.val_videos = splitter.split_video()
        self.load_frame_count()
        self.get_training_dict()
        self.get_validation_dict()
    
    @staticmethod
    def process_video(x):
        root, video, L = x
        key = video
        n_frames = len(os.listdir(os.path.join(root, video)))
        L.append([key.replace('HandStandPushups', 'HandstandPushups'), n_frames])

    def load_frame_count(self):
        pickle_path = os.path.join(self.cfg.ASSETS_PATH, 'UCF101_frame_count_split_{:02d}.pickle'.format(self.split))
        if not os.path.exists(pickle_path):
            
            # ===== Build frame count dict if no pickle file found =====
            self.frame_count = {}
            self.logger.info('Creating frame count dictionary for UCF101')

            # ===== Set up multiprocessing =====
            pool = mp.Pool(processes = 16)
            manager = mp.Manager()
            L = manager.list()
            in_args = []
            for video in os.listdir(self.root):
                in_args.append([self.root, video, L])
            
            # ===== Get frame counts using multiprocessing =====
            for _ in tqdm(pool.imap_unordered(self.process_video, in_args), total = len(in_args)):
                pass 
            
            pool.close()
            pool.join()
            self.frame_count = {k:v for k,v in list(L)}

            with open(pickle_path, 'wb') as f:
                pickle.dump(self.frame_count, f)
            self.logger.info('Saved frame count dictionary to {}'.format(pickle_path))

        else:

            # ===== Load frame count dict if already exists
            with open(pickle_path, 'rb') as f:
                self.frame_count = pickle.load(f)

    def get_training_dict(self):
        self.dict_training = {}
        for video in self.train_videos:
            nb_frame = self.frame_count[video] - 10 + 1
            key = '{} {}'.format(video, nb_frame)
            self.dict_training[key] = self.train_videos[video]

    def get_validation_dict(self):
        self.dict_val = {}
        for video in self.val_videos:
            nb_frame = self.frame_count[video] - 10 + 1
            interval = nb_frame // 19
            for idx in range(19):
                frame = interval * idx + 1
                key = '{} {}'.format(video, frame)
                self.dict_val[key] = self.val_videos[video]

    
    def get_loaders(self):
        # ===== Get transforms =====
        tran = transforms.Compose([
                        transforms.Resize(self.cfg.DATASET.FINETUNING.TRANSFORMATIONS.RESIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = self.cfg.DATASET.FINETUNING.MEAN, std = self.cfg.DATASET.FINETUNING.STD)
        ])
        
        # ===== Make Datasets =====
        train_dataset = UCF_dataset(
            root = self.cfg.DATASET.UCF101.FRAMES_PATH,
            file_dict = self.dict_training,
            mode = 'train',
            transforms = tran
        )

        val_dataset = UCF_dataset(
            root = self.cfg.DATASET.UCF101.FRAMES_PATH,
            file_dict = self.dict_val,
            mode = 'val',
            transforms = tran
        )

        self.logger.info('Created Train / Val splits')
        self.logger.info('Training Videos : {}'.format(len(train_dataset)))
        self.logger.info('Validation Videos : {}'.format(len(val_dataset) // 19))

        # ===== Create loaders =====
        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = self.cfg.TRAIN.FINETUNING.BATCH_SIZE,
            shuffle = True,
            num_workers = self.num_workers 
        )

        val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = self.cfg.TRAIN.FINETUNING.BATCH_SIZE,
            shuffle = True,
            num_workers = self.num_workers
        )

        val_gt = {k:v - 1 for k,v in self.val_videos.items()}

        return train_loader, val_loader, val_gt

if __name__ == '__main__':
    import argparse 
    from config import config, update_config
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default = None)
    parser.add_argument('opts', default = None, nargs = argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    logger = setup_logger()

    train_loader, val_loader, val_gt = UCF101(cfg, logger)

