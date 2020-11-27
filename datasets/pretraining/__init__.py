import torch 
import random 
import numpy as np 
from .augmentation import *
from .kinetics400 import Kinetics400
from torchvision.transforms import Compose 


class PretrainTransforms:
    def __init__(self, cfg):
        img_transforms = []
        tensor_transforms = []
        # ===== Prepare scaling and crop sizes =====
        crop_size = cfg.DATASET.PRETRAINING.TRANSFORMATIONS.CROP_SIZE
        scale_size = cfg.DATASET.PRETRAINING.TRANSFORMATIONS.SCALE_SIZE 

        # ===== Add optional augmentations =====
        if cfg.DATASET.PRETRAINING.TRANSFORMATIONS.HORIZONTAL_FLIP == True:
            img_transforms.append(RandomHorizontalFlip(consistent=True))
        if cfg.DATASET.PRETRAINING.TRANSFORMATIONS.RANDOM_GREY == True:
            img_transforms.append(RandomGray(consistent=False, p=0.5))
        if cfg.DATASET.PRETRAINING.TRANSFORMATIONS.COLOUR_JITTER == True:
            img_transforms.append(ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0))

        # ===== Add Scale and Crop transforms =====
        img_transforms.append(Scale(size=scale_size))
        img_transforms.append(RandomCrop(size=crop_size, consistent = True))

        # ===== Create Rotation Transform =====
        self.Rotation = Rotation()

        # ===== Create Tensor Transformations =====
        tensor_transforms.append(ToTensor())
        tensor_transforms.append(Normalize(
            mean = cfg.DATASET.PRETRAINING.MEAN,
            std = cfg.DATASET.PRETRAINING.STD,
        ))

        self.img_transforms = Compose(img_transforms)
        self.tensor_transforms = Compose(tensor_transforms)

    def __call__(self, anchor_frame, pair_frame):
        
        # ===== Transform Pair Together =====
        anchor_tensor, pair_tensor = self.tensor_transforms(self.img_transforms([anchor_frame, pair_frame]))

        # ===== Transform Rotation, get Rotation GT =====
        rotation_gt = np.random.randint(0,4)
        rotation_intermediate = self.img_transforms([anchor_frame])
        rotation_intermediate = self.Rotation(rotation_intermediate, rotation = 90 * rotation_gt)
        rotation_tensor = self.tensor_transforms(rotation_intermediate)[0]

        return anchor_tensor, pair_tensor, rotation_tensor, rotation_gt

def get_pretraining_dataset(cfg):
    transforms = PretrainTransforms(cfg)
    dataset_name = cfg.DATASET.PRETRAINING.DATASET

    if dataset_name == 'kinetics400':
        dataset = Kinetics400(cfg, transforms)
    else:
        raise NotImplementedError

    n_data = len(dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = cfg.TRAIN.PRETRAINING.BATCH_SIZE,
        shuffle = True,
        num_workers = cfg.WORKERS,
        pin_memory = True,
        sampler = None,
        drop_last = True
    )

    return train_loader, n_data

        