import os

from yacs.config import CfgNode as CN

_C = CN()


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.WORKERS = 8
_C.PRINT_FREQ = 20
_C.ASSETS_PATH = '/scratch1/kni101/TCE/assets'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.KINETICS400 = CN()
_C.DATASET.KINETICS400.FRAMES_PATH = '/datasets/work/d61-eif/source/kinetics400-frames'

_C.DATASET.UCF101 = CN()
_C.DATASET.UCF101.FRAMES_PATH = '/datasets/work/d61-eif/source/UCF-101-twostream/jpegs_256'
_C.DATASET.UCF101.SPLITS_PATH = '/datasets/work/d61-eif/source/UCF-101-twostream/UCF_list'
_C.DATASET.UCF101.SPLIT = 1

_C.DATASET.PRETRAINING = CN() 
_C.DATASET.PRETRAINING.DATASET = 'kinetics400'
_C.DATASET.PRETRAINING.MEAN = [0.485, 0.456, 0.406]
_C.DATASET.PRETRAINING.STD = [0.229, 0.224, 0.225]

_C.DATASET.FINETUNING = CN() 
_C.DATASET.FINETUNING.DATASET = 'UCF101'
_C.DATASET.FINETUNING.MEAN = [0.485, 0.456, 0.406]
_C.DATASET.FINETUNING.STD = [0.229, 0.224, 0.225]


_C.DATASET.PRETRAINING.TRANSFORMATIONS = CN()
_C.DATASET.PRETRAINING.TRANSFORMATIONS.CROP_SIZE = (224,224)
_C.DATASET.PRETRAINING.TRANSFORMATIONS.SCALE_SIZE = (224,224)
_C.DATASET.PRETRAINING.TRANSFORMATIONS.HORIZONTAL_FLIP = True 
_C.DATASET.PRETRAINING.TRANSFORMATIONS.RANDOM_GREY = True 
_C.DATASET.PRETRAINING.TRANSFORMATIONS.COLOUR_JITTER = True 

_C.DATASET.FINETUNING.TRANSFORMATIONS = CN()
_C.DATASET.FINETUNING.TRANSFORMATIONS.RESIZE = (224,224)
_C.DATASET.FINETUNING.TRANSFORMATIONS.CROP_SIZE = (224,224)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TRUNK = 'resnet18'
_C.MODEL.PRETRAINING = CN()
_C.MODEL.PRETRAINING.FC_DIM = 128

_C.MODEL.FINETUNING = CN()
_C.MODEL.FINETUNING.NUM_CLASSES = -1
_C.MODEL.FINETUNING.DROPOUT = 0

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.PRETRAINING = CN()
_C.LOSS.PRETRAINING.ROTATION_WEIGHT = 1

_C.LOSS.PRETRAINING.NCE = CN()
_C.LOSS.PRETRAINING.NCE.NEGATIVES = 8192
_C.LOSS.PRETRAINING.NCE.TEMPERATURE = 0.07
_C.LOSS.PRETRAINING.NCE.MOMENTUM = 0.5


_C.LOSS.PRETRAINING.MINING = CN()
_C.LOSS.PRETRAINING.MINING.USE_MINING = True
_C.LOSS.PRETRAINING.MINING.THRESH_LOW = -1
_C.LOSS.PRETRAINING.MINING.THRESH_HIGH = 1
_C.LOSS.PRETRAINING.MINING.THRESH_RATE = 5
_C.LOSS.PRETRAINING.MINING.MAX_HARD_NEGATIVES_PERCENTAGE = 1

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.PRETRAINING = CN()
_C.TRAIN.PRETRAINING.LEARNING_RATE = 0.03
_C.TRAIN.PRETRAINING.DECAY_EPOCHS = (25,)
_C.TRAIN.PRETRAINING.DECAY_FACTOR = 0.1

_C.TRAIN.PRETRAINING.SAVEDIR = ''
_C.TRAIN.PRETRAINING.MOMENTUM = 0.9
_C.TRAIN.PRETRAINING.WEIGHT_DECAY = 1e-4
_C.TRAIN.PRETRAINING.SAVE_FREQ = 1
_C.TRAIN.PRETRAINING.BATCH_SIZE = 100
_C.TRAIN.PRETRAINING.EPOCHS = 50
_C.TRAIN.PRETRAINING.RESUME = ''
_C.TRAIN.PRETRAINING.FRAME_PADDING = 0

_C.TRAIN.FINETUNING = CN()
_C.TRAIN.FINETUNING.CHECKPOINT = ''
_C.TRAIN.FINETUNING.SAVEDIR = ''
_C.TRAIN.FINETUNING.RESUME = False 
_C.TRAIN.FINETUNING.BATCH_SIZE = 100
_C.TRAIN.FINETUNING.LEARNING_RATE = 0.05
_C.TRAIN.FINETUNING.EPOCHS = 900
_C.TRAIN.FINETUNING.DECAY_EPOCHS = (375,)
_C.TRAIN.FINETUNING.MOMENTUM = 0.9
_C.TRAIN.FINETUNING.WEIGHT_DECAY = 0
_C.TRAIN.FINETUNING.DECAY_FACTOR = 0.1
_C.TRAIN.FINETUNING.VAL_FREQ = 3
# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
_C.VISUALISATION = CN() 
_C.VISUALISATION.TSNE = CN()
_C.VISUALISATION.TSNE.CROP_SIZE = 224
_C.VISUALISATION.TSNE.RESIZE = 256


def update_config(cfg, args):
    cfg.defrost()
    if args.cfg != None:
        cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
