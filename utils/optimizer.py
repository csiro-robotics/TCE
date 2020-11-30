import torch 
import numpy as np 

def get_optimizer(model, config, pretraining): # TODO Move to Utils
    if pretraining == True:
        lr = config.TRAIN.PRETRAINING.LEARNING_RATE
        momentum = config.TRAIN.PRETRAINING.MOMENTUM 
        weight_decay = config.TRAIN.PRETRAINING.WEIGHT_DECAY
    elif pretraining == False:
        lr = config.TRAIN.FINETUNING.LEARNING_RATE
        momentum = config.TRAIN.FINETUNING.MOMENTUM 
        weight_decay = config.TRAIN.FINETUNING.WEIGHT_DECAY

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = lr,
        momentum = momentum,
        weight_decay = weight_decay
    )

    return optimizer

def adjust_learning_rate(epoch, cfg, optimizer, pretraining, logger):
    """Sets the learning rate to the initial LR decayed by the given rate every steep step"""
    if pretraining == True:
        decay_epochs = cfg.TRAIN.PRETRAINING.DECAY_EPOCHS
        lr = cfg.TRAIN.PRETRAINING.LEARNING_RATE
        decay_factor = cfg.TRAIN.PRETRAINING.DECAY_FACTOR
    elif pretraining == False:
        decay_epochs = cfg.TRAIN.FINETUNING.DECAY_EPOCHS
        lr = cfg.TRAIN.FINETUNING.LEARNING_RATE
        decay_factor = cfg.TRAIN.FINETUNING.DECAY_FACTOR


    steps = np.sum(epoch > np.asarray(decay_epochs))
    if steps > 0:
        new_lr = lr * (decay_factor  ** steps)
        logger.info('Learning rate for epoch set to {}'.format(new_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        logger.info('Learning rate for epoch set to {}'.format(lr))
    