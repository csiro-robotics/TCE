import os 
import math 
import time 
import argparse 
import logging 
import torch
import numpy as np 
from tensorboardX import SummaryWriter


from config import config, update_config
from network import *
from datasets.pretraining import get_pretraining_dataset
from loss import get_loss
from utils import *


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

# TODO
# Test Learning Rate Decay 
# Test end-of-epoch
# Test Loading Checkpoints
# Train end-to-end 

def parse_args():
    parser = argparse.ArgumentParser(description='Train TCE Self-Supervised')
    parser.add_argument('--cfg', help = 'Path to config file', type = str, default = None)
    parser.add_argument('opts', help = 'Modify config using the command line', 
                         default = None, nargs=argparse.REMAINDER )
    args = parser.parse_args()
    update_config(config, args)

    return args

def train(epoch, train_loader, model, NCELoss, RotationLoss, optimizer, config, tboard, logger):
    
    # ===== Get total steps, set up meters =====
    total_steps = config.TRAIN.PRETRAINING.EPOCHS * len(train_loader)
    model.train()
    NCELoss.train()

    batch_time = AverageMeter()

    MainLossMeter = AverageMeter()
    RotationLossMeter = AverageMeter()
    TotalLossMeter = AverageMeter()
    RotationAccMeter = AverageMeter()

    tl = config.LOSS.PRETRAINING.MINING.THRESH_LOW
    th = config.LOSS.PRETRAINING.MINING.THRESH_HIGH
    tr = config.LOSS.PRETRAINING.MINING.THRESH_RATE

    for idx, inputs in enumerate(train_loader):
        end = time.time()
        # ===== Prepare data and get threshold ======
        anchor_tensor = inputs['anchor_tensor'].cuda()
        pair_tensor = inputs['pair_tensor'].cuda()
        rotation_tensor = inputs['rotation_tensor'].cuda()
        rotation_gt = inputs['rotation_gt'].cuda()
        inputs['negatives'] = inputs['negatives'].cuda()
        inputs['membank_idx'] = inputs['membank_idx'].cuda()

        step = (epoch - 1) * len(train_loader) + idx + 1
        threshold = tl + (th - tl) * (1 - math.exp(tr * step / total_steps))

        # ===== Forward =====
        bsz = anchor_tensor.size(0)
        inputs['anchor_feature'] = model(anchor_tensor, rotation = False)
        inputs['pair_feature'] = model(pair_tensor, rotation = False)
        rotation_feature = model(rotation_tensor, rotation = True)

        main_loss = NCELoss(inputs, threshold)
        with torch.no_grad():
            rotation_indexes = rotation_feature.max(1)[1]
            rotation_accuracy = int(torch.sum(torch.eq(rotation_indexes, rotation_gt).long())) / bsz
        rotation_loss = RotationLoss(rotation_feature, rotation_gt).mul(config.LOSS.PRETRAINING.ROTATION_WEIGHT)
        total_loss = main_loss + rotation_loss

        # ===== Backward ===== 
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ===== Update Meters =====
        batch_time.update(time.time() - end, bsz)
        MainLossMeter.update(main_loss.item(), bsz)
        RotationLossMeter.update(rotation_loss.item(), bsz)
        TotalLossMeter.update(total_loss.item(), bsz)
        RotationAccMeter.update(rotation_accuracy, bsz)

        # ===== Print and Update Tensorboards
        if idx % config.PRINT_FREQ == 0:
            logger.info('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Main Loss: {mainloss.val:.3f} ({mainloss.avg:.3f})\t'
                'Rotation Loss {rotloss.val:.3f} ({rotloss.avg:.3f})\t'
                'Total Loss {totloss.val:.3f} ({totloss.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                mainloss   = MainLossMeter,
                rotloss  = RotationLossMeter,
                totloss  = TotalLossMeter,
                ))
            tboard.add_scalars("Main Loss", 
                {"Absolute":MainLossMeter.val, "Average":MainLossMeter.avg}, step)
            tboard.add_scalars("Rotation Loss", 
                {"Absolute":RotationLossMeter.val, "Average":RotationLossMeter.avg}, step)
            tboard.add_scalars("Total Loss", 
                {"Absolute":TotalLossMeter.val, "Average":TotalLossMeter.avg}, step)
            tboard.add_scalars("Rotation Accuracy", 
                {"Absolute":RotationAccMeter.val, "Average":RotationAccMeter.avg}, step)

    return_dic = {
        'Main Loss' : {'Average' : MainLossMeter.avg},
        'Rotation Loss' : {'Average' : RotationLossMeter.avg},
        'Total Loss' : {'Average' : TotalLossMeter.avg},
        'Rotation Accuracy' : {'Average' : RotationAccMeter.avg},
    }

    return return_dic 


def main():
    args = parse_args()
    logger = setup_logger()
    logger.info(config)
    if not os.path.exists(config.ASSETS_PATH):
        os.makedirs(config.ASSETS_PATH)
        
    # ===== Create the dataloader =====
    train_loader, n_data = get_pretraining_dataset(config)
    logger.info('Training with {} Train Samples'.format(n_data))

    # ===== Create the model =====
    model = PreTrainNet(config)
    logger.info('Built Model, using {} backbone'.format(config.MODEL.TRUNK))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    logger.info('Training on {} GPUs'.format(torch.cuda.device_count()))

    # ===== Set the optimizer =====
    optimizer = get_optimizer(model, config, pretraining = True )

    # ===== Get the loss  =====
    NCELoss, RotationLoss = get_loss(config, n_data)
    NCELoss = NCELoss.cuda()
    RotationLoss = RotationLoss.cuda()

    # ===== Resume from am earlier checkpoint =====
    start_epoch = 1
    if config.TRAIN.PRETRAINING.RESUME:
        try:
            checkpoint = torch.load(config.TRAIN.PRETRAINING.RESUME)
        except FileNotFoundError:
            raise FileNotFoundError('No Checkpoint found at path {}'.format(config.TRAIN.PRETRAINING.RESUME))

        start_epoch = checkpoint['epoch'] + 1
        # ===== Align checkpoint keys with model =====
        if 'module' in list(checkpoint['state_dict'].keys())[0] and 'module' not in list(model.state_dict().keys())[0]:
            checkpoint['state_dict'] = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items() }
        elif 'module' not in list(checkpoint['state_dict'].keys())[0] and 'module' in list(model.state_dict().keys())[0]:
            checkpoint['state_dict'] = {'module.' + k:v for k,v in checkpoint['state_dict'].items() }

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        NCELoss.load_state_dict(checkpoint['NCELoss'])
        logger.info('Loaded Checkpoint from "{}"'.format(config.TRAIN.PRETRAINING.RESUME))
    else:
        logger.info('Training from Random Initialisation')

    # ===== Set up Save Directory and TensorBoard =====
    assert config.TRAIN.PRETRAINING.SAVEDIR, 'Please specify save directory for model'
    if not os.path.exists(config.TRAIN.PRETRAINING.SAVEDIR):
        os.makedirs(config.TRAIN.PRETRAINING.SAVEDIR)
        os.makedirs(os.path.join(config.TRAIN.PRETRAINING.SAVEDIR, 'checkpoints'))
        os.makedirs(os.path.join(config.TRAIN.PRETRAINING.SAVEDIR, 'tboard'))
        
    tboard = SummaryWriter(logdir = os.path.join(config.TRAIN.PRETRAINING.SAVEDIR, 'tboard'))

    # ===== Train Loop ===== 
    logger.info('Begin training')
    for epoch in range(start_epoch, config.TRAIN.PRETRAINING.EPOCHS + 1):
        adjust_learning_rate(epoch, config, optimizer, pretraining = True, logger = logger)
        logger.info('Training Epoch {}'.format(epoch))

        return_dic = train(
            epoch = epoch,
            train_loader = train_loader,
            model = model,
            NCELoss = NCELoss,
            RotationLoss = RotationLoss,
            optimizer = optimizer,
            config = config,
            tboard = tboard,
            logger = logger,
        )

        for key, value in return_dic.items():
            tboard.add_scalars(key, value, epoch)

        if epoch % config.TRAIN.PRETRAINING.SAVE_FREQ == 0:
            state = {
                'config': config,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'NCELoss': NCELoss.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(config.TRAIN.PRETRAINING.SAVEDIR, 'checkpoints', 'ckpt_epoch_{}.pth'.format(epoch))
            logger.info('Saved Checkpoint to {}'.format(save_file))
            torch.save(state, save_file)

if __name__ == '__main__':
    main()

    









