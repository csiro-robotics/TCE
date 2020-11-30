import os 
import math 
import time 
import argparse 
import logging 
import torch
import torch.nn as nn
import numpy as np 
from tensorboardX import SummaryWriter


from config import config, update_config
from network import *
from datasets.pretraining import get_pretraining_dataset
from datasets.finetuning import *
from loss import get_loss
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train TCE Self-Supervised')
    parser.add_argument('--cfg', help = 'Path to config file', type = str, default = None)
    parser.add_argument('--val', default = False, action = 'store_true', help = 'Just validate checkpoint')
    parser.add_argument('opts', help = 'Modify config using the command line', 
                         default = None, nargs=argparse.REMAINDER )
    args = parser.parse_args()
    update_config(config, args)

    return args


def train(epoch, train_loader, model, criterion, optimizer, cfg, tboard, logger):
    
    # ===== Set up Meters =====
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    # ===== Switch to train mode =====
    model.train()
    

    # ===== Start training on batches =====
    for idx, (data, label) in enumerate(train_loader):
        end = time.time()
        bsz = data[0].size(0)

        # ===== Forwards =====
        data = [x.cuda() for x in data]
        label = label.cuda()


        for i, frame_tensor in enumerate(data):
            if i == 0:
                output = model(frame_tensor)
            else:
                output += model(frame_tensor)

        loss = criterion(output, label)
        prec1, prec5 = accuracy(output, label, topk=(1,5))

        # ===== Backwards =====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== Update meters and do logging =====
        batch_time.update(time.time() - end)
        loss_meter.update(loss.item(), bsz)
        top1_meter.update(prec1)
        top5_meter.update(prec5)

        if idx % cfg.PRINT_FREQ == 0:
            log_step = (epoch-1) * len(train_loader) + idx
            tboard.add_scalars('Train/Loss', {'val': loss_meter.val, 'avg': loss_meter.avg}, log_step)
            tboard.add_scalars('Train/Top1 Acc', {'val': top1_meter.val, 'avg': top1_meter.avg}, log_step)
            tboard.add_scalars('Train/Top5 Acc', {'val': top5_meter.val, 'avg': top5_meter.avg}, log_step)

            info = ('Epoch : {} ({}/{}) | '
                    'BT : {:02f} ({:02f}) | '
                    'Loss: {:02f} ({:02f}) | '
                    'Top1: {:02f} ({:02f}) | '
                    'Top5: {:02f} ({:02f})').format(
                        epoch, idx, len(train_loader),
                        batch_time.val, batch_time.avg,
                        loss_meter.val, loss_meter.avg,
                        top1_meter.val, top1_meter.avg,
                        top5_meter.val, top5_meter.avg
                    )
            logger.info(info)

@torch.no_grad()
def validate(epoch, val_loader, val_gt, model, criterion, cfg, tboard, logger):

    # ===== Switch to eval mode =====
    model.eval()
    video_predictions = {}
    end = time.time()
    pbar = tqdm(total = len(val_loader))

    for idx, (video_keys, data, label) in enumerate(val_loader):
        bsz = data.size(0)
        data = data.cuda()
        label = label.cuda()

        # ===== Calculate network output and pack into video_predictions =====
        output = model(data)
        for i in range(bsz):
            key = video_keys[i]
            pred = output[i]
            if key not in video_predictions.keys():
                video_predictions[key] = []
                video_predictions[key] = pred
            else:
                video_predictions[key] += pred 
        
        pbar.update(1)

    print('\n')
    # ===== Get Eval Top1, Top5, Loss =====
    video_level_preds = torch.zeros(len(video_predictions), 101).float().cuda()
    video_level_labels = torch.zeros(len(video_predictions)).long().cuda()

    for idx, key in enumerate(sorted(video_predictions.keys())):
        video_level_preds[idx] = video_predictions[key] 
        video_level_labels[idx] = val_gt[key]
    
    prec1, prec5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
    loss = criterion(video_level_preds, video_level_labels)

    # ===== Log ===== 
    logger.info('Validation complete for epoch {}, time taken {:02f} seconds'.format(epoch, time.time() - end))
    logger.info('Top1: {:02f} Top5: {:02f} Loss: {:02f}'.format(prec1, prec5, loss.item()))

    tboard.add_scalars('Val/Top1', {'prec1':prec1}, epoch)
    tboard.add_scalars('Val/Top5', {'prec5':prec5}, epoch)
    tboard.add_scalars('Val/Loss', {'loss':loss.item()}, epoch)

    return prec1




def main():
    args = parse_args()
    logger = setup_logger()
    logger.info(config)
    if not os.path.exists(config.ASSETS_PATH):
        os.makedirs(config.ASSETS_PATH)

    # ===== Create the dataloaders =====
    UCF_dataset = UCF101(config, logger)
    train_loader, val_loader, val_gt = UCF_dataset.get_loaders()

    # ===== Create the model =====
    model = FineTuneNet(config)
    logger.info('Built Model, using {} backbone'.format(config.MODEL.TRUNK))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    logger.info('Training on {} GPUs'.format(torch.cuda.device_count()))

    # ===== Set the optimizer =====
    optimizer = get_optimizer(model, config, pretraining = False)

    # ===== Get the loss =====
    criterion = nn.CrossEntropyLoss().cuda()

    # ===== Load checkpoint =====
    if config.TRAIN.FINETUNING.CHECKPOINT:
        checkpoint = torch.load(config.TRAIN.FINETUNING.CHECKPOINT)
        # ===== Align checkpoint keys with model =====
        if 'module' in list(checkpoint['state_dict'].keys())[0] and 'module' not in list(model.state_dict().keys())[0]:
            checkpoint['state_dict'] = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items() }
        elif 'module' not in list(checkpoint['state_dict'].keys())[0] and 'module' in list(model.state_dict().keys())[0]:
            checkpoint['state_dict'] = {'module.' + k:v for k,v in checkpoint['state_dict'].items() }
        
        if not config.TRAIN.FINETUNING.RESUME:
            # ===== Load only backbone parameters for starting finetuning at epoch 0 =====            
            forgiving_load_state_dict(checkpoint['state_dict'], model, logger)
            start_epoch = 1
            best_prec1 = 0
        else:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']       
    else:
        logger.warning('No checkpoint specified for pre-training.  Training from scratch.')
        start_epoch = 1
        best_prec1 = 0

    # ===== Set up save directory and TensorBoard =====
    assert config.TRAIN.FINETUNING.SAVEDIR, 'Please specify save directory'
    if not os.path.exists(config.TRAIN.FINETUNING.SAVEDIR):
        os.makedirs(config.TRAIN.FINETUNING.SAVEDIR)
        os.makedirs(os.path.join(config.TRAIN.FINETUNING.SAVEDIR, 'checkpoints'))
        os.makedirs(os.path.join(config.TRAIN.FINETUNING.SAVEDIR, 'tboard'))

    tboard = SummaryWriter(logdir = os.path.join(config.TRAIN.FINETUNING.SAVEDIR, 'tboard'))
    
    if args.val:
        logger.info('Running in Validation mode')
        validate(
                epoch = start_epoch,
                val_loader = val_loader,
                val_gt = val_gt,
                model = model,
                criterion = criterion,
                cfg = config,
                tboard = tboard,
                logger = logger,
            )
    else:
        # ===== Train Loop =====
        logger.info('Begin training')
        for epoch in range(start_epoch, config.TRAIN.FINETUNING.EPOCHS + 1):
            adjust_learning_rate(epoch, config, optimizer, pretraining = False, logger = logger)
            logger.info('Training epoch {}'.format(epoch))
            
        # ===== Train 1 epoch ======
            train(
                epoch = epoch,
                train_loader = train_loader,
                model = model,
                criterion = criterion,
                optimizer = optimizer,
                cfg = config,
                tboard = tboard,
                logger = logger
            )
            
            # ===== Validate periodically =====
            if epoch % config.TRAIN.FINETUNING.VAL_FREQ == 0:
                logger.info('Validating at epoch {}'.format(epoch))
                prec1 = validate(
                    epoch = epoch,
                    val_loader = val_loader,
                    val_gt = val_gt,
                    model = model,
                    criterion = criterion,
                    cfg = config,
                    tboard = tboard,
                    logger = logger,
                )

                # ===== Save if new best performance =====
                if prec1 > best_prec1:
                    best_prec1 = prec1
                    logger.info('New best top1 precision: {}'.format(best_prec1))
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_prec1': best_prec1
                    }
                    save_path = os.path.join(config.TRAIN.FINETUNING.SAVEDIR, 'checkpoints', 'best_checkpoint.pth')
                    torch.save(checkpoint, save_path)
            
            # ===== Save latest checkpoint =====
            checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_prec1': best_prec1
                    }
            save_path = os.path.join(config.TRAIN.FINETUNING.SAVEDIR, 'checkpoints', 'latest_checkpoint.pth')
            torch.save(checkpoint, save_path)

if __name__ == '__main__':
    main()