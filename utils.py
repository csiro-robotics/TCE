from __future__ import print_function

import os
import torch
import numpy as np
import pandas as pd

def accuracy(output, target, topk=(1,)):
    '''
    Computes the top-1 accuracy @k for the specified values of k
    
    Arguments:
        output : Output predictions from the network
        target : Ground truth labels
        topk : A tuple containing values for k at which we want top-1 accuracy calculated
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Calculate top-k accuracy for each value of k 
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    def __init__(self):
        '''
        Computes and stores the average and current value
        Used to record values for TensorBoard
        '''
        self.reset()

    def reset(self):
        '''
        Resets AverageMeter
        '''
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return None
    def update(self, val, n=1):
        '''
        Updates AverageMeter with latest value
        '''
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return None

def save_checkpoint(state, is_best, checkpoint, model_best):
    '''
    Saves checkpoint from training

    Arguments:
        state : state dictionary of latest epoch of training
        is_best : True if latest epoch of training has the best performance so far, False otherwise
        checkpoint : Path to save latest checkpoint to
        model_best : Path to save checkpoint to if is_best is True
    '''
    # Save latest checkpoint
    torch.save(state, checkpoint)

    # Save checkpoint if it is the best epoch trained so far
    if is_best:
        shutil.copyfile(checkpoint, model_best)
    return None 

def record_info(info,filename,mode):
    '''
    Saves the training history of the network to a csv file, including average batch time, data loading time,
    loss, top-1 accuracy, top-5 accuracy, and learning rate for each training and validation epoch
    
    Arguments : 
        info : Dictionary containing 
    '''
    if mode =='train':
        # Record info for a training epoch
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']
        
    if mode =='test':
        # Record info for a validation epoch
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']
    
    if not os.path.isfile(filename):
        # Create file if does not exist yet
        df.to_csv(filename,index=False,columns=column_names)
    else:
        # Write to existing file
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)
