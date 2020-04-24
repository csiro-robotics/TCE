'''
Finetuning pre-trained resnet weights on action recognition using the stack-of-differences encoder
'''
from __future__ import print_function

import os
import time 
import pickle 
import argparse 
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn 
from torch.autograd import Variable 

from torchvision import transforms 
from tensorboardX import SummaryWriter

from dataloader.UCF101 import UCF101
from models.resnet_finetune import resnet18, resnet34, resnet50, resnet101 
from utils import accuracy, AverageMeter, save_checkpoint, record_info

# Change these paths to reflect the paths to the UCF101 frame and split files on your machine
__UCF101DIREC__ = '/datasets/work/d61-eif/source/UCF-101-twostream/jpegs_256/'
__UCF101SPLITS__ = '/datasets/work/d61-eif/source/UCF-101-twostream/UCF_list/'

def parse_option():
    '''
    Get command line arguments
    '''
    parser = argparse.ArgumentParser(description = 'Finetuning pre-trained resnet weights on action recognition using the stack-of-differences encoder' )

    # Configuration
    parser.add_argument('--epochs', default = 600, type = int, help = 'Number of training epochs')
    parser.add_argument('--batch-size', default = 25, type = int, help = 'Mini-batch size')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on validation set')
    parser.add_argument('--model', default='resnet50', type=str, help='Model to use for network backbone')
    parser.add_argument('--split', default = '01', type = str, help = 'Train / Test Split for dataset, if relevant')
    parser.add_argument('--dataset', default='UCF101', type = str, help = 'Dataset to use for training and testing, UCF101 by default')

    # Optimization
    parser.add_argument('--lr', default=0.05, type = float, help='Initial Learning Rate')
    parser.add_argument('--lr_decay_rate', default = 0.1, type = float, help = 'Rate at which to decay the learning rate at drop epochs')
    parser.add_argument('--lr_decay_epochs', nargs='+', default=[375], help = 'Epochs at which to decay the learning rate ')
    
    # Paths
    parser.add_argument('--weights', default = '', type = str, help = 'Path to initial weights for finetuning')
    parser.add_argument('--resume', default = '', type = str, help = 'Path to partially trained checkopint to resume training')
    parser.add_argument('--savedir', default='savedir')
    # Miscellaneous
    parser.add_argument('--progress', dest='prog', action='store_true', help='Set true to show tqdm progress bar during training')
    
    # Parse arguments
    arg = parser.parse_args()

    return arg 


def main():
    arg = parse_option()

    # Set up save directories and Tensorboard
    if not os.path.exists(arg.savedir):
        os.mkdir(arg.savedir)
    if not os.path.exists(os.path.join(arg.savedir,'tensorboard')):
        os.mkdir(os.path.join(arg.savedir,'tensorboard'))
    writer = SummaryWriter(logdir=os.path.join(arg.savedir,'tensorboard'))
    if not os.path.exists(os.path.join(arg.savedir,'models')):
        os.mkdir(os.path.join(arg.savedir,'models'))
    
    # Prepare DataLoader
    # Add custom datasets here
    if arg.dataset == 'UCF101':
        data_loader = UCF101(
                            batch_size = arg.batch_size,
                            num_workers = 8,
                            path = __UCF101DIREC__,
                            ucf_list = __UCF101SPLITS__,
                            ucf_split = arg.split, 
                            )

    
    train_loader, test_loader, test_video = data_loader.run()
    
    #Model 
    model = Finetuner(
                        nb_epochs = arg.epochs,
                        lr = arg.lr,
                        batch_size = arg.batch_size,
                        resume = arg.resume,
                        evaluate = arg.evaluate,
                        train_loader = train_loader,
                        test_loader = test_loader,
                        test_video = test_video,
                        weights = arg.weights,
                        arg = arg, 
                        writer = writer
    )
    #Training
    model.run()
    writer.close()

class Finetuner():
    def __init__(self, nb_epochs, lr, batch_size, resume, evaluate, train_loader, test_loader, test_video, weights, arg, writer):
        '''
        Class that handles the finetuning model, including training and testing

        Arguments: 
            nb_epochs : Number of training epochs
            lr : Initial learning rate
            batch_size : Batch size
            resume : Path to checkpoint to resume from, None if not resuming
            evaluate : If True, run in evaluation mode
            train_loader : Dataloader for train set
            test_loader : Dataloader for test set
            test_video : List of labels for test set videos
            weights : Path to initial weights for pre-training, 
        '''
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.start_epoch = 0
        self.evaluate = evaluate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.test_video = test_video
        self.weights = weights
        self.arg = arg
        self.writer = writer
    
    def adjust_learning_rate(self, prev_lr, decay_epochs, lr_decay_rate):
        '''
        Adjust the learning rate based on the epoch
        
        Arguments:
            prev_lr : Previous learning rate
            decay_epochs : List of epochs at which to decay the learning rate
            lr_decay_rate : Rate at which to decay the learning rate
        '''
        if self.epoch in decay_epochs:
            lr = prev_lr * lr_decay_rate
            print("Adjusting learning rate to {}".format(lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def build_model(self):
        '''
        Builds the network model for finetuning and evaluation
        '''
        print('==> Build model and setup loss and optimizer')

        # 15 input channels used for six-frame stack of differences network input
        if self.arg.model == "resnet18":
            self.model = resnet18(pretrained=False, channel=15)
        elif self.arg.model == "resnet34":
            self.model = resnet34(pretrained=False, channel=15)
        elif self.arg.model == "resnet50":
            self.model = resnet50(pretrained=False, channel=15)
        elif self.arg.model == "resnet101":
            self.model = resnet101(pretrained=False, channel=15)
        
        # If arg.weights is specified, load initial weights into the network
        if os.path.isfile(self.weights):
            self.model.load_my_state_dict(self.weights)
        else:
            print("No Initial Weights found")

        # Handle multi-GPU training
        self.model = torch.nn.DataParallel(self.model).cuda()

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)

    def resume_and_evaluate(self):
        '''
        Resume a previously loaded checkpoint, and run evaluation if eval mode is specified
        '''
        # Load previous checkpoint and training progress
        if self.resume and not self.weights:
            if os.path.isfile(self.resume):
                print("==> Loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])

                print("\n\n==> Loaded checkpoint '{}' (Epoch {}) (Best Top-1 Accuracy {})\n\n"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> No checkpoint found at '{}'".format(self.resume))

        # Run evaluation if in eval mode
        if self.evaluate:
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        '''
        Finetune and evaluate the model on the specified dataset
        '''
        # Build the model for training and testing
        self.build_model()
        
        # Resume from a previous checkpoint if specified, evaluate if in evaluation mode
        self.resume_and_evaluate()

        # If in evaluation mode, exit
        if self.evaluate:
            return
        
        cudnn.benchmark = True
        is_best = False
        
        # Train loop for each epoch 
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            # Train the network
            self.train_1epoch()

            # Adjust the learning rate
            self.adjust_learning_rate(prev_lr = self.optimizer.param_groups[0]['lr'], 
                                      decay_epochs = self.arg.lr_decay_epochs, 
                                      lr_decay_rate = self.arg.lr_decay_rate
                                      )

            # Evaluate on test set every third epoch
            if self.epoch % 3 == 0:
                prec1, val_loss = self.validate_1epoch()
                is_best = prec1 > self.best_prec1
                
                
                # Save predictions if this epoch was the best
                if is_best:
                    self.best_prec1 = prec1
                    with open('{}_{}_{}.pickle'.format(os.path.join(self.arg.savedir,'spatial_video_preds'), self.arg.model, self.arg.dataset), 'wb') as f:
                        pickle.dump(self.dic_video_level_preds, f)
                    f.close()
                
            # Save latest checkpoint, and update best checkpoint if this epoch is the best 
            save_checkpoint(
                state = {'epoch': self.epoch,
                        'state_dict': self.model.state_dict(),
                        'best_prec1': self.best_prec1,
                        'optimizer' : self.optimizer.state_dict()
                        },
                is_best = is_best,
                checkpoint = '{}_{}_{}.pth.tar'.format(os.path.join(self.arg.savedir,'models','latest_checkpoint'), self.arg.model, self.arg.dataset),
                model_best = '{}_{}_{}.pth.tar'.format(os.path.join(self.arg.savedir,'models','best_checkpoint'), self.arg.model, self.arg.dataset)
                )
    
    def train_1epoch(self):
        '''
        Train for a single epoch
        '''
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Switch to train mode
        self.model.train()    
        end = time.time()

        # Mini-batch training
        if self.arg.prog:
            progress = tqdm(self.train_loader)
        else:
            progress = self.train_loader

        # Iterate over dataset 
        for i, (data,label) in enumerate(progress):
    
            # Measure data loading time
            data_time.update(time.time() - end)
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # Compute loss
            input_var = Variable(data).cuda()
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # Compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Measure elapsed time and record results
            batch_time.update(time.time() - end)
            end = time.time()
            self.writer.add_scalars("Loss", {"val": losses.val, "average": losses.avg}, (self.epoch)*len(self.train_loader) + i)
            self.writer.add_scalars("Acc@1", {"val": top1.val, "average": top1.avg}, (self.epoch)*len(self.train_loader) + i)
            self.writer.add_scalars("Acc@5", {"val": top5.val, "average": top5.avg}, (self.epoch)*len(self.train_loader) + i)

            # Print every 10 iterations
            if i % 10 == 0:
                print('Loss {loss.val:.4f} ({loss.avg:.4f})\tAcc@1 {top1.val:.3f} ({top1.avg:.3f})\tAcc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(loss=losses,
                    top1=top1, top5=top5))
        
        # Write to TensorBoard
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        # Save training info to CSV
        record_info(info, os.path.join(self.arg.savedir,'rgb_train.csv'),'train')

        return None

    def frame2_video_level_accuracy(self):
        '''
        Return predictions for top-1 accuracy, top-5 accuracy and total loss during evaluation
        '''
        
        self.dic_class_level_preds = {}
        correct = 0
        
        # Create numpy arrays containing predictions and ground truths for the evaluated results
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        
        ii = 0
        minus = 1

        # Iterate over test videos
        for name in sorted(self.dic_video_level_preds.keys()):
            
            # Get predictions and ground truth labels for video
            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name]) - 1 
            
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii += 1


            if np.argmax(preds) == (label):
                # Append video to dictionary of correct results if correct prediction
                correct += 1
                if self.dic_class_level_preds.has_key('{}_{}'.format(str(label),'correct')):
                    self.dic_class_level_preds['{}_{}'.format(str(label),'correct')] += 1
                else:
                    self.dic_class_level_preds['{}_{}'.format(str(label),'correct')] = 1
            else:
                # Append video to dictionary of incorrect results if incorrect prediction
                if self.dic_class_level_preds.has_key('{}_{}'.format(str(label),'incorrect')):
                    self.dic_class_level_preds['{}_{}'.format(str(label),'incorrect')] += 1
                else:
                    self.dic_class_level_preds['{}_{}'.format(str(label),'incorrect')] = 1
            self.dic_class_level_preds[label] = name

        

        # Calculate top-1 and top-5 accuracy
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))

        # Calculate evaluation loss and output it as a numpy tensor
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())
        loss = loss.data.cpu().numpy()     

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
        
        return top1, top5, loss

    def validate_1epoch(self):
        '''
        Run validation on the test set
        '''
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        if self.arg.prog:
            progress = tqdm(self.test_loader)
        else:
            progress = self.test_loader

        # Iterate over dataset
        for i, (keys,data,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # Compute Loss
            output = self.model(data_var)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('/',1)[0]
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]

        # Calculate and record top-1 accuracy, top-5 accuracy, and evaluation loss
        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
        losses.update(video_loss)
        top1.update(video_top1)
        top5.update(video_top5)
        self.writer.add_scalars("Eval Loss", {"val": losses.val, "average": losses.avg}, self.epoch)
        self.writer.add_scalars("Eval Acc@1", {"val": top1.val, "average": top1.avg}, self.epoch)
        self.writer.add_scalars("Eval Acc@5", {"val": top5.val, "average": top5.avg}, self.epoch)

        # Record to TensorBoard
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]}
        record_info(info, os.path.join(self.arg.savedir,'rgb_train.csv'),'test')
        return video_top1, video_loss

def softmax(x):
    '''
    Compute softmax values for each set of scores in x
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()





if __name__=='__main__':
    main()
