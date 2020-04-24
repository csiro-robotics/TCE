from __future__ import print_function

import os
import time
import pickle 
import argparse
import numpy as np

import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
from torchvision import transforms

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tsne_utils import fit_tsne
from dataloader.UCF101 import UCF101
from models.resnet import resnet18, resnet34, resnet50, resnet101 



def build_model(model_name, ckpt):
    '''
    Loads the network architecture

    Arguments:
        model : The name of the model architecture to be loaded
        ckpt : Path to pre-trained checkpoint
    '''

    if arg.model == "resnet18":
        model = resnet18(output_size=arg.output_size)
    elif arg.model == "resnet34":
        model = resnet34(output_size=arg.output_size)
    elif arg.model == "resnet50":
        model = resnet50(output_size=arg.output_size)
    elif arg.model == "resnet101":
        model = resnet101(output_size=arg.output_size)
    
    print('Built model : {}'.format(model_name))

    if arg.ckpt is not None:
        # Load model from checkpoint
        print('Loading checkpoint from {}'.format(arg.ckpt))
        checkpoint = torch.load(arg.ckpt)
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        checkpoint = {k.replace('module.', ''):v for k,v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    else:
        print('No checkpoint specified, model initialised with random weights')

    # Move to GPU(s) if available 
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    

    return model 

class to_blob():
    def __init__(self):
        '''
        Class to convert image to blob for network input 
        '''
        self.resize = transforms.Resize(256)
        self.crop = transforms.CenterCrop(224)
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        
    
    def __call__(self, image):
        '''
        Transform an image into a network input blob

        Arguments:
                image : RGB array for network input, stored as a numpy array with pixel values between 0 and 255
        '''
        image = Image.fromarray(image) # Prep for input to torchvision functions
        image.save('/scratch1/kni101/two-stream-clean/test_png.png')
        image = self.resize(image)
        image = self.crop(image)
        image = self.totensor(image)
        image = self.normalize(image)
        image = image.unsqueeze(0) # Set dim 0 to 1 for input to network (batch size = 1)

        # Move to GPU if available
        if torch.cuda.is_available():
            image = image.cuda()
        return image

def parse_option():
    '''
    Get command line arguments
    '''
    parser = argparse.ArgumentParser(description = 't-SNE visualisation code for TCE' )
    
    # Paths
    parser.add_argument('--embeddings', default = None, required = True, help = 'Path to numpy array containing embeddings')
    parser.add_argument('--output', default = None, required = True, help = 'Output for the t-SNE to be saved at')

    arg = parser.parse_args()

    return arg 


    

if __name__ == "__main__":
    arg = parse_option()
    
    # Load embeddings from path
    # Embeddings should have the shape N, D where:
        # N is the number of frames in the video sequence
        # D is the dimensionality of the embedded features
    # Embeddings should be sorted chronologically across the 0th index; i.e. embeddings[0,:] should be the embedded feature of the 0th frame and so on

    with open(arg.embeddings, 'rb') as f:
        embeddings = pickle.load(f)

    # Use t-SNE to reduce the dimensionality of the embeddings to 2 dimensions for plotting
    reduced_embeddings = fit_tsne(embeddings)
    
    
    # Create a t-SNE plot as an image output

    # Set up matplotlib figure
    fig, ax = plt.subplots()
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')

    # Generate color gradient for subsequent frames
    c = plt.cm.plasma(np.linspace(0,1,len(reduced_embeddings[list_idxs, 0])))

    # Plot graph
    for point in list_idxs:
        ax.plot(reduced_embeddings[point, 0], reduced_embeddings[point, 1], '.', color=c[point])


    # Write plot to image file
    fig.canvas.draw()
    tsne = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    tsne = cv2.cvtColor(tsne, cv2.COLOR_RGB2BGR)
    cv2.imwrite(arg.output, tsne)