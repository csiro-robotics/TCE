from __future__ import print_function

import os
import time
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
    
    # Configuration
    parser.add_argument('--model', default = 'resnet50', help = 'Name of network architecture')
    parser.add_argument('--vid', action = 'store_true', help = 'Save t-SNE as a video file, showing embedding behaviour in the video over time')
    parser.add_argument('--output_size', type = int, default = 128, help = 'Size of the output vector')

    # Paths
    parser.add_argument('--input', default = None, required = True, nargs = '+', help = 'Videos to be turned into a t-SNE.  Accepts either video files of directory of frames as an input.  Can take a sequence of videos when making an image file t-SNE')
    parser.add_argument('--output', default = None, required = True, help = 'Output for the t-SNE to be saved at')
    parser.add_argument('--ckpt', default = None, help = 'Path to pre-trained checkpoint')

    arg = parser.parse_args()

    return arg 


    

if __name__ == "__main__":
    arg = parse_option()
    
    # Create class to handle converting images into network inputs
    to_blob = to_blob()
    
    # Build network architecture and load checkpoint if available
    model = build_model(arg.model, arg.ckpt)

    # Create empty lists to store embeddings and video ids
    embeddings = []
    vid_idxs = []

    for vid_idx, video in enumerate(arg.input):
        # Iterate over the video paths provided
        if os.path.isdir(video):
            # If a directory of frames is provided, create a temporally sorted list of frames
            imglist = [os.path.join(video, imname) for imname in sorted(os.listdir(video))]
            n_images = len(imglist)
        else:
            # If a video file is provided, use cv2 to create a video reader
            reader = cv2.VideoCapture(video)
            n_images = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            assert reader.isOpened(), 'Input was neither a directory nor a valid video file.'
        
        for x in range(n_images):
            # Iterate over all the frames in a video to retrieve their embeddings
            if x % 10 == 0:
                print('Running inference on frame {} now'.format(x))
            t1 = time.time()
            if os.path.isdir(video):
                # Load frame from image file
                im = cv2.imread(imglist[x])
            else:
                # Load frame from video reader
                tf, im = reader.read()
                if not tf:
                    print('Reached end of video at frame {} rather than {}'.format(x, n_images))
                    break
            
            # Get image dimensions, convert from BGR to RGB, create network input blob
            height, width = np.shape(im)[:2]
            im = im[:, :, ::-1]
            blob = to_blob(im)
            
            t2 = time.time()

            # Get feature embedding from network, append embedding and video_idx to lists
            feature_embedding = model(blob)
            embeddings.append(feature_embedding.cpu().detach().numpy())
            vid_idxs.append(vid_idx)
            t3 = time.time()
            print('Times: {:.3f}s | {:.3f}s || {:.3f}s total'.format(t2 - t1, t3 - t2, t3-t1))
        
        if not os.path.isdir(video):
            reader.release()

    embeddings = np.array(embeddings)[:, 0, :]
    
    # Use t-SNE to reduce the dimensionality of the embeddings to 2 dimensions for plotting
    reduced_embeddings = fit_tsne(embeddings)
    
    
    # Create Plot 

    if arg.vid:
        # Create an mp4 t-sne visualisation
        # Only provide a single video for visualisation if you're using this argument
        if not os.path.isdir(video):
            # Create video reader if a video file is provided
            reader = cv2.VideoCapture(video)

        # Set up matplotlib figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.spines['bottom'].set_color('0.5')
        ax.spines['top'].set_color('0.5')
        ax.spines['right'].set_color('0.5')
        ax.spines['left'].set_color('0.5')
        lines = []

        # Set up plot to iterate over : line2 plots the current frame in bright red, line1 plots all other frames in blue
        for vid_idx in np.unique(vid_idxs):
            list_idxs = np.where(vid_idxs == vid_idx)
            line1 = ax.plot(reduced_embeddings[list_idxs, 0], reduced_embeddings[list_idxs, 1], '.', color='b')
            line2 = ax.plot(reduced_embeddings[list_idxs[0], 0], reduced_embeddings[list_idxs[0], 1], 'o', color='r')
            lines.append((line1, line2))

        # Create video writer
        writer = cv2.VideoWriter(arg.output, 0x7634706d, 15, (2*width, height))
        
        for x in range(n_images):
            # Iterate over frames
            if x % 10 == 0:
                print('Processing and writing frame {} to {}'.format(x, arg.output), end='\r')
            if os.path.isdir(arg.input[0]):
                    frame = cv.imread(imglist[x])
            else:
                tf, frame = reader.read()
                assert tf
            height, width = np.shape(frame)[:2]

            # Create plot with current frame highlighted in red, draw to BGR numpy array
            for line1, line2 in lines:
                line2[0].set_xdata(reduced_embeddings[x, 0])
                line2[0].set_ydata(reduced_embeddings[x, 1])
            fig.canvas.draw()
            tsne = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
            tsne = cv2.resize(tsne, (width,height))
            tsne = cv2.cvtColor(tsne, cv2.COLOR_RGB2BGR)

            # Plot t-SNE side-by-side with the current BGR frame of the video
            newframe = np.zeros((height, 2*width, 3), dtype=np.uint8)
            newframe[:, :width, :] = tsne
            newframe[:, width:, :] = frame
            writer.write(newframe)

        writer.release()
        print('\n')
    
    else:
        # Create a t-SNE plot as an image output
        # Accepts multiple videos as inputs, plots each video in a seperate colour
        # If only a single video is input, will plot the t-SNE with a gradient of colour to show temporal progression

        # Set up matplotlib figure
        fig, ax = plt.subplots()
        colors = cm.rainbow(np.linspace(0, 1, len(arg.input)))
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.spines['bottom'].set_color('0.5')
        ax.spines['top'].set_color('0.5')
        ax.spines['right'].set_color('0.5')
        ax.spines['left'].set_color('0.5')

        for x in range(len(arg.input)):
            # Iterate over input videos
            
            # Get indexes of embedding list corresponding to this video
            list_idxs = np.where(np.array(vid_idxs) == x)[0]

            if len(arg.input) == 1:
                # Use colour gradient to show temporal progression if only a single video is given as input
                c = plt.cm.plasma(np.linspace(0,1,len(reduced_embeddings[list_idxs, 0])))

                # Plot graph
                for point in list_idxs:
                    ax.plot(reduced_embeddings[point, 0], reduced_embeddings[point, 1], '.', color=c[point])

            else:
                # Plot each different input video in a different colour
                ax.plot(reduced_embeddings[list_idxs, 0], reduced_embeddings[list_idxs, 1], '.', color=colors[x],label=os.path.basename(arg.input[x]))
                ax.legend()


        # Write plot to image file
        fig.canvas.draw()
        tsne = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        tsne = cv2.cvtColor(tsne, cv2.COLOR_RGB2BGR)
        cv2.imwrite(arg.output, tsne)