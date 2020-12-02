import os 
import cv2 
import torch 
import imageio
import argparse
import numpy as np
from glob import glob 
from tqdm import tqdm  
from glob import glob 

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network import PreTrainNet
from config import config, update_config
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train TCE Self-Supervised')
    parser.add_argument('--cfg', help = 'Path to config file', type = str, default = None)
    parser.add_argument('--target', help = 'Path to visualisation target.  Can be a video or folder of images', default = None)
    parser.add_argument('--ckpt', help = 'Checkpoint to visualise', type=str, required = True)
    parser.add_argument('--gif', action = 'store_true', default = False, help = 'Save output as a gif with the corresponding video alongside')
    parser.add_argument('--fps', type = float, help = 'Frames per second for gif', default = 30)
    parser.add_argument('--save', help = 'Save path', default = None)
    parser.add_argument('opts', help = 'Modify config using the command line', 
                         default = None, nargs=argparse.REMAINDER )
    args = parser.parse_args()
    update_config(config, args)

    return args



if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger()

    # ===== Get model =====
    model = PreTrainNet(config).eval()
    model = torch.nn.DataParallel(model).cuda()

    # ===== Load Model Checkpoint =====
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info('Loaded Checkpoint from {}'.format(args.ckpt))


    # ===== Get input transformation class =====
    create_blob =  get_blob(config)    

    # ===== Create either list of frames or videoreader object =====
    is_folder = os.path.isdir(args.target)
    if is_folder:
        imlist = sorted(glob(os.path.join(args.target, '*')))
        n_images = len(imlist)
    else:
        reader = cv2.VideoCapture(args.target)
        n_images = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        assert reader.isOpened(), 'Target was neither directory or a valid video file'

    # ===== Process video frames with network =====
    embeddings = []
    logger.info('Processing {} Frames'.format(n_images))
    for idx in tqdm(range(n_images)):
        if is_folder:
            frame = cv2.imread(imlist[idx])
        else:
            tf, frame = reader.read()
            if not tf:
                logger.info('Reached end of video at frame {} rather than {}'.format(x, n_images))
                break 
    
        # ===== Get image dimensions, convert from BGR to RGB, create input blob =====
        height, width = np.shape(frame)[:2]
        frame = frame[:, :, ::-1]
        blob = create_blob(frame)

        # ===== Get embedding from network =====
        with torch.no_grad():
            embeddings.append(model(blob).cpu().numpy())
    
    if not is_folder:
        reader.release()
    
    # ===== Use TSNE to reduce embeddings to 2D for plotting =====
    embeddings = np.array(embeddings)[:,0,:]
    reduced_embeddings = fit_tsne(embeddings, logger)


    # ===== Create Plot =====
    if args.gif:
        # ===== Create gif if flag set =====
        frames = create_gif_frames(reduced_embeddings, args.target, logger)
        imageio.mimsave(args.save, frames, duration = 1 / args.fps)
    else:
        # ===== Create TSNE single image =====
        fig, ax = plt.subplots()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        
        colormap = plt.cm.plasma(np.linspace(0,1, len(reduced_embeddings)))
        for idx, point in enumerate(reduced_embeddings):
            ax.plot(point[0], point[1], '.', color = colormap[idx])

        # ===== Write plot to image file =====
        fig.canvas.draw()
        tsne = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        Image.fromarray(tsne).save(args.save)




