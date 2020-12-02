import os 
import cv2 
import time
import numpy as np 
from tqdm import tqdm 
from glob import glob 

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import PIL.Image as Image 
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def create_gif_frames(reduced_embeddings, target, logger):
    
    # ===== Set up axis and points =====
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    embeddings_y = reduced_embeddings[:,0]
    embeddings_x = reduced_embeddings[:,1]

    # ===== Set up frame loader =====
    is_folder = os.path.isdir(target)
    if is_folder:
        imlist = sorted(glob(os.path.join(target, '*')))
        n_images = len(imlist)
    else:
        reader = cv2.VideoCapture(target)
        n_images = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        assert reader.isOpened(), 'Target was neither directory or a valid video file'
    # ===== Iterate over frames =====
    frames = []
    logger.info('Beginning frame generation')

    for idx in tqdm(range(n_images)):
        ax.cla()
        # ===== Get frame =====
        if is_folder:
            frame = cv2.imread(imlist[idx])
        else:
            tf, frame = reader.read()
            if not tf:
                logger.info('Reached end of video at frame {} rather than {}'.format(x, n_images))
                break
        h, w = frame.shape[:2]
        frame = frame[:,:,::-1]
        height, width = frame.shape[:2]
        # ===== Plot on graph, highlighting the current frame's embedding =====
        x = list(embeddings_x[:idx]) + list(embeddings_x[idx+1:])
        y = list(embeddings_y[:idx]) + list(embeddings_y[idx+1:])
        ax.plot(x,y,',', color='b')
        ax.plot(embeddings_x[idx], embeddings_y[idx], 'o', color='r')
        fig.canvas.draw()

        tsne = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        tsne = cv2.resize(tsne, (width,height))
        frames.append(np.concatenate((frame, tsne), axis = 1))
        
    return frames 

class get_blob:
    def __init__(self, cfg):
        self.resize = transforms.Resize(cfg.VISUALISATION.TSNE.RESIZE)
        self.crop = transforms.CenterCrop(cfg.VISUALISATION.TSNE.CROP_SIZE)
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean = cfg.DATASET.PRETRAINING.MEAN,
            std = cfg.DATASET.PRETRAINING.STD
        )

    def __call__(self, frame):
        frame = Image.fromarray(frame)
        frame = self.resize(frame)
        frame = self.crop(frame)
        frame = self.totensor(frame)
        frame = self.normalize(frame)
        frame = frame.unsqueeze(0)
        frame = frame.cuda()

        return frame

def fit_pca(input_data, logger, num_dims=0, threshold=0.85, max_error=0.05, debug=False):
    
    optimised = False
    # ===== Fit PCA to num_dims output dimensions of num_dims is not 0 =====
    if num_dims > 0:
        pca = PCA(n_components=num_dims)
        # ===== Fit PCA =====
        reduced = pca.fit_transform(input_data)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        logger.info("PCA transform  fitted. Explained variance in {} dims is {}.".format(num_dims, explained_variance))
        optimised = True
    
    """
    Get data input shape, set variables for the PCA loop
    lower, upper is the lower and upper bounds we are trying to converge for PCA dimensionality to achieve 
    the desired explained_variance, as a fraction of the original input dimensionality
    previous is the previous attempt to fit the PCA's dimensionality
    """
    dims = min(input_data.shape[0], input_data.shape[1])
    lower, upper = (0., 1.)
    previous = -1

    while not optimised:
        # ===== Iterate the PCA until desired explained_variance achieved ===== 

        # ===== Fit a PCA to dimensionality halfway between lower and upper bounds already found ===== 
        num_dims = int(dims * (0.5*(upper - lower) + lower))
        if num_dims == previous:
            # ===== Settle on PCA dimensionality if this iteration has the same number of dimensions as the last ===== 
            num_dims = int(upper * dims)
            optimised = True

        # ===== Fit PCA to num_dims for this iteration ===== 
        t1 = time.time()
        logger.info('Fitting PCA')
        pca = PCA(n_components=num_dims)
        reduced = pca.fit_transform(input_data)
        logger.info('Time Taken = {} seconds'.format(time.time() - t1))
        explained_variance = np.sum(pca.explained_variance_ratio_)
        previous = num_dims

        if debug:
            logger.info('Lower&Upper: ({}, {})\tNumber of dimensions: {}\tExplained Variance: {}'
            .format(lower, upper, num_dims, explained_variance))

        if explained_variance < threshold:
            # ===== Raise lower bound if explained variance is high enough ===== 
            lower = num_dims / dims
        else:
            # ===== Lower upper bound if explained variance is not high enough ===== 
            upper = num_dims / dims

        if upper - lower < max_error:
            # ===== Settle on PCA dimensionality if upper and lower bounds within acceptable range of each other ===== 
            optimised = True

    return reduced

def fit_tsne(input_data, logger, pca=True, pca_threshold=0.85, pca_error=0.05, pca_num_dims=0, num_dims=2, num_iterations=500, debug=False):
    '''
    Performs TSNE on the input data to reduce it to num_dims dimensions.
    Will first perform PCA by default to reduce the number of dimensions and make
    fitting tsne faster

    Arguments:
        input_data : A [SHAPE] [TENSOR, ARRAY?] containing the data for the PCA, where: #TODO
            DIM INFO
        pca : If True, erform some dimensionality reduction with a PCA before the TSNE to reduce computation time
        pca_threshold : Explained variance threshold for PCA 
        pca_error : Acceptable distance between lower and upper bounds for the pca to be considered converged, as a value between
                    0 and 1
        pca_num_dims : If set to a non-zero value, PCA will reduce input to num_dims dimensions
        num_dims : Number of dimensions to reduce to using TSNE
        num_iterations : Number of iterations to run the TSNE for
        debug : If True, print debug information
    '''
    # ===== Reduce data with PCA first if pca=True ===== 
    if pca:
        input_data = fit_pca(input_data, logger, num_dims=pca_num_dims, threshold=pca_threshold, max_error=pca_error, debug=debug)
    t1 = time.time()
    logger.info('Fitting TSNE:  ')
    tsne=TSNE(n_iter=num_iterations, n_components=num_dims)
    
    reduced = tsne.fit_transform(input_data)
    logger.info('Time Taken = {} seconds'.format(time.time() - t1))

    return reduced