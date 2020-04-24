from __future__ import print_function

import time
import numpy as np 

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def fit_pca(input_data, num_dims=0, threshold=0.85, max_error=0.05, debug=False):
    '''
    Performs PCA on the input data. Attempts to find the minimum number of 
    dimensions needed so that the variance ratio of the output data to the input
    data is greater than threshold. If num_dims is non-zero, will instead simply
    perform PCA to output that number of dimensions.

    Arguments:
        input_data : A [SHAPE] [TENSOR, ARRAY?] containing the data for the PCA, where: #TODO
            DIM INFO
        num_dims : The number of dimensions to reduce the input to.  If 0, will reduce to however many
                   dimentions required to surpass the threshold variance ratio
        threshold : Variance ratio of output and input data
        max_error : Set how close upper and lower bounds of the PCA need to be before they're considered converged
        debug : Set True to print PCA debug info
        
    '''

    optimised = False


    if num_dims > 0:
        # Fit PCA to num_dims output dimensions of num_dims is not 0
        pca = PCA(n_components=num_dims)

        # Fit PCA
        reduced = pca.fit_transform(input_data)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print("PCA transform  fitted. Explained variance in {} dims is {}.".format(num_dims, explained_variance))
        optimised = True
    
    # Get data input shape, set variables for the PCA loop
    # lower, upper is the lower and upper bounds we are trying to converge for PCA dimensionality to achieve 
    # the desired explained_variance, as a fraction of the original input dimensionality
    # previous is the previous attempt to fit the PCA's dimensionality
    dims = min(input_data.shape[0], input_data.shape[1])
    lower, upper = (0., 1.)
    previous = -1

    while not optimised:
        # Iterate the PCA until desired explained_variance achieved

        # Fit a PCA to dimensionality halfway between lower and upper bounds already found
        num_dims = int(dims * (0.5*(upper - lower) + lower))
        if num_dims == previous:
            # Settle on PCA dimensionality if this iteration has the same number of dimensions as the last
            num_dims = int(upper * dims)
            optimised = True

        # Fit PCA to num_dims for this iteration
        t1 = time.time()
        print('Fitting PCA : ', end = '\r')
        pca = PCA(n_components=num_dims)
        reduced = pca.fit_transform(input_data)
        print('Time Taken = {} seconds'.format(time.time() - t1))
        explained_variance = np.sum(pca.explained_variance_ratio_)
        previous = num_dims

        if debug:
            # Print debug info if debug = True
            print('Lower&Upper: ({}, {})\tNumber of dimensions: {}\tExplained Variance: {}'
            .format(lower, upper, num_dims, explained_variance))

        if explained_variance < threshold:
            # Raise lower bound if explained variance is high enough
            lower = num_dims / dims
        else:
            # Lower upper bound if explained variance is not high enough
            upper = num_dims / dims

        if upper - lower < max_error:
            # Settle on PCA dimensionality if upper and lower bounds within acceptable range of each other
            optimised = True

    return reduced


def fit_tsne(input_data, pca=True, pca_threshold=0.85, pca_error=0.05, pca_num_dims=0, num_dims=2, num_iterations=500, debug=False):
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
    # Reduce data with PCA first if pca=True
    if pca:
        input_data = fit_pca(input_data, num_dims=pca_num_dims, threshold=pca_threshold, max_error=pca_error, debug=debug)
    print(input_data.shape)
    t1 = time.time()
    print('Fitting TSNE:  ')
    tsne=TSNE(n_iter=num_iterations, n_components=num_dims)
    
    reduced = tsne.fit_transform(input_data)
    print('Time Taken = {} seconds'.format(time.time() - t1))

    return reduced