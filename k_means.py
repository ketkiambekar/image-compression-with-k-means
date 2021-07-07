from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START CODE HERE ***
    H,W,C=image.shape
    #print(H,W,C)
    centroids=[]
    
    #Check if there are num_clusters number of unique values in centroid
    while len(centroids)<num_clusters :

        #Generate Random H,W,C
        H_r = random.randint(0,H-1)
        W_r = random.randint(0,W-1)
        #C_r = random.randint(0,C-1)
        
        #print(image[H_r,W_r])
      
        centroids.append(image[H_r,W_r])
        
    centroids_init=np.array(centroids)
    # *** END CODE HERE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START CODE HERE ***
    max_iter=5
    H, W, C = image.shape
    cluster_info = np.zeros(shape=(H,W,1))
    num_centroids = len(centroids)

    for m in range (0, max_iter):

        #Loop to determine nearest centroid for each pixel
        for i in range(0,H ):
            for j in range(0,W):  
                
                #print(image[i][j])
                distances = centroids - image[i][j]

                dist = np.sum(distances**2, axis=1)
             
                #Color the Dots
                cluster_info[i][j]= np.argmin(dist, axis=0)     

        #Update Centroids
        new_cent=[]
        for k in range(0,num_centroids):
            cluster=[]
            np_cluster=None
            #select all points with centroid k from cluster_info
            for i in range(0,H ):
                for j in range(0,W): 
                    if cluster_info[i][j]==k:
                        cluster.append(image[i][j])

            np_cluster =  np.array(cluster)

            #calculate mean
            new_cent.append( np.mean(cluster, axis=0))
        
        #Add mean to list of new centroids
        new_centroids = np.array(new_cent)
        if np.array_equal(new_centroids, centroids):
            break
        
        centroids = new_centroids
        if m%print_every==0:
            print('iteration {}: Centroids are {} '.format(m,centroids))


    # *** END CODE HERE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START CODE HERE ***
    H, W, C = image.shape

    for i in range (0, H):
        for j in range (0, W):

            #Find closest centroid to the image:
            distances = centroids - image[i][j]
            dist = np.sum(distances**2, axis=1)
            
            image[i][j]= centroids[int(np.argmin(dist, axis=0))]

    # *** END CODE HERE ***
    print(type(image))

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
