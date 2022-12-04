import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    # L2 distances are the 2-norm magnitude of the difference vectors
    return torch.linalg.norm(x-X, dim=1)

def distance_batch(x, X):
    # same as for un-vectorized, but here 'X' is batched so just repeat each 'x'
    # value nPoints time along the 2nd axis so that shapes match for the
    # substraction
    d = X - torch.unsqueeze(x,1).repeat(1,x.shape[0],1)
    return torch.linalg.norm(d, dim=2)

def gaussian(dist, bandwidth):
    # same for both version, the weights are normalized in the update step
    return torch.exp(-torch.square(dist) / (2 * bandwidth*bandwidth))

def update_point(weight, X):
    # weight points with the normalized weights and sum up their coordinates for
    # each dimension to get the updated point
    _weighted = torch.transpose(torch.unsqueeze(weight,0), 0, 1) * X
    return torch.sum(_weighted, dim=0)/torch.sum(weight)

def update_point_batch(weight, X):
    # same as non-vectorized just extend weights so shapes match for the
    # multiplication and sum (for normalization)
    _weighted = torch.unsqueeze(weight,2).repeat(1,1,X.shape[2]) * X

    return torch.sum(_weighted, dim=1)/torch.unsqueeze(torch.sum(weight, dim=1),1).repeat(1,X.shape[2])

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    # create a working copy of X_ for all points
    X_ = torch.unsqueeze(X_, 0).repeat(X_.shape[0],1,1)
    dist = distance_batch(X, X_)
    weight = gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X_)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
#X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
