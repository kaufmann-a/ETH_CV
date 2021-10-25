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
    return np.array([np.linalg.norm(x - cur_x) for cur_x in X])


def distance_batch(x, X):
    res = np.empty(shape=(0, len(X)))
    for x_ in x:
        res = np.append(res, [[np.linalg.norm(x_ - cur_x) for cur_x in X]], axis=0)
    return res


def gaussian(dist, bandwidth):
    res = np.exp(-0.5 * (dist/bandwidth)**2)
    return res

def update_point(weight, X):
    weighted_sum = sum([w * x for w, x in zip(weight, X)])
    avg = weighted_sum / sum(weight)
    return avg


def update_point_batch(weight, X):
    res = []
    for weight_ in weight:
        weighted_sum = sum([w*x for w, x in zip(weight_, X)])
        avg = weighted_sum / sum(weight_)
        res.append(avg)
    return torch.stack(res)


def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_


def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    nr_batches = 10
    nr_datapoints_batch = math.ceil(len(X_) / nr_batches)
    for i in  range(nr_batches):
        dist = distance_batch(X_[i*nr_datapoints_batch:(i+1)*nr_datapoints_batch], X)
        weight = gaussian(dist, bandwidth)
        X_[i * nr_datapoints_batch:(i + 1) * nr_datapoints_batch] = update_point_batch(weight, X_)
    return X_

def meanshift(X):
    X = X.clone()
    counter = 0
    for _ in range(20):
        X = meanshift_step(X)   # slow implementation
        print("Iteration " + str(counter) + " done")
        counter += 1
        # X = meanshift_step_batch(X)   # fast implementation
    return X


scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
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
