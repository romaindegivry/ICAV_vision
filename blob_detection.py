#%% Load Modules

import numpy as np
import cv2

# Edge Detection
from skimage.feature import blob_doh
from skimage.transform import rescale

# Read Images
from scipy.misc import imread
from skimage.color import rgb2gray, rgb2hsv
from skimage import img_as_float

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

#%% Load Image

image_rgb = img_as_float(imread('5_far.jpg'))
image_rgb = rescale(image_rgb, 0.2)
image = rgb2gray(image_rgb)
image_hsv = rgb2hsv(image_rgb)

#%% Blob Detection

trg_colour = np.array([210, 100, 100])

blob_out = blob_doh(image_rgb[:,:,2])

plt.imshow(image_rgb[:,:,2], cmap = cm.gray)

for idx, blob in enumerate(blob_out):
    y, x, sigma = blob
    plt.plot(x, y, 'ob')

plt.show()

#%% Colour Mask

bound = np.array([[0.85 * 255, 0, 0], [0.05 * 255, 255, 255]])

mask = cv2.inRange(image_rgb * 255, bound[0], bound[1])
output = cv2.bitwise_and(image_rgb, image_rgb, mask = mask)

plt.figure()
plt.imshow(output)
plt.show()

#%%
plt.figure()
plt.imshow(image_rgb)
plt.show()