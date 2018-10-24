#%% Load Modules

import numpy as np

# Read Images
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv
from skimage import img_as_float

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

#%% Load Image

img_1 = rgb2hsv(imread('5_slanted.jpg'))
img_2 = rgb2hsv(imread('A_shade2.jpg'))
img_3 = rgb2hsv(imread('z_far.jpg'))

plt.figure()
plt.imshow(img_1, cmap = cm.hsv)

plt.figure()
plt.imshow(img_2, cmap = cm.hsv)

plt.figure()
plt.imshow(img_3, cmap = cm.hsv)

