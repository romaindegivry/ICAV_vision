#%% Load Modules

import numpy as np

# Transformation for Line Detection
from skimage.transform import (hough_line, hough_line_peaks, rescale)
from skimage.transform import probabilistic_hough_line as phl
# Edge Detection
from skimage.filters import sobel
from skimage.feature import canny, blob_doh

# Read Images
from scipy.misc import imread
from skimage.color import rgb2gray
from skimage import img_as_float

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

#%% Load Image

image_rgb = img_as_float(imread('5_slanted.jpg'))
image = rgb2gray(image_rgb)
#image = rescale(image, 0.5)

#%% Sobel Edge Detection

edge_sobel = sobel(image)

edge_sobel[edge_sobel < 0.5 * np.max(edge_sobel)] = 0
edge_sobel[edge_sobel >= 0.5 * np.max(edge_sobel)] = 1

#%% Canny Edge Detection

edge_canny = canny(image, sigma = 5)

#%% Plot Edge Detection of Filters

fig_edge, axes_edge = plt.subplots(1, 2, figsize = (15,6))

axes_edge[0].imshow(edge_sobel, cmap = cm.gray)
axes_edge[0].set_title('Sobel')
axes_edge[0].set_axis_off()

axes_edge[1].imshow(edge_canny, cmap = cm.gray)
axes_edge[1].set_title('Canny')
axes_edge[1].set_axis_off()

filter_input = edge_canny

#%% Hough Transform

# Classic straight-line Hough transform
h, theta, d = hough_line(filter_input)

peaks_output = hough_line_peaks(h, theta, d, threshold = 0.5 * np.max(h))

lines = phl(filter_input)

#%% Plot Hough Transform and Intersection

# Generating Figure 1
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
ax = axes1.ravel()

ax[0].imshow(filter_input, cmap = cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(image, cmap = cm.gray)
for _, angle, dist in zip(*peaks_output):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax[1].plot((0, image.shape[1]), (y0, y1), '-r')

ax[1].set_xlim((0, image.shape[1]))
ax[1].set_ylim((image.shape[0], 0))

ax[1].set_axis_off()
ax[1].set_title('Detected lines')

plt.tight_layout()
plt.show()

# Generating Figure 2
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
ax = axes2.ravel()

ax[0].imshow(filter_input, cmap = cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(image, cmap = cm.gray)
for p1, p2 in lines:
    ax[1].plot((p1[0], p2[0]), (p1[1], p2[1]), '-r')

ax[1].set_xlim((0, image.shape[1]))
ax[1].set_ylim((image.shape[0], 0))

ax[1].set_axis_off()
ax[1].set_title('Detected lines')

plt.tight_layout()
plt.show()

#Code of Hough Space
#ax[2].imshow(np.log(1 + h),
#             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
#             cmap=cm.gray, aspect= 0.1)
#ax[2].set_title('Hough transform')
#ax[2].set_xlabel('Angles (degrees)')
#ax[2].set_ylabel('Distance (pixels)')
#ax[2].axis('image')