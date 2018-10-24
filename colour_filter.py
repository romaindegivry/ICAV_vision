#%% Colour Filtering

#%% Load Modules

import numpy as np
import cv2

from scipy.ndimage import binary_fill_holes

# Rescale
from skimage.transform import rescale, rotate

# Read Images
from scipy.misc import imread
from skimage.color import rgb2hsv, hsv2rgb

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

# Measure Region
from skimage.measure import regionprops, label
from skimage.segmentation import clear_border

#%% Load Image Function

img = rgb2hsv(rescale(imread('5_slanted.jpg'), 0.5))

#%% HSV Colour Bound

bound1 = np.array([[0, 0.1, 0.5], [0.05, 1, 1.0]])
bound2 = np.array([[0.8, 0.1, 0.5], [1.0, 1, 1.0]])

#%% Colour Segmentation

mask_1 = cv2.inRange(img, bound1[0], bound1[1])
mask_2 = cv2.inRange(img, bound2[0], bound2[1])
mask = mask_1 + mask_2

#mask_filter = binary_fill_holes(mask).astype(np.uint8)
mask_filter = mask
rgb_mask = cv2.bitwise_and(img, img, mask = mask_filter)

#%% Show Mask

fig = plt.figure()

fig.add_subplot(1,4,1)
plt.imshow(hsv2rgb(rgb_mask))

fig.add_subplot(1,4,2)
plt.imshow(mask, cmap = cm.gray)

fig.add_subplot(1,4,3)
plt.imshow(mask_filter, cmap = cm.gray)

#%% Region Properties

label_image = label(mask_filter)

prop_area = np.zeros((np.max(label_image), 2))

c_i = 0
reg = regionprops(label_image)
for prop in reg:

    prop_area[c_i, 0] = prop.label
    prop_area[c_i, 1] = prop.area

    c_i += 1

trg_idx = np.argmax(prop_area, axis = 0)[1]

trg_img = label_image == prop_area[trg_idx, 0]
trg_prop = reg[trg_idx]


fig.add_subplot(1,4,4)
plt.imshow(trg_img, cmap = cm.gray)

plt.show()

#%% Target Region Prop

fig = plt.figure()

fig.add_subplot(1,2,1)
plt.imshow(trg_prop.filled_image, cmap = cm.gray)

fig.add_subplot(1,2,2)
plt.imshow(trg_prop.image, cmap = cm.gray)

#%% Rotation

orient = np.rad2deg(trg_prop.orientation)

image_1 = rotate(trg_prop.image, orient)
image_2 = rotate(trg_prop.image, -orient + 90)
image_3 = rotate(trg_prop.image, -orient + 180)
image_4 = rotate(trg_prop.image, -orient + 270)

fig = plt.figure()

fig.add_subplot(1,4,1)
plt.imshow(image_1, cmap = cm.gray)

fig.add_subplot(1,4,2)
plt.imshow(image_2, cmap = cm.gray)

fig.add_subplot(1,4,3)
plt.imshow(image_3, cmap = cm.gray)

fig.add_subplot(1,4,4)
plt.imshow(image_4, cmap = cm.gray)
