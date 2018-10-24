#%% Integrated Script - Static Image

#%% Load Modules

# Main Modules
import numpy as np
import cv2

# Std Lib
import subprocess as subpro
import time

# Read Images
from imageio import imread, imwrite

# Image Manipulation
from skimage.color import hsv2rgb, rgb2hsv
from skimage.transform import rescale, rotate

# Morphology Operations
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries

# Hough Transform
from skimage.transform import probabilistic_hough_line as hough

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

#%% Colour Segmentation

def run_colour_segmentation(img_hsv):

#    # Bounds on HSV Colour Space
    bound1 = np.array([[0, 0.1, 0.5], [0.1, 1.0, 1.0]])
    bound2 = np.array([[0.8, 0.1, 0.5], [1.0, 1.0, 1.0]])

    # Bounds on HSV Colour Space
#    bound1 = np.array([[0, 25, 127], [18, 255, 255]])
#    bound2 = np.array([[144, 25, 127], [180, 255, 255]])

    # Filter Image
    mask1 = cv2.inRange(img_hsv, bound1[0], bound1[1])
    mask2 = cv2.inRange(img_hsv, bound2[0], bound2[1])
    mask = mask1 + mask2

    mask_rgb = hsv2rgb(cv2.bitwise_and(img_hsv, img_hsv, mask = mask))

    return mask_rgb, mask

#%% Morphology Operation

def run_morph(mask):

    # Extract Region Properties
    labelled_mask = label(mask)
    region_mask = regionprops(labelled_mask)

    # Extract Biggest Region
    area = 0
    idx_trg = 0
    count = 0
    for prop in region_mask:
        if prop.area > area:
            area = prop.area
            idx_trg = count

        count += 1

    prop = regionprops((region_mask[idx_trg].image).astype(np.uint8))[0]

    return prop, labelled_mask, region_mask

#%% Hough Transform

def run_hough_transform(mask_filled, theta_in):

    angle_bound = np.deg2rad(np.linspace(*theta_in))

    # Detect Lines
    lines = hough(find_boundaries(mask_filled),
                  theta = angle_bound)

    # Compute Gradients
    gradients = np.zeros(len(lines))

    i = 0
    for p1, p2 in lines:
        gradients[i] = np.divide((p1[1] - p2[1]), (p1[0] - p2[0]))
        i += 1

    orients = np.rad2deg(np.arctan(gradients))

    return lines, gradients, orients

#%% Reorient and Write

def run_tesseract(write_path, trg_region, orients):

    mask = ~(trg_region.filled_image == trg_region.image)
    mask = rescale(mask, 0.5)

    img1 = np.multiply(rotate(mask, orients[1]).astype(np.uint8), 255)
    img2 = np.multiply(rotate(mask, orients[1] + 90).astype(np.uint8), 255)
    img3 = np.multiply(rotate(mask, orients[1] + 180).astype(np.uint8), 255)
    img4 = np.multiply(rotate(mask, orients[1] + 270).astype(np.uint8), 255)

    imwrite('{0}img_{1}.png'.format(write_path, 1), img1)
    cmd_input = 'tesseract {1}img_{0}.png {1}out_{0} -psm 10'.format(1, write_path)
    subpro.run(cmd_input, shell = True, check = True)

    imwrite('{0}img_{1}.png'.format(write_path, 2), img2)
    cmd_input = 'tesseract {1}img_{0}.png {1}out_{0} -psm 10'.format(2, write_path)
    subpro.run(cmd_input, shell = True, check = True)

    imwrite('{0}img_{1}.png'.format(write_path, 3), img3)
    cmd_input = 'tesseract {1}img_{0}.png {1}out_{0} -psm 10'.format(3, write_path)
    subpro.run(cmd_input, shell = True, check = True)

    imwrite('{0}img_{1}.png'.format(write_path, 4), img4)
    cmd_input = 'tesseract {1}img_{0}.png {1}out_{0} -psm 10'.format(4, write_path)
    subpro.run(cmd_input, shell = True, check = True)

    return mask, img1, img2, img3, img4

#%% Read Files

# Plot Option
opt_plot = True

# Start Time
time_0 = time.time()

img_hsv = rgb2hsv(rescale(imread('Z_near.jpg'), 0.5))

#img_hsv = cv2.imread('A_shade3.jpg', cv2.IMREAD_COLOR)
#img_hsv = cv2.resize(img_hsv, (0,0), fx=0.5, fy=0.5)
#img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)

#%% Colour Segmentation

time_1 = time.time()

mask_rgb, mask = run_colour_segmentation(img_hsv)

if opt_plot:
    fig1, ax = plt.subplots(1,2)

    ax = ax.ravel()

    ax[0].imshow(hsv2rgb(img_hsv))
    ax[0].set_title('Original Image')

    ax[1].imshow(mask_rgb)
    ax[1].set_title('Masked RGB')

#%% Filter Noise

time_2 = time.time()

trg_region, labelled_mask, region_mask = run_morph(mask)

if opt_plot:
    fig2, ax = plt.subplots(1,3)

    ax = ax.ravel()

    ax[0].imshow(mask, cmap = cm.gray)
    ax[0].set_title('Mask')

    ax[1].imshow(trg_region.image, cmap = cm.gray)
    ax[1].set_title('Extracted Target')

    ax[2].imshow(trg_region.filled_image, cmap = cm.gray)
    ax[2].set_title('Filled Target')

#%% Hough Transform

time_3 = time.time()

lines_pos, grad_pos, orients_pos = run_hough_transform(trg_region.filled_image, (-2, 92, 45))
lines_neg, grad_neg, orients_neg = run_hough_transform(trg_region.filled_image, (-92, 2, 45))

if opt_plot:
    fig3, ax = plt.subplots(1,2)

    ax = ax.ravel()

    ax[0].imshow(trg_region.filled_image, cmap = cm.gray)
    ax[0].set_title('Detected Lines')

    for p1, p2 in lines_neg:
        ax[0].plot((p1[0], p2[0]), (p1[1], p2[1]), '-r')

    for p1, p2 in lines_pos:
        ax[0].plot((p1[0], p2[0]), (p1[1], p2[1]), '-b')


    ax[1].imshow(trg_region.image, cmap = cm.gray)
    ax[1].set_title('Detected Lines')

    for p1, p2 in lines_neg:
        ax[1].plot((p1[0], p2[0]), (p1[1], p2[1]), '-r')

    for p1, p2 in lines_pos:
        ax[1].plot((p1[0], p2[0]), (p1[1], p2[1]), '-b')


# #%% Rotate and Write

time_4 = time.time()

# mask, img1, img2, img3, img4 = run_tesseract('output/', trg_region, orients_neg)

# with open('output/out_1.txt', 'r') as f:
    # read_data = f.read()
    # print('Image 1: {0}'.format(read_data))

# f.close()

# with open('output/out_2.txt', 'r') as f:
    # read_data = f.read()
    # print('Image 2: {0}'.format(read_data))

# f.close()

# with open('output/out_3.txt', 'r') as f:
    # read_data = f.read()
    # print('Image 3: {0}'.format(read_data))

# f.close()

# with open('output/out_4.txt', 'r') as f:
    # read_data = f.read()
    # print('Image 4: {0}'.format(read_data))

# f.close()

# End Time
time_5 = time.time()

#%% Print

print('Total Time Interval: {} s'.format(time_5 - time_0))
print('Rescale Interval: {} s'.format(time_1 - time_0))
print('Threshold Interval: {} s'.format(time_2 - time_1))
print('Hough Transform: {} s'.format(time_4 - time_2))
print('OCR: {} s'.format(time_5 - time_4))