# Integrated Script - Video Feed

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
from skimage.transform import rescale, rotate

# Morphology Operations
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries

# Hough Transform
from skimage.transform import probabilistic_hough_line as hough


#%% Colour Segmentation

def run_colour_segmentation(img_rgb):

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

#    # Bounds on HSV Colour Space (SkiImage Format)
#    bound1 = np.array([[0, 0.1, 0.0], [0.09, 1.0, 1.0]])
#    bound2 = np.array([[0.95, 0.1, 0.0], [1.0, 1.0, 1.0]])

    # Bounds on HSV Colour Space (OpenCV Format)
    bound1 = np.array([[0, 25, 70], [20, 255, 255]])
    bound2 = np.array([[144, 25, 70], [180, 255, 255]])

    # Filter Image
    mask1 = cv2.inRange(img_hsv, bound1[0], bound1[1])
    mask2 = cv2.inRange(img_hsv, bound2[0], bound2[1])
    mask = mask1 + mask2

    mask_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask = mask)
    mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_HSV2BGR)

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

    row_c, col_c = region_mask[idx_trg].coords[0]

    trg_region = (labelled_mask == labelled_mask[row_c, col_c])

    return trg_region, labelled_mask, region_mask, idx_trg


#%% Hough Transform

def run_hough_transform(mask, theta_in):

    angle_bound = np.deg2rad(np.linspace(*theta_in))

    # Detect Lines
    mask = regionprops(label(mask))

    lines = hough(find_boundaries(mask[0].filled_image),
                  theta = angle_bound)

    # Compute Gradients
    gradients = np.zeros(len(lines))

    i = 0
    eps = 0.00001 # Prevent Divide by Zero Error

    for p1, p2 in lines:
        gradients[i] = np.divide((p1[1] - p2[1]), (p1[0] - p2[0]) + eps)
        i += 1

    orients = np.rad2deg(np.arctan(gradients))

    return lines, gradients, orients, mask[0].centroid


#%% Reorient and Write

def run_tesseract(write_path, trg_region, orients):

    # Detect Lines
    trg_region = regionprops(label(trg_region))[0]

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


#%% Video Capture

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

stream = cv2.VideoCapture(0)

frame_bool = True
count = 0
c_time = time.time()

while frame_bool:

    c_time = time.time()
    count += 1

    frame_bool, frame = stream.read()
    frame = np.array(frame)

    # Get Frame
    cv2.imshow('Video Feed', frame)

    # HSV Segmentation
    mask_rgb, mask = run_colour_segmentation(frame)
    cv2.imshow('Mask', mask)

    if np.any(mask == 255):
        # Morph (Extract Biggest Region)
        trg_region, labelled_mask, region_mask, idx_trg = run_morph(mask)
        trg_region = trg_region.astype(np.uint8)
        trg_img = np.stack([trg_region, trg_region, trg_region], axis = 2) * 255

        # Hough Transform for Orientation
        lines_pos, grad_pos, orients_pos, centroid_region = run_hough_transform(trg_region, (0, 90, 45))
        lines_neg, grad_neg, orients_neg, centroid_region = run_hough_transform(trg_region, (-90, 0, 45))

        cv2.imshow('Target Region', trg_img)

        # Run OCR
        mask, img1, img2, img3, img4 = run_tesseract('output/', trg_region, orients_neg)

        with open('output/out_1.txt', 'r') as f:
            read_data = f.read()

        f.close()

        cv2.putText(img1,'Letter:{0}'.format(read_data),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('Image 1', img1)


        with open('output/out_2.txt', 'r') as f:
            read_data = f.read()

        f.close()

        cv2.putText(img2,'Letter:{0}'.format(read_data),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('Image 2', img2)


        with open('output/out_3.txt', 'r') as f:
            read_data = f.read()

        f.close()

        cv2.putText(img3,'Letter:{0}'.format(read_data),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('Image 3', img3)


        with open('output/out_4.txt', 'r') as f:
            read_data = f.read()

        f.close()

        cv2.putText(img4,'Letter:{0}'.format(read_data),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('Image 4', img4)

        interval = time.time() - c_time
        print('Loop {} Interval : {} s'.format(count, interval))

    else:
        interval = time.time() - c_time
        print('No Target Detected')
        print('Loop {} Interval : {} s'.format(count, interval))


    # Kill Switch
    k = cv2.waitKey(1)

    if k == 27:
        break


cv2.destroyAllWindows()
stream.release()