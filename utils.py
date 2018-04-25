# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 05:24:59 2018

@author: mgriffi3
"""

"""
REFERENCES
    https://github.com/naokishibuya/car-behavioral-cloning
"""

import cv2, os
import numpy as np
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def LoadImage(dataDir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(dataDir, image_file.strip()))

def CropImage(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front

def ResizeImage(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def Rgb2Yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def Preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = CropImage(image)
    image = ResizeImage(image)
    image = Rgb2Yuv(image)
    return image

def ChooseImage(dataDir, center, left, right, steeringAngle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return LoadImage(dataDir, left), steeringAngle + 0.2
    elif choice == 1:
        return LoadImage(dataDir, right), steeringAngle - 0.2
    return LoadImage(dataDir, center), steeringAngle

def FlipImage(image, steeringAngle):
    """
    Randomly flip the image left <-> right, and flip the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steeringAngle = -steeringAngle
    return image, steeringAngle

def TranslateImage(image, steeringAngle, range_x, range_y):
    """
    Randomly shift the image virtically and horizontally.
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steeringAngle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steeringAngle

def AddShadow(image):
    """
    TODO: make this work
    """
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # set 1 below the line and zero otherwise
    # image coordinate system is upside down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def RandomBrightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def Augument(dataDir, center, left, right, steeringAngle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steeringAngle = ChooseImage(dataDir, center, left, right, steeringAngle)
    image, steeringAngle = FlipImage(image, steeringAngle)
    image, steeringAngle = TranslateImage(image, steeringAngle, range_x, range_y)
    #image = AddShadow(image)
    image = RandomBrightness(image)
    return image, steeringAngle

def BatchGenerator(dataDir, imagePaths, steeringAngles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(imagePaths.shape[0]):
            center, left, right = imagePaths[index]
            steeringAngle = steeringAngles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steeringAngle = Augument(dataDir, center, left, right, steeringAngle)
            else:
                image = LoadImage(dataDir, center) 
            # add the image and steering angle to the batch
            images[i] = Preprocess(image)
            steers[i] = steeringAngle
            i += 1
            if i == batch_size:
                break
        yield images, steers
