"""
utils.py
--------
Helper functions for the Self Driving Car (Udacity Simulator) project.
Covers: reading driving_log.csv, balancing steering data, loading data,
image augmentation (Pan, Zoom, Brightness, Flip), preprocessing (NVIDIA style),
and a Keras batch generator for training.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa
import random
from sklearn.utils import shuffle


# -----------------------------------------------------------------
# STEP 1: Extract only the filename (not full system path) from the
# path saved by the simulator (simulator saves full local path).
# -----------------------------------------------------------------
def getName(filePath):
    return filePath.split('\\')[-1].split('/')[-1]


# -----------------------------------------------------------------
# STEP 2: Import driving_log.csv, keep only Center image + Steering,
# and print how many images we found.
# -----------------------------------------------------------------
def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)

    # Keep only the file name, not the full path (simulator stores full local path)
    data['Center'] = data['Center'].apply(getName)

    print('Total Images Imported:', data.shape[0])
    return data


# -----------------------------------------------------------------
# STEP 3: Balance the data — Udacity data has WAY too many steering=0
# samples (driving straight). This biases the model to always predict
# ~0 steering. We bin the steering angles into `nBin` bins and cut
# down any bin that exceeds `samplesPerBin`.
# -----------------------------------------------------------------
def balanceData(data, display=True):
    nBin = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data['Steering'], nBin)

    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])),
                  (samplesPerBin, samplesPerBin))
        plt.title('Steering Angle Distribution (Before Balancing)')
        plt.show()

    removeIndexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if bins[j] <= data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)

    print('Removed Images:', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print('Remaining Images:', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBin)
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])),
                  (samplesPerBin, samplesPerBin))
        plt.title('Steering Angle Distribution (After Balancing)')
        plt.show()

    return data


# -----------------------------------------------------------------
# STEP 4: Convert dataframe into two numpy arrays -> imagesPath, steerings
# -----------------------------------------------------------------
def loadData(path, data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        imagesPath.append(os.path.join(path, 'IMG', indexed_data[0]))
        steering.append(float(indexed_data[3]))

    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering


# -----------------------------------------------------------------
# STEP 5: Image Augmentation -> Pan, Zoom, Brightness, Flip
# (exactly the 4 shown in your diagram). Applied randomly, only
# during training (never on validation data).
# -----------------------------------------------------------------
def augmentImage(imgPath, steering):
    img = cv2.imread(imgPath) if isinstance(imgPath, str) else imgPath

    # PAN (random translation)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    # ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)

    # FLIP (must also flip the steering angle sign!)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


# -----------------------------------------------------------------
# STEP 6: Preprocessing (NVIDIA paper style)
# - Crop out sky (top) and car hood (bottom)
# - Convert to YUV color space (NVIDIA uses YUV, not RGB)
# - Gaussian blur to reduce noise
# - Resize to 200x66 (NVIDIA input size)
# - Normalize pixel values to 0-1
# -----------------------------------------------------------------
def preProcess(img):
    img = img[60:135, :, :]                       # crop sky & hood
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)     # NVIDIA uses YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))               # NVIDIA input size
    img = img / 255.0                              # normalize
    return img


# -----------------------------------------------------------------
# STEP 7: Batch Generator -> generates batches on-the-fly, applying
# augmentation ONLY for training batches, not for validation batches.
# This saves RAM (no need to load/augment entire dataset at once).
# -----------------------------------------------------------------
def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)

            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = cv2.imread(imagesPath[index])
                steering = steeringList[index]

            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield (np.asarray(imgBatch), np.asarray(steeringBatch))