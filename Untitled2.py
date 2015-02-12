# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
from PIL import Image

import sys
import numpy as np
import time
import cv2
import re
import glob
import os

import VideoTexturesUtils as vtu

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

## read frames from sequence of images
sampleData = "palm_tree1/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame-*.png")
mattes = glob.glob(dataFolder + sampleData + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes)

# <codecell>

figure(); imshow(np.load(dataFolder + "palm_tree1/vanilla_distMat.npy"), interpolation='nearest')

# <codecell>

matte = Image.open(mattes[176])
matte = np.reshape(np.array(matte, dtype=np.uint8), (matte.size[1], matte.size[0], 1))

fullResoFrame = Image.open(frames[176])
fullResoFrame = np.array(fullResoFrame, dtype=np.uint8)[:, :, 0:3]

Image.frombytes("RGBA", (fullResoFrame.shape[1], fullResoFrame.shape[0]), np.concatenate((fullResoFrame, matte), axis=-1).tostring()).save(dataFolder + "frame_177.png")

