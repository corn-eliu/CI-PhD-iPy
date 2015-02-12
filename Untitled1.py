# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

from PIL import Image

import numpy as np

import scipy as sp
import scipy.io as sio
import cv2
import cv
import glob
import time
import gc
import re

import sys
import os

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
sampleData = "ribbon2/"
# sampleData = "ribbon1_matted/"
# sampleData = "little_palm1_cropped/"
# sampleData = "ballAnimation/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame-*.png")
mattes = glob.glob(dataFolder + sampleData + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes)

lowThresh = 96
highThresh = lowThresh*2

# <codecell>

img = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
matte = cv2.cvtColor(cv2.imread(mattes[0]), cv2.COLOR_BGR2GRAY)

imgEdges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), lowThresh, highThresh)
matteEdges = cv2.Canny(matte, lowThresh, highThresh)
matteEdges = cv2.dilate(matteEdges, np.ones((6,6),dtype=np.uint8))
matteEdges = cv2.erode(matteEdges, np.ones((6,6),dtype=np.uint8))

# figure(); imshow(imgEdges*(matte/255.0), interpolation='nearest')
figure(); imshow(matteEdges, interpolation='nearest')

# <codecell>

## find points on matte edges
edgePoints = np.argwhere(matteEdges == np.max(matteEdges))
## closes point to top-left (i.e.) [0, 0]
startPoint = edgePoints[np.argmin(np.sum(edgePoints, axis=1)), :]
print startPoint
scatter(startPoint[1], startPoint[0])

# <codecell>

edgePoints

