# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab 
import numpy as np
import scipy as sp
import cv2
import glob
import time
import sys
import os
from scipy import ndimage
from scipy import stats

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

dataFolder = "data/"

# <codecell>

distanceMatrix = np.load("ribbon1DistMat.npy")

# <codecell>

figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

## read frames from sequence of images
sampleData = "ribbon1_matte/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame*.png")
frames = np.sort(frames)
numFrames = len(frames)
print numFrames

# <codecell>

# numFrames = 3
distanceMatrix = np.zeros([numFrames, numFrames])
for i in range(0, numFrames) :
    p = np.ndarray.flatten(np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB))/255.0)
    t = time.time()
#     print p.shape
    for j in range(i+1, numFrames) :
        ## len(q-p) = sqrt(len(q)^2+len(p)^2 - 2*dot(p, q)) where p and q are two images in vector format and 1xN size
        q = np.ndarray.flatten(np.array(cv2.cvtColor(cv2.imread(frames[j]), cv2.COLOR_BGR2RGB))/255.0)
        distanceMatrix[j, i] = distanceMatrix[i, j] = np.sqrt(np.linalg.norm(q)**2+np.linalg.norm(p)**2 - 2*np.dot(p, q))
#         distanceMatrix[j, i] = distanceMatrix[i, j] = np.linalg.norm(q-p)
#         print distanceMatrix[j, i],
    sys.stdout.write('\r' + "Frame " + np.string_(i) + " of " + np.string_(numFrames) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()

# <codecell>

print distanceMatrix

# <codecell>

print distanceMatrix

# <codecell>

print distanceMatrix

