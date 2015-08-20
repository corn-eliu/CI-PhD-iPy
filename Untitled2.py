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
sampleData = "havana/"
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

result = np.zeros(frameSize)

# <codecell>

movie = np.zeros((frameSize[0], frameSize[1], numFrames), dtype=uint8)

# <codecell>

channel = 2
start = time.time()
for f in xrange(numFrames) :
    if np.mod(f, 50) == 0 :
        sys.stdout.write('\r' + "Loading frame " + np.string_(f) + " of " + np.string_(numFrames) + " at " + np.string_(int(f/(time.time()-start))) + "FPS")
        sys.stdout.flush()
    movie[:, :, f] = np.array(Image.open(frames[f]))[:, :, channel]
print
print "done"
result[:, :, channel] = np.copy(np.median(movie, axis=-1))
figure(); imshow(np.array(result, dtype=uint8))

# <codecell>

start = time.time()
for f in xrange(numFrames) :
#     if np.mod(f, 50) == 0 :
    sys.stdout.write('\r' + "Loading frame " + np.string_(f) + " of " + np.string_(numFrames) + " at " + np.string_(int(f/(time.time()-start))) + "FPS")
    sys.stdout.flush()
#     img = np.array(Image.open(frames[f]), dtype=np.uint8)
#     for c in xrange(256) :
#         histograms[c, :, :, :] = histograms[c, :, :, :] + np.array(img[:, :, :] == c, dtype=uint16)
    img = np.array(Image.open(frames[f]), dtype=np.uint8).reshape(np.prod(frameSize))
    histograms[img, arange(np.prod(frameSize))] += 1
print
print "done"
# result[:, :, channel] = np.copy(np.median(movie, axis=-1))
# figure(); imshow(np.array(result, dtype=uint8))

# <codecell>

histograms = np.zeros((256, np.prod(frameSize)), dtype=np.uint16)

# <codecell>

histMax = np.array(np.argmax(histograms, axis=0), dtype=np.uint8)

# <codecell>

Image.fromarray((histMax.reshape(frameSize)).astype(numpy.uint8)).save(outputData + "histMax.png")

# <codecell>

figure(); imshow(histMax.reshape(frameSize))

# <codecell>

img = np.array(Image.open(frames[0]), dtype=np.uint8).reshape(np.prod(frameSize))

# <codecell>

print histMax.reshape(frameSize)[279, 792, :]

# <codecell>

ch = 2
row = 279
col = 792
idx = ch+col*frameSize[-1]+row*frameSize[-1]*frameSize[-2]
print histMax[idx]

figure()
bar(arange(256, dtype=int), np.array(histograms[:, idx], dtype=float)/np.max(histograms[:, 0]), width=1.0)
ax = gca()
ax.set_xlim(left=0, right=256)
ax.set_ylim(bottom=0, top=1.2)

# <codecell>

Image.fromarray((result).astype(numpy.uint8)).save(outputData + "median.png")

# <codecell>

med = np.median(movie, axis=-1)

# <codecell>

figure(); imshow(med)
figure(); imshow(movie[:, :, 0])

# <codecell>

figure(); imshow(np.load(dataFolder + "palm_tree1/vanilla_distMat.npy"), interpolation='nearest')

# <codecell>

matte = Image.open(mattes[176])
matte = np.reshape(np.array(matte, dtype=np.uint8), (matte.size[1], matte.size[0], 1))

fullResoFrame = Image.open(frames[176])
fullResoFrame = np.array(fullResoFrame, dtype=np.uint8)[:, :, 0:3]

Image.frombytes("RGBA", (fullResoFrame.shape[1], fullResoFrame.shape[0]), np.concatenate((fullResoFrame, matte), axis=-1).tostring()).save(dataFolder + "frame_177.png")

