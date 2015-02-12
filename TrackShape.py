# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import scipy as sp
from scipy import io
from scipy import ndimage
import re
import cv2
import sys
import glob

import ActiveShapesUtils as asu

dataLoc = "/home/ilisescu/PhD/iPy/data/flower/"
nameLength = len(filter(None, re.split('/',dataLoc)))

colVals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

# <codecell>

## load frames
frameNames = np.sort(glob.glob(dataLoc + "*.png"))
frameSize = cv2.imread(frameNames[0]).shape
movie = np.zeros(np.hstack([frameSize, len(frameNames)]), dtype=uint8)
print movie.shape
for i in xrange(0, len(frameNames)):
#     im = np.array(cv2.imread(location+frames[i]))/255.0
#     movie[:, :, i] = np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
#     movie[:, :, :, i] = np.array(cv2.imread(frameNames[i]))/255.0
    movie[:, :, :, i] = cv2.cvtColor(cv2.imread(frameNames[i]), cv2.COLOR_BGR2RGB)
    sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(frameNames)))
    sys.stdout.flush()

# <codecell>

## TEMP::HACK:: load snake for first frame
startingSnake = np.load("snakeF1Flower.npy")

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.imshow(movie[:, :, :, 0], interpolation='nearest')
ax.autoscale(False)
ax.plot(startingSnake[:, 1], startingSnake[:, 0], c='y', marker="o")
draw()

# <codecell>

figure(); imshow(distanceMatrix, cmap=cm.RdYlBu)
figure(); imshow(edges)
figure(); imshow(np.power(distanceMatrix, 2), cmap=cm.RdYlBu)

# <codecell>

currentSnake = np.copy(startingSnake)
## processType: 0 = snakes, 1 = shape templates, 2 = NCC, 3 = ??
processParams = dict(processType = 2, useFlow = True, useFarneback = False)
for i in xrange(1, 2) :
    if processParams['processType'] == 0 :
        print "Processing using active shape"
        
        edges = cv2.Canny(movie[:, :, :, i], 100, 200)/255.0
        distanceMatrix = ndimage.distance_transform_edt(1-edges)
        fig = plt.figure(figsize=(10, 10))
        
        if processParams['useFlow'] :
            if processParams['useFarneback'] :
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(movie[:, :, :, i], cv2.COLOR_RGB2GRAY), cv2.cvtColor(movie[:, :, :, i-1], cv2.COLOR_RGB2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
                currentSnake = flow[np.array(currentSnake[:, 0], dtype=np.int),np.array(currentSnake[:, 1], dtype=np.int), :]+currentSnake
            else :
                lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                newSnake, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(movie[:, :, :, i-1], cv2.COLOR_RGB2GRAY), cv2.cvtColor(movie[:, :, :, i], cv2.COLOR_RGB2GRAY), np.reshape(np.array([currentSnake[:, 1], currentSnake[:, 0]], dtype=np.float32).T, [len(currentSnake), 1, 2]), None, **lk_params)
                currentSnake = np.array([newSnake[:, 0, 1], newSnake[:, 0, 0]]).T
        
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.imshow(movie[:, :, :, i], interpolation='nearest')
        ax.autoscale(False) 
        ax.plot(currentSnake[:, 1], currentSnake[:, 0], c='y', marker="o")
        s, = ax.plot(currentSnake[:, 1], currentSnake[:, 0], c='r', marker="o")
        draw()
        currentSnake = asu.optimizeSnake(currentSnake, distanceMatrix, 5, 0.9, s, 10, False)
        
    elif processParams['processType'] == 1 :
        print "Processing using shape templates"
        
    elif processParams['processType'] == 2 :
        print "Processing using NCC"
        
        ## discretize snake
        discreteSnake = currentSnake[0, :]
        for p in xrange(1, len(currentSnake)) :
            if np.linalg.norm(currentSnake[p-1, :]-currentSnake[p, :]) >= 1 :
                discreteSnake = np.vstack((discreteSnake, asu.discretizeLine(currentSnake[p-1:p+1, :])))
        discreteSnake = np.array(discreteSnake, dtype=np.int)
        snakeMask = np.zeros(movie[:, :, :, i].shape[0:2])
        snakeMask[discreteSnake[:, 0], discreteSnake[:, 1]] = 1
        snakeMask = snakeMask[np.min(discreteSnake[:, 0]):np.max(discreteSnake[:, 0]+1), np.min(discreteSnake[:, 1]):np.max(discreteSnake[:, 1]+1)]
        snakeMask = ndimage.morphology.binary_fill_holes(snakeMask)
        
        template = movie[np.min(discreteSnake[:, 0]):np.max(discreteSnake[:, 0]+1), np.min(discreteSnake[:, 1]):np.max(discreteSnake[:, 1]+1), :, i-1]
        template = template*np.repeat(np.reshape(snakeMask, np.hstack((snakeMask.shape, 1))), movie.shape[2], axis=2)
        print template.shape
        figure(); imshow(template, interpolation='nearest')
        
        result = cv2.matchTemplate(movie[:, :, :, i-1], template, cv2.TM_CCORR)
        print result.shape
        figure(); imshow(result, interpolation='nearest')
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        print minLoc, maxLoc
        
        result2 = np.zeros((np.array(movie[:, :, :, i-1].shape[0:2])-np.array(template.shape[0:2])))
        for r in xrange(0, result2.shape[0]) :
            print r, 
            for c in xrange(0, result2.shape[1]) :
                result2[r, c] = np.sum(movie[r:r+template.shape[0], c:c+template.shape[1], :, i-1]*template)
        
        figure(); imshow(result2, interpolation='nearest')
        
        fig = plt.figure(figsize=(10, 10))
        
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.imshow(movie[:, :, :, i-1], interpolation='nearest')
        ax.autoscale(False)
        ax.scatter(minLoc[0], minLoc[1], c='b', marker="o")
        ax.scatter(maxLoc[0], maxLoc[1], c='r', marker="o")
        draw()
        
    elif processParams['processType'] == 3 :
        print "Processing using dunno"
        
    else :
        raise BaseException, "undefined processType"

# <codecell>

print np.arange(0, result2.shape[1])
testTmp = np.repeat(np.reshape(movie[:, :, :, 0], np.hstack((movie.shape[0:3], 1))), result2.shape[1], axis=3)

# <codecell>

idxRows = np.repeat(np.reshape(arange(0, 55), (55, 1)), 55, axis=1)
idxRows += 400
idxCols = np.repeat(np.reshape(arange(0, 55), (1, 55)), 55, axis=0)
idxCols += 600
idxCols = np.repeat(np.reshape(idxCols, np.hstack((idxCols.shape, 1))), 2, axis=2).shape

figure(); imshow(testTmp[idxRows, idxCols, :, :][:, :, :, 280])

# <codecell>

arange(400, 455).shape

# <codecell>

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(movie[:, :, :, i-1], cv2.COLOR_RGB2GRAY), mask = None, **feature_params)
print p0.shape
print "la"
print np.reshape(np.array(currentSnake, dtype=np.float32), [len(currentSnake), 1, 2]).shape

# <codecell>

snakeIdxs = np.array([activeSnake[:, 1], activeSnake[:, 0]]).T

# <codecell>

testSimpleFlow = np.load("testSimpleFlow.npy")
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.imshow(testSimpleFlow, interpolation='nearest')
ax.autoscale(False)
s, = ax.plot(currentSnake[:, 1], currentSnake[:, 0], c='y', marker="o")
draw()

