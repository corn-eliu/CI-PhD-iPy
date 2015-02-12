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
import Image

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import MazeSequenceUtils as msu
import ActiveShapesUtils as asu

dataLoc = "/home/ilisescu/PhD/iPy/data/flower/"
nameLength = len(filter(None, re.split('/',dataLoc)))

colVals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

# <codecell>

## load image
frameName = "/home/ilisescu/PhD/iPy/data/frame-00001-mask.png"
frame = cv2.imread(frameName)

## make bbox of snake
bbox = np.array([[5, 25], [45, 5], [90, 70], [45, 95]])
bbox = np.array([[667, 711], [443, 318], [540, 262], [765, 655]])
## build the snake initialization by supersampling the bbox
numOfPoints = 81
interval = (np.linalg.norm(bbox[0, :]-bbox[1, :])+np.linalg.norm(bbox[1, :]-bbox[2, :]))/(numOfPoints*0.5)
initialSnake = []
for i, j in zip(arange(0, len(bbox)), np.hstack((arange(1, len(bbox)), 0))):
    currentPoint = bbox[i, :]
#     snake.append(currentPoint)
    currentDir = (bbox[j, :]-bbox[i, :])/np.linalg.norm(bbox[j, :]-bbox[i, :])
    while np.linalg.norm(currentPoint-bbox[j, :]) > interval :
        initialSnake.append(currentPoint)
        currentPoint = currentPoint + interval*currentDir
    
    ##divide last remaining bit into half
    initialSnake.append((initialSnake[-1]+bbox[j, :])/2)

initialSnake = np.round(np.array(initialSnake))
initialSnake = np.vstack((initialSnake, initialSnake[0, :]))
print initialSnake.shape


## visualize snake
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.imshow(frame, interpolation='nearest')
ax.autoscale(False)
ax.plot(initialSnake[:, 0], initialSnake[:, 1], c='y', marker="o")
draw()

# <codecell>

## compute canny edge and distance to edge for frame
edges = cv2.Canny(frame, 100, 200)
edges = edges/np.max(edges)
distanceMatrix = ndimage.distance_transform_edt(1-edges)

## get user points
userPoints = np.load("userPoints.npy")
discreteUserPoints = userPoints[0, :]
for i in xrange(1, len(userPoints)) :
    if np.linalg.norm(userPoints[i-1, :]-userPoints[i, :]) >= 1 :
        discreteUserPoints = np.vstack((discreteUserPoints, asu.discretizeLine(userPoints[i-1:i+1, :])))
discreteUserPoints = np.array(discreteUserPoints, dtype=np.int)
userPointsMap = np.ones(frame.shape[0:2])
userPointsMap[discreteUserPoints[:, 1], discreteUserPoints[:, 0]] = 0
userPointsDistance = ndimage.distance_transform_edt(userPointsMap)
# distanceMatrix = (userPointsDistance/np.max(userPointsDistance))*distanceMatrix

figure(); imshow(edges, interpolation='nearest')
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

## show unary costs in an image together with the edges
img = np.array(edges, dtype=float)

for n in xrange(0, len(snakeIdxs)) :
    img[snakeIdxs[n, 0]-2:snakeIdxs[n, 0]+3, snakeIdxs[n, 1]-2:snakeIdxs[n, 1]+3] = np.reshape(unaryCosts[:, n], [neighbourhoodSize, neighbourhoodSize])/np.max(unaryCosts[:, n])

print img.shape

## visualize snake
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.imshow(img, interpolation='nearest')
ax.autoscale(False)
ax.plot(snakeIdxs[:, 1], snakeIdxs[:, 0], c='y', marker="o", )
draw()


# <codecell>

meanSpacing = 0.0
for p in xrange(0, len(initialSnake)) :
    pNext = p+1 if p < len(initialSnake)-1 else 0
    meanSpacing += np.linalg.norm(initialSnake[p, :]-initialSnake[pNext, :])

meanSpacing = meanSpacing/len(initialSnake)
meanSpacing = 22.3076923077
print meanSpacing
neighbourhoodSize = 5

activeSnake = np.copy(initialSnake)
## need to swap columns in snake to get indices in distanceMatrix as snake contains (x, y) coords and I need (r, c)
## x == c, y==r

snakeIdxs = np.array([activeSnake[:, 1], activeSnake[:, 0]]).T

## remove bad points and retry
# clampedSnake = np.vstack((snakeIdxs[0:24, :], snakeIdxs[26, :], snakeIdxs[35:70, :], snakeIdxs[74, :], snakeIdxs[78, :], snakeIdxs[84:len(snakeIdxs), :]))
# snakeIdxs = np.copy(clampedSnake)


fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.imshow(frame, interpolation='nearest')
ax.autoscale(False)
s, = ax.plot(snakeIdxs[:, 1], snakeIdxs[:, 0], c='y', marker="o")
draw()

asu.optimizeSnake(snakeIdxs, distanceMatrix, neighbourhoodSize, 0.9, s, 18, False)

# <codecell>

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.imshow(frame, interpolation='nearest')
ax.autoscale(False)
for i in xrange(0, 2):#len(testkIdxNexts)) :
    ax.plot(np.hstack((testkIdxPrevs[i, 1], testkIdx[1], testkIdxNexts[i, 1])), np.hstack((testkIdxPrevs[i, 0], testkIdx[0], testkIdxNexts[i, 0])), c='y', marker="o")
    print -((testkIdxPrevs[i, :]-2*testkIdx+testkIdxNexts[i, :])*(testkIdxPrevs[i, :]-2*testkIdx+testkIdxNexts[i, :])).sum()
draw()

# <codecell>

testA = testkIdxPrevs[1, :]
testO = testkIdx
testB = testkIdxNexts[1, :]
print testA, testO, testB

testArea = (testA[1]*(testO[0]-testB[0])+testO[1]*(testB[0]-testA[0])+testB[1]*(testA[0]-testO[0]))/2.0
testCurve = (4*testArea)/(np.linalg.norm(testA-testO)*np.linalg.norm(testO-testB)*np.linalg.norm(testB-testA))
print testCurve

# <codecell>

## try out the curvature thing
print snakeIdxs[0:3, :]
testkIdx = np.array([15., 44.])#snakeIdxs[1, :]
testkIdxPrevs = np.zeros((neighbourhoodSize**2, snakeIdxs.shape[1]))
testkIdxNexts = np.zeros((neighbourhoodSize**2, snakeIdxs.shape[1]))
for i in xrange(0, neighbourhoodSize**2) :
    testkIdxPrevs[i, :] = deltaCoords(linearTo2DCoord(i, neighbourhoodSize), neighbourhoodSize)+snakeIdxs[0, :]
    testkIdxNexts[i, :] = deltaCoords(linearTo2DCoord(12, neighbourhoodSize), neighbourhoodSize)+snakeIdxs[2, :]
    
print testkIdx
print testkIdxPrevs
print testkIdxNexts

testcurveTerms = -((testkIdxPrevs-2*testkIdx+testkIdxNexts)*(testkIdxPrevs-2*testkIdx+testkIdxNexts)).sum(axis=1)
print testcurveTerms

# <codecell>

activeSnake = np.zeros_like(snake)
for i in xrange(0, len(snake)) :
    activeSnake[i, :] = deltaCoords(linearTo2DCoord(minCostTraversal[i], neighbourhoodSize), neighbourhoodSize)+snakeIndices[i, :]
    

activeSnake = np.array([activeSnake[:, 1], activeSnake[:, 0]]).T

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.imshow(frame, interpolation='nearest')
ax.autoscale(False)
ax.plot(activeSnake[:, 0], activeSnake[:, 1], c='y', marker="o")
draw()

