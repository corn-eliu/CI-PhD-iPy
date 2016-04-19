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

dataLoc = "/home/ilisescu/PhD/iPy/data/flower/"
nameLength = len(filter(None, re.split('/',dataLoc)))

colVals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

# <codecell>

def insideRect(point, rectSize) :
    width = rectSize[1]
    height = rectSize[0]
    result = point[0] >= 0 and point[1] >= 0 and point[0] <= width and point[1] <= height
#     print "inside rect", result
    return result
    
    
def findSplitSegment(line, rectSize) :
    points = []
    ## line-line intersection using cramer's rule
    a = line[2:4]
    u = line[0:2]
    width = rectSize[1]
    height = rectSize[0]
#     print rectSize
    
    
    # check intersection with x axis
    b = np.array([0, 0])
    v = np.array([1, 0])
    
    A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
    c = np.array([[b[0]-a[0]], [b[1]-a[1]]])
    x = np.linalg.det(np.array([c, A[:, 1]]))/np.linalg.det(A)
    point = np.array(np.round(a+x*u), dtype=int)
#     print point
    
    if insideRect(point, rectSize) :
        points.append(point)
        
        
    ## check intersection with height axis
    b = np.array([0, height])
    A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
    c = np.array([[b[0]-a[0]], [b[1]-a[1]]])
    x = np.linalg.det(np.array([c, A[:, 1]]))/np.linalg.det(A)
    point = np.array(np.round(a+x*u), dtype=int)
#     print point
    
    if insideRect(point, rectSize) :
        isNew = True
        for p in points :
            if (point == p).all() :
                isNew = False
                break
        if isNew :
            points.append(point)
        
        
    # check instersection with y axis
    b = np.array([0, 0])
    v = np.array([0, 1])
    
    A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
    c = np.array([[b[0]-a[0]], [b[1]-a[1]]])
    x = np.linalg.det(np.array([c, A[:, 1]]))/np.linalg.det(A)
    point = np.array(np.round(a+x*u), dtype=int)
#     print point
    
    if insideRect(point, rectSize) :
        isNew = True
        for p in points :
            if (point == p).all() :
                isNew = False
                break
        if isNew :
            points.append(point)
        
    
    ## check intersection with width axis
    b = np.array([width, 0])
    A = np.array([[u[0], -v[0]], [u[1], -v[1]]])
    c = np.array([[b[0]-a[0]], [b[1]-a[1]]])
    x = np.linalg.det(np.array([c, A[:, 1]]))/np.linalg.det(A)
    point = np.array(np.round(a+x*u), dtype=int)
#     print point
    
    if insideRect(point, rectSize) :
        isNew = True
        for p in points :
            if (point == p).all() :
                isNew = False
                break
        if isNew :
            points.append(point)
    
    if len(points) != 2 :
        print "Incorrect number of points in split segment:", len(points) 
        raise BaseException
    return np.array(points)
    
    

# <codecell>

def discretizeLine(segmentExtremes) :
    ## discretize line segment
    lN = np.int(np.linalg.norm(segmentExtremes[1, :]-segmentExtremes[0, :]))
    lDir = (segmentExtremes[1, :]-segmentExtremes[0, :])/np.float(lN)
    discretizedSegment = np.array(np.round(np.repeat(np.reshape(np.array(segmentExtremes[1, :], dtype=int), [len(segmentExtremes[1, :]), 1]), lN+1, axis=1)-np.arange(0, lN+1)*np.reshape(lDir,[len(lDir), 1])), dtype=np.int).T
    return discretizedSegment

# <codecell>

def positiveHalfspaceMask(segmentExtremes, rectSize, inclusive) :
    hsMask = np.zeros(rectSize)
    A = segmentExtremes[0, :]
    B = segmentExtremes[1, :]
    Xs = np.repeat(np.reshape(np.arange(0, rectSize[1], 1), [1, rectSize[1]]), rectSize[0], axis=0)
    Ys = np.repeat(np.reshape(np.arange(0, rectSize[0], 1), [rectSize[0], 1]), rectSize[1], axis=1)
    distances = (B[0]-A[0])*(Ys-A[1])-(B[1]-A[1])*(Xs-A[0])
    
    if inclusive :
        hsMask[np.where(distances>=0)] = 1
    else :
        hsMask[np.where(distances>0)] = 1
    return hsMask

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
    movie[:, :, :, i] = cv2.imread(frameNames[i])
    sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(frameNames)))
    sys.stdout.flush()

# <codecell>

close('all')
## show first image and let user draw some points
fig = plt.figure()
ax = fig.add_subplot(111)
firstFrame = cv2.cvtColor(movie[:, :, :, 0], cv2.COLOR_BGR2RGB)
ax.imshow(firstFrame)
ax.set_autoscale_on(False)
# ax.set_xlim(firstFrame.shape[1])
# ax.set_ylim(firstFrame.shape[0])

userPoints = list()
line, = ax.plot(0, 0, color='b', lw=4.0)
buttonClicked = False

def onclick(event):
    global buttonClicked
    global userPoints
    if event.button == 1 :
        buttonClicked = True
        userPoints = list()
        userPoints.append(np.array([event.xdata, event.ydata]))

def onmove(event):
    global buttonClicked
    global userPoints
    global line
    if buttonClicked == True :
        userPoints.append(np.array([event.xdata, event.ydata]))
        line.set_ydata(np.array(userPoints)[:, 1])
        line.set_xdata(np.array(userPoints)[:, 0])
        draw()
        
def onrelease(event):
    global buttonClicked
    global userPoints
    global line
    if buttonClicked == True and event.button == 1 :
        buttonClicked = False
        userPoints.append(np.array([event.xdata, event.ydata]))
        line.set_ydata(np.array(userPoints)[:, 1])
        line.set_xdata(np.array(userPoints)[:, 0])
        draw()
    
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('motion_notify_event', onmove)
cid = fig.canvas.mpl_connect('button_release_event', onrelease)

# <codecell>

## find bestFitLine that best fits user given points and that will be the skeleton --> one line for now
userPoints = np.round(np.array(userPoints))
line = msu.fitLine2D(userPoints)
## project first and last userPoint onto line
u = line[0:2]
a = line[2:4]

## first row is first point, second row is last point (project first and last user points onto fitted line)
bestFitLine = ((np.dot(userPoints[[0, -1], :], u)/np.dot(u, u))*np.reshape(u, [len(u), 1])).T

## normalized vector from projected point to user point is line normal
lineNormal = (bestFitLine[0, :]-userPoints[0, :])/np.linalg.norm(bestFitLine[0, :]-userPoints[0, :])
## the fitted line is moved along normal by a certain scalar (i.e. distanceToLine)
distanceToLine = np.linalg.norm(a - bestFitLine[0, :] - np.dot(np.dot(a-bestFitLine[0, :], u), u))
## correct for that certain scalar found above
bestFitLine =  bestFitLine - distanceToLine*lineNormal

## show
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(firstFrame)
ax.set_autoscale_on(False)
ax.plot(userPoints[:, 0], userPoints[:, 1], color="b") ## plot user points
ax.plot(bestFitLine[:, 0],bestFitLine[:, 1], color="g") ## plot fitted line segment


# print userPoints

# <codecell>

## compute canny edge for first frame
frame = cv2.cvtColor(movie[:, :, :, 0],cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(frame, 100, 200)

figure(); imshow(edges, interpolation='nearest')

# dilEdges1 = sp.ndimage.morphology.binary_dilation(edges1, iterations=5)
# dilEdges2 = sp.ndimage.morphology.binary_dilation(edges2, iterations=5)

discreteLine = discretizeLine(bestFitLine)
distanceMatrix = np.ones(firstFrame.shape[0:-1], dtype=np.uint)
distanceMatrix[discreteLine[:, 1], discreteLine[:, 0]] = 0
distanceMatrix = ndimage.distance_transform_edt(distanceMatrix)

figure(); imshow(distanceMatrix*edges, interpolation='nearest')

# <codecell>

## now check for each discrete point on line segment what's the closest canny edge in both positive and negative half space

posHalfSpaceClosestEdges = np.zeros_like(discreteLine)
negHalfSpaceClosestEdges = np.zeros_like(discreteLine)

positiveMask = positiveHalfspaceMask(np.vstack((discreteLine[0, :], discreteLine[-1, :])), firstFrame.shape[0:2], True)
figure(); imshow(positiveMask, interpolation='nearest')

dCoords = discreteLine[0, :] - discreteLine[-1, :] ## delta coordinates, dX and dY
discreteNormals = np.array([[-dCoords[1], dCoords[0]],[dCoords[1], -dCoords[0]]])

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.imshow(distanceMatrix*edges, interpolation='nearest')

ax.plot(bestFitLine[:, 0], bestFitLine[:, 1], c='r')

for i in xrange(0, len(discreteLine)) :
    print i, 
    currentPoint = discreteLine[i, :]
    splitPoints = findSplitSegment(np.hstack((discreteNormals[0, :], currentPoint)), [firstFrame.shape[0]-1,firstFrame.shape[1]-1])
    seg = np.vstack((splitPoints[0, :], splitPoints[1, :]))
    dSeg = discretizeLine(seg)
    
    mask = np.zeros(firstFrame.shape[0:-1], dtype=np.uint)
    mask[dSeg[:, 1], dSeg[:, 0]] = 1
    
    filteredDistMat = distanceMatrix*edges*mask*positiveMask
    posHalfSpaceClosestEdges[i, :] = np.argwhere(filteredDistMat == np.min(filteredDistMat[np.where(filteredDistMat > 0)]))[0]
    
    filteredDistMat = distanceMatrix*edges*mask*(1-positiveMask)
    negHalfSpaceClosestEdges[i, :] = np.argwhere(filteredDistMat == np.min(filteredDistMat[np.where(filteredDistMat > 0)]))[0]
    
    ax.scatter(splitPoints[:, 0], splitPoints[:, 1], c='r')
    draw()

ax.scatter(negHalfSpaceClosestEdges[:, 1], negHalfSpaceClosestEdges[:, 0], c='y')

ax.scatter(posHalfSpaceClosestEdges[:, 1], posHalfSpaceClosestEdges[:, 0], c='g')
draw()

# <codecell>

## find distances between closest edges and best fit line
As = np.repeat(np.reshape(line[2:4], [1, 2]), len(testPoints), axis=0)
Ns = np.repeat(np.reshape(line[0:2], [1, 2]), len(testPoints), axis=0)

# remember that negHalfSpaceClosestEdges and posHalfSpaceClosestEdges contain row, column indices not x, y coords
closestEdges = np.array([negHalfSpaceClosestEdges[:, 1], negHalfSpaceClosestEdges[:, 0]]).T 
negHSEdgeDists = np.linalg.norm(As-closestEdges - Ns*np.repeat(np.reshape(np.dot(As-closestEdges, line[0:2]), [len(Ns), 1]), 1, axis=1), axis=1)

closestEdges = np.array([posHalfSpaceClosestEdges[:, 1], posHalfSpaceClosestEdges[:, 0]]).T
posHSEdgeDists = np.linalg.norm(As-closestEdges - Ns*np.repeat(np.reshape(np.dot(As-closestEdges, line[0:2]), [len(Ns), 1]), 1, axis=1), axis=1)


figure(); plot(negHSEdgeDists, 'r', posHSEdgeDists, 'b')

# <codecell>

## compute histogram
negHist, negBins = np.histogram(negHSEdgeDists)
posHist, posBins = np.histogram(posHSEdgeDists)

figure()
plt.bar(negBins[:-1], negHist, color='yellow', width=2)
plt.bar(posBins[:-1], posHist, color='green', width=2)
plt.xlim(min(np.hstack((negBins, posBins))), max(np.hstack((negBins, posBins))))
plt.show()   

# <codecell>

testSortNeg = np.argsort(negHist)
print testSortNeg
print negHist
print negBins
print negHist[testSortNeg]
print negBins[testSortNeg+1]
print negBins[np.argmax(negHist+1)]
print np.sum(negBins[1:len(negBins)]*np.array(negHist, dtype=float)/np.max(negHist))/(len(negBins)-1)
print np.sum((negBins[testSortNeg+1])[-3:]*np.array((negHist[testSortNeg])[-3:], dtype=float)/np.max(negHist))/3.0
print np.sum((negBins[testSortNeg+1])[-3:])/3.0
print np.mean((negBins[testSortNeg+1])[-3:])


# <codecell>

## compute histogram
negHist, negBins = np.histogram(negHSEdgeDists)
posHist, posBins = np.histogram(posHSEdgeDists)

figure()
plt.bar(negBins[:-1], negHist, color='yellow', width=2)
plt.bar(posBins[:-1], posHist, color='green', width=2)
plt.xlim(min(np.hstack((negBins, posBins))), max(np.hstack((negBins, posBins))))
plt.show()

## find 4 corners of box around bone to use as active shape input
# the bbox is aligned to the bestFitLine
bbox = np.zeros([4, 2])

# distance from bestFitLine to edge of bbox in positive and negative halfspace respectively

distType = 2

if distType == 0 :
    ## distance is given by mean distance to closest edge points
    posDist = np.mean(posHSEdgeDists)
    negDist = np.mean(negHSEdgeDists)
elif distType == 1:
    ## distance is given by the right boundary of highest value bin in distance to edge histogram
    posDist = posBins[np.argmax(posHist)+1]
    negDist = negBins[np.argmax(negHist)+1]
elif distType == 2:
    ## distance is the mean of right boundary of top 3 highest value bins in distance to edge histogram
    posDist = np.mean((posBins[np.argsort(posHist)+1])[-3:])
    negDist = np.mean((negBins[np.argsort(negHist)+1])[-3:])

print posDist
print negDist

## HACK:::HACK
inflation = 0.5
inflatedStart = discreteLine[0, :] + (discreteLine[-1, :]-discreteLine[0, :])*(1+inflation*0.1)
print discreteLine[0, :], inflatedStart
inflatedEnd = discreteLine[-1, :] + (discreteLine[0, :]-discreteLine[-1, :])*(1+inflation*0.1)
print discreteLine[1, :], inflatedEnd

bbox[0, :] = inflatedStart + posDist*(1+inflation)*discreteNormals[1, :]/np.linalg.norm(discreteNormals[1, :])
bbox[1, :] = inflatedEnd + posDist*(1+inflation)*discreteNormals[1, :]/np.linalg.norm(discreteNormals[1, :])
bbox[2, :] = inflatedEnd + negDist*(1+inflation)*discreteNormals[0, :]/np.linalg.norm(discreteNormals[0, :])
bbox[3, :] = inflatedStart + negDist*(1+inflation)*discreteNormals[0, :]/np.linalg.norm(discreteNormals[0, :])
print bbox

## inflate bbox


fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.autoscale(False)
ax.plot(bestFitLine[:, 0], bestFitLine[:, 1], c='r')
ax.plot(bbox[:, 0], bbox[:, 1], c='y')
draw()

# discreteLine[-1, :]

# <codecell>

## build the snake initialization by supersampling the bbox
numOfPoints = 100
interval = (np.linalg.norm(bbox[0, :]-bbox[1, :])+np.linalg.norm(bbox[1, :]-bbox[2, :]))/(numOfPoints*0.5)
snake = []
for i, j in zip(arange(0, len(bbox)), np.hstack((arange(1, len(bbox)), 0))):
    currentPoint = bbox[i, :]
#     snake.append(currentPoint)
    currentDir = (bbox[j, :]-bbox[i, :])/np.linalg.norm(bbox[j, :]-bbox[i, :])
    while np.linalg.norm(currentPoint-bbox[j, :]) > interval :
        snake.append(currentPoint)
        currentPoint = currentPoint + interval*currentDir
    
    ##divide last remaining bit into half
    snake.append((snake[-1]+bbox[j, :])/2)

snake = np.array(snake)
print snake.shape

# <codecell>

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.autoscale(False)
ax.plot(bestFitLine[:, 0], bestFitLine[:, 1], c='r')
ax.scatter(snake[:, 0], snake[:, 1], c='y')
draw()

# <codecell>

testBox = np.copy(bbox)
print testBox
testMidPoint = (discreteLine[0, :]+discreteLine[-1, :])/2
testMidPoint = np.repeat(np.reshape(testMidPoint, [1, len(testMidPoint)]), len(testBox), axis=0)
print testMidPoint
testInflated = ((testBox-testMidPoint)*1.2)+testMidPoint

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.imshow(distanceMatrix*edges, interpolation='nearest')
ax.autoscale(False)
ax.plot(bestFitLine[:, 0], bestFitLine[:, 1], c='r')
ax.plot(testBox[:, 0], testBox[:, 1], c='y')
ax.plot(testInflated[:, 0], testInflated[:, 1], c='g')
draw()

# <codecell>

points = []
points.append(np.array([0, 1]))
points.append(np.array([0, 2]))
points.append(np.array([0, 3]))
if any(np.array([0, 5]) == p for p in points):
    print "yay"

# <codecell>

#sign( (Bx-Ax)*(Y-Ay) - (By-Ay)*(X-Ax) )
print np.vstack((discreteLine[0, :], discreteLine[-1, :]))
positiveMask = positiveHalfspaceMask(np.vstack((discreteLine[0, :], discreteLine[-1, :])), firstFrame.shape[0:2], True)
figure(); imshow(positiveMask, interpolation='nearest')

# <codecell>

print discreteLine

# <codecell>

tmp = np.where((distanceMatrix*edges) == np.min((distanceMatrix*edges)[np.nonzero(distanceMatrix*edges)]))
print tmp
ax.scatter(tmp[1], tmp[0], c='g')

# <codecell>

points = findSplitSegment(line, [firstFrame.shape[0]-1,firstFrame.shape[1]-1])


fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.imshow(distanceMatrix*edges, interpolation='nearest')

ax.scatter(points[:, 0], points[:, 1], c='r')
#     ax.scatter(point2[0], point2[1], c='b')
ax.plot(bestFitLine[:, 0], bestFitLine[:, 1], c='r')

