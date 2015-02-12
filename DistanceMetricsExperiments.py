# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import scipy as sp
import cv2
import cv
import glob
import time
import sys
import os

from _emd import emd

from Queue import Queue

from pygraph.classes.digraph import digraph
from pygraph.classes.graph import graph
from pygraph.readwrite.dot import write
from pygraph.algorithms.minmax import shortest_path

import pygraphviz as gv

import VideoTexturesUtils as vtu
import GraphWithValues as gwv

dataFolder = "/home/ilisescu/PhD/data/"

def distance(f1, f2):
#     return np.sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    return np.sqrt( (f1[0] - f2[0])**2  + (f1[1] - f2[1])**2 + (f1[2] - f2[2])**2 )

# <codecell>

## try emd from pdinges

distanceMatrix = np.zeros((numFrames, numFrames))
for r in xrange(0, numFrames) :
    for c in xrange(r+1, numFrames) :
        frame1 = list(np.array(np.reshape(movie[:, :, :, r], (np.prod(movie.shape[0:2]), movie.shape[2])), dtype=np.float32)/255.0)
        frame2 = list(np.array(np.reshape(movie[:, :, :, c], (np.prod(movie.shape[0:2]), movie.shape[2])), dtype=np.float32)/255.0)
#         f1 = []
#         f2 = []
#         for i, j in zip(frame1, frame2):
#             f1.append(i)
#             f2.append(j)
#             print i, tuple(i)
            
#         print len(w1), len(w2)
#         print f1
#         print frame1
        distanceMatrix[r, c] = distanceMatrix[c, r] = emd((frame1, list(arange(0.0, len(frame1)))), (frame2, list(arange(0.0, len(frame1)))), distance)
        print distanceMatrix[r, c]
    print
figure(); imshow(distanceMatrix, interpolation='nearest')
# print emd(([f1[0]], [1.0]), ([f1[10]], [11.0]), distance)

# <codecell>

figure(); imshow(distanceMatrix, interpolation='nearest')

distMat = vtu.filterDistanceMatrix(distanceMatrix, 1, True)
idxCorrection = 1
figure(); imshow(distMat, interpolation='nearest')
gwv.showCustomGraph(distanceMatrix)

# <codecell>

print movie[:, :, :, 0].shape
print np.sum(np.sum(movie[:, :, :, 0], axis=0), axis=0)
print np.sum(movie[:, :, 0, 0]), np.sum(movie[:, :, 1, 0]), np.sum(movie[:, :, 2, 0])
print np.sum(np.array(f1), axis=0)
print len(f1)

# <codecell>

## try the cosine distance thing between difference feature vector and unit vector
def distEuc2(f1, f2) :
    ## gives sum over squared intensity values for each image
    ff1 = np.sum(f1*f1, axis=1)
    ff2 = np.sum(f2*f2, axis=1)
#     print ff1.shape
#     print ff2.shape
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
#     print "askdfh", np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1).shape, np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0).shape
    d = np.sqrt(np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1)+np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0) - 2*np.dot(f1, f2.T))
    return d

idxCorrection = 0
sampleData = "pendulum/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "0*.png")
maxFrames = len(frames)
frames = np.sort(frames)[0:maxFrames]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames

frameSize = cv2.imread(frames[0]).shape
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=np.uint8)
for i in range(0, numFrames) :
    movie[:, :, :, i] = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB), dtype=np.uint8)
    
    sys.stdout.write('\r' + "Loaded frame " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()

print
blocksPerDim = 4
subDivisions = blocksPerDim**2
distanceMatrix = np.zeros((numFrames, numFrames))
for r in xrange(0, numFrames) :#7, 8):#0, numFrames) :
    for c in xrange(r+1, numFrames) :#17, 18):#r+1, numFrames) :
        feats = np.zeros(subDivisions)
        tmp = np.zeros((blocksPerDim, blocksPerDim))
        divHeight = movie.shape[0]/blocksPerDim
        divWidth = movie.shape[1]/blocksPerDim
        for i in xrange(0, blocksPerDim) :
            for j in xrange(0, blocksPerDim) :
                ## compute difference between each corresponding quadrant
#                 figure(); 
#                 subplot(211); imshow(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r])
#                 subplot(212); imshow(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c])
                
                b1 = np.array(np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r], (divHeight*divWidth*movie.shape[-2], 1)).T, dtype=int)
                b2 = np.array(np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], (divHeight*divWidth*movie.shape[-2], 1)).T, dtype=int)
                feats[i*blocksPerDim+j] = distEuc2(b1, b2)#np.sqrt(np.sum(np.power(b1-b2, 2)))#distEuc2(b1, b2)
#                 print i, j, feats[i*blocksPerDim+j]
                tmp[i, j] = feats[i*blocksPerDim+j]
#                 f1.append(distEuc2())
                ## normalize corresponding subDivisions-dim vector
                ## compute dot product between the normalized vector and unit vector
#                 f1.append(np.array(np.sum(np.sum(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r], axis=0), axis=0), dtype=float32))
#                 f2.append(np.array(np.sum(np.sum(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], axis=0), axis=0), dtype=float32))
        
#         print len(f1), len(f2)
#         print feats
        ## normalize corresponding subDivisions-dim vector
        feats /= np.linalg.norm(feats)
#         print r, c, feats
        ## compute dot product between the normalized vector and unit vector
        distanceMatrix[r, c] = distanceMatrix[c, r] = np.exp(-np.dot(feats, np.ones(len(feats))))
#         print distanceMatrix[r, c]
#         figure(); imshow(movie[:, :, :, r])
#         figure(); imshow(movie[:, :, :, c])
#     print

# distanceMatrix /= np.linalg.norm(distanceMatrix)
# figure(); imshow(distanceMatrix, interpolation='nearest')
# figure(); imshow(tmp, interpolation='nearest')

distMat = vtu.filterDistanceMatrix(distanceMatrix, 1, True)
idxCorrection = 1
figure(); imshow(distMat, interpolation='nearest')
gwv.showCustomGraph(distanceMatrix)

# <codecell>

gwv.showCustomGraph(tmp)
figure(); imshow(movie[:, :, :, r])
figure(); imshow(movie[:, :, :, c])

# <codecell>

figure(); 
i = 3
j = 1
subplot(211); imshow(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r])
subplot(212); imshow(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c])

gwv.showCustomGraph(np.sum(np.sqrt(np.power(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r]-movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], 2)), axis=-1))


b1 = np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r], (divHeight*divWidth*movie.shape[-2], 1)).T
b2 = np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], (divHeight*divWidth*movie.shape[-2], 1)).T
print "diff", np.sum(np.sum(np.sqrt(np.power(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r]-movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], 2)), axis=-1))
print np.sqrt(np.sum(np.power(b1-b2, 2))), np.sqrt(np.sum(np.power(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r]-movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], 2)))
print np.sqrt(np.dot(b1-b2, (b1-b2).T))
print distEuc2(b1, b2)

figure(); 
i = 2
j = 1
subplot(211); imshow(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r])
subplot(212); imshow(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c])

# figure(); imshow(np.sqrt(np.power(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r]-movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], 2)), interpolation='nearest')
gwv.showCustomGraph(np.sum(np.sqrt(np.power(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r]-movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], 2)), axis=-1))

b1 = np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r], (divHeight*divWidth*movie.shape[-2], 1)).T
b2 = np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], (divHeight*divWidth*movie.shape[-2], 1)).T
print np.sum(np.sum(np.sqrt(np.power(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r]-movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], 2)), axis=-1))
print np.sqrt(np.sum(np.power(b1-b2, 2))), np.sqrt(np.sum(np.power(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r]-movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], 2)))
print np.sqrt(np.dot(b1-b2, (b1-b2).T)), np.sqrt(np.dot(b1, b1.T)+np.dot(b2, b2.T)-2*np.dot(b1, b2.T))
print distEuc2(b1, b2)

# <codecell>

print np.sum(np.power(b1-b2, 2)), np.dot(b1-b2, (b1-b2).T), np.sum((b1-b2)*(b1-b2))
print np.inner(b1-b2, (b1-b2)), sum((b1-b2)[:]*(b1-b2)[:])
a = np.arange(0, 10800)#np.array([1,2,3,7,9])
b = np.arange(5, 10805)#np.array([2,1,5,8,2])
print np.inner(a, b), np.sum(a*b), a.shape, b.shape
t1 = np.ndarray.flatten(b1-b2)
t2 = np.copy(t1)
print np.sum(t1-t2)
print np.inner(t1, t2), np.sum(t1*t2), t1.shape, t2.shape
print np.sqrt(np.sum(np.power(b1-b2, 2))), np.sqrt(np.sum(b1*b1)+np.sum(b2*b2)-2*np.sum(b1*b2))
print np.sum(b1*b1), np.sum(b2*b2), 2*np.sum(b1*b2), np.sum(b1*b1)+np.sum(b2*b2), np.dot(np.ndarray.flatten(b1), np.ndarray.flatten(b2))
print np.sum(a*a), np.sum(b*b), 2*np.sum(a*b), np.sum(a*a)+np.sum(b*b), np.dot(np.ndarray.flatten(a), np.ndarray.flatten(b))

c1 = np.ndarray.flatten(np.copy(b1[0, 0:50]))
c2 = np.ndarray.flatten(np.copy(b2[0, 0:50]))
print np.sum(c1*c1), np.sum(c2*c2), 2*np.sum(c1*c2), np.sum(c1*c1)+np.sum(c2*c2), np.dot(np.ndarray.flatten(c1), np.ndarray.flatten(c2))
print c1.shape, c1
print c2.shape, c2
print np.dot(c1, c2), np.sum(c1*c2)
print c1*c2

# <codecell>

## how do I reshape to divide data from 1D array
mov = np.copy(movie[:, :, :, 0:3])
print mov.shape
data = np.reshape(mov, [np.prod(mov.shape[0:-1]), mov.shape[-1]]).T
print data.shape
blockedData = data.reshape((data.shape[0]*subDivisions, data.shape[-1]/subDivisions))
print blockedData.shape
figure(); imshow(data[0, :].reshape((240, 240, 3)))
figure(); imshow(blockedData[0, :].reshape((mov.shape[0]/blocksPerDim, mov.shape[1]/blocksPerDim, mov.shape[2])))
figure(); imshow(blockedData[0, :].reshape((mov.shape[0]/subDivisions, mov.shape[1], mov.shape[2])))
# print mov

# <codecell>

## try defining a stencil for each block and use that to index image to get pixel values corresponding to that block
img = np.copy(mov[:, :, :, 0])
figure(), imshow(img)
## stencil for upper left block
stencil = np.zeros(img.shape, dtype=int)
bRows = img.shape[0]/blocksPerDim
bCols = img.shape[1]/blocksPerDim
stencil[0*bRows:0*bRows+bRows, 0*bCols:0*bCols+bCols, :] = np.ones((bRows, bCols, 3))
figure(); imshow(stencil, interpolation='nearest')
block = np.reshape(img[np.argwhere(stencil==1)[:, 0], np.argwhere(stencil==1)[:, 1], np.argwhere(stencil==1)[:, 2]], (60, 60, 3))
figure(); imshow(block)
print img[np.argwhere(stencil==1)[:, 0], np.argwhere(stencil==1)[:, 1], np.argwhere(stencil==1)[:, 2]].shape
print np.argwhere(stencil==1).shape

# <codecell>

frameSize = imageSize
blockSize = movie.shape[-1]
frames = glob.glob(outputData + "/0*.png")
mattes = glob.glob(outputData + "/matte-0*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)
f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
i = 0
# data1 = np.zeros((movie.shape[-1]*subDivisions, bRows*bCols*frameSize[-1]))
data1 = np.zeros((blockSize, bRows*bCols*frameSize[-1], subDivisions))
for f, idx in zip(xrange(i*blockSize, i*blockSize+blockSize), xrange(0, blockSize)) :
    img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0
    if os.path.isfile(mattes[f]) :
        alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY))/255.0
        img *= np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
#         else :
#             img /= 255.0
        
    f1s[:, :, :, idx] = img
    for s in xrange(0, len(stencils)) :
        index = s + idx*len(stencils)
#         data1[index, :] = img[stencils[s]]
        data1[idx, :, s] = img[stencils[s]]

print data1.shape
# data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
# print data1.shape

features = np.zeros((blockSize, blockSize, subDivisions))
for s in xrange(0, subDivisions) :
    features[:, :, s] = distEuc2(data1[:, :, s], data1[:, :, s])

## normalize
norms = np.repeat(np.reshape(np.linalg.norm(features, axis=-1), (blockSize, blockSize, 1)), subDivisions, axis=-1)
norms += np.repeat(np.reshape(np.eye(features.shape[0]), (blockSize, blockSize, 1)), subDivisions, axis=-1)
features /= norms

distMat = np.exp(-np.sum(features, axis=-1))
distMat *= (1-np.eye(distMat.shape[0]))

# <codecell>

gwv.showCustomGraph(distMat)

# <codecell>

figure(); imshow(data1[r, :, 9].reshape((60, 60, 3)), interpolation='nearest')
figure(); imshow(data1[c, :, 9].reshape((60, 60, 3)), interpolation='nearest')
i = 2
j = 1
b1 = np.array(np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r], (divHeight*divWidth*movie.shape[-2], 1)).T, dtype=int)
b2 = np.array(np.reshape(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], (divHeight*divWidth*movie.shape[-2], 1)).T, dtype=int)
figure(); imshow(b1.reshape((60, 60, 3)), interpolation='nearest')
figure(); imshow(b2.reshape((60, 60, 3)), interpolation='nearest')

figure(); imshow(np.array(f1s[:, :, :, r], dtype=uint8))
figure(); imshow(np.array(f1s[:, :, :, c], dtype=uint8))
figure(); imshow(movie[:, :, :, r])
figure(); imshow(movie[:, :, :, c])

print np.sum(data1[r, :, 9]-b1), np.sum(data1[c, :, 9]-b2)

# <codecell>

gwv.showCustomGraph(features[r, c, :].reshape((4, 4)))
print data1[7, :, -1].shape
gwv.showCustomGraph(np.reshape(data1[7, :, -1], (60, 60, 3))[:, :, 0])
print distEuc2(np.reshape(data1[7, :, -1], (1, 10800)), np.reshape(data1[17, :, -1], (1, 10800)))

# <codecell>

print features[r, c, :]
print features.shape
print np.sum(features[0, 1, :], axis=-1)
print np.dot(features[0, 1, :], np.ones(len(features[0, 1, :])))
print np.exp(-np.sum(features, axis=-1)).shape
gwv.showCustomGraph(np.sum(features, axis=-1))
gwv.showCustomGraph(distMat)
print np.sum(features, axis=-1)

# <codecell>

gwv.showCustomGraph(np.sum(feats, axis=-1))
gwv.showCustomGraph(np.linalg.norm(feats, axis=-1))

# <codecell>

figure(); imshow(data1[0, :, 9].reshape((60, 60, 3)))
gwv.showCustomGraph(distEuc2(data1[:, :, 9], data1[:, :, 9]))

# <codecell>

## given block sizes and img sizes build indices representing each block
imageSize = movie.shape[0:-1]
stencils = []#np.zeros(((imageSize[0]/blocksPerDim)*(imageSize[1]/blocksPerDim)*imageSize[-1], imageSize[-1], subDivisions))
# print stencils.shape
bRows = imageSize[0]/blocksPerDim
bCols = imageSize[1]/blocksPerDim
for r in xrange(0, blocksPerDim) :
    for c in xrange(0, blocksPerDim) :
        idx = c + r*blocksPerDim
        stencil = np.zeros(imageSize, dtype=int)
        stencil[r*bRows:r*bRows+bRows, c*bCols:c*bCols+bCols] = np.ones((bRows, bCols, imageSize[-1]))
#         stencils[:, :, idx] = np.argwhere(stencil==1)
        stencils.append(list(np.argwhere(stencil==1).T))
#         print idx

print img[stencils[0]].shape

# <codecell>

tmpMat = np.load(outputData + "/vanilla_distMat.npy")
gwv.showCustomGraph(tmpMat)

# <codecell>

def distEuc2(f1, f2) :
    ## gives sum over squared intensity values for each image
    ff1 = np.sum(f1*f1, axis=1)
    ff2 = np.sum(f2*f2, axis=1)
#     print ff1.shape
#     print ff2.shape
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
#     print "askdfh", np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1).shape, np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0).shape
    d = np.sqrt(np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1)+np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0) - 2*np.dot(f1, f2.T))
    return d
print np.sqrt(np.sum(np.power(b1-b2, 2)))
print (b1-b2).shape, (b2-b1).shape
print np.sqrt(np.dot(b1-b2, (b1-b2).T))
print np.sqrt(np.dot(b1, b1.T) + np.dot(b2, b2.T) - 2*np.dot(b1,b2.T))
print distEuc2(b1, b2)
print np.dot(b1, b1.T)
print np.sum((b1/255)*(b1/255), axis=1)

# <codecell>

## try the emd on pendulum data
idxCorrection = 0
sampleData = "pendulum/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "0*.png")
maxFrames = len(frames)
frames = np.sort(frames)[0:maxFrames]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames

frameSize = cv2.imread(frames[0]).shape
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=np.uint8)
for i in range(0, numFrames) :
    movie[:, :, :, i] = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB), dtype=np.uint8)
    
    sys.stdout.write('\r' + "Loaded frame " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()

print
blocksPerDim = 10
subDivisions = blocksPerDim**2
distanceMatrix = np.zeros((numFrames, numFrames))
for r in xrange(0, numFrames) :
    for c in xrange(r+1, numFrames) :
#         f1 = np.reshape(movie[:, :, :, r], (np.prod(movie.shape[0:2]), movie.shape[2]))
#         f1 = np.array(np.hstack((f1, np.reshape(range(0, len(f1)), (len(f1), 1)))), dtype=np.float)/255.0
#         frame1 = cv.CreateMat(f1.shape[0], f1.shape[1], cv.CV_32FC1)
#         cv.Convert(cv.fromarray(np.ascontiguousarray(f1)), frame1)

#         f2 = np.reshape(movie[:, :, :, c], (np.prod(movie.shape[0:2]), movie.shape[2]))
#         f2 = np.array(np.hstack((f2, np.reshape(range(0, len(f2)), (len(f2), 1)))), dtype=np.float)/255.0
#         frame2 = cv.CreateMat(f2.shape[0], f2.shape[1], cv.CV_32FC1)
#         cv.Convert(cv.fromarray(np.ascontiguousarray(f2)), frame2)
        
# #         print frame1, frame2
#         distanceMatrix[r, c] = distanceMatrix[c, r] = cv.CalcEMD2(frame1, frame2, cv.CV_DIST_L2)
#         print distanceMatrix[r, c]
#         frame1 = np.array(np.reshape(movie[:, :, :, r], (np.prod(movie.shape[0:2]), movie.shape[2])), dtype=np.float32)/255.0
        f1 = []
        f2 = []
        divHeight = movie.shape[0]/blocksPerDim
        divWidth = movie.shape[1]/blocksPerDim
        for i in xrange(0, blocksPerDim) :
            for j in xrange(0, blocksPerDim) :
                f1.append(np.array(np.sum(np.sum(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, r], axis=0), axis=0), dtype=float32))
                f2.append(np.array(np.sum(np.sum(movie[i*divHeight:(i+1)*divHeight, j*divWidth:(j+1)*divWidth, :, c], axis=0), axis=0), dtype=float32))
#         print len(f1), len(f2)
#         print np.sum(np.array(f1), axis=0), np.sum(np.array(f2), axis=0)
        
#         frame2 = list(np.array(np.reshape(movie[:, :, :, c], (np.prod(movie.shape[0:2]), movie.shape[2])), dtype=np.float32)/255.0)
#         print len(frame1), len(frame2)
        distanceMatrix[r, c] = distanceMatrix[c, r] = emd((f1, list(arange(0.0, len(f1)))), (f2, list(arange(0.0, len(f2)))), distance)
        print distanceMatrix[r, c]
    print

figure(); imshow(distanceMatrix, interpolation='nearest')

distMat = vtu.filterDistanceMatrix(distanceMatrix, 1, True)
idxCorrection = 1
figure(); imshow(distMat, interpolation='nearest')
gwv.showCustomGraph(distanceMatrix)

# <codecell>

def gauss(x, mean, variance) :
    normTerm = 1.0#np.sqrt(2*variance*np.pi)
    return np.exp(-np.power(x-mean, 2.0)/(2*variance))/normTerm

def multiGauss(X, Y, theta, mean, variance) :
    normTerm = 1.0#np.sqrt(np.linalg.det(covar)*np.power(2*np.pi, x.shape[0]))
    if len(mean) != 2 or len(variance) != 2 :
        raise BaseException, "multiGauss needs as many means and variances as k, where k=2"
        
    a = (np.cos(theta)**2)/(2*variance[0])+(np.sin(theta)**2)/(2*variance[1])
    b = np.sin(2*theta)/(4*variance[0])+np.sin(2*theta)/(4*variance[1])
    c = (np.sin(theta)**2)/(2*variance[0])+(np.cos(theta)**2)/(2*variance[1])
    
    return np.exp(-(a*np.power(X-mean[0], 2)+2*b*(X-mean[0])*(Y-mean[1])+c*np.power(Y-mean[1], 2)))/normTerm

# <codecell>

def smoothStep(x, mean, interval, steepness) :
    a = mean-np.floor(interval/2.0)
    b = mean-np.floor((interval*steepness)/2.0)
    c = mean+np.ceil((interval*steepness)/2.0)
    d = mean+np.ceil(interval/2.0)
    print a, b, c, d
    
    ## find step from 0 to 1
    step1 = np.clip((x - a)/(b - a), 0.0, 1.0);
    step1 = step1*step1*step1*(step1*(step1*6 - 15) + 10);
    
    ## find step from 1 to 0
    step2 = np.clip((x - d)/(c - d), 0.0, 1.0);
    step2 = step2*step2*step2*(step2*(step2*6 - 15) + 10);
    
    ## combine the two steps together
    result = np.zeros(x.shape)
    result += step1
    result[np.argwhere(step2 != 1.0)] = step2[np.argwhere(step2 != 1.0)]
    return result;

# <codecell>

def getShortestPath(graph, start, end) :
    paths = shortest_path(graph, start)[0]
#     print paths
    curr = end
    path = []
    
    ## no path from start to end
    if curr not in paths :
        return np.array(path), -1
    
    path.append(curr)
    while curr != start :
        curr = paths[curr]
        path.append(curr)
        
    path = np.array(path)[::-1]
    distance = 0
    for i in xrange(1, len(path)) :
        distance += graph.edge_weight((path[i-1], path[i]))
    return path, distance

# <codecell>

# Draw graph as PNG
def drawGraph(graph, name, yes) :
    if yes :
        dot = write(graph)
        gvv = gv.AGraph()
        gvv.from_string(dot)
        gvv.layout(prog='circo')
        gvv.draw(path=name)

# <codecell>

## made up data of a dot moving left to right and top to bottom
frameSize = np.array([7, 7, 3])
# numFrames = 31
# movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=uint8)
# movie[0, 3, :, 0] = movie[1, 3, :, 1] = movie[2, 3, :, 2] = movie[3, 3, :, 3] = movie[4, 3, :, 4] = 255
# movie[5, 3, :, 5] = movie[6, 3, :, 6] = movie[5, 3, :, 7] = movie[4, 3, :, 8] = movie[3, 3, :, 9] = 255
# movie[2, 3, :, 10] = movie[1, 3, :, 11] = movie[0, 3, :, 12] = movie[1, 3, :, 13] = movie[2, 3, :, 14] = 255
# movie[3, 3, :, 15] = movie[3, 2, :, 16] = movie[3, 1, :, 17] = movie[3, 0, :, 18] = movie[3, 1, :, 19] = 255
# movie[3, 2, :, 20] = movie[3, 3, :, 21] = movie[3, 4, :, 22] = movie[3, 5, :, 23] = movie[3, 6, :, 24] = 255
# movie[3, 5, :, 25] = movie[3, 4, :, 26] = movie[3, 3, :, 27] = movie[3, 2, :, 28] = movie[3, 1, :, 29] = 255
# movie[3, 0, :, 30] = 255

numFrames = 31
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=uint8)
movie[3, 0, :, 0] = movie[3, 1, :, 1] = movie[3, 2, :, 2] = movie[3, 3, :, 3] = movie[3, 4, :, 4] = 255
movie[3, 5, :, 5] = movie[3, 6, :, 6] = movie[3, 5, :, 7] = movie[3, 4, :, 8] = movie[3, 3, :, 9] = 255
movie[3, 2, :, 10] = movie[3, 1, :, 11] = movie[3, 0, :, 12] = movie[3, 1, :, 13] = movie[3, 2, :, 14] = 255
movie[3, 3, :, 15] = movie[2, 3, :, 16] = movie[1, 3, :, 17] = movie[0, 3, :, 18] = movie[1, 3, :, 19] = 255
movie[2, 3, :, 20] = movie[3, 3, :, 21] = movie[4, 3, :, 22] = movie[5, 3, :, 23] = movie[6, 3, :, 24] = 255
movie[5, 3, :, 25] = movie[4, 3, :, 26] = movie[3, 3, :, 27] = movie[2, 3, :, 28] = movie[1, 3, :, 29] = 255
movie[0, 3, :, 30] = 255

# numFrames = 26
# movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=uint8)
# movie[3, 0, :, 0] = movie[3, 1, :, 1] = movie[3, 2, :, 2] = movie[3, 3, :, 3] = movie[3, 4, :, 4] = 255
# movie[3, 5, :, 5] = movie[3, 6, :, 6] = movie[3, 5, :, 7] = movie[3, 4, :, 8] = movie[3, 3, :, 9] = 255
# movie[3, 2, :, 10] = movie[3, 1, :, 11] = movie[3, 0, :, 12] = movie[0, 3, :, 13] = movie[1, 3, :, 14] = 255
# movie[2, 3, :, 15] = movie[3, 3, :, 16] = movie[4, 3, :, 17] = movie[5, 3, :, 18] = movie[6, 3, :, 19] = 255
# movie[5, 3, :, 20] = movie[4, 3, :, 21] = movie[3, 3, :, 22] = movie[2, 3, :, 23] = movie[1, 3, :, 24] = 255
# movie[0, 3, :, 25] = 255

# distanceMatrix = np.zeros((numFrames, numFrames))
# for r in xrange(0, numFrames) :
#     for c in xrange(r+1, numFrames) :
#         rowLoc = np.argwhere(movie[:, :, 0, r] == 255)[0]
#         colLoc = np.argwhere(movie[:, :, 0, c] == 255)[0]
#         distanceMatrix[r, c] = distanceMatrix[c, r] = np.sum(np.abs(rowLoc-colLoc))
#         f1 = np.reshape(np.ndarray.flatten(movie[:, :, 0, r]), (movie.shape[0]*movie.shape[1], 1))
#         f1 = np.array(np.hstack((f1, np.reshape(range(0, len(f1)), (len(f1), 1)))), dtype=np.float)/255.0
#         frame1 = cv.CreateMat(f1.shape[0], f1.shape[1], cv.CV_32FC1)
#         cv.Convert(cv.fromarray(np.ascontiguousarray(f1)), frame1)

#         f2 = np.reshape(np.ndarray.flatten(movie[:, :, 0, c]), (movie.shape[0]*movie.shape[1], 1))
#         f2 = np.array(np.hstack((f2, np.reshape(range(0, len(f2)), (len(f2), 1)))), dtype=np.float)/255.0
#         frame2 = cv.CreateMat(f2.shape[0], f2.shape[1], cv.CV_32FC1)
#         cv.Convert(cv.fromarray(np.ascontiguousarray(f2)), frame2)
        
# #         print frame1, frame2
#         distanceMatrix[r, c] = distanceMatrix[c, r] = cv.CalcEMD2(frame1, frame2, cv.CV_DIST_L2)
#         print distanceMatrix[r, c]
#     print
    
distanceMatrix = np.zeros((numFrames, numFrames))
for r in xrange(0, numFrames) :
    for c in xrange(r+1, numFrames) :
        f1 = np.reshape(movie[:, :, :, r], (np.prod(movie.shape[0:2]), movie.shape[2]))
        f1 = np.array(np.hstack((f1, np.reshape(range(0, len(f1)), (len(f1), 1)))), dtype=np.float)/255.0
#         f1 = np.array(np.hstack((f1, np.reshape(np.repeat(1, len(f1)), (len(f1), 1)))), dtype=np.float)/255.0
        frame1 = cv.CreateMat(f1.shape[0], f1.shape[1], cv.CV_32FC1)
        cv.Convert(cv.fromarray(np.ascontiguousarray(f1)), frame1)

        f2 = np.reshape(movie[:, :, :, c], (np.prod(movie.shape[0:2]), movie.shape[2]))
        f2 = np.array(np.hstack((f2, np.reshape(range(0, len(f2)), (len(f2), 1)))), dtype=np.float)/255.0
#         f2 = np.array(np.hstack((f2, np.reshape(np.repeat(1, len(f2)), (len(f2), 1)))), dtype=np.float)/255.0
        frame2 = cv.CreateMat(f2.shape[0], f2.shape[1], cv.CV_32FC1)
        cv.Convert(cv.fromarray(np.ascontiguousarray(f2)), frame2)

        print frame1, frame2
        distanceMatrix[r, c] = distanceMatrix[c, r] = cv.CalcEMD2(frame1, frame2, cv.CV_DIST_L2)

#         f1 = list(np.array(np.ndarray.flatten(movie[:, :, 0, r]), dtype=np.int))
#         f2 = list(np.array(np.ndarray.flatten(movie[:, :, 0, c]), dtype=np.int))
# #         w1 = []
# #         w2 = []
# #         for i, j in zip(f1, f2):
# #             w1.append(i)
# #             w2.append(j)
            
# #         print len(w1), len(w2)
# #         print f1
#         distanceMatrix[r, c] = distanceMatrix[c, r] = emd.emd(range(0, len(f1)), range(0, len(f2)), f1, f2)
        print distanceMatrix[r, c]
    print

## add noise
# distanceMatrix += np.random.random(distanceMatrix.shape)*(1-np.eye(distanceMatrix.shape[0]))
figure(); imshow(distanceMatrix, interpolation='nearest')

distMat = vtu.filterDistanceMatrix(distanceMatrix, 1, True)
idxCorrection = 1
figure(); imshow(distMat, interpolation='nearest')
gwv.showCustomGraph(distanceMatrix)

# <codecell>

p1 = np.array([134, 240, 75])
p2 = np.array([230, 45, 30])

print np.sqrt(np.sum(np.power(p1-p2, 2)))
print np.sqrt(np.power(np.sum(p1)-np.sum(p2), 2))

# <codecell>

def estimateFutureCost(alpha, p, distanceMatrixFilt, weights) :
    
    distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
    distMat = distMatFilt ** p
    
    last = np.copy(distMat)
    current = np.zeros(distMat.shape)
    
    ## while distance between last and current is larger than threshold
    iterations = 0 
    while np.linalg.norm(last - current) > 0.1 : 
        for i in range(distMat.shape[0]-1, -1, -1) :
#             m = np.min(distMat*weights[1:distanceMatrixFilt.shape[1], 0:-1], axis=1)
            m = np.min(distMat*weights, axis=1)
            distMat[i, :] = (distMatFilt[i, :] ** p) + alpha*m
            
        last = np.copy(current)
        current = np.copy(distMat)
        
        sys.stdout.write('\r' + "Iteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
        sys.stdout.flush()
        
        iterations += 1
    
    print
    print 'finished in', iterations, 'iterations'
    
    return distMat

distMatFut = estimateFutureCost(0.999, 2.0, distMat, np.ones(np.array(distMat.shape)-1))#rangeWeights)
figure(); imshow(distMatFut, interpolation='nearest')

# <codecell>

# weighMat = distMatFut
weightMat = distMat[1:distMat.shape[1], 0:-1]
figure(); imshow(weightMat, interpolation='nearest')

# probMat, cumProb = vtu.getProbabilities(weightMat, 1.0, None, True)
idxCorrection = 2#5 #+= 1

## initNodes and ranges for star dot
# initialNodes = np.array([9, 21])-idxCorrection
# intervals = np.array([11, 11])
## initNodes and ranges for ribbon2_matted
initialNodes = np.array([122, 501, 838, 1106])-idxCorrection
intervals = np.array([251, 441, 169, 339])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(probMat, interpolation='nearest')
# ax.set_autoscale_on(False)
# ax.scatter(initialNodes, initialNodes, c="m", marker="s")

## compute rangeCurves using initialNodes before idxCorrection
rangeCurves = np.zeros(weightMat.shape[-1])
for node, inter in zip(initialNodes, intervals) :
    ## the variances should be set according to situation range
#     rangeCurves += gauss(arange(0.0, len(rangeCurves)), float(node), 5.0)
    rangeCurves += smoothStep(arange(0.0, len(rangeCurves)), float(node), inter, 0.4)
# rangeCurves /= np.max(rangeCurves)
figure(); plot(arange(0, len(rangeCurves)), rangeCurves)
print rangeCurves

# <codecell>

initPoints = np.array([122, 501, 838, 1106]) -4
extraPoints = 4
labeledPoints = np.zeros((numClasses, extraPoints+1), dtype=np.int)
for i in xrange(0, len(initPoints)) :
    labeledPoints[i, :] = range(initPoints[i]-extraPoints/2, initPoints[i]+extraPoints/2+1)
print labeledPoints

fl = np.zeros((np.prod(labeledPoints.shape), numClasses))
for i in xrange(0, numClasses) :
    fl[i*(extraPoints+1):(i+1)*(extraPoints+1), i] = 1
print fl

# <codecell>

## do label propagation as zhu 2003
distances = np.copy(distMat)
# distances = np.copy(distMat[1:distMat.shape[1], 0:-1])
# distances = np.copy(distMatFut)

if False :
    ## use dotstar
    numClasses = 2
    ## init labeled points
    labeledPoints = np.array([[9], [21]])-1
    fl = np.zeros((len(labeledPoints), numClasses))
    fl = np.eye(numClasses)
else :
    ## use ribbon2
    numClasses = 4
    ## init labeled points
    labeledPoints = np.array([[122], [501], [838], [1106]]) -4
    fl = np.eye(numClasses)
    
#     labeledPoints = np.array([[22, 122, 222], [281, 501, 721], [754, 838, 922], [956, 1106, 1256]]) -4
#     fl = np.zeros((np.prod(labeledPoints.shape), numClasses))
#     fl[0:3, 0] = 1
#     fl[3:6, 1] = 1
#     fl[6:9, 2] = 1
#     fl[9:, 3] = 1

    initPoints = np.array([122, 501, 838, 1106]) -4
    extraPoints = 0
    labeledPoints = np.zeros((numClasses, extraPoints+1), dtype=np.int)
    for i in xrange(0, len(initPoints)) :
        labeledPoints[i, :] = range(initPoints[i]-extraPoints/2, initPoints[i]+extraPoints/2+1)

    fl = np.zeros((np.prod(labeledPoints.shape), numClasses))
    for i in xrange(0, numClasses) :
        fl[i*(extraPoints+1):(i+1)*(extraPoints+1), i] = 1
    
print numClasses, labeledPoints
print fl

## order w to have labeled nodes at the top-left corner
orderedDist = np.copy(distances)
flatLabeled = np.ndarray.flatten(labeledPoints); print flatLabeled
for i in xrange(0, len(flatLabeled)) :
    #shift flatLabeled[i]-th row up to i-th row and adapt remaining rows
    tmp = np.copy(orderedDist)
    orderedDist[i, :] = tmp[flatLabeled[i], :]
    orderedDist[i+1:, :] = np.vstack((tmp[i:flatLabeled[i], :], tmp[flatLabeled[i]+1:, :]))
    #shift flatLabeled[i]-th column left to i-th column and adapt remaining columns
    tmp = np.copy(orderedDist)
    orderedDist[:, i] = tmp[:, flatLabeled[i]]
    orderedDist[:, i+1:] = np.hstack((tmp[:, i:flatLabeled[i]], tmp[:, flatLabeled[i]+1:]))
#     print len(flatLabeled)+flatLabeled[i]
    
gwv.showCustomGraph(distances)
gwv.showCustomGraph(orderedDist)

## compute weights
w, cumW = vtu.getProbabilities(orderedDist, 0.05, None, False)
gwv.showCustomGraph(w)

l = len(flatLabeled)
n = orderedDist.shape[0]
## compute graph laplacian
L = np.diag(np.sum(w, axis=0)) - w
# gwv.showCustomGraph(L)

## propagate labels
fu = np.dot(np.dot(-np.linalg.inv(L[l:, l:]), L[l:, 0:l]), fl)

## use class mass normalization to normalize label probabilities
q = np.sum(fl)+1
fu_CMN = fu*(np.ones(fu.shape)*(q/np.sum(fu)))

# <codecell>

## iterative approach

# <codecell>

print fu
# print fu_CMN

print flatLabeled
## add labeled frames to fu and plot
labelProbs = np.array(fu[0:flatLabeled[0]])
print labelProbs.shape
for i in xrange(1, len(flatLabeled)) :
#     print flatLabeled[i]+i, flatLabeled[i+1]-i
#     print fu[flatLabeled[i]+i:flatLabeled[i+1]-i, :]
    labelProbs = np.vstack((labelProbs, fl[i-1, :]))
    print labelProbs.shape, flatLabeled[i-1]-(i-1), flatLabeled[i]-i
    labelProbs = np.vstack((labelProbs, fu[flatLabeled[i-1]-(i-1):flatLabeled[i]-i, :]))
    print labelProbs.shape


labelProbs = np.vstack((labelProbs, fl[-1, :]))
labelProbs = np.vstack((labelProbs, fu[flatLabeled[-1]-len(flatLabeled)+1:, :]))
# labelProbs = labelProbs[1:, :]
print labelProbs, labelProbs.shape

clrs = ['r', 'g', 'b', 'y']
fig1 = figure()
xlabel('all points')
fig2 = figure()
xlabel('only unlabeled')
fig3 = figure()
xlabel('only unlabeled + CMN')

for i in xrange(0, numClasses) :
    figure(fig1.number); plot(labelProbs[:, i], clrs[i])
    figure(fig2.number); plot(fu[:, i], clrs[i])
    figure(fig3.number); plot(fu_CMN[:, i], clrs[i])
    
    for node in labeledPoints[i, :] :
        figure(fig1.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])
        figure(fig2.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])
        figure(fig3.number); plot(np.repeat(node, 2), [0, np.max(fu_CMN)], clrs[i])

# <codecell>

print outputData
print labeledPoints+4
np.save(outputData+"labeledPoints.npy", labeledPoints+4)
np.save(outputData+"labelProbs.npy", labelProbs)

# <codecell>

print np.sum(labelProbs, axis=1).shape

# <codecell>

idxCorrection = 0
sampleData = "ribbon2/"
# sampleData = "pendulum/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame*.png")
maxFrames = len(frames)
frames = np.sort(frames)[0:maxFrames]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames
## load full precomputed distMat for dataset
distanceMatrix = np.load(dataFolder + sampleData + "vanilla_distMat.npy")
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

frameSize = cv2.imread(frames[0]).shape
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=np.uint8)
for i in range(0, numFrames) :
    movie[:, :, :, i] = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB), dtype=np.uint8)
    
    sys.stdout.write('\r' + "Loaded frame " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()

# <codecell>

distMat = vtu.filterDistanceMatrix(distanceMatrix, 4, True)
idxCorrection += 4
figure(); imshow(distMat, interpolation='nearest')

# <codecell>

figure(); imshow(np.reshape(distMat[281-8:281, :], (8, len(distMat))), interpolation='nearest')

# <codecell>

kernel = np.reshape(rangeCurves, (len(rangeCurves), 1))
kernel = kernel*kernel.T
# figure(); imshow(kernel, interpolation='nearest')
print kernel.shape, distMat.shape

# <codecell>

def estimateFutureCost(alpha, p, distanceMatrixFilt, weights) :
    
    distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
    distMat = distMatFilt ** p
    
    last = np.copy(distMat)
    current = np.zeros(distMat.shape)
    
    ## while distance between last and current is larger than threshold
    iterations = 0 
    while np.linalg.norm(last - current) > 0.1 : 
        for i in range(distMat.shape[0]-1, -1, -1) :
#             m = np.min(distMat*weights[1:distanceMatrixFilt.shape[1], 0:-1], axis=1)
            m = np.min(distMat*weights, axis=1)
            distMat[i, :] = (distMatFilt[i, :] ** p) + alpha*m
            
        last = np.copy(current)
        current = np.copy(distMat)
        
        sys.stdout.write('\r' + "Iteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
        sys.stdout.flush()
        
        iterations += 1
    
    print
    print 'finished in', iterations, 'iterations'
    
    return distMat

distMatFut = estimateFutureCost(0.999, 2.0, distMat, np.ones(distMat.shape))#rangeWeights)
figure(); imshow(distMatFut, interpolation='nearest')

# <codecell>

p, c = vtu.getProbabilities(distMatFut, 0.02, None, True)
figure(); imshow(c, interpolation='nearest')

# <codecell>

## compute length of shortest path for each pair of nodes (uses sum of edges weight as distance measure)
shortPathLengths = np.ones(weightMat.shape, dtype=uint)*sys.float_info.max

for i in frameGraph.nodes() :
    shtpa = shortest_path(frameGraph, i)
    shortPathLengths[i, shtpa[1].keys()] = shtpa[1].values()
#     for j in frameGraph.nodes() :
#         if i != j :
#             path, distance = getShortestPath(frameGraph, i, j)
#             if distance > 0 : 
#                 shortPathLengths[i, j] = distance
    sys.stdout.write('\r' + "Computed shortest paths for frame " + np.string_(i))
    sys.stdout.flush()

print
sigma = np.mean(shortPathLengths[np.where(shortPathLengths != sys.float_info.max)])
sigma = np.mean(intervals)/2.0#100.0
print 'sigma', sigma
pathProbMat = np.exp((-shortPathLengths)/sigma)
## normalize columnwise instead of rowwise as done with probability based on distMat
# normTerm = np.sum(pathProbMat, axis=1)
# normTerm = cv2.repeat(normTerm, 1, shortPathLengths.shape[1])
normTerm = np.sum(pathProbMat, axis=0)
normTerm = np.repeat(np.reshape(normTerm, (len(normTerm), 1)).T, len(normTerm), axis=0)
pathProbMat = pathProbMat / normTerm
pathProbMat[np.isnan(pathProbMat)] = 0.0
print np.sum(pathProbMat, axis=0)

# <codecell>

figure(); imshow(pathProbMat, interpolation='nearest')

# <codecell>

close("fig1")
## traverse graph starting from initialNode and randomize jump
print initialNodes
currentNode = initialNodes[3]
finalFrames = []
finalFrames.append(currentNode)
print currentNode
sequenceLength = 1.0
maxSeqLength = 100.0
destFrame = initialNodes[3]
destRange = smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[0]), intervals[0], 0.4)
# figure(); plot(arange(0.0, len(rangeCurves)), destRange)
for i in xrange(0, 1000) :
    print
    print "frame", i, "from", currentNode
    neighs = np.array(frameGraph.node_neighbors[currentNode], dtype=int)
#     print neighs,
#     print neighs
    probs = []
    for n in frameGraph.node_neighbors[currentNode] :
        if len(frameGraph.node_neighbors[n]) > 0:
#             probs.append(frameGraph.edge_weight((currentNode, n)))
            probs.append(probMat[currentNode, n])
        else :
            neighs = np.delete(neighs, np.argwhere(neighs == n))
    probs = np.array(probs)

    ## add the probability based on distance to destination
#     print probs, neighs, destFrame
    
    probs /= np.sum(probs)
    
    pathProbs = pathProbMat[neighs, destFrame]
    pathProbs = pathProbs/np.sum(pathProbs)
    
    if not np.isnan(pathProbs).all() :
        probs += pathProbs
    
    probs *= (1+100*destRange[neighs])
    
    ## increase probability of jumping based on how long the consequent sequence has been
    p = np.exp(-np.power(sequenceLength-maxSeqLength, 2)/(maxSeqLength*4.0))
    probs[np.where(neighs == currentNode + 1)] *= 1-p
    probs[np.where(neighs != currentNode + 1)] *= p
    
#     probs = np.power(probs, 1+destRange[neighs])
    
    probs /= np.sum(probs)
    ## pick a random node to go to next
    tmp = np.random.rand()
    randNode = np.round(np.sum(np.cumsum(probs) < tmp))
    newNode = neighs[randNode]#np.argmax(probs)]]
    
    if newNode == currentNode+1:
        sequenceLength += 1.0
    else :
        print "jump",
        sequenceLength = 1.0
    
    currentNode = newNode
    finalFrames.append(currentNode)
    print currentNode, sequenceLength, probs, pathProbMat[neighs, destFrame], destRange[neighs], neighs, tmp
    
    
figure("fig1"); 
for iN in initialNodes :
    plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*iN, 'g')
plot(finalFrames, 'b', np.repeat(destFrame, len(finalFrames)), "r")

# <codecell>

figure(); plot(arange(0.0, len(rangeCurves)), destRange*100)

# <codecell>

import bisect
def shortestPathFromTo(graph, source, weights):
    # Initialization
    dist     = {source: 0}
    previous = {source: None}

    # This is a sorted queue of (dist, node) 2-tuples. The first item in the
    # queue is always either a finalized node that we can ignore or the node
    # with the smallest estimated distance from the source. Note that we will
    # not remove nodes from this list as they are finalized; we just ignore them
    # when they come up.
    q = [(0, source)]

    # The set of nodes for which we have final distances.
    finished = set()

    # Algorithm loop
    while len(q) > 0:
        du, u = q.pop(0)

        # Process reachable, remaining nodes from u
        if u not in finished:
            finished.add(u)
#             print u, "tra"
            for v in xrange(0, len(graph.nodes())):#graph[u]:
#                 print v, 
                if v not in finished:
                    alt = du + weights[u, v]#graph.edge_weight((u, v))
                    if (v not in dist) or (alt < dist[v]):
                        dist[v] = alt
                        previous[v] = u
                        bisect.insort(q, (alt, v))
#                 print

    return previous, dist

# <codecell>

## try and find random path from src to tgt
src = initialNodes[0]
tgt = initialNodes[1]
curr = src
prev = curr
randPath = [src]
cost = 0.0
while curr != tgt :
    curr = np.random.randint(0, probMat.shape[-1])
    randPath.append(curr)
    cost += (1.0/probMat)[curr, prev]
    prev = curr

print cost    
print len(randPath), randPath

# <codecell>

print frameGraph.neighbors(0)
# print frameGraph.nodes()
p, d = shortestPathFromTo(frameGraph, 0, 1.0/probMat)
print p
print d

# <codecell>

## find shortest paths between initNodes in fully connected graph
fullGraph = digraph()
fullGraph.add_nodes(arange(0, weightMat.shape[-1], dtype=int))

for i in xrange(0, len(fullGraph.nodes())) :
    for j in xrange(0, len(fullGraph.nodes())) :
        fullGraph.add_edge((i, j), wt=(1.0/probMat)[i, j])
    print i, 

print len(fullGraph.nodes())
print len(fullGraph.edges())

# <codecell>

print len(fullGraph.edges())
print fullGraph.edge_weight((0, 5)), probMat[0, 3]

# <codecell>

# figure(); imshow(shortPathLengths, interpolation='nearest')
print probMat[1101, 1102], pathProbMat[1102, initialNodes[3]]
print getShortestPath(frameGraph, initialNodes[3], initialNodes[2])

# <codecell>

# finalFrames = np.ndarray.flatten(np.array(strongly_connected_components(subGr))+idxCorrection)
# finalFrames = np.arange(30, 274)
# finalFrames = np.array(finalFrames)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# <codecell>

def update_plot(i, data, scat, img, ax):
#     plt.clf()
#     plt.subplot(211)
#     plt.imshow(movie[:, :, :, data[i]], interpolation='nearest')
#     plt.subplot(212)
    img.set_data(movie[:, :, :, data[i]])
    ax.clear()
    ax.plot(data)
    for iN in initialNodes :
        ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*iN, 'g')
    ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*destFrame, 'r')
    scat = ax.scatter(i, data[i])
    return scat,

visFrames = np.array(finalFrames) + idxCorrection

x = 0
y = visFrames[0]

fig = plt.figure()
plt.subplot(211)
img = plt.imshow(movie[:, :, :, x], interpolation='nearest')
ax = plt.subplot(212)
ax.plot(finalFrames)
for iN in initialNodes :
    ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*iN, 'g')
ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*destFrame, 'r')
scat = ax.scatter(x, y)

ani = animation.FuncAnimation(fig, update_plot, frames=xrange(len(visFrames)),
                              fargs=(visFrames, scat, img, ax), interval=33)
plt.show()

# <codecell>

## visualize frames automatically

def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw

fig=plt.figure()
img = plt.imshow(np.array(cv2.cvtColor(cv2.imread(frames[finalFrames[0]]), cv2.COLOR_BGR2RGB), dtype=np.uint8))
img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
#     img.set_data(movie[:, :, :, finalFrames[0]])
    img.set_data(np.array(cv2.cvtColor(cv2.imread(frames[finalFrames[0]]), cv2.COLOR_BGR2RGB), dtype=np.uint8))
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f) + ' from ' + str(finalFrames[f]))
#     img.set_data(movie[:, :, :, finalFrames[f]])
    img.set_data(np.array(cv2.cvtColor(cv2.imread(frames[finalFrames[f]]), cv2.COLOR_BGR2RGB), dtype=np.uint8))
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalFrames),interval=33,blit=True)

plt.show()

# <codecell>

## visualize frames automatically
# finalFrames = arange(0, numFrames)

def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw

fig=plt.figure()
img = plt.imshow(movie[:, :, :, 0])
img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
img.set_interpolation('nearest')
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
    img.set_data(movie[:, :, :, finalFrames[0]])
#     img.set_data(movie[:, :, :, 0])
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f) + ' from ' + str(finalFrames[f]))
    img.set_data(movie[:, :, :, finalFrames[f]])
#     img.set_data(movie[:, :, :, f])
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalFrames),interval=33,blit=True)

plt.show()

