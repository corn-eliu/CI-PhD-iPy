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

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

## read frames from sequence of images
sampleData = "ballAnimation/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "*.png")
frames = np.sort(frames)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=uint8)
for f, idx in zip(frames, xrange(0, numFrames)) :
    movie[:, :, :, idx] = np.array(cv2.imread(f))

# <codecell>

## compute euclidean distance assuming f is an array where each row is a flattened image (1xN array, N=W*H*Channels)
## euclidean distance defined as the length of the the displacement vector:
## len(q-p) = sqrt(len(q)^2+len(p)^2 - 2*dot(p, q)) where p and q are two images in vector format and 1xN size
def distEuc(f) :
    ## gives sum over squared intensity values for each image
    ff = np.sum(f*f, axis=1)
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
    d = np.sqrt(np.reshape(ff, [len(ff),1])+ff.T - 2*np.dot(f, f.T))
    return d

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

# <codecell>

tmp = distFlow(movie[...,0], movie[...,4])
figure(); imshow(tmp)

# <codecell>

def distFlow(f1, f2) :
    hsv = np.zeros_like(f1)
    hsv[...,1] = 255
    
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)/255.0, cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)/255.0, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    
    return rgb

# <codecell>

## divide data into subblocks
s = time.time()
numBlocks = 1
blockSize = numFrames/numBlocks
print numFrames, numBlocks, blockSize
distanceMatrix = np.zeros([numFrames, numFrames])

for i in xrange(0, numBlocks) :
    
    t = time.time()
    
    ##load row frames
    f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
    for f, idx in zip(xrange(i*blockSize, i*blockSize+blockSize), xrange(0, blockSize)) :
        f1s[:, :, :, idx] = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0

    ##compute distance between every pair of row frames
    data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
    distanceMatrix[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = distEuc(data1)
    
    sys.stdout.write('\r' + "Row Frames " + np.string_(i*blockSize) + " to " + np.string_(i*blockSize+blockSize-1) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()
    print
    
    for j in xrange(i+1, numBlocks) :
        
        t = time.time()
        
        ##load column frames
        f2s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
        for f, idx in zip(xrange(j*blockSize, j*blockSize+blockSize), xrange(0, blockSize)) :
            f2s[:, :, :, idx] = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0
            
        ##compute distance between every pair of row-column frames
        data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
        distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize] = distEuc2(data1, data2)
        distanceMatrix[j*blockSize:j*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize].T
    
        sys.stdout.write('\r' + "Column Frames " + np.string_(j*blockSize) + " to " + np.string_(j*blockSize+blockSize-1) + " in " + np.string_(time.time() - t))
        sys.stdout.flush()
        print

figure(); imshow(distanceMatrix, interpolation='nearest')
print
print "finished in", time.time() - s

# <codecell>

print outputData

# <codecell>

np.save(outputData + "distMat", distanceMatrix)

# <codecell>

s = time.time()
if numFrames > 0 :
    frameSize = cv2.imread(frames[0]).shape
    movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]))
    for i in range(0, numFrames) :
        im = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB))#/255.0
        movie[:, :, :, i] = im#np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
        
print 'Movie has shape', movie.shape
distanceMatrix1 = np.zeros([movie.shape[3], movie.shape[3]])
data = np.reshape(movie/255.0, [np.prod(movie.shape[0:-1]), movie.shape[-1]]).T
distanceMatrix1 = distEuc2(data, data)
figure(); imshow(distanceMatrix1, interpolation='nearest')
print
print "finished in", time.time() - s

# <codecell>

print distanceMatrix1[0:blockSize, blockSize:blockSize+blockSize]

# <codecell>

print distanceMatrix[0:blockSize, blockSize:blockSize+blockSize]

# <codecell>

# numFrames = 3
distanceMatrix = np.zeros([numFrames, numFrames])
s = time.time()
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
figure(); imshow(distanceMatrix, interpolation='nearest')
print
print "finished in", time.time() - s

# <codecell>

print distanceMatrix

# <codecell>

print distanceMatrix

# <codecell>

print distanceMatrix

