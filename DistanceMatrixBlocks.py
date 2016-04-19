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
import gc
from scipy import ndimage
from scipy import stats

from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import GraphWithValues as gwv

# dataFolder = "/home/ilisescu/PhD/data/"
dataFolder = "/media/ilisescu/Data1/PhD/data/"

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
# sampleData = "ribbon2/"
# sampleData = "ribbon1_matted/"
# sampleData = "little_palm1_cropped/"
# sampleData = "ballAnimation/"
sampleData = "candle1/segmentedAndCropped/"
sampleData = "candle2/subset_stabilized/segmentedAndCropped/"
sampleData = "candle3/stabilized/segmentedAndCropped/"
sampleData = "wave2/"
sampleData = "toy/"

outputData = dataFolder+sampleData

## Find pngs in sample data
frames = np.sort(glob.glob(dataFolder + sampleData + "frame-*.png"))
mattes = np.sort(glob.glob(dataFolder + sampleData + "matte-*.png"))
sprite = "aron2"
segmented = np.sort(glob.glob(dataFolder + sampleData + sprite+"-maskedFlow/frame-*.png"))
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes), len(segmented)

# <codecell>

featureMatrix = np.load(outputData + "gridFeatures.npy")
print featureMatrix.shape
# gwv.showCustomGraph(np.sum(features, axis=-1))

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

## divide data into subblocks
s = time.time()
numBlocks = 2
blockSize = numFrames/numBlocks
print numFrames, numBlocks, blockSize
distanceMatrix = np.zeros([numFrames, numFrames])

for i in xrange(0, numBlocks) :
    
    t = time.time()
    
    ##load row frames
    f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
    for f, idx in zip(xrange(i*blockSize, i*blockSize+blockSize), xrange(0, blockSize)) :
        img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB), dtype=np.float64)
        if f < len(segmented) and os.path.isfile(segmented[f]) :
            alpha = np.array(Image.open(segmented[f]), dtype=np.float64)[:, :, -1]/255.0
            f1s[:, :, :, idx] = (img/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))
        else :
            if f < len(mattes) and os.path.isfile(mattes[f]) :
                alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY), dtype=np.float64)/255.0
                f1s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
            else :
                f1s[:, :, :, idx] = img/255.0

    ##compute distance between every pair of row frames
    data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
    print data1.shape
    distanceMatrix[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = distEuc(data1)
    
    sys.stdout.write('\r' + "Row Frames " + np.string_(i*blockSize) + " to " + np.string_(i*blockSize+blockSize-1) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()
    print
    
    for j in xrange(i+1, numBlocks) :
        
        t = time.time()
        
        ##load column frames
        f2s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
        for f, idx in zip(xrange(j*blockSize, j*blockSize+blockSize), xrange(0, blockSize)) :
            img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB), dtype=np.float64)
            if f < len(segmented) and os.path.isfile(segmented[f]) :
                alpha = np.array(Image.open(segmented[f]), dtype=np.float64)[:, :, -1]/255.0
                f2s[:, :, :, idx] = (img/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))
            else :
                if f < len(mattes) and os.path.isfile(mattes[f]) :
                    alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY), dtype=np.float64)/255.0
                    f2s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
                else :
                    f2s[:, :, :, idx] = img/255.0
    #             f2s[:, :, :, idx] = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0
            
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

np.save(dataFolder+sampleData+"vanilla_distMat.npy", distanceMatrix)

# <codecell>

np.save(dataFolder+sampleData+"toy1-vanilla_distMat.npy", distanceMatrix)

# <codecell>

## divide data into subblocks and compute cosine based distance
st = time.time()
numBlocks = 8
blockSize = numFrames/numBlocks
print numFrames, numBlocks, blockSize
distanceMatrix = np.zeros([numFrames, numFrames])

blocksPerDim = 16
subDivisions = blocksPerDim**2
featureMatrix = np.zeros([numFrames, numFrames, subDivisions])

## given block sizes and img sizes build indices representing each block
imageSize = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)).shape
stencils = []
bRows = imageSize[0]/blocksPerDim
bCols = imageSize[1]/blocksPerDim
for r in xrange(0, blocksPerDim) :
    for c in xrange(0, blocksPerDim) :
        idx = c + r*blocksPerDim
        stencil = np.zeros(imageSize, dtype=int)
        stencil[r*bRows:r*bRows+bRows, c*bCols:c*bCols+bCols] = np.ones((bRows, bCols, imageSize[-1]))
        stencils.append(list(np.argwhere(stencil==1).T))
    
for i in xrange(0, numBlocks) :
    
    t = time.time()
    
    ##load row frames
#     f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
    data1 = np.zeros((blockSize, bRows*bCols*frameSize[-1], subDivisions))
    for f, idx in zip(xrange(i*blockSize, i*blockSize+blockSize), xrange(0, blockSize)) :
        img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0
        if os.path.isfile(mattes[f]) :
            alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY))/255.0
            img *= np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
#         else :
#             img /= 255.0
            
#         f1s[:, :, :, idx] = img
    
        ## use stencils to divide the image into blocks
        for s in xrange(0, len(stencils)) :
            index = s + idx*len(stencils)
            data1[idx, :, s] = img[stencils[s]]
    
    ## get l2 distance between sublocks to build the feature vectors for each image
    features = np.zeros((blockSize, blockSize, subDivisions))
    for s in xrange(0, subDivisions) :
        features[:, :, s] = distEuc(data1[:, :, s])
    
    ## normalize
    norms = np.repeat(np.reshape(np.linalg.norm(features, axis=-1), (blockSize, blockSize, 1)), subDivisions, axis=-1)
    norms += np.repeat(np.reshape(np.eye(features.shape[0]), (blockSize, blockSize, 1)), subDivisions, axis=-1)
    features /= norms
    
    distanceMatrix[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = np.exp(-np.sum(features, axis=-1)) * (1-np.eye(features.shape[0]))
    featureMatrix[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize, :] = np.copy(features)
    
    sys.stdout.write('\r' + "Row Frames " + np.string_(i*blockSize) + " to " + np.string_(i*blockSize+blockSize-1) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()
    print
    
    for j in xrange(i+1, numBlocks) :
        
        t = time.time()
        
        ##load column frames
#         f2s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
        data2 = np.zeros((blockSize, bRows*bCols*frameSize[-1], subDivisions))
        for f, idx in zip(xrange(j*blockSize, j*blockSize+blockSize), xrange(0, blockSize)) :
            img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0
            if os.path.isfile(mattes[f]) :
                alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY))/255.0
                img *= np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
#             else :
#                 img = img/255.0
            
#             f2s[:, :, :, idx] = img
            
            ## use stencils to divide the image into blocks
            for s in xrange(0, len(stencils)) :
                index = s + idx*len(stencils)
                data2[idx, :, s] = img[stencils[s]]
        
        ## get l2 distance between sublocks to build the feature vectors for each image
        features = np.zeros((blockSize, blockSize, subDivisions))
        for s in xrange(0, subDivisions) :
            features[:, :, s] = distEuc2(data1[:, :, s], data2[:, :, s])
        
        ## normalize
        norms = np.repeat(np.reshape(np.linalg.norm(features, axis=-1), (blockSize, blockSize, 1)), subDivisions, axis=-1)
#         norms += np.repeat(np.reshape(np.eye(features.shape[0]), (blockSize, blockSize, 1)), subDivisions, axis=-1)
        features /= norms
        
        distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize] = np.exp(-np.sum(features, axis=-1))# * (1-np.eye(features.shape[0]))
        distanceMatrix[j*blockSize:j*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize].T
        featureMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize, :] = np.copy(features)
        featureMatrix[j*blockSize:j*blockSize+blockSize, i*blockSize:i*blockSize+blockSize, :] = np.transpose(featureMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize, :], axes=(1,0,2))
    
        sys.stdout.write('\r' + "Column Frames " + np.string_(j*blockSize) + " to " + np.string_(j*blockSize+blockSize-1) + " in " + np.string_(time.time() - t))
        sys.stdout.flush()
        print

figure(); imshow(distanceMatrix, interpolation='nearest')
print
print "finished in", time.time() - st

# <codecell>

featureMatrix = np.load(outputData + "gridFeatures.npy")

# <codecell>

#gwv.showCustomGraph(distanceMatrix)#featureMatrix[:, :, 9])
np.save(outputData + "gridFeatures.npy", featureMatrix)

# <codecell>

blocksPerDim = 16
gwv.showCustomGraph(featureMatrix[1165, 1166, :].reshape((blocksPerDim, blocksPerDim)))
gwv.showCustomGraph(featureMatrix[717, 1166, :].reshape((blocksPerDim, blocksPerDim)))
# print distanceMatrix[1165, 1166], distanceMatrix[717, 1166]
print np.sum(featureMatrix[1165, 1166, :]), np.sum(featureMatrix[717, 1166, :])
print np.exp(-np.sum(featureMatrix[1165, 1166, :])), np.exp(-np.sum(featureMatrix[717, 1166, :]))
gwv.showCustomGraph(np.sum(featureMatrix, axis=-1))

# <codecell>

print np.sum(featureMatrix[0, 7, :]), np.sum(featureMatrix[0, 12, :]), np.exp(-np.sum(featureMatrix[0, 7, :])), np.exp(-np.sum(featureMatrix[0, 12, :]))

# <codecell>

distanceMatrix16subs = np.copy(distanceMatrix)

# <codecell>

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
distMat *= 1-np.eye(distMat.shape[0])

# <codecell>

print stencils[0]

# <codecell>

f = 0
img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))
if os.path.isfile(mattes[f]) :
    alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY))/255.0
    alphaed = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
else :
    alphaed = img/255.0
figure(); imshow(alphaed)
figure(); imshow(alpha)

# <codecell>

print 1280/8

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

