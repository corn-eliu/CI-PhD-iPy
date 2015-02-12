# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab 

from multiprocessing import Process, Array, Pool
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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import GraphWithValues as gwv

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

def computeDist(args):
    rowIdxStart = args[0]
    rowIdxEnd = args[1]
    colIdxStart = args[2]
    colIdxEnd = args[3]
    
    ## get row frames
    tac = time.time()
    f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
    for f, idx in zip(xrange(rowIdxStart, rowIdxEnd), xrange(0, blockSize)) :
        img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))
        if os.path.isfile(mattes[f]) :
            alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY))/255.0
            f1s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
        else :
            f1s[:, :, :, idx] = img/255.0

    data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
    
    if rowIdxStart == colIdxStart and rowIdxEnd == colIdxEnd :
        ##compute distance between every pair of row frames
        dist = distEuc(data1)
        print "Row Frames", rowIdxStart, "to", rowIdxEnd, "in", time.time()-tac, "secs"
        sys.stdout.flush()
    else :
        ##load column frames
        f2s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
        for f, idx in zip(xrange(colIdxStart, colIdxEnd), xrange(0, blockSize)) :
            img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))
            if os.path.isfile(mattes[f]) :
                alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY))/255.0
                f2s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
            else :
                f2s[:, :, :, idx] = img/255.0
                
        data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
        
        ##compute distance between every pair of row-column frames
        dist = distEuc2(data1, data2)
        
        print "Column Frames", colIdxStart, "to", colIdxEnd, "in", time.time()-tac, "secs"
        sys.stdout.flush()
        
    
    return dist, [rowIdxStart, rowIdxEnd, colIdxStart, colIdxEnd]

def computeDist2(args):
    rowIdxStart = args[0]
    rowIdxEnd = args[1]
    colIdxStart = args[2]
    colIdxEnd = args[3]
    
    ## get row frames
    tac = time.time()
    f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
    for f, idx in zip(xrange(rowIdxStart, rowIdxEnd), xrange(0, blockSize)) :
        img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB), dtype=np.float16)
        if os.path.isfile(mattes[f]) :
            alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY), dtype=np.float16)/255.0
            f1s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
        else :
            f1s[:, :, :, idx] = img/255.0

    data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
    ##compute distance between every pair of row frames
    dist = distEuc(data1)
    print "Row Frames", rowIdxStart, "to", rowIdxEnd, "in", time.time()-tac, "secs"
    sys.stdout.flush()
    
    for j in xrange(rowIdxEnd/blockSize, numBlocks) :
        tac = time.time()
        ##load column frames
        f2s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
        for f, idx in zip(xrange(j*blockSize, j*blockSize+blockSize), xrange(0, blockSize)) :
            img = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB), dtype=np.float16)
            if os.path.isfile(mattes[f]) :
                alpha = np.array(cv2.cvtColor(cv2.imread(mattes[f]), cv2.COLOR_BGR2GRAY), dtype=np.float16)/255.0
                f2s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
            else :
                f2s[:, :, :, idx] = img/255.0
                
        data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
        
        ##compute distance between every pair of row-column frames
        dist = np.concatenate((dist, distEuc2(data1, data2)), axis=-1)
        
        print "Column Frames", j*blockSize, "to", j*blockSize+blockSize, "in", time.time()-tac, "secs"
        sys.stdout.flush()
        del f2s
        gc.collect()
        
    
    return dist, [rowIdxStart, rowIdxEnd, colIdxStart, colIdxEnd]

# <codecell>

mpParams = list()

for i in xrange(0, numBlocks) :
    mpParams.append([i*blockSize, i*blockSize+blockSize, i*blockSize, i*blockSize+blockSize])
    for j in xrange(i+1, numBlocks) :
        mpParams.append([i*blockSize, i*blockSize+blockSize, j*blockSize, j*blockSize+blockSize])
        
print np.array(mpParams)

# <codecell>

for j in xrange(64/blockSize, numBlocks) :
    print j, j*blockSize, j*blockSize+blockSize

# <codecell>

if __name__ == '__main__':
    distMat = np.zeros([numFrames, numFrames])
    tic = time.time()
    p = Pool(processes=2)
    
    numBlocks = 8
    blockSize = numFrames/numBlocks
    print numFrames, numBlocks, blockSize
    sys.stdout.flush()
    mpParams = list()
    
    for i in xrange(0, numBlocks) :
        mpParams.append([i*blockSize, i*blockSize+blockSize, i*blockSize, numFrames])
#         for j in xrange(i+1, numBlocks) :
#             mpParams.append([i*blockSize, i*blockSize+blockSize, j*blockSize, j*blockSize+blockSize])
            
    distsAndIdxs = p.map(computeDist2, mpParams)
    
    ## unpack distances from processes return values and update distMat
    for dist, idx in distsAndIdxs :
        distMat[idx[0]:idx[1], idx[2]:idx[3]] = dist
        
    print
    print "total time", time.time()-tic
    
    print distMat.shape
#     distMat[0:16, 16:32] = 10
    figure(); imshow(distMat, interpolation='nearest')

# <codecell>

print len(distsAndIdxs[0][0])

# <codecell>

fullDistMat = np.copy(distMat)

