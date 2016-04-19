# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab 
import numpy as np
import scipy as sp
import scipy.io as sio
import cv2
import cv
import glob
import time
import sys
import os
from scipy import ndimage
from scipy import stats

from _emd import emd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import GraphWithValues as gwv
import VideoTexturesUtils as vtu

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

## divide data into subblocks and compute cosine based distance
st = time.time()

blocksPerDim = 16
subDivisions = blocksPerDim**2

## given block sizes and img sizes build indices representing each block
imageSize = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)).shape
stencils = []
bRows = imageSize[0]/blocksPerDim
bCols = imageSize[1]/blocksPerDim
for r in xrange(0, blocksPerDim) :
    for c in xrange(0, blocksPerDim) :
        idx = c + r*blocksPerDim
        ## this is for stencilling the images which are 3D
#         stencil = np.zeros(imageSize, dtype=int)
#         stencil[r*bRows:r*bRows+bRows, c*bCols:c*bCols+bCols] = np.ones((bRows, bCols, imageSize[-1]))
#         stencils.append(list(np.argwhere(stencil==1).T))
        ## this is for stencilling the mattes which are 2D
        stencil = np.zeros(imageSize[0:-1], dtype=int)
        stencil[r*bRows:r*bRows+bRows, c*bCols:c*bCols+bCols] = np.ones((bRows, bCols))
        stencils.append(list(np.argwhere(stencil==1).T))
    
features = np.zeros([numFrames, subDivisions])
for i in xrange(0, numFrames) :
    
    t = time.time()
    
    ##load frame
    img = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB))/255.0
    alpha = np.zeros(img.shape[0:-1])
    if os.path.isfile(mattes[i]) :
        alpha = np.array(cv2.cvtColor(cv2.imread(mattes[i]), cv2.COLOR_BGR2GRAY))/255.0
        img *= np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)

    ## use stencils to divide the image into blocks and count number of foreground pixels
    for s in xrange(0, len(stencils)) :
        index = s + idx*len(stencils)
        features[i, s] = len(np.argwhere(alpha[stencils[s]] != 0))
    sys.stdout.write('\r' + "Computed histogram of frame " + np.string_(i) + " of " + np.string_(numFrames) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()
    
print
print "finished in", time.time() - st

## normalize
features /= np.repeat(np.reshape(np.linalg.norm(features, axis=-1), (numFrames, 1)), subDivisions, axis=-1)
figure(); imshow(features.T, interpolation='nearest')

# <codecell>

distMat = sio.loadmat("hist2demd_distMat.mat")["distMat"]
figure(); imshow(distMat, interpolation='nearest')

# <codecell>

import scipy.io as sio
sio.savemat("features.mat", {"features":np.array(features*255, dtype=np.int32)})

# <codecell>

gwv.showCustomGraph(features[717, :].reshape((blocksPerDim, blocksPerDim)))
gwv.showCustomGraph(features[1165, :].reshape((blocksPerDim, blocksPerDim)))
gwv.showCustomGraph(features[1166, :].reshape((blocksPerDim, blocksPerDim)))
print np.dot(features[717, :], features[1166, :])
print np.dot(features[1165, :], features[1166, :])

# <codecell>

def distance1D(f1, f2):
#     return np.sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    return np.sqrt((f1 - f2)**2)#np.sqrt( (f1[0] - f2[0])**2  + (f1[1] - f2[1])**2 + (f1[2] - f2[2])**2 )

def distance2D(f1, f2):
#     return np.sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    return np.sqrt( (f1[0] - f2[0])**2  + (f1[1] - f2[1])**2 )# + (f1[2] - f2[2])**2 )

print "1D"
print emd((list(features[717, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance1D)
print emd((list(features[1165, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance1D)
print "2D"
print emd((list(features[717, :].reshape((blocksPerDim, blocksPerDim))), list(arange(0.0, blocksPerDim))), (list(features[1166, :].reshape((blocksPerDim, blocksPerDim))), list(arange(0.0, blocksPerDim))), distance2D)
print emd((list(features[1165, :].reshape((blocksPerDim, blocksPerDim))), list(arange(0.0, blocksPerDim))), (list(features[1166, :].reshape((blocksPerDim, blocksPerDim))), list(arange(0.0, blocksPerDim))), distance2D)
print "1D with 2D weights"
print emd((list(features[717, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(np.repeat(arange(0.0, blocksPerDim), blocksPerDim))), distance1D)
print emd((list(features[1165, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(np.repeat(arange(0.0, blocksPerDim), blocksPerDim))), distance1D)
# print "cv"
# frame1 = cv.CreateMat(blocksPerDim, blocksPerDim, cv.CV_32FC1)
# cv.Convert(cv.fromarray(np.ascontiguousarray(features[717, :].reshape((blocksPerDim, blocksPerDim)))), frame1)
# frame2 = cv.CreateMat(blocksPerDim, blocksPerDim, cv.CV_32FC1)
# cv.Convert(cv.fromarray(np.ascontiguousarray(features[1166, :].reshape((blocksPerDim, blocksPerDim)))), frame2)
# print frame1, frame2
# print cv.CalcEMD2(frame1, frame2, cv.CV_DIST_L2)
# cv.Convert(cv.fromarray(np.ascontiguousarray(features[1165, :].reshape((blocksPerDim, blocksPerDim)))), frame1)
# print cv.CalcEMD2(frame1, frame2, cv.CV_DIST_L2)

# <codecell>

# import Image
# im = Image.fromarray(np.array(features[717, :].reshape((blocksPerDim, blocksPerDim))*255, dtype=uint8))
# im.save("im717.png")
np.array(features[717, :].reshape((blocksPerDim, blocksPerDim))*255, dtype=uint8)
print np.max(features)

# <codecell>

## compute the distance matrix where distance is dot product between feature vectors
distanceMatrix = np.ones((numFrames, numFrames))# np.zeros((numFrames, numFrames))

def distance(f1, f2):
#     return np.sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    return np.sqrt((f1 - f2)**2)#np.sqrt( (f1[0] - f2[0])**2  + (f1[1] - f2[1])**2 + (f1[2] - f2[2])**2 )

for r in xrange(0, numFrames) :
    for c in xrange(r+1, numFrames) :
#         distanceMatrix[r, c] = distanceMatrix[c, r] = emd((list(features[r, :]), list(arange(0.0, subDivisions))), (list(features[c, :]), list(arange(0.0, subDivisions))), distance)
        distanceMatrix[r, c] = distanceMatrix[c, r] = np.dot(features[r, :], features[c, :])
    print r, 

distanceMatrix = 1 - distanceMatrix
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

# np.save(outputData + "histcos_distMat.npy", distanceMatrix)
distanceMatrix = np.load(outputData + "vanilla_distMat.npy")
# distanceMatrix = np.array(distanceMatrix, dtype=float)/np.max(distanceMatrix)
distMat = vtu.filterDistanceMatrix(distanceMatrix, 4, True)
figure(); imshow(distMat, interpolation='nearest')
# sio.savemat("hist2demd_distMat.mat", {"distMat":distMat})

# <codecell>

def distance(f1, f2):
#     return np.sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    return np.sqrt((f1 - f2)**2)#np.sqrt( (f1[0] - f2[0])**2  + (f1[1] - f2[1])**2 + (f1[2] - f2[2])**2 )

print emd((list(features[1165, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance)
print emd((list(features[717, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance)

# <codecell>

print list(features[1165, :])

# <codecell>

## do label propagation as zhu 2003
# distances = np.copy(distanceMatrix)
distances = np.array(np.copy(distMat), dtype=float)#/np.max(distMat)
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
    extraPoints = 16
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
# gwv.showCustomGraph(cumW)

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

## check that orderedDist is still symmetric
print np.sum(orderedDist[4, :] - orderedDist[:, 4])
print np.sum(orderedDist[10, :] - orderedDist[:, 10])
print np.sum(orderedDist[250, :] - orderedDist[:, 250])
## check that orderedDist has been ordered the right way
print orderedDist[0, 0:50]
print distances[117, list(flatLabeled)]
print distances[117, 0:50]
print 
print orderedDist[3, 0:50]
print distances[496, list(flatLabeled)]
print distances[496, 0:50]
print 
print orderedDist[10, 0:50]
print distances[1102, list(flatLabeled)]
print distances[1102, 0:50]

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

clrs = ['r', 'g', 'b', 'm']
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

sio.savemat("labeledPoints.mat", {"labeledPoints":labeledPoints})

# <codecell>

sio.savemat("hist2demd_mult0.02_labelProbs.mat", {"labelProbs":labelProbs})

