# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import sys
import scipy as sp
from scipy import optimize

import time
import os

from sklearn import ensemble
import cv2

import scipy.io as sio
import glob
import commands

from PIL import Image

import GraphWithValues as gwv
import VideoTexturesUtils as vtu

import ComputeGridFeatures as cgf

# <codecell>

## trying to replicate Neill's example
N = 9 ## 3x3 images
phi_true = sio.loadmat("../MATLAB/phi_true.mat")['phi_true']

## plot gamma prior
alpha = 20
beta = alpha-1

def gammaDistrib(x, a, b) :
    return (b**a)*np.exp(-b*x)*(x**(a-1))/sp.special.gamma(a)

p = np.arange(0, 10+10.0/200, 10.0/200)

figure(); plot(p, gammaDistrib(p, alpha, beta))
gca().set_autoscale_on(False)

nn, bin_edges = np.histogram(phi_true, bins=20)
## pp are bin centers
pp = bin_edges[:-1]+(bin_edges[1]-bin_edges[0])/2.0
scatter(pp[nn>0], nn[nn>0]*np.mean(np.diff(pp)), c='r', marker='o')

## training data
I = 50 ## num images

X_train = sio.loadmat("../MATLAB/X_train.mat")['X_train']
sigma_true = 0.25

w_true = sio.loadmat("../MATLAB/w_true.mat")['w_true']
w_train = sio.loadmat("../MATLAB/w_train.mat")['w_train']


X_test = sio.loadmat("../MATLAB/X_test.mat")['X_test']
w_test = np.dot(X_test.T, phi_true)


## testing
sigma_sq = sigma_true**2

errorsML = np.zeros((I, 1))
errorsMAP = np.zeros((I, 1))


## functions to optimize
def logL_ML(phi, X, w, N) :
    phi = np.reshape(phi, (len(phi), 1))
    result = -0.5 * (1.0/sigma_sq) * np.dot((w - np.dot(X.T, phi)).T, (w - np.dot(X.T, phi)))
    return float(result)

def logL_MAP(phi, X, w, N) :
    phi = np.reshape(phi, (len(phi), 1))
    result = (-0.5 * (1.0/sigma_sq) * np.dot((w - np.dot(X.T, phi)).T, (w - np.dot(X.T, phi))) - 
              np.dot(beta*np.ones((N, 1)).T, phi) + np.dot((alpha-1)*np.ones((N, 1)).T, np.log(phi)))
    return float(result)

def negLogL_ML(phi, X, w, N) :
    phi = np.reshape(phi, (len(phi), 1))
    result = - logL_ML(phi, X, w, N)
    return float(result)#.reshape((1,))

fCallCount = 0
def negLogL_MAP(phi, X, w, N) :
    
    global fCallCount
    fCallCount += 1
    if np.mod(fCallCount, 100) == 0 :
        sys.stdout.write('\r' + "function call count " + np.string_(fCallCount)); sys.stdout.flush()
    
    phi = np.reshape(phi, (len(phi), 1))
    result = - logL_MAP(phi, X, w, N)
    return float(result)#.reshape((1,))

def derNegLogL_ML(phi, X, w, N) :
    phi = np.reshape(phi, (len(phi), 1))
    result = -(1/sigma_sq)*np.dot(X, w-np.dot(X.T, phi))
    return np.ndarray.flatten(result)

derCallCount = 0
def derNegLogL_MAP(phi, X, w, N) :
    
    global derCallCount
    derCallCount += 1
    if np.mod(derCallCount, 100) == 0 :
        sys.stdout.write('\r' + "derivative call count " + np.string_(derCallCount)); sys.stdout.flush()
        
    phi = np.reshape(phi, (len(phi), 1))
    result = (-(1/sigma_sq)*np.dot(X, w-np.dot(X.T, phi)) + beta*np.ones((N, 1)) - 
              ((alpha-1)*np.ones((N, 1)))*(1.0/phi))
    return np.ndarray.flatten(result)

# <codecell>

for i in arange(I) :
    
#     X = X_train[:, :i+1]
#     w = w_train[:i+1, :]
    
    phi0 = np.ones((N, 1))
    
    phi_ML = optimize.fmin_ncg(negLogL_ML, phi0, fprime=derNegLogL_ML, 
                               args=(X_train[:, :i+1], w_train[:i+1, :], N)).reshape(phi0.shape)
    phi_MAP = optimize.fmin_ncg(negLogL_MAP, phi0, fprime=derNegLogL_MAP, 
                                args=(X_train[:, :i+1], w_train[:i+1, :], N)).reshape(phi0.shape)
    
    errorsML[i] = np.linalg.norm(w_test- np.dot(X_test.T, phi_ML))
    errorsMAP[i] = np.linalg.norm(w_test- np.dot(X_test.T, phi_MAP))
    
    print "done", i
    sys.stdout.flush()
    
figure(); plot(errorsML); plot(errorsMAP)

# <codecell>

## example optimization
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
xopt = optimize.fmin_ncg(rosen, x0, fprime=rosen_der)

# <codecell>

# dataPath = "/home/ilisescu/PhD/data/"
dataPath = "/media/ilisescu/Data1/PhD/data/"

# dataSet = "pendulum/"
# dataSet = "tree/"
# dataSet = "splashes_water/"
# dataSet = "small_waterfall/"
# dataSet = "sisi_flag/"
dataSet = "eu_flag_ph_left/"
# dataSet = "ribbon2/"
# dataSet = "candle1/segmentedAndCropped/"
framePaths = np.sort(glob.glob(dataPath + dataSet + "frame*.png"))
numFrames = len(framePaths)
print numFrames
imageSize = np.array(Image.open(framePaths[0])).shape[0:2]

# <codecell>

im = cv2.imread(framePaths[0])
figure(); imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

# <codecell>

### compute l2 dist for images of the splashes_water dataset
resizeRatio = 1.0#0.4#0.5#0.75
doRGB = False
useRange = False
featsRange = np.array([-2, -1, 0, 1, 2])
# featsRange = np.array([-1, 0, 1])
rangeResizeRatios = resizeRatio/2**np.abs(arange(-np.floor(len(featsRange)/2), np.floor(len(featsRange)/2)+1))
baseDimensionality = int(np.prod(np.round(np.array(imageSize)*resizeRatio)))
if doRGB :
    if useRange :
        imagesRGBData = np.zeros((np.sum(baseDimensionality/((resizeRatio/rangeResizeRatios)**2))*3, numFrames), dtype=np.float32)
    else :
        imagesRGBData = np.zeros((baseDimensionality*3, numFrames), dtype=np.float32)
else :
    if useRange :
        imagesGrayData = np.zeros((np.sum(baseDimensionality/((resizeRatio/rangeResizeRatios)**2)), numFrames), dtype=np.float32)
    else :
        imagesGrayData = np.zeros((baseDimensionality, numFrames), dtype=np.float32)

for i in xrange(numFrames) :
    if doRGB :
        if useRange :
            for delta, ratio in zip(featsRange, rangeResizeRatios) :
                print delta+i, ratio, 
            print 
        else :
            imagesRGBData[:, i] = np.ndarray.flatten(np.array(cv2.cvtColor(cv2.resize(cv2.imread(framePaths[i]), 
                                                                                      (0, 0), fx=resizeRatio, fy=resizeRatio, 
                                                                                      interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    else :
        if useRange :
            feats = np.empty(0)
            for delta, ratio in zip(featsRange, rangeResizeRatios) :
                if delta+i >= 0 and delta+i < numFrames :
                    feats = np.concatenate((feats, 
                                            np.ndarray.flatten(np.array(cv2.cvtColor(cv2.resize(cv2.imread(framePaths[delta+i]), 
                                                                                                (0, 0), fx=ratio, fy=ratio, 
                                                                                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)))))
                else :
                    feats = np.concatenate((feats, np.zeros(baseDimensionality/((resizeRatio/ratio)**2))))
            imagesGrayData[:, i] = feats
        else :
            imagesGrayData[:, i] = np.ndarray.flatten(np.array(cv2.cvtColor(cv2.resize(cv2.imread(framePaths[i]), 
                                                                                       (0, 0), fx=resizeRatio, fy=resizeRatio, 
                                                                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)))
        
    sys.stdout.write('\r' + "Loaded image " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()


if doRGB :
    imagesRGBData /= np.float32(255)
    print 
    print imagesRGBData.shape
else :
    imagesGrayData /= np.float32(255)
    print
    print imagesGrayData.shape

# <codecell>

## visualize features from the image data when using ranges to make sure they've been assembled well
# imageFeatsSize = np.array(baseDimensionality/((resizeRatio/rangeResizeRatios)**2), dtype=int)
# print imageFeatsSize
# resizedImageSize = cv2.resize(cv2.imread(framePaths[0]), (0, 0), fx=rangeResizeRatios[1], fy=rangeResizeRatios[1], interpolation=cv2.INTER_AREA).shape[0:2]
# figure(); imshow(imagesGrayData[imageFeatsSize[0]:np.sum(imageFeatsSize[0:2]), 100].reshape(resizedImageSize))
# resizedImageSize = cv2.resize(cv2.imread(framePaths[0]), (0, 0), fx=rangeResizeRatios[0], fy=rangeResizeRatios[0], interpolation=cv2.INTER_AREA).shape[0:2]
# figure(); imshow(imagesGrayData[:imageFeatsSize[0], 100].reshape(resizedImageSize))
# figure(); imshow(imagesGrayData[np.sum(imageFeatsSize[0:2]):, 100].reshape(resizedImageSize))
figure(); imshow(imagesGrayData[:, 100].reshape((360, 640)))

# <codecell>

newL2Dist = np.zeros((numFrames, numFrames))
if doRGB :
    print "using RGB"
else :
    print "using Gray"   
for i in xrange(numFrames) :
#     for j in xrange(i, numFrames) :
#         newL2Dist[i, j] = np.sum(np.sqrt((imagesRGBData[:, i]-imagesRGBData[:, j])**2))
#         newL2Dist[j, i] = newL2Dist[i, j]
    
    if doRGB :
        newL2Dist[i, i:] = np.sqrt(np.sum((imagesRGBData[:, i].reshape((np.prod(np.round(np.array(imageSize)*resizeRatio))*3, 1))-imagesRGBData[:, i:])**2, axis=0))
    else :
        newL2Dist[i, i:] = np.sqrt(np.sum((imagesGrayData[:, i].reshape((np.prod(np.round(np.array(imageSize)*resizeRatio)), 1))-imagesGrayData[:, i:])**2, axis=0))
        
    newL2Dist[i:, i] = newL2Dist[i, i:]
        
    sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()
    
gwv.showCustomGraph(newL2Dist)

# <codecell>

# hogFeats = sio.loadmat(dataPath + dataSet + "allFramesHogs.mat")["hogFeats"]
# hogFeats = sio.loadmat(dataPath + dataSet + "allFramesHogs_NoEncoding.mat")["hogFeats"]
# print hogFeats.shape

## get feats of subsequent frames
goodPairsIdxs = np.array([np.arange(numFrames-1, dtype=int), np.arange(1, numFrames, dtype=int)])
print goodPairsIdxs

useValidatedJumps = True

if useValidatedJumps and os.path.isfile(dataPath+dataSet+"validatedJumps.npy") :
    ## validatedJumps has indices of good jumps which means that it contains indices of distances between i and j+1
    ## so need to take (j+1)-1 to get indices of pairs whos distance has been labelled
    validatedJumps = np.load(dataPath+dataSet+"validatedJumps.npy")
    
    additionalGoodPairsIdxs = np.argwhere(validatedJumps == 1).T
    if additionalGoodPairsIdxs.shape[-1] > 0 :
        goodPairsIdxs = np.concatenate((goodPairsIdxs, additionalGoodPairsIdxs), axis=1)
    
    badPairsIdxs = np.argwhere(validatedJumps == 0).T
    ### why did I do this?
#     badPairsIdxs[1, : ] -= 1
    print additionalGoodPairsIdxs.T
    print badPairsIdxs.T
else :
    ## get feats of random pairings that are considered bad
    numBadExamples = 1000
    minIdxsDiff = 10
    badPairsIdxs = np.sort(np.array([np.random.choice(np.arange(numFrames), numBadExamples), 
                                     np.random.choice(np.arange(numFrames), numBadExamples)]), axis=0)

    print len(np.argwhere(np.abs(badPairsIdxs[0, :]-badPairsIdxs[1, :]) < minIdxsDiff)), "invalid pairs"
    for pairIdx in xrange(numBadExamples) :
        idxDiff = np.abs(badPairsIdxs[0, pairIdx] - badPairsIdxs[1, pairIdx])
        tmp = idxDiff
        newPair = badPairsIdxs[:, pairIdx]
        while idxDiff < minIdxsDiff :
            newPair = np.sort(np.random.choice(np.arange(numFrames), 2))
            idxDiff = np.abs(newPair[0] - newPair[1])
    #     print badPairsIdxs[:, pairIdx], newPair, tmp
        badPairsIdxs[:, pairIdx] = newPair
    #     if badPairsIdxs[pairIdx, 0] - badPairsIdxs[pairIdx, 1] < minIdxsDiff

    # print badPairsIdxs.T
    print len(np.argwhere(np.abs(badPairsIdxs[0, :]-badPairsIdxs[1, :]) < minIdxsDiff)), "invalid pairs"
    print badPairsIdxs

# <codecell>

## copy the example idxs that uses the validated jumps to tmp
# goodPairsIncludingValidation = np.copy(goodPairsIdxs)
# badPairsIncludingValidation = np.copy(badPairsIdxs)
## attach them to the example idxs without validated jumps to see how it looks
# goodPairsIdxs = np.copy(goodPairsIncludingValidation)
# badPairsIdxs = np.concatenate((badPairsIdxs, badPairsIncludingValidation), axis=1)
print goodPairsIdxs.shape
print badPairsIdxs.shape

# <codecell>

allPairsHogs = []
for i in xrange(len(hogFeats)) :
    for j in xrange(i+1, len(hogFeats)) :
        ## ABS DIST
#         allPairsHogs.append(np.sqrt((hogFeats[i, :]-hogFeats[j, :])**2))
        allPairsHogs.append((hogFeats[i, :]-hogFeats[j, :])**2)
        
    sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(len(hogFeats)))
    sys.stdout.flush()

# <codecell>

allPairsImageData = []
for i in xrange(numFrames) :
    for j in xrange(i+1, numFrames) :
        ## ABS DIST
#         allPairsHogs.append(np.sqrt((hogFeats[i, :]-hogFeats[j, :])**2))
        if doRGB :
            allPairsImageData.append((imagesRGBData[:, i]-imagesRGBData[:, j])**np.float32(2))
        else :
            allPairsImageData.append((imagesGrayData[:, i]-imagesGrayData[:, j])**np.float32(2))
#             allPairsImageData.append(imagesGrayData[:, i]-imagesGrayData[:, j])
        
    sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()

# <codecell>

## split examples in half 
# allBadPairs = np.copy(badPairsIdxs)
# allGoodPairs = np.copy(goodPairsIdxs)
numGoodPairs = allGoodPairs.shape[-1]
numBadPairs = allBadPairs.shape[-1]
print numGoodPairs, numBadPairs

goodValidationExamples = np.copy(allGoodPairs[:, numGoodPairs/2:])
badValidationExamples = np.copy(allBadPairs[:, numBadPairs/2:])
print goodValidationExamples.shape, badValidationExamples.shape

goodPairsIdxs = np.copy(allGoodPairs[:, :numGoodPairs/2])
badPairsIdxs = np.copy(allBadPairs[:, :numBadPairs/2])
print goodPairsIdxs.shape, badPairsIdxs.shape

# <codecell>

print goodExamplesData.shape
print badExamplesData.shape
print np.concatenate((goodExamplesData[:, goodPairsToUse], badExamplesData[:, badPairsToUse]), axis=1).shape

# <codecell>

doHogs = False
goodPairsToUse = arange(len(goodPairsIdxs.T))
# goodPairsToUse = arange(len(goodPairsIdxs.T)-len(additionalGoodPairsIdxs.T), len(goodPairsIdxs.T))

badPairsToUse = arange(len(badPairsIdxs.T))
# badPairsToUse = np.delete(badPairsToUse, 12)
# badPairsToUse = badPairsToUse[10:]
if doHogs :
    ## use hog feats as frame features
    ## ABS DIST
    # goodExamplesData = np.sqrt((hogFeats[goodPairsIdxs[0, :], :]-hogFeats[goodPairsIdxs[1, :], :])**2)
    goodExamplesData = ((hogFeats[goodPairsIdxs[0, :], :]-hogFeats[goodPairsIdxs[1, :], :])**2).T
    print goodExamplesData.shape
    ## ABS DIST
    # badExamplesData = np.sqrt((hogFeats[badPairsIdxs[0, :], :]-hogFeats[badPairsIdxs[1, :], :])**2)
    badExamplesData = ((hogFeats[badPairsIdxs[0, :], :]-hogFeats[badPairsIdxs[1, :], :])**2).T
    print badExamplesData.shape
    
    X = np.concatenate((goodExamplesData[:, goodPairsToUse], badExamplesData[:, badPairsToUse]), axis=1)
    w = np.concatenate((np.zeros(len(goodPairsToUse)), 10.0*np.ones(len(badPairsToUse)))).reshape((X.shape[-1], 1))
    N = X.shape[0]
    phi0 = np.ones((N, 1))
    
    sio.savemat(dataPath + dataSet + "trainingExamplesForHogs", {"X":X, "w":w})
else :
    ## use full rgb feats as frame features
    if doRGB :
        goodExamplesData = (imagesRGBData[:, goodPairsIdxs[0, :]]-imagesRGBData[:, goodPairsIdxs[1, :]])**np.float32(2)
        print goodExamplesData.shape
        badExamplesData = (imagesRGBData[:, badPairsIdxs[0, :]]-imagesRGBData[:, badPairsIdxs[1, :]])**np.float32(2)
        print badExamplesData.shape
    else :
        goodExamplesData = (imagesGrayData[:, goodPairsIdxs[0, :]]-imagesGrayData[:, goodPairsIdxs[1, :]])**np.float32(2)
        print goodExamplesData.shape
        badExamplesData = (imagesGrayData[:, badPairsIdxs[0, :]]-imagesGrayData[:, badPairsIdxs[1, :]])**np.float32(2)
        print badExamplesData.shape
    if useRange :
        goodExamplesData = goodExamplesData[:, int(np.floor(len(featsRange)/2)):-int(np.floor(len(featsRange)/2))]
    
#     X = np.concatenate((goodExamplesData, badExamplesData), axis=1)
#     w = np.concatenate((np.zeros(goodExamplesData.shape[-1]), 10.0*np.ones(badExamplesData.shape[-1]))).reshape((X.shape[-1], 1))
    X = np.concatenate((goodExamplesData[:, goodPairsToUse], badExamplesData[:, badPairsToUse]), axis=1)
    w = np.concatenate((np.zeros(len(goodPairsToUse)), 10.0*np.ones(len(badPairsToUse)))).reshape((X.shape[-1], 1))
    N = X.shape[0]
    phi0 = np.ones((N, 1))
    sio.savemat(dataPath + dataSet + "trainingExamplesForImageData", {"X":X, "w":w})
    
print "used", len(goodPairsToUse), "good examples of", goodExamplesData.shape[-1]
print "used", len(badPairsToUse), "bad examples of", badExamplesData.shape[-1]
print N, X.shape, w.shape, phi0.shape

# <codecell>

imageSize = np.array(Image.open(framePaths[0])).shape[0:2]
print imageSize

# <codecell>

## compute grid features for the training examples

gridSize = np.array((10, 10))

stencils3D = cgf.stencil3D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])
stencils2D = cgf.stencil2D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])
# figure(); imshow(im[:, :, 0][stencils2D[5]].reshape((180, 320)))
# figure(); imshow(im[stencils3D[5]].reshape((180, 320, 3)))

useFlow = True
useAugment = True
k = 1.0
L = 10.0

numTypes = 1
if useAugment :
    numTypes += 1
if useFlow :
    numTypes += 1

t = time.time()
## slow stupid way for now
goodExamplesData = np.zeros((numTypes*np.prod(gridSize), len(goodPairsIdxs.T)))
# goodExamplesData = np.concatenate((goodExamplesData, np.zeros((np.prod(gridSize), len(goodPairsIdxs.T)))))
for i, pair in enumerate(goodPairsIdxs.T) :
    
    if useFlow :
        flow = np.array(cv2.calcOpticalFlowFarneback(np.array(imagesGrayData[:, pair[0]].reshape(imageSize)*255, np.uint8),
                                                     np.array(imagesGrayData[:, pair[1]].reshape(imageSize)*255, np.uint8),
                                                     0.5, 3, 15, 3, 5, 1.1, 0), np.float64)
        ## sigmoid
        flow = L/(1.0+np.exp(-k*flow))
    for j, stencil in enumerate(stencils2D) :
        ## add average SSD to feature vector
        goodExamplesData[j, i] = np.average((imagesGrayData[:, pair[0]].reshape(imageSize)[stencil]/1.0-imagesGrayData[:, pair[1]].reshape(imageSize)[stencil]/1.0)**2)
        ## add average flow intensity
        if useFlow :
            goodExamplesData[j+np.prod(gridSize), i] = np.average(np.linalg.norm(np.array([flow[:, :, 0][stencil], flow[:, :, 1][stencil]]), axis=0))
        
        ## add augmentation to x
        if useAugment :
            shift = 2
            if not useFlow :
                shift = 1
            goodExamplesData[j+shift*np.prod(gridSize), i] = np.average(((1.0-imagesGrayData[:, pair[0]].reshape(imageSize)[stencil])/1.0-imagesGrayData[:, pair[1]].reshape(imageSize)[stencil]/1.0)**2+
                                                                        ((1.0-imagesGrayData[:, pair[1]].reshape(imageSize)[stencil])/1.0-imagesGrayData[:, pair[0]].reshape(imageSize)[stencil]/1.0)**2)
        
    sys.stdout.write('\r' + "Computed features for good pair " + np.string_(i+1) + " of " + np.string_(len(goodPairsIdxs.T)))
    sys.stdout.flush()
print
badExamplesData = np.zeros((numTypes*np.prod(gridSize), len(badPairsIdxs.T)))
# badExamplesData = np.concatenate((badExamplesData, np.zeros((np.prod(gridSize), len(badPairsIdxs.T)))))
for i, pair in enumerate(badPairsIdxs.T) :
    
    if useFlow :
        flow = np.array(cv2.calcOpticalFlowFarneback(np.array(imagesGrayData[:, pair[0]].reshape(imageSize)*255, np.uint8),
                                                     np.array(imagesGrayData[:, pair[1]].reshape(imageSize)*255, np.uint8),
                                                     0.5, 3, 15, 3, 5, 1.1, 0), np.float64)
        ## sigmoid
        flow = L/(1.0+np.exp(-k*flow))
    
    for j, stencil in enumerate(stencils2D) :
        ## add SSD to feature vector
        badExamplesData[j, i] = np.average((imagesGrayData[:, pair[0]].reshape(imageSize)[stencil]/1.0-imagesGrayData[:, pair[1]].reshape(imageSize)[stencil]/1.0)**2)
        ## add average flow intensity
        if useFlow :
            badExamplesData[j+np.prod(gridSize), i] = np.average(np.linalg.norm(np.array([flow[:, :, 0][stencil], flow[:, :, 1][stencil]]), axis=0))
            
        ## add augmentation to x
        if useAugment :
            shift = 2
            if not useFlow :
                shift = 1
            badExamplesData[j+shift*np.prod(gridSize), i] = np.average(((1.0-imagesGrayData[:, pair[0]].reshape(imageSize)[stencil])/1.0-imagesGrayData[:, pair[1]].reshape(imageSize)[stencil]/1.0)**2+
                                                                       ((1.0-imagesGrayData[:, pair[1]].reshape(imageSize)[stencil])/1.0-imagesGrayData[:, pair[0]].reshape(imageSize)[stencil]/1.0)**2)
        
    sys.stdout.write('\r' + "Computed features for bad pair " + np.string_(i+1) + " of " + np.string_(len(badPairsIdxs.T)))
    sys.stdout.flush()
print 
print time.time() - t
print goodPairsIdxs.shape
print badPairsIdxs.shape

# <codecell>

## compute grid features for the training examples

gridSize = np.array((10, 10))

stencils3D = cgf.stencil3D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])
stencils2D = cgf.stencil2D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])
# figure(); imshow(im[:, :, 0][stencils2D[5]].reshape((180, 320)))
# figure(); imshow(im[stencils3D[5]].reshape((180, 320, 3)))

useFlow = True
useAugment = True
k = 1.0
L = 10.0

numTypes = 1
if useAugment :
    numTypes += 1
if useFlow :
    numTypes += 1

t = time.time()
## slow stupid way for now
goodExamplesData = np.zeros((numTypes*np.prod(gridSize), len(goodPairsIdxs.T)))
# goodExamplesData = np.concatenate((goodExamplesData, np.zeros((np.prod(gridSize), len(goodPairsIdxs.T)))))
for i, pair in enumerate(goodPairsIdxs.T) :
    frame1 = cv2.cvtColor(np.array(Image.open(framePaths[pair[0]])), cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(np.array(Image.open(framePaths[pair[1]])), cv2.COLOR_RGB2GRAY)
    
    if useFlow :
        flow = np.array(cv2.calcOpticalFlowFarneback(frame1, frame2,
                                                     0.5, 3, 15, 3, 5, 1.1, 0), np.float64)
        ## sigmoid
        flow = L/(1.0+np.exp(-k*flow))
    for j, stencil in enumerate(stencils2D) :
        ## add average SSD to feature vector
        goodExamplesData[j, i] = np.average((frame1[stencil]/255.0-frame2[stencil]/255.0)**2)
        ## add average flow intensity
        if useFlow :
            goodExamplesData[j+np.prod(gridSize), i] = np.average(np.linalg.norm(np.array([flow[:, :, 0][stencil], flow[:, :, 1][stencil]]), axis=0))
        
        ## add augmentation to x
        if useAugment :
            shift = 2
            if not useFlow :
                shift = 1
            goodExamplesData[j+shift*np.prod(gridSize), i] = np.average(((255.0-frame1[stencil])/255.0-frame2[stencil]/255.0)**2+
                                                                        ((255.0-frame2[stencil])/255.0-frame1[stencil]/255.0)**2)
        
    sys.stdout.write('\r' + "Computed features for good pair " + np.string_(i+1) + " of " + np.string_(len(goodPairsIdxs.T)))
    sys.stdout.flush()
print
badExamplesData = np.zeros((numTypes*np.prod(gridSize), len(badPairsIdxs.T)))
# badExamplesData = np.concatenate((badExamplesData, np.zeros((np.prod(gridSize), len(badPairsIdxs.T)))))
for i, pair in enumerate(badPairsIdxs.T) :
    frame1 = cv2.cvtColor(np.array(Image.open(framePaths[pair[0]])), cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(np.array(Image.open(framePaths[pair[1]])), cv2.COLOR_RGB2GRAY)
    
    if useFlow :
        flow = np.array(cv2.calcOpticalFlowFarneback(frame1, frame2,
                                                     0.5, 3, 15, 3, 5, 1.1, 0), np.float64)
        ## sigmoid
        flow = L/(1.0+np.exp(-k*flow))
    
    for j, stencil in enumerate(stencils2D) :
        ## add SSD to feature vector
        badExamplesData[j, i] = np.average((frame1[stencil]/255.0-frame2[stencil]/255.0)**2)
        ## add average flow intensity
        if useFlow :
            badExamplesData[j+np.prod(gridSize), i] = np.average(np.linalg.norm(np.array([flow[:, :, 0][stencil], flow[:, :, 1][stencil]]), axis=0))
            
        ## add augmentation to x
        if useAugment :
            shift = 2
            if not useFlow :
                shift = 1
            badExamplesData[j+shift*np.prod(gridSize), i] = np.average(((255.0-frame1[stencil])/255.0-frame2[stencil]/255.0)**2+
                                                                       ((255.0-frame2[stencil])/255.0-frame1[stencil]/255.0)**2)
        
    sys.stdout.write('\r' + "Computed features for bad pair " + np.string_(i+1) + " of " + np.string_(len(badPairsIdxs.T)))
    sys.stdout.flush()
print 
print time.time() - t
print goodPairsIdxs.shape
print badPairsIdxs.shape

# <codecell>

## split examples in half 
# allBadPairs = np.copy(badPairsIdxs)
# allGoodPairs = np.copy(goodPairsIdxs)
numGoodPairs = allGoodPairs.shape[-1]
numBadPairs = allBadPairs.shape[-1]
print numGoodPairs, numBadPairs

goodValidationExamples = np.copy(allGoodPairs[:, numGoodPairs/2:])
badValidationExamples = np.copy(allBadPairs[:, numBadPairs/2:])
print goodValidationExamples.shape, badValidationExamples.shape

goodPairsIdxs = np.copy(allGoodPairs[:, :numGoodPairs/2])
badPairsIdxs = np.copy(allBadPairs[:, :numBadPairs/2])
print goodPairsIdxs.shape, badPairsIdxs.shape

# <codecell>

import datetime
print "started at",
print datetime.datetime.now()
goodExamplesMovement = []
for j, pair in enumerate(goodPairsIdxs.T) :
    
    ## find "how much left-movement there is" between two frames (low numbers means more right movement)
    frame1 = np.array(Image.open(framePaths[pair[0]]))
    frame2 = np.array(Image.open(framePaths[pair[1]]))
    imageSize = frame1.shape[0:2]
    patchSize = np.array([64, 64])
    spacing = 32.0 ## how far away a patch is from the previous one

    movementSigma = 8.0
    numSamples = 50
    patchYs = np.arange(4.0*movementSigma, imageSize[0]-patchSize[0]-4.0*movementSigma, spacing)
    patchYs = patchYs.reshape((1, len(patchYs)))
    patchXs = np.arange(4.0*movementSigma, imageSize[1]-patchSize[1]-4.0*movementSigma, spacing)
    patchXs = patchXs.reshape((1, len(patchXs)))
    patchLocations = np.array(np.concatenate((patchYs.repeat(len(patchXs.T)),
                                              patchXs.repeat(len(patchYs.T), axis=0).flatten())).reshape((2, len(patchXs.T)*len(patchYs.T))), int)

#     print patchLocations.shape

    # figure(); imshow(frame1)
    scores = np.zeros(patchLocations.shape[-1])
    for i, loc in enumerate(patchLocations.T) : ## [:, [0, 1282, 2967]]
    #     print loc
        ## changing the x-mean by - movementSigma as I want to go left
        mean = np.array([loc[1]-2.0*movementSigma, loc[0]])
        covariance = np.array([[movementSigma, 0.0], [0.0, movementSigma/10.0]])
        ## samples contains (x, y) coords
        samples = np.random.multivariate_normal(mean, covariance, numSamples)
        weights = sp.stats.multivariate_normal.pdf(samples, mean, covariance)
        weights /= np.sum(weights)
    #     print samples
        ## displacedLocs contains (row, col) coords
        displacedLocs = np.array(np.round(samples).T[::-1], int)
        validLocs = np.negative(np.any([np.any(displacedLocs < 0, axis=0), displacedLocs[0, :] >= imageSize[0]-patchSize[0], displacedLocs[1, :] >= imageSize[1]-patchSize[1]], axis=0))
        frame2Patches = [frame2[l[0]:l[0]+patchSize[0], l[1]:l[1]+patchSize[1]] for l in displacedLocs[:, validLocs].T]
        frame1Patch = frame1[loc[0]:loc[0]+patchSize[0], loc[1]:loc[1]+patchSize[1], :].reshape((1, patchSize[0], patchSize[0], frame1.shape[-1])).repeat(len(frame2Patches), axis=0)
        scores[i] = np.sum(np.sum(np.sqrt((frame1Patch.reshape((len(frame2Patches), np.prod(patchSize)*frame1.shape[-1]))/255.0-
                                           np.array(frame2Patches).reshape((len(frame2Patches), np.prod(patchSize)*frame1.shape[-1]))/255.0)**2), axis=-1)*np.ones_like(weights[validLocs]))

        sys.stdout.write('\r' + "Done patch " + np.string_(i) + " of " + np.string_(patchLocations.shape[-1]) + " for good pair " + np.string_(j) + " of " + np.string_(len(goodPairsIdxs.T)))
        sys.stdout.flush()
    goodExamplesMovement.append(np.copy(scores))
print
badExamplesMovement = []
for j, pair in enumerate(badPairsIdxs.T) :
    
    ## find "how much left-movement there is" between two frames (low numbers means more right movement)
    frame1 = np.array(Image.open(framePaths[pair[0]]))
    frame2 = np.array(Image.open(framePaths[pair[1]]))
    imageSize = frame1.shape[0:2]
    patchSize = np.array([64, 64])
    spacing = 32.0 ## how far away a patch is from the previous one

    movementSigma = 8.0
    numSamples = 50
    patchYs = np.arange(4.0*movementSigma, imageSize[0]-patchSize[0]-4.0*movementSigma, spacing)
    patchYs = patchYs.reshape((1, len(patchYs)))
    patchXs = np.arange(4.0*movementSigma, imageSize[1]-patchSize[1]-4.0*movementSigma, spacing)
    patchXs = patchXs.reshape((1, len(patchXs)))
    patchLocations = np.array(np.concatenate((patchYs.repeat(len(patchXs.T)),
                                              patchXs.repeat(len(patchYs.T), axis=0).flatten())).reshape((2, len(patchXs.T)*len(patchYs.T))), int)

#     print patchLocations.shape

    # figure(); imshow(frame1)
    scores = np.zeros(patchLocations.shape[-1])
    for i, loc in enumerate(patchLocations.T) : ## [:, [0, 1282, 2967]]
    #     print loc
        ## changing the x-mean by - movementSigma as I want to go left
        mean = np.array([loc[1]-2.0*movementSigma, loc[0]])
        covariance = np.array([[movementSigma, 0.0], [0.0, movementSigma/10.0]])
        ## samples contains (x, y) coords
        samples = np.random.multivariate_normal(mean, covariance, numSamples)
        weights = sp.stats.multivariate_normal.pdf(samples, mean, covariance)
        weights /= np.sum(weights)
    #     print samples
        ## displacedLocs contains (row, col) coords
        displacedLocs = np.array(np.round(samples).T[::-1], int)
        validLocs = np.negative(np.any([np.any(displacedLocs < 0, axis=0), displacedLocs[0, :] >= imageSize[0]-patchSize[0], displacedLocs[1, :] >= imageSize[1]-patchSize[1]], axis=0))
        frame2Patches = [frame2[l[0]:l[0]+patchSize[0], l[1]:l[1]+patchSize[1]] for l in displacedLocs[:, validLocs].T]
        frame1Patch = frame1[loc[0]:loc[0]+patchSize[0], loc[1]:loc[1]+patchSize[1], :].reshape((1, patchSize[0], patchSize[0], frame1.shape[-1])).repeat(len(frame2Patches), axis=0)
        scores[i] = np.sum(np.sum(np.sqrt((frame1Patch.reshape((len(frame2Patches), np.prod(patchSize)*frame1.shape[-1]))/255.0-
                                           np.array(frame2Patches).reshape((len(frame2Patches), np.prod(patchSize)*frame1.shape[-1]))/255.0)**2), axis=-1)*np.ones_like(weights[validLocs]))

        sys.stdout.write('\r' + "Done patch " + np.string_(i) + " of " + np.string_(patchLocations.shape[-1]) + " for bad pair " + np.string_(j) + " of " + np.string_(len(badPairsIdxs.T)))
        sys.stdout.flush()
    badExamplesMovement.append(np.copy(scores))
print

print "ended at",
print datetime.datetime.now()

# <codecell>

np.save("good_examples_flag_movement_left.npy", goodExamplesMovement)
np.save("bad_examples_flag_movement_left.npy", badExamplesMovement)

# <codecell>

patchSize = np.array([64, 64])
spacing = 32.0 ## how far away a patch is from the previous one

displacements = np.array([1, 2, 4, 8, 16])
displacements = np.concatenate((np.array([np.zeros(len(displacements)), displacements], int), ## EAST
                      np.array([displacements, displacements], int), ## SOUT-EAST
                      np.array([displacements, np.zeros(len(displacements))], int), ## SOUTH
                      np.array([displacements, -displacements], int), ## SOUTH-WEST
                      np.array([np.zeros(len(displacements)), -displacements], int), ## WEST
                      np.array([-displacements, -displacements], int), ## NORTH-WEST
                      np.array([-displacements, np.zeros(len(displacements))], int), ## NORTH
                      np.array([-displacements, displacements], int), ## NORTH-EAST
                      ), axis=-1)

patchYs = np.arange(np.max(displacements), imageSize[0]-patchSize[0]-np.max(displacements), spacing)
patchYs = patchYs.reshape((1, len(patchYs)))
patchXs = np.arange(np.max(displacements), imageSize[1]-patchSize[1]-np.max(displacements), spacing)
patchXs = patchXs.reshape((1, len(patchXs)))
patchLocations = np.array(np.concatenate((patchYs.repeat(len(patchXs.T)),
                                          patchXs.repeat(len(patchYs.T), axis=0).flatten())).reshape((2, len(patchXs.T)*len(patchYs.T))), int)

frame1Idxs = np.empty((0, 2), dtype=int)
frame2Idxs = np.empty((0, 2), dtype=int)
for i, locSlice in enumerate(np.arange(patchXs.shape[-1], patchLocations.shape[-1]+1, patchXs.shape[-1])) :
    frame1Idxs = np.concatenate((frame1Idxs, np.repeat(patchLocations[:, locSlice-patchXs.shape[-1]:locSlice], len(displacements.T), axis=1).T), axis=0)
    frame2Idxs = np.concatenate((frame2Idxs, np.array([disp + loc for loc in patchLocations[:, locSlice-patchXs.shape[-1]:locSlice].T for disp in displacements.T])), axis=0)


goodExamplesStarMovement = []
for j, pair in enumerate(goodPairsIdxs.T) :
    t = time.time()
    frame1 = np.array(Image.open(framePaths[pair[0]]))/255.0
    frame2 = np.array(Image.open(framePaths[pair[1]]))/255.0

    goodExamplesStarMovement.append(np.array([np.sqrt(np.sum((frame2[l2[0]:l2[0]+patchSize[0], l2[1]:l2[1]+patchSize[1]]-frame1[l1[0]:l1[0]+patchSize[0], l1[1]:l1[1]+patchSize[1]])**2)) for l2, l1 in zip(frame2Idxs, frame1Idxs)]))
    
    print time.time()-t
    sys.stdout.write('\r' + "Done good pair " + np.string_(j) + " of " + np.string_(len(goodPairsIdxs.T)))
    sys.stdout.flush()
    
badExamplesStarMovement = []
for j, pair in enumerate(badPairsIdxs.T) :
    frame1 = np.array(Image.open(framePaths[pair[0]]))/255.0
    frame2 = np.array(Image.open(framePaths[pair[1]]))/255.0

    badExamplesStarMovement.append(np.array([np.sqrt(np.sum((frame2[l2[0]:l2[0]+patchSize[0], l2[1]:l2[1]+patchSize[1]]-frame1[l1[0]:l1[0]+patchSize[0], l1[1]:l1[1]+patchSize[1]])**2)) for l2, l1 in zip(frame2Idxs, frame1Idxs)]))
    
    sys.stdout.write('\r' + "Done bad pair " + np.string_(j) + " of " + np.string_(len(badPairsIdxs.T)))
    sys.stdout.flush()

# <codecell>

print displacements

# <codecell>

# tmpExamples = np.array([np.sqrt(np.sum((frame2[l2[0]:l2[0]+patchSize[0], l2[1]:l2[1]+patchSize[1]]-frame1[l1[0]:l1[0]+patchSize[0], l1[1]:l1[1]+patchSize[1]])**2)) for l2, l1 in zip(frame2Idxs, frame1Idxs)])
tmp1Idxs = frame1Idxs[0:5, :]
tmp2Idxs = frame2Idxs[0:5, :]

print tmp1Idxs, tmp2Idxs

tmp1Idxs = np.empty((2, 0), int)
tmp2Idxs = np.empty((2, 0), int)
for l1, l2 in zip(frame1Idxs[0:5, :], frame2Idxs[0:5, :]) :
    tmp1Idxs = np.concatenate((tmp1Idxs, np.array(np.meshgrid(arange(l1[0], l1[0]+patchSize[0]), arange(l1[1], l1[1]+patchSize[1]))).reshape((2, np.prod(patchSize)))), axis=1)
    tmp2Idxs = np.concatenate((tmp2Idxs, np.array(np.meshgrid(arange(l2[0], l2[0]+patchSize[0]), arange(l2[1], l2[1]+patchSize[1]))).reshape((2, np.prod(patchSize)))), axis=1)
    
    sys.stdout.write('\r' + "Lala " + np.string_(tmp1Idxs.shape[1]/4096) + " of " + np.string_(len(frame1Idxs)))
    sys.stdout.flush()
    

clear_output()
# bob1 = np.array([np.meshgrid(arange(l[0], l[0]+patchSize[0]), arange(l[1], l[1]+patchSize[1])) for l in tmp1Idxs]).reshape((tmp1Idxs.shape[0], 2, np.prod(patchSize)))
# bob2 = np.array([np.meshgrid(arange(l[0], l[0]+patchSize[0]), arange(l[1], l[1]+patchSize[1])) for l in tmp2Idxs]).reshape((tmp2Idxs.shape[0], 2, np.prod(patchSize)))
# # print np.array(np.meshgrid(arange(5, 7), arange(3, 5))).shape
# print bob1.shape
# print bob1.reshape((5, 2, 64**2))[0, :, :].T.shape
# lala = np.array([frame1[la[0], la[1], :] for la in bob1.reshape((5, 2, 64**2))[0, :, :].T])
# print lala.shape

# print np.argwhere(frame1[bob1.reshape((5, 2, 64**2))[0, 0, :], bob1.reshape((5, 2, 64**2))[0, 1, :]]-lala != 0.0)


# print np.sum(lala, axis=1)[0:16]
# print np.sum(lala, axis=1).reshape((8, 8*64), order='F')[:, 0:2]
# print np.sum(lala, axis=1).reshape((8, 8*64), order='F')[:, 1:2].flatten() - np.sum(lala, axis=1)[8:16]

# print bob1[0, :, :]
# print bob1[1, :, :]
# print bob1.reshape((4096*5, 2))
# print 




# tom = np.array([np.sum(frame1[la[0, :], la[01, :], :], axis=-1) for la in bob1])
# print tom.shape
# print np.argwhere(np.sum(frame1[bob1.reshape((4096*5, 2))[:, 0], bob1.reshape((4096*5, 2))[:, 1], :], axis=-1)[:4096]-tom[0, :] != 0.0)

# <codecell>

t = time.time()
tmp1Idxs = np.array([np.array(np.meshgrid(arange(l1[0], l1[0]+patchSize[0]), 
                                          arange(l1[1], l1[1]+patchSize[1]))).reshape((2, np.prod(patchSize))) for l1 in frame1Idxs]).swapaxes(0, 1).reshape((2, len(frame1Idxs)*np.prod(patchSize)))

tmp2Idxs = np.array([np.array(np.meshgrid(arange(l2[0], l2[0]+patchSize[0]), 
                                          arange(l2[1], l2[1]+patchSize[1]))).reshape((2, np.prod(patchSize))) for l2 in frame2Idxs]).swapaxes(0, 1).reshape((2, len(frame2Idxs)*np.prod(patchSize)))
print time.time() - t
print tmp1Idxs.shape

# <codecell>

print np.sqrt(tmp1Idxs.shape[1])

# <codecell>

t = time.time()
np.sqrt(np.sum(np.sum((frame2[tmp2Idxs[0, :], tmp2Idxs[1, :], :] - frame1[tmp1Idxs[0, :], tmp1Idxs[1, :], :])**2, axis=-1).reshape((len(frame1Idxs), np.prod(patchSize))), axis=-1))
print time.time()-t

# <codecell>

print tmpExamples[0:5]

# <codecell>

labels = ['East', 'South-East', 'South', 'South-West', 'West', 'North-West', 'North', 'North-East']
figure()
for dispDirIdx, col in enumerate(cm.jet(np.arange(0, 255, 36))) :
    print dispDirIdx, col
    vals = [np.average(np.array(badExamplesStarMovement)[:, arange(dispLenIdx+dispDirIdx*5, len(badExamplesStarMovement[0]), len(displacements.T))]) for dispLenIdx in arange(5)]
    print vals
    print
    plot(vals, color=col, marker='.', label=labels[dispDirIdx])
legend()

# <codecell>

# dispLenIdx = 0
# dispDirIdx = 0
print dispDirIdx, dispLenIdx
print np.sum(np.array(goodExamplesStarMovement)[:, arange(dispLenIdx+0*5, len(goodExamplesStarMovement[0]), len(displacements.T))])

# <codecell>

print displacements.T
print arange

# <codecell>

bob = np.zeros(imageSize)
bob[frame2Idxs[:, 0], frame2Idxs[:, 1]] = np.average(np.array(goodExamplesStarMovement), axis=0)
gwv.showCustomGraph(bob)

# <codecell>

print np.arange(patchXs.shape[-1], patchLocations.shape[-1]+1, patchXs.shape[-1]).shape
print patchLocations.shape

# <codecell>

print "bad examples shapes"
print np.array(badExamplesMovement).shape
print badExamplesData.shape
badExamplesData = np.concatenate((badExamplesData, np.array(badExamplesMovement).T), axis=0)
print badExamplesData.shape

print "good examples shapes"
print np.array(goodExamplesMovement).shape
print goodExamplesData.shape
goodExamplesData = np.concatenate((goodExamplesData, np.array(goodExamplesMovement).T), axis=0)
print goodExamplesData.shape

# <codecell>

bob = np.zeros(imageSize)
bob[patchLocations[0, :], patchLocations[1, :]] = badExamplesMovement[10]
gwv.showCustomGraph(bob)

# <codecell>

# badDataBAK = np.copy(badExamplesData)
# goodDataBAK = np.copy(goodExamplesData)

# <codecell>

print np.array(goodExamplesStarMovement).shape

# <codecell>

# gwv.showCustomGraph(np.average(goodDataBAK[np.prod(gridSize):2*np.prod(gridSize), goodPairsToUse], axis=1).reshape(gridSize))
# gwv.showCustomGraph(np.average(badDataBAK[np.prod(gridSize):2*np.prod(gridSize), badPairsToUse], axis=1).reshape(gridSize))
gwv.showCustomGraph(np.average(goodExamplesData[np.prod(gridSize):2*np.prod(gridSize), goodPairsToUse], axis=1).reshape(gridSize))
gwv.showCustomGraph(np.average(badExamplesData[np.prod(gridSize):2*np.prod(gridSize), badPairsToUse], axis=1).reshape(gridSize))

# <codecell>

figure(); imshow((np.average(goodExamplesData[np.prod(gridSize):2*np.prod(gridSize), goodPairsToUse], axis=1).reshape(gridSize)), interpolation='nearest')
cb = colorbar(); cb.set_clim((0.63, 0.83))
figure(); imshow((np.average(badExamplesData[np.prod(gridSize):2*np.prod(gridSize), badPairsToUse], axis=1).reshape(gridSize)), interpolation='nearest')
cb = colorbar(); cb.set_clim((0.63, 0.83))

# <codecell>

goodPairsToUse = arange(len(goodPairsIdxs.T))
# goodPairsToUse = arange(len(goodPairsIdxs.T)-len(additionalGoodPairsIdxs.T), len(goodPairsIdxs.T))
badPairsToUse = arange(len(badPairsIdxs.T))

# X = np.concatenate((goodExamplesData[:3*np.prod(gridSize), goodPairsToUse], badExamplesData[:3*np.prod(gridSize), badPairsToUse]), axis=1)
# X = np.concatenate((goodExamplesData[3*np.prod(gridSize):, goodPairsToUse], badExamplesData[3*np.prod(gridSize):, badPairsToUse]), axis=1)
# X = np.concatenate((np.array(goodExamplesStarMovement).T, np.array(badExamplesStarMovement).T), axis=1)
X = np.concatenate((np.concatenate((np.array(goodExamplesStarMovement).T, goodExamplesData), axis=0), 
                    np.concatenate((np.array(badExamplesStarMovement).T, badExamplesData), axis=0)), axis=1)
w = np.concatenate((np.zeros(len(goodPairsToUse)), 1.0*np.ones(len(badPairsToUse)))).reshape((X.shape[-1], 1))
N = X.shape[0]
phi0 = np.ones((N, 1))
sio.savemat(dataPath + dataSet + "trainingExamplesForImageData", {"X":X, "w":w})
print X.shape

# <codecell>

print 8.8 - np.floor(8.8)

# <codecell>

print np.array(badExamplesStarMovement).T.shape

# <codecell>

# ## now call the matlab script to fit phi
# if doHogs :
#     trainingExamplesLoc = dataPath + dataSet + "trainingExamplesForHogs.mat"
#     phiSaveLoc = dataPath + dataSet + "fittedPhiForHogs.mat"
# else :
#     trainingExamplesLoc = dataPath + dataSet + "trainingExamplesForImageData.mat"
#     phiSaveLoc = dataPath + dataSet + "fittedPhiForImageData.mat"

# matlabCommand = "cd ~/PhD/MATLAB/; matlab -nosplash -nodesktop -nodisplay -r "
# matlabCommand += "\"fitPhiForRegression '" + trainingExamplesLoc + "' '"
# matlabCommand += phiSaveLoc + "'; exit;\""

# stat, output = commands.getstatusoutput(matlabCommand)
# stat /= 256

# if stat == 10 :
#     print "Error when saving result"
# elif stat == 11 :
#     print "Error when loading examples"
# else :
#     print "Optimization completed with status", stat
    
# print output

# <codecell>

## now call the matlab script to fit phi using psi
if doHogs :
    trainingExamplesLoc = dataPath + dataSet + "trainingExamplesForHogs.mat"
    phiSaveLoc = dataPath + dataSet + "fittedPhiForHogsUsingPsi.mat"
else :
    trainingExamplesLoc = dataPath + dataSet + "trainingExamplesForImageData.mat"
    phiSaveLoc = dataPath + dataSet + "fittedPhiForImageDataUsingPsi.mat"

matlabCommand = "cd ~/PhD/MATLAB/; matlab -nosplash -nodesktop -nodisplay -r "
matlabCommand += "\"fitPsiForRegression '" + trainingExamplesLoc + "' '"
matlabCommand += phiSaveLoc + "'; exit;\""

stat, output = commands.getstatusoutput(matlabCommand)
stat /= 256

if stat == 10 :
    print "Error when saving result"
elif stat == 11 :
    print "Error when loading examples"
else :
    print "Optimization completed with status", stat
    
print output

# <codecell>

figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
# bob = np.zeros(imageSize)
# bob[frame2Idxs[:, 0], frame2Idxs[:, 1]] = sio.loadmat(phiSaveLoc)['phi_MAP'][:, 0]
# gwv.showCustomGraph(bob)

# <codecell>

figure()
weightSubset = len(goodExamplesStarMovement[0])
plot(np.sqrt(np.dot(X.T[:len(goodPairsIdxs.T), weightSubset:], sio.loadmat(phiSaveLoc)['phi_MAP'][weightSubset:, :])))
plot(arange(len(goodPairsIdxs.T), len(X.T)), np.sqrt(np.dot(X.T[len(goodPairsIdxs.T):, weightSubset:], sio.loadmat(phiSaveLoc)['phi_MAP'][weightSubset:, :])))
figure()
plot(np.sqrt(np.dot(X.T[:len(goodPairsIdxs.T), weightSubset:], np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'][weightSubset:, :]))))
plot(arange(len(goodPairsIdxs.T), len(X.T)), np.sqrt(np.dot(X.T[len(goodPairsIdxs.T):, weightSubset:], np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'][weightSubset:, :]))))

# <codecell>

## color average and flow intensity on 5x5 grid with sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average and flow intensity on 5x5 grid with sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average and flow intensity on 5x5 grid with sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average and flow intensity on 5x5 grid with sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]
print 4.21937917707/0.303323752561, 4.07502075956/0.298591493111

# <codecell>

## color average on 5x5 grid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average and flow intensity on 5x5 grid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average, flow intensity and complement diff on 5x5 grid 
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average on 10x10 grid wrong flow
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average and flow intensity on 10x10 grid wrong flow
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average and flow intensity on 10x10 grid with sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average and flow intensity on 10x10 grid no sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average, flow intensity and complement diff on 10x10 grid wrong flow
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average, flow intensity and complement diff on 10x10 grid with sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average, flow intensity and complement diff on 10x10 grid with less steep sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average, flow intensity and complement diff on 10x10 grid with even less steep sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

## color average, flow intensity and complement diff on 10x10 grid no sigmoid
print "Weighted", np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2))/X.shape[-1], 
print "L2", np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2))/X.shape[-1]

# <codecell>

figure(); plot(np.average(goodExamplesData[np.prod(gridSize):2*np.prod(gridSize), goodPairsToUse], axis=1))
figure(); plot(np.average(badExamplesData[np.prod(gridSize):2*np.prod(gridSize), badPairsToUse], axis=1))

figure(); plot(2.0/(1.0+np.exp(-np.average(goodExamplesData[np.prod(gridSize):2*np.prod(gridSize), goodPairsToUse], axis=1))))
figure(); plot(2.0/(1.0+np.exp(-np.average(badExamplesData[np.prod(gridSize):2*np.prod(gridSize), badPairsToUse], axis=1))))

# <codecell>

## printing data stats
print "good data stats: "
print "color diff range [", np.min(goodExamplesData[:prod(gridSize), :]), ",", np.max(goodExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print
print "bad data stats: "
print "color diff range [", np.min(badExamplesData[:prod(gridSize), :]), ",", np.max(badExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(badExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(badExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print

## showing the regressed weights and 1 example per label and how they look like after weighting
figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
reshapedWeights = sio.loadmat(phiSaveLoc)['phi_MAP'][:prod(gridSize)].reshape(gridSize)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][prod(gridSize):2*prod(gridSize)].reshape(gridSize)), axis=-1)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][2*prod(gridSize):3*prod(gridSize)].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(reshapedWeights, showColorbar=True)

goodExample = goodExamplesData[:prod(gridSize), 0].reshape(gridSize)
goodExample = np.concatenate((goodExample, goodExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
goodExample = np.concatenate((goodExample, goodExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(goodExample, showColorbar=True)
gwv.showCustomGraph(goodExample*reshapedWeights, showColorbar=True)
print "good example weighted distance", np.sum(goodExample*reshapedWeights)

badExample = badExamplesData[:prod(gridSize), 0].reshape(gridSize)
badExample = np.concatenate((badExample, badExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
badExample = np.concatenate((badExample, badExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(badExample, showColorbar=True)
gwv.showCustomGraph(badExample*reshapedWeights, showColorbar=True)
print "bad example weighted distance", np.sum(badExample*reshapedWeights)

# <codecell>

## printing data stats
print "good data stats: "
print "color diff range [", np.min(goodExamplesData[:prod(gridSize), :]), ",", np.max(goodExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print
print "bad data stats: "
print "color diff range [", np.min(badExamplesData[:prod(gridSize), :]), ",", np.max(badExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(badExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(badExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print

## showing the regressed weights and 1 example per label and how they look like after weighting
figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
reshapedWeights = sio.loadmat(phiSaveLoc)['phi_MAP'][:prod(gridSize)].reshape(gridSize)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][prod(gridSize):2*prod(gridSize)].reshape(gridSize)), axis=-1)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][2*prod(gridSize):3*prod(gridSize)].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(reshapedWeights, showColorbar=True)

goodExample = goodExamplesData[:prod(gridSize), 0].reshape(gridSize)
goodExample = np.concatenate((goodExample, goodExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
goodExample = np.concatenate((goodExample, goodExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(goodExample, showColorbar=True)
gwv.showCustomGraph(goodExample*reshapedWeights, showColorbar=True)
print "good example weighted distance", np.sum(goodExample*reshapedWeights)

badExample = badExamplesData[:prod(gridSize), 0].reshape(gridSize)
badExample = np.concatenate((badExample, badExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
badExample = np.concatenate((badExample, badExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(badExample, showColorbar=True)
gwv.showCustomGraph(badExample*reshapedWeights, showColorbar=True)
print "bad example weighted distance", np.sum(badExample*reshapedWeights)

# <codecell>

## printing data stats
print "good data stats: "
print "color diff range [", np.min(goodExamplesData[:prod(gridSize), :]), ",", np.max(goodExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print
print "bad data stats: "
print "color diff range [", np.min(badExamplesData[:prod(gridSize), :]), ",", np.max(badExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(badExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(badExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print

## showing the regressed weights and 1 example per label and how they look like after weighting
figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
reshapedWeights = sio.loadmat(phiSaveLoc)['phi_MAP'][:prod(gridSize)].reshape(gridSize)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][prod(gridSize):2*prod(gridSize)].reshape(gridSize)), axis=-1)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][2*prod(gridSize):3*prod(gridSize)].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(reshapedWeights, showColorbar=True)

goodExample = goodExamplesData[:prod(gridSize), 0].reshape(gridSize)
goodExample = np.concatenate((goodExample, goodExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
goodExample = np.concatenate((goodExample, goodExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(goodExample, showColorbar=True)
gwv.showCustomGraph(goodExample*reshapedWeights, showColorbar=True)
print "good example weighted distance", np.sum(goodExample*reshapedWeights)

badExample = badExamplesData[:prod(gridSize), 0].reshape(gridSize)
badExample = np.concatenate((badExample, badExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
badExample = np.concatenate((badExample, badExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(badExample, showColorbar=True)
gwv.showCustomGraph(badExample*reshapedWeights, showColorbar=True)
print "bad example weighted distance", np.sum(badExample*reshapedWeights)

# <codecell>

## printing data stats
print "good data stats: "
print "color diff range [", np.min(goodExamplesData[:prod(gridSize), :]), ",", np.max(goodExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print
print "bad data stats: "
print "color diff range [", np.min(badExamplesData[:prod(gridSize), :]), ",", np.max(badExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(badExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(badExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print

## showing the regressed weights and 1 example per label and how they look like after weighting
figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
reshapedWeights = sio.loadmat(phiSaveLoc)['phi_MAP'][:prod(gridSize)].reshape(gridSize)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][prod(gridSize):2*prod(gridSize)].reshape(gridSize)), axis=-1)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][2*prod(gridSize):3*prod(gridSize)].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(reshapedWeights, showColorbar=True)

goodExample = goodExamplesData[:prod(gridSize), 0].reshape(gridSize)
goodExample = np.concatenate((goodExample, goodExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
goodExample = np.concatenate((goodExample, goodExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(goodExample, showColorbar=True)
gwv.showCustomGraph(goodExample*reshapedWeights, showColorbar=True)
print "good example weighted distance", np.sum(goodExample*reshapedWeights)

badExample = badExamplesData[:prod(gridSize), 0].reshape(gridSize)
badExample = np.concatenate((badExample, badExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
badExample = np.concatenate((badExample, badExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(badExample, showColorbar=True)
gwv.showCustomGraph(badExample*reshapedWeights, showColorbar=True)
print "bad example weighted distance", np.sum(badExample*reshapedWeights)

# <codecell>

## printing data stats
print "good data stats: "
print "color diff range [", np.min(goodExamplesData[:prod(gridSize), :]), ",", np.max(goodExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print
print "bad data stats: "
print "color diff range [", np.min(badExamplesData[:prod(gridSize), :]), ",", np.max(badExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(badExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(badExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print

## showing the regressed weights and 1 example per label and how they look like after weighting
figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
reshapedWeights = sio.loadmat(phiSaveLoc)['phi_MAP'][:prod(gridSize)].reshape(gridSize)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][prod(gridSize):2*prod(gridSize)].reshape(gridSize)), axis=-1)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][2*prod(gridSize):3*prod(gridSize)].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(reshapedWeights, showColorbar=True)

goodExample = goodExamplesData[:prod(gridSize), 0].reshape(gridSize)
goodExample = np.concatenate((goodExample, goodExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
goodExample = np.concatenate((goodExample, goodExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(goodExample, showColorbar=True)
gwv.showCustomGraph(goodExample*reshapedWeights, showColorbar=True)
print "good example weighted distance", np.sum(goodExample*reshapedWeights)

badExample = badExamplesData[:prod(gridSize), 0].reshape(gridSize)
badExample = np.concatenate((badExample, badExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
badExample = np.concatenate((badExample, badExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(badExample, showColorbar=True)
gwv.showCustomGraph(badExample*reshapedWeights, showColorbar=True)
print "bad example weighted distance", np.sum(badExample*reshapedWeights)

# <codecell>

## printing data stats
print "good data stats: "
print "color diff range [", np.min(goodExamplesData[:prod(gridSize), :]), ",", np.max(goodExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(goodExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(goodExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print
print "bad data stats: "
print "color diff range [", np.min(badExamplesData[:prod(gridSize), :]), ",", np.max(badExamplesData[:prod(gridSize), :]), "]"
print "flow average range [", np.min(badExamplesData[prod(gridSize):2*prod(gridSize), :]), ",", np.max(badExamplesData[prod(gridSize):2*prod(gridSize), :]), "]"
print "color complement diff range [", np.min(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), ",", np.max(badExamplesData[2*prod(gridSize):3*prod(gridSize), :]), "]"
print

## showing the regressed weights and 1 example per label and how they look like after weighting
figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
reshapedWeights = sio.loadmat(phiSaveLoc)['phi_MAP'][:prod(gridSize)].reshape(gridSize)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][prod(gridSize):2*prod(gridSize)].reshape(gridSize)), axis=-1)
reshapedWeights = np.concatenate((reshapedWeights, sio.loadmat(phiSaveLoc)['phi_MAP'][2*prod(gridSize):3*prod(gridSize)].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(reshapedWeights, showColorbar=True)

goodExample = goodExamplesData[:prod(gridSize), 0].reshape(gridSize)
goodExample = np.concatenate((goodExample, goodExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
goodExample = np.concatenate((goodExample, goodExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(goodExample, showColorbar=True)
gwv.showCustomGraph(goodExample*reshapedWeights, showColorbar=True)
print "good example weighted distance", np.sum(goodExample*reshapedWeights)

badExample = badExamplesData[:prod(gridSize), 0].reshape(gridSize)
badExample = np.concatenate((badExample, badExamplesData[prod(gridSize):2*prod(gridSize), 0].reshape(gridSize)), axis=-1)
badExample = np.concatenate((badExample, badExamplesData[2*prod(gridSize):3*prod(gridSize), 0].reshape(gridSize)), axis=-1)
gwv.showCustomGraph(badExample, showColorbar=True)
gwv.showCustomGraph(badExample*reshapedWeights, showColorbar=True)
print "bad example weighted distance", np.sum(badExample*reshapedWeights)

# <codecell>

sigma = 1.0
mu = 3.0
samples = np.random.multivariate_normal(np.array([mu, mu]), np.array([[sigma, 0.0], [0.0, sigma]]), 50000)
figure(); scatter(samples[:, 0], samples[:, 1], c=sp.stats.multivariate_normal.pdf(samples, np.array([mu, mu]), np.array([[sigma, 0.0], [0.0, sigma]])))
# xlim([-10, 10])
# ylim([-10, 10])
figure(); plot(np.arange(-6, 6, 0.1), sp.stats.multivariate_normal.pdf(np.arange(-6, 6, 0.1), np.array([mu]), np.array([sigma])))

# <codecell>

## find "how much left-movement there is" between two frames (low numbers means more right movement)
frame1 = np.array(Image.open("/media/ilisescu/Data1/PhD/data/splashes_water/frame-00165.png"))
frame2 = np.array(Image.open("/media/ilisescu/Data1/PhD/data/splashes_water/frame-00167.png"))
imageSize = frame1.shape[0:2]
patchSize = np.array([32, 32])
spacing = 16.0 ## how far away a patch is from the previous one

movementSigma = 4.0
numSamples = 500
patchYs = np.arange(4.0*movementSigma, imageSize[0]-patchSize[0]-4.0*movementSigma, spacing)
patchYs = patchYs.reshape((1, len(patchYs)))
patchXs = np.arange(4.0*movementSigma, imageSize[1]-patchSize[1]-4.0*movementSigma, spacing)
patchXs = patchXs.reshape((1, len(patchXs)))
patchLocations = np.array(np.concatenate((patchYs.repeat(len(patchXs.T)),
                                          patchXs.repeat(len(patchYs.T), axis=0).flatten())).reshape((2, len(patchXs.T)*len(patchYs.T))), int)

print patchLocations.shape

# figure(); imshow(frame1)
scores = np.zeros(patchLocations.shape[-1])
for i, loc in enumerate(patchLocations.T) : ## [:, [0, 1282, 2967]]
#     print loc
    ## changing the x-mean by - movementSigma as I want to go left
    mean = np.array([loc[1]-2.0*movementSigma, loc[0]])
    covariance = np.array([[movementSigma, 0.0], [0.0, movementSigma/10.0]])
    ## samples contains (x, y) coords
    samples = np.random.multivariate_normal(mean, covariance, numSamples)
    weights = sp.stats.multivariate_normal.pdf(samples, mean, covariance)
    weights /= np.sum(weights)
#     print samples
    ## displacedLocs contains (row, col) coords
    displacedLocs = np.array(np.round(samples).T[::-1], int)
    validLocs = np.negative(np.any([np.any(displacedLocs < 0, axis=0), displacedLocs[0, :] >= imageSize[0]-patchSize[0], displacedLocs[1, :] >= imageSize[1]-patchSize[1]], axis=0))
    frame2Patches = [frame2[l[0]:l[0]+patchSize[0], l[1]:l[1]+patchSize[1]] for l in displacedLocs[:, validLocs].T]
    frame1Patch = frame1[loc[0]:loc[0]+patchSize[0], loc[1]:loc[1]+patchSize[1], :].reshape((1, patchSize[0], patchSize[0], frame1.shape[-1])).repeat(len(frame2Patches), axis=0)
    scores[i] = np.sum(np.sum(np.sqrt((frame1Patch.reshape((numSamples, np.prod(patchSize)*frame1.shape[-1]))/255.0-
                                       np.array(frame2Patches).reshape((numSamples, np.prod(patchSize)*frame1.shape[-1]))/255.0)**2), axis=-1)*weights)

    sys.stdout.write('\r' + "Done patch " + np.string_(i) + " of " + np.string_(patchLocations.shape[-1]))
    sys.stdout.flush()
#     print np.all(validLocs)
#     print displacedLocs[:, np.negative(validLocs)]
#     figure(); imshow(frame1)
#     scatter(samples[:, 0], samples[:, 1])#, c=sp.stats.multivariate_normal.pdf(samples, np.array([mu, mu]), np.array([[sigma, 0.0], [0.0, sigma]])))
#     scatter(loc[1], loc[0], c='r')
#     xlim([0, imageSize[1]])
#     ylim([imageSize[0], 0])
#     figure(); imshow(frame1[loc[0]:loc[0]+patchSize[0], loc[1]:loc[1]+patchSize[1], :])

# <codecell>

figure(); imshow(frame1)
scatter(samples[:, 0], samples[:, 1])#, c=sp.stats.multivariate_normal.pdf(samples, np.array([mu, mu]), np.array([[sigma, 0.0], [0.0, sigma]])))
scatter(loc[1], loc[0], c='r')
xlim([0, imageSize[1]])
ylim([imageSize[0], 0])
figure(); imshow(frame1[loc[0]:loc[0]+patchSize[0], loc[1]:loc[1]+patchSize[1], :])

# <codecell>

# leftMovement = np.copy(scores)
# rightMovement = np.copy(scores)
bob = np.zeros(imageSize)
bob[patchLocations[0, :], patchLocations[1, :]] = scores
gwv.showCustomGraph(bob)

# <codecell>

print np.sum(leftMovement), np.sum(rightMovement)

# <codecell>

print displacedLocs
patches = [frame2[l[0]:l[0]+patchSize[0], l[1]:l[1]+patchSize[1]] for l in displacedLocs[:, validLocs].T]
# print np.array(patches).shape
figure(); imshow(patches[0])
figure(); imshow(patches[200])
print displacedLocs[:, 0]
print displacedLocs[:, 200]
figure(); imshow(frame1[loc[0]:loc[0]+patchSize[0], loc[1]:loc[1]+patchSize[1], :])
frame1Patch = frame1[loc[0]:loc[0]+patchSize[0], loc[1]:loc[1]+patchSize[1], :].reshape((1, patchSize[0], patchSize[0], frame1.shape[-1])).repeat(len(patches), axis=0)

# <codecell>

print np.min(samples)
tmp = np.random.randint(-2, 33, (2, 10))
print tmp
print np.any(tmp < 0, axis=0)
print tmp[0, :] >= 32
print tmp[1, :] >= 30
print np.negative(np.any([np.any(tmp < 0, axis=0), tmp[0, :] >= 32, tmp[1, :] >= 30], axis=0))

# <codecell>

idxs = np.mgrid[0:np.prod(gridSize)*3, 0:np.prod(gridSize)*3].reshape((2, (np.prod(gridSize)*3)**2))
weightDists = np.zeros((3*np.prod(gridSize), 3*np.prod(gridSize)))
weightDists[idxs[0, :], idxs[1, :]] = np.sqrt((sio.loadmat(phiSaveLoc)['phi_MAP'][idxs[0, :]]-sio.loadmat(phiSaveLoc)['phi_MAP'][idxs[1, :]])**2).flatten()
gwv.showCustomGraph(weightDists)
# noSigmoidWeights = np.copy(sio.loadmat(phiSaveLoc)['phi_MAP'])
# noSigmoidWeightDists = np.copy(weightDists)

# <codecell>

gwv.showCustomGraph(weightDists[np.prod(gridSize):2*np.prod(gridSize), np.prod(gridSize):2*np.prod(gridSize)], showColorbar=True)
gwv.showCustomGraph(noSigmoidWeightDists[np.prod(gridSize):2*np.prod(gridSize), np.prod(gridSize):2*np.prod(gridSize)], showColorbar=True)

# <codecell>

bob = np.random.randint(-100, 100, 300)
figure(); plot(bob)
figure(); plot(1.0/(1.0 + np.exp(-bob)))
figure(); plot(1.0/(1.0 + np.exp(-5.0*bob)))
figure(); plot(1.0/(1.0 + np.exp(-0.2*bob)))

# <codecell>

# fittedPhiUsingPsi = sio.loadmat(phiSaveLoc)['phi_MAP']
# print glob.glob(dataPath+dataSet+"weightsUsingPsi*.npy")
fittedPhiUsingPsi = np.load(glob.glob(dataPath+dataSet+"weightsUsingPsi*.npy")[0])

# <codecell>

# print np.sum(fittedPhiUsingPsi-sio.loadmat(phiSaveLoc)['phi_MAP'])
print fittedPhiUsingPsi
print sio.loadmat(phiSaveLoc)['phi_MAP']
print fittedPhiUsingPsi/sio.loadmat(phiSaveLoc)['phi_MAP']
print np.mean(fittedPhiUsingPsi/sio.loadmat(phiSaveLoc)['phi_MAP'])
print np.mean(np.sqrt((fittedPhiUsingPsi/sio.loadmat(phiSaveLoc)['phi_MAP']-np.mean(fittedPhiUsingPsi/sio.loadmat(phiSaveLoc)['phi_MAP']))**2))

# <codecell>

np.save(dataPath+dataSet+"weightsUsingPsi_"+np.string_(numBadExamples)+"randomEg_"+np.string_(resizeRatio)+"scale.npy", fittedPhiUsingPsi)
print dataPath+dataSet+"weightsUsingPsi_"+np.string_(numBadExamples)+"randomEg_"+np.string_(resizeRatio)+"scale.npy"

# <codecell>

## test to see the distance assigned to the validation pairs
# goodValidationData = (imagesGrayData[:, goodValidationExamples[0, :]]-imagesGrayData[:, goodValidationExamples[1, :]])**np.float32(2)
# badValidationData = (imagesGrayData[:, badValidationExamples[0, :]]-imagesGrayData[:, badValidationExamples[1, :]])**np.float32(2)
print goodValidationData.shape
print badValidationData.shape
figure()
plot(np.sqrt(np.dot(goodValidationData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="g", label="Good validation")
plot(np.sqrt(np.dot(goodExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="y", label="Good training")
plot(np.sqrt(np.dot(badValidationData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="b", label="Bad validation")
plot(np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

figure()
plot(np.sqrt(np.dot(goodValidationData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="g", label="Good validation")
plot(np.sqrt(np.dot(goodExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="y", label="Good training")
plot(np.sqrt(np.dot(badValidationData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="b", label="Bad validation")
plot(np.sqrt(np.dot(badExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

# <codecell>

# img1Name = "frame-00522.png"
# img1Name = "frame-01022.png"
# img2Name = "frame-01023.png"

## good validated
img1Name = "frame-00929.png"
img2Name = "frame-00951.png"

## bad validated
# img1Name = "frame-00331.png"
# img2Name = "frame-00106.png"

# img1 = cv2.imread(dataPath+dataSet+img1Name,0) # queryImage
# img2 = cv2.imread(dataPath+dataSet+img2Name,0) # trainImage
img1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in xrange(len(matches))]

# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
        
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

# figure(); imshow(img3,)

# <codecell>

## Need to draw only good matches, so create a mask
## good matches here are chosen based on how unique they are, i.e. how much worse (the threshold below) 
## is the match with the second best neighbour
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.72*n.distance:
        matchesMask[i]=[1,0]
# print matchesMask

# <codecell>

## visualize
# figure(); imshow(cv2.cvtColor(np.concatenate((cv2.imread(dataPath+dataSet+img1Name), 
#                                               cv2.imread(dataPath+dataSet+img2Name)), axis=1), cv2.COLOR_BGR2RGB))
figure(); imshow(np.concatenate((frame1, frame2), axis=1))

for i, mask in enumerate(matchesMask) :
    if mask[0] == 1 :
#         print i, kp1[i].pt, matches[i][0].trainIdx
        matchIdx = matches[i][0].trainIdx
        plot([kp1[i].pt[0], kp2[matchIdx].pt[0]+img1.shape[1]], [kp1[i].pt[1], kp2[matchIdx].pt[1]])

## alternative vis
# figure(); imshow(np.array(cv2.cvtColor(cv2.imread(dataPath+dataSet+img1Name), cv2.COLOR_BGR2RGB)*.5 +
#                           cv2.cvtColor(cv2.imread(dataPath+dataSet+img2Name), cv2.COLOR_BGR2RGB)*.5, dtype=uint8))
figure(); imshow(np.array(frame1*0.5 + frame2*0.5, dtype=uint8))

dist = 0.0
numPoints = 0
for i, mask in enumerate(matchesMask) :
    if mask[0] == 1 :
#         print i, kp1[i].pt, matches[i][0].trainIdx
        matchIdx = matches[i][0].trainIdx
        plot([kp1[i].pt[0], kp2[matchIdx].pt[0]], [kp1[i].pt[1], kp2[matchIdx].pt[1]])
        scatter(kp1[i].pt[0], kp1[i].pt[1], marker='.')
        dist += np.linalg.norm(np.array(kp1[i].pt) - np.array(kp2[matchIdx].pt))
        numPoints += 1
print dist/numPoints, numPoints

# <codecell>

print additionalGoodPairsIdxs
print goodPairsIdxs[:, -len(additionalGoodPairsIdxs.T):]

# <codecell>

gwv.showCustomGraph(np.average(goodExamplesBAK[np.prod(gridSize):2*np.prod(gridSize), goodPairsToUse], axis=1).reshape(gridSize))
gwv.showCustomGraph(np.average(badExamplesBAK[np.prod(gridSize):2*np.prod(gridSize), badPairsToUse], axis=1).reshape(gridSize))

# <codecell>

gwv.showCustomGraph(np.average(goodExamplesData[:np.prod(gridSize), goodPairsToUse], axis=1).reshape(gridSize))
gwv.showCustomGraph(np.var(goodExamplesData[:np.prod(gridSize), goodPairsToUse], axis=1).reshape(gridSize))
gwv.showCustomGraph(np.average(badExamplesData[:np.prod(gridSize), badPairsToUse], axis=1).reshape(gridSize))
gwv.showCustomGraph(np.var(badExamplesData[:np.prod(gridSize), badPairsToUse], axis=1).reshape(gridSize))
gwv.showCustomGraph(sio.loadmat(phiSaveLoc)['phi_MAP'][:np.prod(gridSize)].reshape(gridSize))
# gwv.showCustomGraph(sio.loadmat(phiSaveLoc)['phi_MAP'].reshape(gridSize)*np.average(X[:np.prod(gridSize), :], axis=1).reshape(gridSize))
# gwv.showCustomGraph(np.average(X[np.prod(gridSize):2*np.prod(gridSize), :], axis=1).reshape(gridSize))
# gwv.showCustomGraph(np.average(X[2*np.prod(gridSize):, :], axis=1).reshape(gridSize))

# <codecell>

print sio.loadmat(phiSaveLoc)['phi_MAP']

# <codecell>

figure(); plot(sio.loadmat(phiSaveLoc)['phi_MAP'])
# gwv.showCustomGraph(sio.loadmat(phiSaveLoc)['phi_MAP'][:16].reshape((4, 4)))

# <codecell>

## plot good vs bad examples before and after fitting psi
figure("Re-weighted")
# additionalGoodPairsIdxs = np.empty(0)
numGood, numBad = [len(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T), len(badExamplesData.T)]
plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="y", label="Good training")
plot(arange(numGood, numGood+numBad), np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="r", label="Bad training")
print "good"
print np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, sio.loadmat(phiSaveLoc)['phi_MAP']))
print "bad"
print np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP']))
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

figure("L2")
plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="y", label="Good training")
plot(arange(numGood, numGood+numBad), np.sqrt(np.dot(badExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

# <codecell>

## plot good vs bad examples before and after fitting psi
figure("Re-weighted")
# additionalGoodPairsIdxs = np.empty(0)
numGood, numBad = [len(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T), len(badExamplesData.T)]
plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="y", label="Good training")
plot(arange(numGood, numGood+numBad), np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="r", label="Bad training")
print "good"
print np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, sio.loadmat(phiSaveLoc)['phi_MAP']))
print "bad"
print np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP']))
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

figure("L2")
plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="y", label="Good training")
plot(arange(numGood, numGood+numBad), np.sqrt(np.dot(badExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

# <codecell>

print additionalGoodPairsIdxs.T
print badPairsIdxs.T

# <codecell>

figure(); imshow(imagesGrayData[:, 928].reshape((720, 1280)))
figure(); imshow(imagesGrayData[:, 953].reshape((720, 1280)))
figure(); imshow(imagesGrayData[:, 928].reshape((720, 1280)))
figure(); imshow(imagesGrayData[:, 950].reshape((720, 1280)))

# <codecell>

## plot good vs bad examples before and after fitting psi
figure("Re-weighted")
# numGood, numBad = [len(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T), len(badExamplesData.T)]
# plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="y", label="Good training")
numGood, numBad = [len(goodExamplesData.T), len(badExamplesData.T)]
plot(np.sqrt(np.dot(goodExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="y", label="Good training")
plot(arange(numGood, numGood+numBad), np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="r", label="Bad training")
print "good"
print np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, sio.loadmat(phiSaveLoc)['phi_MAP']))
print "bad"
print np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP']))
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

figure("L2")
# plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="y", label="Good training")
plot(np.sqrt(np.dot(goodExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="y", label="Good training")
plot(arange(numGood, numGood+numBad), np.sqrt(np.dot(badExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

# <codecell>

print X.shape
print w.shape
print len(list(w))

# <codecell>

testExamples = [1213, 1215, 1217, 1219, 1221, 1223, 1225, 1227, 1229, 1231, 1232]
trainingExamples = np.delete(np.arange(X.shape[1]),
                             np.concatenate((np.arange(goodExamplesData.shape[1]-len(additionalGoodPairsIdxs.T)),
                                             testExamples)))
print testExamples
print trainingExamples

# <codecell>

tic = time.time()
classifier = ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=4, verbose=4)
classifier.fit(list(X[:, trainingExamples].T), w[trainingExamples, 0])
print time.time()-tic

# <codecell>

# figure("Random Forests")
figure("Random Forests - reduced examples")
dists = classifier.predict(list(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T))
dists = np.concatenate((dists, classifier.predict(list(badExamplesData.T))))
plot(dists[:numGood], color="y", label="Good training")
plot(arange(numGood, numGood+numBad), dists[numGood:], color="r", label="Bad training")
gca().autoscale(False)
scatter(np.array(testExamples)-len(goodExamplesData.T)+len(additionalGoodPairsIdxs.T), dists[np.array(testExamples)-len(goodExamplesData.T)+len(additionalGoodPairsIdxs.T)])
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

# <codecell>

figure("Random Forests - timeline pairs")
plot(classifier.predict(list(goodExamplesData[:, :-len(additionalGoodPairsIdxs.T)].T)), color="g")

# <codecell>

### compute l2 distance matrix between labelled training examples
featDists = np.ones((numGood+numBad, numGood+numBad))
for featI, i in zip(np.concatenate((goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, badExamplesData.T), axis=0), xrange(numGood+numBad)) :
    for featJ, j in zip(np.concatenate((goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, badExamplesData.T), axis=0), xrange(numGood+numBad)) :
        featDists[i, j] = np.sqrt(np.sum((featI-featJ)**2))
gwv.showCustomGraph(featDists)
print numGood, numBad

# <codecell>

print np.sum(goodExamplesData[:, -len(additionalGoodPairsIdxs.T)+2]-badExamplesData[:, -4])

# <codecell>

figure(); plot(fittedPhiUsingPsi)
# print phiSaveLoc
resizeRatio = 0.75
resizedImageSize = cv2.resize(cv2.imread(framePaths[0]), (0, 0), fx=resizeRatio, fy=resizeRatio).shape[0:2]
reshapedWeights = fittedPhiUsingPsi.reshape(resizedImageSize)
gwv.showCustomGraph(log(reshapedWeights), showColorbar=True)

# <codecell>

# print fittedPhiUsingPsi.shape, resizedImageSize
# gwv.showCustomGraph(X[:, -1].reshape(resizedImageSize)*reshapedWeights)
# gwv.showCustomGraph(X[:, -1].reshape(resizedImageSize))

## resize training examples into image shape
# resizedTraining = X.reshape((resizedImageSize[0], resizedImageSize[1], X.shape[-1]))
## weight each example by the fitted weights
# weightedTraining = resizedTraining*reshapedWeights.reshape((resizedImageSize[0], resizedImageSize[1], 1)).repeat(X.shape[-1], axis=-1)
## visualize average of the weighted examples and see how good and bad examples get weighted
# gwv.showCustomGraph(np.mean(weightedTraining[:, :, 1207:], axis=-1), showColorbar=True)
gwv.showCustomGraph(np.sum(resizedTraining[:, :, 1207:], axis=-1), showColorbar=True)
# gwv.showCustomGraph(weightedTraining[:, :, -800])

# <codecell>

# validatedJumps = np.load(dataPath+dataSet+"validatedJumps.npy")
# gwv.showCustomGraph(validatedJumps)

good = np.argwhere(validatedJumps == 1).T
bad = np.argwhere(validatedJumps == 0).T

if doRGB :
    resizedImageSize = cv2.resize(cv2.imread(framePaths[0]), (0, 0), fx=resizeRatio, fy=resizeRatio).shape
else :
    resizedImageSize = cv2.resize(cv2.imread(framePaths[0]), (0, 0), fx=resizeRatio, fy=resizeRatio).shape[0:2]
fittedWeights = sio.loadmat(phiSaveLoc)['phi_MAP'].reshape(resizedImageSize)

visPair = bad[:, -1]
print visPair
# figure(); imshow(imagesGrayData[:, visPair[0]].reshape(resizedImageSize))
# figure(); imshow(imagesGrayData[:, visPair[1]].reshape(resizedImageSize))
if doRGB :
    squaredDiff = (imagesRGBData[:, pair[0]].reshape(resizedImageSize)-imagesRGBData[:, pair[1]].reshape(resizedImageSize))**2
else :
    squaredDiff = (imagesGrayData[:, visPair[0]].reshape(resizedImageSize)-imagesGrayData[:, visPair[1]].reshape(resizedImageSize))**2

if doRGB :
    figure('Bad Pair '+np.string_(np.sum(squaredDiff))); imshow(np.array(squaredDiff*np.max(squaredDiff)*255, dtype=uint8))
    figure('Bad Pair Weighted '+np.string_(np.sum(squaredDiff*fittedWeights))); imshow(np.array(squaredDiff*fittedWeights*np.max(squaredDiff*fittedWeights)*255, dtype=uint8))
else :
    gwv.showCustomGraph(squaredDiff, title='Bad Pair '+np.string_(np.sum(squaredDiff)))
    gwv.showCustomGraph(squaredDiff*fittedWeights, 
                        title='Bad Pair Weighted '+np.string_(np.sum(squaredDiff*fittedWeights)))

visPair = good[:, -1]
print visPair
# figure(); imshow(imagesGrayData[:, visPair[0]].reshape(resizedImageSize))
# figure(); imshow(imagesGrayData[:, visPair[1]].reshape(resizedImageSize))
if doRGB :
    squaredDiff = (imagesRGBData[:, visPair[0]].reshape(resizedImageSize)-imagesRGBData[:, visPair[1]].reshape(resizedImageSize))**2
else :
    squaredDiff = (imagesGrayData[:, visPair[0]].reshape(resizedImageSize)-imagesGrayData[:, visPair[1]].reshape(resizedImageSize))**2
    
if doRGB :
    figure('Good Pair '+np.string_(np.sum(squaredDiff))); imshow(np.array(squaredDiff*np.max(squaredDiff)*255, dtype=uint8))
    figure('Good Pair Weighted '+np.string_(np.sum(squaredDiff*fittedWeights))); imshow(np.array(squaredDiff*fittedWeights*np.max(squaredDiff*fittedWeights)*255, dtype=uint8))
else :
    gwv.showCustomGraph(squaredDiff, title='Good Pair '+np.string_(np.sum(squaredDiff)))
    gwv.showCustomGraph(squaredDiff*fittedWeights, 
                        title='Good Pair Weighted '+np.string_(np.sum(squaredDiff*fittedWeights)))

# <codecell>

goodWeighted = []
badWeighted = []
goodL2 = []
badL2 = []
fittedWeights = np.ndarray.flatten(sio.loadmat(phiSaveLoc)['phi_MAP'])
for pair in good.T :
    if doHogs :
        squaredDiff = (hogFeats[pair[0], :]-hogFeats[pair[1], :])**2
    else :
        if doRGB :
            squaredDiff = (imagesRGBData[:, pair[0]]-imagesRGBData[:, pair[1]])**2
        else :
            squaredDiff = (imagesGrayData[:, pair[0]]-imagesGrayData[:, pair[1]])**2
    print pair, np.sqrt(np.sum(squaredDiff)), np.sqrt(np.sum(squaredDiff*fittedWeights))
    goodWeighted.append(np.sqrt(np.sum(squaredDiff*fittedWeights)))
    goodL2.append(np.sqrt(np.sum(squaredDiff)))

print 
for pair in bad.T :
    if doHogs :
        squaredDiff = (hogFeats[pair[0], :]-hogFeats[pair[1], :])**2
    else :
        if doRGB :
            squaredDiff = (imagesRGBData[:, pair[0]]-imagesRGBData[:, pair[1]])**2
        else :
            squaredDiff = (imagesGrayData[:, pair[0]]-imagesGrayData[:, pair[1]])**2
    print pair, np.sqrt(np.sum(squaredDiff)), np.sqrt(np.sum(squaredDiff*fittedWeights))
    badWeighted.append(np.sqrt(np.sum(squaredDiff*fittedWeights)))
    badL2.append(np.sqrt(np.sum(squaredDiff)))
    
## plot good vs bad examples before and after fitting psi
figure()
plot(goodWeighted, color="y", label="Good training")
plot(badWeighted, color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

figure()
plot(goodL2, color="y", label="Good training")
plot(badL2, color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

# <codecell>

[927 950] 27.2093 2.14080814411
[927 951] 28.1342 2.17981672858
[928 950] 29.7045 2.18269318793
[928 951] 28.4334 2.18061420331
[929 951] 31.3144 2.24727272516
[929 952] 29.8655 2.23011209716
[930 952] 33.0149 2.36626978265
[953 927] 31.914 2.28318629996
[974 945] 26.8204 2.25359033979

[126   8] 30.71 3.56370817368
[127   8] 29.2642 3.52514847033
[127   9] 30.4547 3.54506750138
[127  10] 32.5952 3.61926665286
[128   9] 29.0482 3.51348336548
[128  10] 30.7595 3.56129860681
[129  10] 29.1621 3.49770694837
[130  11] 29.6627 3.47948731979
[330 105] 30.3913 3.27191115629
[330 106] 32.0219 3.29588710884
[928 949] 30.8524 2.20138769731
[928 953] 30.0233 2.24931922881
[929 942] 40.126 2.43132457755
[929 950] 32.6549 2.25081500298
[946 922] 26.5016 2.09064952641
[949 925] 26.4918 2.10779238819
[950 928] 29.7045 2.18269318793
[ 959 1029] 35.1725 2.51904705615
[960 941] 34.9789 2.36658488158
[960 950] 35.2708 2.28535194038

# <codecell>

print np.sqrt(np.dot(badExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))
figure(); imshow((imagesGrayData[:, bad[0, -1]].reshape(resizedImageSize)-imagesGrayData[:, bad[1, -1]].reshape(resizedImageSize))**2)
figure(); imshow(badExamplesData[:, -1].reshape(resizedImageSize))

# <codecell>

print badPairsIdxs

# <codecell>

for pair in good.T :
    squaredDiff = (imagesGrayData[:, pair[0]].reshape(resizedImageSize)-imagesGrayData[:, pair[1]].reshape(resizedImageSize))**2
    print pair, np.sum(squaredDiff), np.sum(squaredDiff*fittedWeights)

print 
for pair in bad.T :
    squaredDiff = (imagesGrayData[:, pair[0]].reshape(resizedImageSize)-imagesGrayData[:, pair[1]].reshape(resizedImageSize))**2
    print pair, np.sum(squaredDiff), np.sum(squaredDiff*fittedWeights)

# <codecell>

for pair in good.T :
    squaredDiff = (imagesGrayData[:, pair[0]].reshape(resizedImageSize)-imagesGrayData[:, pair[1]].reshape(resizedImageSize))**2
    print pair, np.sum(squaredDiff), np.sum(squaredDiff*fittedWeights)

print 
for pair in bad.T :
    squaredDiff = (imagesGrayData[:, pair[0]].reshape(resizedImageSize)-imagesGrayData[:, pair[1]].reshape(resizedImageSize))**2
    print pair, np.sum(squaredDiff), np.sum(squaredDiff*fittedWeights)

# <codecell>

gwv.showCustomGraph((imagesGrayData[:, 926].reshape(resizedImageSize)-imagesGrayData[:, 927].reshape(resizedImageSize))**2)
gwv.showCustomGraph((imagesGrayData[:, 949].reshape(resizedImageSize)-imagesGrayData[:, 950].reshape(resizedImageSize))**2)

gwv.showCustomGraph((imagesGrayData[:, 958].reshape(resizedImageSize)-imagesGrayData[:, 959].reshape(resizedImageSize))**2)
gwv.showCustomGraph((imagesGrayData[:, 1028].reshape(resizedImageSize)-imagesGrayData[:, 1029].reshape(resizedImageSize))**2)

# <codecell>

phiUsingPsiFromUserLabels = np.copy(fittedPhiUsingPsi)
print phiUsingPsiFromUserLabels

# <codecell>

print fittedPhiUsingPsi

# <codecell>

## get indices of all pairs to compute
allPairsImageIndices = []
for i in xrange(numFrames) :
    for j in xrange(i+1, numFrames) :
        allPairsImageIndices.append(np.array([i, j]))
        
allPairsImageIndices = np.array(allPairsImageIndices)
    
## load fitted phi and compute the predicted w given this phi
# fittedPhi = sio.loadmat(phiSaveLoc)['phi_MAP']
# fittedPhi = sio.loadmat("../MATLAB/phi_MAPtrust.mat")['phi_MAP']
# fittedPhi = np.copy(fittedPhiUsingPsi)
fittedPhi = np.ones_like(fittedPhiUsingPsi, dtype=float32)
print fittedPhi.shape
splitSize = 1000
# doHogs = False
if doHogs :
    predictedW = np.dot(np.array(allPairsHogs), fittedPhi)
#     predictedW = np.dot(np.array(allPairsHogs), np.ones_like(fittedPhi))
else :
    predictedW = np.empty((0), dtype=float32)
    for i, j in zip(xrange(len(allPairsImageIndices)/splitSize+1), xrange(splitSize)) :
        if doRGB :
            allPairsImageData = np.zeros((splitSize, imagesRGBData.shape[0]), dtype=float32)
        else :
            allPairsImageData = np.zeros((splitSize, imagesGrayData.shape[0]), dtype=float32)
#         for pair in allPairsImageIndices[i*splitSize:(i+1)*splitSize] :
#             if doRGB :
#                 allPairsImageData[j, :] = (imagesRGBData[:, pair[0]]-imagesRGBData[:, pair[1]])**np.float32(2)
#             else :
#                 allPairsImageData[j, :] = (imagesGrayData[:, pair[0]]-imagesGrayData[:, pair[1]])**np.float32(2)

        pairs = allPairsImageIndices[i*splitSize:(i+1)*splitSize, :]
        if doRGB :
            allPairsImageData = ((imagesRGBData[:, pairs[:, 0]]-imagesRGBData[:, pairs[:, 1]])**np.float32(2)).T
        else :
            allPairsImageData = ((imagesGrayData[:, pairs[:, 0]]-imagesGrayData[:, pairs[:, 1]])**np.float32(2)).T
            
        predictedW = np.concatenate((predictedW, np.ndarray.flatten(np.sqrt(np.dot(allPairsImageData, fittedPhi)))))
#         predictedW = np.sqrt(np.dot(np.array(allPairsImageData), np.ones_like(fittedPhi, dtype=np.float32)))
            
        sys.stdout.write('\r' + "Done with split " + np.string_(i) + " of " + np.string_(len(allPairsImageIndices)/splitSize))
        sys.stdout.flush()

regressedDist = np.ones((numFrames, numFrames))
flatRegressedDist = list(np.copy(predictedW))
for i in xrange(numFrames-1) :
    regressedDist[i, i+1:] = flatRegressedDist[:numFrames-(i+1)]
    regressedDist[i+1:, i] = regressedDist[i, i+1:]
    del flatRegressedDist[:numFrames-(i+1)]

# <codecell>

print allPairsImageData.shape

# <codecell>

print diffs.shape

# <codecell>

print len(allPairsImageData)/1000+1
print len(allPairsImageData[397*1000:398*1000])

# <codecell>

# gwv.showCustomGraph(regressedDist)
gwv.showCustomGraph(np.load(dataPath+dataSet+"vanilla_distMat.npy"))

# <codecell>

randDist = np.copy(regressedDist)

# <codecell>

print fittedPhi

# <codecell>

# np.save(dataPath+dataSet+"psiWeighted_allOnesWeights_distMat.npy", regressedDist)
np.save(dataPath+dataSet+"psiWeighted_1000randomBadExamples_distMat.npy", regressedDist)
# np.save(dataPath+dataSet+"psiWeighted_userLabelledExamples_distMat.npy", regressedDist)

# <codecell>

np.log(1+np.exp(np.log(np.exp(np.ones(5))-1)))

# <codecell>

gwv.showCustomGraph(np.load(dataPath+dataSet+"l2_distMat.npy"))

# <codecell>

# gwv.showCustomGraph(regressedDist, title='pendulum - Regressed using phi')
gwv.showCustomGraph(regressedDist, title='pendulum - usingPsi')

# <codecell>

i = 0
tmp1 = np.sqrt(np.sum((imagesGrayData[:, i].reshape((np.prod(np.array(imageSize)*resizeRatio), 1))-imagesGrayData[:, i:])**2, axis=0))
print tmp1
tmp2 = imagesGrayData[:, i]-imagesGrayData[:, i+1]
figure(); plot(tmp2 - np.power(allPairsImageData[0], np.float32(0.5)))

# <codecell>

### use random forests instead of the phi thingy
# from sklearn import ensemble

tic = time.time()
regressor = ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=4, verbose=0)
regressor.fit(list(X.T), list(np.ndarray.flatten(w/np.max(w))))
print "regressor trained in", time.time()-tic; sys.stdout.flush()

# <codecell>

tic = time.time()
dists = regressor.predict(allPairsHogs)
print "distance regressed in", time.time()-tic; sys.stdout.flush()

# <codecell>

forestsDist = np.ones((numFrames, numFrames))
flatRegressedDist = list(np.copy(dists))
for i in xrange(numFrames-1) :
    forestsDist[i, i+1:] = flatRegressedDist[:numFrames-(i+1)]
    forestsDist[i+1:, i] = forestsDist[i, i+1:]
    del flatRegressedDist[:numFrames-(i+1)]

# <codecell>

gwv.showCustomGraph(forestsDist, title='Regressed by forests')

# <codecell>

gwv.showCustomGraph(np.load(dataPath+dataSet+"64adj10.avi_distanceMatrix.npy"))

# <codecell>

## save for matlab if necessary
sio.savemat("hogs_training_examples", {"X":X, "w":w})
# sio.savemat("hogs_training_examples_pairs", {"badExamples":badPairsIdxs, "goodExamples":goodPairsIdxs})
# sio.savemat("all_hogs_pair_feats", {"all_pairs_feats":np.array(allPairsHogs).T})
# sio.savemat("/media/ilisescu/Data1/imagesRGB_data", {"imagesRGBData":imagesRGBData})

# <codecell>

## now estimate phi based on training X and w

iterNum = 0
def printStats(xk) :
    global iterNum
    global phi0
    global derCallCount; derCallCount = 0
    global fCallCount; fCallCount = 0
    print 
    print iterNum, np.mean(np.abs(xk-phi0)); sys.stdout.flush()
    iterNum += 1

tic = time.time()
phi_MAP = optimize.fmin_ncg(negLogL_MAP, phi0, fprime=derNegLogL_MAP, callback=printStats,
                            args=(X, w, N)).reshape(phi0.shape)
print "optimized in", time.time()-tic; sys.stdout.flush()

# <codecell>

## use phi_MAP computed by matlab using downsampled RGB images (every 64th pixel only) to compute dists in the next bit
phi_MAP = sio.loadmat("../MATLAB/phi_MAP.mat")['phi_MAP']
print phi_MAP.shape

downsampledRGB = imagesRGBData.reshape((imagesRGBData.shape[0]/3, 3, 672))
downsampledRGB = downsampledRGB[arange(0, downsampledRGB.shape[0], 64), :, :]
downsampledRGB = downsampledRGB.reshape(downsampledRGB.shape[0]*3, 672)
print downsampledRGB.shape

# <codecell>

dists = np.empty((0, 1))
for i in arange(downsampledRGB.shape[-1]) :
    allPairsRGB = []
    for j in xrange(i+1, downsampledRGB.shape[-1]) :
        ## ABS DIST
#         allPairsHogs.append(np.sqrt((hogFeats[i, :]-hogFeats[j, :])**2))
        allPairsRGB.append((downsampledRGB[:, i]-downsampledRGB[:, j])**2)
    
#     newDists = 
    dists = np.concatenate((dists, np.dot(np.array(allPairsRGB), phi_MAP)))
    
    sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(downsampledRGB.shape[-1]))
    sys.stdout.flush()

# <codecell>

# dists = sio.loadmat("../MATLAB/predW.mat")['predW']
# dists = sio.loadmat("../MATLAB/l2W.mat")['l2W']
numFrames = downsampledRGB.shape[-1]
regressedDist = np.ones((numFrames, numFrames))
flatRegressedDist = list(np.copy(dists))
for i in xrange(numFrames-1) :
    regressedDist[i, i+1:] = flatRegressedDist[:numFrames-(i+1)]
    regressedDist[i+1:, i] = regressedDist[i, i+1:]
    del flatRegressedDist[:numFrames-(i+1)]

# <codecell>

gwv.showCustomGraph(regressedDist)

# <codecell>

np.save("/media/ilisescu/Data1/PhD/data/Videos/6489810.avi_learnedPhiRGBfeats.npy", regressedDist)

# <codecell>

# l2Dist = np.load("/media/ilisescu/Data1/PhD/data/Videos/6489810.avi_distanceMatrix.npy")
gwv.showCustomGraph(l2Dist)
gwv.showCustomGraph(vtu.filterDistanceMatrix(l2Dist, 4, False))

