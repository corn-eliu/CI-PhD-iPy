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


import cv2

import scipy.io as sio
import glob
import commands

from PIL import Image

import GraphWithValues as gwv
import VideoTexturesUtils as vtu

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
# dataSet = "eu_flag_ph_left/"
# dataSet = "ribbon2/"
dataSet = "candle1/segmentedAndCropped/"
framePaths = np.sort(glob.glob(dataPath + dataSet + "frame*.png"))
numFrames = len(framePaths)
print numFrames
imageSize = np.array(Image.open(framePaths[0])).shape[0:2]

# <codecell>

im = cv2.imread(framePaths[0])
figure(); imshow(im)

# <codecell>

### compute l2 dist for images of the splashes_water dataset
resizeRatio = 0.75#0.5
doRGB = False
useRange = False
featsRange = np.array([-2, -1, 0, 1, 2])
# featsRange = np.array([-1, 0, 1])
rangeResizeRatios = resizeRatio/2**np.abs(arange(-np.floor(len(featsRange)/2), np.floor(len(featsRange)/2)+1))
baseDimensionality = np.prod(np.round(np.array(imageSize)*resizeRatio))
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
imageFeatsSize = np.array(baseDimensionality/((resizeRatio/rangeResizeRatios)**2), dtype=int)
print imageFeatsSize
resizedImageSize = cv2.resize(cv2.imread(framePaths[0]), (0, 0), fx=rangeResizeRatios[1], fy=rangeResizeRatios[1], interpolation=cv2.INTER_AREA).shape[0:2]
figure(); imshow(imagesGrayData[imageFeatsSize[0]:np.sum(imageFeatsSize[0:2]), 100].reshape(resizedImageSize))
resizedImageSize = cv2.resize(cv2.imread(framePaths[0]), (0, 0), fx=rangeResizeRatios[0], fy=rangeResizeRatios[0], interpolation=cv2.INTER_AREA).shape[0:2]
figure(); imshow(imagesGrayData[:imageFeatsSize[0], 100].reshape(resizedImageSize))
figure(); imshow(imagesGrayData[np.sum(imageFeatsSize[0:2]):, 100].reshape(resizedImageSize))

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
# print hogFeats.shape

## get feats of subsequent frames
goodPairsIdxs = np.array([np.arange(numFrames-1, dtype=int), np.arange(1, numFrames, dtype=int)])
print goodPairsIdxs

useValidatedJumps = False

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
    print badPairsIdxs
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

doHogs = False
if doHogs :
    ## use hog feats as frame features
    ## ABS DIST
    # goodExamplesData = np.sqrt((hogFeats[goodPairsIdxs[0, :], :]-hogFeats[goodPairsIdxs[1, :], :])**2)
    goodExamplesData = (hogFeats[goodPairsIdxs[0, :], :]-hogFeats[goodPairsIdxs[1, :], :])**2
    print goodExamplesData.shape
    ## ABS DIST
    # badExamplesData = np.sqrt((hogFeats[badPairsIdxs[0, :], :]-hogFeats[badPairsIdxs[1, :], :])**2)
    badExamplesData = (hogFeats[badPairsIdxs[0, :], :]-hogFeats[badPairsIdxs[1, :], :])**2
    print badExamplesData.shape
    
    X = np.concatenate((goodExamplesData, badExamplesData)).T
    w = np.concatenate((np.zeros(len(goodExamplesData)), 10.0*np.ones(len(badExamplesData)))).reshape((X.shape[-1], 1))
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
    
    X = np.concatenate((goodExamplesData, badExamplesData), axis=1)
    w = np.concatenate((np.zeros(goodExamplesData.shape[-1]), 10.0*np.ones(badExamplesData.shape[-1]))).reshape((X.shape[-1], 1))
    N = X.shape[0]
    phi0 = np.ones((N, 1))
    sio.savemat(dataPath + dataSet + "trainingExamplesForImageData", {"X":X, "w":w})

print N, X.shape, w.shape, phi0.shape

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

# fittedPhi = sio.loadmat(phiSaveLoc)['phi_MAP']

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

print additionalGoodPairsIdxs
print goodPairsIdxs[:, -len(additionalGoodPairsIdxs.T):]

# <codecell>

## plot good vs bad examples before and after fitting psi
figure()
plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="y", label="Good training")
plot(np.sqrt(np.dot(badExamplesData.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

figure()
plot(np.sqrt(np.dot(goodExamplesData[:, -len(additionalGoodPairsIdxs.T):].T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="y", label="Good training")
plot(np.sqrt(np.dot(badExamplesData.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), color="r", label="Bad training")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

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

