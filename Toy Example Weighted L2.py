# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import sys
import scipy as sp
from scipy import optimize

import itertools

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

## make small toy example with known L2 distance for 3 examples
greyImage1 = np.ones((1, 2, 3), dtype=int)*127 + np.random.randint(-10, 10, 3)
greyImage1[:, 1, :] = np.random.randint(120, 134, 3)##(0, 255, 3)
# greyImage1[0, 1, 2] = 0 ## setting last dimension to 0 and see if that dimension is leveraged to get the good examples right

greyImage2 = np.ones((1, 2, 3), dtype=int)*127 + np.random.randint(-10, 10, 3)
greyImage2[:, 1, :] = np.random.randint(120, 134, 3)##(0, 255, 3)
# greyImage2[0, 1, 2] = 0

blackImage1 = np.zeros((1, 2, 3), dtype=int) + np.random.randint(0, 10, 3)
blackImage1[:, 1, :] = np.random.randint(120, 134, 3)##(0, 255, 3)
# blackImage1[0, 1, 2] = 127

blackImage2 = np.zeros((1, 2, 3), dtype=int) + np.random.randint(0, 10, 3)
blackImage2[:, 1, :] = np.random.randint(120, 134, 3)##(0, 255, 3)
# blackImage2[0, 1, 2] = 127

whiteImage1 = np.ones((1, 2, 3), dtype=int)*255 + np.random.randint(-10, 0, 3)
whiteImage1[:, 1, :] = np.random.randint(120, 134, 3)##(0, 255, 3)
# whiteImage1[0, 1, 2] = 255

whiteImage2 = np.ones((1, 2, 3), dtype=int)*255 + np.random.randint(-10, 0, 3)
whiteImage2[:, 1, :] = np.random.randint(120, 134, 3)##(0, 255, 3)
# whiteImage2[0, 1, 2] = 255

# X = np.array([(blackImage1.flatten()/255.0-blackImage1.flatten()/255.0)**2,
#               (whiteImage1.flatten()/255.0-whiteImage2.flatten()/255.0)**2,
#               (greyImage1.flatten()/255.0-greyImage2.flatten()/255.0)**2,
#               (greyImage1.flatten()/255.0-blackImage1.flatten()/255.0)**2,
#               (greyImage1.flatten()/255.0-blackImage2.flatten()/255.0)**2,
#               (greyImage2.flatten()/255.0-blackImage1.flatten()/255.0)**2,
#               (greyImage2.flatten()/255.0-blackImage2.flatten()/255.0)**2,
#               (greyImage1.flatten()/255.0-whiteImage1.flatten()/255.0)**2,
#               (greyImage1.flatten()/255.0-whiteImage2.flatten()/255.0)**2,
#               (greyImage2.flatten()/255.0-whiteImage1.flatten()/255.0)**2,
#               (greyImage2.flatten()/255.0-whiteImage2.flatten()/255.0)**2,
#               (whiteImage1.flatten()/255.0-blackImage1.flatten()/255.0)**2,
#               (whiteImage1.flatten()/255.0-blackImage2.flatten()/255.0)**2,
#               (whiteImage2.flatten()/255.0-blackImage1.flatten()/255.0)**2,
#               (whiteImage2.flatten()/255.0-blackImage2.flatten()/255.0)**2]).T
# w = np.array([[0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]]).T
X = np.array([(blackImage1.flatten()/255.0-blackImage1.flatten()/255.0)**2,
              (whiteImage1.flatten()/255.0-whiteImage2.flatten()/255.0)**2,
              (whiteImage1.flatten()/255.0-blackImage1.flatten()/255.0)**2]).T
w = np.array([[0.0, 0.0, 100.0]]).T

N = X.shape[0]
phi0 = np.ones((N, 1))
sio.savemat(dataPath + "trainingExamplesForSmallToyExample", {"X":X, "w":w})
print N, X.shape, w.shape, phi0.shape

## now call the matlab script to fit phi using psi
trainingExamplesLoc = dataPath + "trainingExamplesForSmallToyExample.mat"
phiSaveLoc = dataPath + "fittedPhiForSmallToyExampleUsingPsi.mat"

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

print sio.loadmat(phiSaveLoc)['phi_MAP']
toyDist = np.zeros((6, 6))
for imgI, i in zip([greyImage1.flatten(), greyImage2.flatten(), 
                    whiteImage1.flatten(), whiteImage2.flatten(), 
                    blackImage1.flatten(), blackImage2.flatten()], arange(6)):
    for imgJ, j in zip([greyImage1.flatten(), greyImage2.flatten(), 
                        whiteImage1.flatten(), whiteImage2.flatten(), 
                        blackImage1.flatten(), blackImage2.flatten()], arange(6)):
#         toyDist[i, j] = np.sqrt(np.dot(((imgI/255.0-imgJ/255.0)**2).reshape((1, 6)), sio.loadmat(phiSaveLoc)['phi_MAP']))
#         toyDist[i, j] = np.sqrt(np.dot(((imgI[0:3]/255.0-imgJ[0:3]/255.0)**2).reshape((1, 3)), sio.loadmat(phiSaveLoc)['phi_MAP'][0:3]))
        toyDist[i, j] = np.sqrt(np.dot(((imgI/255.0-imgJ/255.0)**2).reshape((1, 6)), np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))
#         toyDist[i, j] = np.sqrt(np.dot(((imgI[0:3]/255.0-imgJ[0:3]/255.0)**2).reshape((1, 3)), np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])[0:3]))
        
gwv.showCustomGraph(toyDist, "L2 distance for toy example with random pixel-best examples")

# <headingcell level=2>

# White vs Grey vs Black example

# <codecell>

## automatically generated toy example
## number of toy images for white, grey and black pixels
numToyImages = [50, 50, 50]
## make the 2D images with a white/grey/black pixel and a random pixel
toyImages = np.concatenate((np.array([np.ones(numToyImages[0])*255 + np.random.randint(-10, 0, numToyImages[0]), np.random.randint(0, 255, numToyImages[0])]),
                            np.array([np.ones(numToyImages[1])*127 + np.random.randint(-5, 5, numToyImages[1]), np.random.randint(0, 255, numToyImages[1])]),
                            np.array([np.zeros(numToyImages[2]) + np.random.randint(0, 10, numToyImages[2]), np.random.randint(0, 255, numToyImages[2])])), axis=-1)

# print toyImages

## generating all possible indices for good examples
goodExamplesIdxs = np.zeros((np.sum(numToyImages), np.sum(numToyImages)))
goodExamplesIdxs[0:numToyImages[0], 
                 0:numToyImages[0]] = np.triu(np.ones((numToyImages[0], numToyImages[0])), k=1) ## white vs white
goodExamplesIdxs[numToyImages[0]:np.sum(numToyImages[:2]), 
                 numToyImages[0]:np.sum(numToyImages[:2])] = np.triu(np.ones((numToyImages[1], numToyImages[1])), k=1) ## grey vs grey
goodExamplesIdxs[np.sum(numToyImages[:2]):np.sum(numToyImages), 
                 np.sum(numToyImages[:2]):np.sum(numToyImages)] = np.triu(np.ones((numToyImages[2], numToyImages[2])), k=1) ## black vs black
goodExamplesIdxs = np.argwhere(goodExamplesIdxs)
print goodExamplesIdxs.shape[0], "possible good examples"

## generating all possible indices for bad examples
badExamplesIdxs = np.zeros((np.sum(numToyImages), np.sum(numToyImages)))
# badExamplesIdxs[0:numToyImages[0], numToyImages[0]:np.sum(numToyImages[:2])] = 1 ## white vs grey
badExamplesIdxs[0:numToyImages[0], np.sum(numToyImages[:2]):] = 1 ## white vs black
# badExamplesIdxs[numToyImages[0]:np.sum(numToyImages[:2]), np.sum(numToyImages[:2]):] = 1 ## grey vs black
badExamplesIdxs = np.argwhere(badExamplesIdxs)
print badExamplesIdxs.shape[0], "possible bad examples"

randomGoodPairs = goodExamplesIdxs[np.random.choice(np.arange(len(goodExamplesIdxs)), size=2300, replace=False), :]
randomBadPairs = badExamplesIdxs[np.random.choice(np.arange(len(badExamplesIdxs)), size=2300, replace=False), :]

# <headingcell level=2>

# White to Black and Back Animation example

# <codecell>

animationLength = 20
numLoops = 10
numToyImages = animationLength*numLoops
## animation goes from 0 to 255 and back in animationLength frames
animFrameIdxs = np.repeat(np.concatenate((np.arange(0, animationLength/2+1),
                                          np.arange(0, animationLength/2+1)[1:-1][::-1])).reshape((1, animationLength)),
                          numLoops, axis=-0).flatten()
# toyImages = (animFrameIdxs*((255.0*2.0)/animationLength)).reshape((1, animationLength*numLoops))
# ## add noise to animation colors
# toyImages = np.clip(toyImages + np.random.randint(-5, 5, toyImages.shape[-1]), 0, 255)
# ## add random dimension
# toyImages = np.array(np.concatenate((toyImages, np.random.randint(0, 255, toyImages.shape[-1]).reshape((1, toyImages.shape[-1])))), int)
# print toyImages


desiredIdxs = np.zeros((animationLength*numLoops, animationLength*numLoops))

## good pairings are instances where the same color is show as stored in animFrameIdxs

## this allows for switching animation direction when reaching half-way mark
# desiredIdxs = np.array([animFrameIdxs == i for i in animFrameIdxs], dtype=int)*2
## this one doesn't
desiredIdxs = np.array([np.mod(np.arange(animationLength*numLoops),
                               animationLength) == i for i in np.mod(np.arange(animationLength*numLoops),
                                                                     animationLength)], dtype=int)*2

## bad pairings are instances where different colors are picked
turnDist = 2 ## distance from animation switch (i.e. when I switch going from white to black to going from black to white and viceversa)
desiredIdxs += cv2.erode(np.array(1.0 - np.array([animFrameIdxs == i for i in animFrameIdxs], dtype=int), np.uint8), np.ones((turnDist*2+1, turnDist*2+1), np.uint8))
# desiredIdxs += cv2.erode(np.array(1.0 - desiredIdxs/np.max(desiredIdxs), np.uint8), np.ones((turnDist*2+1, turnDist*2+1), np.uint8))

desiredIdxs = np.roll(desiredIdxs, 1, axis=-1)
desiredIdxs = np.triu(desiredIdxs, k=1)
gwv.showCustomGraph(desiredIdxs)

goodExamplesIdxs = np.argwhere(desiredIdxs==2)
badExamplesIdxs = np.argwhere(desiredIdxs==1)

print goodExamplesIdxs.shape
print badExamplesIdxs.shape


randomGoodPairs = goodExamplesIdxs[np.random.choice(np.arange(len(goodExamplesIdxs)), size=1000, replace=False), :]
randomBadPairs = badExamplesIdxs[np.random.choice(np.arange(len(badExamplesIdxs)), size=1000, replace=False), :]

# <codecell>

# toyImages = np.copy(toyImages[:2, :])
useAugment = True
augmentSymmetric = True
useTemporal = False

## computing feature vectures for all pairs
allPairsIdxs = np.argwhere(np.triu(np.ones((np.sum(numToyImages), np.sum(numToyImages)))))#, k=1))
allPairs = (toyImages[:, allPairsIdxs[:, 0]]/1.0-toyImages[:, allPairsIdxs[:, 1]]/1.0)**2
## add augmentation to x
if useAugment :
    if augmentSymmetric :
        print "using symmetric augmentation"
#         allPairs = np.concatenate((allPairs, 
#                                    (((255-toyImages[0, allPairsIdxs[:, 0]])/255.0-toyImages[0, allPairsIdxs[:, 1]]/255.0)**2+
#                                     ((255-toyImages[0, allPairsIdxs[:, 1]])/255.0-toyImages[0, allPairsIdxs[:, 0]]/255.0)**2).reshape((1, len(allPairsIdxs)))))
        allPairs = np.concatenate((allPairs,
                                   (((255-toyImages[:, allPairsIdxs[:, 0]])/1.0-toyImages[:, allPairsIdxs[:, 1]]/1.0)**2+
                                    ((255-toyImages[:, allPairsIdxs[:, 1]])/1.0-toyImages[:, allPairsIdxs[:, 0]]/1.0)**2).reshape((toyImages.shape[0], len(allPairsIdxs)))))
    else :
        print "using non-symmetric augmentation"
        allPairs = np.concatenate((allPairs, 
                                   ((255-toyImages[:, allPairsIdxs[:, 0]])/255.0-toyImages[:, allPairsIdxs[:, 1]]/255.0).reshape((toyImages.shape[0], len(allPairsIdxs)))**2))
        
if useTemporal :
    print "using temporal features augmentation"
    shiftedIdxs1 = allPairsIdxs[:, 0]+1
    np.place(shiftedIdxs1, shiftedIdxs1==numToyImages, -1)
    shiftedIdxs2 = allPairsIdxs[:, 1]+1
    np.place(shiftedIdxs2, shiftedIdxs2==numToyImages, -1)
    allPairs = np.concatenate((allPairs,
                               ((toyImages[0, allPairsIdxs[:, 0]]/255.0-toyImages[0, allPairsIdxs[:, 1]-1]/255.0)**2).reshape((1, len(allPairsIdxs))),
                               ((toyImages[0, shiftedIdxs1]/255.0-toyImages[0, allPairsIdxs[:, 1]]/255.0)**2).reshape((1, len(allPairsIdxs))),
                               ((toyImages[0, allPairsIdxs[:, 0]-1]/255.0-toyImages[0, allPairsIdxs[:, 1]-1]/255.0)**2).reshape((1, len(allPairsIdxs))),
                               ((toyImages[0, shiftedIdxs1]/255.0-toyImages[0, shiftedIdxs2]/255.0)**2).reshape((1, len(allPairsIdxs)))))
        
print allPairs.shape

# <codecell>

print np.max(allPairs[0:2, :])
print np.max(toyImages)

# <codecell>

print X.T
print w
print goodExamplesToUse
print badExamplesToUse
print toyImages[:, 42]
print toyImages[:, 117]

# <codecell>

## number of good and bad examples to use in training per try
numExamples = np.array([[1, 1],
                        [3, 3],
                        [9, 9],
                        [27, 27],
                        [81, 81],
                        [243, 243],
                        [729, 729],
                        [2187, 2187]])
# numExamples = numExamples[:-1, :]

## weights fitted for each combination of numbers of examples
fittedWeights = []
## sum of squared differences between regressed distances and given labels w
trainingErrors = []
## sum of squared differences between l2 distance between training pairs and the desired labels w
l2DistErrors = []
## distance matrices using the regressed weights
regressedDistanceMats = []


for currentTry in numExamples[0:] :
    print currentTry
    goodExamplesToUse = randomGoodPairs[:currentTry[0], :] # goodExamplesIdxs[np.random.choice(np.arange(len(goodExamplesIdxs)), size=currentTry[0], replace=False), :]
    badExamplesToUse = randomBadPairs[:currentTry[1], :] # badExamplesIdxs[np.random.choice(np.arange(len(badExamplesIdxs)), size=currentTry[1], replace=False), :]
#     print toyImages[0, goodExamplesToUse[:, 0]]
#     print toyImages[0, goodExamplesToUse[:, 1]]
    print np.min(np.sqrt((toyImages[0, goodExamplesToUse[:, 0]]-toyImages[0, goodExamplesToUse[:, 1]])**2)),
    print np.max(np.sqrt((toyImages[0, goodExamplesToUse[:, 0]]-toyImages[0, goodExamplesToUse[:, 1]])**2))
    print
#     print toyImages[0, badExamplesToUse[:, 0]]
#     print toyImages[0, badExamplesToUse[:, 1]]
    print np.min(np.sqrt((toyImages[0, badExamplesToUse[:, 0]]-toyImages[0, badExamplesToUse[:, 1]])**2)),
    print np.max(np.sqrt((toyImages[0, badExamplesToUse[:, 0]]-toyImages[0, badExamplesToUse[:, 1]])**2))
    
    goodExamplesData = (toyImages[:, goodExamplesToUse[:, 0]]/1.0-toyImages[:, goodExamplesToUse[:, 1]]/1.0)**2
    ## add augmentation to x
    if useAugment :
        if augmentSymmetric :
#             goodExamplesData = np.concatenate((goodExamplesData, 
#                                                (((255-toyImages[0, goodExamplesToUse[:, 0]])/255.0-toyImages[0, goodExamplesToUse[:, 1]]/255.0)**2+
#                                                 ((255-toyImages[0, goodExamplesToUse[:, 1]])/255.0-toyImages[0, goodExamplesToUse[:, 0]]/255.0)**2).reshape((1, len(goodExamplesToUse)))))
            goodExamplesData = np.concatenate((goodExamplesData, 
                                               (((255-toyImages[:, goodExamplesToUse[:, 0]])/1.0-toyImages[:, goodExamplesToUse[:, 1]]/1.0)**2+
                                                ((255-toyImages[:, goodExamplesToUse[:, 1]])/1.0-toyImages[:, goodExamplesToUse[:, 0]]/1.0)**2).reshape((toyImages.shape[0], len(goodExamplesToUse)))))

        else :
            goodExamplesData = np.concatenate((goodExamplesData, 
                                               ((255-toyImages[:, goodExamplesToUse[:, 0]])/255.0-toyImages[:, goodExamplesToUse[:, 1]]/255.0).reshape((toyImages.shape[0], len(goodExamplesToUse)))**2))
    
    if useTemporal :
        shiftedIdxs1 = goodExamplesToUse[:, 0]+1
        np.place(shiftedIdxs1, shiftedIdxs1==numToyImages, -1)
        shiftedIdxs2 = goodExamplesToUse[:, 1]+1
        np.place(shiftedIdxs2, shiftedIdxs2==numToyImages, -1)
        goodExamplesData = np.concatenate((goodExamplesData,
                                           ((toyImages[0, goodExamplesToUse[:, 0]]/255.0-toyImages[0, goodExamplesToUse[:, 1]-1]/255.0)**2).reshape((1, len(goodExamplesToUse))),
                                           ((toyImages[0, shiftedIdxs1]/255.0-toyImages[0, goodExamplesToUse[:, 1]]/255.0)**2).reshape((1, len(goodExamplesToUse))),
                                           ((toyImages[0, goodExamplesToUse[:, 0]-1]/255.0-toyImages[0, goodExamplesToUse[:, 1]-1]/255.0)**2).reshape((1, len(goodExamplesToUse))),
                                           ((toyImages[0, shiftedIdxs1]/255.0-toyImages[0, shiftedIdxs2]/255.0)**2).reshape((1, len(goodExamplesToUse)))))
    
    badExamplesData = (toyImages[:, badExamplesToUse[:, 0]]/1.0-toyImages[:, badExamplesToUse[:, 1]]/1.0)**2
    ## add augmentation to x
    if useAugment :
        if augmentSymmetric :
#             badExamplesData = np.concatenate((badExamplesData, 
#                                               (((255-toyImages[0, badExamplesToUse[:, 0]])/255.0-toyImages[0, badExamplesToUse[:, 1]]/255.0)**2+
#                                                ((255-toyImages[0, badExamplesToUse[:, 1]])/255.0-toyImages[0, badExamplesToUse[:, 0]]/255.0)**2).reshape((1, len(badExamplesToUse)))))
            badExamplesData = np.concatenate((badExamplesData, 
                                              (((255-toyImages[:, badExamplesToUse[:, 0]])/1.0-toyImages[:, badExamplesToUse[:, 1]]/1.0)**2+
                                               ((255-toyImages[:, badExamplesToUse[:, 1]])/1.0-toyImages[:, badExamplesToUse[:, 0]]/1.0)**2).reshape((toyImages.shape[0], len(badExamplesToUse)))))
        else :
            badExamplesData = np.concatenate((badExamplesData, 
                                              ((255-toyImages[:, badExamplesToUse[:, 0]])/255.0-toyImages[:, badExamplesToUse[:, 1]]/255.0).reshape((toyImages.shape[0], len(badExamplesToUse)))**2))
            
    
    if useTemporal :
        shiftedIdxs1 = badExamplesToUse[:, 0]+1
        np.place(shiftedIdxs1, shiftedIdxs1==numToyImages, -1)
        shiftedIdxs2 = badExamplesToUse[:, 1]+1
        np.place(shiftedIdxs2, shiftedIdxs2==numToyImages, -1)
        badExamplesData = np.concatenate((badExamplesData,
                                          ((toyImages[0, badExamplesToUse[:, 0]]/255.0-toyImages[0, badExamplesToUse[:, 1]-1]/255.0)**2).reshape((1, len(badExamplesToUse))),
                                          ((toyImages[0, shiftedIdxs1]/255.0-toyImages[0, badExamplesToUse[:, 1]]/255.0)**2).reshape((1, len(badExamplesToUse))),
                                          ((toyImages[0, badExamplesToUse[:, 0]-1]/255.0-toyImages[0, badExamplesToUse[:, 1]-1]/255.0)**2).reshape((1, len(badExamplesToUse))),
                                          ((toyImages[0, shiftedIdxs1]/255.0-toyImages[0, shiftedIdxs2]/255.0)**2).reshape((1, len(badExamplesToUse)))))
    
    X = np.concatenate((goodExamplesData, badExamplesData), axis=1)
    
    print X.shape
    
    w = np.array([np.concatenate((np.ones(currentTry[0]), np.zeros(currentTry[1])))]).T

    N = X.shape[0]
    phi0 = np.ones((N, 1))
    sio.savemat(dataPath + "trainingExamplesForSmallToyExample", {"X":X, "w":w})
    print N, X.shape, w.shape, phi0.shape

    ## now call the matlab script to fit phi using psi
    trainingExamplesLoc = dataPath + "trainingExamplesForSmallToyExample.mat"
    phiSaveLoc = dataPath + "fittedPhiForSmallToyExampleUsingPsi.mat"

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

#     print output
    sys.stdout.flush()
    fittedWeights.append(sio.loadmat(phiSaveLoc)['phi_MAP'])
    trainingErrors.append(np.sum(np.sqrt((np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP']))-w)**2)))
    l2DistErrors.append(np.sum(np.sqrt((np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))-w)**2)))
    
    toyRegDistMat = np.zeros((np.sum(numToyImages), np.sum(numToyImages)))
    toyRegDistMat[allPairsIdxs[:, 0], allPairsIdxs[:, 1]] = np.sqrt(np.dot(allPairs.T, sio.loadmat(phiSaveLoc)['phi_MAP']))[:, 0]
    toyRegDistMat[allPairsIdxs[:, 1], allPairsIdxs[:, 0]] = toyRegDistMat[allPairsIdxs[:, 0], allPairsIdxs[:, 1]]
    regressedDistanceMats.append(toyRegDistMat)

## distance matrix using the all one weights (i.e L2)
toyL2DistMat = np.zeros((np.sum(numToyImages), np.sum(numToyImages)))
toyL2DistMat[allPairsIdxs[:, 0], allPairsIdxs[:, 1]] = np.sqrt(np.dot(allPairs.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))[:, 0]
toyL2DistMat[allPairsIdxs[:, 1], allPairsIdxs[:, 0]] = toyL2DistMat[allPairsIdxs[:, 0], allPairsIdxs[:, 1]]
gwv.showCustomGraph(toyL2DistMat)

# <codecell>

print allPairs.T[:20, :]
print allPairsIdxs[:20, :]
print toyImages[:, allPairsIdxs[:20, 0]]
print toyImages[:, allPairsIdxs[:20, 1]]
print randomGoodPairs
print toyImages[:, 81], toyImages[:, 121]
# print goodExamplesIdxs[:100, :]

# <codecell>

gwv.showCustomGraph(toyL2DistMat)

# <codecell>

## plot training error wrt l2 distance errors
figure()
wErrHandle, = plot(np.array(numExamples)[0:, 0], trainingErrors/np.array(numExamples)[0:, 0], 'b.-')
l2ErrHandle, = plot(np.array(numExamples)[0:, 0], l2DistErrors/np.array(numExamples)[0:, 0], 'g.-')
legend([wErrHandle, l2ErrHandle], ['Weighted L2 Error', 'Normal L2 Error'])
## plot how fitted weights change
figure()
legendHandles = plot(numExamples[0:], np.array(fittedWeights)[:, :, 0])
# legend(legendHandles, ['Color Dimension Weight', 'Random Dimension Weight', 'Color Complement Dimension Weight'])
legend(legendHandles, ['Color Dimension Weight', 'Random Dimension Weight', 'i minus j-1', 'i+1 minus j', 'i-1 minus j-1', 'i+1 minus j+1'])
gca().text(.5,1.05,"Regressed weights",
        horizontalalignment='center',
        transform=gca().transAxes)
## plot regressed distance vs l2 distance for given X
figure()
wDistHandle, = plot(np.sqrt(np.dot(X.T, sio.loadmat(phiSaveLoc)['phi_MAP'])), c='y', zorder=0)
# l2DistHandle, = plot(np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), 'g.')
markerColors = np.zeros(len(X.T))
markerColors[X[0, :] <= 0.25] = 0
markerColors[np.all((X[0, :] < 0.75, X[0, :] > 0.25), axis=0)] = 1
markerColors[X[0, :] >= 0.75] = 2
# l2DistHandle = scatter(arange(len(X.T)), np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), c=markerColors, marker='d', cmap='brg', edgecolors='none', alpha=.9)
l2DistHandle = scatter(arange(len(X.T)), np.sqrt(np.dot(X.T, np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP']))), c='b', marker='d', cmap='brg', edgecolors='none', alpha=.9)
legend([wDistHandle, l2DistHandle], ['Weighted L2 Distance', 'Normal L2 Distance (brg colors for 1st dim diff)'])

# <codecell>

print np.array(fittedWeights)

# <codecell>

la = np.array([[[  5.75370303e-06],
  [  1.98026170e-05],
  [  6.08683678e-06],
  [  4.13045091e-05]],

 [[  3.30627742e-06],
  [  8.23000196e-06],
  [  6.45439079e-06],
  [  7.30920523e-06]],

 [[  1.73742169e-06],
  [  5.67529264e-06],
  [  6.91068515e-06],
  [  6.85317573e-06]],

 [[  8.61059748e-07],
  [  6.43230557e-06],
  [  7.44591657e-06],
  [  3.71514172e-06]],

 [[  3.15566188e-07],
  [  8.39728250e-06],
  [  7.16540943e-06],
  [  3.17361636e-06]],

 [[  1.12285405e-07],
  [  7.36157139e-06],
  [  7.20373431e-06],
  [  3.61047410e-06]],

 [[  3.68339760e-08],
  [  7.35595027e-06],
  [  7.03334438e-06],
  [  3.92868692e-06]],

 [[  1.16844047e-08],
  [  8.20602475e-06],
  [  6.97588908e-06],
  [  3.94293922e-06]]])
bla = np.array([[[  3.51116019e-01],
  [  6.50612416e-01],
  [  5.06810605e-01],
  [  8.60520556e-01]],

 [[  2.01972293e-01],
  [  4.05914611e-01],
  [  4.31706000e-01],
  [  3.81429868e-01]],

 [[  1.11742089e-01],
  [  3.05022226e-01],
  [  4.56300265e-01],
  [  3.83076751e-01]],

 [[  5.71748338e-02],
  [  3.59400142e-01],
  [  4.87272101e-01],
  [  2.22844256e-01]],

 [[  2.07700178e-02],
  [  5.22134669e-01],
  [  4.67620609e-01],
  [  2.00241503e-01]],

 [[  7.34309186e-03],
  [  4.70209500e-01],
  [  4.68964352e-01],
  [  2.32635911e-01]],

 [[  2.39958005e-03],
  [  4.75510550e-01],
  [  4.57546808e-01],
  [  2.54771782e-01]],

 [[  7.60242149e-04],
  [  5.32588937e-01],
  [  4.53678580e-01],
  [  2.56171434e-01]]])

print bla/la

# <codecell>

print np.array(fittedWeights)

# <codecell>

print goodExamplesToUse
print np.sqrt(X.T)*255
# print toyImages[:, 114], toyImages[:, 115]
# print goodExamplesData.T
# print ((toyImages[:, goodExamplesToUse[0, 0]]/255.0-toyImages[:, goodExamplesToUse[0, 1]]/255.0)**2).T
# print toyImages[:, goodExamplesToUse[0, 0]], toyImages[:, goodExamplesToUse[0, 1]]
# print 153/255.0, 128/255.0, 153/255.0- 128/255.0

# <codecell>

bob = -1
# bob += 1
gwv.showCustomGraph(regressedDistanceMats[bob])
gca().text(.5,1.05,np.string_(np.array(numExamples)[bob, 0]) + " training examples",
        horizontalalignment='center',
        transform=gca().transAxes)

# <codecell>

gwv.showCustomGraph(vtu.filterDistanceMatrix(toyL2DistMat, 4, True))
gwv.showCustomGraph(vtu.filterDistanceMatrix(regressedDistanceMats[1], 4, True))

# <codecell>

id1 = 9
id2 = 1
im1 = np.concatenate((toyImages[:, id1], [toyImages[0, id1]], [255.0-toyImages[0, id1]]))
im2 = np.concatenate((toyImages[:, id2], [255.0-toyImages[0, id2]], [toyImages[0, id2]]))
print im1, im2
print allPairsIdxs
print allPairs[:, 1]
print (im1/255.0-im2/255.0)**2
print np.sqrt(np.sum((im1/255.0-im2/255.0)**2)), np.sqrt(np.sum((im2/255.0-im1/255.0)**2))

# <codecell>

la = np.zeros_like(regressedDistanceMats[0])
la[allPairsIdxs[:, 0], allPairsIdxs[:, 1]] = np.sqrt(np.dot(allPairs.T, sio.loadmat(phiSaveLoc)['phi_MAP']))[:, 0]
la2 = np.copy(allPairs)
la2[-1, :] = ((255-toyImages[0, allPairsIdxs[:, 1]])/255.0-toyImages[0, allPairsIdxs[:, 0]]/255.0)**2
la[allPairsIdxs[:, 1], allPairsIdxs[:, 0]] = np.sqrt(np.dot(la2.T, sio.loadmat(phiSaveLoc)['phi_MAP']))[:, 0]
gwv.showCustomGraph(la)

# <codecell>

gwv.showCustomGraph(la-regressedDistanceMats[bob])

