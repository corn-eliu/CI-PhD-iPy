# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import opengm
import numpy as np
import cv2
import time
import os
from PIL import Image

import scipy.io as sio
import sys
import glob
import GraphWithValues as gwv

# <codecell>

DICT_SPRITE_NAME = 'sprite_name'
DICT_SEQUENCE_NAME = "semantic_sequence_name"
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MEDIAN_COLOR = 'median_color'
PATCH_BORDER = 0.4

# dataPath = "/home/ilisescu/PhD/data/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "wave2/"
# dataSet = "wave1/"
# dataSet = "wave3/"
# dataSet = "theme_park_sunny/"
dataSet = "windows/"
dataSet = "digger/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
bgImage = np.array(Image.open(dataPath + dataSet + "median.png"))[:, :, :3]

allXs = np.arange(bgImage.shape[1], dtype=np.float32).reshape((1, bgImage.shape[1])).repeat(bgImage.shape[0], axis=0)
allYs = np.arange(bgImage.shape[0], dtype=np.float32).reshape((bgImage.shape[0], 1)).repeat(bgImage.shape[1], axis=1)

trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())

## merge tracked sprite with bg
spriteIdx = 0
sequenceLength = len(trackedSprites[spriteIdx][DICT_BBOXES])
print numOfFrames

# <codecell>

def multivariateNormal(data, mean, var, normalized = True) :
    if (data.shape[0] != mean.shape[0] or np.any(data.shape[0] != np.array(var.shape)) 
        or len(var.shape) != 2 or var.shape[0] != var.shape[1]) :
        raise Exception("Data shapes don't agree data(" + np.string_(data.shape) + ") mean(" + np.string_(mean.shape) + 
                        ") var(" + np.string_(var.shape) + ")")
        
    D = float(data.shape[0])
    n = (1/(np.power(2.0*np.pi, D/2.0)*np.sqrt(np.linalg.det(var))))
    if normalized :
        p = n*np.exp(-0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1))
    else :
        p = np.exp(-0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1))
        
    return p

def minusLogMultivariateNormal(data, mean, var, normalized = True) :
    if (data.shape[0] != mean.shape[0] or np.any(data.shape[0] != np.array(var.shape)) 
        or len(var.shape) != 2 or var.shape[0] != var.shape[1]) :
        raise Exception("Data shapes don't agree data(" + np.string_(data.shape) + ") mean(" + np.string_(mean.shape) + 
                        ") var(" + np.string_(var.shape) + ")")
    
    D = float(data.shape[0])
    n = -0.5*np.log(np.linalg.det(var))-(D/2.0)*np.log(2.0*np.pi)
    if normalized :
        p = n -0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1)
    else :
        p = -0.5*np.sum(np.dot((data-mean).T, np.linalg.inv(var))*(data-mean).T, axis=-1)
        
    return -p

def vectorisedMinusLogMultiNormal(dataPoints, means, var, normalized = True) :
    if (dataPoints.shape[1] != means.shape[1] or np.any(dataPoints.shape[1] != np.array(var.shape)) 
        or len(var.shape) != 2 or var.shape[0] != var.shape[1]) :
        raise Exception("Data shapes don't agree data(" + np.string_(dataPoints.shape) + ") mean(" + np.string_(means.shape) + 
                        ") var(" + np.string_(var.shape) + ")")
    
    D = float(dataPoints.shape[1])
    n = -0.5*np.log(np.linalg.det(var))-(D/2.0)*np.log(2.0*np.pi)
    
    ## this does 0.5*dot(dot(data-mean, varInv), data-mean)
    varInv = np.linalg.inv(var)
    dataMinusMean = dataPoints-means
    
    ps = []
    for i in xrange(int(D)) :
        ps.append(np.sum((dataMinusMean)*varInv[:, i], axis=-1))
    
    ps = np.array(ps).T
    
    ps = -0.5*np.sum(ps*(dataMinusMean), axis=-1)
    
    if normalized :
        return n-ps
    else :
        return -ps

# <codecell>

def getSpritePatch(sprite, frameKey, frameWidth, frameHeight) :
    """Computes sprite patch based on its bbox
    
        \t  sprite      : dictionary containing relevant sprite data
        \t  frameKey    : the key of the frame the sprite patch is taken from
        \t  frameWidth  : width of original image
        \t  frameHeight : height of original image
           
        return: spritePatch, offset, patchSize,
                [left, top, bottom, right] : array of booleans telling whether the expanded bbox touches the corresponding border of the image"""
    
    ## get the bbox for the current sprite frame, make it larger and find the rectangular patch to work with
    ## boundaries of the patch [min, max]
    
    ## returns sprite patch based on bbox and returns it along with the offset [x, y] and it's size [rows, cols]
    
    ## make bbox bigger
    largeBBox = sprite[DICT_BBOXES][frameKey].T
    ## move to origin
    largeBBox = np.dot(np.array([[-sprite[DICT_BBOX_CENTERS][frameKey][0], 1.0, 0.0], 
                                 [-sprite[DICT_BBOX_CENTERS][frameKey][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## make bigger
    largeBBox = np.dot(np.array([[0.0, 1.0 + PATCH_BORDER, 0.0], 
                                 [0.0, 0.0, 1.0 + PATCH_BORDER]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## move back tooriginal center
    largeBBox = np.dot(np.array([[sprite[DICT_BBOX_CENTERS][frameKey][0], 1.0, 0.0], 
                                 [sprite[DICT_BBOX_CENTERS][frameKey][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    
    xBounds = np.zeros(2); yBounds = np.zeros(2)
    
    ## make sure xBounds are in between 0 and width and yBounds are in between 0 and height
    xBounds[0] = np.max((0, np.min(largeBBox[0, :])))
    xBounds[1] = np.min((frameWidth, np.max(largeBBox[0, :])))
    yBounds[0] = np.max((0, np.min(largeBBox[1, :])))
    yBounds[1] = np.min((frameHeight, np.max(largeBBox[1, :])))
    
    offset = np.array([np.round(np.array([xBounds[0], yBounds[0]]))], dtype=int).T # [x, y]
    patchSize = np.array(np.round(np.array([yBounds[1]-yBounds[0], xBounds[1]-xBounds[0]])), dtype=int) # [rows, cols]
    
    spritePatch = np.array(Image.open(sprite[DICT_FRAMES_LOCATIONS][frameKey]))[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :]
    
    return spritePatch, offset, patchSize, [np.min((largeBBox)[0, :]) > 0.0 ,
                                            np.min((largeBBox)[1, :]) > 0.0 ,
                                            np.max((largeBBox)[1, :]) < frameHeight,
                                            np.max((largeBBox)[0, :]) < frameWidth]


def getPatchPriors(bgPatch, spritePatch, offset, patchSize, sprite, frameKey, prevFrameKey = None, prevFrameAlphaLoc = "",
                   prevMaskImportance = 0.8, prevMaskDilate = 13, prevMaskBlurSize = 31, prevMaskBlurSigma = 2.5,
                   diffPatchImportance = 0.015, diffPatchMultiplier = 1000.0, useOpticalFlow = True, useDiffPatch = False) :
    """Computes priors for background and sprite patches
    
        \t  bgPatch             : background patch
        \t  spritePatch         : sprite patch
        \t  offset              : [x, y] position of patches in the coordinate system of the original images
        \t  patchSize           : num of [rows, cols] per patches
        \t  sprite              : dictionary containing relevant sprite data
        \t  frameKey            : the key of the frame the sprite patch is taken from
        \t  prevFrameKey        : the key of the previous frame
        \t  prevFrameAlphaLoc   : location of the previous frame
        \t  prevMaskImportance  : balances the importance of the prior based on the remapped mask of the previous frame
        \t  prevMaskDilate      : amount of dilation to perform on previous frame's mask
        \t  prevMaskBlurSize    : size of the blurring kernel perfomed on previous frame's mask
        \t  prevMaskBlurSigma   : variance of the gaussian blurring perfomed on previous frame's mask
        \t  diffPatchImportance : balances the importance of the prior based on difference of patch to background
        \t  diffPatchMultiplier : multiplier that changes the scaling of the difference based cost
        \t  useOpticalFlow      : modify sprite prior by the mask of the previous frame
        \t  useDiffPatch        : modify bg prior by difference of sprite to bg patch
           
        return: bgPrior, spritePrior"""
    
    ## get uniform prior for bg patch
    bgPrior = -np.log(np.ones(patchSize)/np.prod(patchSize))
    
    ## get prior for sprite patch
    spritePrior = np.zeros(patchSize)
    xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
    ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))
    data = np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys)))))
    
    ## get covariance and means of prior on patch by using the bbox
    spriteBBox = sprite[DICT_BBOXES][frameKey].T
    segment1 = spriteBBox[:, 0] - spriteBBox[:, 1]
    segment2 = spriteBBox[:, 1] - spriteBBox[:, 2]
    sigmaX = np.linalg.norm(segment1)/3.7
    sigmaY = np.linalg.norm(segment2)/3.7
    
    rotRadians = sprite[DICT_BBOX_ROTATIONS][frameKey]
    
    rotMat = np.array([[np.cos(rotRadians), -np.sin(rotRadians)], [np.sin(rotRadians), np.cos(rotRadians)]])
    
    means = np.reshape(sprite[DICT_BBOX_CENTERS][frameKey], (2, 1)) - offset
    covs = np.dot(np.dot(rotMat.T, np.array([[sigmaX**2, 0.0], [0.0, sigmaY**2]])), rotMat)
    
    spritePrior = np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F')
    
    ## change the spritePrior using optical flow stuff
    if useOpticalFlow and prevFrameKey != None :
        prevFrameName = sprite[DICT_FRAMES_LOCATIONS][prevFrameKey].split('/')[-1]
        nextFrameName = sprite[DICT_FRAMES_LOCATIONS][frameKey].split('/')[-1]
        
        if os.path.isfile(prevFrameAlphaLoc+prevFrameName) :
            alpha = np.array(Image.open(prevFrameAlphaLoc+prevFrameName))[:, :, -1]/255.0

            flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(Image.open(dataPath+dataSet+nextFrameName)), cv2.COLOR_RGB2GRAY), 
                                                cv2.cvtColor(np.array(Image.open(dataPath+dataSet+prevFrameName)), cv2.COLOR_RGB2GRAY), 
                                                0.5, 3, 15, 3, 5, 1.1, 0)
        
            ## remap alpha according to flow
            remappedFg = cv2.remap(alpha, flow[:, :, 0]+allXs, flow[:, :, 1]+allYs, cv2.INTER_LINEAR)
            ## get patch
            remappedFgPatch = remappedFg[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
            remappedFgPatch = cv2.GaussianBlur(cv2.morphologyEx(remappedFgPatch, cv2.MORPH_DILATE, 
                                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prevMaskDilate, prevMaskDilate))), 
                                               (prevMaskBlurSize, prevMaskBlurSize), prevMaskBlurSigma)

            spritePrior = (1.0-prevMaskImportance)*spritePrior + prevMaskImportance*(-np.log((remappedFgPatch+0.01)/np.sum(remappedFgPatch+0.01)))
    
    
    if useDiffPatch :
        ## change the background prior to give higher cost for pixels to be classified as background if the difference between bgPatch and spritePatch is high
        diffPatch = np.reshape(vectorisedMinusLogMultiNormal(spritePatch.reshape((np.prod(patchSize), 3)), 
                                                             bgPatch.reshape((np.prod(patchSize), 3)), 
                                                             np.eye(3)*diffPatchMultiplier, True), patchSize)
        bgPrior = (1.0-diffPatchImportance)*bgPrior + diffPatchImportance*diffPatch
        
    
    return bgPrior, spritePrior
    

def mergePatches(bgPatch, spritePatch, bgPrior, spritePrior, offset, patchSize, touchedBorders, scribble = None, useCenterSquare = True, useGradients = False) :
    """Computes pixel labels using graphcut given two same size patches
    
        \t  bgPatch         : background patch
        \t  spritePatch     : sprite patch
        \t  bgPrior         : background prior
        \t  spritePrior     : sprite prior
        \t  offset          : [x, y] position of patches in the coordinate system of the original images
        \t  patchSize       : num of [rows, cols] per patches
        \t  touchedBorders  : borders of the image touched by the enlarged bbox
        \t  useCenterSquare : forces square of pixels in the center of the patch to be classified as foreground
        \t  useGradients    : uses the gradient weighted pairwise cost
           
        return: reshapedLabels = labels for each pixel"""

    t = time.time()
    ## merge two overlapping patches

    h = patchSize[0]
    w = patchSize[1]
    
    patAPixs = np.empty(0, dtype=np.uint)
    patBPixs = np.empty(0, dtype=np.uint)
    
    ## force small square of size squarePadding*2 + 1 around center of patch to come from patch B (i.e. the car)
    if useCenterSquare :
        squarePadding = 6
        rows = np.ndarray.flatten(np.arange((h/2)-squarePadding, (h/2)+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
        cols = np.ndarray.flatten(np.arange((w/2)-squarePadding, (w/2)+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
        patBPixs = np.unique(np.concatenate((patBPixs, np.array(rows + cols*h, dtype=np.uint))))
    
    ## force one ring of pixels on the edge of the patch to come from patch A (i.e. the bg) (unless that column/row is intersected by the bbox)
#     if np.min((largeBBox)[0, :]) > 0.0 :
    if touchedBorders[0] :
#         print "adding left column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h, dtype=np.uint)[1:-1])))
    else :
#         print "adding left column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h, dtype=np.uint)[1:-1])))
#     if np.min((largeBBox)[1, :]) > 0.0 :
    if touchedBorders[1] :
#         print "adding top row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1])))
    else :
#         print "adding top row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1])))
#     if np.max((largeBBox)[1, :]) < bgImage.shape[0] :
    if touchedBorders[2] :
#         print "adding bottom row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1]+h-1)))
    else :
#         print "adding bottom row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=np.uint)[1:-1]+h-1)))
#     if np.max((largeBBox)[0, :]) < bgImage.shape[1] :
    if touchedBorders[3] :
#         print "adding right column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=np.uint)[1:-1])))
    else :
#         print "adding right column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(h*(w-1), h*w, dtype=np.uint)[1:-1])))
    
#     patBPixs = np.empty(0)

    ## deal with scribble if present
    if scribble != None :
        ## find indices of bg (blue) and fg (green) pixels in scribble
        bgPixs = np.argwhere(np.all((scribble[:, :, 0] == 0, scribble[:, :, 1] == 0, scribble[:, :, 2] == 255), axis=0))
        bgPixs[:, 0] -= offset[1]
        bgPixs[:, 1] -= offset[0]
        bgPixs = bgPixs[np.all(np.concatenate(([bgPixs[:, 0] >= 0], 
                                               [bgPixs[:, 1] >= 0], 
                                               [bgPixs[:, 0] < h], 
                                               [bgPixs[:, 1] < w])).T, axis=-1), :]
        fgPixs = np.argwhere(np.all((scribble[:, :, 0] == 0, scribble[:, :, 1] == 255, scribble[:, :, 2] == 0), axis=0))
        fgPixs[:, 0] -= offset[1]
        fgPixs[:, 1] -= offset[0]
        fgPixs = fgPixs[np.all(np.concatenate(([fgPixs[:, 0] >= 0], 
                                               [fgPixs[:, 1] >= 0], 
                                               [fgPixs[:, 0] < h], 
                                               [fgPixs[:, 1] < w])).T, axis=-1), :]
        
        

        ## for simplicity keep track of fixed pixels in a new patch-sized array
        fixedPixels = np.zeros(patchSize)
        ## get fixed pixels from other params first
        ## 1 == bg pixels (get 2d coords from 1d first)
        if len(patAPixs) > 0 :
            fixedPixels[np.array(np.mod(patAPixs, patchSize[0]), dtype=np.uint), np.array(patAPixs/patchSize[0], dtype=np.uint)] = 1
        ## 2 == fg pixels (get 2d coords from 1d first)
        if len(patAPixs) > 0 :
            fixedPixels[np.array(np.mod(patBPixs, patchSize[0]), dtype=np.uint), np.array(patBPixs/patchSize[0], dtype=np.uint)] = 2
        
        if len(bgPixs) > 0 :
            fixedPixels[bgPixs[:, 0], bgPixs[:, 1]] = 1
        if len(fgPixs) > 0 :
            fixedPixels[fgPixs[:, 0], fgPixs[:, 1]] = 2

        ## turn back to 1d indices
        patAPixs = np.argwhere(fixedPixels == 1)
        patAPixs = np.sort(patAPixs[:, 0] + patAPixs[:, 1]*patchSize[0])
        patBPixs = np.argwhere(fixedPixels == 2)
        patBPixs = np.sort(patBPixs[:, 0] + patBPixs[:, 1]*patchSize[0])
        gwv.showCustomGraph(fixedPixels)
    
    patA = np.copy(bgPatch/255.0)
    patB = np.copy(spritePatch/255.0)
    
#     print "patch setup", time.time() - t
    t = time.time()
    
    if useGradients :
        sobelX = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

        labels, unaryCosts, pairCosts, graphModel = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, 0.001, 
                                                           bgPrior.reshape(np.prod(patchSize), order='F'),
                                                           spritePrior.reshape(np.prod(patchSize), order='F'),
                                                           cv2.filter2D(bgPatch, cv2.CV_32F, sobelX),
                                                           cv2.filter2D(bgPatch, cv2.CV_32F, sobelX.T),
                                                           cv2.filter2D(spritePatch, cv2.CV_32F, sobelX),
                                                           cv2.filter2D(spritePatch, cv2.CV_32F, sobelX.T))
    else :
        labels, unaryCosts, pairCosts, graphModel = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, 0.001, 
                                                           bgPrior.reshape(np.prod(patchSize), order='F'),
                                                           spritePrior.reshape(np.prod(patchSize), order='F'))
    
#     print "total solving", time.time() - t
    t = time.time()
        
    return labels

# <codecell>

def getGraphcutOnOverlap(patchA, patchB, patchAPixels, patchBPixels, multiplier, unaryPriorPatchA, unaryPriorPatchB,
                         patchAGradX = None, patchAGradY = None, patchBGradX = None, patchBGradY = None) :
    """Computes pixel labels using graphcut given two same size patches
    
        \t  patchA           : patch A
        \t  patchB           : patch B
        \t  patchAPixels     : pixels that are definitely to be taken from patch A
        \t  patchBPixels     : pixels that are definitely to be taken from patch B
        \t  multiplier       : sigma multiplier for rgb space normal
        \t  unaryPriorPatchA : prior cost ditribution for patchA labels
        \t  unaryPriorPatchB : prior cost ditribution for patchB labels
           
        return: reshapedLabels = labels for each pixel"""
    
    t = time.time()
    if np.all(patchA.shape != patchB.shape) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
        
    if patchA.dtype != np.float64 or patchB.dtype != np.float64 :
        raise Exception("The two specified patches are not of type float64! Check there is no overflow when computing costs")
    
    h, width = patchA.shape[0:2]
    maxCost = 10000000.0#np.sys.float_info.max
    
    s = time.time()
    ## build graph
    numLabels = 2
    numNodes = h*width+numLabels
    gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels)
    
    ## Last 2 nodes are patch A and B respectively
    idxPatchANode = numNodes - 2
    idxPatchBNode = numNodes - 1
    
        
    ## get unary functions
    unaries = np.zeros((numNodes,numLabels))
    
    ## fix label for nodes representing patch A and B to have label 0 and 1 respectively
    unaries[idxPatchANode, :] = [0.0, maxCost]
    unaries[idxPatchBNode, :] = [maxCost, 0.0]
    
    ## set unaries based on the priors given for both patches
    unaries[0:h*width, 0] = unaryPriorPatchA
    unaries[0:h*width, 1] = unaryPriorPatchB
    
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, np.arange(0, numNodes, 1))
    
    
    ## get factor indices for the overlap grid of pixels
    stmp = time.time()
#     pairIndices = np.array(opengm.secondOrderGridVis(width,h,True))
    pairIndices = getGridPairIndices(width, h)
#     print "pairIndices took", time.time()-stmp, "seconds"
#     sys.stdout.flush()
    ## get pairwise functions for those nodes
#     pairwise = np.zeros(len(pairIndices))
#     for pair, i in zip(pairIndices, arange(len(pairIndices))) :
#         sPix = np.array([int(np.mod(pair[0],h)), int(pair[0]/h)])
#         tPix = np.array([int(np.mod(pair[1],h)), int(pair[1]/h)])
        
# #         pairwise[i] = norm(patchA[sPix[0], sPix[1], :] - patchB[sPix[0], sPix[1], :])
# #         pairwise[i] += norm(patchA[tPix[0], tPix[1], :] - patchB[tPix[0], tPix[1], :])

#         pairwise[i] = minusLogMultivariateNormal(patchA[sPix[0], sPix[1], :].reshape((3, 1)), patchB[sPix[0], sPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
#         pairwise[i] += minusLogMultivariateNormal(patchA[tPix[0], tPix[1], :].reshape((3, 1)), patchB[tPix[0], tPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
        
#         fid = gm.addFunction(np.array([[0.0, pairwise[i]],[pairwise[i], 0.0]]))
#         gm.addFactor(fid, pair)
        
    sPixs = np.array([np.mod(pairIndices[:, 0],h), pairIndices[:, 0]/h], dtype=int).T
    tPixs = np.array([np.mod(pairIndices[:, 1],h), pairIndices[:, 1]/h], dtype=int).T
    
    pairwise = vectorisedMinusLogMultiNormal(patchA[sPixs[:, 0], sPixs[:, 1], :], patchB[sPixs[:, 0], sPixs[:, 1], :], np.eye(3)*multiplier, False)
    pairwise += vectorisedMinusLogMultiNormal(patchA[tPixs[:, 0], tPixs[:, 1], :], patchB[tPixs[:, 0], tPixs[:, 1], :], np.eye(3)*multiplier, False)
#     print np.min(pairwise), np.max(pairwise), pairwise
    if False and patchAGradX != None and patchAGradY != None and patchBGradX != None and patchBGradY != None :
#         pairwise /= ((vectorisedMinusLogMultiNormal(patchAGradX[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchAGradX[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchAGradX[tPixs[:, 0], tPixs[:, 1], :], np.zeros_like(patchAGradX[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradX[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradX[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradX[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradX[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchAGradY[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchAGradY[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchAGradY[tPixs[:, 0], tPixs[:, 1], :], np.zeros_like(patchAGradY[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradY[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradY[sPixs[:, 0], sPixs[:, 1], :]), np.eye(3)*multiplier, False)+
#                      vectorisedMinusLogMultiNormal(patchBGradY[sPixs[:, 0], sPixs[:, 1], :], np.zeros_like(patchBGradY[tPixs[:, 0], tPixs[:, 1], :]), np.eye(3)*multiplier, False))/1000.0+0.00001)
        denominator = (np.sqrt(np.sum(patchAGradX[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchAGradX[tPixs[:, 0], tPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradX[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradX[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchAGradY[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchAGradY[tPixs[:, 0], tPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradY[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1))+
                     np.sqrt(np.sum(patchBGradY[sPixs[:, 0], sPixs[:, 1], :]**2, axis=-1)))
    
        pairwise /= ((np.max(denominator) - denominator)+0.000001)
    
#     print np.min(pairwise), np.max(pairwise), pairwise
    fids = gm.addFunctions(np.array([[0.0, 1.0],[1.0, 0.0]]).reshape((1, 2, 2)).repeat(len(pairwise), axis=0)*
                           pairwise.reshape((len(pairwise), 1, 1)).repeat(2, axis=1).repeat(2, axis=2))
    
    gm.addFactors(fids, pairIndices)
            
    
    # add function used for connecting the patch variables
    fid = gm.addFunction(np.array([[0.0, maxCost],[maxCost, 0.0]]))
    
    # connect patch A to definite patch A pixels
    if len(patchAPixels) > 0 :
        patchAFactors = np.hstack((patchAPixels.reshape((len(patchAPixels), 1)), np.ones((len(patchAPixels), 1), dtype=np.uint)*idxPatchANode))
        gm.addFactors(fid, patchAFactors)
    
    # connect patch B to definite patch B pixels
    if len(patchBPixels) > 0 :
        patchBFactors = np.hstack((patchBPixels.reshape((len(patchBPixels), 1)), np.ones((len(patchBPixels), 1), dtype=np.uint)*idxPatchBNode))
        gm.addFactors(fid, patchBFactors)
    
#     print "graph setup", time.time() - t
    t = time.time()
#     print "graph setup took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    graphCut = opengm.inference.GraphCut(gm=gm)
    graphCut.infer()
#     print "graph inference took", time.time()-s, "seconds"
#     sys.stdout.flush()
    
    labels = np.array(graphCut.arg(), dtype=int)
    
    reshapedLabels = np.reshape(np.copy(labels[0:-numLabels]), patchA.shape[0:2], 'F')
    
#     print "solving", time.time() - t
    t = time.time()
#     print gm
#     print gm.evaluate(labels)
    
    return reshapedLabels, unaries, pairwise, gm

# <codecell>

def getGridPairIndices(width, height) :
## deal with pixels that have East and South neighbours i.e. all of them apart from last column and last row
    pairIdxs = np.zeros(((width*height-(width+height-1))*2, 2), dtype=int)
## each column contains idxs [0, h-2]
    idxs = np.arange(0, height-1, dtype=int).reshape((height-1, 1)).repeat(width-1, axis=-1)
## each column contains idxs [0, h-2]+h*i where i is the column index 
## (i.e. now I have indices of all nodes in the grid apart from last col and row)
    idxs += (np.arange(0, width-1)*height).reshape((1, width-1)).repeat(height-1, axis=0)
    # figure(); imshow(idxs)
## now flatten idxs and repeat once so that I have the idx for each node that has E and S neighbours twice
    idxs = np.ndarray.flatten(idxs.T).repeat(2)
## idxs for each "left" node (that is connected to the edge) are the ones just computed
    pairIdxs[:, 0] = idxs
## idxs for each "right" node are to the E and S so need to sum "left" idx to height and to 1
# print np.ndarray.flatten(np.array([[patchSize[0]], [1]]).repeat(np.prod(patchSize)-(np.sum(patchSize)-1), axis=-1).T)
    pairIdxs[:, 1] = idxs + np.ndarray.flatten(np.array([[height], [1]]).repeat(width*height-(width+height-1), axis=-1).T)
    
## deal with pixels that have only East neighbours
## get "left" nodes
    leftNodes = np.arange(height-1, width*height-1, height)
## now connect "left" nodes to the nodes to their East (i.e. sum to height) and add them to the list of pair indices
    pairIdxs = np.concatenate((pairIdxs, np.array([leftNodes, leftNodes+height]).T), axis=0)
    
## deal with pixels that have only South neighbours
## get "top" nodes
    topNodes = np.arange(width*height-height, width*height-1)
## now connect "to" nodes to the nodes to their South (i.e. sum to 1) and add them to the list of pair indices
    pairIdxs = np.concatenate((pairIdxs, np.array([topNodes, topNodes+1]).T), axis=0)
    
    return pairIdxs

# <codecell>

def computeMattedImage(frameIdx, spriteIdx, framePathsIdxs, imageWidth, imageHeight, outputPath) :
    startTime = time.time()
    ## returns rgba
    frameName =  trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][framePathsIdxs[frameIdx]].split('/')[-1]
    
    t = time.time()
    spritePatch, offset, patchSize, touchedBorders = getSpritePatch(trackedSprites[spriteIdx], framePathsIdxs[frameIdx], 
                                                                    imageWidth, imageHeight)
    bgPatch = np.copy(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
#             print "patches", time.time() - t
    t = time.time()

    if frameIdx > 0 :
        bgPrior, spritePrior = getPatchPriors(bgPatch, spritePatch, offset, patchSize, trackedSprites[spriteIdx],
                                              framePathsIdxs[frameIdx], 
                                              prevFrameKey=framePathsIdxs[frameIdx-1], 
                                              prevFrameAlphaLoc=outputPath,
                                              useOpticalFlow=True,
                                              useDiffPatch=False,
                                              prevMaskImportance=0.8,
                                              prevMaskDilate=13,
                                              prevMaskBlurSize=31,
                                              prevMaskBlurSigma=2.5,
                                              diffPatchImportance=0.01,
                                              diffPatchMultiplier=1000.0)
#         print "priors with flow", time.time() - t
        t = time.time()
    else :
        bgPrior, spritePrior = getPatchPriors(bgPatch, spritePatch, offset, patchSize, trackedSprites[spriteIdx],
                                              framePathsIdxs[frameIdx],
                                              useOpticalFlow=True,
                                              useDiffPatch=False,
                                              prevMaskImportance=0.8,
                                              prevMaskDilate=13,
                                              prevMaskBlurSize=31,
                                              prevMaskBlurSigma=2.5,
                                              diffPatchImportance=0.01,
                                              diffPatchMultiplier=1000.0)
#         print "priors without flow", time.time() - t
#         figure(); imshow(bgPrior)
#         figure(); imshow(spritePrior)
        t = time.time()

#             scribble = None
#             if self.scribble.format() == QtGui.QImage.Format.Format_RGB888 :
#                 scribble = np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 3))[:, :, [2, 1, 0]]
#             elif self.scribble.format() == QtGui.QImage.Format.Format_RGB32 :
#                 scribble = np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 4))[:, :, [2, 1, 0]]

    if os.path.isfile(outputPath+"scribble-"+frameName) :
        scribbleIm = np.array(Image.open(outputPath+"scribble-"+frameName), dtype=np.uint8)
    else :
        scribbleIm = np.ones((720, 1280, 3), dtype=np.uint8)*np.uint8(255)

    labels = mergePatches(bgPatch, spritePatch, bgPrior, spritePrior, offset, patchSize, touchedBorders,
                          scribble=scribbleIm,
                          useCenterSquare=True,
                          useGradients=False)
#     figure(); imshow(labels)
#             print "merging", time.time() - t
    t = time.time()

    outputPatch = np.zeros((bgPatch.shape[0], bgPatch.shape[1], bgPatch.shape[2]+1), dtype=np.uint8)
    for i in xrange(labels.shape[0]) :
        for j in xrange(labels.shape[1]) :
            if labels[i, j] == 0 :
                ## patA stands for the bgPatch but I want to set the pixels here to 0 to save space
                outputPatch[i, j, 0:-1] = 0#bgPatch[i, j, :]
            else :
                outputPatch[i, j, 0:-1] = spritePatch[i, j, :]
                outputPatch[i, j, -1] = 255

    currentFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], bgImage.shape[2]+1), dtype=np.uint8)
    currentFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = np.copy(outputPatch)
#             print "putting together the frame", time.time() - t
    t = time.time()
#     figure(); imshow(currentFrame)

#     Image.fromarray((currentFrame).astype(np.uint8)).save(outputPath + frameName)
#             print "saving", time.time() - t
    t = time.time()
    print "done in", time.time() - startTime


im = np.array(Image.open(dataPath+dataSet+"median.png"))#[:, :, :3]
imageWidth = im.shape[1]
imageHeight = im.shape[0]

for spriteIdx in np.arange(len(trackedSprites))[2:3] :
    print trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]
    framePathsIdxs = np.sort(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())
    outputPath = dataPath + dataSet + trackedSprites[spriteIdx][DICT_SEQUENCE_NAME] + "-maskedFlow/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
        
    for frameIdx in np.arange(len(framePathsIdxs))[146:147] :
        computeMattedImage(frameIdx, spriteIdx, framePathsIdxs, imageWidth, imageHeight, outputPath)

# <codecell>

%pylab
# figure(); imshow(im)
im.shape

# <codecell>

# import shutil
# for i in xrange(1267, 1407) :
#     shutil.copyfile("/media/ilisescu/Data1/PhD/data/wave1/aron1-maskedFlow/scribble-frame-01266.png",
#                     "/media/ilisescu/Data1/PhD/data/wave1/aron1-maskedFlow/scribble-frame-{0:05d}.png".format(i))

