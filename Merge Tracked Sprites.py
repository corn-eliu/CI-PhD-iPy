# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
sys.path.append('/opt/pcaflow-master/')
import opengm
import numpy as np
import cv2
import time
import os
# import graph_tool as gt
from PIL import Image
from PySide import QtCore, QtGui

import scipy.io as sio
import GraphWithValues as gwv
import sys
import glob

app = QtGui.QApplication(sys.argv)

# <codecell>

S_IDX = 0
T_IDX = 1
ST_COST = 2
S_LABEL = 3
T_LABEL = 4
S_A_COLOR = np.arange(5, 8, dtype=int)
S_B_COLOR = np.arange(8, 11, dtype=int)
T_A_COLOR = np.arange(11, 14, dtype=int)
T_B_COLOR = np.arange(14, 17, dtype=int)
print S_IDX, T_IDX, ST_COST, S_LABEL, T_LABEL, S_A_COLOR, S_B_COLOR, T_A_COLOR, T_B_COLOR

DICT_SPRITE_NAME = 'sprite_name'
DICT_BBOX_AFFINES = 'bbox_affines'
DICT_NUM_FRAMES = 'num_frames'
DICT_FRAMES_LOCATIONS = 'frame_locs'

## used for enlarging bbox used to decide size of patch around it (percentage)
PATCH_BORDER = 0.4

# <codecell>

## load the tracked sprites
DICT_SPRITE_NAME = 'sprite_name'
<<<<<<< HEAD
DICT_SEQUENCE_NAME = "semantic_sequence_name"
=======
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
# DICT_BBOX_AFFINES = 'bbox_affines'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
# DICT_NUM_FRAMES = 'num_frames'
# DICT_START_FRAME = 'start_frame'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MEDIAN_COLOR = 'median_color'

<<<<<<< HEAD
# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
# dataSet = "wave2/"
# dataSet = "wave1/"
# dataSet = "wave3/"
# dataSet = "windows/"
dataSet = "digger/"
=======
dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
<<<<<<< HEAD
bgImage = np.array(Image.open(dataPath + dataSet + "median.png"))[:, :, :3]

allXs = arange(bgImage.shape[1], dtype=float32).reshape((1, bgImage.shape[1])).repeat(bgImage.shape[0], axis=0)
allYs = arange(bgImage.shape[0], dtype=float32).reshape((bgImage.shape[0], 1)).repeat(bgImage.shape[1], axis=1)
=======
bgImage = np.array(Image.open(dataPath + dataSet + "median.png"))
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
<<<<<<< HEAD
    print trackedSprites[-1][DICT_SEQUENCE_NAME]
=======
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

## merge tracked sprite with bg
spriteIdx = 0
sequenceLength = len(trackedSprites[spriteIdx][DICT_BBOXES])

# <codecell>

<<<<<<< HEAD
# ## finds mean color for each sprite
# bgIm = np.array(Image.open(dataPath+dataSet+"median.png"))

# for i in arange(len(trackedSprites)) :
#     maskDir = dataPath + dataSet + trackedSprites[i][DICT_SEQUENCE_NAME] + "-masked-blended"
    
#     medianCols = []
#     count = 0
#     lenSprite = len(trackedSprites[i][DICT_FRAMES_LOCATIONS])
    
#     for f in np.sort(trackedSprites[i][DICT_FRAMES_LOCATIONS].keys())[int(lenSprite*0.15):-int(lenSprite*0.15)] :
#         count += 1
#         frameName = trackedSprites[i][DICT_FRAMES_LOCATIONS][f].split(os.sep)[-1]
#         im = np.array(cv2.cvtColor(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED), cv2.COLOR_BGRA2RGBA), dtype=np.uint8)
# #         center = np.array(trackedSprites[i][DICT_BBOX_CENTERS][f], dtype=int)[::-1]
# #         squarePadding = 20
# #         rows = np.ndarray.flatten(arange((center[0])-squarePadding, 
# #                                          (center[0])+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
# #         cols = np.ndarray.flatten(arange((center[1])-squarePadding, 
# #                                          (center[1])+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
        
# #         medianCols.append(np.median(im[rows, cols, :-1], axis=0))
        
# #         visiblePixels = np.argwhere(im[:, :, -1] != 0)
# #         medianCols.append(np.mean(im[visiblePixels[:, 0], visiblePixels[:, 1], :-1], axis=0))

#         diffIm = (np.sum(np.abs(bgIm-im[:, :, :-1]), axis=-1)*im[:, :, -1]/255.0)
#         relevantPixels = np.argwhere(diffIm/np.max(diffIm) > 0.5)
    
#         medianCols.append(np.mean(im[relevantPixels[:, 0], relevantPixels[:, 1], :-1], axis=0))
        
#         sys.stdout.write('\r' + "Processed image " + np.string_(count) + " (" + np.string_(len(trackedSprites[i][DICT_FRAMES_LOCATIONS])) + ")")
#         sys.stdout.flush()
    
#     medianRGB = np.median(np.array(medianCols), axis=0)
#     normed = medianRGB/np.linalg.norm(medianRGB)
    
#     trackedSprites[i][DICT_MEDIAN_COLOR] = np.array(255/np.max(normed)*normed, dtype=int)
    
#     print 
#     print dataPath + dataSet + "sprite-" + "{0:04}".format(i) + "-" + trackedSprites[i][DICT_SEQUENCE_NAME] + ".npy", trackedSprites[i][DICT_MEDIAN_COLOR]
#     np.save(dataPath + dataSet + "sprite-" + "{0:04}".format(i) + "-" + trackedSprites[i][DICT_SEQUENCE_NAME] + ".npy", trackedSprites[i])
=======
bgIm = np.array(Image.open(dataPath+dataSet+"median.png"))

for i in arange(len(trackedSprites)) :
    maskDir = dataPath + dataSet + trackedSprites[i][DICT_SPRITE_NAME] + "-masked-blended"
    
    medianCols = []
    count = 0
    lenSprite = len(trackedSprites[i][DICT_FRAMES_LOCATIONS])
    
    for f in np.sort(trackedSprites[i][DICT_FRAMES_LOCATIONS].keys())[int(lenSprite*0.15):-int(lenSprite*0.15)] :
        count += 1
        frameName = trackedSprites[i][DICT_FRAMES_LOCATIONS][f].split(os.sep)[-1]
        im = np.array(cv2.cvtColor(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED), cv2.COLOR_BGRA2RGBA), dtype=np.uint8)
#         center = np.array(trackedSprites[i][DICT_BBOX_CENTERS][f], dtype=int)[::-1]
#         squarePadding = 20
#         rows = np.ndarray.flatten(arange((center[0])-squarePadding, 
#                                          (center[0])+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
#         cols = np.ndarray.flatten(arange((center[1])-squarePadding, 
#                                          (center[1])+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
        
#         medianCols.append(np.median(im[rows, cols, :-1], axis=0))
        
#         visiblePixels = np.argwhere(im[:, :, -1] != 0)
#         medianCols.append(np.mean(im[visiblePixels[:, 0], visiblePixels[:, 1], :-1], axis=0))

        diffIm = (np.sum(np.abs(bgIm-im[:, :, :-1]), axis=-1)*im[:, :, -1]/255.0)
        relevantPixels = np.argwhere(diffIm/np.max(diffIm) > 0.5)
    
        medianCols.append(np.mean(im[relevantPixels[:, 0], relevantPixels[:, 1], :-1], axis=0))
        
        sys.stdout.write('\r' + "Processed image " + np.string_(count) + " (" + np.string_(len(trackedSprites[i][DICT_FRAMES_LOCATIONS])) + ")")
        sys.stdout.flush()
    
    medianRGB = np.median(np.array(medianCols), axis=0)
    normed = medianRGB/np.linalg.norm(medianRGB)
    
    trackedSprites[i][DICT_MEDIAN_COLOR] = np.array(255/np.max(normed)*normed, dtype=int)
    
    print 
    print dataPath + dataSet + "sprite-" + "{0:04}".format(i) + "-" + trackedSprites[i][DICT_SPRITE_NAME] + ".npy", trackedSprites[i][DICT_MEDIAN_COLOR]
    np.save(dataPath + dataSet + "sprite-" + "{0:04}".format(i) + "-" + trackedSprites[i][DICT_SPRITE_NAME] + ".npy", trackedSprites[i])

# <codecell>

# print im.shape
# print np.median(np.array(medianCols), axis=0)
# imshow(im[rows, cols, :-1].reshape((5, 5, 3)))
figure(); imshow(im)
figure(); imshow(bgIm)
diffIm = (np.sum(np.abs(bgIm-im[:, :, :-1]), axis=-1)*im[:, :, -1]/255.0)
relevantPixels = np.argwhere(diffIm/np.max(diffIm) > 0.5)
gwv.showCustomGraph(diffIm/np.max(diffIm)> 0.5)

# <codecell>

print rows
print cols
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

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
    gm = opengm.gm(numpy.ones(numNodes,dtype=opengm.label_type)*numLabels)
    
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
    gm.addFactors(fids, arange(0, numNodes, 1))
    
    
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
        patchAFactors = np.hstack((patchAPixels.reshape((len(patchAPixels), 1)), np.ones((len(patchAPixels), 1), dtype=uint)*idxPatchANode))
        gm.addFactors(fid, patchAFactors)
    
    # connect patch B to definite patch B pixels
    if len(patchBPixels) > 0 :
        patchBFactors = np.hstack((patchBPixels.reshape((len(patchBPixels), 1)), np.ones((len(patchBPixels), 1), dtype=uint)*idxPatchBNode))
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

compatibilityMat = np.zeros((sprite1NumFrames, sprite2NumFrames))
numLabels = np.max((sprite1NumFrames, sprite2NumFrames))
# gwv.showCustomGraph(sprite2DistMat)
## adapt compatibility mats of all sprites with fewer frames
# compatibilityMat = np.concatenate((compatibilityMat, np.ones((numLabels-sprite1NumFrames, numLabels))*maxCost))
compatibilityMat = np.concatenate((compatibilityMat, np.zeros((numLabels-sprite1NumFrames, numLabels))))
compatibilityMat[sprite1NumFrames:, sprite1NumFrames:] = maxCost
compatibilityMat[4, 7] = compatibilityMat[7, 4] = maxCost#-100.0
# compatibilityMat = np.triu(compatibilityMat)+np.triu(compatibilityMat, 1).T
gwv.showCustomGraph(compatibilityMat)

# <codecell>

maxCost = 1000001.0

sprite1NumFrames = 700
sprite2NumFrames = 1300

sprite1DistMat = (1.0-np.eye(sprite1NumFrames, k=1))*maxCost
sprite1DistMat[-1, 0] = 0.0
sprite1DistMat[3, 5] = 10.0
# sprite1DistMat = sprite1DistMat.T

sprite2DistMat = (1.0-np.eye(sprite2NumFrames, k=1))*maxCost
sprite2DistMat[-1, 0] = 0.0
# sprite2DistMat = sprite2DistMat.T
compatibilityMat = np.zeros((sprite1NumFrames, sprite2NumFrames))
# gwv.showCustomGraph(sprite1DistMat)
# gwv.showCustomGraph(sprite2DistMat)
numLabels = np.max((sprite1NumFrames, sprite2NumFrames))
print numLabels
## adapt dist mats of all sprites with fewer frames
tmp = np.ones((numLabels, numLabels))*maxCost
tmp[:sprite1DistMat.shape[0], :sprite1DistMat.shape[1]] = sprite1DistMat
sprite1DistMat = tmp
# gwv.showCustomGraph(sprite1DistMat)
# gwv.showCustomGraph(sprite2DistMat)
## adapt compatibility mats of all sprites with fewer frames
compatibilityMat = np.concatenate((compatibilityMat, np.ones((numLabels-sprite1NumFrames, numLabels))*maxCost))
compatibilityMat[4, 7] = maxCost#-100.0
# compatibilityMat = np.triu(compatibilityMat)+np.triu(compatibilityMat, 1).T
# gwv.showCustomGraph(compatibilityMat)

seqLengths = [sprite1NumFrames, sprite2NumFrames]
distMats = [sprite1DistMat, sprite2DistMat]
compatibilityMats = {'00':np.zeros((numLabels, numLabels)),
                     '11':np.zeros((numLabels, numLabels)),
                     '01':compatibilityMat}

t = time.time()
tracks = np.random.randint(0, 2, 6) # [0, 1, 0]
startFrames = np.random.randint(0, 180, 6) # [2, 5, 0]

N = 20
numTracks = len(tracks)
numNodes = N*numTracks

gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels)

## cycle through all tracks eventually but for now have 1 sprite per track

## FIRST ADD THE ROWS OF THE GRAPH WITH THEIR UNARIES AND PAIRWISE
for trackEntity, startFrame, i in zip(tracks, startFrames, arange(len(tracks))) :
    
    ## THE UNARIES SHOULD DEPEND ON THE SEMANTICS PER ENTITY AND WITH MAX COST FOR THE LABELS
    ## THAT REPRESENT FRAMES THAT DO NOT EXIST FOR A GIVEN SPRITE (I.E. LEN(SPRITE_FRAMES) < NUM_LABELS)
    
    ## unaries
    unaries = np.zeros((N, numLabels))
    unaries[:, seqLengths[trackEntity]:] = maxCost
    unaries[0, :] = maxCost; unaries[0, startFrame] = 0.0 ## sets the third frame as the starting frame for sprite 1
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, arange(N*i, N*i+N))
    
    
    pairIndices = np.array([np.arange(N-1), np.arange(1, N)]).T + N*i
    
    ## add function for row-nodes pairwise cost
    fid = gm.addFunction(distMats[trackEntity])
    ## add second order factors
    gm.addFactors(fid, pairIndices)
    
## SECOND ADD THE PAIRWISE BETWEEN ROWS
for i, j in np.argwhere(np.triu(np.ones((len(tracks), len(tracks))), 1)) :
#     print i, j
#     print np.string_(np.min([tracks[i], tracks[j]])) + np.string_(np.max([tracks[i], tracks[j]]))
    pairIndices = np.array([np.arange(N*i, N*i+N), np.arange(N*j, N*j+N)]).T
#     print pairIndices
    
    ## add function for column-nodes pairwise cost
    if tracks[i] <= tracks[j] :
        fid = gm.addFunction(compatibilityMats[np.string_(np.min([tracks[i], tracks[j]])) + np.string_(np.max([tracks[i], tracks[j]]))])
    else :
        fid = gm.addFunction(compatibilityMats[np.string_(np.min([tracks[i], tracks[j]])) + np.string_(np.max([tracks[i], tracks[j]]))].T)
    ## add second order factors
    gm.addFactors(fid, pairIndices)

print time.time() - t
print gm

# <codecell>

print N
print startFrames
print tracks
t = time.time()
inferer = opengm.inference.TrwsExternal(gm=gm)
inferer.infer()

labels = np.array(inferer.arg(), dtype=int)
print time.time() - t

print gm.evaluate(labels)

for i in xrange(len(tracks)) :
    print labels[N*i:N*i+N]

# <codecell>

maxCost = 1000001.0

sprite1NumFrames = 200
sprite2NumFrames = 300

sprite1DistMat = (1.0-np.eye(sprite1NumFrames, k=1))*maxCost
sprite1DistMat[-1, 0] = 0.0
sprite1DistMat[3, 5] = 10.0
# sprite1DistMat = sprite1DistMat.T

sprite2DistMat = (1.0-np.eye(sprite2NumFrames, k=1))*maxCost
sprite2DistMat[-1, 0] = 0.0
# sprite2DistMat = sprite2DistMat.T
compatibilityMat = np.zeros((sprite1NumFrames, sprite2NumFrames))
# gwv.showCustomGraph(sprite1DistMat)
# gwv.showCustomGraph(sprite2DistMat)
# numLabels = np.max((sprite1NumFrames, sprite2NumFrames))
# print numLabels
## adapt dist mats of all sprites with fewer frames
# tmp = np.ones((numLabels, numLabels))*maxCost
# tmp[:sprite1DistMat.shape[0], :sprite1DistMat.shape[1]] = sprite1DistMat
# sprite1DistMat = tmp
# # gwv.showCustomGraph(sprite1DistMat)
# # gwv.showCustomGraph(sprite2DistMat)
# ## adapt compatibility mats of all sprites with fewer frames
# compatibilityMat = np.concatenate((compatibilityMat, np.ones((numLabels-sprite1NumFrames, numLabels))*maxCost))
compatibilityMat[4, 7] = maxCost#-100.0
# compatibilityMat = np.triu(compatibilityMat)+np.triu(compatibilityMat, 1).T
# gwv.showCustomGraph(compatibilityMat)

seqLengths = np.array([sprite1NumFrames, sprite2NumFrames])
distMats = [sprite1DistMat, sprite2DistMat]
compatibilityMats = {'00':np.zeros((seqLengths[0], seqLengths[0])),
                     '11':np.zeros((seqLengths[1], seqLengths[1])),
                     '01':compatibilityMat}

t = time.time()
# tracks = np.random.randint(0, 2, 6) # [0, 1, 0]
# tracks = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
startFrames = np.random.randint(180, 195, 6) # [2, 5, 0]

N = 51
numTracks = len(tracks)
numNodes = N*numTracks

gm = opengm.gm(seqLengths[tracks].repeat(N))

## cycle through all tracks eventually but for now have 1 sprite per track

## FIRST ADD THE ROWS OF THE GRAPH WITH THEIR UNARIES AND PAIRWISE
for trackEntity, startFrame, i in zip(tracks, startFrames, arange(len(tracks))) :
    
    ## THE UNARIES SHOULD DEPEND ON THE SEMANTICS PER ENTITY AND WITH MAX COST FOR THE LABELS
    ## THAT REPRESENT FRAMES THAT DO NOT EXIST FOR A GIVEN SPRITE (I.E. LEN(SPRITE_FRAMES) < NUM_LABELS)
    
    ## unaries
    unaries = np.zeros((N, seqLengths[trackEntity]))
#     unaries[:, seqLengths[trackEntity]:] = maxCost
    unaries[0, :] = maxCost; unaries[0, startFrame] = 0.0 ## sets the third frame as the starting frame for sprite 1
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, arange(N*i, N*i+N))
    
    
    pairIndices = np.array([np.arange(N-1), np.arange(1, N)]).T + N*i
    
    ## add function for row-nodes pairwise cost
    fid = gm.addFunction(distMats[trackEntity])
    ## add second order factors
    gm.addFactors(fid, pairIndices)
    
## SECOND ADD THE PAIRWISE BETWEEN ROWS
for i, j in np.argwhere(np.triu(np.ones((len(tracks), len(tracks))), 1)) :
#     print i, j
#     print np.string_(np.min([tracks[i], tracks[j]])) + np.string_(np.max([tracks[i], tracks[j]]))
    pairIndices = np.array([np.arange(N*i, N*i+N), np.arange(N*j, N*j+N)]).T
#     print pairIndices
    
    ## add function for column-nodes pairwise cost
    if tracks[i] <= tracks[j] :
        fid = gm.addFunction(compatibilityMats[np.string_(np.min([tracks[i], tracks[j]])) + np.string_(np.max([tracks[i], tracks[j]]))])
    else :
        fid = gm.addFunction(compatibilityMats[np.string_(np.min([tracks[i], tracks[j]])) + np.string_(np.max([tracks[i], tracks[j]]))].T)
    ## add second order factors
    gm.addFactors(fid, pairIndices)

print time.time() - t
print gm
print tracks

# <codecell>

print N
print startFrames
print tracks
t = time.time()
inferer = opengm.inference.TrwsExternal(gm=gm)
inferer.infer()

labels = np.array(inferer.arg(), dtype=int)
print time.time() - t

print gm.evaluate(labels)

for i in xrange(len(tracks)) :
    print labels[N*i:N*i+N]

# <codecell>

def get3WayLabelling(patchA, patchB, patchC, patchAPixels, patchBPixels, patchCPixels, multiplier, unaryPriorPatchA, unaryPriorPatchB, unaryPriorPatchC) :
    """Computes pixel labels using graphcut given two same size patches
    
        \t  patchA           : patch A
        \t  patchB           : patch B
        \t  patchAPixels     : pixels that are definitely to be taken from patch A
        \t  patchBPixels     : pixels that are definitely to be taken from patch B
        \t  multiplier       : sigma multiplier for rgb space normal
        \t  unaryPriorPatchA : prior cost ditribution for patchA labels
        \t  unaryPriorPatchB : prior cost ditribution for patchB labels
           
        return: reshapedLabels = labels for each pixel"""
    
    if np.all(patchA.shape != patchB.shape) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
        
    if patchA.dtype != np.float64 or patchB.dtype != np.float64 :
        raise Exception("The two specified patches are not of type float64! Check there is no overflow when computing costs")
    
    h, width = patchA.shape[0:2]
    maxCost = 10000000.0#np.sys.float_info.max
    
    ## build graph
    numLabels = 3
    numNodes = h*width+numLabels
    gm = opengm.gm(numpy.ones(numNodes,dtype=opengm.label_type)*numLabels)
    
    ## Last 3 nodes are patch A, B and C respectively
    idxPatchANode = numNodes - 3
    idxPatchBNode = numNodes - 2
    idxPatchCNode = numNodes - 1
    
        
    ## get unary functions
    unaries = np.zeros((numNodes,numLabels))
    
    print h, width, unaries.shape
    
    ## fix label for nodes representing patch A and B to have label 0 and 1 respectively
    unaries[idxPatchANode, :] = [0.0, maxCost, maxCost]
    unaries[idxPatchBNode, :] = [maxCost, 0.0, maxCost]
    unaries[idxPatchCNode, :] = [maxCost, maxCost, 0.0]
    
    ## set unaries based on the priors given for both patches
    unaries[0:h*width, 0] = unaryPriorPatchA
    unaries[0:h*width, 1] = unaryPriorPatchB
    unaries[0:h*width, 2] = unaryPriorPatchC
    
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, arange(0, numNodes, 1))
    
    
    ## get factor indices for the overlap grid of pixels
    pairIndices = np.array(opengm.secondOrderGridVis(width,h,True))
    ## get pairwise functions for those nodes
    pairwiseAB = np.zeros((len(pairIndices), 1))
    pairwiseAC = np.zeros((len(pairIndices), 1))
    pairwiseBC = np.zeros((len(pairIndices), 1))
    for pair, i in zip(pairIndices, arange(len(pairIndices))) :
        sPix = np.array([int(np.mod(pair[0],h)), int(pair[0]/h)])
        tPix = np.array([int(np.mod(pair[1],h)), int(pair[1]/h)])
        
#         pairwise[i] = norm(patchA[sPix[0], sPix[1], :] - patchB[sPix[0], sPix[1], :])
#         pairwise[i] += norm(patchA[tPix[0], tPix[1], :] - patchB[tPix[0], tPix[1], :])

        pairwiseAB[i] = minusLogMultivariateNormal(patchA[sPix[0], sPix[1], :].reshape((3, 1)), patchB[sPix[0], sPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
        pairwiseAB[i] += minusLogMultivariateNormal(patchA[tPix[0], tPix[1], :].reshape((3, 1)), patchB[tPix[0], tPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
        
        pairwiseAC[i] = minusLogMultivariateNormal(patchA[sPix[0], sPix[1], :].reshape((3, 1)), patchC[sPix[0], sPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
        pairwiseAC[i] += minusLogMultivariateNormal(patchA[tPix[0], tPix[1], :].reshape((3, 1)), patchC[tPix[0], tPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
        
#         pairwiseBC[i] = minusLogMultivariateNormal(patchB[sPix[0], sPix[1], :].reshape((3, 1)), patchC[sPix[0], sPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
#         pairwiseBC[i] += minusLogMultivariateNormal(patchB[tPix[0], tPix[1], :].reshape((3, 1)), patchC[tPix[0], tPix[1], :].reshape((3, 1)), np.eye(3)*multiplier, False)
        ## sPix is above tPix
        if sPix[0] < tPix[0] :
            pairwiseBC[i] = gradientCostsY[sPix[0], sPix[1]] + gradientCostsY[tPix[0], tPix[1]]
        ## sPix is to the left of tPix
        elif sPix[1] < tPix[1] :
            pairwiseBC[i] = gradientCostsY[sPix[0], sPix[1]] + gradientCostsY[tPix[0], tPix[1]]
        else :
            print "OH OH", sPix, tPix
            
#         pairwiseBC[i] = maxCost
        
        pairwiseCB = np.copy(pairwiseBC[i])
#         if sPix[0] < tPix[0] :
#             pairwiseCB = pairwiseCB*10.0
        
        fid = gm.addFunction(np.array([[0.0, pairwiseAB[i], pairwiseAC[i]],[pairwiseAB[i], 0.0, pairwiseBC[i]],[pairwiseAC[i], pairwiseBC[i], 0.0]]))
        gm.addFactor(fid, pair)
            
    
    # add function used for connecting the patch variables
    fid = gm.addFunction(np.array([[0.0, maxCost, maxCost],[maxCost, 0.0, maxCost],[maxCost, maxCost, 0.0]]))
    
    # connect patch A to definite patch A pixels
    if len(patchAPixels) > 0 :
        patchAFactors = np.hstack((patchAPixels.reshape((len(patchAPixels), 1)), np.ones((len(patchAPixels), 1), dtype=uint)*idxPatchANode))
        gm.addFactors(fid, patchAFactors)
    
    # connect patch B to definite patch B pixels
    if len(patchBPixels) > 0 :
        patchBFactors = np.hstack((patchBPixels.reshape((len(patchBPixels), 1)), np.ones((len(patchBPixels), 1), dtype=uint)*idxPatchBNode))
        gm.addFactors(fid, patchBFactors)
        
    # connect patch B to definite patch B pixels
    if len(patchCPixels) > 0 :
        patchCFactors = np.hstack((patchCPixels.reshape((len(patchCPixels), 1)), np.ones((len(patchCPixels), 1), dtype=uint)*idxPatchCNode))
        gm.addFactors(fid, patchCFactors)
    
    
#     graphCut = opengm.inference.GraphCut(gm=gm)
    trws = opengm.inference.TrwsExternal(gm=gm)
    trws.infer()
    
    labels = np.array(trws.arg(), dtype=int)
    print labels.shape
    
    reshapedLabels = np.reshape(np.copy(labels[0:-numLabels]), patchA.shape[0:2], 'F')
    print gm
    
    return reshapedLabels, unaries, np.hstack((pairwiseAB, pairwiseAC, pairwiseBC)), gm

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
    
    patAPixs = np.empty(0, dtype=uint)
    patBPixs = np.empty(0, dtype=uint)
    
    ## force small square of size squarePadding*2 + 1 around center of patch to come from patch B (i.e. the car)
    if useCenterSquare :
        squarePadding = 6
        rows = np.ndarray.flatten(arange((h/2)-squarePadding, (h/2)+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
        cols = np.ndarray.flatten(arange((w/2)-squarePadding, (w/2)+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
        patBPixs = np.unique(np.concatenate((patBPixs, np.array(rows + cols*h, dtype=uint))))
    
    ## force one ring of pixels on the edge of the patch to come from patch A (i.e. the bg) (unless that column/row is intersected by the bbox)
#     if np.min((largeBBox)[0, :]) > 0.0 :
    if touchedBorders[0] :
#         print "adding left column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h, dtype=uint)[1:-1])))
    else :
#         print "adding left column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h, dtype=uint)[1:-1])))
#     if np.min((largeBBox)[1, :]) > 0.0 :
    if touchedBorders[1] :
#         print "adding top row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1])))
    else :
#         print "adding top row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1])))
#     if np.max((largeBBox)[1, :]) < bgImage.shape[0] :
    if touchedBorders[2] :
#         print "adding bottom row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1]+h-1)))
    else :
#         print "adding bottom row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1]+h-1)))
#     if np.max((largeBBox)[0, :]) < bgImage.shape[1] :
    if touchedBorders[3] :
#         print "adding right column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=uint)[1:-1])))
    else :
#         print "adding right column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(h*(w-1), h*w, dtype=uint)[1:-1])))
    
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
            fixedPixels[np.array(np.mod(patAPixs, patchSize[0]), dtype=uint), np.array(patAPixs/patchSize[0], dtype=uint)] = 1
        ## 2 == fg pixels (get 2d coords from 1d first)
        if len(patAPixs) > 0 :
            fixedPixels[np.array(np.mod(patBPixs, patchSize[0]), dtype=uint), np.array(patBPixs/patchSize[0], dtype=uint)] = 2
        
        if len(bgPixs) > 0 :
            fixedPixels[bgPixs[:, 0], bgPixs[:, 1]] = 1
        if len(fgPixs) > 0 :
            fixedPixels[fgPixs[:, 0], fgPixs[:, 1]] = 2

        ## turn back to 1d indices
        patAPixs = np.argwhere(fixedPixels == 1)
        patAPixs = np.sort(patAPixs[:, 0] + patAPixs[:, 1]*patchSize[0])
        patBPixs = np.argwhere(fixedPixels == 2)
        patBPixs = np.sort(patBPixs[:, 0] + patBPixs[:, 1]*patchSize[0])
    
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

bgImage = np.array(Image.open("inception.png"))[:, :, 0:3]
bgPatch = np.array(Image.open("taraBG.png"))[:, :, 0:3]
spritePatch = np.array(Image.open("tara.png"))[:, :, 0:3]

allXs = arange(bgPatch.shape[1], dtype=float32).reshape((1, bgPatch.shape[1])).repeat(bgPatch.shape[0], axis=0)
allYs = arange(bgPatch.shape[0], dtype=float32).reshape((bgPatch.shape[0], 1)).repeat(bgPatch.shape[1], axis=1)

##### spritePatch, offset, patchSize, touchedBorders = getSpritePatch(trackedSprites[spriteIdx], f, bgImage.shape[1], bgImage.shape[0]) #####
patchSize = np.array(bgPatch.shape[0:2])
offset = np.array([[0], [0]])
touchedBorders = np.array([True, True, True, True])

##### bgPrior, spritePrior = getPatchPriors(bgPatch, spritePatch, offset, patchSize, trackedSprites[spriteIdx], f, 
#                                           prevFrameKey=np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys())[frameCount-1], prevFrameAlphaLoc=outputPath,
#                                           prevMaskImportance=0.2) #####

## get uniform prior for bg patch
bgPrior = -np.log(np.ones(patchSize)/np.prod(patchSize))

## get prior for sprite patch
spritePrior = np.zeros(patchSize)
xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))
data = np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys)))))

## get covariance and means of prior on patch by using the bbox
spriteBBox = np.array([[76, 48], [76+388, 48], [76+388, 48+374], [76, 48+374]], np.float).T### sprite[DICT_BBOXES][frameKey].T
segment1 = spriteBBox[:, 0] - spriteBBox[:, 1]
segment2 = spriteBBox[:, 1] - spriteBBox[:, 2]
sigmaX = np.linalg.norm(segment1)/3.7
sigmaY = np.linalg.norm(segment2)/3.7

rotRadians = 0.0## sprite[DICT_BBOX_ROTATIONS][frameKey]

rotMat = np.array([[np.cos(rotRadians), -np.sin(rotRadians)], [np.sin(rotRadians), np.cos(rotRadians)]])

means = np.reshape(np.array([76+388/2, 48+374/2]), (2, 1)) - offset ### np.reshape(sprite[DICT_BBOX_CENTERS][frameKey], (2, 1)) - offset
covs = np.dot(np.dot(rotMat.T, np.array([[sigmaX**2, 0.0], [0.0, sigmaY**2]])), rotMat)

spritePrior = np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F')

labels = mergePatches(bgPatch, spritePatch, bgPrior, spritePrior, offset, patchSize, touchedBorders, useCenterSquare=False)

outputPatch = np.zeros((bgPatch.shape[0], bgPatch.shape[1], bgPatch.shape[2]+1), dtype=uint8)
for i in xrange(labels.shape[0]) :
    for j in xrange(labels.shape[1]) :
        if labels[i, j] == 0 :
            ## patA stands for the bgPatch but I want to set the pixels here to 0 to save space
            outputPatch[i, j, 0:-1] = 0#bgPatch[i, j, :]
        else :
            outputPatch[i, j, 0:-1] = spritePatch[i, j, :]
            outputPatch[i, j, -1] = 255

            
bgOffset = np.array([762, 304])
currentFrame = np.array(bgImage, np.uint8)
currentFrame[bgOffset[0]:bgOffset[0]+patchSize[0], bgOffset[1]:bgOffset[1]+patchSize[1], :] = (currentFrame[bgOffset[0]:bgOffset[0]+patchSize[0], bgOffset[1]:bgOffset[1]+patchSize[1], :]*
                                                                                               (1.0-outputPatch[:, :, -1]/255.0).reshape((patchSize[0], patchSize[1], 1))+
                                                                                               outputPatch[:, :, :-1]*(outputPatch[:, :, -1]/255.0).reshape((patchSize[0], patchSize[1], 1)))

#     Image.fromarray((currentFrame).astype(numpy.uint8)).save(outputPath + trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f].split('/')[-1])
figure(); imshow(currentFrame)

# <codecell>

figure(); imshow(currentFrame)

# <codecell>

## load the tracked sprites
trackedSprites = [{
                   DICT_SPRITE_NAME:'havana_red_car_plusrot', 
                   DICT_BBOX_AFFINES:sio.loadmat("../ASLA tracker/result/havana_red_car_plusrot/Result/result.mat")['result'], 
                   DICT_NUM_FRAMES:0, 
                   DICT_FRAMES_LOCATIONS:[]
                   }, 
                  {
                   DICT_SPRITE_NAME:'havana_bus', 
                   DICT_BBOX_AFFINES:sio.loadmat("../ASLA tracker/result/havana_bus/Result/result.mat")['result'], 
                   DICT_NUM_FRAMES:0, 
                   DICT_FRAMES_LOCATIONS:[]
                   }
                  ]

numOfSprites = len(trackedSprites)
## setting number of frames from the number of tracked bboxes
for i in arange(numOfSprites) :
    trackedSprites[i][DICT_NUM_FRAMES] = len(trackedSprites[i][DICT_BBOX_AFFINES])
## setting frame locations for the tracked sprites
for i in arange(numOfSprites) :
    trackedSprites[i][DICT_FRAMES_LOCATIONS] = np.sort(glob.glob("../ASLA tracker/Datasets/" + trackedSprites[i][DICT_SPRITE_NAME] + "/*.png"))

## default corners of bbox to be transformed by the affine matrix transformation using function below
## [x, y] coords for each corner
bboxDefaultCorners = np.array([ [1,-16,-16], [1,16,-16], [1,16,16], [1,-16,16], [1,-16,-16] ]).T
def getAffMat(p) :
    return np.array([[p[0], p[2], p[3]], [p[1], p[4], p[5]]]);

# <codecell>

## load background image
bgImage = np.array(Image.open("../data/havana/median.png"))

## frame index of sprites to merge together
currentFramePerSprite = [263, 160]

## get the bboxes for each sprite at current frame and find the offset and size of the subpatch of the total image to work with
## boundaries of the patch [min, max]
xBounds = np.array([bgImage.shape[1], 0.0])
yBounds = np.array([bgImage.shape[0], 0.0])
figure(); imshow(bgImage)
for i in arange(numOfSprites) :
    ## plot bbox
    spriteBBox = np.dot(getAffMat(trackedSprites[i][DICT_BBOX_AFFINES][currentFramePerSprite[i], :]), bboxDefaultCorners)
    plot(spriteBBox[0, :], spriteBBox[1, :])
    
    ## make bbox bigger
    largeBBox = np.dot(np.array([[0.0, 0.0, 1.0+PATCH_BORDER], [0.0, 1.0+PATCH_BORDER, 0.0]]), bboxDefaultCorners)
    ## transform according to affine transformation
    largeBBox = np.dot(getAffMat(trackedSprites[i][DICT_BBOX_AFFINES][currentFramePerSprite[i], :]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    
    xBounds[0] = np.min((xBounds[0], np.min(largeBBox[0, :])))
    xBounds[1] = np.max((xBounds[1], np.max(largeBBox[0, :])))
    yBounds[0] = np.min((yBounds[0], np.min(largeBBox[1, :])))
    yBounds[1] = np.max((yBounds[1], np.max(largeBBox[1, :])))
    
offset = np.array([np.round(np.array([xBounds[0], yBounds[0]]))], dtype=int).T # [x, y]
patchSize = np.array(np.round(np.array([yBounds[1]-yBounds[0], xBounds[1]-xBounds[0]])), dtype=int) # [rows, cols]
print offset, patchSize

plot([offset[0], offset[0]+patchSize[1], offset[0]+patchSize[1], offset[0], offset[0]], 
     [offset[1], offset[1], offset[1]+patchSize[0], offset[1]+patchSize[0], offset[1]])

# <codecell>

## get image patches based on offset and patchSize
bgPatch = np.copy(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
figure(); imshow(bgPatch)

spritePatches = np.zeros((numOfSprites, patchSize[0], patchSize[1], bgImage.shape[-1]), dtype=uint8)
for i in arange(numOfSprites) :
    spritePatches[i, :, :, :] = np.array(Image.open(trackedSprites[i][DICT_FRAMES_LOCATIONS][currentFramePerSprite[i]]))[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :]
    figure(); imshow(spritePatches[i, :, :, :])

# <codecell>

figure(); imshow(np.sum(np.power(spritePatches[0]-bgPatch, 2), axis=-1))

# <codecell>

## precompute pixel pairs for all edges in the patch
gridEdges1D = np.array(opengm.secondOrderGridVis(patchSize[1],patchSize[0],True))
gridEdges2D = np.zeros((len(gridEdges1D), 4))

gridEdges2D[:, 0] = np.mod(gridEdges1D[:, 0], patchSize[0])
gridEdges2D[:, 1] = np.array(gridEdges1D[:, 0]/patchSize[0], dtype=int)
gridEdges2D[:, 2] = np.mod(gridEdges1D[:, 1], patchSize[0])
gridEdges2D[:, 3] = np.array(gridEdges1D[:, 1]/patchSize[0], dtype=int)

# <codecell>

## get uniform prior for bg patch
bgPrior = -np.log(np.ones(patchSize)/np.prod(patchSize))

## get priors for all sprite patches
spritePriors = np.zeros((numOfSprites, patchSize[0], patchSize[1]))
xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))
data = np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys)))))

for i in arange(numOfSprites) :
    ## get covariance and means of prior on patch by using the bbox
    spriteBBox = np.dot(getAffMat(trackedSprites[i][DICT_BBOX_AFFINES][currentFramePerSprite[i], :]), bboxDefaultCorners)
    segment1 = spriteBBox[:, 0] - spriteBBox[:, 1]
    segment2 = spriteBBox[:, 1] - spriteBBox[:, 2]
    sigmaX = np.linalg.norm(segment1)/2.0
    sigmaY = np.linalg.norm(segment2)/2.0

    ## find rotation as described here http://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation
    A11 = getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][f, :])[0, 1]
    A21 = getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][f, :])[1, 1]
    rotRadians = np.pi-np.arctan(A21/A11)

    rotMat = np.array([[np.cos(rotRadians), -np.sin(rotRadians)], [np.sin(rotRadians), np.cos(rotRadians)]])
    
    means = np.reshape(trackedSprites[i][DICT_BBOX_AFFINES][currentFramePerSprite[i], 0:2], (2, 1)) - offset
    covs = np.dot(np.dot(rotMat.T, np.array([[sigmaX**2, 0.0], [0.0, sigmaY**2]])), rotMat)
    
    print sigmaX, sigmaY, rotRadians, means
    spritePriors[i, :, :] = np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F')
    
    
    figure(); imshow(spritePriors[i])
#     gwv.showCustomGraph(np.reshape(multivariateNormal(data, means, covs, True), patchSize, order='F'))
#     gwv.showCustomGraph(np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F'))

# <codecell>

figure(); imshow(spritePatches[0])
sobelY = np.array([[-1, -1, -1], [0, 0, 0], [1, 2, 1]])
sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gradientsY = cv2.filter2D(cv2.cvtColor(spritePatches[0], cv2.cv.CV_RGB2GRAY)/255.0, -1, sobelY)
gradientsX = cv2.filter2D(cv2.cvtColor(spritePatches[0], cv2.cv.CV_RGB2GRAY)/255.0, -1, sobelX)
figure(); imshow(gradientsY, interpolation='nearest', cmap=get_cmap("Greys"))
figure(); imshow(gradientsX, interpolation='nearest', cmap=get_cmap("Greys"))

# <codecell>

gradientCostsY = np.exp(np.abs(gradientsY)/(0.3*np.mean(np.abs(gradientsY))))
gradientCostsY /= np.sum(gradientCostsY)
gradientCostsY = -log(gradientCostsY)/10.0
gradientCostsX = np.exp(np.abs(gradientsX)/(0.3*np.mean(np.abs(gradientsX))))
gradientCostsX /= np.sum(gradientCostsX)
gradientCostsX = -log(gradientCostsX)/10.0
#figure(); imshow
gwv.showCustomGraph(gradientCostsY)#, interpolation='nearest', cmap=get_cmap("Greys"))
#figure(); imshow
gwv.showCustomGraph(gradientCostsY)#, interpolation='nearest', cmap=get_cmap("Greys"))

# <codecell>

print np.max(gradientCostsX), np.max(gradientCostsY), np.min(gradientCostsX), np.min(gradientCostsY)

# <codecell>

## get depth prior for bg by fitting a plane to the floor
userDefinedPoints = np.array([[291, 713], [336, 945], [337, 642], [398, 879]], dtype=float)
defaultPlanePoints = np.array([[0, 0], [0, patchSize[1]], [patchSize[0], 0], [patchSize[0], patchSize[1]]], dtype=float)
# defaultPlanePoints = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
figure(); imshow(bgImage)
scatter(userDefinedPoints[:, 1], userDefinedPoints[:, 0])

# hom = cv2.findHomography(defaultPlanePoints, userDefinedPoints)[0]
hom = cv2.findHomography(userDefinedPoints, defaultPlanePoints)[0]
print hom

# <codecell>

figure();
transformedGrid = np.dot(hom, np.concatenate((userDefinedPoints.T, np.ones((1, userDefinedPoints.T.shape[-1]))), axis=0))
transformedGrid /= transformedGrid[-1, :]
scatter(transformedGrid[1, :], transformedGrid[0, :])

# <codecell>

print transformedGrid

# <codecell>

print np.concatenate((userDefinedPoints.T, np.ones((1, userDefinedPoints.T.shape[-1]))), axis=0)

# <codecell>

figure(); imshow(bgImage)
xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))

pointGrid = np.vstack((ys.reshape((1, len(ys))), xs.reshape((1, len(xs)))))
transformedGrid = np.dot(hom, np.concatenate((pointGrid, np.ones((1, pointGrid.shape[-1]))), axis=0))
transformedGrid /= transformedGrid[-1, :]
scatter(transformedGrid[1, 0:-1:100], transformedGrid[0, 0:-1:100])
# scatter(data[1, 0:-1:100], data[0, 0:-1:100], marker='.')

# <codecell>

## get grid of points with origin in the middle of the image
gridXs = np.ndarray.flatten(np.arange(-bgImage.shape[1]/2, bgImage.shape[1]/2, dtype=float).reshape((bgImage.shape[1], 1)).repeat(bgImage.shape[0], axis=-1))
gridYs = np.ndarray.flatten(np.arange(-bgImage.shape[0]/2, bgImage.shape[0]/2, dtype=float).reshape((1, bgImage.shape[0])).repeat(bgImage.shape[1], axis=0))

pointGrid = np.vstack((gridYs.reshape((1, len(gridYs))), gridXs.reshape((1, len(gridXs))), np.ones((1, len(gridXs)))))

# <codecell>

gridXs = np.ndarray.flatten(np.arange(bgImage.shape[1], dtype=float).reshape((bgImage.shape[1], 1)).repeat(bgImage.shape[0], axis=-1))
gridYs = np.ndarray.flatten(np.arange(bgImage.shape[0], dtype=float).reshape((1, bgImage.shape[0])).repeat(bgImage.shape[1], axis=0))

pointGrid = np.vstack((gridYs.reshape((1, len(gridYs))), gridXs.reshape((1, len(gridXs))), np.ones((1, len(gridXs)))))

# <codecell>

tmp = np.dot(hom, pointGrid)
tmp /= tmp[-1, :]
tmp[:, np.any(tmp > 1000, axis=0)] = 1000
tmp[:, np.any(tmp < -1000, axis=0)] = -1000

# print tmp.shape
gwv.showCustomGraph(tmp[0, :].reshape(bgImage.shape[0:2], order='F'))

# <codecell>

print np.any(tmp > 1000, axis=0).shape

# <codecell>

print np.max(tmp[1, :])

# <codecell>

## merge two overlapping patches

h = patchSize[0]
w = patchSize[1]

## force one ring of pixels on the edge of the patch to come from patch A (i.e. the bg)
patAPixs = np.arange(0, h, dtype=uint)
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint))))
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)+h-1)))
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=uint))))

## force small square of size squarePadding*2 + 1 around center of patch to come from patch B (i.e. the car)
squarePadding = 6
rows = np.ndarray.flatten(arange((h/2)-squarePadding, (h/2)+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
cols = np.ndarray.flatten(arange((w/2)-squarePadding, (w/2)+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
patBPixs = rows + cols*h
patBPixs = np.empty(0)

patA = np.copy(bgPatch/255.0)
patB = np.copy(spritePatches[0]/255.0)

labels, unaryCosts, pairCosts, graphModel = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, 0.001, 
                                                   bgPrior.reshape(np.prod(patchSize), order='F'),
                                                   spritePriors[0].reshape(np.prod(patchSize), order='F'))

figure(); imshow(labels, interpolation='nearest')

outputPatch = np.zeros(patA.shape, dtype=uint8)
for i in xrange(labels.shape[0]) :
    for j in xrange(labels.shape[1]) :
        if labels[i, j] == 0 :
            outputPatch[i, j, :] = patA[i, j, :]*255
        else :
            outputPatch[i, j, :] = patB[i, j, :]*255
            
figure(); imshow(outputPatch, interpolation='nearest')

# <codecell>

## merge three overlapping patches

h = patchSize[0]
w = patchSize[1]

## force one ring of pixels on the edge of the patch to come from patch A (i.e. the bg)
patAPixs = np.arange(0, h, dtype=uint)
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint))))
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)+h-1)))
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=uint))))

## force small square of size squarePadding*2 + 1 around center of patch to come from patch B (i.e. the car)
squarePadding = 6
rows = np.ndarray.flatten(arange((h/2)-squarePadding, (h/2)+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
cols = np.ndarray.flatten(arange((w/2)-squarePadding, (w/2)+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
patBPixs = rows + cols*h
patBPixs = np.empty(0)
patCPixs = np.empty(0)

patA = np.copy(bgPatch/255.0)
patB = np.copy(spritePatches[0]/255.0)
patC = np.copy(spritePatches[1]/255.0)

labels, unaryCosts, pairCosts, graphModel = get3WayLabelling(patA, patB, patC, patAPixs, patBPixs, patCPixs, 0.005, 
                                                   bgPrior.reshape(np.prod(patchSize), order='F'),
                                                   spritePriors[0].reshape(np.prod(patchSize), order='F'),
                                                   spritePriors[1].reshape(np.prod(patchSize), order='F'))
# labels, unaryCosts, pairCosts = get3WayLabelling(patA, patB, patC, patAPixs, patBPixs, patCPixs, 0.005, 
#                                                    bgPrior.reshape(np.prod(patchSize), order='F'),
#                                                    (spritePriors[0]*(1.0-weightedDiffAB/np.max(weightedDiffAB))).reshape(np.prod(patchSize), order='F'),
#                                                    (spritePriors[1]*(1.0-weightedDiffAC/np.max(weightedDiffAC))).reshape(np.prod(patchSize), order='F'))

figure(); imshow(labels, interpolation='nearest')

outputPatch = np.zeros(patA.shape, dtype=uint8)
for i in xrange(labels.shape[0]) :
    for j in xrange(labels.shape[1]) :
        if labels[i, j] == 0 :
            outputPatch[i, j, :] = patA[i, j, :]*255
        elif labels[i, j] == 1 :
            outputPatch[i, j, :] = patB[i, j, :]*255
        else :
            outputPatch[i, j, :] = patC[i, j, :]*255
            
figure(); imshow(outputPatch, interpolation='nearest')

# <codecell>

## load the tracked sprites
DICT_SPRITE_NAME = 'sprite_name'
# DICT_BBOX_AFFINES = 'bbox_affines'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
# DICT_NUM_FRAMES = 'num_frames'
# DICT_START_FRAME = 'start_frame'
DICT_FRAMES_LOCATIONS = 'frame_locs'

dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
bgImage = np.array(Image.open(dataPath + dataSet + "median.png"))

trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())

## merge tracked sprite with bg
spriteIdx = 0
sequenceLength = len(trackedSprites[spriteIdx][DICT_BBOXES])
showFigs = False

outputPath = dataPath + dataSet + trackedSprites[spriteIdx][DICT_SPRITE_NAME] + "-mergedWithBG/"

if outputPath != None and not os.path.isdir(outputPath):
    os.makedirs(outputPath)

if showFigs :
    figure(); imshow(bgImage)

print "processing", trackedSprites[spriteIdx][DICT_SPRITE_NAME]

startTime = time.time()
for f, frameCount in zip(np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys()), xrange(len(trackedSprites[spriteIdx][DICT_BBOXES].keys()))):#[1109:1110]:#sequenceLength):
    ## get the bbox for the current sprite frame, make it larger and find the rectangular patch to work with
    ## boundaries of the patch [min, max]
    
    s = time.time()
    xBounds = np.array([bgImage.shape[1], 0.0])
    yBounds = np.array([bgImage.shape[0], 0.0])
    
    
    if showFigs :
        ## plot bbox
        spriteBBox = np.vstack((trackedSprites[spriteIdx][DICT_BBOXES][f], trackedSprites[spriteIdx][DICT_BBOXES][f][0, :])).T
        plot(spriteBBox[0, :], spriteBBox[1, :])
    
    ## make bbox bigger
    largeBBox = trackedSprites[spriteIdx][DICT_BBOXES][f].T
    ## move to origin
    largeBBox = np.dot(np.array([[-trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][0], 1.0, 0.0], 
                                 [-trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## make bigger
    largeBBox = np.dot(np.array([[0.0, 1.0 + PATCH_BORDER, 0.0], 
                                 [0.0, 0.0, 1.0 + PATCH_BORDER]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## move back to original center
    largeBBox = np.dot(np.array([[trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][0], 1.0, 0.0], 
                                 [trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    
    
    ## make sure xBounds are in between 0 and width and yBounds are in between 0 and height
    xBounds[0] = np.max((0, np.min(largeBBox[0, :])))
    xBounds[1] = np.min((bgImage.shape[1], np.max(largeBBox[0, :])))
    yBounds[0] = np.max((0, np.min(largeBBox[1, :])))
    yBounds[1] = np.min((bgImage.shape[0], np.max(largeBBox[1, :])))
    
#     print xBounds, yBounds
    
    offset = np.array([np.round(np.array([xBounds[0], yBounds[0]]))], dtype=int).T # [x, y]
    patchSize = np.array(np.round(np.array([yBounds[1]-yBounds[0], xBounds[1]-xBounds[0]])), dtype=int) # [rows, cols]
#     print offset, patchSize
    
    if showFigs :
        plot([offset[0], offset[0]+patchSize[1], offset[0]+patchSize[1], offset[0], offset[0]], 
             [offset[1], offset[1], offset[1]+patchSize[0], offset[1]+patchSize[0], offset[1]])

    ## get image patches based on offset and patchSize
    bgPatch = np.copy(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
    
    if showFigs :
        figure(); imshow(bgPatch)
    
    
    spritePatch = np.copy(np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f]))[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
    
    if showFigs :
        figure(); imshow(spritePatch)
        
    
#     print "patch fetch took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
        
#     ## precompute pixel pairs for all edges in the patch
#     gridEdges1D = np.array(opengm.secondOrderGridVis(patchSize[1],patchSize[0],True))
#     print "grideEdges1D took", time.time()-s, "seconds"
#     sys.stdout.flush()
#     s = time.time()

#     gridEdges2D = np.zeros((len(gridEdges1D), 4))
    
#     gridEdges2D[:, 0] = np.mod(gridEdges1D[:, 0], patchSize[0])
#     gridEdges2D[:, 1] = np.array(gridEdges1D[:, 0]/patchSize[0], dtype=int)
#     gridEdges2D[:, 2] = np.mod(gridEdges1D[:, 1], patchSize[0])
#     gridEdges2D[:, 3] = np.array(gridEdges1D[:, 1]/patchSize[0], dtype=int)

#     print "grideEdges2D took", time.time()-s, "seconds"
#     sys.stdout.flush()
#     s = time.time()

    ## get uniform prior for bg patch
    bgPrior = -np.log(np.ones(patchSize)/np.prod(patchSize))
    
    ## get priors for all sprite patches
    spritePrior = np.zeros(patchSize)
    xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
    ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))
    data = np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys)))))
    
    ## get covariance and means of prior on patch by using the bbox
    spriteBBox = trackedSprites[spriteIdx][DICT_BBOXES][f].T
    segment1 = spriteBBox[:, 0] - spriteBBox[:, 1]
    segment2 = spriteBBox[:, 1] - spriteBBox[:, 2]
    sigmaX = np.linalg.norm(segment1)/3.7
    sigmaY = np.linalg.norm(segment2)/3.7

#     ## find rotation as described here http://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation
#     A11 = getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][f, :])[0, 1]
#     A21 = getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][f, :])[1, 1]
# #     rotRadians = np.pi-np.arctan(A21/A11)
#     rotRadians = np.arccos(np.dot((segment1)/np.linalg.norm(segment1), np.array([1.0, 0.0])))
#     b = np.array([1.0, 0.0])
#     ## arctan2 is computed based on formula: a[0]*b[1]-b[0]*a[1], a[0]*b[0]+a[1]*b[1]) where b = [1, 0] and a = segment1
#     rotRadians = np.mod(np.arctan2(-segment1[1], segment1[0]),2*np.pi)
#     print "rotation", rotRadians*180.0/np.pi
    rotRadians = trackedSprites[spriteIdx][DICT_BBOX_ROTATIONS][f]
    
    rotMat = np.array([[np.cos(rotRadians), -np.sin(rotRadians)], [np.sin(rotRadians), np.cos(rotRadians)]])
    
    means = np.reshape(trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f], (2, 1)) - offset
    covs = np.dot(np.dot(rotMat.T, np.array([[sigmaX**2, 0.0], [0.0, sigmaY**2]])), rotMat)
    
#     print sigmaX, sigmaY, rotRadians, means
    spritePrior = np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F')
    
#     print "sprite prior took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    
    
    if showFigs :
        figure(); imshow(spritePrior)
#         gwv.showCustomGraph(np.reshape(multivariateNormal(data, means, covs, True), patchSize, order='F'))
#         gwv.showCustomGraph(np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F'))

    ## merge two overlapping patches

    h = patchSize[0]
    w = patchSize[1]
    
    patAPixs = np.empty(0, dtype=uint)
    patBPixs = np.empty(0, dtype=uint)
    
    ## force small square of size squarePadding*2 + 1 around center of patch to come from patch B (i.e. the car)
    squarePadding = 6
    rows = np.ndarray.flatten(arange((h/2)-squarePadding, (h/2)+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
    cols = np.ndarray.flatten(arange((w/2)-squarePadding, (w/2)+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
    patBPixs = np.unique(np.concatenate((patBPixs, np.array(rows + cols*h, dtype=uint))))
    
    ## force one ring of pixels on the edge of the patch to come from patch A (i.e. the bg) (unless that column/row is intersected by the bbox)
    if np.min((largeBBox)[0, :]) > 0.0 :
#         print "adding left column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h, dtype=uint)[1:-1])))
    else :
#         print "adding left column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h, dtype=uint)[1:-1])))
    if np.min((largeBBox)[1, :]) > 0.0 :
#         print "adding top row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1])))
    else :
#         print "adding top row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1])))
    if np.max((largeBBox)[1, :]) < bgImage.shape[0] :
#         print "adding bottom row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1]+h-1)))
    else :
#         print "adding bottom row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1]+h-1)))
    if np.max((largeBBox)[0, :]) < bgImage.shape[1] :
#         print "adding right column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=uint)[1:-1])))
    else :
#         print "adding right column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(h*(w-1), h*w, dtype=uint)[1:-1])))
    
#     patBPixs = np.empty(0)
    
    patA = np.copy(bgPatch/255.0)
    patB = np.copy(spritePatch/255.0)
    
#     print "patch pixels took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    
    labels, unaryCosts, pairCosts, graphModel = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, 0.001, 
                                                       bgPrior.reshape(np.prod(patchSize), order='F'),
                                                       spritePrior.reshape(np.prod(patchSize), order='F'))
    
#     print "graphcut took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    
    if showFigs :
        figure(); imshow(labels, interpolation='nearest')
    
    outputPatch = np.zeros(patA.shape, dtype=uint8)
    for i in xrange(labels.shape[0]) :
        for j in xrange(labels.shape[1]) :
            if labels[i, j] == 0 :
                outputPatch[i, j, :] = patA[i, j, :]*255
            else :
                outputPatch[i, j, :] = patB[i, j, :]*255
    
    if showFigs :
        figure(); imshow(outputPatch, interpolation='nearest')
        
        
    currentFrame = np.copy(bgImage)
    currentFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = np.copy(outputPatch)
    
    frameMask = np.zeros(bgImage.shape, dtype=np.uint8)
#     frameMask[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = labels*255
    ## offest is (x, y) whereas argwhere returns (row, col)
    fgIdxs = np.argwhere(labels == 1) + offset[::-1].T
    frameMask[fgIdxs[:, 0], fgIdxs[:, 1], :] = 255
    
    if showFigs :
        figure(); imshow(currentFrame)
        figure(); imshow(np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f])))
    
#     saveLoc = outputPath + trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f].split('/')[-1]
    
    Image.fromarray((currentFrame).astype(numpy.uint8)).save(outputPath + trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f].split('/')[-1])
    Image.fromarray((frameMask).astype(numpy.uint8)).save(outputPath + "mask-" + trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f].split('/')[-1])
    
#     print "saving took", time.time()-s, "seconds"
#     sys.stdout.flush()
    sys.stdout.write('\r' + "Done " + np.string_(frameCount+1) + " images of " + np.string_(len(trackedSprites[spriteIdx][DICT_BBOXES].keys())))
    sys.stdout.flush()
    
print 
print "total time:", time.time() - startTime

# <codecell>

## load the tracked sprites
DICT_SPRITE_NAME = 'sprite_name'
# DICT_BBOX_AFFINES = 'bbox_affines'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
# DICT_NUM_FRAMES = 'num_frames'
# DICT_START_FRAME = 'start_frame'
DICT_FRAMES_LOCATIONS = 'frame_locs'

# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
dataSet = "clouds_subsample10/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
    print trackedSprites[-1][DICT_SPRITE_NAME]

# <codecell>

## load the tracked sprites
DICT_SPRITE_NAME = 'sprite_name'
# DICT_BBOX_AFFINES = 'bbox_affines'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
# DICT_NUM_FRAMES = 'num_frames'
# DICT_START_FRAME = 'start_frame'
DICT_FRAMES_LOCATIONS = 'frame_locs'

<<<<<<< HEAD
dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
=======
# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
dataSet = "theme_park_sunny/"
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
bgImage = np.array(Image.open(dataPath + dataSet + "median.png"))

trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())

## merge tracked sprite with bg
<<<<<<< HEAD
spriteIdx = 2
=======
spriteIdx = 0
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
sequenceLength = len(trackedSprites[spriteIdx][DICT_BBOXES])
showFigs = True

outputPath = dataPath + dataSet + trackedSprites[spriteIdx][DICT_SPRITE_NAME] + "-maskedFlow/" #"-masked/"

if outputPath != None and not os.path.isdir(outputPath):
    os.makedirs(outputPath)

if showFigs :
    figure(); imshow(bgImage)
    
print "processing", trackedSprites[spriteIdx][DICT_SPRITE_NAME]

allXs = arange(bgImage.shape[1], dtype=float32).reshape((1, bgImage.shape[1])).repeat(bgImage.shape[0], axis=0)
allYs = arange(bgImage.shape[0], dtype=float32).reshape((bgImage.shape[0], 1)).repeat(bgImage.shape[1], axis=1)

startTime = time.time()
for f, frameCount in zip(np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys())[92:93], arange(len(trackedSprites[spriteIdx][DICT_BBOXES].keys()))[92:93]):#[1109:1110]:#sequenceLength):
    ## get the bbox for the current sprite frame, make it larger and find the rectangular patch to work with
    ## boundaries of the patch [min, max]
    
    s = time.time()
    xBounds = np.array([bgImage.shape[1], 0.0])
    yBounds = np.array([bgImage.shape[0], 0.0])
    
    
    if showFigs :
        ## plot bbox
        spriteBBox = np.vstack((trackedSprites[spriteIdx][DICT_BBOXES][f], trackedSprites[spriteIdx][DICT_BBOXES][f][0, :])).T
        plot(spriteBBox[0, :], spriteBBox[1, :])
    
    ## make bbox bigger
    largeBBox = trackedSprites[spriteIdx][DICT_BBOXES][f].T
    ## move to origin
    largeBBox = np.dot(np.array([[-trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][0], 1.0, 0.0], 
                                 [-trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## make bigger
    largeBBox = np.dot(np.array([[0.0, 1.0 + PATCH_BORDER, 0.0], 
                                 [0.0, 0.0, 1.0 + PATCH_BORDER]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    ## move back tooriginal center
    largeBBox = np.dot(np.array([[trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][0], 1.0, 0.0], 
                                 [trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f][1], 0.0, 1.0]]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    
    
    ## make sure xBounds are in between 0 and width and yBounds are in between 0 and height
    xBounds[0] = np.max((0, np.min(largeBBox[0, :])))
    xBounds[1] = np.min((bgImage.shape[1], np.max(largeBBox[0, :])))
    yBounds[0] = np.max((0, np.min(largeBBox[1, :])))
    yBounds[1] = np.min((bgImage.shape[0], np.max(largeBBox[1, :])))
    
#     print xBounds, yBounds
    
    offset = np.array([np.round(np.array([xBounds[0], yBounds[0]]))], dtype=int).T # [x, y]
    patchSize = np.array(np.round(np.array([yBounds[1]-yBounds[0], xBounds[1]-xBounds[0]])), dtype=int) # [rows, cols]
#     print offset, patchSize
    
    if showFigs :
        plot([offset[0], offset[0]+patchSize[1], offset[0]+patchSize[1], offset[0], offset[0]], 
             [offset[1], offset[1], offset[1]+patchSize[0], offset[1]+patchSize[0], offset[1]])

    ## get image patches based on offset and patchSize
    bgPatch = np.copy(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
    
    if showFigs :
        figure(); imshow(bgPatch)
    
    
    spritePatch = np.copy(np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f]))[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
    
    if showFigs :
        figure(); imshow(spritePatch)
        
    
#     print "patch fetch took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
        
#     ## precompute pixel pairs for all edges in the patch
#     gridEdges1D = np.array(opengm.secondOrderGridVis(patchSize[1],patchSize[0],True))
#     print "grideEdges1D took", time.time()-s, "seconds"
#     sys.stdout.flush()
#     s = time.time()

#     gridEdges2D = np.zeros((len(gridEdges1D), 4))
    
#     gridEdges2D[:, 0] = np.mod(gridEdges1D[:, 0], patchSize[0])
#     gridEdges2D[:, 1] = np.array(gridEdges1D[:, 0]/patchSize[0], dtype=int)
#     gridEdges2D[:, 2] = np.mod(gridEdges1D[:, 1], patchSize[0])
#     gridEdges2D[:, 3] = np.array(gridEdges1D[:, 1]/patchSize[0], dtype=int)

#     print "grideEdges2D took", time.time()-s, "seconds"
#     sys.stdout.flush()
#     s = time.time()

    ## get uniform prior for bg patch
    bgPrior = -np.log(np.ones(patchSize)/np.prod(patchSize))
    
    ## get priors for all sprite patches
    spritePrior = np.zeros(patchSize)
    xs = np.ndarray.flatten(np.arange(patchSize[1], dtype=float).reshape((patchSize[1], 1)).repeat(patchSize[0], axis=-1))
    ys = np.ndarray.flatten(np.arange(patchSize[0], dtype=float).reshape((1, patchSize[0])).repeat(patchSize[1], axis=0))
    data = np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys)))))
    
    ## get covariance and means of prior on patch by using the bbox
    spriteBBox = trackedSprites[spriteIdx][DICT_BBOXES][f].T
    segment1 = spriteBBox[:, 0] - spriteBBox[:, 1]
    segment2 = spriteBBox[:, 1] - spriteBBox[:, 2]
    sigmaX = np.linalg.norm(segment1)/3.7
    sigmaY = np.linalg.norm(segment2)/3.7

#     ## find rotation as described here http://math.stackexchange.com/questions/612006/decomposing-an-affine-transformation
#     A11 = getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][f, :])[0, 1]
#     A21 = getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][f, :])[1, 1]
# #     rotRadians = np.pi-np.arctan(A21/A11)
#     rotRadians = np.arccos(np.dot((segment1)/np.linalg.norm(segment1), np.array([1.0, 0.0])))
#     b = np.array([1.0, 0.0])
#     ## arctan2 is computed based on formula: a[0]*b[1]-b[0]*a[1], a[0]*b[0]+a[1]*b[1]) where b = [1, 0] and a = segment1
#     rotRadians = np.mod(np.arctan2(-segment1[1], segment1[0]),2*np.pi)
#     print "rotation", rotRadians*180.0/np.pi
    rotRadians = trackedSprites[spriteIdx][DICT_BBOX_ROTATIONS][f]
    
    rotMat = np.array([[np.cos(rotRadians), -np.sin(rotRadians)], [np.sin(rotRadians), np.cos(rotRadians)]])
    
    means = np.reshape(trackedSprites[spriteIdx][DICT_BBOX_CENTERS][f], (2, 1)) - offset
    covs = np.dot(np.dot(rotMat.T, np.array([[sigmaX**2, 0.0], [0.0, sigmaY**2]])), rotMat)
    
#     print sigmaX, sigmaY, rotRadians, means
    spritePrior = np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F')
    
#     print "sprite prior took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    
    
    if showFigs :
        figure(); imshow(spritePrior)
#         gwv.showCustomGraph(np.reshape(multivariateNormal(data, means, covs, True), patchSize, order='F'))
#         gwv.showCustomGraph(np.reshape(minusLogMultivariateNormal(data, means, covs, True), patchSize, order='F'))

    ## change the background prior to give higher cost for pixels to be classified as background if the difference between bgPatch and spritePatch is high
    diffPatch = np.reshape(vectorisedMinusLogMultiNormal(spritePatch.reshape((np.prod(patchSize), 3)), 
                                                         bgPatch.reshape((np.prod(patchSize), 3)), 
                                                         np.eye(3)*1000.0, True), patchSize)#, order='F')
#     diffPatch = (remappedFgPatch+10.0)*100.0 + diffPatch
    alpha = 0.985
#     bgPrior = alpha*bgPrior + (1.0-alpha)*diffPatch
#     bgPrior *= remappedFgPatch

    ## stuff using optical flow
    if frameCount > 0 :
        prevFrameName = trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys())[frameCount-1]].split('/')[-1]
        nextFrameName = trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys())[frameCount]].split('/')[-1]
#         print frameCount
#         print outputPath+prevFrameName
#         print dataPath+dataSet+prevFrameName
#         print dataPath+dataSet+nextFrameName
        
#         img1 = np.array(Image.open(dataPath+dataSet+prevFrameName))
#         img2 = np.array(Image.open(dataPath+dataSet+nextFrameName))

#         flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(Image.open(dataPath+dataSet+prevFrameName)), cv2.COLOR_RGB2GRAY), 
#                                             cv2.cvtColor(np.array(Image.open(dataPath+dataSet+nextFrameName)), cv2.COLOR_RGB2GRAY), 
#                                             0.5, 3, 15, 3, 5, 1.1, 0)
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(Image.open(dataPath+dataSet+nextFrameName)), cv2.COLOR_RGB2GRAY), 
                                            cv2.cvtColor(np.array(Image.open(dataPath+dataSet+prevFrameName)), cv2.COLOR_RGB2GRAY), 
                                            0.5, 3, 15, 3, 5, 1.1, 0)
        alpha = np.array(Image.open(outputPath+prevFrameName))[:, :, -1]/255.0
#         fgIdxs = np.argwhere(alpha != 0)
#         remappedFgIdxs = np.array(np.round(fgIdxs+flow[fgIdxs[:, 0], fgIdxs[:, 1]][:, ::-1]), dtype=int)
#         remappedFg = np.zeros_like(alpha)
#         remappedFg[remappedFgIdxs[:, 0], remappedFgIdxs[:, 1]] = 1
        
        remappedFg = cv2.remap(alpha, flow[:, :, 0]+allXs, flow[:, :, 1]+allYs, cv2.INTER_LINEAR)
        
        remappedFgPatch = remappedFg[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
        
#         remappedFgPatch = cv2.GaussianBlur(remappedFgPatch, (31, 31), 2.5)
        remappedFgPatch = cv2.GaussianBlur(cv2.morphologyEx(remappedFgPatch, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))), (31, 31), 2.5)
        
        param = 0.4
#         bgPrior *= (remappedFgPatch+param)

        
#         bgPrior *= (remappedFgPatch+param)/(1.0+param)
#         bgPrior *= 1.15
        param = 0.2
        spritePrior = param*spritePrior + (1.0-param)*(-np.log((remappedFgPatch+0.01)/np.sum(remappedFgPatch+0.01)))
    

    ## merge two overlapping patches

    h = patchSize[0]
    w = patchSize[1]
    
    patAPixs = np.empty(0, dtype=uint)
    patBPixs = np.empty(0, dtype=uint)
    
    ## force small square of size squarePadding*2 + 1 around center of patch to come from patch B (i.e. the car)
    squarePadding = 6
    rows = np.ndarray.flatten(arange((h/2)-squarePadding, (h/2)+squarePadding+1).reshape((squarePadding*2+1, 1)).repeat(squarePadding*2+1, axis=-1))
    cols = np.ndarray.flatten(arange((w/2)-squarePadding, (w/2)+squarePadding+1).reshape((1, squarePadding*2+1)).repeat(squarePadding*2+1, axis=0))
    patBPixs = np.unique(np.concatenate((patBPixs, np.array(rows + cols*h, dtype=uint))))
    
    ## force one ring of pixels on the edge of the patch to come from patch A (i.e. the bg) (unless that column/row is intersected by the bbox)
    if np.min((largeBBox)[0, :]) > 0.0 :
#         print "adding left column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h, dtype=uint)[1:-1])))
    else :
#         print "adding left column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h, dtype=uint)[1:-1])))
    if np.min((largeBBox)[1, :]) > 0.0 :
#         print "adding top row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1])))
    else :
#         print "adding top row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1])))
    if np.max((largeBBox)[1, :]) < bgImage.shape[0] :
#         print "adding bottom row to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1]+h-1)))
    else :
#         print "adding bottom row to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)[1:-1]+h-1)))
    if np.max((largeBBox)[0, :]) < bgImage.shape[1] :
#         print "adding right column to A"
        patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=uint)[1:-1])))
    else :
#         print "adding right column to B"
        patBPixs = np.unique(np.concatenate((patBPixs, np.arange(h*(w-1), h*w, dtype=uint)[1:-1])))
    
#     patBPixs = np.empty(0)
    
    patA = np.copy(bgPatch/255.0)
    patB = np.copy(spritePatch/255.0)
    
#     print "patch pixels took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    
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
    
#     print "graphcut took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    
    if showFigs :
        figure(); imshow(labels, interpolation='nearest')
    
    outputPatch = np.zeros((patA.shape[0], patA.shape[1], patA.shape[2]+1), dtype=uint8)
    for i in xrange(labels.shape[0]) :
        for j in xrange(labels.shape[1]) :
            if labels[i, j] == 0 :
                ## patA stands for the bgPatch but I want to set the pixels here to 0 to save space
                outputPatch[i, j, 0:-1] = 0#patA[i, j, :]*255
            else :
                outputPatch[i, j, 0:-1] = patB[i, j, :]*255
                outputPatch[i, j, -1] = 255
    
    if showFigs :
        figure(); imshow(outputPatch, interpolation='nearest')
        
    currentFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], bgImage.shape[2]+1), dtype=uint8)
    currentFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = np.copy(outputPatch)
    
    if showFigs :
        figure(); imshow(currentFrame)
        figure(); imshow(np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f])))
    
    Image.fromarray((currentFrame).astype(numpy.uint8)).save(outputPath + trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f].split('/')[-1])
    
#     print "saving took", time.time()-s, "seconds"
#     sys.stdout.flush()
    sys.stdout.write('\r' + "Done " + np.string_(frameCount+1) + " images of " + np.string_(len(trackedSprites[spriteIdx][DICT_BBOXES].keys())))
    sys.stdout.flush()
    
print 
print "total time:", time.time() - startTime

# <codecell>

## load the tracked sprites
DICT_SPRITE_NAME = 'sprite_name'
# DICT_BBOX_AFFINES = 'bbox_affines'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
# DICT_NUM_FRAMES = 'num_frames'
# DICT_START_FRAME = 'start_frame'
DICT_FRAMES_LOCATIONS = 'frame_locs'

dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
bgImage = np.array(Image.open(dataPath + dataSet + "median.png"))

trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())

## merge tracked sprite with bg
spriteIdx = 2
sequenceLength = len(trackedSprites[spriteIdx][DICT_BBOXES])
showFigs = False

outputPath = dataPath + dataSet + trackedSprites[spriteIdx][DICT_SPRITE_NAME] + "-maskedFlow/" #"-masked/"

if outputPath != None and not os.path.isdir(outputPath):
    os.makedirs(outputPath)

if showFigs :
    figure(); imshow(bgImage)
    
print "processing", trackedSprites[spriteIdx][DICT_SPRITE_NAME]

allXs = arange(bgImage.shape[1], dtype=float32).reshape((1, bgImage.shape[1])).repeat(bgImage.shape[0], axis=0)
allYs = arange(bgImage.shape[0], dtype=float32).reshape((bgImage.shape[0], 1)).repeat(bgImage.shape[1], axis=1)

startTime = time.time()
for f, frameCount in zip(np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys())[2:3], arange(len(trackedSprites[spriteIdx][DICT_BBOXES].keys()))[2:3]):#[1109:1110]:#sequenceLength):
    
    spritePatch, offset, patchSize, touchedBorders = getSpritePatch(trackedSprites[spriteIdx], f, bgImage.shape[1], bgImage.shape[0])
    bgPatch = np.copy(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])

    bgPrior, spritePrior = getPatchPriors(bgPatch, spritePatch, offset, patchSize, trackedSprites[spriteIdx], f, 
                                          prevFrameKey=np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys())[frameCount-1], prevFrameAlphaLoc=outputPath,
                                          prevMaskImportance=0.2)

    labels = mergePatches(bgPatch, spritePatch, bgPrior, spritePrior, offset, patchSize, touchedBorders, useCenterSquare=False,
                               scribble=np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 3)))

    outputPatch = np.zeros((bgPatch.shape[0], bgPatch.shape[1], bgPatch.shape[2]+1), dtype=uint8)
    for i in xrange(labels.shape[0]) :
        for j in xrange(labels.shape[1]) :
            if labels[i, j] == 0 :
                ## patA stands for the bgPatch but I want to set the pixels here to 0 to save space
                outputPatch[i, j, 0:-1] = 0#bgPatch[i, j, :]
            else :
                outputPatch[i, j, 0:-1] = spritePatch[i, j, :]
                outputPatch[i, j, -1] = 255

    currentFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], bgImage.shape[2]+1), dtype=uint8)
    currentFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = np.copy(outputPatch)

#     Image.fromarray((currentFrame).astype(numpy.uint8)).save(outputPath + trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][f].split('/')[-1])
    figure(); imshow(currentFrame)
    
print 
print "total time:", time.time() - startTime

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text="", parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.segmentedImage = None
        self.originalImage = None
        self.bgImage = None
        self.originalImageOpacity = 0.2
        
        self.scribbleImage = None
        self.scribbleOpacity = 0.4
        
    def setSegmentedImage(self, segmentedImage) : 
        self.segmentedImage = segmentedImage.copy()
        self.setMinimumSize(self.segmentedImage.size())
        self.update()
        
    def setOriginalImage(self, originalImage) : 
        self.originalImage = originalImage.copy()
        self.setMinimumSize(self.originalImage.size())
        self.update()
        
    def setOriginalImageOpacity(self, originalImageOpacity) : 
        self.originalImageOpacity = originalImageOpacity
        self.update()
        
    def setBackgroundImage(self, bgImage) :
        self.bgImage = bgImage.copy()
        self.setMinimumSize(self.bgImage.size())            
        self.update()
        
    def setScribbleImage(self, scribbleImage) :
        self.scribbleImage = scribbleImage.copy()
        self.update()
        
    def setScribbleOpacity(self, scribbleOpacity) : 
        self.scribbleOpacity = scribbleOpacity
        self.update()
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        
        if self.segmentedImage != None and self.originalImage != None and self.bgImage != None :
            upperLeft = ((self.width()-self.originalImage.width())/2, (self.height()-self.originalImage.height())/2)
            ## draw background
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.bgImage)
            
            ## draw rect
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 32, 32, 127)))
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 0)))
            painter.drawRect(QtCore.QRect(upperLeft[0], upperLeft[1], self.originalImage.width(), self.originalImage.height()))
            
            ## draw originalImage
            painter.setOpacity(self.originalImageOpacity)
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.originalImage)
            
            ## draw segmentedImage
            painter.setOpacity(1.0)
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.segmentedImage)
            
            if self.scribbleImage != None :
                painter.setOpacity(self.scribbleOpacity)
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_Multiply)
                painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.scribbleImage)
                
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.spriteIdx = 1
                
        self.createGUI()
        
        self.changeSprite(self.spriteIdx)
        
        self.setWindowTitle("Sprite Segmentation")
        self.resize(1700, 900)
        
        self.frameIdx = 0
        self.scribbling = False
        self.lastPoint = QtCore.QPoint(0, 0)
        self.stopSegmenting = False
        
        
        im = np.ascontiguousarray(np.array(Image.open(dataPath+dataSet+"median.png"))[:, :, :3])
        
        self.imageWidth = im.shape[1]
        self.imageHeight = im.shape[0]
        
        self.scribble = QtGui.QImage(QtCore.QSize(self.imageWidth, self.imageHeight), QtGui.QImage.Format_RGB888)
        self.scribble.fill(QtGui.QColor.fromRgb(255, 255, 255))
        self.frameLabel.setScribbleImage(self.scribble)
        
        ## HACK ##
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setBackgroundImage(qim)
        
        self.setFrameImage()
        
#         self.settingAnchorPoint = False
        self.prevMousePosition = QtCore.QPoint(0, 0)
        
        
        self.setFocus() 
        
    def setFrameImage(self, refresh = False) :
        if self.frameIdx >= 0 and self.frameIdx < self.numFrames :
            ## returns rgba but for whatever reason it needs to be bgra for qt to display it properly
            im = self.getMattedImage(refresh)
            im = np.ascontiguousarray(np.copy(im[:, :, [2, 1, 0, 3]]))

            ## HACK ##
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
            self.frameLabel.setSegmentedImage(qim)
            self.frameInfo.setText(trackedSprites[self.spriteIdx][DICT_FRAMES_LOCATIONS][self.framePathsIdxs[self.frameIdx]])
            
            im = np.ascontiguousarray(Image.open(trackedSprites[self.spriteIdx][DICT_FRAMES_LOCATIONS][self.framePathsIdxs[self.frameIdx]]))
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            self.frameLabel.setOriginalImage(qim)
            
            
            frameName =  trackedSprites[self.spriteIdx][DICT_FRAMES_LOCATIONS][self.framePathsIdxs[self.frameIdx]].split('/')[-1]
            
            if os.path.isfile(self.outputPath+"scribble-"+frameName) :
                im = np.ascontiguousarray(Image.open(self.outputPath+"scribble-"+frameName))
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
                
                self.scribble.fill(QtGui.QColor.fromRgb(255, 255, 255))
                painter = QtGui.QPainter(self.scribble)
                painter.drawImage(QtCore.QPoint(0, 0), qim)
#                 self.scribble = newScribble
#                 self.scribble = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            else :
#                 self.scribble = QtGui.QImage(QtCore.QSize(self.imageWidth, self.imageHeight), QtGui.QImage.Format_RGB888)
                self.scribble.fill(QtGui.QColor.fromRgb(255, 255, 255))
            self.frameLabel.setScribbleImage(self.scribble)
                
        
    def getMattedImage(self, refresh) :
        startTime = time.time()
        ## returns rgba
        frameName =  trackedSprites[self.spriteIdx][DICT_FRAMES_LOCATIONS][self.framePathsIdxs[self.frameIdx]].split('/')[-1]

        if os.path.isfile(self.outputPath+frameName) and not refresh :
            return np.array(Image.open(self.outputPath+frameName))
        else :
            t = time.time()
            spritePatch, offset, patchSize, touchedBorders = getSpritePatch(trackedSprites[self.spriteIdx], self.framePathsIdxs[self.frameIdx], 
                                                                            self.imageWidth, self.imageHeight)
            bgPatch = np.copy(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :])
#             print "patches", time.time() - t
            t = time.time()

            if self.frameIdx > 0 :
                bgPrior, spritePrior = getPatchPriors(bgPatch, spritePatch, offset, patchSize, trackedSprites[self.spriteIdx],
                                                      self.framePathsIdxs[self.frameIdx], 
                                                      prevFrameKey=self.framePathsIdxs[self.frameIdx-1], 
                                                      prevFrameAlphaLoc=self.outputPath,
                                                      useOpticalFlow=self.doUseOpticalFlowPriorBox.isChecked(),
                                                      useDiffPatch=self.doUsePatchDiffPriorBox.isChecked(),
                                                      prevMaskImportance=self.prevMaskImportanceSpinBox.value(),
                                                      prevMaskDilate=self.prevMaskDilateSpinBox.value(),
                                                      prevMaskBlurSize=self.prevMaskBlurSizeSpinBox.value(),
                                                      prevMaskBlurSigma=self.prevMaskBlurSigmaSpinBox.value(),
                                                      diffPatchImportance=self.diffPatchImportanceSpinBox.value(),
                                                      diffPatchMultiplier=self.diffPatchMultiplierSpinBox.value())
#                 print "priors with flow", time.time() - t
#                 gwv.showCustomGraph(spritePrior)
                t = time.time()
            else :
                bgPrior, spritePrior = getPatchPriors(bgPatch, spritePatch, offset, patchSize, trackedSprites[self.spriteIdx],
                                                      self.framePathsIdxs[self.frameIdx],
                                                      useOpticalFlow=self.doUseOpticalFlowPriorBox.isChecked(),
                                                      useDiffPatch=self.doUsePatchDiffPriorBox.isChecked(),
                                                      prevMaskImportance=self.prevMaskImportanceSpinBox.value(),
                                                      prevMaskDilate=self.prevMaskDilateSpinBox.value(),
                                                      prevMaskBlurSize=self.prevMaskBlurSizeSpinBox.value(),
                                                      prevMaskBlurSigma=self.prevMaskBlurSigmaSpinBox.value(),
                                                      diffPatchImportance=self.diffPatchImportanceSpinBox.value(),
                                                      diffPatchMultiplier=self.diffPatchMultiplierSpinBox.value())
#                 print "priors without flow", time.time() - t
                t = time.time()
            
#             scribble = None
#             if self.scribble.format() == QtGui.QImage.Format.Format_RGB888 :
#                 scribble = np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 3))[:, :, [2, 1, 0]]
#             elif self.scribble.format() == QtGui.QImage.Format.Format_RGB32 :
#                 scribble = np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 4))[:, :, [2, 1, 0]]
            
            labels = mergePatches(bgPatch, spritePatch, bgPrior, spritePrior, offset, patchSize, touchedBorders,
                                  scribble=np.frombuffer(self.scribble.constBits(), dtype=uint8).reshape((self.imageHeight, self.imageWidth, 3)),
                                  useCenterSquare=self.doUseCenterSquareBox.isChecked(),
                                  useGradients=self.doUseGradientsCostBox.isChecked())
#             print "merging", time.time() - t
            t = time.time()

            outputPatch = np.zeros((bgPatch.shape[0], bgPatch.shape[1], bgPatch.shape[2]+1), dtype=uint8)
            for i in xrange(labels.shape[0]) :
                for j in xrange(labels.shape[1]) :
                    if labels[i, j] == 0 :
                        ## patA stands for the bgPatch but I want to set the pixels here to 0 to save space
                        outputPatch[i, j, 0:-1] = 0#bgPatch[i, j, :]
                    else :
                        outputPatch[i, j, 0:-1] = spritePatch[i, j, :]
                        outputPatch[i, j, -1] = 255

            currentFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], bgImage.shape[2]+1), dtype=uint8)
            currentFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = np.copy(outputPatch)
#             print "putting together the frame", time.time() - t
            t = time.time()
            
            Image.fromarray((currentFrame).astype(numpy.uint8)).save(self.outputPath + frameName)
#             print "saving", time.time() - t
            t = time.time()
    
            print "segmented in", time.time() - startTime

            return currentFrame

    def changeSprite(self, idx) :
        self.spriteIdx = idx
        self.framePathsIdxs = np.sort(trackedSprites[self.spriteIdx][DICT_FRAMES_LOCATIONS].keys())
        self.numFrames = len(self.framePathsIdxs)
        
        self.outputPath = dataPath + dataSet + trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME] + "-maskedFlow/"
        if not os.path.isdir(self.outputPath) :
            os.mkdir(self.outputPath)
        
        self.frameSpinBox.setRange(0, self.numFrames-1)
        self.frameSlider.setMaximum(self.numFrames-1)
        
        self.frameSpinBox.setValue(0)
        
    def changeFrame(self, idx, refresh = False) :
        self.frameIdx = idx
        self.setFrameImage(refresh)
        
    def refreshSegmentation(self) :
        self.setFrameImage(True)
        
    def segmentSequence(self) :
        self.stopSegmenting = False
        startFrame = self.frameIdx
        for frameIdx in xrange(startFrame, self.numFrames) :
            self.changeFrame(frameIdx, True)
            QtGui.QApplication.processEvents()
            if self.stopSegmenting :
                self.stopSegmenting = False
                break;
    
    def setOriginalImageOpacity(self, opacity) :
        self.frameLabel.setOriginalImageOpacity(opacity/100.0)
        
    def setScribbleOpacity(self, opacity) :
        self.frameLabel.setScribbleOpacity(opacity/100.0)
    
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_Return :
            self.stopSegmenting = True
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton or event.button() == QtCore.Qt.RightButton :
            sizeDiff = (self.frameLabel.size() - self.frameLabel.bgImage.size())/2
            mousePos = event.pos() - self.frameLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
            
            self.lastPoint = mousePos
            self.scribbling = True
 
    def mouseMoveEvent(self, event):
        if ((event.buttons() & QtCore.Qt.LeftButton) or (event.buttons() & QtCore.Qt.RightButton)) and self.scribbling:
            sizeDiff = (self.frameLabel.size() - self.frameLabel.bgImage.size())/2
            mousePos = event.pos() - self.frameLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
            
            if event.buttons() & QtCore.Qt.LeftButton :
                ## foreground
                penColor = QtGui.QColor.fromRgb(0, 255, 0)
            else :
                ## background
                penColor = QtGui.QColor.fromRgb(0, 0, 255)
                
            if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier :
                ## delete
                penColor = QtGui.QColor.fromRgb(255, 255, 255)
            
            self.drawLineTo(mousePos, penColor)
 
    def mouseReleaseEvent(self, event):
        if (event.button() == QtCore.Qt.LeftButton or event.button() == QtCore.Qt.RightButton) and self.scribbling:
            sizeDiff = (self.frameLabel.size() - self.frameLabel.bgImage.size())/2
            mousePos = event.pos() - self.frameLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
            
            if event.buttons() & QtCore.Qt.LeftButton :
                ## foreground
                penColor = QtGui.QColor.fromRgb(0, 255, 0)
            else :
                ## background
                penColor = QtGui.QColor.fromRgb(0, 0, 255)
                
            if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier :
                ## delete
                penColor = QtGui.QColor.fromRgb(255, 255, 255)
                
            self.drawLineTo(mousePos, penColor)
            
            self.scribbling = False
            
            if self.frameIdx >= 0 and self.frameIdx < self.numFrames :
                frameName =  trackedSprites[self.spriteIdx][DICT_FRAMES_LOCATIONS][self.framePathsIdxs[self.frameIdx]].split('/')[-1]
                self.scribble.save(self.outputPath+"scribble-" + frameName)
 
    def drawLineTo(self, endPoint, penColor):
        painter = QtGui.QPainter(self.scribble)
        penWidth = 20
            
        painter.setPen(QtGui.QPen(penColor, penWidth,
                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
 
        self.lastPoint = QtCore.QPoint(endPoint)
        
        self.frameLabel.setScribbleImage(self.scribble)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel()
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameSpinBox = QtGui.QSpinBox()
#         self.frameSpinBox.setRange(0, self.numFrames-1)
        self.frameSpinBox.setSingleStep(1)
        
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
#         self.frameSlider.setMinimum(0)
#         self.frameSlider.setMaximum(self.numFrames-1)
        
        controlsGroup = QtGui.QGroupBox("Controls")
        controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
                                             "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        controlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        
        
        self.doUseOpticalFlowPriorBox = QtGui.QCheckBox()
        self.doUseOpticalFlowPriorBox.setChecked(True)
        
        self.prevMaskImportanceSpinBox = QtGui.QDoubleSpinBox()
        self.prevMaskImportanceSpinBox.setRange(0.0, 1.0)
        self.prevMaskImportanceSpinBox.setSingleStep(0.01)
        self.prevMaskImportanceSpinBox.setValue(0.8)
        
        self.prevMaskDilateSpinBox = QtGui.QSpinBox()
        self.prevMaskDilateSpinBox.setRange(1, 33)
        self.prevMaskDilateSpinBox.setSingleStep(2)
        self.prevMaskDilateSpinBox.setValue(13)
        
        self.prevMaskBlurSizeSpinBox = QtGui.QSpinBox()
        self.prevMaskBlurSizeSpinBox.setRange(1, 65)
        self.prevMaskBlurSizeSpinBox.setSingleStep(2)
        self.prevMaskBlurSizeSpinBox.setValue(31)
        
        self.prevMaskBlurSigmaSpinBox = QtGui.QDoubleSpinBox()
        self.prevMaskBlurSigmaSpinBox.setRange(0.5, 5.0)
        self.prevMaskBlurSigmaSpinBox.setSingleStep(0.1)
        self.prevMaskBlurSigmaSpinBox.setValue(2.5)
        
        
        
        self.doUsePatchDiffPriorBox = QtGui.QCheckBox()
        
        self.diffPatchImportanceSpinBox = QtGui.QDoubleSpinBox()
        self.diffPatchImportanceSpinBox.setRange(0.0, 1.0)
        self.diffPatchImportanceSpinBox.setSingleStep(0.001)
        self.diffPatchImportanceSpinBox.setValue(0.015)
        
        self.diffPatchMultiplierSpinBox = QtGui.QDoubleSpinBox()
        self.diffPatchMultiplierSpinBox.setRange(1.0, 10000.0)
        self.diffPatchMultiplierSpinBox.setSingleStep(10.0)
        self.diffPatchMultiplierSpinBox.setValue(1000.0)
        
        
        
        self.doUseGradientsCostBox = QtGui.QCheckBox()
        
        self.doUseCenterSquareBox = QtGui.QCheckBox()
        self.doUseCenterSquareBox.setChecked(True)
        
        
        
        self.refreshSegmentationButton = QtGui.QPushButton("&Refresh Segmentation")
        
        self.segmentSequenceButton = QtGui.QPushButton("&Segment Sequence")
        
        
        
        self.spriteIdxSpinBox = QtGui.QSpinBox()
        self.spriteIdxSpinBox.setRange(0, len(trackedSprites)-1)
        self.spriteIdxSpinBox.setValue(self.spriteIdx)
        
        self.originalImageOpacitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.originalImageOpacitySlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.originalImageOpacitySlider.setMinimum(0)
        self.originalImageOpacitySlider.setMaximum(100)
        self.originalImageOpacitySlider.setValue(20)
        
        
        self.scribbleOpacitySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scribbleOpacitySlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.scribbleOpacitySlider.setMinimum(0)
        self.scribbleOpacitySlider.setMaximum(100)
        self.scribbleOpacitySlider.setValue(40)
        
        
        ## SIGNALS ##
        
        self.frameSpinBox.valueChanged[int].connect(self.frameSlider.setValue)
        self.frameSlider.valueChanged[int].connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.changeFrame)
        
        self.doUseOpticalFlowPriorBox.stateChanged.connect(self.refreshSegmentation)
        self.prevMaskImportanceSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.prevMaskDilateSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.prevMaskBlurSizeSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.prevMaskBlurSigmaSpinBox.editingFinished.connect(self.refreshSegmentation)
        
        self.doUsePatchDiffPriorBox.stateChanged.connect(self.refreshSegmentation)
        self.diffPatchImportanceSpinBox.editingFinished.connect(self.refreshSegmentation)
        self.diffPatchMultiplierSpinBox.editingFinished.connect(self.refreshSegmentation)
        
        self.doUseGradientsCostBox.stateChanged.connect(self.refreshSegmentation)
        self.doUseCenterSquareBox.stateChanged.connect(self.refreshSegmentation)
        
        self.refreshSegmentationButton.clicked.connect(self.refreshSegmentation)
        
        self.spriteIdxSpinBox.valueChanged.connect(self.changeSprite)
        
        self.segmentSequenceButton.clicked.connect(self.segmentSequence)
        
        self.originalImageOpacitySlider.valueChanged[int].connect(self.setOriginalImageOpacity)
        self.scribbleOpacitySlider.valueChanged[int].connect(self.setScribbleOpacity)
        
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QGridLayout()
        controlsLayout.addWidget(QtGui.QLabel("Use Optical Flow Prior"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.doUseOpticalFlowPriorBox, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Previous Mask Importance"), 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.prevMaskImportanceSpinBox, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Previous Mask Dilation"), 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.prevMaskDilateSpinBox, 2, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Previous Mask Blur Size"), 3, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.prevMaskBlurSizeSpinBox, 3, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Previous Mask Blur Sigma"), 4, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.prevMaskBlurSigmaSpinBox, 4, 1, 1, 1, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(QtGui.QLabel("Use Patch Difference Prior"), 5, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.doUsePatchDiffPriorBox, 5, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Patch Difference Importance"), 6, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.diffPatchImportanceSpinBox, 6, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Patch Difference Multiplier"), 7, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.diffPatchMultiplierSpinBox, 7, 1, 1, 1, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(QtGui.QLabel("Use Gradients Cost"), 8, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.doUseGradientsCostBox, 8, 1, 1, 1, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(QtGui.QLabel("Force FG Center Square"), 9, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.doUseCenterSquareBox, 9, 1, 1, 1, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(self.refreshSegmentationButton, 10, 0, 1, 2, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(QtGui.QLabel("Sprite"), 11, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.spriteIdxSpinBox, 11, 1, 1, 1, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(self.segmentSequenceButton, 12, 0, 1, 2, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(QtGui.QLabel("Original Image Opacity"), 13, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.originalImageOpacitySlider, 13, 1, 1, 2, QtCore.Qt.AlignLeft)
        
        controlsLayout.addWidget(QtGui.QLabel("Scribble Opacity"), 14, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.scribbleOpacitySlider, 14, 1, 1, 2, QtCore.Qt.AlignLeft)
        
        controlsGroup.setLayout(controlsLayout)

        controlsVLayout = QtGui.QVBoxLayout()
        controlsVLayout.addWidget(controlsGroup)
        controlsVLayout.addStretch()
        
        sliderLayout = QtGui.QHBoxLayout()
        sliderLayout.addWidget(self.frameSlider)
        sliderLayout.addWidget(self.frameSpinBox)
        
        frameLayout = QtGui.QVBoxLayout()
        frameLayout.addWidget(self.frameLabel)
        frameLayout.addWidget(self.frameInfo)
        frameLayout.addLayout(sliderLayout)
        
        mainLayout.addLayout(controlsVLayout)
        mainLayout.addLayout(frameLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

def computeMattedImage(frameIdx, spriteIdx, framePathsIdxs, imageWidth, imageHeight, outputPath) :
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
#                 print "priors with flow", time.time() - t
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
#                 print "priors without flow", time.time() - t
        t = time.time()

#             scribble = None
#             if self.scribble.format() == QtGui.QImage.Format.Format_RGB888 :
#                 scribble = np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 3))[:, :, [2, 1, 0]]
#             elif self.scribble.format() == QtGui.QImage.Format.Format_RGB32 :
#                 scribble = np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 4))[:, :, [2, 1, 0]]

    labels = mergePatches(bgPatch, spritePatch, bgPrior, spritePrior, offset, patchSize, touchedBorders,
                          scribble=np.ones((720, 1280, 3), dtype=np.uint8)*np.uint8(255),
                          useCenterSquare=True,
                          useGradients=False)
#             print "merging", time.time() - t
    t = time.time()

    outputPatch = np.zeros((bgPatch.shape[0], bgPatch.shape[1], bgPatch.shape[2]+1), dtype=uint8)
    for i in xrange(labels.shape[0]) :
        for j in xrange(labels.shape[1]) :
            if labels[i, j] == 0 :
                ## patA stands for the bgPatch but I want to set the pixels here to 0 to save space
                outputPatch[i, j, 0:-1] = 0#bgPatch[i, j, :]
            else :
                outputPatch[i, j, 0:-1] = spritePatch[i, j, :]
                outputPatch[i, j, -1] = 255

    currentFrame = np.zeros((bgImage.shape[0], bgImage.shape[1], bgImage.shape[2]+1), dtype=uint8)
    currentFrame[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :] = np.copy(outputPatch)
#             print "putting together the frame", time.time() - t
    t = time.time()

    Image.fromarray((currentFrame).astype(numpy.uint8)).save(outputPath + frameName)
#             print "saving", time.time() - t
    t = time.time()


im = np.array(Image.open(dataPath+dataSet+"median.png"))
imageWidth = im.shape[1]
imageHeight = im.shape[0]

for spriteIdx in arange(len(trackedSprites))[1:2] :
    print trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]
    framePathsIdxs = np.sort(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())
    outputPath = dataPath + dataSet + trackedSprites[spriteIdx][DICT_SEQUENCE_NAME] + "-maskedFlow/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
        
    for frameIdx in arange(len(framePathsIdxs))[:1] :
        computeMattedImage(frameIdx, spriteIdx, framePathsIdxs, imageWidth, imageHeight, outputPath)

# <codecell>

print outputPath

# <codecell>

from pcaflow import PCAFlow
import warnings
# To read images
from scipy.misc import imread

# To display
from pcaflow.utils.viz_flow import viz_flow
    
PATH_PC_U = '/opt/pcaflow-master/data/PC_U.npy'
PATH_PC_V = '/opt/pcaflow-master/data/PC_V.npy'
PATH_COV = '/opt/pcaflow-master/data/COV_SINTEL.npy'
PATH_COV_SUBLAYER = '/opt/pcaflow-master/data/COV_SINTEL_SUBLAYER.npy'

# <codecell>

### Compute using PCA-Layers.
P = PCAFlow.PCAFlow(
        pc_file_u=PATH_PC_U,
        pc_file_v=PATH_PC_V,
        covfile=PATH_COV,
        covfile_sublayer=PATH_COV_SUBLAYER,
        preset='pcalayers_sintel',
        )

### Compute using PCA-Flow.
# P = PCAFlow.PCAFlow(
#        pc_file_u=PATH_PC_U,
#        pc_file_v=PATH_PC_V,
#        covfile=PATH_COV,
#        preset='pcaflow_sintel',
#        )

# <codecell>

### Once the object is created, it can be used like this:
I1 = imread('/opt/pcaflow-master/image1.png')
I2 = imread('/opt/pcaflow-master/image2.png')
# I1 = imread(dataPath+dataSet+"frame-01110.png")
# I2 = imread(dataPath+dataSet+"frame-01111.png")

try :
    P.push_back(I1)
    P.push_back(I2)

# Compute flow
    u,v = P.compute_flow()
except DeprecationWarning :
    print

### Use this if you want to just get the motion descriptor
#u,v,data = P.compute_flow(return_additional=['weights',])
#descriptor = data['weights']

I_flow = viz_flow(u,v)

figure()
subplot(221)
imshow(I1)
title('First image')
subplot(222)
imshow(I_flow)
title('Flow colormap')
subplot(223)
imshow(u)
title('Horizontal component')
subplot(224)
imshow(v)
title('Vertical component')

show()

# <codecell>

gwv.showCustomGraph(u)

# <codecell>

tmp = window.scribble.copy()

# <codecell>

print np.frombuffer(tmp.constBits(), dtype=uint8).shape

# <codecell>

## from qimage to array
scribble = np.copy(np.frombuffer(window.scribble.constBits(), dtype=uint8)).reshape((720, 1280, 3))
# bgPixs = np.argwhere(np.all((scribble[:, :, 0] == 0, scribble[:, :, 1] == 0), axis=0))
# fgPixs = np.argwhere(np.all((scribble[:, :, 0] == 0, scribble[:, :, 2] == 0), axis=0))
figure(); imshow(scribble)

# <codecell>

print patchSize, offset
# fixedPixels = np.zeros(patchSize)
# ## 1 == bg pixels
# fixedPixels[np.mod(APixels, patchSize[0]), np.array(APixels/patchSize[0], dtype=int)] = 1
# ## 2 == fg pixels
# fixedPixels[np.mod(BPixels, patchSize[0]), np.array(BPixels/patchSize[0], dtype=int)] = 2

# fixedPixels[bgPixs[:, 0]-offset[1], bgPixs[:, 1]-offset[0]] = 1
# fixedPixels[fgPixs[:, 0]-offset[1], fgPixs[:, 1]-offset[0]] = 2
# gwv.showCustomGraph(fixedPixels)

# <codecell>

## turn to 1D indices
patAPixs = np.argwhere(fixedPixels == 1)
patAPixs = np.sort(patAPixs[:, 0] + patAPixs[:, 1]*patchSize[0])
patBPixs = np.argwhere(fixedPixels == 2)
patBPixs = np.sort(patBPixs[:, 0] + patBPixs[:, 1]*patchSize[0])
print patAPixs
print patBPixs
tmp = np.zeros_like(fixedPixels)
tmp[np.mod(patAPixs, patchSize[0]), np.array(patAPixs/patchSize[0], dtype=int)] = 1
tmp[np.mod(patBPixs, patchSize[0]), np.array(patBPixs/patchSize[0], dtype=int)] = 2
gwv.showCustomGraph(tmp)

# <codecell>

print APixels.shape

# <codecell>

figure(); imshow(np.frombuffer(window.scribble.constBits(), dtype=uint8).reshape((720, 1280, 4))[:, :, [2, 1, 0]])

# <codecell>

figure(); imshow(spritePatch)
sobelX = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
figure(); imshow(cv2.filter2D(spritePatch, cv2.CV_32F, sobelX.T)[:, :, 0])

# <codecell>

print np.array([[-1, 0, 1],
                                                                 [-2, 0, 2],
                                                                 [-1, 0, 1]])

# <codecell>

gwv.showCustomGraph(remappedFgPatch)

# <codecell>

gwv.showCustomGraph((remappedFgPatch+0.01)/np.sum(remappedFgPatch+0.01))
gwv.showCustomGraph(-np.log((remappedFgPatch+0.01)/np.sum(remappedFgPatch+0.01)))

# <codecell>

gwv.showCustomGraph(remappedFgPatch)
gwv.showCustomGraph(cv2.GaussianBlur(remappedFgPatch, (31, 31), 2.5))
gwv.showCustomGraph(bgPrior)

# <codecell>

img1 = np.array(Image.open(dataPath+dataSet+"white_bus1-maskedFlow/frame-00001.png"))
alpha = img1[:, :, -1]/255.0
img1 = np.array(Image.open(dataPath+dataSet+"frame-00001.png"))
img2 = np.array(Image.open(dataPath+dataSet+"frame-00002.png"))
flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), 
                                    cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), 
                                    0.5, 3, 15, 3, 5, 1.1, 0)
figure(); imshow(alpha)

# <codecell>

allXs = arange(bgImage.shape[1], dtype=float32).reshape((1, bgImage.shape[1])).repeat(bgImage.shape[0], axis=0)
allYs = arange(bgImage.shape[0], dtype=float32).reshape((bgImage.shape[0], 1)).repeat(bgImage.shape[1], axis=1)


remapped = cv2.remap(alpha, flow[:, :, 0]+allXs, flow[:, :, 1]+allYs, cv2.INTER_LINEAR)
figure(); imshow(remapped)

# <codecell>

fgIdxs = np.argwhere(alpha != 0)
remappedFgIdxs = np.array(np.round(fgIdxs+flow[fgIdxs[:, 0], fgIdxs[:, 1]][:, ::-1]), dtype=int)
remappedFg = np.zeros_like(alpha)
remappedFg[remappedFgIdxs[:, 0], remappedFgIdxs[:, 1]] = 1
remappedFgPatch = remappedFg[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
# figure(); imshow(alpha)
# figure(); imshow(remappedFg)
gwv.showCustomGraph(remappedFgPatch)
tmp = cv2.GaussianBlur(remappedFgPatch, (31, 31), 2.5)
gwv.showCustomGraph(np.copy(tmp))
tmp[np.argwhere(remappedFgPatch == np.max(remappedFgPatch))[:, 0], np.argwhere(remappedFgPatch != 0)[:, 1]] = np.max(remappedFgPatch)
gwv.showCustomGraph(tmp)

# <codecell>

# print cv2.getGaussianKernel(31, 2.5)
gwv.showCustomGraph(cv2.GaussianBlur(cv2.morphologyEx(remappedFgPatch, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))), (33, 33), 2.5))

# <codecell>

remappedFgPatch = remappedFg[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]]
# gwv.showCustomGraph(diffPatch)
gwv.showCustomGraph(remappedFgPatch)
# gwv.showCustomGraph((remappedFgPatch+10.0)*100.0 + diffPatch)

# <codecell>

bob = currentFrame[:, :, -1].reshape((720, 1280, 1))/255.0
figure(); imshow(np.array(bgImage*(1.0 - bob)+currentFrame[:, :, :-1]*bob, dtype=uint8))

# <codecell>

Image.fromarray(np.array(bgImage*(1.0 - bob)+currentFrame[:, :, :-1]*bob, dtype=uint8)).save("tralalala.png")

# <codecell>

print flow[fgIdxs[:, 0], fgIdxs[:, 1]]
print flow[fgIdxs[:, 0], fgIdxs[:, 1]][:, ::-1]

# <codecell>

gwv.showCustomGraph(bgPrior)

# <codecell>

# diffPatch = np.sum((spritePatch-bgPatch)**2.0, axis=-1)
# diffPatch /= np.max(diffPatch)
diffPatch = np.reshape(vectorisedMinusLogMultiNormal(spritePatch.reshape((np.prod(patchSize), 3)), 
                                                     bgPatch.reshape((np.prod(patchSize), 3)), 
                                                     np.eye(3)*1000.0, True), patchSize)#, order='F')
alpha = 0.6

gwv.showCustomGraph(diffPatch)
gwv.showCustomGraph(spritePrior)
gwv.showCustomGraph(bgPrior)

# <codecell>

tmp = np.zeros(patchSize)
tmp[np.array(np.mod(patAPixs, patchSize[0]), dtype=int), np.array(patAPixs/patchSize[0], dtype=int)] += 1
tmp[np.array(np.mod(patBPixs, patchSize[0]), dtype=int), np.array(patBPixs/patchSize[0], dtype=int)] += 2
figure(); imshow(tmp, interpolation='nearest')

# <codecell>

figure(); imshow(bgImage)

transform = getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][f, :])
transform[1, 0] = 500
transform[0, 0] = 400

## get the bbox for the current sprite frame, make it larger and find the rectangular patch to work with
## boundaries of the patch [min, max]
xBounds = np.array([bgImage.shape[1], 0.0])
yBounds = np.array([bgImage.shape[0], 0.0])

## plot bbox
spriteBBox = np.dot(transform, bboxDefaultCorners)
plot(spriteBBox[0, :], spriteBBox[1, :])


largeBBox = np.dot(np.array([[0.0, 0.0, 1.0+PATCH_BORDER], [0.0, 1.0+PATCH_BORDER, 0.0]]), bboxDefaultCorners)
## transform according to affine transformation
largeBBox = np.dot(transform, np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
plot(largeBBox[0, :], largeBBox[1, :])

## make sure xBounds are in between 0 and width and yBounds are in between 0 and height
xBounds[0] = np.max((0, np.min(largeBBox[0, :])))
xBounds[1] = np.min((bgImage.shape[1], np.max(largeBBox[0, :])))
yBounds[0] = np.max((0, np.min(largeBBox[1, :])))
yBounds[1] = np.min((bgImage.shape[0], np.max(largeBBox[1, :])))

#     print xBounds, yBounds

offset = np.array([np.round(np.array([xBounds[0], yBounds[0]]))], dtype=int).T # [x, y]
patchSize = np.array(np.round(np.array([yBounds[1]-yBounds[0], xBounds[1]-xBounds[0]])), dtype=int) # [rows, cols]

plot([offset[0], offset[0]+patchSize[1], offset[0]+patchSize[1], offset[0], offset[0]], 
     [offset[1], offset[1], offset[1]+patchSize[0], offset[1]+patchSize[0], offset[1]])

# <codecell>

if np.min((largeBBox)[0, :]) > 0.0 :
    print "adding left column"
#     patAPixs = np.arange(0, h, dtype=uint)
if np.min((largeBBox)[1, :]) > 0.0 :
    print "adding top row"
#     patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint))))
if np.max((largeBBox)[1, :]) < bgImage.shape[0] :
    print "adding bottom row"
#     patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint)+h-1)))
if np.max((largeBBox)[0, :]) < bgImage.shape[1] :
    print "adding right column"
#     patAPixs = np.unique(np.concatenate((patAPixs, np.arange(h*(w-1), h*w, dtype=uint))))

# <codecell>

labs = np.copy(labels).reshape(np.prod(patchSize), order='F')
cutEdges = np.zeros(patchSize)
cutSpriteEdges = np.zeros(patchSize)
for factor in graphModel.factors(order=2) :
    varIdxs = factor.variableIndices
    if varIdxs[0] < np.prod(patchSize) and varIdxs[1] < np.prod(patchSize) :
        sPix = np.array([int(np.mod(varIdxs[0],h)), int(varIdxs[0]/h)])
        tPix = np.array([int(np.mod(varIdxs[1],h)), int(varIdxs[1]/h)])
        if (labs[varIdxs[0]] == 1 and labs[varIdxs[1]] == 2) or (labs[varIdxs[1]] == 1 and labs[varIdxs[0]] == 2) :
            print varIdxs, factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            cutSpriteEdges[sPix[0], sPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            cutSpriteEdges[tPix[0], tPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            
        if labs[varIdxs[0]] - labs[varIdxs[1]] != 0 :
            cutEdges[sPix[0], sPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            cutEdges[tPix[0], tPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            
gwv.showCustomGraph(cutEdges)
gwv.showCustomGraph(cutSpriteEdges)
print np.max(cutEdges), np.unique(np.sort(np.ndarray.flatten(cutEdges)))[1]
print np.max(cutSpriteEdges), np.unique(np.sort(np.ndarray.flatten(cutSpriteEdges)))[1]
print np.sum(cutEdges)
print np.sum(cutSpriteEdges)

# <codecell>

labs = np.copy(labels).reshape(np.prod(patchSize), order='F')
cutEdges = np.zeros(patchSize)
cutSpriteEdges = np.zeros(patchSize)
for factor in graphModel.factors(order=2) :
    varIdxs = factor.variableIndices
    if varIdxs[0] < np.prod(patchSize) and varIdxs[1] < np.prod(patchSize) :
        sPix = np.array([int(np.mod(varIdxs[0],h)), int(varIdxs[0]/h)])
        tPix = np.array([int(np.mod(varIdxs[1],h)), int(varIdxs[1]/h)])
        if (labs[varIdxs[0]] == 1 and labs[varIdxs[1]] == 2) or (labs[varIdxs[1]] == 1 and labs[varIdxs[0]] == 2) :
#             print varIdxs, factor.max()
            cutSpriteEdges[sPix[0], sPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            cutSpriteEdges[tPix[0], tPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            
        if labs[varIdxs[0]] - labs[varIdxs[1]] != 0 :
            cutEdges[sPix[0], sPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            cutEdges[tPix[0], tPix[1]] = factor[labs[varIdxs[0]], labs[varIdxs[1]]]
            
gwv.showCustomGraph(cutEdges)
gwv.showCustomGraph(cutSpriteEdges)
print np.max(cutEdges), np.unique(np.sort(np.ndarray.flatten(cutEdges)))[1]
# print np.max(cutSpriteEdges), np.unique(np.sort(np.ndarray.flatten(cutSpriteEdges)))[1]
print np.sum(cutEdges)
print np.sum(cutSpriteEdges)

# <codecell>

print factor[0, 2]

# <codecell>

tmp = np.all(np.concatenate(((diffAB < diffAC).reshape((223, 171, 1)), (labels == 1).reshape((223, 171, 1))), axis=-1), axis=-1)
tmp = (diffAC - diffAB) > 0.08

tmpPatch = np.zeros(patA.shape, dtype=uint8)
for i in xrange(labels.shape[0]) :
    for j in xrange(labels.shape[1]) :
        if labels[i, j] == 0 :
            tmpPatch[i, j, :] = patA[i, j, :]*255
        elif labels[i, j] == 1 and not tmp[i, j] :
            tmpPatch[i, j, :] = patB[i, j, :]*255
        else :
            tmpPatch[i, j, :] = patC[i, j, :]*255
            
figure(); imshow(tmpPatch, interpolation='nearest')

# <codecell>

figure(); imshow(spritePatches[1])

# <codecell>

diffAB = np.sum(np.power(patA - patB, 2), axis=-1)

diffAC = np.sum(np.power(patA - patC, 2), axis=-1)

diffBC = np.sum(np.power(patB - patC, 2), axis=-1)

sumDiffABAC = np.copy(diffAB+diffAC)

maxVal = np.max((np.max(diffAB), np.max(diffAC), np.max(diffBC), np.max(sumDiffABAC)))

figure(); imshow(diffAB, vmin=0, vmax=maxVal)
figure(); imshow(diffAC, vmin=0, vmax=maxVal)
figure(); imshow(diffBC, vmin=0, vmax=maxVal)
figure(); imshow(sumDiffABAC, vmin=0, vmax=maxVal)
print maxVal

# <codecell>

figure(); imshow(spritePriors[0], interpolation='nearest')
figure(); imshow(spritePriors[0]*(1.0-diffAB/np.max(diffAB)), interpolation='nearest')

figure(); imshow(spritePriors[1], interpolation='nearest')
figure(); imshow(spritePriors[1]*(1.0-diffAC/np.max(diffAC)), interpolation='nearest')

# <codecell>

weightedDiffAB = np.exp(-spritePriors[0])/np.sum(np.exp(-spritePriors[0]))*diffAB
weightedDiffAC = np.exp(-spritePriors[1])/np.sum(np.exp(-spritePriors[1]))*diffAC

figure(); imshow(spritePriors[0]*(1.0-weightedDiffAB/np.max(weightedDiffAB)), interpolation='nearest')
figure(); imshow(spritePriors[1]*(1.0-weightedDiffAC/np.max(weightedDiffAC)), interpolation='nearest')

# <codecell>

figure(); imshow(np.exp(-spritePriors[0]))

# <codecell>

## cut a patch from input image based on enlarged bbox
spriteIdx = 1
path = "../data/havana/cutPatches/"
for i in arange(trackedSprites[spriteIdx][DICT_NUM_FRAMES]) :
    img = np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][i]))
    
    ## make bbox bigger
    largeBBox = np.dot(np.array([[0.0, 0.0, 1.2], [0.0, 1.2, 0.0]]), bboxDefaultCorners)
    ## transform according to affine transformation
    largeBBox = np.dot(getAffMat(trackedSprites[spriteIdx][DICT_BBOX_AFFINES][i, :]), 
                        np.vstack((np.ones((1, largeBBox.shape[1])), largeBBox)))
    
    offS = np.array([np.round(np.array([np.min(largeBBox[0, :]), np.min(largeBBox[1, :])]))], dtype=int).T # [x, y]
    pSize = np.array(np.round(np.array([np.max(largeBBox[1, :])-np.min(largeBBox[1, :]), 
                                            np.max(largeBBox[0, :])-np.min(largeBBox[0, :])])), dtype=int) # [rows, cols]
    
    
    Image.fromarray((img[offS[1]:offS[1]+pSize[0], offS[0]:offS[0]+pSize[1]]).astype(numpy.uint8)).save(path+trackedSprites[spriteIdx][DICT_SPRITE_NAME] +"/" + np.string_(i) + ".png")
    
#     figure(); imshow(img)
#     plot([offset[0], offset[0]+patchSize[1], offset[0]+patchSize[1], offset[0], offset[0]], 
#      [offset[1], offset[1], offset[1]+patchSize[0], offset[1]+patchSize[0], offset[1]])

# <codecell>

## visualise an image of the pairwise costs for a given graph (the even indices are nodes and odd are edges, hence the doubled patchSize)
for i in arange(pairCosts.shape[-1]) :
    edgeMapImg = -0.01*np.ones(np.array(patchSize)*2)
    
    edgeMapImg[np.array((gridEdges2D[:, 2]*2+gridEdges2D[:, 0]*2)/2, dtype=int), 
               np.array((gridEdges2D[:, 3]*2+gridEdges2D[:, 1]*2)/2, dtype=int)] = np.copy(pairCosts[:, i])#/np.max(pairCosts[:, i]))
    
    figure(); imshow(np.copy(edgeMapImg), interpolation='nearest', vmin=0.0, vmax=np.max(pairCosts))

# <codecell>

# gwv.showCustomGraph(gradientsY)
figure(); imshow(labels, interpolation='nearest')

# <codecell>

## save an image
Image.fromarray((outputPatch).astype(numpy.uint8)).save("havana_car150_merged.png")

# <codecell>

## visualize patch definite pixels
tmp = np.zeros(patchSize)
tmp[np.mod(patAPixs, h), np.array(patAPixs/h, dtype=int)] = 1
figure(); imshow(tmp, interpolation='nearest')

# <codecell>

def dda(points, imgSize):
    ## digital differential analyzer
    ## points is 2XN array where points[:, 0] = [x, y] coords
    result = np.zeros(imgSize, dtype=int)
    
    for i in arange(points.shape[-1]-1) :
        m = (points[1, i+1] - points[1, i])/(points[0, i+1] - points[0, i])
        xStart = points[0, i]
        yStart = points[1, i]
        xEnd = points[0, i+1]
        yEnd = points[1, i+1]
        x = float(np.copy(xStart))
        y = float(np.copy(yStart))
        print "i=", i, xStart, yStart, xEnd, yEnd, m
        result[y, x] = 1
        if np.abs(m) <= 1.0 :
            while int(np.round(x)) != int(np.round(xEnd)) or int(np.round(y)) != int(np.round(yEnd)) :
                if xStart < xEnd :
                    x += 1
                    y += m
                else :
                    x -= 1
                    y -= m
                    
#                 print int(np.round(x)), int(np.round(y)), int(np.round(xEnd)), int(np.round(yEnd))
                result[int(np.round(y)), int(np.round(x))] = 1
        else :
            while int(np.round(x)) != int(np.round(xEnd)) or int(np.round(y)) != int(np.round(yEnd)) :
                if yStart < yEnd :
                    y += 1
                    x += 1.0/m
                else :
                    y -= 1
                    x -= 1.0/m
                    
#                 print int(np.round(x)), int(np.round(y)), int(np.round(xEnd)), int(np.round(yEnd))
                result[int(np.round(y)), int(np.round(x))] = 1
                
    return result

tmp = dda(bbox, carImg.shape[0:2])
figure(); imshow(tmp, interpolation='nearest')

# <codecell>

### visualize pairwise costs image
def interpolate(val, y0, x0, y1, x1):
    return (val-x0)*(y1-y0)/(x1-x0) + y0;

def base(val):
    if val <= -0.75 :
        return 0
    elif val <= -0.25 :
        return interpolate(val, 0.0, -0.75, 1.0, -0.25)
    elif val <= 0.25 :
        return 1.0
    elif val <= 0.75 :
        return interpolate(val, 1.0, 0.25, 0.0, 0.75)
    else :
        return 0.0


def red(gray) :
    return base(gray - 0.5 )

def green(gray) :
    return base(gray)

def blue(gray) :
    return base(gray + 0.5)

