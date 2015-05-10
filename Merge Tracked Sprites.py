# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import opengm
import numpy as np
import cv2
import time
import os
# import graph_tool as gt
from PIL import Image
import scipy.io as sio
import GraphWithValues as gwv
import sys
import glob

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

def getGraphcutOnOverlap(patchA, patchB, patchAPixels, patchBPixels, multiplier, unaryPriorPatchA, unaryPriorPatchB) :
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
    
#     print "graph setup took", time.time()-s, "seconds"
#     sys.stdout.flush()
    s = time.time()
    graphCut = opengm.inference.GraphCut(gm=gm)
    graphCut.infer()
#     print "graph inference took", time.time()-s, "seconds"
#     sys.stdout.flush()
    
    labels = np.array(graphCut.arg(), dtype=int)
    
    reshapedLabels = np.reshape(np.copy(labels[0:-numLabels]), patchA.shape[0:2], 'F')
#     print gm
#     print gm.evaluate(labels)
    
    return reshapedLabels, unaries, pairwise, gm

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
for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
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

dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

trackedSprites = []
for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
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
for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
    trackedSprites.append(np.load(sprite).item())

## merge tracked sprite with bg
spriteIdx = 11
sequenceLength = len(trackedSprites[spriteIdx][DICT_BBOXES])
showFigs = False

outputPath = dataPath + dataSet + trackedSprites[spriteIdx][DICT_SPRITE_NAME] + "-masked/"

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

