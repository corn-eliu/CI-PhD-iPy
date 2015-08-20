# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import sys
import scipy as sp

import cv2
import time
import os
import scipy.io as sio
import glob

from PIL import Image
from PySide import QtCore, QtGui

import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import opengm

import shutil

app = QtGui.QApplication(sys.argv)

# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
dataSet = "candle1/segmentedAndCropped"
formatString = "{:05d}.png"

# <codecell>

DICT_ORIGINAL_DATASET = 'original_dataset'
DICT_FRAME_LOCS = 'frame_locs'
DICT_SEQUENCE_LENGTH = 'sequence_length'
DICT_ANCHOR_POINT = 'anchor_point'

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

def imgIndicesToLinear(rowRange, colRange, channels, imgSize) :
    ## repeat col indices channels times to be able to index the 3 channels
    colsWithChannels = colRange.repeat(channels)*channels+np.repeat([np.arange(channels)], len(colRange), axis=0).flatten()
    idxs = colsWithChannels.reshape((1, len(colsWithChannels))).repeat(len(rowRange), axis=0).flatten()+(rowRange*imgSize[1]*channels).repeat(len(colsWithChannels))
    return idxs

def getLinearIndicesOfFrameOverlap(aFrameSize, bFrameSize, aAnchorPoint, bAnchorPoint) :
    ## translation vector for matching b's frame coordinate system to the a's frame one
    translation = aAnchorPoint-bAnchorPoint
    
    ## moving b's frame corners into a's frame coordinate system
    bFrameCorners = np.array([[0, 0], bFrameSize[::-1]]) + translation
    
    ## overlap corners in a's frame coordinate system
    aOverlapCorners = np.array([[np.max([bFrameCorners[0, 0], 0]), 
                                np.max([bFrameCorners[0, 1], 0])],
                               [np.min([bFrameCorners[1, 0], aFrameSize[1]]), 
                                np.max([bFrameCorners[1, 1], aFrameSize[0]])]])
    
    ## get overlap corners in b's frame coordinate system
    bOverlapCorners = aOverlapCorners-translation

    aLinearIndices = imgIndicesToLinear(np.arange(aOverlapCorners[0, 1], aOverlapCorners[1, 1]),
                                        np.arange(aOverlapCorners[0, 0], aOverlapCorners[1, 0]), 3, aFrameSize)
    bLinearIndices = imgIndicesToLinear(np.arange(bOverlapCorners[0, 1], bOverlapCorners[1, 1]),
                                        np.arange(bOverlapCorners[0, 0], bOverlapCorners[1, 0]), 3, bFrameSize)
    
    return aLinearIndices, bLinearIndices

# iThFrameLinearIndices, jThFrameLinearIndices = getLinearIndicesOfFrameOverlap(iThLoopFrameSize, jThLoopFrameSize, iThLoopAnchorPoint, jThLoopAnchorPoint)

# <codecell>

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
# s = time.time()
# vectorisedMinusLogMultiNormal(semanticDist.reshape((len(semanticDist), 1)), np.array([0.0]).reshape((1, 1)), np.array([0.0001]).reshape((1, 1)), True)
# print time.time() - s
# s = time.time()
# vectorisedMinusLogMultiNormal(semanticLabels, np.array(desiredLabel).reshape((1, 2)), np.eye(2)*0.0001, True)

# <codecell>

### this is done using matrices
def solveSparseDynProgMRF(unaryCosts, pairwiseCosts, nodesConnectedToLabel) :
    ## assumes unaryCosts has 1 row for each label and 1 col for each variable
    ## assumes arrow heads are rows and arrow tails are cols in pairwiseCosts
    
    ## use the unary and pairwise costs to compute the min cost paths at each node
    # each column represents point n and each row says the index of the k-state that is chosen for the min cost path
    minCostPaths = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]], dtype=int)
    # contains the min cost to reach a certain state k (i.e. row) for point n (i.e. column)
    minCosts = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]])
    # the first row of minCosts is just the unary cost
    minCosts[:, 0] = unaryCosts[:, 0]
    minCostPaths[:, 0] = np.arange(0, unaryCosts.shape[0])        
    
    k = unaryCosts.shape[0]
    for n in xrange(1, unaryCosts.shape[1]) :
        costsPerVariableLabelEdge = minCosts[nodesConnectedToLabel, n-1]
        costsPerVariableLabelEdge += pairwiseCosts[arange(len(pairwiseCosts)).reshape((len(pairwiseCosts), 1)).repeat(nodesConnectedToLabel.shape[-1], axis=-1), nodesConnectedToLabel]
        costsPerVariableLabelEdge += unaryCosts[:, n].reshape((len(unaryCosts), 1)).repeat(nodesConnectedToLabel.shape[-1], axis=-1)
        minCostsIdxs = np.argmin(costsPerVariableLabelEdge, axis=-1)
        ## minCosts
        minCosts[:, n] = costsPerVariableLabelEdge[arange(len(unaryCosts)), minCostsIdxs]
        ## minCostPaths
        minCostPaths[:, n] = nodesConnectedToLabel[arange(len(unaryCosts)), minCostsIdxs]
    
    
    ## now find the min cost path starting from the right most n with lowest cost
    minCostTraversal = np.zeros(unaryCosts.shape[1], dtype=np.int)
    ## last node is the node where the right most node with lowest cost
    minCostTraversal[-1] = np.argmin(minCosts[:, -1]) #minCostPaths[np.argmin(minCosts[:, -1]), -1]
    if np.min(minCosts[:, -1]) == np.inf :
        minCostTraversal[-1] = np.floor((unaryCosts.shape[0])/2)
    
    for i in xrange(len(minCostTraversal)-2, -1, -1) :
        minCostTraversal[i] = minCostPaths[minCostTraversal[i+1], i+1]
        
    return minCostTraversal, np.min(minCosts[:, -1]), minCostPaths

# <codecell>

## setup precomputed loops for each semantics --> should save loops in this format in the first place when I make them
loopedSemantics = []

## loop one from candle1
loopedSemantics.append({
                        DICT_ORIGINAL_DATASET:"candle1/segmentedAndCropped",
                        DICT_FRAME_LOCS:[dataPath+"candle1/segmentedAndCropped/frame-{:05d}.png".format(i+1) for i in xrange(922, 1122+1)],
                        DICT_SEQUENCE_LENGTH:201,
                        DICT_ANCHOR_POINT:np.array([360, 700])
                        })

## loop two from candle1
loopedSemantics.append({
                        DICT_ORIGINAL_DATASET:"candle1/segmentedAndCropped",
                        DICT_FRAME_LOCS:[dataPath+"candle1/segmentedAndCropped/frame-{:05d}.png".format(i+1) for i in xrange(549, 654+1)],
                        DICT_SEQUENCE_LENGTH:106,
                        DICT_ANCHOR_POINT:np.array([360, 700])
                        })
## loop for candle left
loopedSemantics.append({
                        DICT_ORIGINAL_DATASET:"candle3/stabilized/segmentedAndCropped/",
                        DICT_FRAME_LOCS:[dataPath+"candle3/stabilized/segmentedAndCropped/frame-{:05d}.png".format(i+1) for i in xrange(512, 723+1)],
                        DICT_SEQUENCE_LENGTH:212,
                        DICT_ANCHOR_POINT:np.array([550, 500])
                        })
## loop for candle right
loopedSemantics.append({
                        DICT_ORIGINAL_DATASET:"candle3/stabilized/segmentedAndCropped/",
                        DICT_FRAME_LOCS:[dataPath+"candle3/stabilized/segmentedAndCropped/frame-{:05d}.png".format(i+1) for i in xrange(1072, 1268+1)],
                        DICT_SEQUENCE_LENGTH:197,
                        DICT_ANCHOR_POINT:np.array([550, 500])
                        })

## do some integrity checks
for semantic in loopedSemantics :
    if len(semantic[DICT_FRAME_LOCS]) <= 0:
        raise Exception("Loop does not contain any frames")
    if len(semantic[DICT_FRAME_LOCS]) != semantic[DICT_SEQUENCE_LENGTH] :
        raise Exception("Sequence length does not match number of frame locations")
        
overlapFrames = 5
      
transitionSequences = []
transitionSequences.append({
                        DICT_ORIGINAL_DATASET:"candle3/stabilized/segmentedAndCropped/",
                        DICT_FRAME_LOCS:[dataPath+"candle3/stabilized/segmentedAndCropped/frame-{:05d}.png".format(i+1) for i in xrange(0, 511+overlapFrames+1)],
                        DICT_SEQUENCE_LENGTH:517,
                        DICT_ANCHOR_POINT:np.array([550, 500])})

transitionSequences.append({
                        DICT_ORIGINAL_DATASET:"candle3/stabilized/segmentedAndCropped/",
                        DICT_FRAME_LOCS:[dataPath+"candle3/stabilized/segmentedAndCropped/frame-{:05d}.png".format(i+1) for i in xrange(724-overlapFrames, 1071+overlapFrames+1)],
                        DICT_SEQUENCE_LENGTH:358,
                        DICT_ANCHOR_POINT:np.array([550, 500])})

transitionSequences.append({
                        DICT_ORIGINAL_DATASET:"candle3/stabilized/segmentedAndCropped/",
                        DICT_FRAME_LOCS:[dataPath+"candle3/stabilized/segmentedAndCropped/frame-{:05d}.png".format(i+1) for i in xrange(1269-overlapFrames, 1513+1)],
                        DICT_SEQUENCE_LENGTH:250,
                        DICT_ANCHOR_POINT:np.array([550, 500])})

## do some integrity checks
for transition in transitionSequences :
    if len(transition[DICT_FRAME_LOCS]) <= 0:
        raise Exception("Sequence does not contain any frames")
    if len(transition[DICT_FRAME_LOCS]) != transition[DICT_SEQUENCE_LENGTH] :
        raise Exception("Sequence length does not match number of frame locations")

# <codecell>

## compute distance matrix between frames of different loops
# interLoopDistances = {}
# s = time.time()
# for i in xrange(len(loopedSemantics)) :
    
#     t = time.time()
#     iThLoopFrameSize = np.array(cv2.imread(loopedSemantics[i][DICT_FRAME_LOCS][0]).shape[0:2])
#     iThLoopFrames = np.zeros((loopedSemantics[i][DICT_SEQUENCE_LENGTH], np.prod(iThLoopFrameSize)*3), dtype=np.float32)
    
#     for frame in xrange(loopedSemantics[i][DICT_SEQUENCE_LENGTH]) :
#         iThLoopFrames[frame, :] = np.array(cv2.cvtColor(cv2.imread(loopedSemantics[i][DICT_FRAME_LOCS][frame]), 
#                                                         cv2.COLOR_BGR2RGB), dtype=np.float32).reshape((iThLoopFrames.shape[1]))/255.0
#     print "Loaded frames of loop", i, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
    
#     if np.string_(i) not in interLoopDistances.keys() :
#         t = time.time()
#         interLoopDistances[np.string_(i)] = distEuc(iThLoopFrames)
#         print "Computed distances in loop", i, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
    
#     for j in xrange(i+1, len(loopedSemantics)) :
        
#         t = time.time()
#         jThLoopFrameSize = np.array(cv2.imread(loopedSemantics[j][DICT_FRAME_LOCS][0]).shape[0:2])
#         jThLoopFrames = np.zeros((loopedSemantics[j][DICT_SEQUENCE_LENGTH], np.prod(jThLoopFrameSize)*3), dtype=np.float32)
        
#         for frame in xrange(loopedSemantics[j][DICT_SEQUENCE_LENGTH]) :
#             jThLoopFrames[frame, :] = np.array(cv2.cvtColor(cv2.imread(loopedSemantics[j][DICT_FRAME_LOCS][frame]), 
#                                                             cv2.COLOR_BGR2RGB), dtype=np.float32).reshape((jThLoopFrames.shape[1]))/255.0
#         print "Loaded frames of loop", j, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
        
#         if np.string_(i)+np.string_(j) not in interLoopDistances.keys() :
#             t = time.time()
#             if np.any(iThLoopFrameSize != jThLoopFrameSize) :
#                 iThFrameLinearIndices, jThFrameLinearIndices = getLinearIndicesOfFrameOverlap(iThLoopFrameSize, jThLoopFrameSize,
#                                                                                               loopedSemantics[i][DICT_ANCHOR_POINT],
#                                                                                               loopedSemantics[j][DICT_ANCHOR_POINT])
#                 interLoopDistances[np.string_(i)+np.string_(j)] = distEuc2(iThLoopFrames[:, iThFrameLinearIndices],
#                                                                            jThLoopFrames[:, jThFrameLinearIndices])
#             else :
#                 interLoopDistances[np.string_(i)+np.string_(j)] = distEuc2(iThLoopFrames, jThLoopFrames)

#             print "Computed distance between", i, "and", j, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
        
# print "Done in", np.string_(time.time() - s), "seconds"
# del iThLoopFrames, jThLoopFrames

# np.save(dataPath + "candle_semantics_interLoopDistances.npy", interLoopDistances)
interLoopDistances = np.load(dataPath+"candle_semantics_interLoopDistances.npy").item(0)

# <codecell>

# ## distances between transition sequences
# interSequenceDistances = {}
# s = time.time()
# for i in xrange(len(transitionSequences)) :
    
#     t = time.time()
#     iThSequenceFrameSize = np.array(cv2.imread(transitionSequences[i][DICT_FRAME_LOCS][0]).shape[0:2])
#     iThSequenceFrames = np.zeros((transitionSequences[i][DICT_SEQUENCE_LENGTH], np.prod(iThSequenceFrameSize)*3), dtype=np.float32)
    
#     for frame in xrange(transitionSequences[i][DICT_SEQUENCE_LENGTH]) :
#         iThSequenceFrames[frame, :] = np.array(cv2.cvtColor(cv2.imread(transitionSequences[i][DICT_FRAME_LOCS][frame]), 
#                                                         cv2.COLOR_BGR2RGB), dtype=np.float32).reshape((iThSequenceFrames.shape[1]))/255.0
#     print "Loaded frames of sequence", i, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
    
#     if np.string_(i) not in interSequenceDistances.keys() :
#         t = time.time()
#         interSequenceDistances[np.string_(i)] = distEuc(iThSequenceFrames)
#         print "Computed distances in sequence", i, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
    
#     for j in xrange(i+1, len(transitionSequences)) :
        
#         t = time.time()
#         jThSequenceFrameSize = np.array(cv2.imread(transitionSequences[j][DICT_FRAME_LOCS][0]).shape[0:2])
#         jThSequenceFrames = np.zeros((transitionSequences[j][DICT_SEQUENCE_LENGTH], np.prod(jThSequenceFrameSize)*3), dtype=np.float32)
        
#         for frame in xrange(transitionSequences[j][DICT_SEQUENCE_LENGTH]) :
#             jThSequenceFrames[frame, :] = np.array(cv2.cvtColor(cv2.imread(transitionSequences[j][DICT_FRAME_LOCS][frame]), 
#                                                             cv2.COLOR_BGR2RGB), dtype=np.float32).reshape((jThSequenceFrames.shape[1]))/255.0
#         print "Loaded frames of sequence", j, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
            
#         if np.string_(i)+np.string_(j) not in interSequenceDistances.keys() :
#             t = time.time()
#             if np.any(iThSequenceFrameSize != jThSequenceFrameSize) :
#                 iThFrameLinearIndices, jThFrameLinearIndices = getLinearIndicesOfFrameOverlap(iThSequenceFrameSize, jThSequenceFrameSize,
#                                                                                               transitionSequences[i][DICT_ANCHOR_POINT],
#                                                                                               transitionSequences[j][DICT_ANCHOR_POINT])
#                 interSequenceDistances[np.string_(i)+np.string_(j)] = distEuc2(iThSequenceFrames[:, iThFrameLinearIndices],
#                                                                            jThSequenceFrames[:, jThFrameLinearIndices])
#             else :
#                 interSequenceDistances[np.string_(i)+np.string_(j)] = distEuc2(iThSequenceFrames, jThSequenceFrames)

#             print "Computed distance between", i, "and", j, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
        
# print "Done in", np.string_(time.time() - s), "seconds"
# del iThSequenceFrames, jThSequenceFrames
# np.save(dataPath + "candle_semantics_interSequenceDistancess.npy", interSequenceDistances)
interSequenceDistances = np.load(dataPath+"candle_semantics_interSequenceDistancess.npy").item(0)

# <codecell>

# loopToTransitionDistances = {}
# s = time.time()
# for i in xrange(len(transitionSequences)) :
    
#     t = time.time()
#     transitionFrameSize = np.array(cv2.imread(transitionSequences[i][DICT_FRAME_LOCS][0]).shape[0:2])
#     transitionFrames = np.zeros((transitionSequences[i][DICT_SEQUENCE_LENGTH], np.prod(transitionFrameSize)*3), dtype=np.float32)
    
#     for frame in xrange(transitionSequences[i][DICT_SEQUENCE_LENGTH]) :
#         transitionFrames[frame, :] = np.array(cv2.cvtColor(cv2.imread(transitionSequences[i][DICT_FRAME_LOCS][frame]), 
#                                                         cv2.COLOR_BGR2RGB), dtype=np.float32).reshape((transitionFrames.shape[1]))/255.0
#     print "Loaded frames of sequence", i, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
    
#     for j in xrange(len(loopedSemantics)) :
        
#         t = time.time()
#         loopFrameSize = np.array(cv2.imread(loopedSemantics[j][DICT_FRAME_LOCS][0]).shape[0:2])
#         loopFrames = np.zeros((loopedSemantics[j][DICT_SEQUENCE_LENGTH], np.prod(loopFrameSize)*3), dtype=np.float32)
        
#         for frame in xrange(loopedSemantics[j][DICT_SEQUENCE_LENGTH]) :
#             loopFrames[frame, :] = np.array(cv2.cvtColor(cv2.imread(loopedSemantics[j][DICT_FRAME_LOCS][frame]), 
#                                                             cv2.COLOR_BGR2RGB), dtype=np.float32).reshape((loopFrames.shape[1]))/255.0
#         print "Loaded frames of loop", j, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
        
#         if np.string_(i)+np.string_(j) not in loopToTransitionDistances.keys() :
#             t = time.time()
#             if np.any(transitionFrameSize != loopFrameSize) :
#                 transitionLinearIndices, loopLinearIndices = getLinearIndicesOfFrameOverlap(transitionFrameSize, loopFrameSize,
#                                                                                               transitionSequences[i][DICT_ANCHOR_POINT],
#                                                                                               loopedSemantics[j][DICT_ANCHOR_POINT])
#                 loopToTransitionDistances[np.string_(i)+np.string_(j)] = distEuc2(transitionFrames[:, transitionLinearIndices],
#                                                                            loopFrames[:, loopLinearIndices])
#             else :
#                 loopToTransitionDistances[np.string_(i)+np.string_(j)] = distEuc2(transitionFrames, loopFrames)

#             print "Computed distance between", i, "and", j, "in", np.string_(time.time() - t), "seconds"; sys.stdout.flush()
        
# print "Done in", np.string_(time.time() - s), "seconds"
# del transitionFrames, loopFrames
# np.save(dataPath + "candle_semantics_loopToTransitionDistances.npy", loopToTransitionDistances)
loopToTransitionDistances = np.load(dataPath+"candle_semantics_loopToTransitionDistances.npy").item(0)

# <codecell>

# gwv.showCustomGraph(interLoopDistances['02'])
## make full distance matrix between looped semantics
totalNumberOfFramesInLoops = 0
for semantics in loopedSemantics :
    totalNumberOfFramesInLoops += semantics[DICT_SEQUENCE_LENGTH]
    
fullLoopedSemanticsDistMat = (1-np.eye(totalNumberOfFramesInLoops))*300
currentRow = 0
colDelta = 0
for i in xrange(len(loopedSemantics)) :
    iLength = loopedSemantics[i][DICT_SEQUENCE_LENGTH]
    fullLoopedSemanticsDistMat[currentRow:currentRow+iLength, currentRow:currentRow+iLength] = interLoopDistances[np.string_(i)]
    colDelta += iLength
    currentCol = colDelta
    for j in xrange(i+1, len(loopedSemantics)) :
        print i, j, currentRow, currentCol
        jLength = loopedSemantics[j][DICT_SEQUENCE_LENGTH]
        fullLoopedSemanticsDistMat[currentRow:currentRow+iLength, currentCol:currentCol+jLength] = interLoopDistances[np.string_(i)+np.string_(j)]
        fullLoopedSemanticsDistMat[currentCol:currentCol+jLength, currentRow:currentRow+iLength] = interLoopDistances[np.string_(i)+np.string_(j)].T
        currentCol += jLength
    currentRow += iLength
    
# gwv.showCustomGraph(fullLoopedSemanticsDistMat)

# <codecell>

# gwv.showCustomGraph(interLoopDistances['02'])
## make full distance matrix between transition sequences
totalNumberOfFramesInTransitions = 0
for sequence in transitionSequences :
    totalNumberOfFramesInTransitions += sequence[DICT_SEQUENCE_LENGTH]
    
fullTransitionSequenceDistMat = (1-np.eye(totalNumberOfFramesInTransitions))*300
currentRow = 0
colDelta = 0
for i in xrange(len(transitionSequences)) :
    iLength = transitionSequences[i][DICT_SEQUENCE_LENGTH]
    fullTransitionSequenceDistMat[currentRow:currentRow+iLength, currentRow:currentRow+iLength] = interSequenceDistances[np.string_(i)]
    colDelta += iLength
    currentCol = colDelta
    for j in xrange(i+1, len(transitionSequences)) :
        print i, j, currentRow, currentCol
        jLength = transitionSequences[j][DICT_SEQUENCE_LENGTH]
        fullTransitionSequenceDistMat[currentRow:currentRow+iLength, currentCol:currentCol+jLength] = interSequenceDistances[np.string_(i)+np.string_(j)]
        fullTransitionSequenceDistMat[currentCol:currentCol+jLength, currentRow:currentRow+iLength] = interSequenceDistances[np.string_(i)+np.string_(j)].T
        currentCol += jLength
    currentRow += iLength
    
# gwv.showCustomGraph(fullTransitionSequenceDistMat)

# <codecell>

# gwv.showCustomGraph(interLoopDistances['02'])
## make full distance matrix between transition sequences and loops
    
fullTransitionsVsLoopsDistMat = np.zeros((totalNumberOfFramesInTransitions, totalNumberOfFramesInLoops))
currentRow = 0
for i in xrange(len(transitionSequences)) :
    iLength = transitionSequences[i][DICT_SEQUENCE_LENGTH]
    currentCol = 0
    for j in xrange(len(loopedSemantics)) :
        print i, j, currentRow, currentCol
        jLength = loopedSemantics[j][DICT_SEQUENCE_LENGTH]
        fullTransitionsVsLoopsDistMat[currentRow:currentRow+iLength, currentCol:currentCol+jLength] = loopToTransitionDistances[np.string_(i)+np.string_(j)]
        currentCol += jLength
    currentRow += iLength
    
# gwv.showCustomGraph(fullTransitionsVsLoopsDistMat)

# <codecell>

## combine al distance matrices together
fullDistanceMatrix = np.concatenate((fullLoopedSemanticsDistMat, fullTransitionsVsLoopsDistMat.T), axis=-1)
fullDistanceMatrix = np.concatenate((fullDistanceMatrix, np.concatenate((fullTransitionsVsLoopsDistMat, fullTransitionSequenceDistMat), axis=-1)), axis=0)
gwv.showCustomGraph(fullDistanceMatrix)

# <codecell>

## make the transition matrix where the arrow heads are rows and arrow tails are cols (i.e. I can go from frame colIdx to frame rowIdx)
# transitionCosts = np.zeros((totalNumberOfFrames, totalNumberOfFrames))
# currentRow = 0
# colDelta = 0
# for i in xrange(len(loopedSemantics)) :
#     iLength = loopedSemantics[i][DICT_SEQUENCE_LENGTH]
#     ## I roll each distance matrix left by 1 column which means that to go to frame i from frame j, I check the distance between frame j+1 and i which is now at location (i, j)
#     transitionCosts[currentRow:currentRow+iLength, currentRow:currentRow+iLength] = np.roll(interLoopDistances[np.string_(i)], -1, axis=-1)
#     colDelta += iLength
#     currentCol = colDelta
#     for j in xrange(i+1, len(loopedSemantics)) :
#         print i, j, currentRow, currentCol
#         jLength = loopedSemantics[j][DICT_SEQUENCE_LENGTH]
#         transitionCosts[currentRow:currentRow+iLength, currentCol:currentCol+jLength] = np.roll(interLoopDistances[np.string_(i)+np.string_(j)], -1, axis=-1)
#         transitionCosts[currentCol:currentCol+jLength, currentRow:currentRow+iLength] = np.roll(interLoopDistances[np.string_(i)+np.string_(j)].T, -1, axis=-1)
#         currentCol += jLength
#     currentRow += iLength
    
# gwv.showCustomGraph(transitionCosts)

# transitionCosts = transitionCosts / np.max(transitionCosts)
# badJumps = np.argwhere(transitionCosts > 0.15)
# transitionCosts[badJumps[:, 0], badJumps[:, 1]] = 1000000.0
# ## add some number to the costs to give some cost to following the timeline, which should not influence following the timeline loop-wise but should reduce the length of
# ## transition animations
# transitionCosts += 0.01
# gwv.showCustomGraph(np.log(transitionCosts))

# <codecell>

## make the transition matrix where the arrow heads are rows and arrow tails are cols (i.e. I can go from frame colIdx to frame rowIdx)
## use all transition matrices now
transitionCosts = np.copy(fullDistanceMatrix)
separationIndices = []
separationIndices.append(0)
for i in xrange(len(loopedSemantics)) :
    separationIndices.append(separationIndices[-1]+loopedSemantics[i][DICT_SEQUENCE_LENGTH])
for i in xrange(len(transitionSequences)) :
    separationIndices.append(separationIndices[-1]+transitionSequences[i][DICT_SEQUENCE_LENGTH])

for i, j in zip(separationIndices[:-1], separationIndices[1:]) :
    for k, l in zip(separationIndices[:-1], separationIndices[1:]) :
        transitionCosts[i:j, k:l] = np.roll(transitionCosts[i:j, k:l], -1, axis=-1)
        
        if j > separationIndices[len(loopedSemantics)] or l > separationIndices[len(loopedSemantics)] :
            ## dealing with transition sequences that cannot loop at the end
            print i, j, k, l
            ## set last column and row to maxCost
            transitionCosts[i:j, l-1] = np.max(fullDistanceMatrix)
            transitionCosts[j-1, k:l] = np.max(fullDistanceMatrix)
#         print i, j, k, l 
        
    
gwv.showCustomGraph(transitionCosts / np.max(transitionCosts))

transitionCosts = transitionCosts / np.max(transitionCosts)
# transitionCosts = transitionCosts*1000.0
# transitionCosts = transitionCosts / np.max(transitionCosts)
badJumps = np.argwhere(transitionCosts > 0.25)
transitionCosts[badJumps[:, 0], badJumps[:, 1]] = 1000000.0
## add some number to the costs to give some cost to following the timeline, which should not influence following the timeline loop-wise but should reduce the length of
## transition animations
transitionCosts += 0.1

## get rid of short jumps ?
if True :
    ## can jump from frame in first column to frame in second column
    lowCostTransitions = np.argwhere(transitionCosts != np.max(transitionCosts), )
    jumpLength = np.abs(lowCostTransitions[:, 0]-lowCostTransitions[:, 1])
    shortJumps = np.all(np.array([jumpLength > 1, jumpLength < 10]), axis=0)
    shortJumps = np.any(np.array([shortJumps, jumpLength == 0]), axis=0)

    transitionCosts[lowCostTransitions[shortJumps, :][:, 0], lowCostTransitions[shortJumps, :][:, 1]] = np.max(transitionCosts)


gwv.showCustomGraph(np.log(transitionCosts))

# <codecell>

def getAnimation(pairwise, fullSequenceSemantics, desiredSemantics, framesToProduce, startFrame, distVariance) :
    ## The lower distVariance, the higher the penalty for showing a frame having something different from desiredSemantics
    
    
    ## now try and do the optimization completely vectorized
    ## number of edges connected to each label node of variable n (pairwise stores node at arrow tail as cols and at arrow head as rows)
    maxEdgesPerLabel = np.max(np.sum(np.array(pairwise != np.max(pairwise), dtype=int), axis=-1))
    ## initialize this to index of connected label node with highest edge cost (which is then used as padding)
    ## it contains for each label node of variable n (indexed by rows), all the label nodes of variable n-1 it is connected to by non infinite cost edge (indexed by cols)
    nodesConnectedToLabel = np.argmax(pairwise, axis=-1).reshape((len(pairwise), 1)).repeat(maxEdgesPerLabel, axis=-1)
    
    ##### OLD WAY OF GETTING THE SPARSE INDICES #####
#     sparseIndices = np.where(pairwise.T != np.max(pairwise))
#     # print sparseIndices
#     tailIndices = sparseIndices[0]
#     headIndices = sparseIndices[1]

#     ## this contains which label of variable n-1 is connected to which label of variable n
#     indicesInLabelSpace = [list(tailIndices[np.where(headIndices == i)[0]]) for i in np.unique(headIndices)]
    
    ##### NEW WAY OF GETTING THE SPARSE INDICES #####
    sparseIndices = np.where(pairwise != np.max(pairwise))
    # print sparseIndices
    tailIndices = sparseIndices[1]
    headIndices = sparseIndices[0]
    
    indicesInLabelSpace = [tailIndices[headIndices == i] for i in np.arange(len(pairwise))]

    for headLabel, tailLabels in zip(arange(0, len(nodesConnectedToLabel)), indicesInLabelSpace) :
        nodesConnectedToLabel[headLabel, 0:len(tailLabels)] = tailLabels

#     print nodesConnectedToLabel.shape

    unaries = vectorisedMinusLogMultiNormal(fullSequenceSemantics, desiredSemantics.reshape((1, len(loopedSemantics))), np.eye(len(loopedSemantics))*distVariance, True)
    unaries = unaries.reshape((len(pairwise), 1)).repeat(framesToProduce, axis=-1)
    unaries[:, 0] = 1000000.0
    unaries[startFrame, 0] = 0.0

    # gwv.showCustomGraph(unaries)
    # print "computed costs for sprite", spriteIndices[idx], "in", time.time() - tic; sys.stdout.flush()
    # tic = time.time()
    # minCostTraversal, minCost = solveMRF(unaries, pairwise)
    #     minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)
    minCostTraversal, minCost, paths = solveSparseDynProgMRF(unaries, pairwise, nodesConnectedToLabel)
    return minCostTraversal#, minCost, unaries

# <codecell>

## give semantics to each looped semantics and node to the transition frames
sequenceSemantics = np.empty((0, len(loopedSemantics)))
for s in xrange(len(loopedSemantics)) :
    sems = np.zeros((loopedSemantics[s][DICT_SEQUENCE_LENGTH], len(loopedSemantics)))
    sems[:, s] = 1.0
    sequenceSemantics = np.concatenate((sequenceSemantics, sems))
sequenceSemantics = np.concatenate((sequenceSemantics, np.zeros((totalNumberOfFramesInTransitions, len(loopedSemantics)))))

# <codecell>

gwv.showCustomGraph(transitionCosts)

# <codecell>

semanticsToUse = np.array([0, 1, 2, 3])
sequenceLength = 100

semanticsFrameRanges = [0]
for semantics in loopedSemantics :
    semanticsFrameRanges.append(semantics[DICT_SEQUENCE_LENGTH])
semanticsFrameRanges = np.cumsum(np.array(semanticsFrameRanges))
semanticsFrameRanges = np.array([semanticsFrameRanges[:-1], semanticsFrameRanges[1:]]).T
print semanticsFrameRanges
    
## find transitions between each semantics and the others
semanticsCombinations = []
for s in semanticsToUse :
    for t in semanticsToUse :
        if s != t :
            semanticsCombinations.append(np.array([s, t]))

frameInterval = 10
transitionAnimations = []
for combination in semanticsCombinations :
    transitionAnimations.append([])
    print combination
    fromSemantics, toSemantics = combination
    desiredSemantics = np.zeros(len(loopedSemantics))
    desiredSemantics[toSemantics] = 1.0
    print desiredSemantics
    startFrames = np.arange(semanticsFrameRanges[fromSemantics, 0], semanticsFrameRanges[fromSemantics, 1], frameInterval)
    for startFrame in startFrames :
        animation = getAnimation(transitionCosts, sequenceSemantics, desiredSemantics, sequenceLength, startFrame, 0.0005)
        for f in xrange(len(animation)) :
            if np.all(sequenceSemantics[animation[f], :] == desiredSemantics) :
                transitionAnimations[-1].append(animation[:f+1])
                break
            if f == len(animation)-1 :
                print "Animation does not reach desired semantics"
    
        sys.stdout.write('\r' + "Found animation from " + np.string_(combination[0]) + " to " + np.string_(combination[1]) +
                         " starting at " + np.string_(startFrame) + " (" + np.string_(startFrames[-1]) + ")")
        sys.stdout.flush()
    print 


# desiredSemantics = np.array([0.0, 1.0, 0.0, 0.0])
# startFrame = 320

# print getAnimation(transitionCosts, sequenceSemantics, sequenceLength, startFrame)

# <codecell>

allNeededFrames = np.empty(0, dtype=int)

for s in semanticsToUse :
    allNeededFrames = np.concatenate((allNeededFrames, np.arange(semanticsFrameRanges[s, 0], semanticsFrameRanges[s, 1])))
    
for t in transitionAnimations :
    for animation in t :
        allNeededFrames = np.concatenate((allNeededFrames, animation))
        
allNeededFrames = np.unique(allNeededFrames)
print allNeededFrames

## make an array where each element tells me where the last frame of that looped semantic or transition sequence is
frameSequencesLengthCumSum = []
for s in loopedSemantics :
    frameSequencesLengthCumSum.append(s[DICT_SEQUENCE_LENGTH])
for t in transitionSequences :
    frameSequencesLengthCumSum.append(t[DICT_SEQUENCE_LENGTH])
frameSequencesLengthCumSum = np.cumsum(np.array(frameSequencesLengthCumSum))
print frameSequencesLengthCumSum

## which sequence each frame comes from (if id is bigger than len(loopedSemantics), then id indexes transitionSequences)
sequencePerFrame = np.zeros_like(allNeededFrames)
for i in xrange(len(allNeededFrames)) :
    sequencePerFrame[i] = np.sum(allNeededFrames[i] >= frameSequencesLengthCumSum)
    
print sequencePerFrame

## actual indices of the frames in their sequences rather than indices in the transition matrix
allNeededFramesInSequenceIdxs = allNeededFrames-np.concatenate(([0], frameSequencesLengthCumSum))[sequencePerFrame]
print allNeededFramesInSequenceIdxs

## now build frameLocs using the above info
frameLocs = []
for i in xrange(len(allNeededFrames)) :
    if sequencePerFrame[i] < len(loopedSemantics) :
        frameLocs.append(loopedSemantics[sequencePerFrame[i]][DICT_FRAME_LOCS][allNeededFramesInSequenceIdxs[i]])
    else :
        frameLocs.append(transitionSequences[sequencePerFrame[i]-len(loopedSemantics)][DICT_FRAME_LOCS][allNeededFramesInSequenceIdxs[i]])
print frameLocs

# <codecell>

## build the full graph now based on used looped semantics and the transition animation
semanticsGraph = []
## keep track of what loop frame has been put in each node (row idx is index of node)
loopFramesInNodes = np.empty(0, dtype=int)
## add all the nodes used by the looped semantics I'm using in the graph
for s in semanticsToUse :
    for f in xrange(loopedSemantics[s][DICT_SEQUENCE_LENGTH]) :
        globalFrameIdx = semanticsFrameRanges[s, 0] + f
        semanticsGraph.append(np.array([int(np.argwhere(allNeededFrames == globalFrameIdx)), s]))
        loopFramesInNodes = np.concatenate((loopFramesInNodes, [globalFrameIdx]), axis=1)
        
## add connections between loop nodes to build the actual nodes
for s in semanticsToUse :
    for f in xrange(loopedSemantics[s][DICT_SEQUENCE_LENGTH]) :
        globalFrameIdx = semanticsFrameRanges[s, 0] + f
        nodeIdx = int(np.argwhere(loopFramesInNodes == globalFrameIdx))
        neighbourNodeIdx = int(np.argwhere(loopFramesInNodes == semanticsFrameRanges[s, 0] + np.mod(f+1, loopedSemantics[s][DICT_SEQUENCE_LENGTH])))
        semanticsGraph[nodeIdx] = np.concatenate((semanticsGraph[nodeIdx], [neighbourNodeIdx]), axis=1)

## go through transitionAnimations and add them to semanticsGraph
for s in xrange(len(semanticsCombinations)) :
    fromSemantics, toSemantics = semanticsCombinations[s]
    ## transition animations are already using global frame idxs
    for transitionAnimation in transitionAnimations[s] : #[0:1] :
#         print fromSemantics, toSemantics
#         print transitionAnimation
#         print "from node", int(np.argwhere(loopFramesInNodes == transitionAnimation[0])), "to node", int(np.argwhere(loopFramesInNodes == transitionAnimation[-1]))
        ## add node for second frame in animation
        semanticsGraph.append(np.array([int(np.argwhere(allNeededFrames == transitionAnimation[1])), toSemantics]))
        ## add edge from first frame in animation (i.e. a node from loop fromSemantics) to second frame which is a completely new node
        nodeIdx = int(np.argwhere(loopFramesInNodes == transitionAnimation[0]))
        semanticsGraph[nodeIdx] = np.concatenate((semanticsGraph[nodeIdx], [len(semanticsGraph)-1]), axis=1)
        
        ## add edge between new node for frame and previous node
        for frame in transitionAnimation[2:-1] :
            semanticsGraph.append(np.array([int(np.argwhere(allNeededFrames == frame)), toSemantics]))
            semanticsGraph[-2] = np.concatenate((semanticsGraph[-2], [len(semanticsGraph)-1]), axis=1)
#             print frame
            
        ## add edge between last frame's node and the last frame in the animation (i.e. a node from loop toSemantics)
        semanticsGraph[-1] = np.concatenate((semanticsGraph[-1], [int(np.argwhere(loopFramesInNodes == transitionAnimation[-1]))]), axis=1)

# <codecell>

if not os.path.isdir(dataPath+"precomputedSemanticsGraph/") :
    os.mkdir(dataPath+"precomputedSemanticsGraph/")
csvFile = open(dataPath+"precomputedSemanticsGraph/semanticsGraph.csv", 'w')
## [num_semantics, num_nodes, num_frames]
csvFile.write("## [num_semantics, num_nodes, num_frames]\n")
csvFile.write(np.string_(len(semanticsToUse)) + "," + np.string_(len(semanticsGraph)) + "," + np.string_(len(frameLocs)) + "\n")

## HACK to just try the UI ##
csvFile.write("## [frame_image_id, semantics_id, connected_node1, ...]\n")
## format semantics graph and save as csv
semanticsGraphStrings = []
for node in semanticsGraph :
    ## [frame_image_id, semantics_id, connected_node1, ...]
    semanticsGraphStrings.append(','.join("{0:0d}".format(n) for n in node ) + "\n")
# print semanticsGraphs
csvFile.writelines(semanticsGraphStrings)
csvFile.close()
# np.savetxt(dataPath+"precomputedSemanticsGraph/semanticsGraph.csv", semanticsGraph, delimiter=',')

# <codecell>

## copy frames to right folder
for frame in xrange(len(frameLocs)) :
    shutil.copyfile(frameLocs[frame], dataPath+"precomputedSemanticsGraph/frame-{0:05d}.png".format(frame+1))

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text="", parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.image = None
        self.color = QtGui.QColor(QtCore.Qt.black)
        
    def setImage(self, image) : 
        self.image = image.copy()
        self.setMinimumSize(self.image.size())
        self.update()
        
    def setBackgroundColor(self, color) :
        self.color = color
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        
        if self.image != None :
#             upperLeft = ((self.width()-self.image.width())/2, (self.height()-self.image.height())/2)
            upperLeft = (self.width()/2-self.image.width()/2, self.height()-self.image.height())
    
            ## draw background
            painter.setBrush(QtGui.QBrush(self.color))
            painter.setPen(QtGui.QPen(self.color, 0, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawRect(QtCore.QRect(upperLeft[0], upperLeft[1], self.image.width(), self.image.height()))
            
            ## draw image
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.image)
                
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.setWindowTitle("Precomputed Semantic Looping")
        self.resize(1280, 720)
        
        self.playIcon = QtGui.QIcon("play.png")
        self.pauseIcon = QtGui.QIcon("pause.png")
        self.doPlaySequence = False
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.renderOneFrame)
        self.lastRenderTime = time.time()
        self.oldInfoText = ""
        
        self.createGUI()
        
        self.semanticsGraphLocation = dataPath+"precomputedSemanticsGraph/"
        self.loadSemanticsGraph(self.semanticsGraphLocation)
        
        self.currentNode = 0
        self.currentSemantics = 0
        self.renderOneFrame()
        
        self.setFocus()
        
    def loadSemanticsGraph(self, location) :
    
        self.semanticsGraph = []
        self.numSemantics = 0
        self.numNodes = 0
        self.numFrames = 0
        
        if not os.path.isfile(location+"semanticsGraph.csv") :
            print "No semantics graph at given location"
            self.close()
        else :
            csvFile = open(location+"semanticsGraph.csv", 'r')
            readInfo = False
            for line in csvFile :
                if not line.startswith("##") :
                    if not readInfo :
                        readInfo = True
                        self.numSemantics, self.numNodes, self.numFrames = np.array(line.split(","), dtype=int)
                    else :
                        self.semanticsGraph.append(np.array(line.split(","), dtype=int))
            
            csvFile.close()
            
            
        ## here I should really load the sorted list of frames in the location where the semanticGraph is stored
#         self.frameLocs = np.sort(glob.glob(location+"frame*.png"))
#         if len(self.frameLocs) != self.numFrames :
#             print "Number of frames in folder doesn't match number of frames in graph"
#             self.close()
        
        self.frameLocs = frameLocs
        
        self.allFrames = []
        for frame in self.frameLocs :
            self.allFrames.append(np.ascontiguousarray(cv2.imread(frame, cv2.CV_LOAD_IMAGE_UNCHANGED)))
        
    def renderOneFrame(self) :
        ## get background image
        if self.currentNode >= 0 and self.currentNode < len(self.semanticsGraph) :
            frameIdx = self.semanticsGraph[self.currentNode][0]
            if frameIdx >= 0 and frameIdx < self.numFrames :
#                 im = np.ascontiguousarray(cv2.imread(self.frameLocs[frameIdx], cv2.CV_LOAD_IMAGE_UNCHANGED))
                im = self.allFrames[frameIdx]
                img = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
#                 self.frameLabel.setFixedSize(img.width(), img.height())
                self.frameLabel.setImage(img)

                self.frameInfo.setText("Rendering at " + np.string_(int(1.0/(time.time() - self.lastRenderTime))) + " FPS\n" + 
                                       "Current Semantics " + np.string_(self.currentSemantics) + "\n" + 
                                       np.string_(self.currentNode) + " --> " + self.frameLocs[frameIdx])
                self.lastRenderTime = time.time()
                self.stepInGraph()
    
    def stepInGraph(self) :
        if self.currentNode >= 0 and self.currentNode < len(self.semanticsGraph) :
            neighbourNodes = self.semanticsGraph[self.currentNode][2:]
            for node in neighbourNodes :
                if self.semanticsGraph[node][1] == self.currentSemantics :
                    self.currentNode = node
                    break
                else :
                    self.currentNode = node
            
    def closeEvent(self, event) :
        print "closing"
        sys.stdout.flush()
            
    def deleteGeneratedSequence(self) :
        del self.generatedSequence
        self.generatedSequence = []
        
        ## update sliders
        self.frameIdxSlider.setMaximum(0)
        self.frameIdxSpinBox.setRange(0, 0)
        
        self.frameInfo.setText("Info text")
        
        self.frameIdxSpinBox.setValue(0)
            
    def keyPressEvent(self, e) :
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            pressedIdx = np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9))
            if pressedIdx >= 0 and pressedIdx < self.numSemantics :
                print "show semantics", pressedIdx
                self.currentSemantics = pressedIdx
                sys.stdout.flush()
            
    
#     def eventFilter(self, obj, event) :
#         if obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseMove :
#             self.mouseMoved(event)
#             return True
#         elif obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseButtonPress :
#             self.mousePressed(event)
#             return True
#         elif obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseButtonRelease :
#             self.mouseReleased(event)
#             return True
#         elif (obj == self.frameIdxSpinBox or obj == self.frameIdxSlider) and event.type() == QtCore.QEvent.Type.KeyPress :
#             self.keyPressEvent(event)
#             return True
#         return QtGui.QWidget.eventFilter(self, obj, event)
    
    def playSequenceButtonPressed(self) :
        if self.doPlaySequence :
            self.doPlaySequence = False
            self.playSequenceButton.setIcon(self.playIcon)
            self.playTimer.stop()
            
            self.frameInfo.setText(self.oldInfoText)
        else :
            self.lastRenderTime = time.time()
            self.doPlaySequence = True
            self.playSequenceButton.setIcon(self.pauseIcon)
            self.playTimer.start()
            
            self.oldInfoText = self.frameInfo.text()
            
    def setRenderFps(self, value) :
        self.playTimer.setInterval(1000/value)
        
    def setBackgroundColor(self) :
        newBgColor = QtGui.QColorDialog.getColor(QtCore.Qt.black, self, "Choose Background Color")
        self.frameLabel.setBackgroundColor(newBgColor)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frameLabel.installEventFilter(self)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 60)
        self.renderFpsSpinBox.setSingleStep(1)
        self.renderFpsSpinBox.setValue(30)
        
        self.playSequenceButton = QtGui.QToolButton()
        self.playSequenceButton.setToolTip("Play Generated Sequence")
        self.playSequenceButton.setCheckable(False)
        self.playSequenceButton.setShortcut(QtGui.QKeySequence("Alt+P"))
        self.playSequenceButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        self.playSequenceButton.setIcon(self.playIcon)
        
        self.setBackgroundColorButton = QtGui.QPushButton("&Background Color")
        
        
        ## SIGNALS ##
        
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        
        self.playSequenceButton.clicked.connect(self.playSequenceButtonPressed)
        self.setBackgroundColorButton.clicked.connect(self.setBackgroundColor)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        
        renderingControls = QtGui.QGroupBox("Rendering Controls")
        renderingControls.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        renderingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        renderingControlsLayout = QtGui.QVBoxLayout()
        renderingControlsLayout.addWidget(self.playSequenceButton)
        renderingControlsLayout.addWidget(self.renderFpsSpinBox)
        renderingControlsLayout.addWidget(self.setBackgroundColorButton)
        renderingControls.setLayout(renderingControlsLayout)        
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(renderingControls)
        
        frameHLayout = QtGui.QHBoxLayout()
        frameHLayout.addStretch()
        frameHLayout.addWidget(self.frameLabel)
        frameHLayout.addStretch()
        
        frameVLayout = QtGui.QVBoxLayout()
        frameVLayout.addStretch()
        frameVLayout.addLayout(frameHLayout)
        frameVLayout.addStretch()
        frameVLayout.addWidget(self.frameInfo)
        
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameVLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

im = window.spriteListTable.itemDelegateForRow(11).iconImage
qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32)
qim.save("tralala.png")

