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

app = QtGui.QApplication(sys.argv)

DICT_SPRITE_NAME = 'sprite_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SPRITE_IDX = 'sprite_idx' # stores the index in the self.trackedSprites array of the sprite used in the generated sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_MEDIAN_COLOR = 'median_color'

dataPath = "/media/ilisescu/Data1/PhD/data/"
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

# <codecell>

## load 
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
<<<<<<< HEAD:Semantic Looping Sprites (with all methods and hacks for them).py
    if DICT_SPRITE_NAME not in trackedSprites[-1] :
        del trackedSprites[-1]
#     print trackedSprites[-1][DICT_SPRITE_NAME]
=======
    print trackedSprites[-1][DICT_SPRITE_NAME]
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36:Semantic Looping.py

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

def getMRFCosts(semanticLabels, desiredSemantics, startFrame, sequenceLength) :
    """Computes the unary and pairwise costs for a given sprite
    
        \t  semanticLabels   : the semantic labels assigned to the frames in the sprite sequence
        \t  desiredSemantics : the desired label combination
        \t  startFrame       : starting frame for given sprite (used to constrain which frame to start from)
        \t  sequenceLength   : length of sequence to produce (i.e. number of variables to assign a label k \belongs [0, N] where N is number of frames for sprite)
           
        return: unaries  = unary costs for each node in the graph
                pairwise = pairwise costs for each edge in the graph"""
    
    maxCost = 10000000.0
    ## k = num of semantic labels as there should be semantics attached to each frame
    k = len(semanticLabels)
    
    ## unaries are dictated by semantic labels and by startFrame
    
    # start with uniform distribution for likelihood
    likelihood = np.ones((k, sequenceLength))/(k*sequenceLength)
    
#     # set probability of start frame to 1 and renormalize
#     if startFrame >= 0 and startFrame < k :
#         likelihood[startFrame, 0] = 1.0
#         likelihood /= np.sum(likelihood)
    
    # get the costs associated to agreement of the assigned labels to the desired semantics
    # the variance should maybe depend on k so that when there are more frames in a sprite, the variance is higher so that even if I have to follow the timeline for a long time
    # the cost deriveing from the unary cost does not become bigger than the single pairwise cost to break to go straight to the desired semantic label
    # but for now the sprite sequences are not that long and I'm not expecting them to be many orders of magnitude longer 
    # (variance would have to be 5 or 6 orders of magnitude smaller to make breaking the timeline cheaper than following it)
    distVariance = 0.001#0.001
    numSemantics = semanticLabels.shape[-1]
#     semanticsCosts = vectorisedMinusLogMultiNormal(semanticLabels, np.array(desiredSemantics).reshape((1, numSemantics)), np.eye(numSemantics)*distVariance, True)
    semanticsCosts = np.zeros((k, desiredSemantics.shape[0]))
    for i in xrange(desiredSemantics.shape[0]) :
        semanticsCosts[:, i] = vectorisedMinusLogMultiNormal(semanticLabels, desiredSemantics[i, :].reshape((1, numSemantics)), np.eye(numSemantics)*distVariance, True)
    
    if desiredSemantics.shape[0] < sequenceLength :
        semanticsCosts = semanticsCosts.reshape((k, 1)).repeat(sequenceLength, axis=-1)
    
    # set unaries to minus log of the likelihood + minus log of the semantic labels' distance to the 
    unaries = -np.log(likelihood) + semanticsCosts#.reshape((k, 1)).repeat(sequenceLength, axis=-1)
#     unaries = semanticsCosts.reshape((k, 1)).repeat(sequenceLength, axis=-1)
    
# #     # set cost of start frame to 0 NOTE: not sure if I should use this or the above with the renormalization
#     if startFrame >= 0 and startFrame < k :
#         unaries[startFrame, 0] = 0.0
    if startFrame >= 0 and startFrame < k :
        unaries[:, 0] = maxCost
        unaries[startFrame, 0] = 0.0
    
    ## pairwise are dictated by time constraint and looping ability (i.e. jump probability)
    
    # first dimension is k_n, second represents k_n-1 and last dimension represents all the edges going from graph column w_n-1 to w_n
    pairwise = np.zeros([k, k, sequenceLength-1])
    
    # to enforce timeline give low cost to edge between w_n-1(k = i) and w_n(k = i+1) which can be achieved using
    # an identity matrix with diagonal shifted down by one because only edges from column i-1 and k = j to column i and k=j+1 are viable
    timeConstraint = np.eye(k, k=-1)
    # also allow the sprite to keep looping on label 0 (i.e. show only sprite frame 0 which is the empty frame) so make edge from w_n-1(k=0) to w_n(k=0) viable
    timeConstraint[0, 0] = 1.0
    # also allow the sprite to keep looping from the last frame if necessary so allow to go 
    # from last column (i.e. edge starts from w_n-1(k=last frame)) to second row because first row represents empty frame (i.e. edge goes to w_n(k=1))
    timeConstraint[1, k-1] = 1.0
    # also allow the sprite to go back to the first frame (i.e. empty frame) so allow a low cost edge 
    # from last column (i.e. edge starts from w_n-1(k=last frame)) to first row (i.e. edge goes to w_n(k=0))
    timeConstraint[0, k-1] = 1.0
    
    ## NOTE: don't do all the normal distribution wanking for now: just put very high cost to non viable edges but I'll need something more clever when I try to actually loop a video texture
    ## I would also have to set the time constraint edges' costs to something different from 0 to allow for quicker paths (but more expensive individually) to be chosen when
    ## the semantic label changes
#     timeConstraint /= np.sum(timeConstraint) ## if I normalize here then I need to set mean of gaussian below to what the new max is
#     timeConstraint = vectorisedMinusLogMultiNormal(timeConstraint.reshape((k*k, 1)), np.array([np.max(timeConstraint)]).reshape((1, 1)), np.array([distVariance]).reshape((1, 1)), True)
    timeConstraint = (1.0 - timeConstraint)*maxCost
    
    pairwise = timeConstraint
    
    return unaries.T, pairwise.T

# <codecell>

def smoothstep(delay) :
    # Scale, and clamp x to 0..1 range
    edge0 = 0.0
    edge1 = 1.0
    x = np.arange(0.0, 1.0, 1.0/(delay+1))
    x = np.clip((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    return (x*x*x*(x*(x*6 - 15) + 10))[1:]

def toggleLabelsSmoothly(labels, delay) :
    newLabels = np.roll(labels, 1)
    steps = smoothstep(delay)
    result = np.zeros((delay, labels.shape[-1]))
    ## where diff is less than zero, label prob went from 0 to 1
    result[:, np.argwhere(labels-newLabels < 0)[0, 1]] = steps
    ## where diff is greater than zero, label prob went from 1 to 0
    result[:, np.argwhere(labels-newLabels > 0)[0, 1]] = 1.0 - steps
    return result

print toggleLabelsSmoothly(np.array([[0.0, 1.0]]), 5)

# <codecell>

def synchedSequence2FullOverlap(spriteSequences, spritesTotalLength) :
    ## given synched sequences and corresponding sprites sequence lengths, return the full overlapping sequences assuming I'm following 
    ## the sprites' timeline so all this will become a mess as soon as I start looping
    ## or maybe not really as long as the length of the sequence I'm generating is long enough or actually if I'm looping, I would
    ## probably have to opportunity to jump around in the sprite's timeline so maybe there's no problem if the sequence is short
    if spriteSequences.shape[0] < 1 :
        raise Exception("Empty spriteSequences")
        
    if len(np.argwhere(np.any(spriteSequences < 0, axis=0))) == spriteSequences.shape[-1] :
        return None
#         raise Exception("Invalid spriteSequences")
        
    remainingFrames = spritesTotalLength-spriteSequences[:, -1]-1
#     print remainingFrames
        
    fullSequences = np.hstack((spriteSequences, np.zeros((spriteSequences.shape[0], np.max(remainingFrames)), dtype=int)))
    
    for i in xrange(spriteSequences.shape[0]) :
        fullSequences[i, spriteSequences.shape[-1]:] = np.arange(spriteSequences[i, -1]+1, spriteSequences[i, -1]+1+np.max(remainingFrames), dtype=int)
        
    ## get rid of pairs where the frame index is larger than the sprite length
    fullSequences = fullSequences[:, np.ndarray.flatten(np.argwhere(np.all(fullSequences < np.array(spritesTotalLength).reshape(2, 1), axis=0)))]
    
    ## get rid of pairs where the frame index is negative (due to the fact that I'm showing the 0th frame i.e. invisible sprite)
    fullSequences = fullSequences[:, np.ndarray.flatten(np.argwhere(np.all(fullSequences >= 0, axis=0)))]
    
    return fullSequences

# print synchedSequence2FullOverlap(np.vstack((minCostTraversalExistingSprite.reshape((1, len(minCostTraversalExistingSprite)))-1,
#                                              minCostTraversal.reshape((1, len(minCostTraversal)))-1)), spriteTotalLength)

# <codecell>

def aabb2obbDist(aabb, obb, verbose = False) :
    if verbose :
        figure(); plot(aabb[:, 0], aabb[:, 1])
        plot(obb[:, 0], obb[:, 1])
    minDist = 100000000.0
    colors = ['r', 'g', 'b', 'y']
    for i, j in zip(arange(4), np.mod(arange(1, 5), 4)) :
        m = (obb[j, 1] - obb[i, 1]) / (obb[j, 0] - obb[i, 0])
        b = obb[i, 1] - (m * obb[i, 0]);
        ## project aabb points onto obb segment
        projPoints = np.dot(np.hstack((aabb, np.ones((len(aabb), 1)))), np.array([[1, m, -m*b], [m, m**2, b]]).T)/(m**2+1)
        if np.all(np.negative(np.isnan(projPoints))) :
            ## find distances
            dists = aabb2pointsDist(aabb, projPoints)#np.linalg.norm(projPoints-aabb, axis=-1)
            ## find closest point
            closestPoint = np.argmin(dists)
            ## if rs is between 0 and 1 the point is on the segment
            rs = np.sum((obb[j, :]-obb[i, :])*(aabb-obb[i, :]), axis=1)/(np.linalg.norm(obb[j, :]-obb[i, :])**2)
            if verbose :
                print projPoints
                scatter(projPoints[:, 0], projPoints[:, 1], c=colors[i])
                print dists
                print closestPoint
                print rs
            ## if closestPoint is on the segment
            if rs[closestPoint] > 0.0 and rs[closestPoint] < 1.0 :
#                 print "in", aabb2pointDist(aabb, projPoints[closestPoint, :])
                minDist = np.min((minDist, aabb2pointDist(aabb, projPoints[closestPoint, :])))
            else :
#                 print "out", aabb2pointDist(aabb, obb[i, :]), aabb2pointDist(aabb, obb[j, :])
                minDist = np.min((minDist, aabb2pointDist(aabb, obb[i, :]), aabb2pointDist(aabb, obb[j, :])))

    return minDist


def aabb2pointDist(aabb, point) :
    dx = np.max((np.min(aabb[:, 0]) - point[0], 0, point[0] - np.max(aabb[:, 0])))
    dy = np.max((np.min(aabb[:, 1]) - point[1], 0, point[1] - np.max(aabb[:, 1])))
    return np.sqrt(dx**2 + dy**2);

def aabb2pointsDist(aabb, points) :
    dx = np.max(np.vstack((np.min(aabb[:, 0]) - points[:, 0], np.zeros(len(points)), points[:, 0] - np.max(aabb[:, 0]))), axis=0)
    dy = np.max(np.vstack((np.min(aabb[:, 1]) - points[:, 1], np.zeros(len(points)), points[:, 1] - np.max(aabb[:, 1]))), axis=0)
    return np.sqrt(dx**2 + dy**2);


def getShiftedSpriteTrackDist(firstSprite, secondSprite, shift) :
    
    spriteTotalLength = np.zeros(2, dtype=int)
    spriteTotalLength[0] = len(firstSprite[DICT_BBOX_CENTERS])
    spriteTotalLength[1] = len(secondSprite[DICT_BBOX_CENTERS])
    
    ## find the overlapping sprite subsequences
    ## length of overlap is the minimum between length of the second sequence and length of the first sequence - the advantage it has n the second sequence
    overlapLength = np.min((spriteTotalLength[0]-shift, spriteTotalLength[1]))
    
    frameRanges = np.zeros((2, overlapLength), dtype=int)
    frameRanges[0, :] = np.arange(shift, overlapLength + shift)
    frameRanges[1, :] = np.arange(overlapLength)
    
    totalDistance, distances = getOverlappingSpriteTracksDistance(firstSprite, secondSprite, frameRanges)
    
    return totalDistance, distances, frameRanges


def getOverlappingSpriteTracksDistance(firstSprite, secondSprite, frameRanges, doEarlyOut = True, verbose = False) :
#     ## for now the distance is only given by the distance between bbox center but can add later other things like bbox overlapping region
#     bboxCenters0 = np.array([firstSprite[DICT_BBOX_CENTERS][x] for x in np.sort(firstSprite[DICT_BBOX_CENTERS].keys())[frameRanges[0, :]]])
#     bboxCenters1 = np.array([secondSprite[DICT_BBOX_CENTERS][x] for x in np.sort(secondSprite[DICT_BBOX_CENTERS].keys())[frameRanges[1, :]]])
    
#     centerDistance = np.linalg.norm(bboxCenters0-bboxCenters1, axis=1)
    
#     totDist = np.min(centerDistance)
#     allDists = centerDistance
    
    firstSpriteKeys = np.sort(firstSprite[DICT_BBOX_CENTERS].keys())
    secondSpriteKeys = np.sort(secondSprite[DICT_BBOX_CENTERS].keys())
    allDists = np.zeros(frameRanges.shape[-1])
    for i in xrange(frameRanges.shape[-1]) :            
        allDists[i] = getSpritesBBoxDist(firstSprite[DICT_BBOX_ROTATIONS][firstSpriteKeys[frameRanges[0, i]]],
                                          firstSprite[DICT_BBOXES][firstSpriteKeys[frameRanges[0, i]]], 
                                          secondSprite[DICT_BBOXES][secondSpriteKeys[frameRanges[1, i]]])
        
        if verbose and np.mod(i, frameRanges.shape[-1]/100) == 0 :
            sys.stdout.write('\r' + "Computed image pair " + np.string_(i) + " of " + np.string_(frameRanges.shape[-1]))
            sys.stdout.flush()
        
        ## early out since you can't get lower than 0
        if doEarlyOut and allDists[i] == 0.0 :
            break
            
    totDist = np.min(allDists)
#     return np.sum(centerDistance)/len(centerDistance), centerDistance    
    return totDist, allDists

def getSpritesBBoxDist(theta, bbox1, bbox2) :
    rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    bbox1 = np.dot(rotMat, bbox1.T).T
    bbox2 = np.dot(rotMat, bbox2.T).T
    ## if the bboxes coincide then the distance is set to 0
    if np.all(np.abs(bbox1 - bbox2) <= 10**-10) :
        return 0.0
    else :
        return aabb2obbDist(bbox1, bbox2)

# <codecell>

# totalDistance, distances, frameRanges = getShiftedSpriteTrackDist(trackedSprites[spriteIndices[whosFirst]], 
#                                                                       trackedSprites[spriteIndices[whosFirst-1]], 553)
# print frameRanges
# print np.argwhere(np.isnan(distances))
# i=278
# firstSprite = trackedSprites[spriteIndices[whosFirst]]
# secondSprite = trackedSprites[spriteIndices[whosFirst-1]]
# firstSpriteKeys = np.sort(firstSprite[DICT_BBOX_CENTERS].keys())
# secondSpriteKeys = np.sort(secondSprite[DICT_BBOX_CENTERS].keys())

# theta = firstSprite[DICT_BBOX_ROTATIONS][firstSpriteKeys[frameRanges[0, i]]]
# rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# bbox1 = np.dot(rotMat, firstSprite[DICT_BBOXES][firstSpriteKeys[frameRanges[0, i]]].T).T
# bbox2 = np.dot(rotMat, secondSprite[DICT_BBOXES][secondSpriteKeys[frameRanges[1, i]]].T).T

# print aabb2obbDist(bbox1, bbox2)

# <codecell>

def solveMRF(unaries, pairwise) :
    ## build graph
    numLabels = unaries.shape[1]
    chainLength = unaries.shape[0]
    gm = opengm.gm(numpy.ones(chainLength,dtype=opengm.label_type)*numLabels)
    
    # add unary functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, arange(0, chainLength, 1))
    
    ## add pairwise function
    fid = gm.addFunction(pairwise)
    pairIndices = np.hstack((np.arange(chainLength-1, dtype=int).reshape((chainLength-1, 1)), 
                             np.arange(1, chainLength, dtype=int).reshape((chainLength-1, 1))))
    # add second order factors
    gm.addFactors(fid, pairIndices)
    
    dynProg = opengm.inference.DynamicProgramming(gm)
    tic = time.time()
    dynProg.infer()
    print "bla", time.time() - tic
    
    labels = np.array(dynProg.arg(), dtype=int)
    print gm
    
    return labels, gm.evaluate(labels)

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
        minCostTraversal[i] = minCostPaths[ minCostTraversal[i+1], i+1]
        
    return minCostTraversal, np.min(minCosts[:, -1])

# <codecell>

### THIS CAN BE USED WHEN SPRITE INDICES HAVE CHANGED AND WANT TO UPDATE THE PRECOMPUTED DISTANCES DICT PROPERLY ####
# tmp = glob.glob(dataPath + dataSet + "sprite*.npy")[0]
# oldVsNew = []
# for sprite, i in zip(glob.glob(dataPath + dataSet + "sprite*.npy"), arange(len(glob.glob(dataPath + dataSet + "sprite*.npy")))) :
#     for t in xrange(len(trackedSprites)) :
#         if trackedSprites[t][DICT_SPRITE_NAME] == np.load(sprite).item()[DICT_SPRITE_NAME] :
#             print trackedSprites[t][DICT_SPRITE_NAME], i, t
#             oldVsNew.append([i, t])
#             break
            
# oldVsNew = np.array(oldVsNew)
# oldVsNew = oldVsNew[np.ndarray.flatten(np.argsort(oldVsNew[:, 1])), :]
# print oldVsNew

# newPrecomputedDistances = {}
# for i in xrange(len(trackedSprites)) :
#     for j in xrange(i, len(trackedSprites)) :
#         ## for each i and j of the new sprite idxs get the corresponding old idxs
#         print i, j, "-", oldVsNew[i, 0], oldVsNew[j, 0], 
#         ## if the indices I'm asking for are not sorted I need to invert the shifts, that is negative shifts are now positive and viceversa
#         ## because sprite indices were sorted when computing the old distances
#         if oldVsNew[i, 0] > oldVsNew[j, 0] :
#             print "reverse"
#             tmp = precomputedDistances[np.string_(oldVsNew[j, 0])+np.string_(oldVsNew[i, 0])]
#             newPrecomputedDistances[np.string_(i)+np.string_(j)] = {-x:tmp[x] for x in tmp.keys()}
#         else :
#             print "don't reverse"
#             newPrecomputedDistances[np.string_(i)+np.string_(j)] = precomputedDistances[np.string_(oldVsNew[i, 0])+np.string_(oldVsNew[j, 0])]
            
# np.save(dataPath + dataSet + "precomputedDistances.npy", newPrecomputedDistances)        

# <codecell>

## precompute all distances for all possible sprite pairings
# precomputedDistances = {}
# for i in arange(len(trackedSprites)) :
#     for j in xrange(i, len(trackedSprites)) :
#         possibleShifts = np.arange(-len(trackedSprites[i][DICT_BBOX_CENTERS])+1, 
#                            len(trackedSprites[j][DICT_BBOX_CENTERS]), dtype=int)
        
#         allDistances = {} #np.zeros(len(possibleShifts))
#         for shift in possibleShifts :
#             if shift < 0 :
#                 totalDistance, distances, frameRanges = getShiftedSpriteTrackDist(trackedSprites[i], 
#                                                                                   trackedSprites[j], -shift)
#             else :
#                 totalDistance, distances, frameRanges = getShiftedSpriteTrackDist(trackedSprites[j], 
#                                                                                   trackedSprites[i], shift)
                
#             allDistances[shift] = totalDistance
            
#             sys.stdout.write('\r' + "Done " + np.string_(shift-possibleShifts[0]) + " shifts of " + np.string_(len(possibleShifts)))
#             sys.stdout.flush()
            
#         print
#         print "done with pair", i, j
#         precomputedDistances[np.string_(i)+np.string_(j)] = allDistances

# np.save(dataPath + dataSet + "precomputedDistances.npy", precomputedDistances)

precomputedDistances = np.load(dataPath + dataSet + "precomputedDistances.npy").item()

# <codecell>

# figure(); plot([precomputedDistances['03'][x] for x in np.sort(precomputedDistances['03'].keys())])
# totalDistance, distances, frameRanges = getShiftedSpriteTrackDist(trackedSprites[3], trackedSprites[0], 1020-400)
# figure(); plot(distances)
# print getOverlappingSpriteTracksDistance(trackedSprites[0], trackedSprites[3], np.array([[400], [1020]]))

# <codecell>

## load all sprite patches
# preloadedSpritePatches = []
# currentSpriteImages = []
# del preloadedSpritePatches
# preloadedSpritePatches = []
# for sprite in trackedSprites :
<<<<<<< HEAD:Semantic Looping Sprites (with all methods and hacks for them).py
# #     maskDir = dataPath + dataSet + sprite[DICT_SPRITE_NAME] + "-masked-blended"
#     maskDir = dataPath + dataSet + sprite[DICT_SPRITE_NAME] + "-maskedFlow-blended"
=======
#     maskDir = dataPath + dataSet + sprite[DICT_SPRITE_NAME] + "-masked-blended"
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36:Semantic Looping.py
#     del currentSpriteImages
#     currentSpriteImages = []
#     for frameKey in np.sort(sprite[DICT_FRAMES_LOCATIONS].keys()) :
#         frameName = sprite[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
        
#         if os.path.isdir(maskDir) and os.path.exists(maskDir+"/"+frameName) :
#             im = np.array(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED), dtype=np.uint8)
            
#             visiblePixels = np.argwhere(im[:, :, -1] != 0)
#             topLeft = np.min(visiblePixels, axis=0)
#             patchSize = np.max(visiblePixels, axis=0) - topLeft + 1
            
#             currentSpriteImages.append({'top_left_pos':topLeft, 'sprite_colors':im[visiblePixels[:, 0], visiblePixels[:, 1], :], 
#                                         'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize})
# #             currentSpriteImages.append(im[topLeft[0]:topLeft[0]+patchSize[0]+1, topLeft[1]:topLeft[1]+patchSize[1]+1])
#         else :
# #             im = np.ascontiguousarray(Image.open(sprite[DICT_FRAMES_LOCATIONS][frameIdx]), dtype=np.uint8)
#             currentSpriteImages.append(None)
        
#         sys.stdout.write('\r' + "Loaded image " + np.string_(len(currentSpriteImages)) + " (" + np.string_(len(sprite[DICT_FRAMES_LOCATIONS])) + ")")
#         sys.stdout.flush()
#     preloadedSpritePatches.append(np.copy(currentSpriteImages))
#     print
#     print "done with sprite", sprite[DICT_SPRITE_NAME]

# np.save(dataPath + dataSet + "preloadedSpritePatches.npy", preloadedSpritePatches)

preloadedSpritePatches = list(np.load(dataPath + dataSet + "preloadedSpritePatches.npy"))

# <codecell>

## go through the generated sequence and check that SPRITE_IDX matches the index in tracked sprites
## load sprites 
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())

print len(trackedSprites)
for sequenceName in np.sort(glob.glob(dataPath+dataSet+"generatedSequence-debug*")) :
    print sequenceName
    generatedSequence = list(np.load(sequenceName))
    print len(generatedSequence)
    for s in xrange(len(generatedSequence)) :
        print generatedSequence[s][DICT_SPRITE_NAME], generatedSequence[s][DICT_SPRITE_IDX],
        for i in xrange(len(trackedSprites)) :
            if trackedSprites[i][DICT_SPRITE_NAME] == generatedSequence[s][DICT_SPRITE_NAME] :
    #             print trackedSprites[i][DICT_SPRITE_NAME], i
                print i
                generatedSequence[s][DICT_SPRITE_IDX] = i
    
        np.save(sequenceName, generatedSequence)

# <codecell>

############################################ TEST TEST TEST TEST ############################################ 
spriteIdx = -1
for i in xrange(len(trackedSprites)) :
    if trackedSprites[i][DICT_SPRITE_NAME] == 'red_car1' :
        spriteIdx = i
        break
print "using sprite", trackedSprites[spriteIdx][DICT_SPRITE_NAME]

## semantics for reshuffling are binary i.e. each frame has label 1 (i.e. sprite visible) and extra frame at beginning has label 0
semanticLabels = np.zeros((len(trackedSprites[spriteIdx][DICT_BBOX_CENTERS])+1, 2))
## label 0 means sprite not visible (i.e. only show the first empty frame)
semanticLabels[0, 0] = 1.0
semanticLabels[1:, 1] = 1.0
delay = 8
desiredLabel = np.array([1.0, 0.0]).reshape((1, 2))#.repeat(300-delay/2, axis=0)
desiredLabel = np.concatenate((desiredLabel, toggleLabelsSmoothly(np.array([[1.0, 0.0]]), delay)))
desiredLabel = np.concatenate((desiredLabel, np.array([0.0, 1.0]).reshape((1, 2)).repeat(600-delay, axis=0)))
# semanticDist = np.sum(np.power(semanticLabels-desiredLabel, 2), axis=-1)
desiredLabel = window.burstSemanticsToggle(np.array([1.0, 0.0]), 300, 2, 20)
# desiredLabel = np.array([1.0, 0.0]).reshape((1, 2)).repeat(300, axis=0)

tic = time.time()
unaries, pairwise = getMRFCosts(semanticLabels, desiredLabel, 0, 300)
print "computed costs in", time.time() - tic; sys.stdout.flush()
# gwv.showCustomGraph(unaries[1:, :], title="unaries")
tic = time.time()
minCostTraversal, minCost = solveMRF(unaries, pairwise)
print "solved in", time.time() - tic; sys.stdout.flush()
print minCostTraversal

# <codecell>

## start with generating two sequences one for an instance of red_car1 and one for white_bus1 such that they would be colliding
spriteIndices = np.array([0, 3])
sequenceStartFrames = np.array([0, 580])
sequenceLength = 450
generatedSequences = np.zeros((len(spriteIndices), sequenceLength), dtype=int)
allUnaries = []
allPairwise = []
allNodesConnectedToLabel = []

## now generate initial conflicting sequences to be resolved later

for idx in xrange(len(spriteIndices)) :
    spriteTotalLength = len(trackedSprites[spriteIndices[idx]][DICT_BBOX_CENTERS])

    spriteSemanticLabels = np.zeros((spriteTotalLength+1, 2))
    ## label 0 means sprite not visible (i.e. only show the first empty frame)
    spriteSemanticLabels[0, 0] = 1.0
    spriteSemanticLabels[1:, 1] = 1.0

    tic = time.time()
    if idx == 0 : 
        spriteDesiredSemantics = np.array([1.0, 0.0]).reshape((1, 2))
        spriteDesiredSemantics = np.concatenate((spriteDesiredSemantics, toggleLabelsSmoothly(np.array([1.0, 0.0]).reshape((1, 2)), 8)))
        spriteDesiredSemantics = np.concatenate((spriteDesiredSemantics, np.roll(np.array([1.0, 0.0]).reshape((1, 2)), 1).repeat(sequenceLength-8-1, axis=0)))
        
        unaries, pairwise = getMRFCosts(spriteSemanticLabels, spriteDesiredSemantics, sequenceStartFrames[idx], sequenceLength)
    else :
        unaries, pairwise = getMRFCosts(spriteSemanticLabels, np.array([1.0, 0.0]).reshape((1, 2)).repeat(sequenceLength, axis=0), sequenceStartFrames[idx], sequenceLength)
    allUnaries.append(unaries)
    allPairwise.append(pairwise)
    
    jumpCosts = spriteDistMats[idx][1:, :-1]**2
    viableJumps = np.argwhere(jumpCosts < 0.2)
    viableJumps = viableJumps[np.ndarray.flatten(np.argwhere(jumpCosts[viableJumps[:, 0], viableJumps[:, 1]] > 0.1))]
    ## add 1 to indices because of the 0th frame, and then 5, 4 from the filtering and 1 from going from distances to costs
    allPairwise[idx][viableJumps[:, 0]+1+1+4, viableJumps[:, 1]+1+1+4] = jumpCosts[viableJumps[:, 0], viableJumps[:, 1]]
    
    ## now try and do the optimization completely vectorized
    ## number of edges connected to each label node of variable n (pairwise stores node at arrow tail as cols and at arrow head as rows)
    maxEdgesPerLabel = np.max(np.sum(np.array(pairwise.T != np.max(pairwise.T), dtype=int), axis=-1))
    ## initialize this to index of connected label node with highest edge cost (which is then used as padding)
    ## it contains for each label node of variable n (indexed by rows), all the label nodes of variable n-1 it is connected to by non infinite cost edge (indexed by cols)
    nodesConnectedToLabel = np.argmax(pairwise.T, axis=-1).reshape((len(pairwise.T), 1)).repeat(maxEdgesPerLabel, axis=-1)
    
    sparseIndices = np.where(pairwise != np.max(pairwise))
    # print sparseIndices
    tailIndices = sparseIndices[0]
    headIndices = sparseIndices[1]
    
    ## this contains which label of variable n-1 is connected to which label of variable n
    indicesInLabelSpace = [list(tailIndices[np.where(headIndices == i)[0]]) for i in np.unique(headIndices)]
    
    for headLabel, tailLabels in zip(arange(0, len(nodesConnectedToLabel)), indicesInLabelSpace) :
        nodesConnectedToLabel[headLabel, 0:len(tailLabels)] = tailLabels    
        
    allNodesConnectedToLabel.append(nodesConnectedToLabel)
    
    # gwv.showCustomGraph(unaries)
    print "computed costs for sprite", spriteIndices[idx], "in", time.time() - tic; sys.stdout.flush()
    tic = time.time()
    # minCostTraversal, minCost = solveMRF(unaries, pairwise)
#     minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)
    minCostTraversal, minCost = solveSparseDynProgMRF(allUnaries[idx].T, allPairwise[idx].T, allNodesConnectedToLabel[idx])
    
    print "solved traversal for sprite", spriteIndices[idx] , "in", time.time() - tic; sys.stdout.flush()
    generatedSequences[idx, :] = minCostTraversal

print generatedSequences

count = 0
while True :
    print "iteration", count, 
    ## get distance between every pairing of frames in the generatedSequences
    areCompatible = np.zeros(generatedSequences.shape[-1], dtype=bool)
    areCompatible[np.any(generatedSequences < 1, axis=0)] = True
    
    compatibDist, allCompatibDists = getOverlappingSpriteTracksDistance(trackedSprites[spriteIndices[0]], trackedSprites[spriteIndices[1]], generatedSequences)
    print "incompatibilities to solve", len(np.argwhere(allCompatibDists <= 1.0)); sys.stdout.flush()
#     print allCompatibDists
    
    areCompatible[allCompatibDists > 1.0] = True
    
#     if not np.all(areCompatible) :
#         for idx in arange(len(spriteIndices))[0:1] :
            
#             allUnaries[idx][np.negative(areCompatible), generatedSequences[idx, np.negative(areCompatible)]] += 1000.0 #10000000.0
#             ## if I fix spriteIndices[1] then I can find out all combinations of frames between spriteIndices[0] and the generated sequence for spriteIndices[1]
# #             gwv.showCustomGraph(allUnaries[idx])
            
#             tic = time.time()
#             minCostTraversal, minCost = solveSparseDynProgMRF(allUnaries[idx].T, allPairwise[idx].T, allNodesConnectedToLabel[idx])
            
#             print "solved traversal for sprite", spriteIndices[idx] , "in", time.time() - tic; sys.stdout.flush()
# #             print minCostTraversal, minCost
            
#             generatedSequences[idx, :] = minCostTraversal
#     else :
#         break
    
    count += 1
    if count > 0 :
        break
print generatedSequences

# <codecell>

spriteTotLength = len(trackedSprites[0][DICT_BBOX_CENTERS])

spriteSequence = generatedSequences[1, :][generatedSequences[1, :]-1 >= 0]#.reshape((1, sequenceLength)).repeat(spriteTotLength)-1
# bob = np.vstack((np.ndarray.flatten(arange(spriteTotLength).reshape((1, spriteTotLength)).repeat(sequenceLength, axis=0)), 
#                  generatedSequences[1, :].reshape((1, sequenceLength)).repeat(spriteTotLength)-1))
bob = np.vstack((np.ndarray.flatten(arange(spriteTotLength).reshape((1, spriteTotLength)).repeat(len(spriteSequence), axis=0)), 
                 spriteSequence.reshape((1, len(spriteSequence))).repeat(spriteTotLength)-1))

# dist, allDists = getOverlappingSpriteTracksDistance(trackedSprites[spriteIndices[0]], trackedSprites[spriteIndices[1]], bob, False, True)
## just get the dists from spritesCompatibility
allDists = spritesCompatibility[bob[0, :], bob[1, :]].reshape((len(spriteSequence), spriteTotLength))

# <codecell>

gwv.showCustomGraph(allDists.reshape((len(spriteSequence), spriteTotLength)))
# print allDists.reshape((sequenceLength, spriteTotLength))

# <codecell>

## if I fix spriteIndices[1] then I can find out all combinations of frames between spriteIndices[0] and the generated sequence for spriteIndices[1]
idx = 0
modifiedUnaries = np.copy(allUnaries[idx])
incompatiblePairs = np.argwhere(allDists.reshape((len(spriteSequence), spriteTotLength)) <= 1.0)
# incompatiblePairs = incompatiblePairs[np.ndarray.flatten(np.argwhere(allDists.reshape((sequenceLength, spriteTotLength))[incompatiblePairs[:, 0], incompatiblePairs[:, 1]] >= 0.1)), :]
modifiedUnaries[incompatiblePairs[:, 0], incompatiblePairs[:, 1]+1] = 1e7
gwv.showCustomGraph(modifiedUnaries)

tic = time.time()
minCostTraversal, minCost = solveSparseDynProgMRF(modifiedUnaries.T, allPairwise[idx].T, allNodesConnectedToLabel[idx])
print "solved traversal for sprite", spriteIndices[idx] , "in", time.time() - tic; sys.stdout.flush()
print minCostTraversal, minCost

# <codecell>

# gwv.showCustomGraph(spriteDistMats[1])
tmp = np.copy(spritesCompatibility)
tmp[sequenceStartFrames[0]-1, 5:-5+1] += spriteDistMats[1][1:, :-1][sequenceStartFrames[1]-1, :]
gwv.showCustomGraph(tmp)

# <codecell>

print trackedSprites[spriteIndices[idx]][DICT_SPRITE_NAME]

# <codecell>

debugSequence = []
for idx in xrange(len(spriteIndices)) :
    debugSequence.append({
                          DICT_SPRITE_NAME:trackedSprites[spriteIndices[idx]][DICT_SPRITE_NAME],
                          DICT_SPRITE_IDX:spriteIndices[idx],
                          DICT_SEQUENCE_FRAMES:generatedSequences[idx, :],
                          DICT_DESIRED_SEMANTICS:np.array([1.0, 0.0]).reshape((1, 2)).repeat(sequenceLength, axis=0)
                          })
    if idx == 0 :
        debugSequence[idx][DICT_SEQUENCE_FRAMES] = minCostTraversal#np.ones(sequenceLength)*200
np.save(dataPath+dataSet+"generatedSequence-debug.npy", debugSequence)

# <codecell>

incompatibleDistances = np.copy(spritesCompatibility)
incompatiblePairs = np.argwhere(incompatibleDistances <= 1.0)
incompatibleDistances[incompatiblePairs[:, 0], incompatiblePairs[:, 1]] = 1e7
gwv.showCustomGraph(incompatibleDistances)

# <codecell>

## compute L2 distance between bbox centers for given sprites
spriteDistMats = []
for idx in xrange(len(spriteIndices)) :
    bboxCenters = np.array([trackedSprites[spriteIndices[idx]][DICT_BBOX_CENTERS][x] for x in np.sort(trackedSprites[spriteIndices[idx]][DICT_BBOX_CENTERS].keys())])
    l2DistMat = np.zeros((len(bboxCenters), len(bboxCenters)))
    for c in xrange(len(bboxCenters)) :
        l2DistMat[c, c:] = np.linalg.norm(bboxCenters[c].reshape((1, 2)).repeat(len(bboxCenters)-c, axis=0) - bboxCenters[c:], axis=1)
        l2DistMat[c:, c] = l2DistMat[c, c:]
            
    spriteDistMats.append(vtu.filterDistanceMatrix(l2DistMat, 4, False))

# <codecell>

## compute compatibility distance for every frame pairing between sprites
## compute L2 distance between bbox centers for given sprites
spritesCompatibility = np.zeros((len(trackedSprites[spriteIndices[0]][DICT_BBOX_CENTERS]), len(trackedSprites[spriteIndices[1]][DICT_BBOX_CENTERS])))

for frame in np.arange(len(trackedSprites[spriteIndices[0]][DICT_BBOX_CENTERS])) :
    spriteTotLength = len(trackedSprites[spriteIndices[1]][DICT_BBOX_CENTERS])
    frameRanges = np.vstack((np.ones(spriteTotLength, dtype=int)*frame, np.arange(spriteTotLength, dtype=int).reshape((1, spriteTotLength))))
    compatibilityDist, allCompatibilityDists = getOverlappingSpriteTracksDistance(trackedSprites[spriteIndices[0]], trackedSprites[spriteIndices[1]], frameRanges, False)
    spritesCompatibility[frame, :] = allCompatibilityDists
    
    sys.stdout.write('\r' + "Computed frame " + np.string_(frame) + " of " + np.string_(len(trackedSprites[spriteIndices[0]][DICT_BBOX_CENTERS])))
    sys.stdout.flush()

# <codecell>

figure(); plot(np.min(spritesCompatibility, axis=1))

# <codecell>

gwv.showCustomGraph(spritesCompatibility)

# <codecell>

tmp = np.clip(spriteDistMats[0], 0.1, 0.3)#*(spriteDistMats[0] <= 1.0)
gwv.showCustomGraph(tmp)
# probs, cumProbs = vtu.getProbabilities(spriteDistMats[0][1:, 0:-1], 0.01, None, True)
probs, cumProbs = vtu.getProbabilities(tmp[1:, 0:-1], 0.001, None, False)
gwv.showCustomGraph(probs)
gwv.showCustomGraph(cumProbs)

# <codecell>

spriteIndex = 0
gwv.showCustomGraph(allPairwise[spriteIndex])
gwv.showCustomGraph(spriteDistMats[spriteIndex])

pairwiseWithDist = np.ones_like(allPairwise[spriteIndex])* 5
jumpCosts = spriteDistMats[spriteIndex][1:, :-1]**2
viableJumps = np.argwhere(jumpCosts < 0.2)
viableJumps = viableJumps[np.ndarray.flatten(np.argwhere(jumpCosts[viableJumps[:, 0], viableJumps[:, 1]] > 0.1))]
## add 1 to indices because of the 0th frame, and then 5, 4 from the filtering and 1 from going from distances to costs
pairwiseWithDist[viableJumps[:, 0]+1, viableJumps[:, 1]+1] = jumpCosts[viableJumps[:, 0], viableJumps[:, 1]]
gwv.showCustomGraph(pairwiseWithDist)

# <codecell>

print viableJumps.shape

# <codecell>

# np.all((jumpCosts < 2.0, jumpCosts > 0.1), axis=0)
print np.argwhere(jumpCosts[viableJumps[:, 0], viableJumps[:, 1]] > 0.5)

# <codecell>

minCostTraversal, minCost = solveSparseDynProgMRF(allUnaries[spriteIndex].T, pairwiseWithDist.T, allNodesConnectedToLabel[spriteIndex])
print minCostTraversal

# <codecell>

gwv.showCustomGraph(tmp.reshape(spriteDistMats[spriteIndex].shape))

# <codecell>

np.argwhere(tmp.reshape(spriteDistMats[spriteIndex].shape) < 0)

# <headingcell level=2>

# From here on there's stuff using the hardcoded compatibility measure and assumes the sprite loops till the end of its sequence without jumping around in the timeline

# <codecell>

## load tracked sprites
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
## load generated sequence
generatedSequence = list(np.load(dataPath + dataSet + "generatedSequence-2015-05-02_19:41:52.npy"))

# <codecell>

## try to learn from a labelled incompatibility and use it to not repeat the error
## I know there's an incompatibility between two instances of red_car1 at frame 431 of the generated sequence loaded above
labelledSequenceFrame = 413
## here find which sprites were labelled eventually when I have a scribble but for now I just need to find the 2 sprites that I
## saw were incompatible
for seq in generatedSequence :
    print seq[DICT_SPRITE_NAME], seq[DICT_SEQUENCE_FRAMES][labelledSequenceFrame]
    ## here eventually check if scribble touches the current sprite by for instance checking whether scribble intersects the
    ## sprite's bbox at frame seq[DICT_SEQUENCE_FRAMES][labelledSequenceFrame]

## for now I know the two red_car1 sprites that collide at frame 413 are number 2 and 6 in the generatedSequence
incompatibleSpriteTracks = np.array([2, 6])
spriteIndices = [] ## index of given sprite in trackedSprites
spriteSemanticLabels = []
print
print "incompatible sprites", incompatibleSpriteTracks, ":",
for i in xrange(len(incompatibleSpriteTracks)) :
    print generatedSequence[incompatibleSpriteTracks[i]][DICT_SPRITE_NAME] + "(", 
    print np.string_(int(generatedSequence[incompatibleSpriteTracks[i]][DICT_SEQUENCE_FRAMES][labelledSequenceFrame])) + ")",
    
    spriteIndices.append(generatedSequence[incompatibleSpriteTracks[i]][DICT_SPRITE_IDX])
    
    ## compute semantic labels for each sprite which for the reshuffling case are binary 
    ## i.e. each frame has label 1 (i.e. sprite visible) and extra frame at beginning has label 0
    semanticLabels = np.zeros((len(trackedSprites[spriteIndices[i]][DICT_BBOX_CENTERS])+1, 2))
    ## label 0 means sprite not visible (i.e. only show the first empty frame)
    semanticLabels[0, 0] = 1.0
    semanticLabels[1:, 1] = 1.0
    
    spriteSemanticLabels.append(semanticLabels)
print


## given that I know what sprites are incompatible and in what configuration, how do I go about getting a compatibility measure for all configurations?
## and is there a way to get a compatibility measure for all the sprites combinations and all their configurations from it?

## for now just use sprite center distance to characterise a certain configuration and later on use other cues like bbox overlap and RGB L2 distance
## compute bbox center distance between the given configuration of incompatible sprites
if len(incompatibleSpriteTracks) == 2 :
    ## subtracting 1 here because in generatedSequence there is an extra frame for each frame denoting the sprite being invisible
    spriteFrame = np.zeros(2, dtype=int)
    spriteFrame[0] = generatedSequence[incompatibleSpriteTracks[0]][DICT_SEQUENCE_FRAMES][labelledSequenceFrame]-1
    spriteFrame[1] = generatedSequence[incompatibleSpriteTracks[1]][DICT_SEQUENCE_FRAMES][labelledSequenceFrame]-1
    
    shift = np.abs(spriteFrame[0]-spriteFrame[1])
    whosFirst = int(np.argmax(spriteFrame))
    print whosFirst
    
    totalDistance, distances, frameRanges = getShiftedSpriteTrackDist(trackedSprites[spriteIndices[whosFirst]], 
                                                                      trackedSprites[spriteIndices[whosFirst-1]], shift)
    print totalDistance
    
    possibleShifts = np.arange(-len(trackedSprites[spriteIndices[0]][DICT_BBOX_CENTERS])+1, 
                               len(trackedSprites[spriteIndices[1]][DICT_BBOX_CENTERS]), dtype=int)
#     allDistances = np.zeros(len(possibleShifts))
#     for shift, i in zip(possibleShifts, xrange(len(allDistances))) :
#         if shift < 0 :
#             totalDistance, distances, frameRanges = getShiftedSpriteTrackDist(trackedSprites[spriteIndices[0]], 
#                                                                               trackedSprites[spriteIndices[1]], -shift)
#         else :
#             totalDistance, distances, frameRanges = getShiftedSpriteTrackDist(trackedSprites[spriteIndices[1]], 
#                                                                               trackedSprites[spriteIndices[0]], shift)
            
#         allDistances[i] = totalDistance
        
#         sys.stdout.write('\r' + "Done " + np.string_(i) + " shifts of " + np.string_(len(allDistances)))
#         sys.stdout.flush()

# <codecell>

sequenceLength = 101
toggleDelay = 8
## simulating situation where I'm inserting sprite 6 into the loaded generatedSequence at startFrame
## this makes sure that I get the labelled incompatible situation (i.e. sprite 2 is at frame 72 and sprite 6 at 111) if I don't do anything about compatibility
startFrame = 413#-111-toggleDelay/2

## semantics for sprite 6
spriteDesiredSemantics = np.array([1.0, 0.0]).reshape((1, 2))
spriteDesiredSemantics = np.concatenate((spriteDesiredSemantics, toggleLabelsSmoothly(np.array([1.0, 0.0]).reshape((1, 2)), toggleDelay)))
spriteDesiredSemantics = np.concatenate((spriteDesiredSemantics, np.roll(np.array([1.0, 0.0]).reshape((1, 2)), 1).repeat(sequenceLength-toggleDelay-1, axis=0)))

## also need to simulate a situation where the semantics for a sprite are toggled before a certain sprite has been toggled but is already part of the 
## sequence which is really the situation I'm in in the loaded generated sequence (i.e. I added sprite 2 at time t and then added sprite 6 at a later
## time but before sprite 2 in the sequence time line so, chronologically, sprite 6 is added later but is incompatible with a sprite the will be shown
## later in the sequence) but I guess when I add a new sprite, I need to check if it's compatible with all the sprites in the sequence although I'm not sure
## how that would work


conflictingSpriteTotalLength = len(trackedSprites[spriteIndices[1]][DICT_BBOX_CENTERS])
print conflictingSpriteTotalLength

conflictingSpriteSemanticLabels = np.zeros((conflictingSpriteTotalLength+1, 2))
## label 0 means sprite not visible (i.e. only show the first empty frame)
conflictingSpriteSemanticLabels[0, 0] = 1.0
conflictingSpriteSemanticLabels[1:, 1] = 1.0

## now optimize the sprite I'm changing the semantic label of (i.e. I want to show now)
## optimize without caring about compatibility
tic = time.time()
unaries, pairwise = getMRFCosts(conflictingSpriteSemanticLabels, spriteDesiredSemantics, 0, sequenceLength)

## now try and do the optimization completely vectorized
## number of edges connected to each label node of variable n (pairwise stores node at arrow tail as cols and at arrow head as rows)
maxEdgesPerLabel = np.max(np.sum(np.array(pairwise.T != np.max(pairwise.T), dtype=int), axis=-1))
## initialize this to index of connected label node with highest edge cost (which is then used as padding)
## it contains for each label node of variable n (indexed by rows), all the label nodes of variable n-1 it is connected to by non infinite cost edge (indexed by cols)
nodesConnectedToLabel = np.argmax(pairwise.T, axis=-1).reshape((len(pairwise.T), 1)).repeat(maxEdgesPerLabel, axis=-1)

sparseIndices = np.where(pairwise != np.max(pairwise))
# print sparseIndices
tailIndices = sparseIndices[0]
headIndices = sparseIndices[1]

## this contains which label of variable n-1 is connected to which label of variable n
indicesInLabelSpace = [list(tailIndices[np.where(headIndices == i)[0]]) for i in np.unique(headIndices)]

for headLabel, tailLabels in zip(arange(0, len(nodesConnectedToLabel)), indicesInLabelSpace) :
    nodesConnectedToLabel[headLabel, 0:len(tailLabels)] = tailLabels    

# gwv.showCustomGraph(unaries)
print "computed costs for sprite", incompatibleSpriteTracks[1], "in", time.time() - tic; sys.stdout.flush()
tic = time.time()
# minCostTraversal, minCost = solveMRF(unaries, pairwise)
minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)

print "solved traversal for sprite", incompatibleSpriteTracks[1] , "in", time.time() - tic; sys.stdout.flush()
print minCostTraversal, minCost

count = 0
unariesToUpdate = np.zeros_like(unaries, dtype=np.bool)
while True :
    ## check whether the sprite is compatible with existing sprites in the generated sequence
    tic = time.time()
    for i in xrange(len(generatedSequence)) :
        if i != incompatibleSpriteTracks[1] and i == 7 :
            overlappingSequence = np.array(generatedSequence[i][DICT_SEQUENCE_FRAMES][startFrame:startFrame+sequenceLength], dtype=int)
#             print overlappingSequence
            
            spriteTotalLength = len(trackedSprites[generatedSequence[i][DICT_SPRITE_IDX]][DICT_BBOX_CENTERS])
            spriteSemanticLabels = np.zeros((spriteTotalLength+1, 2))
            ## label 0 means sprite not visible (i.e. only show the first empty frame)
            spriteSemanticLabels[0, 0] = 1.0
            spriteSemanticLabels[1:, 1] = 1.0
            
#             print spriteTotalLength
#             print overlappingSequence
#             print minCostTraversal
            isCompatible = np.zeros(len(minCostTraversal), dtype=np.bool)
            
            ## if the semantic labels are different, the sprites are compatible with each in the reshuffling case but need to figure out how to deal with this
            ## in a general way
#             isCompatible[np.all(spriteSemanticLabels[overlappingSequence] != conflictingSpriteSemanticLabels[minCostTraversal], axis = 1)] = True
            ### HACK ??? ### if one of the frame is 0 it means the two sprites are compatible
            isCompatible[np.any(np.array(np.vstack((overlappingSequence.reshape((1, len(overlappingSequence))),
                                                    minCostTraversal.reshape((1, len(minCostTraversal))))), dtype=int) == 0, axis = 0)] = True
#             print isCompatible
            frameRanges = synchedSequence2FullOverlap(np.array(np.vstack((overlappingSequence.reshape((1, len(overlappingSequence)))-1,
                                                                          minCostTraversal.reshape((1, len(minCostTraversal)))-1)), dtype=int), 
                                                      np.array((spriteTotalLength, conflictingSpriteTotalLength)))
#             print frameRanges
            
            if frameRanges != None :
#                 totalDistance, distances = getOverlappingSpriteTracksDistance(trackedSprites[generatedSequence[i][DICT_SPRITE_IDX]], trackedSprites[0], frameRanges)
                
                ## references precomputedDistances instead of recomputing
                
                spriteIdxs = np.array([generatedSequence[i][DICT_SPRITE_IDX], generatedSequence[incompatibleSpriteTracks[1]][DICT_SPRITE_IDX]])
                sortIdxs = np.argsort(spriteIdxs)
                pairing = np.string_(spriteIdxs[sortIdxs][0]) + np.string_(spriteIdxs[sortIdxs][1])
                pairingShift = frameRanges[sortIdxs, 0][1]-frameRanges[sortIdxs, 0][0]
                totalDistance = precomputedDistances[pairing][pairingShift]
                
                print totalDistance, precomputedDistances[pairing][pairingShift], pairing, pairingShift, frameRanges[sortIdxs, 0]
                
                ## find all pairs of frame that show the same label as the desired label (i.e. [0.0, 1.0])
                tmp = np.all(spriteSemanticLabels[overlappingSequence] == conflictingSpriteSemanticLabels[minCostTraversal], axis=1)
                if totalDistance > 50.0 : 
                    isCompatible[np.all((np.all(conflictingSpriteSemanticLabels[minCostTraversal] == np.array([0.0, 1.0]), axis=1), tmp), axis=0)] = True
            else :
                print "sprites not overlapping"
            
#             print isCompatible
    print "la", time.time() - tic
    count += 1
    if np.any(np.negative(isCompatible)) :
        ## when I do the check for all the sprites in the sequence I would have to take an AND over all the isCompatible arrays but now I know there's only 1 sprite
        
        ## keep track of unaries to change
        unariesToUpdate[np.arange(len(minCostTraversal), dtype=int)[np.negative(isCompatible)], minCostTraversal[np.negative(isCompatible)]] = True
        ## change the unaries to increase the cost for the frames where the isCompatible is False
#         unaries[np.arange(len(minCostTraversal), dtype=int)[np.negative(isCompatible)], minCostTraversal[np.negative(isCompatible)]] += 1000.0
        unaries[np.argwhere(unariesToUpdate)[:, 0], np.argwhere(unariesToUpdate)[:, 1]] += 1000.0
    #     gwv.showCustomGraph(unaries)
        tic = time.time()
#         minCostTraversal, minCost = solveMRF(unaries, pairwise)
        minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)
        if True or np.mod(count, 10) == 0 :
            print "iterarion", count, ": solved traversal for sprite", incompatibleSpriteTracks[1] , "in", time.time() - tic; sys.stdout.flush()
#             print minCostTraversal, minCost
        
        if count == 200 :
            break
    else :
        print "done biatch"
        print minCostTraversal
        break

# <codecell>

tmp = 2
for cost, path, i in zip(minCosts[:, tmp], minCostPaths[:, tmp], xrange(1, minCosts.shape[0])) :
    if np.mod(i-1, 5) == 0 :
        print "{0:03d} - {1:03d}\t".format(i-1, i+3), 
    print cost, "{0:03d}".format(int(path)), "\t",
    if np.mod(i, 5) == 0 :
        print

# <codecell>

DRAW_FIRST_FRAME = 'first_frame'
DRAW_LAST_FRAME = 'last_frame'
DRAW_COLOR = 'color'

class SemanticsSlider(QtGui.QSlider) :
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None) :
        super(SemanticsSlider, self).__init__(orientation, parent)
        style = "QSlider::handle:horizontal { background: #cccccc; width: 25px; border-radius: 0px; } "
        style += "QSlider::groove:horizontal { background: #dddddd; } "
        self.setStyleSheet(style)
        
        self.semanticsToDraw = []
        self.numOfFrames = 1
        self.selectedSemantics = 0
        
    def setSelectedSemantics(self, selectedSemantics) :
        self.selectedSemantics = selectedSemantics
        
    def setSemanticsToDraw(self, semanticsToDraw, numOfFrames) :
        self.semanticsToDraw = semanticsToDraw
        self.numOfFrames = float(numOfFrames)
        
        desiredHeight = len(self.semanticsToDraw)*7
        self.setFixedHeight(desiredHeight)
        
        self.resize(self.width(), self.height())
        self.update()
        
    def paintEvent(self, event) :
        super(SemanticsSlider, self).paintEvent(event)
        
        painter = QtGui.QPainter(self)
        
        ## draw semantics
        
        yCoord = 0.0
        for i in xrange(len(self.semanticsToDraw)) :
            col = self.semanticsToDraw[i][DRAW_COLOR]

            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255)))
            startX =  self.semanticsToDraw[i][DRAW_FIRST_FRAME]/self.numOfFrames*self.width()
            endX =  self.semanticsToDraw[i][DRAW_LAST_FRAME]/self.numOfFrames*self.width()

            if self.selectedSemantics == i :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 127), 1, 
                                              QtCore.Qt.DashLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+0.5, endX-startX, 5)

            else :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 63), 1, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+0.5, endX-startX, 5)


            yCoord += 7


        ## draw slider

        ## the slider is 2 pixels wide so remove 1.0 from X coord
        sliderXCoord = np.max((self.sliderPosition()/self.numOfFrames*self.width()-1.0, 0.0))
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 0), 0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0, 128)))
        painter.drawRect(sliderXCoord, 0, 2, self.height())

# <codecell>

class ListDelegate(QtGui.QItemDelegate):
    
    def __init__(self, parent=None) :
        super(ListDelegate, self).__init__(parent)
        
        self.setBackgroundColor(QtGui.QColor.fromRgb(245, 245, 245))
        self.setIconImage(QtGui.QImage(QtCore.QSize(0, 0), QtGui.QImage.Format_ARGB32))

    def setBackgroundColor(self, bgColor) :
        self.bgColor = bgColor
    
    def setIconImage(self, iconImage) :
        self.iconImage = np.ascontiguousarray(np.copy(iconImage))
    
    def drawDisplay(self, painter, option, rect, text):
        painter.save()
        
        colorRect = QtCore.QRect(rect.left()+rect.height(), rect.top(), rect.width()-rect.height(), rect.height())
        selectionRect = rect
        iconRect = QtCore.QRect(rect.left(), rect.top(), rect.height(), rect.height())

        # draw colorRect
        painter.setBrush(QtGui.QBrush(self.bgColor))
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(128, 128, 128, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
        painter.drawRect(colorRect)
        
        ## draw iconRect
        qim = QtGui.QImage(self.iconImage.data, self.iconImage.shape[1], self.iconImage.shape[0], 
                                                       self.iconImage.strides[0], QtGui.QImage.Format_ARGB32)
        painter.drawImage(iconRect, qim.scaled(iconRect.size()))
        
        ## draw selectionRect
        if option.state & QtGui.QStyle.State_Selected:
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0, 0)))
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 64, 64, 255), 5, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
            painter.drawRect(selectionRect)
        
        # set text color
        painter.setPen(QtGui.QPen(QtCore.Qt.black))
        if option.state & QtGui.QStyle.State_Selected:
            painter.setFont(QtGui.QFont("Helvetica [Cronyx]", 11, QtGui.QFont.Bold))
        else :
            painter.setFont(QtGui.QFont("Helvetica [Cronyx]", 11))
            
        painter.drawText(colorRect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignCenter, text)

        painter.restore()

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.setMouseTracking(True)
        
        self.image = None
        self.overlay = None
        
    def setImage(self, image) : 
        self.image = image.copy()
        self.update()

    def setOverlay(self, overlay) :
        self.overlay = overlay.copy()
        self.update()
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        if self.image != None :
            painter.drawImage(QtCore.QPoint(0, 0), self.image)
            
        if self.overlay != None :
            painter.drawImage(QtCore.QPoint(0, 0), self.overlay)
        
    def setPixmap(self, pixmap) :
        if pixmap.width() > self.width() :
            super(ImageLabel, self).setPixmap(pixmap.scaledToWidth(self.width()))
        else :
            super(ImageLabel, self).setPixmap(pixmap)
        
    def resizeEvent(self, event) :
        if self.pixmap() != None :
            if self.pixmap().width() > self.width() :
                self.setPixmap(self.pixmap().scaledToWidth(self.width()))
                
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.setWindowTitle("Looping the Unloopable")
        self.resize(1280, 720)
        self.LIST_SECTION_SIZE = 60
        
        self.playIcon = QtGui.QIcon("play.png")
        self.pauseIcon = QtGui.QIcon("pause.png")
        self.doPlaySequence = False
        
        self.createGUI()
        
        self.isScribbling = False
        
        self.prevPoint = None
        
        self.trackedSprites = []
        self.currentSpriteIdx = -1
        
        self.frameIdx = 0
        self.overlayImg = QtGui.QImage(QtCore.QSize(100, 100), QtGui.QImage.Format_ARGB32)
        
        ## get background image
        im = np.ascontiguousarray(Image.open(dataPath + dataSet + "median.png"))
        self.bgImage = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setFixedSize(self.bgImage.width(), self.bgImage.height())
        self.frameLabel.setImage(self.bgImage)
        
        self.loadTrackedSprites()
        
        self.generatedSequence = []
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.renderOneFrame)
        self.lastRenderTime = time.time()
        self.oldInfoText = ""
        
        self.EXTEND_LENGTH = 100 + 1 ## since I get rid of the frist frame from the generated sequence because it's forced to be the one already showing
        self.TOGGLE_DELAY = 8
        self.BURST_ENTER_DELAY = 2
        self.BURST_EXIT_DELAY = 20
        
        self.DO_EXTEND = 0
        self.DO_TOGGLE = 1
        self.DO_BURST = 2
        
        if len(glob.glob(dataPath+dataSet+"generatedSequence-*")) > 0 :
            ## load latest sequence
            self.generatedSequence = list(np.load(np.sort(glob.glob(dataPath+dataSet+"generatedSequence-2*"))[-1]))
#             print "loaded", np.sort(glob.glob(dataPath+dataSet+"generatedSequence-*"))[-1]
#             self.generatedSequence = list(np.load(dataPath+dataSet+"generatedSequence-2015-07-07_23:35:48.npy"))
#             self.generatedSequence = generatedSequence
#             self.generatedSequence = list(np.load(dataPath+dataSet+"generatedSequence-debug.npy"))
            if len(self.generatedSequence) > 0 :
                ## update sliders
                self.frameIdxSlider.setMaximum(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                self.frameIdxSpinBox.setRange(0, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                
                self.frameInfo.setText("Generated sequence length: " + np.string_(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])))
                
                self.showFrame(self.frameIdx)
                
            self.setSemanticsToDraw()
        
        self.frameIdxSlider.setSelectedSemantics(-1)
        
        self.setFocus()
        
    def renderOneFrame(self) :
        idx = self.frameIdx + 1
        if idx >= 0 and len(self.generatedSequence) > 0 : #idx < len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) :
            self.frameIdxSpinBox.setValue(np.mod(idx, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])))
            
        self.frameInfo.setText("Rendering at " + np.string_(int(1.0/(time.time() - self.lastRenderTime))) + " FPS")
        self.lastRenderTime = time.time()
    
    def showFrame(self, idx) :
        if idx >= 0 and len(self.generatedSequence) > 0 and idx < len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) :
            self.frameIdx = idx
            
            if self.bgImage != None and self.overlayImg.size() != self.bgImage.size() :
                self.overlayImg = self.overlayImg.scaled(self.bgImage.size())
            ## empty image
            self.overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
            
            ## go through all the sprites used in the sequence
            for s in xrange(len(self.generatedSequence)) :
                ## index in self.trackedSprites of current sprite in self.generatedSequence
                spriteIdx = int(self.generatedSequence[s][DICT_SPRITE_IDX])
                ## index of current sprite frame to visualize
                sequenceFrameIdx = int(self.generatedSequence[s][DICT_SEQUENCE_FRAMES][self.frameIdx]-1)
                ## -1 stands for not shown or eventually for more complicated sprites as the base frame to keep showing when sprite is frozen
                ## really in the sequence I have 0 for the static frame but above I do -1
                if sequenceFrameIdx >= 0 :
                    ## the trackedSprites data is indexed (i.e. the keys) by the frame indices in the original full sequence and keys are not sorted
                    frameToShowIdx = np.sort(self.trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())[sequenceFrameIdx]
                    try :
                        if spriteIdx >= 0 and spriteIdx < len(preloadedSpritePatches) and sequenceFrameIdx < len(preloadedSpritePatches[spriteIdx]) :
                            ## the QImage for this frame has been preloaded
                            self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, self.drawSpritesBox.isChecked(), 
                                             self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked(), preloadedSpritePatches[spriteIdx][sequenceFrameIdx])
                        else :
                            self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, self.drawSpritesBox.isChecked(), 
                                             self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked())
                    except :
                        self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, self.drawSpritesBox.isChecked(), 
                                         self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked())
            
            self.frameLabel.setFixedSize(self.overlayImg.width(), self.overlayImg.height())
            self.frameLabel.setOverlay(self.overlayImg)
            
    def drawOverlay(self, sprite, frameIdx, doDrawSprite, doDrawBBox, doDrawCenter, spritePatch = None) :
        if self.overlayImg != None :            
            painter = QtGui.QPainter(self.overlayImg)
            
            ## draw sprite
            if doDrawSprite :
                if spritePatch != None :
                    reconstructedImg = np.ascontiguousarray(np.zeros((spritePatch['patch_size'][0], spritePatch['patch_size'][1], 4)), dtype=np.uint8)
                    reconstructedImg[spritePatch['visible_indices'][:, 0], spritePatch['visible_indices'][:, 1], :] = spritePatch['sprite_colors']
                    reconstructedQImage = QtGui.QImage(reconstructedImg.data, reconstructedImg.shape[1], reconstructedImg.shape[0], 
                                                       reconstructedImg.strides[0], QtGui.QImage.Format_ARGB32)
                    
                    painter.drawImage(QtCore.QRect(spritePatch['top_left_pos'][1], spritePatch['top_left_pos'][0],
                                                   spritePatch['patch_size'][1], spritePatch['patch_size'][0]), reconstructedQImage)
                else :
                    ## maybe save all this data in trackedSprites by modifying it in "Merge Tracked Sprites"
                    frameName = sprite[DICT_FRAMES_LOCATIONS][frameIdx].split(os.sep)[-1]
                    maskDir = dataPath + dataSet + sprite[DICT_SPRITE_NAME] + "-masked"
                    
                    if os.path.isdir(maskDir) and os.path.exists(maskDir+"/"+frameName) :
    #                     mask = np.array(Image.open(maskDir+"/mask-"+frameName))[:, :, 0]
    #                     ## for whatever reason for this to work the image needs to be BGR
    #                     im = np.concatenate((cv2.imread(sprite[DICT_FRAMES_LOCATIONS][frameIdx]), mask.reshape(np.hstack((mask.shape, 1)))), axis=-1)
    #                     im = np.ascontiguousarray(im)
                        im = np.ascontiguousarray(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED))
                        img = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32)
                    else :
                        im = np.ascontiguousarray(Image.open(sprite[DICT_FRAMES_LOCATIONS][frameIdx]))
                        img = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
                    
                    painter.drawImage(QtCore.QPoint(0, 0), img)
            
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 3, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            ## draw bbox
            if doDrawBBox :
                if DICT_FOOTPRINTS in sprite.keys() :
                    bbox = sprite[DICT_FOOTPRINTS][frameIdx]
                else :
                    bbox = sprite[DICT_BBOXES][frameIdx]
                    
                for p1, p2 in zip(np.mod(arange(4), 4), np.mod(arange(1, 5), 4)) :
                    painter.drawLine(QtCore.QPointF(bbox[p1, 0], bbox[p1, 1]), QtCore.QPointF(bbox[p2, 0], bbox[p2, 1]))
            
            ## draw bbox center
            if doDrawCenter :
                painter.drawPoint(QtCore.QPointF(sprite[DICT_BBOX_CENTERS][frameIdx][0], sprite[DICT_BBOX_CENTERS][frameIdx][1]))
            
#     def changeSprite(self, row) :
#         if len(self.trackedSprites) > row :
#             self.currentSpriteIdx = row
            
#         self.setFocus()
        
    def changeSprite(self, row) :
        if len(self.trackedSprites) > row.row() :
            self.currentSpriteIdx = row.row()
        
        print "selected sprite", self.currentSpriteIdx; sys.stdout.flush()
            
        self.setFocus()
            
    def loadTrackedSprites(self) :
        ## going to first frame of first sprite if there were no sprites before loading
        goToNewSprite = len(self.trackedSprites) == 0
        for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
            self.trackedSprites.append(np.load(sprite).item())
        
        self.setSpriteList()
        if len(self.trackedSprites) > 0 and goToNewSprite :
            self.spriteListTable.selectRow(0)
            
#     def setSpriteList(self) :
#         self.spriteListTable.setRowCount(0)
#         if len(self.trackedSprites) > 0 :
#             self.spriteListTable.setRowCount(len(self.trackedSprites))
            
#             for i in xrange(0, len(self.trackedSprites)):
#                 self.spriteListTable.setItem(i, 0, QtGui.QTableWidgetItem(self.trackedSprites[i][DICT_SPRITE_NAME]))
                
#                 if DICT_MEDIAN_COLOR in self.trackedSprites[i].keys() :
#                     col = self.trackedSprites[i][DICT_MEDIAN_COLOR]
# #                     tmp = QtGui.QTableWidgetItem(np.string_(i))
# #                     tmp.setStyleSheet("QTableWidgetItem { background: #cc0000; width: 25px; border-radius: 0px; } ")
# #                     tmp.setBackground(QtGui.QColor.fromRgb(0, 0, 0, 255))#col[0], col[1], col[2], 255))
# #                     self.spriteListTable.setVerticalHeaderItem(i, tmp)
# #                     self.spriteListTable.verticalHeaderItem(i).setBackground(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
#                     self.spriteListTable.item(i, 0).setBackground(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
# #             self.spriteListTable.verticalHeader().setStyleSheet("::section { border: 2px; } ")
#         else :
#             self.spriteListTable.setRowCount(1)
#             self.spriteListTable.setItem(0, 0, QtGui.QTableWidgetItem("No tracked sprites"))
            
    def setSpriteList(self) :
#         self.spriteListTable.setRowCount(0)
        if len(self.trackedSprites) > 0 :
            self.spriteListModel.setRowCount(len(self.trackedSprites))
#             self.spriteListModel.setColumnCount(2)
            self.delegateList = []
    
            bgImg = np.array(Image.open(dataPath+dataSet+"median.png"))
            for i in xrange(0, len(self.trackedSprites)):
                self.delegateList.append(ListDelegate())
                self.spriteListTable.setItemDelegateForRow(i, self.delegateList[-1])
                
#                 self.spriteListModel.setItem(i, 0, QtGui.QStandardItem(np.string_(i)))
                ## set sprite name
                self.spriteListModel.setItem(i, 0, QtGui.QStandardItem(self.trackedSprites[i][DICT_SPRITE_NAME]))
    
                ## set sprite icon
                numFrames = len(preloadedSpritePatches[i])
                framePadding = int(numFrames*0.2)
                bestFrame = framePadding
                for j in xrange(framePadding, numFrames-framePadding) :
                    patchSize = preloadedSpritePatches[i][j]['patch_size']
                    bestPatchSize = preloadedSpritePatches[i][bestFrame]['patch_size']
                    if np.abs(patchSize[0]*1.0/patchSize[1]-1) < np.abs(bestPatchSize[0]*1.0/bestPatchSize[1]-1) and np.prod(patchSize) > 60**2 :
                        bestFrame = j
                        
                squareImg = np.ascontiguousarray(self.getSquareSpriteIconImg(bgImg, preloadedSpritePatches[i][bestFrame]))
                self.spriteListTable.itemDelegateForRow(i).setIconImage(squareImg)
                
                if DICT_MEDIAN_COLOR in self.trackedSprites[i].keys() :
                    col = self.trackedSprites[i][DICT_MEDIAN_COLOR]
#                     self.spriteListTable.item(i, 0).setBackground(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
                    self.spriteListTable.itemDelegateForRow(i).setBackgroundColor(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255))
                    
            self.currentSpriteIdx = 0
                    
        else :
            self.spriteListModel.setRowCount(1)
#             self.spriteListModel.setColumnCount(1)
            
            self.delegateList = [ListDelegate()]
            self.spriteListTable.setItemDelegateForRow(0, self.delegateList[-1])
            self.spriteListModel.setItem(0, 0, QtGui.QStandardItem("No tracked sprites"))
            
    def getSquareSpriteIconImg(self, bgImg, spritePatch) :
        spriteTopLeft = spritePatch['top_left_pos']
        spritePatchSize = spritePatch['patch_size']
        spriteIconImg = bgImg[spriteTopLeft[0] : spriteTopLeft[0] + spritePatchSize[0], spriteTopLeft[1] : spriteTopLeft[1] + spritePatchSize[1], [2, 1, 0]]
        spriteIconImg[spritePatch['visible_indices'][:, 0], spritePatch['visible_indices'][:, 1], :] = spritePatch['sprite_colors'][:, 0:-1]
        sizeDiff = np.abs(spritePatchSize[0] - spritePatchSize[1])
        additiveBeginning = int(np.floor(sizeDiff/2.0))
        if spritePatchSize[0] > spritePatchSize[1] :
            newTopLeft = np.copy(spriteTopLeft)
            newTopLeft[1] -= additiveBeginning
            newPatchSize = np.copy(spritePatchSize)
            newPatchSize[1] += sizeDiff
            squareIconImg = bgImg[newTopLeft[0] : newTopLeft[0] + newPatchSize[0], newTopLeft[1] : newTopLeft[1] + newPatchSize[1], [2, 1, 0]]
            squareIconImg[:, additiveBeginning:additiveBeginning+spritePatchSize[1], :] = spriteIconImg
            finalPatch = np.ones((squareIconImg.shape[0], squareIconImg.shape[1], 4), dtype=np.uint8)*255
            finalPatch[:, :, :-1] = squareIconImg
#             return np.array(squareIconImg, dtype=np.uint8)
        elif spritePatchSize[0] < spritePatchSize[1] :
            newTopLeft = np.copy(spriteTopLeft)
            newTopLeft[0] -= additiveBeginning
            newPatchSize = np.copy(spritePatchSize)
            newPatchSize[0] += sizeDiff
            squareIconImg = bgImg[newTopLeft[0] : newTopLeft[0] + newPatchSize[0], newTopLeft[1] : newTopLeft[1] + newPatchSize[1], [2, 1, 0]]
            squareIconImg[additiveBeginning:additiveBeginning+spritePatchSize[0], :, :] = spriteIconImg
            finalPatch = np.ones((squareIconImg.shape[0], squareIconImg.shape[1], 4), dtype=np.uint8)*255
            finalPatch[:, :, :-1] = squareIconImg
#             return np.array(squareIconImg, dtype=np.uint8)
        else :
            finalPatch = np.ones((spriteIconImg.shape[0], spriteIconImg.shape[1], 4), dtype=np.uint8)*255
            finalPatch[:, :, :-1] = spriteIconImg
        
#         figure(); imshow(finalPatch)
        return cv2.resize(finalPatch, (self.LIST_SECTION_SIZE, self.LIST_SECTION_SIZE), interpolation=cv2.INTER_AREA) #finalPatch
#         return np.array(spriteIconImg, dtype=np.uint8)
            
    def closeEvent(self, event) :
        print "closing"
        sys.stdout.flush()
        ## saving sequence
        if self.autoSaveBox.isChecked() and len(self.generatedSequence) > 0 :
            fileName = "generatedSequence-" + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            np.save(dataPath + dataSet + fileName, self.generatedSequence)
            print "saved new sequence as \"", fileName, "\""
            
    def deleteGeneratedSequence(self) :
        del self.generatedSequence
        self.generatedSequence = []
        
        ## update sliders
        self.frameIdxSlider.setMaximum(0)
        self.frameIdxSpinBox.setRange(0, 0)
        
        self.frameInfo.setText("Info text")
        
        self.frameIdxSpinBox.setValue(0)
    
    def mousePressed(self, event):
#         print event.pos()
#         sys.stdout.flush()
        if event.button() == QtCore.Qt.LeftButton :
            self.isScribbling = True
            print "left button clicked"
        elif event.button() == QtCore.Qt.RightButton :
            print "right button clicked"
        
        sys.stdout.flush()
                
    def mouseMoved(self, event):
        if self.isScribbling :
            print "scribbling"
            
    def mouseReleased(self, event):
        if self.isScribbling :
            self.isScribbling = False
            
    def leaveOneOutExtension(self, leaveOutIdx, extensionLength, startFrame) :
        extendMode = {}
        for i in xrange(len(self.generatedSequence)) :
            if i != leaveOutIdx :
                extendMode[i] = self.DO_EXTEND
        if len(extendMode) > 0 :
            print "extending existing sprites by", extensionLength, "frames and leaving out", leaveOutIdx
            self.extendSequence(self.extendSequenceTracksSemantics(extensionLength, extendMode), 
                                startFrame)
            
    def keyPressEvent(self, e) :
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            pressedIdx = np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9))
            print "pressed key", pressedIdx
            
        elif e.key() == QtCore.Qt.Key_Space :
            if len(self.generatedSequence) > 0 :
                self.frameIdxSpinBox.setValue(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
            extendMode = {}
            for i in xrange(len(self.generatedSequence)) :
                extendMode[i] = self.DO_EXTEND
            if len(self.generatedSequence) > 0 :
                self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), self.frameIdx)
        elif e.key() == QtCore.Qt.Key_T :
#             print self.currentSpriteIdx
            if self.currentSpriteIdx >= 0 and self.currentSpriteIdx < len(self.trackedSprites) :
                print "spawning sprite", self.trackedSprites[self.currentSpriteIdx][DICT_SPRITE_NAME]
                ## spawn new sprite
                self.addNewSpriteTrackToSequence(self.currentSpriteIdx)

                if len(self.generatedSequence) > 0 :
                    ## extend existing sprites if necessary
                    if self.frameIdx > len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH :
#                         print self.frameIdx, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]), self.EXTEND_LENGTH, 1, self.frameIdx-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1
                        self.leaveOneOutExtension(len(self.generatedSequence)-1, 
                                                  self.frameIdx-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1, 
                                                  len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                        
                    ## now toggle new sprite to visible
#                     print "toggling new sprite"
                    extendMode = {}
                    extendMode[len(self.generatedSequence)-1] = self.DO_TOGGLE
                    self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
                                        self.frameIdx, self.resolveCompatibilityBox.isChecked())
                    
                    ## check whether desired label has been reached and if not extend (need to actually do this but for now just check the sprite frame is larger than 0)
                    while self.generatedSequence[-1][DICT_SEQUENCE_FRAMES][-1] < 1 :
#                         print "extending new sprite because semantics not reached"
                        ## extend existing sprites if necessary
                        if len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1 > len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH :
                            self.leaveOneOutExtension(len(self.generatedSequence)-1, 
                                                      len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1, 
                                                      len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                            
                        extendMode[len(self.generatedSequence)-1] = self.DO_EXTEND
                        self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
                                            len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1, 
                                            self.resolveCompatibilityBox.isChecked())
                        
#                     print "toggling back new sprite"
                    ## toggle it back to not visible
                    extendMode[len(self.generatedSequence)-1] = self.DO_TOGGLE
                    self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
                                        len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
                    
                    additionalFrames = len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES]) - len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])
                    if additionalFrames < 0 :
                        ## extend new sprite's sequence to match total sequence's length
#                         print "extending new sprite's sequence"
                        extendMode = {}
                        extendMode[len(self.generatedSequence)-1] = self.DO_EXTEND
                        self.extendSequence(self.extendSequenceTracksSemantics(-additionalFrames+1, extendMode), 
                                            len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
                    elif additionalFrames > 0 :
                        ## extend existing sprites' sequences to match the new total sequence's length because of newly added sprite
#                         print "extending existing sprites' sequences"
                        extendMode = {}
                        for i in xrange(len(self.generatedSequence)-1) :
                            extendMode[i] = self.DO_EXTEND
                        self.extendSequence(self.extendSequenceTracksSemantics(additionalFrames+1, extendMode), 
                                            len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                        
                self.showFrame(self.frameIdx)   
            
        sys.stdout.flush()
        
    def checkIsCompatible(self, spriteIdx, spriteSequence, sequenceStartFrame) :
        isSpriteCompatible = np.ones(len(spriteSequence), dtype=np.bool)
        if spriteIdx >= 0 and spriteIdx < len(self.generatedSequence) :
            spriteTotLength = len(self.trackedSprites[self.generatedSequence[spriteIdx][DICT_SPRITE_IDX]][DICT_BBOX_CENTERS])
            spriteSemanticLabels = np.zeros((spriteTotLength+1, 2))
            ## label 0 means sprite not visible (i.e. only show the first empty frame)
            spriteSemanticLabels[0, 0] = 1.0
            spriteSemanticLabels[1:, 1] = 1.0
            for i in xrange(len(self.generatedSequence)) :
                if i != spriteIdx :
                    overlappingSpriteSequence = np.array(self.generatedSequence[i][DICT_SEQUENCE_FRAMES][sequenceStartFrame:sequenceStartFrame+len(spriteSequence)], dtype=int)
#                     print overlappingSpriteSequence
#                     print spriteSequence
                    
                    overlappingSpriteTotLength = len(self.trackedSprites[self.generatedSequence[i][DICT_SPRITE_IDX]][DICT_BBOX_CENTERS])
                    overlappingSpriteSemanticLabels = np.zeros((overlappingSpriteTotLength+1, 2))
                    ## label 0 means sprite not visible (i.e. only show the first empty frame)
                    overlappingSpriteSemanticLabels[0, 0] = 1.0
                    overlappingSpriteSemanticLabels[1:, 1] = 1.0
                    
                    areSpritesCompatible = np.zeros(len(spriteSequence), dtype=np.bool)
                    
                    ## if the semantic labels are different, the sprites are compatible with each in the reshuffling case but need to figure out how to deal with this
                    ## in a general way
        #             areSpritesCompatible[np.all(spriteSemanticLabels[overlappingSpriteSequence] != conflictingSpriteSemanticLabels[spriteSequence], axis = 1)] = True
                    ### HACK ??? ### if one of the frame is 0 it means the two sprites are compatible
                    areSpritesCompatible[np.any(np.array(np.vstack((overlappingSpriteSequence.reshape((1, len(overlappingSpriteSequence))),
                                                                    spriteSequence.reshape((1, len(spriteSequence))))), dtype=int) == 0, axis = 0)] = True
                    
                    frameRanges = synchedSequence2FullOverlap(np.array(np.vstack((overlappingSpriteSequence.reshape((1, len(overlappingSpriteSequence)))-1,
                                                                                  spriteSequence.reshape((1, len(spriteSequence)))-1)), dtype=int), 
                                                              np.array((overlappingSpriteTotLength, spriteTotLength)))
                    
#                     print np.array(np.vstack((overlappingSpriteSequence.reshape((1, len(overlappingSpriteSequence))),
#                                                                     spriteSequence.reshape((1, len(spriteSequence))))), dtype=int)
                    
#                     print frameRanges
                    
                    if frameRanges != None :
#                         totalDistance, distances = getOverlappingSpriteTracksDistance(self.trackedSprites[self.generatedSequence[i][DICT_SPRITE_IDX]], 
#                                                                                       self.trackedSprites[self.generatedSequence[spriteIdx][DICT_SPRITE_IDX]], frameRanges)
                
                        spriteIdxs = np.array([self.generatedSequence[i][DICT_SPRITE_IDX], self.generatedSequence[spriteIdx][DICT_SPRITE_IDX]])
                        sortIdxs = np.argsort(spriteIdxs)
                        pairing = np.string_(spriteIdxs[sortIdxs][0]) + np.string_(spriteIdxs[sortIdxs][1])
                        pairingShift = frameRanges[sortIdxs, 0][1]-frameRanges[sortIdxs, 0][0]
                        totalDistance = precomputedDistances[pairing][pairingShift]
                
#                         print "lala", totalDistance, precomputedDistances[pairing][pairingShift], spriteIdxs, pairing, pairingShift, spriteIdx, frameRanges[sortIdxs, 0]
                        
                        ## find all pairs of frame that show the same label as the desired label (i.e. [0.0, 1.0])
                        tmp = np.all(overlappingSpriteSemanticLabels[overlappingSpriteSequence] == spriteSemanticLabels[spriteSequence], axis=1)
                        if totalDistance > 5.0 : 
                            areSpritesCompatible[np.all((np.all(spriteSemanticLabels[spriteSequence] == np.array([0.0, 1.0]), axis=1), tmp), axis=0)] = True
#                     else :
#                         print "sprites not overlapping"
#                     print "areSpritesCompatible", areSpritesCompatible
                    
                    isSpriteCompatible = np.all((isSpriteCompatible, areSpritesCompatible), axis=0)
                    
        return isSpriteCompatible
            
    def extendSequence(self, desiredSemantics, startingFrame, resolveCompatibility = False) :
        self.frameInfo.setText("Extending sequence")
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        
        for i in desiredSemantics.keys() :
            if i >= 0 and i < len(self.generatedSequence) :
                spriteIdx = self.generatedSequence[i][DICT_SPRITE_IDX]
                
                ### HACK ### semanticLabels for each sprite are just saying that the first frame of the sprite sequence means sprite not visible and the rest mean it is
                semanticLabels = np.zeros((len(self.trackedSprites[spriteIdx][DICT_BBOX_CENTERS])+1, 2))
                ## label 0 means sprite not visible (i.e. only show the first empty frame)
                semanticLabels[0, 0] = 1.0
                semanticLabels[1:, 1] = 1.0
                
                ## set starting frame
                if len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES]) == 0 :
                    desiredStartFrame = 0
                else :
                    desiredStartFrame = self.generatedSequence[i][DICT_SEQUENCE_FRAMES][startingFrame]
                
                ## do dynamic programming optimization
                unaries, pairwise = getMRFCosts(semanticLabels, desiredSemantics[i], desiredStartFrame, len(desiredSemantics[i]))
#                 print unaries
#                 print pairwise
#                 print desiredSemantics[i]
#                 print desiredStartFrame, len(desiredSemantics[i])
                
                ## now try and do the optimization completely vectorized
                ## number of edges connected to each label node of variable n (pairwise stores node at arrow tail as cols and at arrow head as rows)
                maxEdgesPerLabel = np.max(np.sum(np.array(pairwise.T != np.max(pairwise.T), dtype=int), axis=-1))
                ## initialize this to index of connected label node with highest edge cost (which is then used as padding)
                ## it contains for each label node of variable n (indexed by rows), all the label nodes of variable n-1 it is connected to by non infinite cost edge (indexed by cols)
                nodesConnectedToLabel = np.argmax(pairwise.T, axis=-1).reshape((len(pairwise.T), 1)).repeat(maxEdgesPerLabel, axis=-1)
                
                sparseIndices = np.where(pairwise != np.max(pairwise))
                # print sparseIndices
                tailIndices = sparseIndices[0]
                headIndices = sparseIndices[1]
                
                ## this contains which label of variable n-1 is connected to which label of variable n
                indicesInLabelSpace = [list(tailIndices[np.where(headIndices == headIdx)[0]]) for headIdx in np.unique(headIndices)]
                
                for headLabel, tailLabels in zip(arange(0, len(nodesConnectedToLabel)), indicesInLabelSpace) :
                    nodesConnectedToLabel[headLabel, 0:len(tailLabels)] = tailLabels
                    
#                 print nodesConnectedToLabel
                                
                minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)#solveMRF(unaries, pairwise)
                
                if resolveCompatibility :
#                     print "resolving compatibility", i, startingFrame
                    
                    self.frameInfo.setText("Optimizing sequence - resolving compatibility")
                    QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                    
                    isSpriteCompatible = self.checkIsCompatible(i, minCostTraversal, startingFrame)
#                     print isSpriteCompatible
#                     print len(isSpriteCompatible)
                    count = 0
                    while True :
                        count += 1
                        if np.any(np.negative(isSpriteCompatible)) :
                            ## when I do the check for all the sprites in the sequence I would have to take an AND over all the isSpriteCompatible arrays but now I know there's only 1 sprite
                            
                            ## change the unaries to increase the cost for the frames where the isSpriteCompatible is False
                            unaries[np.arange(len(minCostTraversal), dtype=int)[np.negative(isSpriteCompatible)], minCostTraversal[np.negative(isSpriteCompatible)]] += 1000.0
                        #     gwv.showCustomGraph(unaries)
                            tic = time.time()
                            minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)#solveMRF(unaries, pairwise)
#                             if True or np.mod(count, 20) == 0 :
#                                 print "iteration", count, ": solved traversal for sprite", i , "in", time.time() - tic, 
#                                 print "num of zeros:", len(np.argwhere(minCostTraversal == 0)); sys.stdout.flush()
                            
#                                 self.frameInfo.setText("Optimizing sequence - resolving compatibility " + np.string_(count))
#                                 QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
#                                 print minCostTraversal, minCost

                            isSpriteCompatible = self.checkIsCompatible(i, minCostTraversal, startingFrame)
#                             if count == 200 :
#                                 break
                        else :
                            print "done"
#                             print minCostTraversal
                            break
#                     gwv.showCustomGraph(unaries[1:, :])
                
                ## update dictionary
                # don't take the first frame of the minCostTraversal as it would just repeat the last seen frame
#                 print i
#                 print self.generatedSequence[i][DICT_SEQUENCE_FRAMES][:startingFrame+1]
#                 print minCostTraversal[1:]
                self.generatedSequence[i][DICT_SEQUENCE_FRAMES] = np.hstack((self.generatedSequence[i][DICT_SEQUENCE_FRAMES][:startingFrame+1], minCostTraversal[1:]))
                self.generatedSequence[i][DICT_DESIRED_SEMANTICS] = np.vstack((self.generatedSequence[i][DICT_DESIRED_SEMANTICS][:startingFrame+1],
                                                                               desiredSemantics[i][1:, :]))
                
                
                ## update sliders
                self.frameIdxSlider.setMaximum(len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES])-1)
                self.frameIdxSpinBox.setRange(0, len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES])-1)
                
                self.frameInfo.setText("Generated sequence length: " + np.string_(len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES])))
                QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                    
#                 print "sequence with", len(self.generatedSequence), "sprites, extended by", len(desiredSemantics[i])-1, "frames"

        self.setSemanticsToDraw()
    
    def extendSequenceTracksSemantics(self, length, mode) :
        spriteSemantics = {}
        ## mode contains the way the track is extended (one of DO_EXTEND, DO_TOGGLE, DO_BURST)
        for i in mode.keys() :
            if i >= 0 and i < len(self.generatedSequence) :
                if len(self.generatedSequence[i][DICT_DESIRED_SEMANTICS]) > 0 :
                    currentSemantics = self.generatedSequence[i][DICT_DESIRED_SEMANTICS][-1, :].reshape((1, 2))
                else :
                    ## hardcoded desired "not show" label
                    currentSemantics = np.array([1.0, 0.0]).reshape((1, 2))
                    
                desiredSemantics = np.array([1.0, 0.0]).reshape((1, 2)).repeat(length, axis=0)
                if mode[i] == self.DO_EXTEND :
                    ## extend semantics
                    desiredSemantics = currentSemantics.repeat(length, axis=0)
                elif mode[i] == self.DO_TOGGLE :
                    ## toggle semantics
                    desiredSemantics = self.toggleSequenceTrackSemantics(currentSemantics, length, self.TOGGLE_DELAY)
                elif mode[i] == self.DO_BURST :
                    ## burst toggle semantics from current to toggle and back to current
                    desiredSemantics = self.burstSemanticsToggle(currentSemantics, length, self.BURST_ENTER_DELAY, self.BURST_EXIT_DELAY)
                        
                spriteSemantics[i] = desiredSemantics
                    
        return spriteSemantics
    
    def toggleSequenceTrackSemantics(self, startSemantics, length, toggleDelay) :
        desiredSemantics = startSemantics.reshape((1, 2))
        desiredSemantics = np.concatenate((desiredSemantics, toggleLabelsSmoothly(startSemantics.reshape((1, 2)), toggleDelay)))
        desiredSemantics = np.concatenate((desiredSemantics, np.roll(startSemantics.reshape((1, 2)), 1).repeat(length-toggleDelay-1, axis=0)))
        
        return desiredSemantics
        
    def burstSemanticsToggle(self, startSemantics, length, enterDelay, exitDelay):
        desiredSemantics = startSemantics.reshape((1, 2))
        desiredSemantics = np.concatenate((desiredSemantics, toggleLabelsSmoothly(startSemantics.reshape((1, 2)), enterDelay)))
        desiredSemantics = np.concatenate((desiredSemantics, np.roll(startSemantics.reshape((1, 2)), 1).repeat(length-2*(enterDelay+exitDelay), axis=0)))
        desiredSemantics = np.concatenate((desiredSemantics, toggleLabelsSmoothly(np.roll(startSemantics.reshape((1, 2)), 1).reshape((1, 2)), exitDelay)))
        desiredSemantics = np.concatenate((desiredSemantics, startSemantics.reshape((1, 2)).repeat(enterDelay+exitDelay-1, axis=0)))
        
        return desiredSemantics
    
    def addNewSpriteTrackToSequence(self, spriteIdx) :
        if spriteIdx >= 0 and spriteIdx < len(self.trackedSprites) :
#             print "adding new sprite to sequence"
            self.generatedSequence.append({
                                           DICT_SPRITE_IDX:spriteIdx,
                                           DICT_SPRITE_NAME:self.trackedSprites[spriteIdx][DICT_SPRITE_NAME],
                                           DICT_SEQUENCE_FRAMES:np.empty(0, dtype=int),
                                           DICT_DESIRED_SEMANTICS:np.empty((0, 2), dtype=float)#[],
#                                                DICT_FRAME_SEMANTIC_TOGGLE:[]
                                          })
            if len(self.generatedSequence) > 1 :
                ## just hardcode filling the new sprite's sequence of frames and semantics to the "not shown" label
                maxFrames = len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])
                self.generatedSequence[-1][DICT_DESIRED_SEMANTICS] = np.array([1.0, 0.0]).reshape((1, 2)).repeat(maxFrames, axis=0)
                self.generatedSequence[-1][DICT_SEQUENCE_FRAMES] = np.zeros(maxFrames)
                
        
    def eventFilter(self, obj, event) :
        if obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseMove :
            self.mouseMoved(event)
            return True
        elif obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseButtonPress :
            self.mousePressed(event)
            return True
        elif obj == self.frameLabel and event.type() == QtCore.QEvent.Type.MouseButtonRelease :
            self.mouseReleased(event)
            return True
        elif (obj == self.frameIdxSpinBox or obj == self.frameIdxSlider) and event.type() == QtCore.QEvent.Type.KeyPress :
            self.keyPressEvent(event)
            return True
        return QtGui.QWidget.eventFilter(self, obj, event)
    
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
        
    def setSemanticsToDraw(self) :
        if len(self.generatedSequence) > 0  :
            self.semanticsToDraw = []
            numOfFrames = 1
            for i in xrange(0, len(self.generatedSequence)):
                spriteIdx = self.generatedSequence[i][DICT_SPRITE_IDX]
                if DICT_MEDIAN_COLOR in self.trackedSprites[spriteIdx].keys() :
                    col = self.trackedSprites[spriteIdx][DICT_MEDIAN_COLOR]
                else :
                    col = np.array([0, 0, 0])
                ## if the sprite is ever visible
                if len(np.argwhere(self.generatedSequence[i][DICT_SEQUENCE_FRAMES] != 0)) > 0 :
                    self.semanticsToDraw.append({
                                                    DRAW_COLOR:col,
                                                    DRAW_FIRST_FRAME:np.argwhere(self.generatedSequence[i][DICT_SEQUENCE_FRAMES] != 0)[0, 0], 
                                                    DRAW_LAST_FRAME:np.argwhere(self.generatedSequence[i][DICT_SEQUENCE_FRAMES] != 0)[-1, 0]
                                                })
                else :
                    print "invisible sprite in generated sequence"
                numOfFrames = np.max((numOfFrames, len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES])-0.5))
                
            self.frameIdxSlider.setSemanticsToDraw(self.semanticsToDraw, numOfFrames)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frameLabel.installEventFilter(self)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameIdxSlider = SemanticsSlider(QtCore.Qt.Horizontal)
        self.frameIdxSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameIdxSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.frameIdxSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.frameIdxSlider.setMinimum(0)
        self.frameIdxSlider.setMaximum(0)
        self.frameIdxSlider.setTickInterval(50)
        self.frameIdxSlider.setSingleStep(1)
        self.frameIdxSlider.setPageStep(100)
        self.frameIdxSlider.installEventFilter(self)
    
        self.frameIdxSpinBox = QtGui.QSpinBox()
        self.frameIdxSpinBox.setRange(0, 0)
        self.frameIdxSpinBox.setSingleStep(1)
        self.frameIdxSpinBox.installEventFilter(self)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 60)
        self.renderFpsSpinBox.setSingleStep(1)
        self.renderFpsSpinBox.setValue(30)
        
#         self.spriteListTable = QtGui.QTableWidget(1, 1)
#         self.spriteListTable.horizontalHeader().setStretchLastSection(True)
#         self.spriteListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Tracked sprites"))
#         self.spriteListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Fixed)
#         self.spriteListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
#         self.spriteListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
#         self.spriteListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
#         self.spriteListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
#         self.spriteListTable.setItem(0, 0, QtGui.QTableWidgetItem("No tracked sprites"))
#         self.spriteListTable.setStyleSheet("QTableWidget::item:selected { background-color: rgba(255, 0, 0, 30) }")


        self.spriteListModel = QtGui.QStandardItemModel(1, 1)
        self.spriteListModel.setHorizontalHeaderLabels(["Tracked sprites"])
        self.spriteListModel.setItem(0, 0, QtGui.QStandardItem("No tracked sprites"))
        
        self.spriteListTable = QtGui.QTableView()
        self.spriteListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.spriteListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.spriteListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.spriteListTable.horizontalHeader().setStretchLastSection(True)
        self.spriteListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Fixed)
        self.spriteListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.spriteListTable.verticalHeader().setVisible(False)
        self.spriteListTable.verticalHeader().setDefaultSectionSize(self.LIST_SECTION_SIZE)

        self.delegateList = [ListDelegate()]
        self.spriteListTable.setItemDelegateForRow(0, self.delegateList[-1])
        self.spriteListTable.setModel(self.spriteListModel)
        
        
        
        self.drawSpritesBox = QtGui.QCheckBox("Render Sprites")
        self.drawSpritesBox.setChecked(True)
        self.drawBBoxBox = QtGui.QCheckBox("Render Bounding Box")
        self.drawCenterBox = QtGui.QCheckBox("Render BBox Center")
        
        self.playSequenceButton = QtGui.QToolButton()
        self.playSequenceButton.setToolTip("Play Generated Sequence")
        self.playSequenceButton.setCheckable(False)
        self.playSequenceButton.setShortcut(QtGui.QKeySequence("Alt+P"))
        self.playSequenceButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        self.playSequenceButton.setIcon(self.playIcon)
        
        self.autoSaveBox = QtGui.QCheckBox("Autosave")
#         self.autoSaveBox.setChecked(True)
        
        self.deleteSequenceButton = QtGui.QPushButton("Delete Sequence")
        
        self.resolveCompatibilityBox = QtGui.QCheckBox("Resolve Compatibility")
        self.resolveCompatibilityBox.setChecked(True)
        
        
        ## SIGNALS ##
        
        self.frameIdxSlider.valueChanged[int].connect(self.frameIdxSpinBox.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.frameIdxSlider.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.showFrame)
        
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        
#         self.spriteListTable.currentCellChanged.connect(self.changeSprite)
#         self.spriteListTable.cellPressed.connect(self.changeSprite)
        
        self.spriteListTable.clicked.connect(self.changeSprite)
        
        self.playSequenceButton.clicked.connect(self.playSequenceButtonPressed)
        self.deleteSequenceButton.clicked.connect(self.deleteGeneratedSequence)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        
        renderingControls = QtGui.QGroupBox("Rendering Controls")
        renderingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        renderingControlsLayout = QtGui.QVBoxLayout()
        renderingControlsLayout.addWidget(self.drawSpritesBox)
        renderingControlsLayout.addWidget(self.drawBBoxBox)
        renderingControlsLayout.addWidget(self.drawCenterBox)
        renderingControlsLayout.addWidget(self.playSequenceButton)
        renderingControlsLayout.addWidget(self.renderFpsSpinBox)
        renderingControls.setLayout(renderingControlsLayout)
        
        
        sequenceControls = QtGui.QGroupBox("Sequence Controls")
        sequenceControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        sequenceControlsLayout = QtGui.QVBoxLayout()
        sequenceControlsLayout.addWidget(self.resolveCompatibilityBox)
        sequenceControlsLayout.addWidget(self.deleteSequenceButton)
        sequenceControlsLayout.addWidget(self.autoSaveBox)
        sequenceControls.setLayout(sequenceControlsLayout)
        
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.spriteListTable)
        controlsLayout.addWidget(renderingControls)
        controlsLayout.addWidget(sequenceControls)
        
        sliderLayout = QtGui.QHBoxLayout()
        sliderLayout.addWidget(self.frameIdxSlider)
        sliderLayout.addWidget(self.frameIdxSpinBox)
        
        frameHLayout = QtGui.QHBoxLayout()
        frameHLayout.addStretch()
        frameHLayout.addWidget(self.frameLabel)
        frameHLayout.addStretch()
        
        frameVLayout = QtGui.QVBoxLayout()
        frameVLayout.addStretch()
        frameVLayout.addLayout(frameHLayout)
        frameVLayout.addStretch()
        frameVLayout.addLayout(sliderLayout)
        frameVLayout.addWidget(self.frameInfo)
        
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameVLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

<<<<<<< HEAD:Semantic Looping Sprites (with all methods and hacks for them).py
## try the compositing
sprite1 = np.array(Image.open(dataPath+dataSet+"blue_orange_bus1-maskedFlow-blended/frame-02880.png"))
sprite2 = np.array(Image.open(dataPath+dataSet+"white_bus1-maskedFlow-blended/frame-01047.png"))
bgPlusSprite1 = sprite1[:, :, :-1]*(sprite1[:, :, -1].reshape((720, 1280, 1))/255.0)
bgPlusSprite1 = np.array(bgPlusSprite1 + Image.open(dataPath+dataSet+"median.png")*(1.0-sprite1[:, :, -1].reshape((720, 1280, 1))/255.0), dtype=np.uint8)

bgPlusSprite2 = sprite2[:, :, :-1]*(sprite2[:, :, -1].reshape((720, 1280, 1))/255.0)
bgPlusSprite2 = np.array(bgPlusSprite2 + Image.open(dataPath+dataSet+"median.png")*(1.0-sprite2[:, :, -1].reshape((720, 1280, 1))/255.0), dtype=np.uint8)

diffColor1 = np.sum((Image.open(dataPath+dataSet+"median.png")-bgPlusSprite1)**2, axis=-1)
diffColor2 = np.sum((Image.open(dataPath+dataSet+"median.png")-bgPlusSprite2)**2, axis=-1)
# diffColor1 += (sprite1[:, :, -1] == 0)*np.max(diffColor1)
# diffColor2 += (sprite2[:, :, -1] == 0)*np.max(diffColor2)

print np.max(diffColor1)
print np.max(diffColor2)
thresh = 200
# firstIf = np.argwhere(diffColor2 > thresh)
# secondIf = np.argwhere(diffColor1 > thresh)
# thridIf = np.argwhere(diffColor1 > diffColor2)

# result = np.copy(bgPlusSprite2)
# result[firstIf[:, 0], firstIf[:, 1], :] = bgPlusSprite2[firstIf[:, 0], firstIf[:, 1], :]
# result[secondIf[:, 0], secondIf[:, 1], :] = bgPlusSprite1[secondIf[:, 0], secondIf[:, 1], :]
# result[thridIf[:, 0], thridIf[:, 1], :] = bgPlusSprite1[thridIf[:, 0], thridIf[:, 1], :]

result = np.zeros_like(bgPlusSprite1)
for i in xrange(diffColor1.shape[0]) :
    for j in xrange(diffColor1.shape[1]) :
            
        if diffColor1[i, j] > thresh and sprite1[i, j, -1] != 0 :
            result[i, j, :] = sprite1[i, j, :-1]
            
        if diffColor2[i, j] > thresh and sprite2[i, j, -1] != 0 :
            result[i, j, :] = sprite2[i, j, :-1]
            
#         elif diffColor1[i, j] > diffColor2[i, j] :
#             result[i, j, :] = bgPlusSprite1[i, j, :]
            
#         else :
#             result[i, j, :] = bgPlusSprite2[i, j, :]
            
figure(); imshow(result)
# Image.fromarray(result).save("/home/ilisescu/result.png")

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

## used for enlarging bbox used to decide size of patch around it (percentage)
PATCH_BORDER = 0.4
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

# <codecell>

print trackedSprites[5][DICT_SPRITE_NAME]

# <codecell>

fullSprite1Prior = np.zeros(bgImage.shape[0:2])
fullSprite1Prior[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]] = np.copy(1.0-spritePrior/np.max(spritePrior))
gwv.showCustomGraph(fullSprite1Prior)

# <codecell>

bgImage = np.array(Image.open(dataPath+dataSet+"median.png"))
## Sprite below
# f = 2889
# spriteIdx = 2
# f = 4758
# spriteIdx = 4
# f = 4377
# spriteIdx = 5
# f = 1046      ### composite this with C to show compositing efficiencey against simple thresholding
# spriteIdx = 8
# f = 4731
# spriteIdx = 4
# f = 2933
# spriteIdx = 2
f = 850      ### composite this with A to show compositing efficiencey against simple thresholding
spriteIdx = 8
# f = 1190     ### composite this with B to show compositing efficiencey against simple thresholding
# spriteIdx = 8
sprite1 = np.array(Image.open(dataPath+dataSet+trackedSprites[spriteIdx][DICT_SPRITE_NAME]+"-maskedFlow-blended/frame-{0:05d}.png".format(f+1)))
spritePatch, offset, patchSize, touchedBorders = getSpritePatch(trackedSprites[spriteIdx], f, bgImage.shape[1], bgImage.shape[0])
bgPrior, spritePrior = getPatchPriors(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :], 
                                      spritePatch, offset, patchSize, trackedSprites[spriteIdx], f)
fullSprite1Prior = np.zeros(bgImage.shape[0:2])
fullSprite1Prior[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]] = np.copy(1.0-spritePrior/np.max(spritePrior))
fullSprite1Prior /= np.max(fullSprite1Prior)

## Sprite above
# f = 1046
# f = 996
# spriteIdx = 8
# f = 1182     ### this is C
# spriteIdx = 8
# f = 2933
# spriteIdx = 2
# f = 4373
# spriteIdx = 5
f = 2927     ### this is A
spriteIdx = 2
# f = 1046
# spriteIdx = 8
# f = 4648     ### this is B
# spriteIdx = 1
sprite2 = np.array(Image.open(dataPath+dataSet+trackedSprites[spriteIdx][DICT_SPRITE_NAME]+"-maskedFlow-blended/frame-{0:05d}.png".format(f+1)))
spritePatch, offset, patchSize, touchedBorders = getSpritePatch(trackedSprites[spriteIdx], f, bgImage.shape[1], bgImage.shape[0])
bgPrior, spritePrior = getPatchPriors(bgImage[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1], :], 
                                      spritePatch, offset, patchSize, trackedSprites[spriteIdx], f)
fullSprite2Prior = np.zeros(bgImage.shape[0:2])
fullSprite2Prior[offset[1]:offset[1]+patchSize[0], offset[0]:offset[0]+patchSize[1]] = np.copy(1.0-spritePrior/np.max(spritePrior))
fullSprite2Prior /= np.max(fullSprite2Prior)


#######

bgImage = np.array(Image.open(dataPath+dataSet+"median.png"))

compositedImage = np.copy(sprite1[:, :, :-1])*(sprite1[:, :, -1].reshape((720, 1280, 1))/255.0)
compositedImage += np.copy(bgImage)*(1.0-sprite1[:, :, -1].reshape((720, 1280, 1))/255.0)
compositedImage = (compositedImage*(1.0-sprite2[:, :, -1].reshape((720, 1280, 1))/255.0) + 
                   np.copy(sprite2[:, :, :-1])*(sprite2[:, :, -1].reshape((720, 1280, 1))/255.0))
compositedImage = np.array(compositedImage, dtype=np.uint8)

thresh = 1.2
# thresh = 1.31
ambiguousIdxs = np.argwhere(np.all(((sprite1[:, :, -1] != 0).reshape((sprite1.shape[0], sprite1.shape[1], 1)),
                         (sprite2[:, :, -1] != 0).reshape((sprite2.shape[0], sprite2.shape[1], 1))), axis=0)[:, :, -1])

alpha = 0.0235
kSize = 15
sigma = 11
diffSprite1 = np.zeros(bgImage.shape[0:2])
diffSprite1[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1]] = np.sqrt(np.sum((bgImage[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :]-
                                                                        sprite1[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :-1])**2, axis=-1))
diffSprite1 = cv2.GaussianBlur(diffSprite1, (kSize, kSize), sigma)
# diffSprite1 = cv2.adaptiveBilateralFilter(np.array(diffSprite1, dtype=np.float32), (8, 8), 5)
diffSprite1 = diffSprite1*alpha+(1.0-alpha)*fullSprite1Prior

diffSprite2 = np.zeros(bgImage.shape[0:2])
diffSprite2[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1]] = np.sqrt(np.sum((bgImage[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :]-
                                                                        sprite2[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :-1])**2, axis=-1))
diffSprite2 = cv2.GaussianBlur(diffSprite2, (kSize, kSize), sigma)#*1.2
diffSprite2 = diffSprite2*alpha+(1.0-alpha)*fullSprite2Prior

print np.max(diffSprite1), np.max(diffSprite2)

tmp = np.zeros(bgImage.shape[0:2])
for (i, j) in ambiguousIdxs :
    if diffSprite2[i, j] > thresh :
        compositedImage[i, j, :] = sprite2[i, j, :-1]
        tmp[i, j] = 1
    elif diffSprite1[i, j] > thresh :
        compositedImage[i, j, :] = sprite1[i, j, :-1]
        tmp[i, j] = 2
    elif diffSprite1[i, j] > diffSprite2[i, j] :
        compositedImage[i, j, :] = sprite1[i, j, :-1]
        tmp[i, j] = 2
    else :
        compositedImage[i, j, :] = sprite2[i, j, :-1]
        tmp[i, j] = 1
#     if diffSprite1 > thresh:
#         compositedImage[i, j, :] = sprite1[i, j, :-1]

#     if diffSprite2 > thresh :
#         compositedImage[i, j, :] = sprite2[i, j, :-1]




colorDiffSprite1 = np.sqrt(np.sum((bgImage/255.0 - sprite1[:, :, :-1]/255.0)**2, axis=-1))
colorDiffSprite1 *= sprite1[:, :, -1]
colorDiffSprite1 = cv2.GaussianBlur(colorDiffSprite1, (kSize, kSize), sigma)

thresholdedSprite1Alpha = np.zeros((bgImage.shape[0], bgImage.shape[1], 1))
thresh = 39.0
for (i, j) in np.argwhere(sprite1[:, :, -1] != 0) :
#     if diffSprite1[i, j] > thresh :
#     if np.sqrt(np.sum((bgImage[i, j, :] - sprite1[i, j, :-1])**2, axis=-1)) > thresh :
    if colorDiffSprite1[i, j] > thresh :
        thresholdedSprite1Alpha[i, j] = 1

        
colorDiffSprite2 = np.sqrt(np.sum((bgImage/255.0 - sprite2[:, :, :-1]/255.0)**2, axis=-1))
colorDiffSprite2 *= sprite2[:, :, -1]
colorDiffSprite2 = cv2.GaussianBlur(colorDiffSprite2, (kSize, kSize), sigma)

thresholdedSprite2Alpha = np.zeros((bgImage.shape[0], bgImage.shape[1], 1))
for (i, j) in np.argwhere(sprite2[:, :, -1] != 0) :
#     if diffSprite2[i, j] > thresh :
#     if np.sqrt(np.sum((bgImage[i, j, :] - sprite2[i, j, :-1])**2, axis=-1)) > thresh :
    if colorDiffSprite2[i, j] > thresh :
        thresholdedSprite2Alpha[i, j] = 1
    
figure(); imshow(compositedImage)
# gwv.showCustomGraph(tmp)
# gwv.showCustomGraph(diffSprite1)
# gwv.showCustomGraph(diffSprite2)
# gwv.showCustomGraph(thresholdedSprite1Alpha[:, :, -1])
# gwv.showCustomGraph(thresholdedSprite2Alpha[:, :, -1])
figure(); imshow(np.array(bgImage*(1.0-np.any((thresholdedSprite1Alpha == 1, thresholdedSprite2Alpha == 1), axis=0)) + 
                          sprite1[:, :, :-1]*(np.all((thresholdedSprite1Alpha == 1, thresholdedSprite2Alpha == 0), axis=0)) +
                          sprite2[:, :, :-1]*thresholdedSprite2Alpha, dtype=np.uint8))

# <codecell>

figure()
imshow(np.concatenate((cv2.cvtColor(np.array(Image.open("/media/ilisescu/Data1/PhD/data/wave1/frame-00025.png")), cv2.COLOR_RGB2GRAY).reshape((720, 1280, 1)),
                       cv2.cvtColor(np.array(Image.open("/media/ilisescu/Data1/PhD/data/wave2/frame-00009.png")), cv2.COLOR_RGB2GRAY).reshape((720, 1280, 1)),
                       cv2.cvtColor(np.array(Image.open("/media/ilisescu/Data1/PhD/data/wave3/frame-00028.png")), cv2.COLOR_RGB2GRAY).reshape((720, 1280, 1))), axis=-1))

# <codecell>

print np.concatenate((np.array(Image.open("/media/ilisescu/Data1/PhD/data/wave1/frame-00025.png"))[:, :, 0].reshape((720, 1280, 1)), 
                       np.array(Image.open("/media/ilisescu/Data1/PhD/data/wave2/frame-00009.png"))[:, :, 1].reshape((720, 1280, 1)), 
                       np.array(Image.open("/media/ilisescu/Data1/PhD/data/wave3/frame-00028.png"))[:, :, 2].reshape((720, 1280, 1))), axis=-1).shape

# <codecell>

# tmp = np.zeros(compositedImage.shape[0:2])
# tmp[ambiguousIdxs[:, 0],ambiguousIdxs[:, 1]] = 1
gwv.showCustomGraph(tmp)

# <codecell>

# Image.fromarray(compositedImage).save("/home/ilisescu/PhD/compositedVsThresholded/whiteBusVsWhiteBus/composited.png")
Image.fromarray(np.array(bgImage*(1.0-np.any((thresholdedSprite1Alpha == 1, thresholdedSprite2Alpha == 1), axis=0)) + 
                          sprite1[:, :, :-1]*(np.all((thresholdedSprite1Alpha == 1, thresholdedSprite2Alpha == 0), axis=0)) +
                          sprite2[:, :, :-1]*thresholdedSprite2Alpha, dtype=np.uint8)).save("/home/ilisescu/PhD/compositedVsThresholded/whiteBusVsWhiteBus/thresholded6.png")

# <codecell>

gwv.showCustomGraph(colorDiffSprite2)

# <codecell>

gwv.showCustomGraph(diffSprite2)
gwv.showCustomGraph(fullSprite2Prior)
alpha = 0.05
gwv.showCustomGraph(diffSprite2*alpha+(1.0-alpha)*(fullSprite2Prior/np.max(fullSprite2Prior)))

# <codecell>

# gwv.showCustomGraph(cv2.morphologyEx(tmp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))
diffSprite1 = np.zeros(bgImage.shape[0:2])
diffSprite1[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1]] = np.sqrt(np.sum((bgImage[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :]-
                                                                        sprite1[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :-1])**2, axis=-1))
diffSprite2 = np.zeros(bgImage.shape[0:2])
diffSprite2[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1]] = np.sqrt(np.sum((bgImage[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :]-
                                                                        sprite2[ambiguousIdxs[:, 0], ambiguousIdxs[:, 1], :-1])**2, axis=-1))
gwv.showCustomGraph(cv2.GaussianBlur(diffSprite1, (15, 15), 5))

# <codecell>

tmp = np.zeros_like(bgPlusSprite1)
ambiguousIdxs = np.argwhere(np.all(((sprite1[:, :, -1] != 0).reshape((sprite1.shape[0], sprite1.shape[1], 1)),
                         (sprite2[:, :, -1] != 0).reshape((sprite2.shape[0], sprite2.shape[1], 1))), axis=0)[:, :, -1])
tmp[idxs[:, 0], idxs[:, 1]] = 255
figure(); imshow(tmp)

# <codecell>

np.all(((sprite1[:, :, -1] != 0).reshape((sprite1.shape[0], sprite1.shape[1], 1)),
                         (sprite2[:, :, -1] != 0).reshape((sprite2.shape[0], sprite2.shape[1], 1))), axis=-1).shape

# <codecell>

figure(); imshow(diffColor2)

# <codecell>

=======
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36:Semantic Looping.py
im = window.spriteListTable.itemDelegateForRow(11).iconImage
qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32)
qim.save("tralala.png")

# <codecell>

for i in xrange(5):#len(preloadedSpritePatches)) :
    numFrames = len(preloadedSpritePatches[i])
    framePadding = int(numFrames*0.2)
    bestFrame = framePadding
    for j in xrange(framePadding, numFrames-framePadding) :
        patchSize = preloadedSpritePatches[i][j]['patch_size']
        bestPatchSize = preloadedSpritePatches[i][bestFrame]['patch_size']
        if np.abs(patchSize[0]*1.0/patchSize[1]-1) < np.abs(bestPatchSize[0]*1.0/bestPatchSize[1]-1) and  np.prod(patchSize) > 50**2 :
#             print np.abs(patchSize[0]*1.0/patchSize[1]-1)
            bestFrame = j
    print i, bestFrame, 
    spritePatch = preloadedSpritePatches[i][bestFrame]
    reconstructedImg = np.ascontiguousarray(np.zeros((spritePatch['patch_size'][0], spritePatch['patch_size'][1], 4)), dtype=np.uint8)
    reconstructedImg[spritePatch['visible_indices'][:, 0], spritePatch['visible_indices'][:, 1], :] = spritePatch['sprite_colors']
    print reconstructedImg.shape
#     figure(); imshow(reconstructedImg)

# <codecell>

figure(); imshow(reconstructedImg)

# <codecell>

print window.generatedSequence[0][DICT_SEQUENCE_FRAMES]
print np.argwhere(window.generatedSequence[0][DICT_SEQUENCE_FRAMES] != 0)[0, 0]
print np.argwhere(window.generatedSequence[0][DICT_SEQUENCE_FRAMES] != 0)[-1, 0]

# <codecell>

img = np.array(Image.open("/media/ilisescu/Data1/PhD/data/clouds_subsample10/median.png"))

spriteImg = np.array(Image.open("/media/ilisescu/Data1/PhD/data/clouds_subsample10/cloud2-masked-blended/frame-19501.png"))
spriteColorLocations = np.argwhere(spriteImg[:, :, -1] != 0)
img[spriteColorLocations[:, 0], spriteColorLocations[:, 1], :] = spriteImg[spriteColorLocations[:, 0], spriteColorLocations[:, 1], :-1]
    
Image.fromarray(np.array(img, dtype=np.uint8)).save("tralala2.png")

# <codecell>

bgImg = np.array(Image.open(dataPath + dataSet + "median.png"))
# bgImg = QtGui.QImage(bgImg.data, bgImg.shape[1], bgImg.shape[0], 
#                                            bgImg.strides[0], QtGui.QImage.Format_RGB888)
# img = QtGui.QImage(1280, 720, QtGui.QImage.Format_ARGB32)
# painter = QtGui.QPainter(img)

# <codecell>

for frame in arange(len(window.generatedSequence[0][DICT_SEQUENCE_FRAMES])) : #[251:252] :
#     img.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
    img = np.array(Image.open(dataPath + dataSet + "median.png"))
#     painter.drawImage(QtCore.QPoint(0, 0), bgImg)
    for s in xrange(len(window.generatedSequence)) :
        realFrameIdx = int(window.generatedSequence[s][DICT_SEQUENCE_FRAMES][frame]-1)
        if realFrameIdx >= 0 :
            spriteIdx = window.generatedSequence[s][DICT_SPRITE_IDX]
            sprite = window.trackedSprites[spriteIdx]
            frameName = sprite[DICT_FRAMES_LOCATIONS][np.sort(sprite[DICT_FRAMES_LOCATIONS].keys())[realFrameIdx]].split(os.sep)[-1]
            spriteImg = np.array(Image.open(dataPath + dataSet + sprite[DICT_SPRITE_NAME] + "-masked/" + frameName))
            spriteColorLocations = np.argwhere(spriteImg[:, :, -1] != 0)
            img[spriteColorLocations[:, 0], spriteColorLocations[:, 1], :] = spriteImg[spriteColorLocations[:, 0], spriteColorLocations[:, 1], :-1]
    
    Image.fromarray(np.array(img, dtype=np.uint8)).save(dataPath + dataSet + "generatedSequenceImgs/frame-{0:05}".format(frame+1) + ".png")
    sys.stdout.write('\r' + "Done " + np.string_(frame) + " frame of " + np.string_(len(window.generatedSequence[0][DICT_SEQUENCE_FRAMES])))
    sys.stdout.flush()

# <codecell>

figure(); imshow(spriteImg[:, :, :-1])

# <codecell>

figure(); imshow(np.array(reconstructedImg, dtype=np.uint8))

# <codecell>

print len(currentSpriteImages)
print len(preloadedSpritePatches)

# <codecell>

generatedSequence = np.copy(window.generatedSequence)

# <codecell>

def buildDynProgMRF(semanticLabels, desiredSemantics, startFrame, sequenceLength) :
    """Builds the MRF formulation for a given sprite
    
        \t  semanticLabels   : the semantic labels assigned to the frames in the sprite sequence
        \t  desiredSemantics : the desired label combination
        \t  startFrame       : starting frame for given sprite (used to constrain which frame to start from)
        \t  sequenceLength   : length of sequence to produce (i.e. number of variables to assign a label k \belongs [0, N] where N is number of frames for sprite)
           
        return: unaries  = unary costs for each node in the graph
                pairwise = pairwise costs for each edge in the graph"""
    
    maxCost = 10000000.0
    ## k = num of semantic labels as there should be semantics attached to each frame
    k = len(semanticLabels)
    
    ## unaries are dictated by semantic labels and by startFrame
    
    # start with uniform distribution for likelihood
    likelihood = np.ones((k, sequenceLength))/(k*sequenceLength)
    
#     # set probability of start frame to 1 and renormalize
#     if startFrame >= 0 and startFrame < k :
#         likelihood[startFrame, 0] = 1.0
#         likelihood /= np.sum(likelihood)
    
    # get the costs associated to agreement of the assigned labels to the desired semantics
    # the variance should maybe depend on k so that when there are more frames in a sprite, the variance is higher so that even if I have to follow the timeline for a long time
    # the cost deriveing from the unary cost does not become bigger than the single pairwise cost to break to go straight to the desired semantic label
    # but for now the sprite sequences are not that long and I'm not expecting them to be many orders of magnitude longer 
    # (variance would have to be 5 or 6 orders of magnitude smaller to make breaking the timeline cheaper than following it)
    distVariance = 0.001#0.001
    numSemantics = semanticLabels.shape[-1]
    semanticsCosts = vectorisedMinusLogMultiNormal(semanticLabels, np.array(desiredSemantics).reshape((1, numSemantics)), np.eye(numSemantics)*distVariance, True)
    
    # set unaries to minus log of the likelihood + minus log of the semantic labels' distance to the 
    unaries = -np.log(likelihood) + semanticsCosts.reshape((k, 1)).repeat(sequenceLength, axis=-1)
#     unaries = semanticsCosts.reshape((k, 1)).repeat(sequenceLength, axis=-1)
    
# #     # set cost of start frame to 0 NOTE: not sure if I should use this or the above with the renormalization
#     if startFrame >= 0 and startFrame < k :
#         unaries[startFrame, 0] = 0.0
    if startFrame >= 0 and startFrame < k :
        unaries[:, 0] = maxCost
        unaries[startFrame, 0] = 0.0
    
    ## pairwise are dictated by time constraint and looping ability (i.e. jump probability)
    
    # first dimension is k_n, second represents k_n-1 and last dimension represents all the edges going from graph column w_n-1 to w_n
    pairwise = np.zeros([k, k, sequenceLength-1])
    
    # to enforce timeline give low cost to edge between w_n-1(k = i) and w_n(k = i+1) which can be achieved using
    # an identity matrix with diagonal shifted down by one because only edges from column i-1 and k = j to column i and k=j+1 are viable
    timeConstraint = np.eye(k, k=-1)
    # also allow the sprite to keep looping on label 0 (i.e. show only sprite frame 0 which is the empty frame) so make edge from w_n-1(k=0) to w_n(k=0) viable
    timeConstraint[0, 0] = 1.0
    # also allow the sprite to keep looping from the last frame if necessary so allow to go 
    # from last column (i.e. edge starts from w_n-1(k=last frame)) to second row because first row represents empty frame (i.e. edge goes to w_n(k=1))
    timeConstraint[1, k-1] = 1.0
    # also allow the sprite to go back to the first frame (i.e. empty frame) so allow a low cost edge 
    # from last column (i.e. edge starts from w_n-1(k=last frame)) to first row (i.e. edge goes to w_n(k=0))
    timeConstraint[0, k-1] = 1.0
    
    ## NOTE: don't do all the normal distribution wanking for now: just put very high cost to non viable edges but I'll need something more clever when I try to actually loop a video texture
    ## I would also have to set the time constraint edges' costs to something different from 0 to allow for quicker paths (but more expensive individually) to be chosen when
    ## the semantic label changes
#     timeConstraint /= np.sum(timeConstraint) ## if I normalize here then I need to set mean of gaussian below to what the new max is
#     timeConstraint = vectorisedMinusLogMultiNormal(timeConstraint.reshape((k*k, 1)), np.array([np.max(timeConstraint)]).reshape((1, 1)), np.array([distVariance]).reshape((1, 1)), True)
    timeConstraint = (1.0 - timeConstraint)*maxCost
    
    pairwise = timeConstraint.reshape((k, k, 1)).repeat(sequenceLength-1, axis=-1)
    
    return unaries, pairwise

# <codecell>

def solveDynProgMRF(unaryCosts, pairwiseCosts) :
    ## use the unary and pairwise costs to compute the min cost paths at each node
    # each column represents point n and each row says the index of the k-state that is chosen for the min cost path
    minCostPaths = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]], dtype=int)
    # contains the min cost to reach a certain state k (i.e. row) for point n (i.e. column)
    minCosts = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]])
    # the first row of minCosts is just the unary cost
    minCosts[:, 0] = unaryCosts[:, 0]
    minCostPaths[:, 0] = np.arange(0, unaryCosts.shape[0])
    tmp = 0
    
    k = unaryCosts.shape[0]
    for n in xrange(1, unaryCosts.shape[1]) :
#         tic = time.time()
        costs = minCosts[:, n-1].reshape((k, 1)).repeat(k, axis=-1) + unaryCosts[:, n].reshape((1, k)).repeat(k, axis=0) + pairwiseCosts.T
#         minCostPaths[:, n] = np.ndarray.flatten(np.argmin(costs, axis=0))
#         minCosts[:, n] = np.min(costs, axis=0)
        minCostPaths[:, n] = np.ndarray.flatten(np.argmin(costs, axis=0))
        minCosts[:, n] = costs[minCostPaths[:, n], np.arange(len(costs))]
#         print time.time() -tic
    
#     if saveCosts :
#         costsMat = {}
#         costsMat['minCosts'] = minCosts
#         costsMat['minCostPaths'] = minCostPaths
#         sp.io.savemat("minCosts.mat", costsMat)
    
    ## now find the min cost path starting from the right most n with lowest cost
    minCostTraversal = np.zeros(unaryCosts.shape[1])
    ## last node is the node where the right most node with lowest cost
    minCostTraversal[-1] = np.argmin(minCosts[:, -1]) #minCostPaths[np.argmin(minCosts[:, -1]), -1]
    if np.min(minCosts[:, -1]) == np.inf :
        minCostTraversal[-1] = np.floor((unaryCosts.shape[0])/2)
    
    for i in xrange(len(minCostTraversal)-2, -1, -1) :
#         print i
        minCostTraversal[i] = minCostPaths[minCostTraversal[i+1], i+1]
#     print minCostTraversal
    
#     if isLooping :
#         minCostTraversal[0] = minCostTraversal[-1]
        
#     print np.min(minCosts[:, -1])
#     print minCostTraversal
    
#     return minCosts, minCostPaths, minCostTraversal, tmp
    return minCostTraversal, np.min(minCosts[:, -1]), minCosts, minCostPaths

# <codecell>

tic = time.time()
minCostTraversal, minCost, minCosts, minCostPaths = solveDynProgMRF(unaries.T, pairwise.T)
print "solved in", time.time() - tic; sys.stdout.flush()

# <codecell>

gwv.showCustomGraph(pairwise)

# <codecell>

### this is done using the sparse indexing
def solveSparseDynProgMRF(unaryCosts, pairwiseCosts) :
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
    tmp = 0
    
    sparseIndices = np.where(pairwiseCosts.T != np.max(pairwiseCosts))
    tailIndices = sparseIndices[0]
    headIndices = sparseIndices[1]
    ## this contains which label of variable n-1 is connected to which label of variable n
    indicesInLabelSpace = [list(tailIndices[np.where(headIndices == i)[0]]) for i in np.unique(headIndices)]
    ## this contains index of edge each label of variable n is connected to
    indicesInEdgeSpace = [list(np.where(headIndices == i)[0]) for i in np.unique(headIndices)]
    
    k = unaryCosts.shape[0]
    for n in xrange(1, unaryCosts.shape[1]) :
#         costs = minCosts[:, n-1].reshape((k, 1)).repeat(k, axis=-1) + unaryCosts[:, n].reshape((1, k)).repeat(k, axis=0) + pairwiseCosts.T
#         minCosts[:, n] = np.min(costs, axis=0)
#         minCostPaths[:, n] = np.ndarray.flatten(np.argmin(costs, axis=0))
        prevCosts = minCosts[tailIndices, n-1]
        currentUnaries = unaryCosts[headIndices, n]
        totalCurrentCosts = prevCosts + pairwiseCosts[headIndices, tailIndices] + currentUnaries
        
        minCostPaths[:, n] = [x[np.argmin(totalCurrentCosts[x])] for x in indicesInLabelSpace]
        ## min cost path
#         print minCostIdx
        minCosts[:, n] = totalCurrentCosts[[x[np.argmin(totalCurrentCosts[x])] for x in indicesInEdgeSpace]] ##totalCurrentCosts[minCostPaths[:, n]]
    
#     if saveCosts :
#         costsMat = {}
#         costsMat['minCosts'] = minCosts
#         costsMat['minCostPaths'] = minCostPaths
#         sp.io.savemat("minCosts.mat", costsMat)
    
    ## now find the min cost path starting from the right most n with lowest cost
    minCostTraversal = np.zeros(unaryCosts.shape[1])
    ## last node is the node where the right most node with lowest cost
    minCostTraversal[-1] = np.argmin(minCosts[:, -1]) #minCostPaths[np.argmin(minCosts[:, -1]), -1]
    if np.min(minCosts[:, -1]) == np.inf :
        minCostTraversal[-1] = np.floor((unaryCosts.shape[0])/2)
    
    for i in xrange(len(minCostTraversal)-2, -1, -1) :
#         print i
        minCostTraversal[i] = minCostPaths[minCostTraversal[i+1], i+1]
#     print minCostTraversal
    
#     if isLooping :
#         minCostTraversal[0] = minCostTraversal[-1]
        
#     print np.min(minCosts[:, -1])
#     print minCostTraversal
    
#     return minCosts, minCostPaths, minCostTraversal, tmp
    return minCostTraversal, np.min(minCosts[:, -1]), minCosts, minCostPaths

# <codecell>

## now try and do the optimization completely vectorized
## number of edges connected to each label node of variable n (pairwise stores node at arrow tail as cols and at arrow head as rows)
maxEdgesPerLabel = np.max(np.sum(np.array(pairwiseCosts != np.max(pairwiseCosts), dtype=int), axis=-1))
## initialize this to index of connected label node with highest edge cost (which is then used as padding)
## it contains for each label node of variable n (indexed by rows), all the label nodes of variable n-1 it is connected to by non infinite cost edge (indexed by cols)
nodesConnectedToLabel = np.argmax(pairwiseCosts, axis=-1).reshape((len(pairwiseCosts), 1)).repeat(maxEdgesPerLabel, axis=-1)

sparseIndices = np.where(pairwiseCosts.T != np.max(pairwiseCosts.T))
# print sparseIndices
tailIndices = sparseIndices[0]
headIndices = sparseIndices[1]

## this contains which label of variable n-1 is connected to which label of variable n
indicesInLabelSpace = [list(tailIndices[np.where(headIndices == i)[0]]) for i in np.unique(headIndices)]

for headLabel, tailLabels in zip(arange(0, len(nodesConnectedToLabel)), indicesInLabelSpace) :
    nodesConnectedToLabel[headLabel, 0:len(tailLabels)] = tailLabels
    
tic = time.time()
minCostTraversalSparse, minCostSparse = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)
print "solved in", time.time() - tic; sys.stdout.flush()
print minCostTraversalSparse
print minCostTraversal-minCostTraversalSparse
print minCost-minCostSparse
print minCosts-minCostsSparse
print minCostPaths-minCostPathsSparse

