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
import opengm

app = QtGui.QApplication(sys.argv)

DICT_SPRITE_NAME = 'sprite_name'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SPRITE_IDX = 'sprite_idx' # stores the index in the self.trackedSprites array of the sprite used in the generated sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed

dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

# <codecell>

## load 
trackedSprites = []
for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
    trackedSprites.append(np.load(sprite).item())

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
            dists = np.linalg.norm(projPoints-aabb, axis=-1)
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
                minDist = np.min((minDist, aabb2pointDist(aabb, projPoints[closestPoint, :])))
            else :
                minDist = np.min((minDist, aabb2pointDist(aabb, obb[i, :]), aabb2pointDist(aabb, obb[j, :])))

    return minDist


def aabb2pointDist(aabb, point) :
    dx = np.max((np.min(aabb[:, 0]) - point[0], 0, point[0] - np.max(aabb[:, 0])))
    dy = np.max((np.min(aabb[:, 1]) - point[1], 0, point[1] - np.max(aabb[:, 1])))
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


def getOverlappingSpriteTracksDistance(firstSprite, secondSprite, frameRanges) :
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
        theta = firstSprite[DICT_BBOX_ROTATIONS][firstSpriteKeys[frameRanges[0, i]]]
        rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        bbox1 = np.dot(rotMat, firstSprite[DICT_BBOXES][firstSpriteKeys[frameRanges[0, i]]].T).T
        bbox2 = np.dot(rotMat, secondSprite[DICT_BBOXES][secondSpriteKeys[frameRanges[1, i]]].T).T
        ## if the bboxes coincide then the distance is set to 0
        if np.all(np.abs(bbox1 - bbox2) <= 10**-10) :
            allDists[i] = 0.0
        else :
            allDists[i] = aabb2obbDist(bbox1, bbox2)
        
        ## early out since you can't get lower than 0
        if allDists[i] == 0.0 :
            break
            
    totDist = np.min(allDists)
#     return np.sum(centerDistance)/len(centerDistance), centerDistance    
    return totDist, allDists

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
        minCostTraversal[i] = minCostPaths[minCostTraversal[i+1], i+1]
        
    return minCostTraversal, np.min(minCosts[:, -1])

# <codecell>

## precompute all distances for all possible sprite pairings
# precomputedDistances = {}
# for i in xrange(len(trackedSprites)) :
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

## load all sprite patches
preloadedSpritePatches = []
currentSpriteImages = []
del preloadedSpritePatches
preloadedSpritePatches = []
for sprite in window.trackedSprites :
    maskDir = dataPath + dataSet + sprite[DICT_SPRITE_NAME] + "-masked"
    del currentSpriteImages
    currentSpriteImages = []
    for frameKey in np.sort(sprite[DICT_FRAMES_LOCATIONS].keys()) :
        frameName = sprite[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
        
        if os.path.isdir(maskDir) and os.path.exists(maskDir+"/"+frameName) :
            im = np.array(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED), dtype=np.uint8)
            
            visiblePixels = np.argwhere(im[:, :, -1] != 0)
            topLeft = np.min(visiblePixels, axis=0)
            patchSize = np.max(visiblePixels, axis=0) - topLeft + 1
            
            currentSpriteImages.append({'top_left_pos':topLeft, 'sprite_colors':im[visiblePixels[:, 0], visiblePixels[:, 1], :], 
                                        'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize})
#             currentSpriteImages.append(im[topLeft[0]:topLeft[0]+patchSize[0]+1, topLeft[1]:topLeft[1]+patchSize[1]+1])
        else :
#             im = np.ascontiguousarray(Image.open(sprite[DICT_FRAMES_LOCATIONS][frameIdx]), dtype=np.uint8)
            currentSpriteImages.append(None)
        
        sys.stdout.write('\r' + "Loaded image " + np.string_(len(currentSpriteImages)) + " (" + np.string_(len(sprite[DICT_FRAMES_LOCATIONS])) + ")")
        sys.stdout.flush()
    preloadedSpritePatches.append(np.copy(currentSpriteImages))
    print
    print "done with sprite", sprite[DICT_SPRITE_NAME]

# <codecell>

## go through the generated sequence and check that SPRITE_IDX matches the index in tracked sprites
## load sprites 
trackedSprites = []
for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
    trackedSprites.append(np.load(sprite).item())

print len(trackedSprites)
for sequenceName in np.sort(glob.glob(dataPath+dataSet+"generatedSequence-*")) :
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

## load tracked sprites
trackedSprites = []
for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
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

figure(); plot(possibleShifts, allDistances)
xlim(possibleShifts[0], possibleShifts[-1])

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
        
        self.EXTEND_LENGTH = 100 + 1 ## since I get rid of the frist frame from the generated sequence because it's forced to be the one already showing
        self.TOGGLE_DELAY = 8
        self.BURST_ENTER_DELAY = 2
        self.BURST_EXIT_DELAY = 20
        
        self.DO_EXTEND = 0
        self.DO_TOGGLE = 1
        self.DO_BURST = 2
        
        if len(glob.glob(dataPath+dataSet+"generatedSequence-*")) > 0 :
            ## load latest sequence
            self.generatedSequence = list(np.load(np.sort(glob.glob(dataPath+dataSet+"generatedSequence-*"))[-1]))
            print "loaded", np.sort(glob.glob(dataPath+dataSet+"generatedSequence-*"))[-1]
#             self.generatedSequence = list(np.load(dataPath+dataSet+"generatedSequence-2015-07-07_23:35:48.npy"))
#             self.generatedSequence = generatedSequence
            if len(self.generatedSequence) > 0 :
                ## update sliders
                self.frameIdxSlider.setMaximum(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                self.frameIdxSpinBox.setRange(0, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                
                self.frameInfo.setText("Generated sequence length: " + np.string_(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])))
                
                self.showFrame(self.frameIdx)
        
        self.setFocus()
        
    def renderOneFrame(self) :
        idx = self.frameIdx + 1
        if idx >= 0 and len(self.generatedSequence) > 0 : #idx < len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) :
            self.frameIdxSpinBox.setValue(np.mod(idx, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])))
    
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
                    if spriteIdx >= 0 and spriteIdx < len(preloadedSpritePatches) and sequenceFrameIdx < len(preloadedSpritePatches[spriteIdx]) :
                        ## the QImage for this frame has been preloaded
#                         print "tralala", sequenceFrameIdx, self.frameIdx, s
#                         print "drawing preloaded"
                        self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, self.drawSpritesBox.isChecked(), 
                                         self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked(), preloadedSpritePatches[spriteIdx][sequenceFrameIdx])
                    else :
#                         print "loading image"
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
                bbox = sprite[DICT_BBOXES][frameIdx]
                for p1, p2 in zip(np.mod(arange(4), 4), np.mod(arange(1, 5), 4)) :
                    painter.drawLine(QtCore.QPointF(bbox[p1, 0], bbox[p1, 1]), QtCore.QPointF(bbox[p2, 0], bbox[p2, 1]))
            
            ## draw bbox center
            if doDrawCenter :
                painter.drawPoint(QtCore.QPointF(sprite[DICT_BBOX_CENTERS][frameIdx][0], sprite[DICT_BBOX_CENTERS][frameIdx][1]))
            
    def changeSprite(self, row) :
        if len(self.trackedSprites) > row :
            self.currentSpriteIdx = row
            
        self.setFocus()
            
    def loadTrackedSprites(self) :
        ## going to first frame of first sprite if there were no sprites before loading
        goToNewSprite = len(self.trackedSprites) == 0
        for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
            self.trackedSprites.append(np.load(sprite).item())
        
        self.setSpriteList()
        if len(self.trackedSprites) > 0 and goToNewSprite :
            self.spriteListTable.selectRow(0)
            
    def setSpriteList(self) :
        self.spriteListTable.setRowCount(0)
        if len(self.trackedSprites) > 0 :
            self.spriteListTable.setRowCount(len(self.trackedSprites))
            
            for i in xrange(0, len(self.trackedSprites)):
                self.spriteListTable.setItem(i, 0, QtGui.QTableWidgetItem(self.trackedSprites[i][DICT_SPRITE_NAME]))
        else :
            self.spriteListTable.setRowCount(1)
            self.spriteListTable.setItem(0, 0, QtGui.QTableWidgetItem("No tracked sprites"))
            
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
            print "pressed key", pressedIdx#,
#             if pressedIdx >= 0 and pressedIdx < len(self.trackedSprites) :
#                 print "i.e. sprite", self.trackedSprites[pressedIdx][DICT_SPRITE_NAME]
# #                 self.toggleSpriteSemantics(pressedIdx)
#                 ## spawn new sprite
#                 self.addNewSpriteTrackToSequence(pressedIdx)

#                 if len(self.generatedSequence) > 0 :
#                     ## extend existing sprites if necessary
#                     if self.frameIdx > len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH :
#                         print self.frameIdx, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]), self.EXTEND_LENGTH, 1, self.frameIdx-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1
#                         self.leaveOneOutExtension(len(self.generatedSequence)-1, 
#                                                   self.frameIdx-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1, 
#                                                   len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                        
#                     ## now toggle new sprite to visible
#                     print "toggling new sprite"
#                     extendMode = {}
#                     extendMode[len(self.generatedSequence)-1] = self.DO_TOGGLE
#                     self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
#                                         self.frameIdx, self.resolveCompatibilityBox.isChecked())
                    
#                     ## check whether desired label has been reached and if not extend (need to actually do this but for now just check the sprite frame is larger than 0)
#                     while self.generatedSequence[-1][DICT_SEQUENCE_FRAMES][-1] < 1 :
#                         print "extending new sprite because semantics not reached"
#                         ## extend existing sprites if necessary
#                         if len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1 > len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH :
#                             self.leaveOneOutExtension(len(self.generatedSequence)-1, 
#                                                       len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1, 
#                                                       len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                            
#                         extendMode[len(self.generatedSequence)-1] = self.DO_EXTEND
#                         self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
#                                             len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1, 
#                                             self.resolveCompatibilityBox.isChecked())
                        
#                     print "toggling back new sprite"
#                     ## toggle it back to not visible
#                     extendMode[len(self.generatedSequence)-1] = self.DO_TOGGLE
#                     self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
#                                         len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
                    
#                     additionalFrames = len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES]) - len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])
#                     if additionalFrames < 0 :
#                         ## extend new sprite's sequence to match total sequence's length
#                         print "extending new sprite's sequence"
#                         extendMode = {}
#                         extendMode[len(self.generatedSequence)-1] = self.DO_EXTEND
#                         self.extendSequence(self.extendSequenceTracksSemantics(-additionalFrames+1, extendMode), 
#                                             len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
#                     elif additionalFrames > 0 :
#                         ## extend existing sprites' sequences to match the new total sequence's length because of newly added sprite
#                         print "extending existing sprites' sequences"
#                         extendMode = {}
#                         for i in xrange(len(self.generatedSequence)-1) :
#                             extendMode[i] = self.DO_EXTEND
#                         self.extendSequence(self.extendSequenceTracksSemantics(additionalFrames+1, extendMode), 
#                                             len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                        
#                 self.showFrame(self.frameIdx)                        
#             else :
#                 print
        elif e.key() == QtCore.Qt.Key_Space :
            if len(self.generatedSequence) > 0 :
                self.frameIdxSpinBox.setValue(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
            extendMode = {}
            for i in xrange(len(self.generatedSequence)) :
                extendMode[i] = self.DO_EXTEND
            if len(self.generatedSequence) > 0 :
                self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), self.frameIdx)
        elif e.key() == QtCore.Qt.Key_T :
            print self.currentSpriteIdx
            if self.currentSpriteIdx >= 0 and self.currentSpriteIdx < len(self.trackedSprites) :
                print "spawining sprite", self.trackedSprites[self.currentSpriteIdx][DICT_SPRITE_NAME]
                ## spawn new sprite
                self.addNewSpriteTrackToSequence(self.currentSpriteIdx)

                if len(self.generatedSequence) > 0 :
                    ## extend existing sprites if necessary
                    if self.frameIdx > len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH :
                        print self.frameIdx, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]), self.EXTEND_LENGTH, 1, self.frameIdx-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1
                        self.leaveOneOutExtension(len(self.generatedSequence)-1, 
                                                  self.frameIdx-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1, 
                                                  len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                        
                    ## now toggle new sprite to visible
                    print "toggling new sprite"
                    extendMode = {}
                    extendMode[len(self.generatedSequence)-1] = self.DO_TOGGLE
                    self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
                                        self.frameIdx, self.resolveCompatibilityBox.isChecked())
                    
                    ## check whether desired label has been reached and if not extend (need to actually do this but for now just check the sprite frame is larger than 0)
                    while self.generatedSequence[-1][DICT_SEQUENCE_FRAMES][-1] < 1 :
                        print "extending new sprite because semantics not reached"
                        ## extend existing sprites if necessary
                        if len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1 > len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH :
                            self.leaveOneOutExtension(len(self.generatedSequence)-1, 
                                                      len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1-(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) - self.EXTEND_LENGTH) + 1, 
                                                      len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                            
                        extendMode[len(self.generatedSequence)-1] = self.DO_EXTEND
                        self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
                                            len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1, 
                                            self.resolveCompatibilityBox.isChecked())
                        
                    print "toggling back new sprite"
                    ## toggle it back to not visible
                    extendMode[len(self.generatedSequence)-1] = self.DO_TOGGLE
                    self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), 
                                        len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
                    
                    additionalFrames = len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES]) - len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])
                    if additionalFrames < 0 :
                        ## extend new sprite's sequence to match total sequence's length
                        print "extending new sprite's sequence"
                        extendMode = {}
                        extendMode[len(self.generatedSequence)-1] = self.DO_EXTEND
                        self.extendSequence(self.extendSequenceTracksSemantics(-additionalFrames+1, extendMode), 
                                            len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
                    elif additionalFrames > 0 :
                        ## extend existing sprites' sequences to match the new total sequence's length because of newly added sprite
                        print "extending existing sprites' sequences"
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
                    
                    print np.array(np.vstack((overlappingSpriteSequence.reshape((1, len(overlappingSpriteSequence))),
                                                                    spriteSequence.reshape((1, len(spriteSequence))))), dtype=int)
                    
#                     print frameRanges
                    
                    if frameRanges != None :
#                         totalDistance, distances = getOverlappingSpriteTracksDistance(self.trackedSprites[self.generatedSequence[i][DICT_SPRITE_IDX]], 
#                                                                                       self.trackedSprites[self.generatedSequence[spriteIdx][DICT_SPRITE_IDX]], frameRanges)
                
                        spriteIdxs = np.array([self.generatedSequence[i][DICT_SPRITE_IDX], self.generatedSequence[spriteIdx][DICT_SPRITE_IDX]])
                        sortIdxs = np.argsort(spriteIdxs)
                        pairing = np.string_(spriteIdxs[sortIdxs][0]) + np.string_(spriteIdxs[sortIdxs][1])
                        pairingShift = frameRanges[sortIdxs, 0][1]-frameRanges[sortIdxs, 0][0]
                        totalDistance = precomputedDistances[pairing][pairingShift]
                
                        print "lala", totalDistance, precomputedDistances[pairing][pairingShift], spriteIdxs, pairing, pairingShift, spriteIdx, frameRanges[sortIdxs, 0]
                        
                        ## find all pairs of frame that show the same label as the desired label (i.e. [0.0, 1.0])
                        tmp = np.all(overlappingSpriteSemanticLabels[overlappingSpriteSequence] == spriteSemanticLabels[spriteSequence], axis=1)
                        if totalDistance > 5.0 : 
                            areSpritesCompatible[np.all((np.all(spriteSemanticLabels[spriteSequence] == np.array([0.0, 1.0]), axis=1), tmp), axis=0)] = True
                    else :
                        print "sprites not overlapping"
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
                print unaries
                print pairwise
                print desiredSemantics[i]
                print desiredStartFrame, len(desiredSemantics[i])
                
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
                    print "resolving compatibility", i, startingFrame
                    
                    self.frameInfo.setText("Optimizing sequence - resolving compatibility")
                    QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
                    
                    isSpriteCompatible = self.checkIsCompatible(i, minCostTraversal, startingFrame)
                    print isSpriteCompatible
                    print len(isSpriteCompatible)
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
                            if True or np.mod(count, 20) == 0 :
                                print "iteration", count, ": solved traversal for sprite", i , "in", time.time() - tic, 
                                print "num of zeros:", len(np.argwhere(minCostTraversal == 0)); sys.stdout.flush()
                            
                                self.frameInfo.setText("Optimizing sequence - resolving compatibility " + np.string_(count))
                                QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
#                                 print minCostTraversal, minCost

                            isSpriteCompatible = self.checkIsCompatible(i, minCostTraversal, startingFrame)
#                             if count == 200 :
#                                 break
                        else :
                            print "done"
#                             print minCostTraversal
                            break
                    gwv.showCustomGraph(unaries[1:, :])
                
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
                    
                print "sequence with", len(self.generatedSequence), "sprites, extended by", len(desiredSemantics[i])-1, "frames"
    
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
            print "adding new sprite to sequence"
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
        else :
            self.doPlaySequence = True
            self.playSequenceButton.setIcon(self.pauseIcon)
            self.playTimer.start()
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frameLabel.installEventFilter(self)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameIdxSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
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
        
        self.spriteListTable = QtGui.QTableWidget(1, 1)
        self.spriteListTable.horizontalHeader().setStretchLastSection(True)
        self.spriteListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Tracked sprites"))
        self.spriteListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Fixed)
        self.spriteListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.spriteListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.spriteListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.spriteListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.spriteListTable.setItem(0, 0, QtGui.QTableWidgetItem("No tracked sprites"))
        
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
        self.autoSaveBox.setChecked(True)
        
        self.deleteSequenceButton = QtGui.QPushButton("Delete Sequence")
        
        self.resolveCompatibilityBox = QtGui.QCheckBox("Resolve Compatibility")
        self.resolveCompatibilityBox.setChecked(True)
        
        
        ## SIGNALS ##
        
        self.frameIdxSlider.valueChanged[int].connect(self.frameIdxSpinBox.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.frameIdxSlider.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.showFrame)
        
        self.spriteListTable.currentCellChanged.connect(self.changeSprite)
        self.spriteListTable.cellPressed.connect(self.changeSprite)
        
        self.playSequenceButton.clicked.connect(self.playSequenceButtonPressed)
        self.deleteSequenceButton.clicked.connect(self.deleteGeneratedSequence)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.spriteListTable)
        controlsLayout.addWidget(self.drawSpritesBox)
        controlsLayout.addWidget(self.drawBBoxBox)
        controlsLayout.addWidget(self.drawCenterBox)
        controlsLayout.addWidget(self.playSequenceButton)
        controlsLayout.addWidget(self.autoSaveBox)
        controlsLayout.addWidget(self.deleteSequenceButton)
        controlsLayout.addWidget(self.resolveCompatibilityBox)
        
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

figure(); imshow(preloadedSpritePatches[-1][-1]['sprite_patch'])

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

