# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import sys

import cv2
import time
import os
import scipy.io as sio
import glob
import random

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

precomputedDistances = np.load(dataPath + dataSet + "precomputedDistances.npy").item()
## load 
trackedSprites = []
for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
    trackedSprites.append(np.load(sprite).item())
    
## load all sprite patches
preloadedSpritePatches = []
currentSpriteImages = []
del preloadedSpritePatches
preloadedSpritePatches = []
for sprite in trackedSprites :
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
        
        self.setWindowTitle("Fogger++")
        self.resize(1280, 720)
        
        self.playIcon = QtGui.QIcon("play.png")
        self.pauseIcon = QtGui.QIcon("pause.png")
        self.doPlay = False
        
        self.createGUI()
        
        self.isDirty = False
        self.upPressed = False
        self.leftPressed = False
        self.downPressed = False
        self.rightPressed = False
        
        self.playerPos = np.array([1098.0, 255.0])
        self.PLAYER_SPEED = 10.0
        self.goalPoint = np.array([579.0, 688.0])
        
        self.EXTEND_LENGTH = 20 + 1 ## since I get rid of the frist frame from the generated sequence because it's forced to be the one already showing
        self.TOGGLE_DELAY = 8
        self.BURST_ENTER_DELAY = 2
        self.BURST_EXIT_DELAY = 20
        
        self.trackedSprites = []
        self.spriteToSpawnIdx = 0
        
        self.DO_EXTEND = 0
        self.DO_TOGGLE = 1
        self.DO_BURST = 2
        
        self.frameIdx = 0
        self.overlayImg = QtGui.QImage(QtCore.QSize(100, 100), QtGui.QImage.Format_ARGB32)
        
        ## get background image
        im = np.ascontiguousarray(Image.open(dataPath + dataSet + "median.png"))
        self.bgImage = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setFixedSize(self.bgImage.width(), self.bgImage.height())
        self.frameLabel.setImage(self.bgImage)
        
        ## get depth image 
        self.depthImg = np.array(Image.open(dataPath + dataSet + "medianApproxDepth.png"))
        
        self.loadTrackedSprites()
        
        self.precomputedSequence = []
#         self.generatedSequence = []
        
#         if len(glob.glob(dataPath+dataSet+"generatedSequence-*")) > 0 :
#             ## load latest sequence
#             self.generatedSequence = list(np.load(np.sort(glob.glob(dataPath+dataSet+"generatedSequence-*"))[-1]))
# #             self.generatedSequence = generatedSequence
#             if len(self.generatedSequence) > 0 :
#                 ## update sliders
# #                 self.frameIdxSlider.setMaximum(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
# #                 self.frameIdxSpinBox.setRange(0, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                
#                 self.frameInfo.setText("Generated sequence length: " + np.string_(len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])))
                
#                 self.showFrame(self.frameIdx)
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.renderOneFrame)
        self.playTimer.start()
        
        self.setFocus()
        
    def renderOneFrame(self) :
#         idx = self.frameIdx
        
        ## if playing, show next frame in the sequence
        if self.doPlay :
#             idx = self.frameIdx + 1
#             if idx >= 0 and len(self.precomputedSequence) > 0 : 
#                 idx = np.mod(idx, len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES]))
            self.isDirty = True
            
        if self.upPressed :
            self.updatePlayerPos(np.array([0.0, -self.PLAYER_SPEED]))
            self.isDirty = True
        if self.leftPressed :
            self.updatePlayerPos(np.array([-self.PLAYER_SPEED, 0.0]))
            self.isDirty = True
        if self.downPressed :
            self.updatePlayerPos(np.array([0.0, self.PLAYER_SPEED]))
            self.isDirty = True
        if self.rightPressed :
            self.updatePlayerPos(np.array([self.PLAYER_SPEED, 0.0]))
            self.isDirty = True
        
        if self.isDirty :
#             self.showFrame(idx)
            self.showFrame(0)
            
        for s in xrange(len(self.precomputedSequence)) :
            ## index in self.trackedSprites of current sprite in self.precomputedSequence
            spriteIdx = self.precomputedSequence[s][DICT_SPRITE_IDX]
            ## index of current sprite frame to visualize
            sequenceFrameIdx = self.precomputedSequence[s][DICT_SEQUENCE_FRAMES][0]-1
            ## -1 stands for not shown or eventually for more complicated sprites as the base frame to keep showing when sprite is frozen
            ## really in the sequence I have 0 for the static frame but above I do -1
            if sequenceFrameIdx >= 0 :
                ## the trackedSprites data is indexed (i.e. the keys) by the frame indices in the original full sequence and keys are not sorted
                frameToShowIdx = np.sort(self.trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())[sequenceFrameIdx]
                
                bbox = self.trackedSprites[spriteIdx][DICT_BBOXES][frameToShowIdx]
                bbox = np.vstack((bbox, bbox[0, :]))
                if cv2.pointPolygonTest(np.array(bbox, dtype=np.float32), (self.playerPos[0], self.playerPos[1]), False) >= 0 :
                    self.frameInfo.setText("You Died !!!")
                    self.playButton.click()
                    self.restartGame()
                
        if np.linalg.norm(self.playerPos-self.goalPoint) < 21.0 :
            self.frameInfo.setText("You Win !!!")
            self.playButton.click()
            self.restartGame()
            
        ## delete first frame of each sequence
        for s in xrange(len(self.precomputedSequence)) :
            self.precomputedSequence[s][DICT_SEQUENCE_FRAMES] = self.precomputedSequence[s][DICT_SEQUENCE_FRAMES][1:]
            self.precomputedSequence[s][DICT_DESIRED_SEMANTICS] = self.precomputedSequence[s][DICT_DESIRED_SEMANTICS][1:]
            
        if len(self.precomputedSequence) > 0 and len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES]) < self.EXTEND_LENGTH :
            self.extendFullSequence(self.EXTEND_LENGTH, len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES])-1)
            
            self.spriteToSpawnIdx = random.choice(arange(len(window.trackedSprites)))
            self.spawnSprite()
            
                
        self.isDirty = False
        
    
    def showFrame(self, idx) :
        if idx >= 0 and len(self.precomputedSequence) > 0 and idx < len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES]) :
#             self.frameIdx = idx
            
            if self.bgImage != None and self.overlayImg.size() != self.bgImage.size() :
                self.overlayImg = self.overlayImg.scaled(self.bgImage.size())
            ## empty image
            self.overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
            
            ## go through all the sprites used in the sequence
            for s in xrange(len(self.precomputedSequence)) :
                ## index in self.trackedSprites of current sprite in self.precomputedSequence
                spriteIdx = self.precomputedSequence[s][DICT_SPRITE_IDX]
                ## index of current sprite frame to visualize
#                 sequenceFrameIdx = self.precomputedSequence[s][DICT_SEQUENCE_FRAMES][self.frameIdx]-1
                sequenceFrameIdx = self.precomputedSequence[s][DICT_SEQUENCE_FRAMES][0]-1
                ## -1 stands for not shown or eventually for more complicated sprites as the base frame to keep showing when sprite is frozen
                ## really in the sequence I have 0 for the static frame but above I do -1
                if sequenceFrameIdx >= 0 :
                    ## the trackedSprites data is indexed (i.e. the keys) by the frame indices in the original full sequence and keys are not sorted
                    frameToShowIdx = np.sort(self.trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())[sequenceFrameIdx]
                    if spriteIdx >= 0 and spriteIdx < len(preloadedSpritePatches) and sequenceFrameIdx < len(preloadedSpritePatches[spriteIdx]) :
                        ## the QImage for this frame has been preloaded
#                         print "tralala", sequenceFrameIdx, self.frameIdx, s
#                         print "drawing preloaded"
                        self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, True, 
                                         False, True, False, preloadedSpritePatches[spriteIdx][sequenceFrameIdx])
                    else :
#                         print "loading image"
                        self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, True, False, True, False)
                    
#                     self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, self.drawSpritesBox.isChecked(), 
#                                      self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked())
#                     self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, True, False, True, False)
            
            self.drawOverlay(None, None, False, True, False, False)
            
            self.frameLabel.setFixedSize(self.overlayImg.width(), self.overlayImg.height())
            self.frameLabel.setOverlay(self.overlayImg)
            
    def drawOverlay(self, sprite, frameIdx, doDrawSprite, doDrawPlayer, doDrawBBox, doDrawCenter, spritePatch = None) :
        if self.overlayImg != None :
            painter = QtGui.QPainter(self.overlayImg)
            
            if sprite != None and frameIdx != None :
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
            
            ## draw player dot
            if doDrawPlayer :
#                 size = np.log(500.0/float(self.depthImg[self.playerPos[1], self.playerPos[0], 0]))*2
#                 size = (255.0-float(self.depthImg[self.playerPos[1], self.playerPos[0], 0]))/10.0
                size = log(255.0/float(self.depthImg[self.playerPos[1], self.playerPos[0], 0]))*6
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), size, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPoint(QtCore.QPointF(self.playerPos[0], self.playerPos[1]))
            
                
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 0, 0, 255), 40, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPoint(QtCore.QPointF(self.goalPoint[0], self.goalPoint[1]))
                
    def getOptimizedSequence(self, sprite, spriteIdx, competingSequences, desiredSemantics, desiredStartFrame, resolveCompatibility = False) :
#         spriteIdx = self.precomputedSequence[i][DICT_SPRITE_IDX]
                
        ### HACK ### semanticLabels for each sprite are just saying that the first frame of the sprite sequence means sprite not visible and the rest mean it is
        semanticLabels = np.zeros((len(sprite[DICT_BBOX_CENTERS])+1, 2))
        ## label 0 means sprite not visible (i.e. only show the first empty frame)
        semanticLabels[0, 0] = 1.0
        semanticLabels[1:, 1] = 1.0
        
        ## do dynamic programming optimization
        unaries, pairwise = getMRFCosts(semanticLabels, desiredSemantics, desiredStartFrame, len(desiredSemantics))
        
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
                        
        minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)#solveMRF(unaries, pairwise)
        
        if resolveCompatibility :
            
            isSpriteCompatible = self.checkIsCompatible(sprite, spriteIdx, minCostTraversal, competingSequences)
            count = 0
            while True :
                if np.any(np.negative(isSpriteCompatible)) :
                    if count == 0 :
                        return None
                    ## when I do the check for all the sprites in the sequence I would have to take an AND over all the isSpriteCompatible arrays but now I know there's only 1 sprite
                    
                    ## change the unaries to increase the cost for the frames where the isSpriteCompatible is False
                    unaries[np.arange(len(minCostTraversal), dtype=int)[np.negative(isSpriteCompatible)], minCostTraversal[np.negative(isSpriteCompatible)]] += 1000.0
                    tic = time.time()
                    minCostTraversal, minCost = solveSparseDynProgMRF(unaries.T, pairwise.T, nodesConnectedToLabel)#solveMRF(unaries, pairwise)
#                     if True or np.mod(count, 20) == 0 :
#                         print "iteration", count, ": solved traversal for sprite", sprite[DICT_SPRITE_NAME] , "in", time.time() - tic, 
#                         print "num of zeros:", len(np.argwhere(minCostTraversal == 0)); sys.stdout.flush()

                    isSpriteCompatible = self.checkIsCompatible(sprite, spriteIdx, minCostTraversal, competingSequences)
                else :
                    print "done"
                    break
                count += 1
                    
        return minCostTraversal
    
    def checkIsCompatible(self, sprite, spriteIdx, spriteSequence, competingSequences) :
        
        isSpriteCompatible = np.ones(len(spriteSequence), dtype=np.bool)
        
        
        spriteTotLength = len(sprite[DICT_BBOX_CENTERS])
        spriteSemanticLabels = np.zeros((spriteTotLength+1, 2))
        ## label 0 means sprite not visible (i.e. only show the first empty frame)
        spriteSemanticLabels[0, 0] = 1.0
        spriteSemanticLabels[1:, 1] = 1.0
            
        for competingSequence in competingSequences :
            overlappingSpriteSequence = np.array(competingSequence[DICT_SEQUENCE_FRAMES], dtype=int)
            
            overlappingSpriteTotLength = len(self.trackedSprites[competingSequence[DICT_SPRITE_IDX]][DICT_BBOX_CENTERS])
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
            
            if frameRanges != None :
            
                spriteIdxs = np.array([competingSequence[DICT_SPRITE_IDX], spriteIdx])
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
            
            isSpriteCompatible = np.all((isSpriteCompatible, areSpritesCompatible), axis=0)
                    
        return isSpriteCompatible
    
    def extendSequenceTrackSemantics(self, currentSemantics, length, mode) :
        
        desiredSemantics = np.array([1.0, 0.0]).reshape((1, 2)).repeat(length, axis=0)
        if mode == self.DO_EXTEND :
            ## extend semantics
            desiredSemantics = currentSemantics.repeat(length, axis=0)
        elif mode == self.DO_TOGGLE :
            ## toggle semantics
            desiredSemantics = self.toggleSequenceTrackSemantics(currentSemantics, length, self.TOGGLE_DELAY)
        elif mode == self.DO_BURST :
            ## burst toggle semantics from current to toggle and back to current
            desiredSemantics = self.burstSemanticsToggle(currentSemantics, length, self.BURST_ENTER_DELAY, self.BURST_EXIT_DELAY)
            
        return desiredSemantics
                
    def extendPrecomputedSequence(self, sequenceIdx, extensionLength, startFrame) :
#         for i in xrange(len(self.precomputedSequence)) :
        if sequenceIdx >= 0 and sequenceIdx < len(self.precomputedSequence) :
            if len(self.precomputedSequence[sequenceIdx][DICT_DESIRED_SEMANTICS]) > 0 :
                currentSemantics = self.precomputedSequence[sequenceIdx][DICT_DESIRED_SEMANTICS][-1, :].reshape((1, 2))
            else :
                ## hardcoded desired "not show" label
                currentSemantics = np.array([1.0, 0.0]).reshape((1, 2))
                
            desiredSemantics = self.extendSequenceTrackSemantics(currentSemantics, extensionLength, self.DO_EXTEND)
            
            if len(self.precomputedSequence[sequenceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
                desiredStartFrame = 0
            else :
                desiredStartFrame = self.precomputedSequence[sequenceIdx][DICT_SEQUENCE_FRAMES][startFrame]
                
            spriteSequence = self.getOptimizedSequence(self.trackedSprites[self.precomputedSequence[sequenceIdx][DICT_SPRITE_IDX]], 
                                                       self.precomputedSequence[sequenceIdx][DICT_SPRITE_IDX], 
                                                       [self.precomputedSequence[i] for i in np.argwhere(np.arange(len(self.precomputedSequence)) != sequenceIdx)], 
                                                       desiredSemantics, desiredStartFrame, resolveCompatibility = False)
            
            if spriteSequence != None :
                ## update dictionary
                # don't take the first frame of the spriteSequence as it would just repeat the last seen frame
                self.precomputedSequence[sequenceIdx][DICT_SEQUENCE_FRAMES] = np.hstack((self.precomputedSequence[sequenceIdx][DICT_SEQUENCE_FRAMES][:startFrame+1], 
                                                                               spriteSequence[1:]))
                self.precomputedSequence[sequenceIdx][DICT_DESIRED_SEMANTICS] = np.vstack((self.precomputedSequence[sequenceIdx][DICT_DESIRED_SEMANTICS][:startFrame+1],
                                                                               desiredSemantics[1:, :]))
                
                return True
            
        return False
    
    def spawnSprite(self) :
        
        if self.spriteToSpawnIdx >= 0 and self.spriteToSpawnIdx < len(self.trackedSprites) :
            print "spawning sprite", self.trackedSprites[self.spriteToSpawnIdx][DICT_SPRITE_NAME]
            ## spawn new sprite
#             self.addNewSpriteTrackToSequence(self.spriteToSpawnIdx)
            self.spriteToSpawn = {
                                   DICT_SPRITE_IDX:self.spriteToSpawnIdx,
                                   DICT_SPRITE_NAME:self.trackedSprites[self.spriteToSpawnIdx][DICT_SPRITE_NAME],
                                   DICT_SEQUENCE_FRAMES:np.empty(0, dtype=int),
                                   DICT_DESIRED_SEMANTICS:np.empty((0, 2), dtype=float)
                                 }
            extensionLength = self.EXTEND_LENGTH
            if len(self.precomputedSequence) > 0 :
                extensionLength = len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES])
        
            desiredSemantics = self.extendSequenceTrackSemantics(np.array([1.0, 0.0]).reshape((1, 2)), extensionLength, self.DO_TOGGLE)
            
            tic = time.time()
            spriteSequence = self.getOptimizedSequence(self.trackedSprites[self.spriteToSpawnIdx], self.spriteToSpawnIdx, 
                                                       self.precomputedSequence, desiredSemantics, 0, resolveCompatibility = True)
            print "1", time.time() - tic
            
            tic = time.time()
            if spriteSequence != None :
                ## update dictionary
                # don't take the first frame of the spriteSequence as it would just repeat the last seen frame
                self.spriteToSpawn[DICT_SEQUENCE_FRAMES] = np.hstack((self.spriteToSpawn[DICT_SEQUENCE_FRAMES], spriteSequence[1:]))
                self.spriteToSpawn[DICT_DESIRED_SEMANTICS] = np.vstack((self.spriteToSpawn[DICT_DESIRED_SEMANTICS], desiredSemantics[1:, :]))
                
                self.precomputedSequence.append(self.spriteToSpawn)
                
                ## toggle the sprite back
                desiredSemantics = self.extendSequenceTrackSemantics(self.precomputedSequence[-1][DICT_DESIRED_SEMANTICS][-1, :].reshape((1, 2)), 
                                                                     self.EXTEND_LENGTH, self.DO_TOGGLE)
            
                spriteSequence = self.getOptimizedSequence(self.trackedSprites[self.spriteToSpawnIdx], self.spriteToSpawnIdx, 
                                                           self.precomputedSequence, desiredSemantics, 
                                                           self.precomputedSequence[-1][DICT_SEQUENCE_FRAMES][-1], resolveCompatibility = False)
                
                ## update dictionary
                # don't take the first frame of the spriteSequence as it would just repeat the last seen frame
                self.precomputedSequence[-1][DICT_SEQUENCE_FRAMES] = np.hstack((self.precomputedSequence[-1][DICT_SEQUENCE_FRAMES], spriteSequence[1:]))
                self.precomputedSequence[-1][DICT_DESIRED_SEMANTICS] = np.vstack((self.precomputedSequence[-1][DICT_DESIRED_SEMANTICS], desiredSemantics[1:, :]))
                
                
                ## check which sprite has the most frames (the last one or the existing ones?) and extend
                additionalFrames = len(self.precomputedSequence[-1][DICT_SEQUENCE_FRAMES]) - len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES])
                if additionalFrames < 0 :
                    ## extend new sprite's sequence to match total sequence's length
                    print "extending new sprite's sequence by", -additionalFrames+1
                    self.extendPrecomputedSequence(len(self.precomputedSequence)-1, -additionalFrames+1, 
                                                   len(self.precomputedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
                    
                elif additionalFrames > 0 :
                    ## extend existing sprites' sequences to match the new total sequence's length because of newly added sprite
                    print "extending existing sprites' sequences by", additionalFrames+1
                    self.leaveOneOutExtension(len(self.precomputedSequence)-1, additionalFrames+1, 
                                              len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES])-1)
            print "2", time.time() - tic
                

    def extendFullSequence(self, extensionLength, startFrame) :
        for i in xrange(len(self.precomputedSequence)) :
            self.extendPrecomputedSequence(i, extensionLength, startFrame)
    
    def leaveOneOutExtension(self, leaveOutIdx, extensionLength, startFrame) :
        for i in xrange(len(self.precomputedSequence)) :
            if i != leaveOutIdx :
                self.extendPrecomputedSequence(i, extensionLength, startFrame)
#         extendMode = {}
#         for i in xrange(len(self.precomputedSequence)) :
#             if i != leaveOutIdx :
#                 extendMode[i] = self.DO_EXTEND
#         if len(extendMode) > 0 :
#             print "extending existing sprites by", extensionLength, "frames and leaving out", leaveOutIdx
#             self.extendSequence(self.extendSequenceTracksSemantics(extensionLength, extendMode), startFrame)
    
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
    
#     def leaveOneOutExtension(self, leaveOutIdx, extensionLength, startFrame) :
#         extendMode = {}
#         for i in xrange(len(self.precomputedSequence)) :
#             if i != leaveOutIdx :
#                 extendMode[i] = self.DO_EXTEND
#         if len(extendMode) > 0 :
#             print "extending existing sprites by", extensionLength, "frames and leaving out", leaveOutIdx
#             self.extendSequence(self.extendSequenceTracksSemantics(extensionLength, extendMode), 
#                                 startFrame)
            
    def restartGame(self) :
        self.isDirty = False
        self.upPressed = False
        self.leftPressed = False
        self.downPressed = False
        self.rightPressed = False
        
        self.playerPos = np.array([1098.0, 255.0])
        self.PLAYER_SPEED = 10.0
        self.goalPoint = np.array([579.0, 688.0])
        
        self.precomputedSequence = []
        
#         self.frameIdx = 0
#         self.showFrame(self.frameIdx)
            
    def loadTrackedSprites(self) :
        ## going to first frame of first sprite if there were no sprites before loading
#         goToNewSprite = len(self.trackedSprites) == 0
        for sprite in glob.glob(dataPath + dataSet + "sprite*.npy") :
            self.trackedSprites.append(np.load(sprite).item())
            
    def closeEvent(self, event) :
        print "closing"
        sys.stdout.flush()
            
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_Up and not e.isAutoRepeat() :
            self.upPressed = True
#             self.updatePlayerPos(np.array([0.0, -self.PLAYER_SPEED]))
        if e.key() == QtCore.Qt.Key_Left and not e.isAutoRepeat() :
            self.leftPressed = True
#             self.updatePlayerPos(np.array([-self.PLAYER_SPEED, 0.0]))
        if e.key() == QtCore.Qt.Key_Down and not e.isAutoRepeat() :
            self.downPressed = True
#             self.updatePlayerPos(np.array([0.0, self.PLAYER_SPEED]))
        if e.key() == QtCore.Qt.Key_Right and not e.isAutoRepeat() :
            self.rightPressed = True
#             self.updatePlayerPos(np.array([self.PLAYER_SPEED, 0.0]))

#         if not e.isAutoRepeat() :
#             print "press", self.upPressed, self.leftPressed, self.downPressed, self.rightPressed, 
#             if e.key() == QtCore.Qt.Key_Up :
#                 print "up"
#             if e.key() == QtCore.Qt.Key_Left :
#                 print "left"
#             if e.key() == QtCore.Qt.Key_Down :
#                 print "down"
#             if e.key() == QtCore.Qt.Key_Right :
#                 print "right"
#             sys.stdout.flush()
    
    def keyReleaseEvent(self, e) :
        if e.key() == QtCore.Qt.Key_Up and not e.isAutoRepeat() :
            self.upPressed = False
        if e.key() == QtCore.Qt.Key_Left and not e.isAutoRepeat() :
            self.leftPressed = False
        if e.key() == QtCore.Qt.Key_Down and not e.isAutoRepeat() :
            self.downPressed = False
        if e.key() == QtCore.Qt.Key_Right and not e.isAutoRepeat() :
            self.rightPressed = False
            
#         if not e.isAutoRepeat() :
#             print "release", self.upPressed, self.leftPressed, self.downPressed, self.rightPressed, 
#             if e.key() == QtCore.Qt.Key_Up :
#                 print "up"
#             if e.key() == QtCore.Qt.Key_Left :
#                 print "left"
#             if e.key() == QtCore.Qt.Key_Down :
#                 print "down"
#             if e.key() == QtCore.Qt.Key_Right :
#                 print "right"
#             sys.stdout.flush()
    
    def updatePlayerPos(self, delta) :
        oldPlayerPos = self.playerPos
        
        newPlayerPos = self.playerPos + delta
        
        ## check x is in bounds
        if newPlayerPos[0] < 0 :
            newPlayerPos[0] = 0.0
        elif newPlayerPos[0] >= self.bgImage.width() :
            newPlayerPos[0] = self.bgImage.width()-1
        
        ## check y is in bounds
        if newPlayerPos[1] < 0 :
            newPlayerPos[1] = 0.0
        elif newPlayerPos[1] >= self.bgImage.height() :
            newPlayerPos[1] = self.bgImage.height()-1
        
        self.playerPos = newPlayerPos
        ## check depth
        ## if location at new x position is invisible then I reached depth bounds
        if self.depthImg[newPlayerPos[1], newPlayerPos[0], -1] == 0 :
            ## if new pos is not good try old x and new y
            if self.depthImg[newPlayerPos[1], oldPlayerPos[0], -1] != 0 :
                self.playerPos[0] = oldPlayerPos[0]
            ## if new pos is not good try old y and new x
            if self.depthImg[oldPlayerPos[1], newPlayerPos[0], -1] != 0 :
                self.playerPos[1] = oldPlayerPos[1]
        
#         print self.depthImg[self.playerPos[1], self.playerPos[0], 0]; sys.stdout.flush()
        
    def eventFilter(self, obj, event) :
        if obj == self.frameLabel and event.type() == QtCore.QEvent.Type.KeyPress :
            self.keyPressEvent(event)
            return True
        if obj == self.frameLabel and event.type() == QtCore.QEvent.Type.KeyRelease :
            self.keyPressEvent(event)
            return True
        return QtGui.QWidget.eventFilter(self, obj, event)
    
    def playButtonPressed(self) :
        if self.doPlay :
            self.doPlay = False
            self.playButton.setIcon(self.playIcon)
        else :
            self.doPlay = True
            self.playButton.setIcon(self.pauseIcon)
            
        self.spriteToSpawnIdx = random.choice(arange(len(window.trackedSprites)))
        self.spawnSprite()
        
#         self.renderOneFrame()
        
#         if len(self.precomputedSequence) > 0 :
#             self.extendFullSequence(self.EXTEND_LENGTH, len(self.precomputedSequence[0][DICT_SEQUENCE_FRAMES])-1)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frameLabel.installEventFilter(self)
        self.frameLabel.installEventFilter(self)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.playButton = QtGui.QToolButton()
        self.playButton.setToolTip("Play Generated Sequence")
        self.playButton.setCheckable(False)
        self.playButton.setShortcut(QtGui.QKeySequence("Alt+P"))
        self.playButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        self.playButton.setIcon(self.playIcon)        
        
        ## SIGNALS ##
        
        self.playButton.clicked.connect(self.playButtonPressed)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        
        frameHLayout = QtGui.QHBoxLayout()
        frameHLayout.addStretch()
        frameHLayout.addWidget(self.frameLabel)
        frameHLayout.addStretch()
        
        frameVLayout = QtGui.QVBoxLayout()
        frameVLayout.addStretch()
        frameVLayout.addLayout(frameHLayout)
        frameVLayout.addStretch()
        frameVLayout.addWidget(self.playButton)
#         frameVLayout.addLayout(sliderLayout)
        frameVLayout.addWidget(self.frameInfo)
        
#         mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameVLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

print window.precomputedSequence[]

# <codecell>

print len(window.trackedSprites) if False else 15

# <codecell>

img = cv2.imread(dataPath + dataSet + 'median.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

# <codecell>

print contours[0]

# <codecell>

print bbox.reshape((5, 1, 2))

# <codecell>

bbox = window.trackedSprites[0][DICT_BBOXES][2000]
bbox = np.vstack((bbox, bbox[0, :]))
figure()
plot(bbox[:, 0], bbox[:, 1])
point = np.array([1000.0, 461, 0])
scatter(point[0], point[1])
print cv2.pointPolygonTest(np.array(bbox, dtype=np.float32), (1000, 461), False)

