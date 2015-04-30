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
    
    pairwise = timeConstraint
    
    return unaries.T, pairwise.T

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
    dynProg.infer()
    
    labels = np.array(dynProg.arg(), dtype=int)
    
    return labels

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
desiredLabel = np.array([0.0, 1.0])
# semanticDist = np.sum(np.power(semanticLabels-desiredLabel, 2), axis=-1)

tic = time.time()
unaries, pairwise = getMRFCosts(semanticLabels, desiredLabel, 0, 600)
print "computed costs in", time.time() - tic; sys.stdout.flush()
tic = time.time()
gwv.showCustomGraph(unaries, title="unaries")
minCostTraversal = solveMRF(unaries, pairwise)
print "solved in", time.time() - tic; sys.stdout.flush()
tic = time.time()
print minCostTraversal

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
        
        self.createGUI()
        
        self.setWindowTitle("Looping the Unloopable")
        self.resize(1280, 720)
        
        self.isScribbling = False
        
        self.prevPoint = None
        
        self.trackedSprites = []
        self.currentSpriteIdx = -1
        
        self.frameIdx = -1
#         self.frameImg = None
        self.overlayImg = QtGui.QImage(QtCore.QSize(100, 100), QtGui.QImage.Format_ARGB32)
#         self.showFrame(self.frameIdx)
        
        im = np.ascontiguousarray(Image.open(dataPath + dataSet + "median.png"))
        self.bgImage = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        
        self.frameLabel.setFixedSize(self.bgImage.width(), self.bgImage.height())
        self.frameLabel.setImage(self.bgImage)
        
        self.loadTrackedSprites()
        
        self.generatedSequence = []
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/60)
        self.playTimer.start()
        self.playTimer.timeout.connect(self.renderOneFrame)
        
        self.setFocus()
        
    def renderOneFrame(self) :
        idx = self.frameIdx + 1
        if idx >= 0 and len(self.generatedSequence) > 0 : #idx < len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) :
            self.showFrame(np.mod(idx, len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])))
    
    def showFrame(self, idx) :
        if idx >= 0 and len(self.generatedSequence) > 0 and idx < len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES]) :
            self.frameIdx = idx
            ## HACK ##
    #         im = np.ascontiguousarray(Image.open((dataPath+dataSet+formatString).format(self.frameIdx+1)))
            spriteIdx = self.generatedSequence[0][DICT_SPRITE_IDX]
            frameToShowIdx = self.generatedSequence[0][DICT_SEQUENCE_FRAMES][self.frameIdx]-1
            tmp = np.sort(self.trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())[frameToShowIdx]
            
            im = np.ascontiguousarray(Image.open(self.trackedSprites[0][DICT_FRAMES_LOCATIONS][tmp]))
            self.frameImg = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            
            self.frameLabel.setFixedSize(self.frameImg.width(), self.frameImg.height())
            self.frameLabel.setImage(self.frameImg)
            
#             self.frameInfo.setText(frameLocs[self.frameIdx])
            
#             if self.currentSpriteIdx < len(self.trackedSprites) and self.currentSpriteIdx >= 0 :
#                 ## set self.bbox to bbox computed for current frame if it exists
#                 if not self.tracking :
#                     if self.frameIdx in self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES].keys() :
#                         self.bbox[TL_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TL_IDX, 0])
#                         self.bbox[TL_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TL_IDX, 1])
#                         self.bbox[TR_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TR_IDX, 0])
#                         self.bbox[TR_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TR_IDX, 1])
#                         self.bbox[BR_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BR_IDX, 0])
#                         self.bbox[BR_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BR_IDX, 1])
#                         self.bbox[BL_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BL_IDX, 0])
#                         self.bbox[BL_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BL_IDX, 1])
#                         self.bbox[-1] = self.bbox[TL_IDX]
                        
#                         self.centerPoint.setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx][0])
#                         self.centerPoint.setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx][1])
                        
#                         if self.drawOverlay(False) :
#                             self.frameLabel.setOverlay(self.overlayImg)
                            
#                         self.bboxIsSet = True
#                     else :
#                         if self.drawOverlay(False, False, False) :
#                             self.frameLabel.setOverlay(self.overlayImg)
            
    def updateBBox(self) :
        if self.settingBBox :
#             print "settingBBox"
            self.bbox[TR_IDX] = QtCore.QPointF(self.bbox[BR_IDX].x(), self.bbox[TL_IDX].y())
            self.bbox[BL_IDX] = QtCore.QPointF(self.bbox[TL_IDX].x(), self.bbox[BR_IDX].y())
            self.bbox[-1] = self.bbox[TL_IDX]
            
            tl = self.bbox[TL_IDX]
            br = self.bbox[BR_IDX]
            self.centerPoint = QtCore.QPointF(min((tl.x(), br.x())) + (max((tl.x(), br.x())) - min((tl.x(), br.x())))/2.0, 
                                              min((tl.y(), br.y())) + (max((tl.y(), br.y())) - min((tl.y(), br.y())))/2.0)
            
            if self.drawOverlay(False) :
                self.frameLabel.setOverlay(self.overlayImg)
            
    def drawOverlay(self, doDrawFeats = True, doDrawBBox = True, doDrawCenter = True) :
        if self.frameImg != None :
            if self.overlayImg.size() != self.frameImg.size() :
                self.overlayImg = self.overlayImg.scaled(self.frameImg.size())
            
            ## empty image
            self.overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
            
            painter = QtGui.QPainter(self.overlayImg)
            
            ## draw bbox
            if doDrawBBox :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 3, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                for p1, p2 in zip(self.bbox[0:-1], self.bbox[1:]) :
                    painter.drawLine(p1, p2)
            
            ## draw bbox center
            if doDrawCenter :
                painter.drawPoint(self.centerPoint)
            
            ## draw tracked features
            if doDrawFeats :
                if self.tracker != None and self.tracker.has_result :
                    
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for point in self.tracker.tracked_keypoints[:, 0:-1] :
                        painter.drawEllipse(QtCore.QPointF(point[0], point[1]), 3, 3)
                        
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for point in self.tracker.votes[:, :2] :
                        painter.drawEllipse(QtCore.QPointF(point[0], point[1]), 3, 3)
                    
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 0, 0, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    for point in self.tracker.outliers[:, :2] :
                        painter.drawEllipse(QtCore.QPointF(point[0], point[1]), 3, 3)
                    
            return True
        else :
            return False
            
    def changeSprite(self, row) :
#         print "changingSprite"
        if len(self.trackedSprites) > row :
            self.currentSpriteIdx = row
#             print "sprite: ", self.trackedSprites[self.currentSpriteIdx][DICT_SPRITE_NAME]
            sys.stdout.flush()
            ## go to the first frame that has been tracked
            if len(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES].keys()) > 0 :
                self.frameIdxSpinBox.setValue(np.min(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES].keys()))
            ## go to the first frame in video
            else :
                self.frameIdxSpinBox.setValue(0)
            
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
#         self.saveTrackedSprites()
    
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
            
    def keyPressEvent(self, e) :
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            pressedIdx = np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9))
            print "pressed key", pressedIdx,
            if pressedIdx >= 0 and pressedIdx < len(self.trackedSprites) :
                print "i.e. sprite", self.trackedSprites[pressedIdx][DICT_SPRITE_NAME]
                self.toggleSpriteSemantics(pressedIdx)
            else :
                print
            
        sys.stdout.flush()
        
    def toggleSpriteSemantics(self, spriteIdx) :
        if spriteIdx >= 0 and spriteIdx < len(self.trackedSprites) :
            ## for now only allow one instance for each sprite
            # if the desired sprite has not been added to the sequence yet
            isInSequence = False
            desiredSpriteIdx = -1
            for i in xrange(len(self.generatedSequence)) :
                if spriteIdx == self.generatedSequence[i][DICT_SPRITE_IDX] :
                    isInSequence = True
                    desiredSpriteIdx = i
                    break
            if not isInSequence :
                print "adding new sprite to sequence"
                self.generatedSequence.append({
                                               DICT_SPRITE_IDX:spriteIdx,
                                               DICT_SPRITE_NAME:self.trackedSprites[spriteIdx][DICT_SPRITE_NAME],
                                               DICT_SEQUENCE_FRAMES:np.empty(0, dtype=int),
                                               DICT_DESIRED_SEMANTICS:[],
                                               DICT_FRAME_SEMANTIC_TOGGLE:[]
                                              })
                
                desiredSpriteIdx = len(self.generatedSequence)-1
            
            if desiredSpriteIdx >= 0 and desiredSpriteIdx < len(self.generatedSequence) :
                ## set the desired semantics
                if len(self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS]) == 0 :
                    ## start by asking to show the sprite
                    desiredSemantics = np.array([0.0, 1.0])
                else :
                    ## slide last requested semantics by 1 to toggle the opposite semantic label
                    desiredSemantics = np.roll(self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS][-1], 1)
                    
                ## set starting frame
                if len(self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES]) == 0 :
                    desiredStartFrame = 0
                else :
                    desiredStartFrame = self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES][self.frameIdx]
                    
                print desiredSemantics, desiredStartFrame
                
                self.semanticLabels = np.zeros((len(self.trackedSprites[spriteIdx][DICT_BBOX_CENTERS])+1, 2))
                ## label 0 means sprite not visible (i.e. only show the first empty frame)
                self.semanticLabels[0, 0] = 1.0
                self.semanticLabels[1:, 1] = 1.0
                
                self.unaries, self.pairwise = getMRFCosts(self.semanticLabels, desiredSemantics, desiredStartFrame, 600)
                self.minCostTraversal = solveMRF(self.unaries, self.pairwise)
                
                ## update dictionary
                self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS].append(desiredSemantics)
                self.generatedSequence[desiredSpriteIdx][DICT_FRAME_SEMANTIC_TOGGLE].append(self.frameIdx)
                # don't take the first frame of the minCostTraversal as it would just repeat the last seen frame
                self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES] = np.hstack((self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES][:self.frameIdx], 
                                                                                            self.minCostTraversal[1:]))
                
                print self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES]
                
        
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
        self.frameIdxSlider.setTickInterval(100)
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
        
#         self.newSpriteButton = QtGui.QPushButton("&New Sprite")
        
#         self.deleteCurrentSpriteBBoxButton = QtGui.QPushButton("Delete BBox")
#         self.setCurrentSpriteBBoxButton = QtGui.QPushButton("Set BBox")
        
        
        ## SIGNALS ##
        
        self.frameIdxSlider.valueChanged[int].connect(self.frameIdxSpinBox.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.frameIdxSlider.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.showFrame)
        
        self.spriteListTable.currentCellChanged.connect(self.changeSprite)
        self.spriteListTable.cellPressed.connect(self.changeSprite)
        
#         self.newSpriteButton.clicked.connect(self.createNewSprite)
        
#         self.deleteCurrentSpriteBBoxButton.clicked.connect(self.deleteCurrentSpriteFrameBBox)
#         self.setCurrentSpriteBBoxButton.clicked.connect(self.setCurrentSpriteFrameBBox)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.spriteListTable)
#         controlsLayout.addWidget(self.newSpriteButton)
#         controlsLayout.addWidget(self.deleteCurrentSpriteBBoxButton)
#         controlsLayout.addWidget(self.setCurrentSpriteBBoxButton)
        
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
        frameVLayout.addWidget(self.frameInfo)
        frameVLayout.addStretch()
        frameVLayout.addLayout(sliderLayout)
        
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameVLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_() 

# <codecell>

np.array([0.0, 1.0])
print np.roll(np.array([0.0, 1.0]), 1)
print np.roll(np.array([1.0, 0.0]), 1)

# <codecell>

print minCostTraversal
bob = np.empty(0, dtype=int)
bob = np.hstack((bob[:0], minCostTraversal))
bob = np.concatenate((bob, minCostTraversal))
print bob

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
    minCostPaths = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]])
    # contains the min cost to reach a certain state k (i.e. row) for point n (i.e. column)
    minCosts = np.zeros([unaryCosts.shape[0], unaryCosts.shape[1]])
    # the first row of minCosts is just the unary cost
    minCosts[:, 0] = unaryCosts[:, 0]
    minCostPaths[:, 0] = np.arange(0, unaryCosts.shape[0])
    tmp = 0
    
    k = unaryCosts.shape[0]
    for n in xrange(1, unaryCosts.shape[1]) :
        costs = minCosts[:, n-1].reshape((k, 1)).repeat(k, axis=-1) + unaryCosts[:, n].reshape((1, k)).repeat(k, axis=0) + pairwiseCosts[:, :, n-1].T
        minCosts[:, n] = np.min(costs, axis=0)
        minCostPaths[:, n] = np.ndarray.flatten(np.argmin(costs, axis=0))
    
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
        
    print np.min(minCosts[:, -1])
    print minCostTraversal
    
    return minCosts, minCostPaths, minCostTraversal, tmp

