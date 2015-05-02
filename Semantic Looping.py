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
#     semanticsCosts = vectorisedMinusLogMultiNormal(semanticLabels, np.array(desiredSemantics).reshape((1, numSemantics)), np.eye(numSemantics)*distVariance, True)
    semanticsCosts = np.zeros((k, desiredSemantics.shape[0]))
    for i in xrange(desiredSemantics.shape[0]) :
        semanticsCosts[:, i] = vectorisedMinusLogMultiNormal(semanticLabels, desiredSemantics[i, :].reshape((1, numSemantics)), np.eye(numSemantics)*distVariance, True)
    
    if desiredSemantics.shape[0] < sequenceLength :
        print "lalala"
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

print toggleLabelsSmoothly(np.array([[0.0, 1.0]]), 4)

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

tic = time.time()
unaries, pairwise = getMRFCosts(semanticLabels, desiredLabel, 0, 300)
print "computed costs in", time.time() - tic; sys.stdout.flush()
tic = time.time()
gwv.showCustomGraph(unaries[1:, :], title="unaries")
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
                spriteIdx = self.generatedSequence[s][DICT_SPRITE_IDX]
                ## index of current sprite frame to visualize
                sequenceFrameIdx = self.generatedSequence[s][DICT_SEQUENCE_FRAMES][self.frameIdx]-1
                ## -1 stands for not shown or eventually for more complicated sprites as the base frame to keep showing when sprite is frozen
                ## really in the sequence I have 0 for the static frame but above I do -1
                if sequenceFrameIdx >= 0 :
                    ## the trackedSprites data is indexed (i.e. the keys) by the frame indices in the original full sequence and keys are not sorted
                    frameToShowIdx = np.sort(self.trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())[sequenceFrameIdx]
                    
                    self.drawOverlay(self.trackedSprites[spriteIdx], frameToShowIdx, self.drawSpritesBox.isChecked(), 
                                     self.drawBBoxBox.isChecked(), self.drawCenterBox.isChecked())
            
            self.frameLabel.setFixedSize(self.overlayImg.width(), self.overlayImg.height())
            self.frameLabel.setOverlay(self.overlayImg)
            
    def drawOverlay(self, sprite, frameIdx, doDrawSprite, doDrawBBox, doDrawCenter) :
        if self.overlayImg != None :            
            painter = QtGui.QPainter(self.overlayImg)
            
            ## draw sprite
            if doDrawSprite :
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
        ## saving sequence
        if self.autoSaveBox.isChecked() and len(self.generatedSequence) > 0 :
            np.save(dataPath + dataSet + "generatedSequence-" + datetime.datetime.now().strftime('%Y-%M-%d_%H:%M:%S'), self.generatedSequence)
            
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
            
    def keyPressEvent(self, e) :
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            pressedIdx = np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9))
            print "pressed key", pressedIdx,
            if pressedIdx >= 0 and pressedIdx < len(self.trackedSprites) :
                print "i.e. sprite", self.trackedSprites[pressedIdx][DICT_SPRITE_NAME]
#                 self.toggleSpriteSemantics(pressedIdx)
                self.addNewSpriteTrackToSequence(pressedIdx)
    
                if len(self.generatedSequence) > 0 :
                    extendMode = {}
                    extendMode[len(self.generatedSequence)-1] = self.DO_BURST
                    self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), self.frameIdx)
                    
                    additionalFrames = len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES]) - len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])
                    if additionalFrames < 0 :
                        ## extend new sprite's sequence to match total sequence's length
                        print "extending new sprite's sequence"
                        extendMode = {}
                        extendMode[len(self.generatedSequence)-1] = self.DO_EXTEND
                        self.extendSequence(self.extendSequenceTracksSemantics(-additionalFrames+1, extendMode), len(self.generatedSequence[-1][DICT_SEQUENCE_FRAMES])-1)
                    elif additionalFrames > 0 :
                        ## extend existing sprites' sequences to match the new total sequence's length because of newly added sprite
                        print "extending existing sprites' sequences"
                        extendMode = {}
                        for i in xrange(len(self.generatedSequence)-1) :
                            extendMode[i] = self.DO_EXTEND
                        self.extendSequence(self.extendSequenceTracksSemantics(additionalFrames+1, extendMode), len(self.generatedSequence[0][DICT_SEQUENCE_FRAMES])-1)
                        
                self.showFrame(self.frameIdx)                        
            else :
                print
        elif e.key() == QtCore.Qt.Key_Space :
            extendMode = {}
            for i in xrange(len(self.generatedSequence)) :
                extendMode[i] = self.DO_EXTEND
            if len(self.generatedSequence) > 0 :
                self.extendSequence(self.extendSequenceTracksSemantics(self.EXTEND_LENGTH, extendMode), self.frameIdx)
            
        sys.stdout.flush()
            
    def extendSequence(self, desiredSemantics, startingFrame) :
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
                minCostTraversal = solveMRF(unaries, pairwise)
                
                ## update dictionary
                # don't take the first frame of the minCostTraversal as it would just repeat the last seen frame
                self.generatedSequence[i][DICT_SEQUENCE_FRAMES] = np.hstack((self.generatedSequence[i][DICT_SEQUENCE_FRAMES][:startingFrame+1], minCostTraversal[1:]))
                self.generatedSequence[i][DICT_DESIRED_SEMANTICS] = np.vstack((self.generatedSequence[i][DICT_DESIRED_SEMANTICS][:startingFrame+1],
                                                                               desiredSemantics[i][1:, :]))
                
                ## update sliders
                self.frameIdxSlider.setMaximum(len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES])-1)
                self.frameIdxSpinBox.setRange(0, len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES])-1)
                
                self.frameInfo.setText("Generated sequence length: " + np.string_(len(self.generatedSequence[i][DICT_SEQUENCE_FRAMES])))
                    
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
                                               DICT_DESIRED_SEMANTICS:np.empty((0, 2), dtype=float)#[],
#                                                DICT_FRAME_SEMANTIC_TOGGLE:[]
                                              })
                
                desiredSpriteIdx = len(self.generatedSequence)-1
            
            if desiredSpriteIdx >= 0 and desiredSpriteIdx < len(self.generatedSequence) :
                ## set the desired semantics
                if len(self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS]) == 0 :
                    ## start by asking to show the sprite
                    desiredSemantics = np.array([0.0, 1.0]).reshape((1, 2)).repeat(600, axis=0)
                else :
                    ## slide last requested semantics by 1 to toggle the opposite semantic label
                    desiredSemantics = np.roll(self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS][-1], 1).reshape((1, 2)).repeat(600, axis=0)
                    
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
#                 self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS].append(desiredSemantics)
                self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS] = np.vstack((self.generatedSequence[desiredSpriteIdx][DICT_DESIRED_SEMANTICS][:self.frameIdx+1],
                                                                                              desiredSemantics[1:, :]))
#                 self.generatedSequence[desiredSpriteIdx][DICT_FRAME_SEMANTIC_TOGGLE].append(self.frameIdx)
                # don't take the first frame of the minCostTraversal as it would just repeat the last seen frame
                self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES] = np.hstack((self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES][:self.frameIdx+1], 
                                                                                            self.minCostTraversal[1:]))
                
                ## update sliders
                self.frameIdxSlider.setMaximum(len(self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES])-1)
                self.frameIdxSpinBox.setRange(0, len(self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES])-1)
                
                self.frameInfo.setText("Generated sequence length: " + np.string_(len(self.generatedSequence[desiredSpriteIdx][DICT_SEQUENCE_FRAMES])))
                
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

