# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import sys
import scipy as sp
from IPython.display import clear_output

import cv2
import time
import os
import scipy.io as sio
import glob
import itertools

from PIL import Image
from PySide import QtCore, QtGui

import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import SemanticsDefinitionTabGUI as sdt
import SemanticLoopingTabGUI as slt
import opengm
import soundfile as sf

from matplotlib.patches import Rectangle

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

app = QtGui.QApplication(sys.argv)

DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SEQUENCE_IDX = 'semantic_sequence_idx' # index of the instantiated sem sequence in the list of all used sem sequences for a synthesised sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_OFFSET = "instance_offset"
DICT_SCALE = "instance_scale"
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_USED_SEQUENCES = "used_semantic_sequences"
DICT_SEQUENCE_INSTANCES = "sequence_instances"
DICT_SEQUENCE_BG = "sequence_background_image"
DICT_SEQUENCE_LOCATION = "sequence_location"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"

GRAPH_MAX_COST = 10000000.0

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

# <codecell>

# for sprite in np.sort(glob.glob(dataPath + dataSet + "semantic_sequence*.npy"))[0:] :
#     ## for a given semantic sequence, load all frames and precompute visible patches (for fast rendering)
#     seqLoc = dataPath + dataSet
#     sequence = np.load(sprite).item()
#     seqName = sequence[DICT_SEQUENCE_NAME]
# #     sequence[DICT_MASK_LOCATION] = seqLoc+seqName+"-maskedFlow-blended/"
# #     currentSequencePatches = {}
# #     for frameKey in np.sort(sequence[DICT_FRAMES_LOCATIONS].keys()) :
# #         frameName = sequence[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
# #         maskDir = sequence[DICT_MASK_LOCATION]

# #         if os.path.isdir(maskDir) and os.path.exists(maskDir+"/"+frameName) :
# #             im = np.array(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED), dtype=np.uint8)

# #             visiblePixels = np.argwhere(im[:, :, -1] != 0)
# #             topLeft = np.min(visiblePixels, axis=0)
# #             patchSize = np.max(visiblePixels, axis=0) - topLeft + 1

# #             currentSequencePatches[frameKey] = {'top_left_pos':topLeft, 'sprite_colors':im[visiblePixels[:, 0], visiblePixels[:, 1], :], 
# #                                                'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}

# #         sys.stdout.write('\r' + "Loaded image " + np.string_(len(currentSequencePatches)) + " (" + np.string_(len(sequence[DICT_FRAMES_LOCATIONS])) + ")")
# #         sys.stdout.flush()

# #     sequence[DICT_PATCHES_LOCATION] = seqLoc+"preloaded_patches-"+seqName+".npy"
# #     np.save(sequence[DICT_PATCHES_LOCATION], currentSequencePatches)
# #     np.save(seqLoc+"semantic_sequence-"+seqName+".npy", sequence)
# #     print 

#     ## "computing" the hacky transitino matrix for the car sprites --> assuming the invisible frame is the last one
#     k = len(sequence[DICT_FRAME_SEMANTICS])
#     timeConstraint = np.eye(k, k=1)
#     timeConstraint[-1, -1] = 1.0 ## the last one can stay in place
#     timeConstraint[-1, 0] = 1.0 ## the last one can go to the first one
#     timeConstraint[-2, 0] = 1.0 ## the second to last can go to the first one
#     timeConstraint = (1.0 - timeConstraint)*maxCost
# #     gwv.showCustomGraph(timeConstraint)
# #     sequence[DICT_TRANSITION_COSTS_LOCATION] = seqLoc+"transition_costs-"+seqName+".npy"
# #     np.save(sequence[DICT_TRANSITION_COSTS_LOCATION], timeConstraint)
# #     np.save(seqLoc+"semantic_sequence-"+seqName+".npy", sequence)


#     print sequence[DICT_SEQUENCE_NAME]
#     print sequence.keys()
#     print len(sequence[DICT_BBOX_CENTERS]), len(sequence[DICT_FOOTPRINTS]), len(sequence[DICT_BBOX_ROTATIONS]), len(sequence[DICT_FRAME_SEMANTICS]),
#     print len(sequence[DICT_BBOXES]), len(sequence[DICT_FRAMES_LOCATIONS])
# #     np.save(sprite, sequence)

# <codecell>

# for seq in np.sort(glob.glob(dataPath + dataSet + "generatedSequence*.npy"))[0:] :
#     s = np.load(seq)
#     for i in xrange(len(s)) :
#         s[i][DICT_SEQUENCE_NAME] = s[i]["sprite_name"]
#         del s[i]["sprite_name"]
#         s[i][DICT_SEQUENCE_IDX] = s[i]["sprite_idx"]
#         del s[i]["sprite_idx"]
#     print len(s), seq
# #     np.save(seq, s)
# #     s[DICT_SEQUENCE_NAME] = s["sprite_name"]
# #     del s["sprite_name"]
# #     np.save(sprite, s)
# #     print s[DICT_SEQUENCE_NAME]
# #     s[DICT_SEQUENCE_LOCATION] = dataPath+dataSet+"semantic_sequence-"+s[DICT_SEQUENCE_NAME]+".npy"
# #     print sprite, s[DICT_SEQUENCE_NAME]
# #     print s.keys()
# #     print s[DICT_SEQUENCE_LOCATION]
# #     print len(s[DICT_BBOX_CENTERS])
# #     np.save(s[DICT_SEQUENCE_LOCATION], s)

# <codecell>

# #### computes inter sprite compatibilities as bbox distances
# baseLoc = "/home/ilisescu/PhD/data/synthesisedSequences/newHavana/"
# synthSequence = np.load(baseLoc+"synthesised_sequence.npy").item()

# for seq1Idx in xrange(len(synthSequence[DICT_USED_SEQUENCES])) :
#     for seq2Idx in xrange(seq1Idx, len(synthSequence[DICT_USED_SEQUENCES])) :
#         seq1 = np.load(synthSequence[DICT_USED_SEQUENCES][seq1Idx]).item()
#         seq2 = np.load(synthSequence[DICT_USED_SEQUENCES][seq2Idx]).item()
        
# #         print seq1[DICT_SEQUENCE_NAME], seq2[DICT_SEQUENCE_NAME]
#         print "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy"
        
        

#         bboxDistance = np.zeros((len(seq1[DICT_BBOXES]), len(seq2[DICT_BBOXES])))
#         avgTime = 0.0
#         for i, iKey in enumerate(np.sort(seq1[DICT_BBOXES].keys())[0:]) :
#             t = time.time()
#             if iKey not in seq1[DICT_FRAMES_LOCATIONS] :
#                 bboxDistance[i, :] = 10000.0
#             else :
#                 for j, jKey in enumerate(np.sort(seq2[DICT_BBOXES].keys())[0:]) :
#                     if jKey not in seq2[DICT_FRAMES_LOCATIONS] :
#                         bboxDistance[i, j] = 10000.0
#                     else :
#                         bboxDistance[i, j] = getSpritesBBoxDist(seq1[DICT_BBOX_ROTATIONS][iKey],
#                                                                 seq1[DICT_BBOXES][iKey],
#                                                                 seq2[DICT_BBOXES][jKey])

#             avgTime = (avgTime*i + time.time()-t)/(i+1)
#             remainingTime = avgTime*(len(seq1[DICT_BBOXES])-i-1)/60.0

#             if np.mod(i, 5) == 0 :
#                 sys.stdout.write('\r' + "Done row " + np.string_(i) + " of " + np.string_(len(seq1[DICT_BBOXES])) +
#                                  " (avg time: " + np.string_(avgTime) + " secs --- remaining: " +
#                                  np.string_(int(np.floor(remainingTime))) + ":" + np.string_(int((remainingTime - np.floor(remainingTime))*60)) + ")")
#                 sys.stdout.flush()
#         print        
#         np.save(baseLoc + "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy", bboxDistance)

# <codecell>

class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.showLoading(False)
        
        self.setWindowTitle("Looping the Unloopable")
        self.resize(1920, 950)
        
        self.readyForVT = False
        self.firstLoad = True
        self.dataLocation = ""
    
    def openSequence(self) :
        return 
        
    def tabChanged(self, tabIdx) :
        if tabIdx == 0 :
            self.semanticsDefinitionTab.setFocus()
        elif tabIdx == 1 :
            self.semanticLoopingTab.setFocus()

    def closeEvent(self, event) :
        self.semanticLoopingTab.cleanup()
            
    def showLoading(self, show) :
        if show :
            self.loadingLabel.setText("Loading... Please wait")
            self.loadingWidget.setVisible(True)
            self.infoLabel.setVisible(False)
        else :
            self.loadingWidget.setVisible(False)
            self.infoLabel.setVisible(True)
            
#     def lockGUI(self, lock):
        
#         self.openVideoButton.setEnabled(False)#not lock)
#         self.openSequenceButton.setEnabled(not lock)
        
#         if self.readyForVT :
#             self.videoTexturesTab.lockGUI(lock)
#         else :
#             self.videoTexturesTab.lockGUI(True)
#             if self.tabWidget.currentIndex() == 1 and not self.firstLoad :
#                 QtGui.QMessageBox.warning(self, "Pre-processing not ready",
#                         "<p align='center'>The pre-processing step has not been completed<br>"
#                         "Please return to the pre-processing tab and compute a distance matrix</p>")
#                 self.tabWidget.setCurrentIndex(0)
            
#         self.preProcessingTab.lockGUI(lock)
#         self.labellingTab.lockGUI(lock)
        
    def createGUI(self) :
        
        ## WIDGETS ##

        self.infoLabel = QtGui.QLabel("No data loaded")
        self.infoLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.infoLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
#         self.openVideoButton = QtGui.QPushButton("Open &Video")
#         self.openVideoButton.setEnabled(False)
        self.openSequenceButton = QtGui.QPushButton("Open &Sequence")
        
        self.loadingLabel = QtGui.QLabel("Loading... Please wait!")
        self.loadingLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.loadingLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
        movie = QtGui.QMovie("loader.gif")
        self.loadingSpinner = QtGui.QLabel()
        self.loadingSpinner.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.loadingSpinner.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.loadingSpinner.setMovie(movie)
        movie.start()
        
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab(self, "/media/ilisescu/Data1/PhD/data/theme_park_sunny/")#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab(self, "/media/ilisescu/Data1/PhD/data/windows/")#dataPath+dataSet)
        self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab(self, "/media/ilisescu/Data1/PhD/data/digger/")#dataPath+dataSet)
        
#         self.semanticLoopingTab = SemanticLoopingTab(self, 100, dataPath+"synthesisedSequences/waveFull/synthesised_sequence.npy")
#         self.semanticLoopingTab = SemanticLoopingTab(self, 100, dataPath+"synthesisedSequences/waveFullBusier/synthesised_sequence.npy")
#         self.semanticLoopingTab = SemanticLoopingTab(self, 250, dataPath+"synthesisedSequences/theme_park/synthesised_sequence.npy")
#         self.semanticLoopingTab = SemanticLoopingTab(self, 250, dataPath+"synthesisedSequences/theme_park_mixedCompatibility/synthesised_sequence.npy")
#         self.semanticLoopingTab = SemanticLoopingTab(self, 250, dataPath+"synthesisedSequences/tetris/synthesised_sequence.npy")
        self.semanticLoopingTab = slt.SemanticLoopingTab(self, 500, dataPath+"synthesisedSequences/havana_new_semantics/synthesised_sequence.npy")
#         self.semanticLoopingTab = SemanticLoopingTab(self, 100, dataPath+"synthesisedSequences/multipleCandles/synthesised_sequence.npy")

        self.tabWidget = QtGui.QTabWidget()
        self.tabWidget.addTab(self.semanticsDefinitionTab, self.tr("Define Semantics"))
        self.tabWidget.addTab(self.semanticLoopingTab, self.tr("Loop Semantics"))
        
        ## SIGNALS ##
        
        self.openSequenceButton.clicked.connect(self.openSequence)
        
        self.tabWidget.currentChanged.connect(self.tabChanged)
        
        ## LAYOUTS ##
        
        mainBox = QtGui.QGroupBox("Main Controls")
        mainBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        mainBoxLayout = QtGui.QHBoxLayout()
        
        self.loadingWidget = QtGui.QWidget()
        loadingLayout = QtGui.QHBoxLayout()
        loadingLayout.addWidget(self.loadingSpinner)
        loadingLayout.addWidget(self.loadingLabel)
        self.loadingWidget.setLayout(loadingLayout)
        
        mainBoxLayout.addWidget(self.loadingWidget)
        mainBoxLayout.addWidget(self.infoLabel)
        mainBoxLayout.addStretch()
        
        buttonLayout = QtGui.QVBoxLayout()
        buttonLayout.addWidget(self.openSequenceButton)
        
        mainBoxLayout.addLayout(buttonLayout)
        mainBox.setLayout(mainBoxLayout)
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.tabWidget)
        mainLayout.addWidget(mainBox)
        self.setLayout(mainLayout)

# <codecell>

# compatibilityMats = {}
# compatibilityMats["00"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--black_car1.npy")/50)*10.0
# compatibilityMats["01"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--blue_car1.npy")/50)*10.0
# compatibilityMats["11"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-blue_car1--blue_car1.npy")/50)*10.0

# <codecell>

# synthSeq = np.load(dataPath+"synthesisedSequences/havanaComplex/synthesised_sequence.npy").item()
# synthSeq = np.load(dataPath+"synthesisedSequences/theme_park/synthesised_sequence.npy").item()
synthSeq = np.load(dataPath+"synthesisedSequences/theme_park_mixedCompatibility/synthesised_sequence.npy").item()
usedSequences = synthSeq[DICT_USED_SEQUENCES]
semanticSequences = []
for sequence in usedSequences :
    semanticSequences.append(np.load(sequence).item())

compatibilityMats = {}
baseLoc = "/".join(semanticSequences[0][DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"
print baseLoc
for i in xrange(len(semanticSequences)) :
    for j in xrange(i, len(semanticSequences)) :
        print 
        print np.string_(i)+"-"+np.string_(j), semanticSequences[i][DICT_SEQUENCE_NAME], semanticSequences[j][DICT_SEQUENCE_NAME]
        correctLoc = baseLoc+"inter_sequence_compatibility-bbox_dist-"+semanticSequences[i][DICT_SEQUENCE_NAME]+"--"+semanticSequences[j][DICT_SEQUENCE_NAME]+".npy"
        transposedLoc = baseLoc+"inter_sequence_compatibility-bbox_dist-"+semanticSequences[j][DICT_SEQUENCE_NAME]+"--"+semanticSequences[i][DICT_SEQUENCE_NAME]+".npy"
        
#         print correctLoc
#         print transposedLoc
        if os.path.isfile(correctLoc) :
            print "using correct", correctLoc
            compatibilityMats[np.string_(i)+"-"+np.string_(j)] = np.exp(-np.load(correctLoc)/50)*10.0
            continue
        print "skipping correct"
        
        if os.path.isfile(transposedLoc) :
            print "using transposed", transposedLoc
            compatibilityMats[np.string_(i)+"-"+np.string_(j)] = np.exp(-np.load(transposedLoc).T/50)*10.0
            continue
            
        print "OOOOPS"

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

with open("/home/ilisescu/PhD/data/havana/preloaded_patches-blue_car2.npy") as f :
    tmp = np.load(f)
#     print tmp.item().keys()

# <codecell>

synthSeq = np.load("E:/PhD/data/synthesisedSequences/tetris/synthesised_sequence.npy").item()
usedSequences = synthSeq[DICT_USED_SEQUENCES]
semanticSequences = []
for sequence in usedSequences :
    if ON_WINDOWS : 
        sequence = "E:/" + "/".join(sequence.split("/")[4:])
    semanticSequences.append(np.load(sequence).item())
#     print semanticSequences[-1][DICT_SEQUENCE_NAME]

overlayImg = QtGui.QImage(QtCore.QSize(1280, 720), QtGui.QImage.Format_ARGB32)
overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))

painter = QtGui.QPainter(overlayImg)
painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)

for sequenceInstance in synthSeq[DICT_SEQUENCE_INSTANCES] :
    seqIdx = sequenceInstance[DICT_SEQUENCE_IDX]
    offset = sequenceInstance[DICT_OFFSET]
    scale = sequenceInstance[DICT_SCALE]
    sequence = semanticSequences[seqIdx]
    frameKey = sequence[DICT_BBOXES].keys()[0]
#     print sequence[DICT_BBOXES][frameKey], scale, offset    
                
    scaleTransf = np.array([[scale[0], 0.0], [0.0, scale[1]]])
    offsetTransf = np.array([[offset[0]], [offset[1]]])
    
    if offset[0] == 0 and offset[1] == 0 :
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, 255), 1, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    else :
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 1, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        
    bbox = sequence[DICT_BBOXES][frameKey]
    transformedBBox = (np.dot(scaleTransf, bbox.T) + offsetTransf)
    
    x, y = transformedBBox[:, 0]
    width, height = transformedBBox[:, 2] - transformedBBox[:, 0]
    painter.drawRoundedRect(x, y, width, height, 3, 3)

#     for p1, p2 in zip(np.mod(arange(4), 4), np.mod(arange(1, 5), 4)) :
#         painter.drawLine(QtCore.QPointF(transformedBBox[0, p1], transformedBBox[1, p1]), QtCore.QPointF(transformedBBox[0, p2], transformedBBox[1, p2]))
    
    
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), 1, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    painter.drawText(transformedBBox[0, 0]+5, transformedBBox[1, 0], 
                     transformedBBox[0, 2]-transformedBBox[0, 0]-5, 20, QtCore.Qt.AlignLeft, np.string_(seqIdx))

print overlayImg.save("C:/Users/ilisescu/Desktop/bboxes.png")
painter.end()
del painter

# <codecell>

paths = np.sort(glob.glob("E:/PhD/data/wave1/semantic_sequence-*.npy"))
semanticSequences = []
for sequence in paths :
    semanticSequences.append(np.load(sequence).item())
    print semanticSequences[-1][DICT_SEQUENCE_NAME]

overlayImg = QtGui.QImage(QtCore.QSize(1280, 720), QtGui.QImage.Format_ARGB32)
overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))

painter = QtGui.QPainter(overlayImg)
painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)

for seqIdx in xrange(len(semanticSequences)) :
    offset = np.array([0.0, 0.0])
    scale = np.array([1.0, 1.0])
    sequence = semanticSequences[seqIdx]
    frameKey = np.sort(sequence[DICT_BBOXES].keys())[0]
#     frameKey = np.sort(sequence[DICT_BBOXES].keys())[450]
#     if seqIdx == 7 :
#         frameKey = np.sort(sequence[DICT_BBOXES].keys())[-70]
#         print sequence[DICT_BBOXES][frameKey], scale, offset    
                
    scaleTransf = np.array([[scale[0], 0.0], [0.0, scale[1]]])
    offsetTransf = np.array([[offset[0]], [offset[1]]])
    
    if offset[0] == 0 and offset[1] == 0 :
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, 255), 3, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    else :
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 1, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        
    bbox = sequence[DICT_BBOXES][frameKey]
    transformedBBox = (np.dot(scaleTransf, bbox.T) + offsetTransf)
    
    x, y = transformedBBox[:, 0]
    width, height = transformedBBox[:, 2] - transformedBBox[:, 0]
    painter.drawRoundedRect(x, y, width, height, 3, 3)

#     for p1, p2 in zip(np.mod(arange(4), 4), np.mod(arange(1, 5), 4)) :
#         painter.drawLine(QtCore.QPointF(transformedBBox[0, p1], transformedBBox[1, p1]), QtCore.QPointF(transformedBBox[0, p2], transformedBBox[1, p2]))
    
    
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), 1, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    painter.drawText(transformedBBox[0, 0]+5, transformedBBox[1, 0], 
                     transformedBBox[0, 2]-transformedBBox[0, 0]-5, 20, QtCore.Qt.AlignLeft, np.string_(seqIdx))

print overlayImg.save("C:/Users/ilisescu/Desktop/bboxes.png")
painter.end()
del painter

# <codecell>

synthSeq = np.load(dataPath+"synthesisedSequences/digger/synthesised_sequence.npy").item()
usedSequences = synthSeq[DICT_USED_SEQUENCES]
semanticSequences = []
for sequence in usedSequences :
    semanticSequences.append(np.load(sequence).item())
    print semanticSequences[-1][DICT_SEQUENCE_NAME]
    
## 0 is compatible, 1 is incompatible
## digger semantics (scooping, dropping)
## truck semantics (moving, receiving dirt)
## digger is rows, truck is columns
## if digger is scooping, truck can move or receive
## if digger is dropping, truck can't move, just receive
hardcodedSemanticCompatibility = np.array([[0, 0], [1, 0]])
print hardcodedSemanticCompatibility
semanticCompatibilityCost = np.zeros((semanticSequences[0][DICT_FRAME_SEMANTICS].shape[0],
                                      semanticSequences[1][DICT_FRAME_SEMANTICS].shape[0]), bool)
thresh = 0.7
for incompatibleCombination in np.argwhere(hardcodedSemanticCompatibility == 1) :
    print incompatibleCombination
    seq1Sems = (semanticSequences[0][DICT_FRAME_SEMANTICS][:, incompatibleCombination[0]] > thresh).reshape((len(semanticSequences[0][DICT_FRAME_SEMANTICS]), 1))
    seq2Sems = (semanticSequences[1][DICT_FRAME_SEMANTICS][:, incompatibleCombination[1]] > thresh).reshape((1, len(semanticSequences[1][DICT_FRAME_SEMANTICS])))
    
    semanticCompatibilityCost = semanticCompatibilityCost | (seq1Sems & seq2Sems)
    

compatibilityMats = {}
compatibilityMats["0-0"] = np.zeros((len(semanticSequences[0][DICT_FRAME_SEMANTICS]), len(semanticSequences[0][DICT_FRAME_SEMANTICS])))
compatibilityMats["0-1"] = semanticCompatibilityCost*GRAPH_MAX_COST
compatibilityMats["1-1"] = np.zeros((len(semanticSequences[1][DICT_FRAME_SEMANTICS]), len(semanticSequences[1][DICT_FRAME_SEMANTICS])))

# <codecell>

gwv.showCustomGraph(compatibilityMats["0-1"])

# <codecell>

synthSeq = np.load(dataPath+"synthesisedSequences/multipleCandles/synthesised_sequence.npy").item()
usedSequences = synthSeq[DICT_USED_SEQUENCES]
semanticSequences = []
for sequence in usedSequences :
    semanticSequences.append(np.load(sequence).item())
    print semanticSequences[-1][DICT_SEQUENCE_NAME]
    
## 0 is compatible, 1 is incompatible
## candle has 3 labels, and two candles are compatible only if they show the same label
hardcodedSemanticCompatibility = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
print hardcodedSemanticCompatibility
semanticCompatibilityCost = np.zeros((semanticSequences[0][DICT_FRAME_SEMANTICS].shape[0],
                                      semanticSequences[0][DICT_FRAME_SEMANTICS].shape[0]), bool)
thresh = 0.9
for incompatibleCombination in np.argwhere(hardcodedSemanticCompatibility == 1) :
    print incompatibleCombination
    seq1Sems = (semanticSequences[0][DICT_FRAME_SEMANTICS][:, incompatibleCombination[0]] > thresh).reshape((len(semanticSequences[0][DICT_FRAME_SEMANTICS]), 1))
    seq2Sems = (semanticSequences[0][DICT_FRAME_SEMANTICS][:, incompatibleCombination[1]] > thresh).reshape((1, len(semanticSequences[0][DICT_FRAME_SEMANTICS])))
    
    semanticCompatibilityCost = semanticCompatibilityCost | (seq1Sems & seq2Sems)
    

compatibilityMats = {}
compatibilityMats["0-0"] = semanticCompatibilityCost*GRAPH_MAX_COST

# <codecell>

## newest compute transition costs
filterSize = 4
threshPercentile = 0.1 ## percentile of transitions to base threshold on
minJumpLength = 20
onlyBackwards = True ## indicates if only backward jumps need filtering out (i.e. the syntehsised sequence can be sped up but not slowed down)
loopOnLast = True ## indicates if an empty frame has been added at the end of the sequence that the synthesis can keep showing without concenquences
sigmaMultiplier = 0.002
for seqLoc in np.sort(glob.glob("/home/ilisescu/PhD/data/street/semantic_sequence*.npy"))[1:2] :
    testSequence = np.load(seqLoc).item()
    print testSequence[DICT_SEQUENCE_NAME]
    distMat = np.load(testSequence['sequence_precomputed_distance_matrix_location'])
#     gwv.showCustomGraph(distMat)
    
    ## filter to preserve dynamics
    kernel = np.eye(filterSize*2+1)
    
    optimizedDistMat = cv2.filter2D(distMat, -1, kernel)
    correction = 1
    
#     gwv.showCustomGraph(optimizedDistMat)
    
    ## init costs
#     testCosts = np.zeros_like(optimizedDistMat)
#     testCosts[0:-1, 0:-1] = np.copy(optimizedDistMat[1:, 0:-1])
#     testCosts[-1, 1:] = optimizedDistMat
    testCosts = np.copy(np.roll(optimizedDistMat, 1, axis=1))    
    
    # find threshold to use based on percentile
    thresh = np.sort(testCosts.flatten())[int(len(testCosts.flatten())*threshPercentile)]
    print "THRESH", thresh
    
    sigma = np.average(testCosts)*sigmaMultiplier
    
    ## don't want to jump too close so increase costs in a window
    if onlyBackwards :
        tmp = (np.triu(np.ones(optimizedDistMat.shape), k=2) +
               np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
               np.eye(optimizedDistMat.shape[0], k=1))
    else :
        tmp = (np.triu(np.ones(optimizedDistMat.shape), k=minJumpLength) +
               np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
               np.eye(optimizedDistMat.shape[0], k=1))
    tmp[tmp == 0] = 10.0
    testCosts *= tmp
    
    
    ## actual filtering
    invalidJumps = testCosts > thresh
    testCosts[invalidJumps] = GRAPH_MAX_COST
    testCosts[np.negative(invalidJumps)] = np.exp(testCosts[np.negative(invalidJumps)]/sigma)
    
    
#     ## adding extra rows and columns to compensate for the index shift indicated by correction
#     testCosts = np.concatenate((testCosts,
#                                 np.ones((testCosts.shape[0], correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
#     testCosts = np.concatenate((testCosts,
#                                 np.ones((correction, testCosts.shape[1]))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=0)
    
#     ## setting transition from N-1 to N to minCost
#     testCosts[-2, -1] = np.min(testCosts)
    
    if loopOnLast :
        ## setting the looping from the last frame
        testCosts[-2, 0] = 0.0#np.min(testCosts)
        ## setting the looping from the empty frame and in place looping
        testCosts[-1, 0] = testCosts[-1, -1] = 0.0#np.min(testCosts)
    else :
        testCosts[-1, 0] = np.max(testCosts)
    
    gwv.showCustomGraph(testCosts)
    
    testSequence[DICT_TRANSITION_COSTS_LOCATION] = "/".join(seqLoc.split("/")[:-1])+"/"+"transition_costs_no_normalization-"+testSequence[DICT_SEQUENCE_NAME]+".npy"
    print 
    print testSequence[DICT_TRANSITION_COSTS_LOCATION], testCosts.shape
    print "------------------"
#     np.save(testSequence[DICT_TRANSITION_COSTS_LOCATION], testCosts)
#     np.save(testSequence[DICT_SEQUENCE_LOCATION], testSequence)

# <codecell>

gwv.showCustomGraph(np.load("/home/ilisescu/PhD/data/street/transition_costs_no_normalization-blue_car1.npy"))

# <codecell>

gwv.showCustomGraph(np.load(np.load("/home/ilisescu/PhD/data/street/semantic_sequence-blue_car1.npy").item()[DICT_TRANSITION_COSTS_LOCATION]))

# <codecell>

## compute transition costs
# synthSequence = np.load(dataPath+"synthesisedSequences/wave-batch_ish_changed_slightly_where_asking_sems_too_long/synthesised_sequence.npy").item()
# synthSequence = np.load(dataPath+"synthesisedSequences/lullaby/synthesised_sequence.npy").item()
# synthSequence = np.load(dataPath+"synthesisedSequences/newHavana/synthesised_sequence.npy").item()

isLooping = True ## tells me that I'm using a sprite that loops back and I added an extra frame to it
loopOnLast = True
onlyBackwards = False
# for seqLoc in synthSequence[DICT_USED_SEQUENCES] :
# for seqLoc in np.sort(glob.glob("/home/ilisescu/PhD/data/havana/semantic_sequence*.npy"))[[0]] :
# for seqLoc in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/elevators/semantic_sequence*.npy")) :
# for seqLoc in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence*.npy"))[0:1] :
# for seqLoc in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/toy/semantic_sequence*.npy"))[0:1] :
for seqLoc in np.sort(glob.glob("/home/ilisescu/PhD/data/street/semantic_sequence*.npy"))[2:3] :
    testSequence = np.load(seqLoc).item()
    # print testSequence[DICT_TRANSITION_COSTS_LOCATION]
#     distMat = np.load("/".join(seqLoc.split("/")[:-1])+"/"+testSequence[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy")
#     distMat = np.load("/".join(seqLoc.split("/")[:-1])+"/"+testSequence[DICT_SEQUENCE_NAME]+"-new_overlap_norm_distMat.npy")
    distMat = np.load(testSequence['sequence_precomputed_distance_matrix_location'])
    ## filter ##
    filterSize = 4
#     optimizedDistMat = vtu.filterDistanceMatrix(distMat, filterSize, True)
    gwv.showCustomGraph(distMat)
    
    if False :
        coeff = special.binom(filterSize*2, range(0, filterSize*2 +1))
        kernel = np.eye(len(coeff))
        kernel = kernel*coeff/np.sum(coeff)
    else :
        kernel = np.eye(filterSize*2+1)
        
    optimizedDistMat = cv2.filter2D(distMat, -1, kernel)

    ## if using vanilla
    if True :
        optimizedDistMat = optimizedDistMat[1:optimizedDistMat.shape[1], 0:-1]
        correction = 1
    else :
        correction = 0
    
    ################# THIS ACCOUNTS FOR THE FACT THAT I DIDN'T NORMALIZE THE DISTMAT BY NUM OF PIXELS FOR THE WAVE SEQUENCES #################
#     maxArea = 0.0
#     for bboxKey in np.sort(testSequence[DICT_BBOXES].keys()) :
#         maxArea = np.max((maxArea, np.prod(np.max(testSequence[DICT_BBOXES][bboxKey], axis=0) - np.min(testSequence[DICT_BBOXES][bboxKey], axis=0))))
#     print maxArea
#     optimizedDistMat /= (maxArea*2) # np.max(optimizedDistMat)
    
    ##########################################################################################################################################
    
    ## exponential
#     testCosts = np.exp(np.copy(optimizedDistMat)/(np.average(optimizedDistMat)*0.1))
#     testCosts = np.exp(np.copy(1.0+optimizedDistMat)/(np.average(1.0+optimizedDistMat)*0.002))

    testCosts = np.copy(optimizedDistMat)
    
    threshPercentile = 0.1
#     threshPercentile = 0.05
        
    print np.max(testCosts), np.min(testCosts), np.sort(testCosts.flatten())[int(len(testCosts.flatten())*threshPercentile)], int(len(testCosts.flatten())*threshPercentile)
    thresh = np.sort(testCosts.flatten())[int(len(testCosts.flatten())*threshPercentile)]
    print np.sort(testCosts.flatten())
    print "THRESH", thresh
    
    ## don't want to jump too close so increase costs in a window
    minJumpLength = 20
    if onlyBackwards :
        tmp = (np.triu(np.ones(optimizedDistMat.shape), k=2) +
               np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
               np.eye(optimizedDistMat.shape[0], k=1))
    else :
        tmp = (np.triu(np.ones(optimizedDistMat.shape), k=minJumpLength) +
               np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
               np.eye(optimizedDistMat.shape[0], k=1))
    tmp[tmp == 0] = 100.0
#     optimizedDistMat *= tmp
    testCosts *= tmp
    #########################################
    
    print np.max(testCosts), np.min(testCosts), np.sort(testCosts.flatten())[int(len(testCosts.flatten())*threshPercentile)], int(len(testCosts.flatten())*threshPercentile)

    ## exponential
    testCosts = np.exp(np.copy(testCosts)/(np.average(testCosts)*0.01)) ## this does the same as above and interestingly has the same semantic importance as above after binary search WTF???

    ######## I'M FULL OF SHIT --> THEY SHOULD ALL WORK, PROBLEM WAS THAT TARA'S SPRITE I WAS LOOKING AT HAPPENED TO SWITCH LABEL AROUND THE 100 FRAME MARK 
    ######## (JUMPING OVER THE SMOOTHSTEP TRANSITION MOST LIKELY)
    ######## SO WITHOUT A HIGH ENOUGH SEMANTIC IMPORTANCE, THE LAST FRAMES OF THE FIRST SEQUENCE WOULDN'T GET THE RIGHT SEMANTIC AND THEN FROM THERE ON
    ######## IT'S PRETTY HARD TO SWITCH THE LABEL IMMEDITALEY WITHOUT THE SMOOTHSTEP TRANSITION

    #########################################
    ## do the thresholding based on how many jumps I want to keep per frame
    desiredPercentage = 0.1 ## desired percentage of transitions to keep
#     desiredPercentage = 0.05 ## desired percentage of transitions to keep
    jumpsToKeep = int(testCosts.shape[0]*desiredPercentage)
    testCosts[np.arange(testCosts.shape[0]).repeat(testCosts.shape[0]-jumpsToKeep),
                           np.argsort(testCosts, axis=-1)[:, jumpsToKeep:].flatten()] = GRAPH_MAX_COST
    invalidJumps = testCosts > thresh
#     testCosts[testCosts > thresh] = GRAPH_MAX_COST
    
    ## exponential
#     testCosts = np.exp(np.copy(optimizedDistMat)/(np.average(optimizedDistMat)*0.1))
#     testCosts = np.exp(np.copy(1.0+testCosts)/(np.average(1.0+testCosts)*0.05))
#     testCosts[testCosts > thresh] = GRAPH_MAX_COST


    ## adding extra rows and columns so that the optimized matrix has the same dimensions as distMat
    ## for the indices that were cut out I put zero cost for jumps to frames that can still be used after optimization
    if isLooping :
        ########## it means I deal with sprites like the havana cars ##########
#         testCosts = np.concatenate((np.ones((testCosts.shape[0], filterSize))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts),
#                                     testCosts,
#                                     (1.0-np.eye(testCosts.shape[0], filterSize+correction+1, k=-testCosts.shape[0]+1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
#         ## the +1 is because if a sprite is looping, the last frame is the empty frame
#         testCosts = np.concatenate(((1.0-np.eye(filterSize, distMat.shape[0]+1, k=1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts),
#                                     testCosts,
#                                     (1.0-np.eye(filterSize+correction+1, distMat.shape[0]+1, k=distMat.shape[0]+1-filterSize-correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=0)

        testCosts = np.concatenate((testCosts,
                                    (1.0-np.eye(testCosts.shape[0], correction+1, k=-testCosts.shape[0]+1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
        ## the +1 is because if a sprite is looping, the last frame is the empty frame
        testCosts = np.concatenate((testCosts,
                                    (1.0-np.eye(correction+1, distMat.shape[0]+1, k=distMat.shape[0]+1-correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=0)

        if loopOnLast :
            ## setting the looping from the last frame
            testCosts[-2, 0] = 0.0#np.min(testCosts)
            ## setting the looping from the empty frame and in place looping
            testCosts[-1, 0] = testCosts[-1, -1] = 0.0#np.min(testCosts)
        else :
            testCosts[-1, 0] = 0.0#np.min(testCosts)
    else :
#         testCosts = np.concatenate((np.ones((testCosts.shape[0], filterSize))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts),
#                                     testCosts,
#                                     np.ones((testCosts.shape[0], filterSize+correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
#         testCosts = np.concatenate((np.roll(np.concatenate((np.zeros((filterSize, 1)) + np.min(testCosts),
#                                                             np.ones((filterSize, distMat.shape[0]-1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1), filterSize, axis=1),
#                                     testCosts,
#                                     np.roll(np.concatenate((np.zeros((filterSize+correction, 1)) + np.min(testCosts),
#                                                             np.ones((filterSize+correction, distMat.shape[0]-1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1), filterSize, axis=1)), axis=0)
        
        testCosts = np.concatenate((testCosts,
                                    np.ones((testCosts.shape[0], correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
        testCosts = np.concatenate((testCosts,
                                    np.roll(np.concatenate((np.zeros((correction, 1)) + np.min(testCosts),
                                                            np.ones((correction, distMat.shape[0]-1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1), 0, axis=1)), axis=0)




    gwv.showCustomGraph(testCosts)
#     print testSequence[DICT_TRANSITION_COSTS_LOCATION]
    testSequence[DICT_TRANSITION_COSTS_LOCATION] = "/".join(seqLoc.split("/")[:-1])+"/"+"transition_costs_no_normalization-"+testSequence[DICT_SEQUENCE_NAME]+".npy"
    print 
    print testSequence[DICT_TRANSITION_COSTS_LOCATION], testCosts.shape
    print "------------------"
#     np.save(testSequence[DICT_TRANSITION_COSTS_LOCATION], testCosts)
#     np.save(testSequence[DICT_SEQUENCE_LOCATION], testSequence)

# <codecell>

print 

# <codecell>

print np.load("/home/ilisescu/PhD/data/havana/transition_costs_no_normalization-black_car1.npy")[21, 336]
gwv.showCustomGraph(np.load("/home/ilisescu/PhD/data/havana/transition_costs_no_normalization-black_car1.npy"))
print testCosts[21, 336]

# <codecell>

gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/toy/transition_costs_no_normalization_-toy1.npy"))
tmp = np.argwhere(np.load("/media/ilisescu/Data1/PhD/data/toy/transition_costs_no_normalization_-toy1.npy") != GRAPH_MAX_COST)
print [len(np.argwhere(tmp[:, 0] == i)) for i in xrange(606)]

# <codecell>

print len(np.argwhere(testCosts != GRAPH_MAX_COST))/float(896*896)

# <codecell>

gwv.showCustomGraph(invalidJumps)

# <codecell>

tmp = np.load("/home/ilisescu/PhD/data/havana/transition_costs_no_normalization_-black_car1.npy")
np.min(tmp[tmp != GRAPH_MAX_COST]-testCosts[tmp != GRAPH_MAX_COST])
# gwv.showCustomGraph(tmp-testCosts)

# <codecell>

## code to turn a sprite straight from the merging UI into a semantic sequence
# seqLoc = "/media/ilisescu/Data1/PhD/data/theme_park_sunny/"
# seqLoc = "/media/ilisescu/Data1/PhD/data/windows/"
# seqLoc = "/media/ilisescu/Data1/PhD/data/digger/"
seqLoc = "/home/ilisescu/PhD/data/havana/"

loopingSequences = np.ones(4, bool) #np.array([False, False, True, True, True, True, True, True, True, True, True, True, True])

for spriteLoc, loopingSequence in zip(np.sort(glob.glob(seqLoc+"sprite-*.npy"))[[1, 5, 6, 9]], loopingSequences) :
    sequence = np.load(spriteLoc).item()
    seqName = sequence[DICT_SEQUENCE_NAME]
    print seqName, loopingSequence#, sequence.keys()
    
    if loopingSequence :
        print "adding extra frame for looping"
        sequence[DICT_BBOXES][np.max(sequence[DICT_FRAMES_LOCATIONS].keys())+1] = np.zeros((4, 2))
        sequence[DICT_BBOX_CENTERS][np.max(sequence[DICT_FRAMES_LOCATIONS].keys())+1] = np.zeros(2)
        sequence[DICT_BBOX_ROTATIONS][np.max(sequence[DICT_FRAMES_LOCATIONS].keys())+1] = 0.0
        
        if DICT_FOOTPRINTS in sequence.keys() :
            sequence[DICT_FOOTPRINTS][np.max(sequence[DICT_FRAMES_LOCATIONS].keys())+1] = np.zeros((4, 2))

    if DICT_MASK_LOCATION not in sequence.keys() :
        print "setting mask location"
        sequence[DICT_MASK_LOCATION] = seqLoc + sequence[DICT_SEQUENCE_NAME] + "-maskedFlow-blended/"

    if DICT_FOOTPRINTS not in sequence.keys() :
        print "setting footprints"
        sequence[DICT_FOOTPRINTS] = sequence[DICT_BBOXES]

    if DICT_SEQUENCE_LOCATION not in sequence.keys() :
        print "setting sequence location"
        sequence[DICT_SEQUENCE_LOCATION] = seqLoc + "semantic_sequence-" + sequence[DICT_SEQUENCE_NAME] + ".npy"

    if True and DICT_PATCHES_LOCATION not in sequence.keys() :
        print "setting preloaded patches"
#         currentSequencePatches = {}
#         for frameKey in np.sort(sequence[DICT_FRAMES_LOCATIONS].keys()) :
#             frameName = sequence[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
#             maskDir = sequence[DICT_MASK_LOCATION]

#             if os.path.isdir(maskDir) and os.path.exists(maskDir+"/"+frameName) :
#                 im = np.array(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED), dtype=np.uint8)

#                 visiblePixels = np.argwhere(im[:, :, -1] != 0)
#                 topLeft = np.min(visiblePixels, axis=0)
#                 patchSize = np.max(visiblePixels, axis=0) - topLeft + 1

#                 currentSequencePatches[frameKey] = {'top_left_pos':topLeft, 'sprite_colors':im[visiblePixels[:, 0], visiblePixels[:, 1], :], 
#                                                    'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize}

#             sys.stdout.write('\r' + "Loaded image " + np.string_(len(currentSequencePatches)) + " (" + np.string_(len(sequence[DICT_FRAMES_LOCATIONS])) + ")")
#             sys.stdout.flush()
#         print 
        sequence[DICT_PATCHES_LOCATION] = seqLoc+"preloaded_patches-"+seqName+".npy"
#         np.save(sequence[DICT_PATCHES_LOCATION], currentSequencePatches)

    if DICT_FRAME_SEMANTICS not in sequence.keys() :
        print "setting frame semantics",
        if loopingSequence :
            print "LOOPING"
            sequence[DICT_FRAME_SEMANTICS] = np.concatenate((np.concatenate((np.zeros((len(sequence[DICT_FRAMES_LOCATIONS]), 1)),
                                                                             np.ones((len(sequence[DICT_FRAMES_LOCATIONS]), 1))), axis=1),
                                                             np.array([[1.0, 0.0]])), axis=0)
        else :
            print
            sequence[DICT_FRAME_SEMANTICS] = np.ones((len(sequence[DICT_FRAMES_LOCATIONS]), 1))
    else :
        if loopingSequence :
            print "adding [1, 0] semantics at the end"
            sequence[DICT_FRAME_SEMANTICS] = np.concatenate((np.concatenate((np.zeros((len(sequence[DICT_FRAME_SEMANTICS]), 1)), sequence[DICT_FRAME_SEMANTICS]), axis=1),
                                                             np.array([[1.0, 0.0, 0.0, 0.0]])), axis=0)
    print sequence[DICT_FRAME_SEMANTICS].shape
    if True and DICT_TRANSITION_COSTS_LOCATION not in sequence.keys() :
        print "setting transition costs location"
        sequence[DICT_TRANSITION_COSTS_LOCATION] = seqLoc + seqName + "-vanilla_distMat.npy"

    np.save(sequence[DICT_SEQUENCE_LOCATION], sequence)
#     print sequence.keys()
    print "saved", sequence[DICT_SEQUENCE_LOCATION]
    
#     print np.max(sequence[DICT_BBOXES].keys()), np.max(sequence[DICT_BBOX_CENTERS].keys()), np.max(sequence[DICT_BBOX_ROTATIONS].keys()), 
#     print np.max(sequence[DICT_FOOTPRINTS].keys()), np.max(sequence[DICT_FRAMES_LOCATIONS].keys())

# <codecell>

print sequence[DICT_FRAME_SEMANTICS]

# <codecell>

# for seqLoc in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/windows/semantic_sequence-window_*.npy")) :
#     seq = np.load(seqLoc).item()
#     if seq[DICT_ICON_FRAME_KEY] < np.min(seq[DICT_BBOXES].keys()) or seq[DICT_ICON_FRAME_KEY] > np.max(seq[DICT_BBOXES].keys()) :
#         print seqLoc
#         print np.sort(seq[DICT_BBOXES].keys())
#         print np.sort(seq[DICT_FRAMES_LOCATIONS].keys())
#         print seq[DICT_ICON_FRAME_KEY]
#         seq[DICT_ICON_FRAME_KEY] = np.min(seq[DICT_BBOXES].keys())
#         np.save(seqLoc, seq)

# <codecell>

## takes a non looping sprite and extends it with all frames mirrored time-wise
for spriteLoc in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/windows/sprite*.npy"))[0:1] :
    baseLoc = "/".join(spriteLoc.split("/")[:-1]) + "/"
    print spriteLoc
    sprite = np.load(spriteLoc).item()
    print sprite.keys()
    print len(sprite[DICT_FRAMES_LOCATIONS].keys()), np.sort(sprite[DICT_FRAMES_LOCATIONS].keys())
    sortedKeys = np.sort(sprite[DICT_FRAMES_LOCATIONS].keys())[:-1][::-1]
    maxKey = np.max(sprite[DICT_FRAMES_LOCATIONS].keys())
    for i, key in enumerate(sortedKeys) :
        newKey = maxKey + i + 1
#         print key, newKey
        sprite[DICT_BBOX_CENTERS][newKey] = sprite[DICT_BBOX_CENTERS][key]
        sprite[DICT_BBOX_ROTATIONS][newKey] = sprite[DICT_BBOX_ROTATIONS][key]
        sprite[DICT_BBOXES][newKey] = sprite[DICT_BBOXES][key]
        sprite[DICT_FRAMES_LOCATIONS][newKey] = baseLoc+"frame-{0:05d}.png".format(newKey+1)
        shutil.copyfile(baseLoc+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-{0:05d}.png".format(key+1),
                        baseLoc+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-{0:05d}.png".format(newKey+1))
    sprite[DICT_FRAME_SEMANTICS] = np.concatenate((sprite[DICT_FRAME_SEMANTICS], sprite[DICT_FRAME_SEMANTICS][:-1, :][::-1, :]))
    
    np.save(spriteLoc, sprite)

# <codecell>

# labelProbs = sprite[DICT_FRAME_SEMANTICS]
# fig1 = figure()
# clrs = np.arange(0.0, 1.0+1.0/(2-1), 1.0/(2-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
# stackplot(np.arange(len(labelProbs)), np.row_stack(tuple([i for i in labelProbs.T])), colors=clrs)
# print np.sort(sprite[DICT_BBOX_CENTERS].keys())
# print np.sort(sprite[DICT_BBOX_ROTATIONS].keys())
# print np.sort(sprite[DICT_BBOXES].keys())
# print np.sort(sprite[DICT_FRAMES_LOCATIONS].keys())

# <codecell>

synthSeqLocation = dataPath+"synthesisedSequences/tetris/synthesised_sequence.npy"
synthSeq = np.load(synthSeqLocation).item()
# usedSequences = synthSeq[DICT_USED_SEQUENCES]
# semanticSequences = []
# for sequence in usedSequences :
#     semanticSequences.append(np.load(sequence).item())

TOGGLE_DELAY = 4
# EXTEND_LENGTH = TOGGLE_DELAY*6 +1
EXTEND_LENGTH = TOGGLE_DELAY*3 +1
with open("/media/ilisescu/Data1/PhD/data/windows/tetris/6x9_sess2.txt") as f:
    lines = f.readlines()
    gridSize = np.array(lines[0].split(",")[1:3], int)[::-1] ## (rows, cols)
    lines = np.concatenate((["D,121231231233,"+"".join(np.zeros(np.prod(gridSize), int).astype(np.string_))+"\n"], lines[1:]))
    for line in lines[0:] :
#         print np.array(list(line.split(",")[-1])[:-1], int).reshape(gridSize, order='C')
#         sys.stdout.flush()
        instructions = np.array(list(line.split(",")[-1])[:-1], int)
        print instructions
        
        
        for instance, instruction in enumerate(instructions[0:]) :
            numSemantics = synthSeq[DICT_SEQUENCE_INSTANCES][instance][DICT_DESIRED_SEMANTICS].shape[1]
            

            ## take current semantics
            desiredSemantics = synthSeq[DICT_SEQUENCE_INSTANCES][instance][DICT_DESIRED_SEMANTICS][-1, :].reshape((1, numSemantics))
#             print desiredSemantics

            if synthSeq[DICT_SEQUENCE_INSTANCES][instance][DICT_DESIRED_SEMANTICS][-1, instruction] != 1.0 :
                toggledlabels = toggleAllLabelsSmoothly(desiredSemantics[-1, :], instruction, TOGGLE_DELAY) #toggleLabelsSmoothly(np.array([[1.0, 0.0]]), self.TOGGLE_DELAY)
                desiredSemantics = np.concatenate((desiredSemantics, toggledlabels)) #np.zeros((self.TOGGLE_DELAY, numSemantics))))
#                 print desiredSemantics

#                 ## do impulse
#                 ## pad the tip with new semantics
#                 tmp = np.zeros((1, numSemantics))
#                 tmp[0, instruction] = 1.0
#                 desiredSemantics = np.concatenate((desiredSemantics, tmp.repeat(TOGGLE_DELAY*2, axis=0)))
#                 ## toggle back to default
#                 toggledlabels = toggleAllLabelsSmoothly(desiredSemantics[-1, :], 0, TOGGLE_DELAY)
#                 desiredSemantics = np.concatenate((desiredSemantics, toggledlabels))

#                 ## pad remaining with default semantics
#                 tmp = np.zeros((1, numSemantics))
#                 tmp[0, instruction] = 1.0
#                 desiredSemantics = np.concatenate((desiredSemantics, tmp.repeat(EXTEND_LENGTH-4*TOGGLE_DELAY-1, axis=0)))

                
                ## pad remaining with default semantics
                tmp = np.zeros((1, numSemantics))
                tmp[0, instruction] = 1.0
                desiredSemantics = np.concatenate((desiredSemantics, tmp.repeat(EXTEND_LENGTH-TOGGLE_DELAY-1, axis=0)))

            else :
                desiredSemantics = desiredSemantics.repeat(EXTEND_LENGTH, axis=0)
                
            synthSeq[DICT_SEQUENCE_INSTANCES][instance][DICT_DESIRED_SEMANTICS] = np.concatenate((synthSeq[DICT_SEQUENCE_INSTANCES][instance][DICT_DESIRED_SEMANTICS][:-1, :],
                                                                                                  desiredSemantics))
        
#         print desiredSemantics
#         print desiredSemantics.shape
#         print EXTEND_LENGTH
#         print synthSeq[DICT_SEQUENCE_INSTANCES][instance][DICT_DESIRED_SEMANTICS].shape
#         time.sleep(0.2)
#         clear_output()


# print synthSeq[DICT_SEQUENCE_INSTANCES][instance][DICT_DESIRED_SEMANTICS]

# <codecell>

np.save(synthSeqLocation, synthSeq)

# <codecell>

#  synthSeq[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS]
# print instance
labelProbs = synthSeq[DICT_SEQUENCE_INSTANCES][8][DICT_DESIRED_SEMANTICS]
fig1 = figure()
clrs = np.arange(0.0, 1.0+1.0/(2-1), 1.0/(2-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
stackplot(np.arange(len(labelProbs)), np.row_stack(tuple([i for i in labelProbs.T])), colors=clrs)

# <codecell>

# gwv.showCustomGraph(compatibilityMats["2-7"])
# print window.semanticLoopingTab.semanticSequences[1][DICT_SEQUENCE_NAME]
# print window.semanticLoopingTab.semanticSequences[0][DICT_SEQUENCE_NAME]

# <codecell>

blueCarFrames = ("389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406"+
 " 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424"+
 " 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442"+
 " 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460"+
 " 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478"+
 " 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496"+
 " 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514"+
 " 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532"+
 " 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550"+
 " 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568"+
 " 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586"+
 " 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604"+
 " 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622"+
 " 623 624 625 626 627 628 629 630   0   1   2   3   4   5   6   7   8   9"+
 " 10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27"+
 " 28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45"+
 " 46  47  48  49  50  51  52  53  54  55  56  57  58").split(" ")
blueCarFrames = np.array(blueCarFrames)[np.array(blueCarFrames) != '']
# print ",".join(blueCarFrames)
blueCarFrames = np.array([389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,
                          412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,
                          435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,
                          458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,
                          481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,
                          504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,
                          527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,
                          550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,
                          573,574,575,576,577,578,579,580,581,582,583,584,585,586,587,588,589,590,591,592,593,594,595,
                          596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,
                          619,620,621,622,623,624,625,626,627,628,629,630,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
                          18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
                          49,50,51,52,53,54,55,56,57,58])

redCarFrames = ("335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352"+
" 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370"+
" 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388"+
" 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406"+
" 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424"+
" 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442"+
" 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460"+
" 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478"+
" 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496"+
" 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514"+
" 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532"+
" 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549   0"+
" 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18"+
" 19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36"+
" 37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54"+
" 55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72"+
" 73  74  75  76  77  78  79  80  81  82  83  84  85").split(" ")
redCarFrames = np.array(redCarFrames)[np.array(redCarFrames) != ""]
# print ",".join(redCarFrames)
redCarFrames = np.array([335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,
                         359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,
                         383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,
                         407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,
                         431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,
                         455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,
                         479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,
                         503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,
                         527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,0,1,
                         2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
                         37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,
                         69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85])

print compatibilityMats["2-7"][blueCarFrames, redCarFrames]

# <codecell>

# ## this repeats the first bbox to the rest of the frames
# dataset = "theme_park_sunny"
# startFrame = 1442
# frameLocs = np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/"+dataset+"/frame-0*.png"))
# print frameLocs.shape
# for s in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/"+dataset+"/sprite*.npy"))[-1:] :
#     print s
#     sprite = np.load(s).item()
# #     sprite[DICT_ICON_FRAME_KEY] = np.min(sprite[DICT_BBOXES].keys())
#     startFrame = np.min(sprite[DICT_BBOXES].keys())
#     print startFrame
#     for i in xrange(startFrame+1, startFrame+1016) :
#         sprite[DICT_BBOXES][i] = sprite[DICT_BBOXES][startFrame]
#         sprite[DICT_BBOX_ROTATIONS][i] = sprite[DICT_BBOX_ROTATIONS][startFrame]
#         sprite[DICT_BBOX_CENTERS][i] = sprite[DICT_BBOX_CENTERS][startFrame]
#         sprite[DICT_FRAMES_LOCATIONS][i] = "/media/ilisescu/Data1/PhD/data/"+dataset+"/frame-{0:05d}.png".format(i+1)
        
#     print len(sprite[DICT_BBOXES])
# #     np.save(s, sprite)

# <codecell>

class TempWindow() :
    def __init__(self, synthesisedSequence):
        self.EXTEND_LENGTH = 301
        self.semanticSequences = []
        self.preloadedTransitionCosts = {}
        for index, seq in enumerate(synthesisedSequence[DICT_USED_SEQUENCES]) :
            self.semanticSequences.append(np.load(seq).item())
            if DICT_TRANSITION_COSTS_LOCATION in self.semanticSequences[-1].keys() :
                self.preloadedTransitionCosts[index] = np.load(self.semanticSequences[-1][DICT_TRANSITION_COSTS_LOCATION])#/GRAPH_MAX_COST*100.0
                print "loaded", self.semanticSequences[-1][DICT_TRANSITION_COSTS_LOCATION]

def getNewFramesForSequenceFull(self, synthesisedSequence, instancesToUse, instancesLengths, startingFrame, resolveCompatibility = True, numSteps=10, costsAlpha=0.1, compatibilityAlpha=0.65) :

    gm = opengm.gm(instancesLengths.repeat(self.EXTEND_LENGTH))

    self.allUnaries = []

    for i, instanceIdx in enumerate(instancesToUse) : # xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
        seqIdx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
        desiredSemantics = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS][startingFrame:startingFrame+self.EXTEND_LENGTH, :]

        if len(desiredSemantics) != self.EXTEND_LENGTH :
            raise Exception("desiredSemantics length is not the same as EXTEND_LENGTH")

        ################ FIND DESIRED START FRAME ################ 
        if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
            desiredStartFrame = 0
        else :
            desiredStartFrame = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]

        distVariance = 1.0/50.0 ##self.semanticsImportanceSpinBox.value() ##0.0005

        ################ GET UNARIES ################
        self.unaries = vectorisedMultiNormalMultipleMeans(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

        ## normalizing to turn into probabilities
        self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
        impossibleLabels = self.unaries <= 0.0
        ## cost is -log(prob)
        self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
        ## if prob == 0.0 then set maxCost
        self.unaries[impossibleLabels] = GRAPH_MAX_COST


        ## force desiredStartFrame to be the first frame of the new sequence
        self.unaries[:, 0] = GRAPH_MAX_COST
        self.unaries[desiredStartFrame, 0] = 0.0
        
        self.unaries = costsAlpha*self.unaries

        self.allUnaries.append(np.copy(self.unaries.T))

        ## add unaries to the graph
        fids = gm.addFunctions(self.unaries.T)
        # add first order factors
        gm.addFactors(fids, arange(self.EXTEND_LENGTH*i, self.EXTEND_LENGTH*i+self.EXTEND_LENGTH))


        ################ GET PAIRWISE ################
        pairIndices = np.array([np.arange(self.EXTEND_LENGTH-1), np.arange(1, self.EXTEND_LENGTH)]).T + self.EXTEND_LENGTH*i

        ## add function for row-nodes pairwise cost
        fid = gm.addFunction((1.0-costsAlpha)*(1.0-compatibilityAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        ## add second order factors
        gm.addFactors(fid, pairIndices)

    ################ ADD THE PAIRWISE BETWEEN ROWS ################
    if resolveCompatibility :
        for i, j in np.argwhere(np.triu(np.ones((len(instancesToUse), len(instancesToUse))), 1)) :
            seq1Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instancesToUse[i]][DICT_SEQUENCE_IDX]
            seq2Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instancesToUse[j]][DICT_SEQUENCE_IDX]
            pairIndices = np.array([np.arange(self.EXTEND_LENGTH*i, self.EXTEND_LENGTH*i+self.EXTEND_LENGTH), 
                                    np.arange(self.EXTEND_LENGTH*j, self.EXTEND_LENGTH*j+self.EXTEND_LENGTH)]).T
#             print pairIndices

            ## add function for column-nodes pairwise cost
            if seq1Idx <= seq2Idx :
                fid = gm.addFunction((1.0-costsAlpha)*compatibilityAlpha*np.copy(compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))]))
                print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx])),
                print compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].shape
            else :
                fid = gm.addFunction((1.0-costsAlpha)*compatibilityAlpha*np.copy(compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].T))
                print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used Transposed comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))
            ## add second order factors
            gm.addFactors(fid, pairIndices)
            
    print gm; sys.stdout.flush()

    t = time.time()
    inferer = opengm.inference.TrwsExternal(gm=gm)#, parameter=opengm.InfParam(steps=numSteps, useRandomStart=True))
    inferer.infer()
    print "solved in", time.time() - t

    return np.array(inferer.arg(), dtype=int), gm


def getNewFramesForSequenceIterative(self, synthesisedSequence, instancesToUse, instancesLengths, lockedInstances, startingFrame, resolveCompatibility = False, costsAlpha=0.5, compatibilityAlpha=0.5) :

    self.allUnaries = []
    
    self.synthesisedFrames = {}
    totalCost = 0.0
    for instanceIdx, instanceLength, lockedInstance in zip(instancesToUse, instancesLengths, lockedInstances) : # xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
        
        gm = opengm.gm(np.array([instanceLength]).repeat(self.EXTEND_LENGTH))
        
        seqIdx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
        desiredSemantics = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS][startingFrame:startingFrame+self.EXTEND_LENGTH, :]
        
        if lockedInstance : 
            if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]) != self.EXTEND_LENGTH :
                raise Exception("not enough synthesised frames")
            else :
                self.synthesisedFrames[instanceIdx] = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]
                print "locked instance", instanceIdx
                print self.synthesisedFrames[instanceIdx]
                continue

        if len(desiredSemantics) != self.EXTEND_LENGTH :
            raise Exception("desiredSemantics length is not the same as EXTEND_LENGTH")

        ################ FIND DESIRED START FRAME ################ 
        if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
            desiredStartFrame = 0
        else :
            desiredStartFrame = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]

        distVariance = 1.0/2.0 ##self.semanticsImportanceSpinBox.value() ##0.0005

        ################ GET UNARIES ################
        self.unaries = vectorisedMultiNormalMultipleMeans(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

        ## normalizing to turn into probabilities
        self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
        impossibleLabels = self.unaries <= 0.0
        ## cost is -log(prob)
        self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
        ## if prob == 0.0 then set maxCost
        self.unaries[impossibleLabels] = GRAPH_MAX_COST


        ## force desiredStartFrame to be the first frame of the new sequence
        self.unaries[:, 0] = GRAPH_MAX_COST
        self.unaries[desiredStartFrame, 0] = 0.0
        
        #### minimizing totalCost = a * unary + (1 - a) * (b * vert_link + (1-b)*horiz_link) = a*unary + (1-a)*b*sum(vert_link) + (1-a)*(1-b)*horiz_link
        #### where a = costsAlpha, b = compatibilityAlpha, 
        
        compatibilityCosts = np.zeros_like(self.unaries)
        if resolveCompatibility :
            seq1Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
            for instance2Idx in np.sort(self.synthesisedFrames.keys()) :
                seq2Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instance2Idx][DICT_SEQUENCE_IDX]
                print "considering sequences", seq1Idx, seq2Idx, self.synthesisedFrames.keys()
                
#                 if instance2Idx != 1 :
#                     continue
                
                if seq1Idx <= seq2Idx :
#                     self.unaries = (1.0-compatibilityAlpha)*self.unaries + compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].T[self.synthesisedFrames[instance2Idx], :].T
                    compatibilityCosts += (1.0-costsAlpha)*compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].T[self.synthesisedFrames[instance2Idx], :].T
                    
                    print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx])),
                    print compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].shape
                else :
#                     self.unaries = (1.0-compatibilityAlpha)*self.unaries + compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))][self.synthesisedFrames[instance2Idx], :].T
                    compatibilityCosts += (1.0-costsAlpha)*compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))][self.synthesisedFrames[instance2Idx], :].T
                    
                    print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used Transposed comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))        
#         ## doing the alpha*unaries + (1-alpha)*pairwise thingy
#         self.unaries *= costsAlpha
        self.unaries = costsAlpha*self.unaries + compatibilityCosts
        

        self.allUnaries.append(np.copy(self.unaries.T))
        

        ## add unaries to the graph
        fids = gm.addFunctions(self.unaries.T)
        # add first order factors
        gm.addFactors(fids, arange(self.EXTEND_LENGTH))


        ################ GET PAIRWISE ################
        pairIndices = np.array([np.arange(self.EXTEND_LENGTH-1), np.arange(1, self.EXTEND_LENGTH)]).T

#         ## add function for row-nodes pairwise cost doing the alpha*unaries + (1-alpha)*pairwise thingy at the same time
#         fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        if resolveCompatibility :
            fid = gm.addFunction((1.0-costsAlpha)*(1.0-compatibilityAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        else :
            fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        ## add second order factors
        gm.addFactors(fid, pairIndices)        
            
        print gm; sys.stdout.flush()

        t = time.time()
        inferer = opengm.inference.DynamicProgramming(gm=gm)
        inferer.infer()
        print "solved in", time.time() - t, "cost", gm.evaluate(inferer.arg())
        print np.array(inferer.arg(), dtype=int)
        totalCost += gm.evaluate(inferer.arg())
        self.synthesisedFrames[instanceIdx] = np.array(inferer.arg(), dtype=int)
        
    return self.synthesisedFrames, totalCost
#     return np.array(inferer.arg(), dtype=int), gm

# <codecell>



print len([c for c in itertools.permutations(tmp, 4)])

# <codecell>

compatibilityMats = {}
compatibilityMats["00"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--black_car1.npy")/50)*10.0
compatibilityMats["01"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--blue_car1.npy")/50)*10.0
compatibilityMats["11"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-blue_car1--blue_car1.npy")/50)*10.0

# <codecell>

gwv.showCustomGraph(compatibilityMats["01"])

# <codecell>

compatibilityMats = {}
compatibilityMats["00"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--black_car1.npy"))*10.0
compatibilityMats["01"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--blue_car1.npy"))*10.0
compatibilityMats["11"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-blue_car1--blue_car1.npy"))*10.0

# <codecell>

compatibilityMats = {}
compatibilityMats["00"] = cv2.GaussianBlur(np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--black_car1.npy"))*10.0, (51, 51), 20.0)
compatibilityMats["01"] = cv2.GaussianBlur(np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--blue_car1.npy"))*10.0, (51, 51), 20.0)
compatibilityMats["11"] = cv2.GaussianBlur(np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-blue_car1--blue_car1.npy"))*10.0, (51, 51), 20.0)

# <codecell>

compatibilityMats = {}
compatibilityMats["00"] = np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--black_car1.npy")
compatibilityMats["01"] = np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--blue_car1.npy")
compatibilityMats["11"] = np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-blue_car1--blue_car1.npy")

# <codecell>

compatibilityMats = {}
compatibilityMats["00"] = np.zeros_like(np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--black_car1.npy"))
compatibilityMats["01"] = np.zeros_like(np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--blue_car1.npy"))
compatibilityMats["11"] = np.zeros_like(np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-blue_car1--blue_car1.npy"))

# <codecell>

synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/synthesised_sequence.npy").item()

frameIdx = 0

tempWindow = TempWindow(synthSeq)

#### NEW WAY ####

instancesToUse = []
instancesLengths = []
maxFrames = 0
t = time.time()
for i in xrange(len(synthSeq[DICT_SEQUENCE_INSTANCES])) :

#     availableDesiredSemantics = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]) - self.frameIdx
#     if availableDesiredSemantics < self.EXTEND_LENGTH :
#         ## the required desired semantics by copying the last one
#         print "extended desired semantics for", i,
#         lastSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][-1, :]
#         self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS] = np.concatenate((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS],
#                                                                                                        lastSemantics.reshape((1, len(lastSemantics))).repeat(self.EXTEND_LENGTH-availableDesiredSemantics, axis=0)))
#     else :
#         print "didn't extend semantics for", i
    desiredSemantics = synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][frameIdx:frameIdx+tempWindow.EXTEND_LENGTH, :]
    print "num of desired semantics =", desiredSemantics.shape[0], "(", len(synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]), ")",
    print 

    seqIdx = synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]

    if seqIdx in tempWindow.preloadedTransitionCosts.keys() :
        instancesToUse.append(i)
        instancesLengths.append(len(tempWindow.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS]))
#                 newFrames = self.getNewFramesForSequenceInstanceQuick(i, self.semanticSequences[seqIdx],
#                                                                       self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value(),
#                                                                       desiredSemantics, self.frameIdx, framesToNotUse)
    else :
        print "ERROR: cannot extend instance", i, "because the semantic sequence", seqIdx, "does not have preloadedTransitionCosts"
        break
        
instancesToUse = np.array(instancesToUse)
instancesLengths = np.array(instancesLengths)
print instancesToUse, instancesLengths
newFramesNewWay, gm = getNewFramesForSequenceFull(tempWindow, synthSeq, np.array(instancesToUse), np.array(instancesLengths), frameIdx)
# getNewFramesForSequenceIterative(tempWindow, synthSeq, np.array(instancesToUse), np.array(instancesLengths), np.ones(len(instancesToUse), bool), frameIdx, True)
# gm = window.semanticLoopingTab.getNewFramesForSequenceFull(synthSeq, np.array(instancesToUse), np.array(instancesLengths), frameIdx, True)

# selectedSequences = np.array([1, 2])
# ## using Peter's idea
# if True :
# #     print selectedSequences, instancesToUse, np.array([instancesToUse != selectedSequence for selectedSequence in selectedSequences]).all(axis=0)
#     notSelected = np.array([instancesToUse != selectedSequence for selectedSequence in selectedSequences]).all(axis=0)
#     notSelectedInstances = instancesToUse[notSelected]
#     selectedSequences = instancesToUse[np.negative(notSelected)]
#     for s in xrange(len(selectedSequences)) : #permutation in itertools.permutations(selectedSequences, len(selectedSequences)) :
# #         print np.concatenate((notSelectedInstances, permutation)), np.concatenate((np.ones(len(notSelectedInstances), bool), np.zeros(len(permutation), bool)))
#         reorderedInstances = np.concatenate((notSelectedInstances, np.roll(selectedSequences, s)))
#         reorderedLengths = np.concatenate((instancesLengths[notSelected], np.roll(instancesLengths[np.negative(notSelected)], s)))
#         lockedInstances = np.concatenate((np.ones(len(instancesToUse)-1, bool), [False]))
#         print reorderedInstances, reorderedLengths, lockedInstances
#         print 
# #         getNewFramesForSequenceIterative(tempWindow, synthSeq, reorderedInstances, reorderedLengths, lockedInstances, frameIdx-50, True, 0.3, 0.7)
#         print 


print "new way done in", time.time() - t



# print gm.evaluate(newFramesNewWay)
# print gm
# # print 757.411073682 + 827.424882297 + 805.571144802 + 717.196980651 + 739.054008251 + 765.178588429
# print newFramesNewWay.reshape((2, tempWindow.EXTEND_LENGTH))
# newFrames1 = newFramesNewWay.reshape((2, tempWindow.EXTEND_LENGTH))[0, :]
# newFrames2 = newFramesNewWay.reshape((2, tempWindow.EXTEND_LENGTH))[1, :]

# <codecell>

instancesToUse = np.array([0, 1, 2, 3])
instancesLengths = np.array([10, 31, 51,12])
print instancesToUse, instancesLengths
# newFramesNewWay, gm = getNewFramesForSequenceFull(tempWindow, synthSeq, np.array(instancesToUse), np.array(instancesLengths), frameIdx, True)
# getNewFramesForSequenceIterative(tempWindow, synthSeq, np.array(instancesToUse), np.array(instancesLengths), np.ones(len(instancesToUse), bool), frameIdx, True)
# gm = window.semanticLoopingTab.getNewFramesForSequenceFull(synthSeq, np.array(instancesToUse), np.array(instancesLengths), frameIdx, True)

selectedSequences = np.array([-1])
## using Peter's idea
if True :
#     print selectedSequences, instancesToUse, np.array([instancesToUse != selectedSequence for selectedSequence in selectedSequences]).all(axis=0)
    notSelected = np.array([instancesToUse != selectedSequence for selectedSequence in selectedSequences]).all(axis=0)
    notSelectedInstances = instancesToUse[notSelected]
    selectedSequences = instancesToUse[np.negative(notSelected)]
    print notSelectedInstances
    if len(selectedSequences) > 1 :
        for s in xrange(len(selectedSequences)) : #permutation in itertools.permutations(selectedSequences, len(selectedSequences)) :
    #         print np.concatenate((notSelectedInstances, permutation)), np.concatenate((np.ones(len(notSelectedInstances), bool), np.zeros(len(permutation), bool)))
            reorderedInstances = np.concatenate((notSelectedInstances, np.roll(selectedSequences, s)))
            reorderedLengths = np.concatenate((instancesLengths[notSelected], np.roll(instancesLengths[np.negative(notSelected)], s)))
            lockedInstances = np.concatenate((np.ones(len(instancesToUse)-1, bool), [False]))
    else :
        print notSelectedInstances, instancesLengths[notSelected], np.zeros(len(notSelectedInstances), bool)
        

# <codecell>

synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/wave-tagging_bad_jumps/synthesised_sequence.npy").item()
# print synthSeq[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS][68:169]
usedSeq = np.load(synthSeq[DICT_USED_SEQUENCES][1]).item()
unaries = np.zeros((usedSeq[DICT_FRAME_SEMANTICS].shape[0], 101))
desiredSems = np.argmax(synthSeq[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS][68:169, :], axis=1)
thresholdedSems = synthSeq[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS][np.arange(68, 169), desiredSems] > 0.75
print desiredSems * thresholdedSems
print (1-desiredSems) * thresholdedSems

for i, des in enumerate(desiredSems * thresholdedSems) :
    if des == 1 :
        unaries[:, i] = usedSeq[DICT_FRAME_SEMANTICS][:, 1] <= 0.75
        
for i, des in enumerate((1-desiredSems) * thresholdedSems) :
    if des == 1 :
        unaries[:, i] = usedSeq[DICT_FRAME_SEMANTICS][:, 0] <= 0.75

# for i, sem in enumerate(usedSeq[DICT_FRAME_SEMANTICS]) :
#     print sem[np.argmax(sem)] > 0.75
#     if np.argmax(sem) == 0 and sem[np.argmax(sem)] <= 0.75 :
#         unaries[i, ((1-desiredSems) * thresholdedSems).astype(bool)] = 1
#     elif 
gwv.showCustomGraph(unaries)

# <codecell>

# usedSeq[DICT_TRANSITION_COSTS_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave2/transition_costs_no_normalization_-tara2.npy"
# np.save(usedSeq[DICT_SEQUENCE_LOCATION], usedSeq)
print usedSeq[DICT_TRANSITION_COSTS_LOCATION]

# <codecell>

gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/wave2/transition_costs-learned_thresholded_filtered-tara2.npy"))

# <codecell>

# print window.semanticLoopingTab.synthesisedSequence[DICT_USED_SEQUENCES][1]
testSequence = np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()
# print testSequence[DICT_TRANSITION_COSTS_LOCATION]
distMat = np.load("/home/ilisescu/PhD/data/havana/"+testSequence[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy")
## filter ##
filterSize = 4
optimizedDistMat = vtu.filterDistanceMatrix(distMat, filterSize, True)

isLooping = True

## if using vanilla
if True :
    optimizedDistMat = optimizedDistMat[1:optimizedDistMat.shape[1], 0:-1]
    correction = 1
else :
    correction = 0

## don't want to jump too close so increase costs in a window
minJumpLength = 1
tmp = (np.triu(np.ones(optimizedDistMat.shape), k=minJumpLength) +
       np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
       np.eye(optimizedDistMat.shape[0], k=1))
tmp[tmp == 0] = 10.0
optimizedDistMat *= tmp
#########################################


# thresholdedCosts = np.copy(sequenceTransitionCost) #np.load("/media/ilisescu/Data1/PhD/data/wave2/transition_costs-tara2.npy")

## thresholded distMat
# testCosts = np.copy(optimizedDistMat) ## turns into a hunt for best semantic importance parameter and it ends with wither not switching label or switching to it immediately
## without threshold it does pretty much the same but the fact that it has more jumps means it can use slightly worse frames in terms of semantics but better in terms of jumps

## constant multiplier
# testCosts = np.copy(optimizedDistMat)*100.0 ## does the same as thresholded

## squared
# testCosts = np.copy(optimizedDistMat)**2 ## this does the same as above and interestingly has the same semantic importance as above after binary search

## exponential
testCosts = np.exp(np.copy(optimizedDistMat)/(np.average(optimizedDistMat)*0.2)) ## this does the same as above and interestingly has the same semantic importance as above after binary search WTF???

######## I'M FULL OF SHIT --> THEY SHOULD ALL WORK, PROBLEM WAS THAT THE TARA SPRITE I WAS LOOKING AT HAPPENED TO SWITCH LABEL AROUND THE 100 FRAME MARK 
######## (JUMPING OVER THE SMOOTHSTEP TRANSITION MOST LIKELY)
######## SO WITHOUT A HIGH ENOUGH SEMANTIC IMPORTANCE, THE LAST FRAMES OF THE FIRST SEQUENCE WOULDN'T GET THE RIGHT SEMANTIC AND THEN FROM THERE ON
######## IT'S PRETTY HARD TO SWITCH THE LABEL IMMEDITALEY WITHOUT THE SMOOTHSTEP TRANSITION


## constant addition
# testCosts = np.copy(optimizedDistMat)+2.0 ## does the same as above and all use the same semantic importance value!!!!!



#########################################
## do the thresholding based on how many jumps I want to keep per frame
desiredPercentage = 0.1 ## desired percentage of transitions to keep
jumpsToKeep = int(testCosts.shape[0]*desiredPercentage)
testCosts[np.arange(testCosts.shape[0]).repeat(testCosts.shape[0]-jumpsToKeep),
                       np.argsort(testCosts, axis=-1)[:, jumpsToKeep:].flatten()] = GRAPH_MAX_COST


## adding extra rows and columns so that the optimized matrix has the same dimensions as distMat
## for the indices that were cut out I put zero cost for jumps to frames that can still be used after optimization
if isLooping :
    ## it means I deal with sprites like the havana cars
    testCosts = np.concatenate((np.ones((testCosts.shape[0], filterSize))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts),
                                testCosts,
                                (1.0-np.eye(testCosts.shape[0], filterSize+correction+1, k=-testCosts.shape[0]+1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
    ## the +1 is because if a sprite is looping, the last frame is the empty frame
    testCosts = np.concatenate(((1.0-np.eye(filterSize, distMat.shape[0]+1, k=1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts),
                                testCosts,
                                (1.0-np.eye(filterSize+correction+1, distMat.shape[0]+1, k=distMat.shape[0]+1-filterSize-correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=0)
    ## setting the looping from the last frame
    testCosts[-2, 0] = np.min(testCosts)
    ## setting the looping from the empty frame and in place looping
    testCosts[-1, 0] = testCosts[-1, -1] = np.min(testCosts)
else :
    testCosts = np.concatenate((np.ones((testCosts.shape[0], filterSize))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts),
                                testCosts,
                                np.ones((testCosts.shape[0], filterSize+correction))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1)
    testCosts = np.concatenate((np.roll(np.concatenate((np.zeros((filterSize, 1)) + np.min(testCosts),
                                                        np.ones((filterSize, distMat.shape[0]-1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1), filterSize, axis=1),
                                testCosts,
                                np.roll(np.concatenate((np.zeros((filterSize+correction, 1)) + np.min(testCosts),
                                                        np.ones((filterSize+correction, distMat.shape[0]-1))*(np.max(testCosts)-np.min(testCosts)) + np.min(testCosts)), axis=1), filterSize, axis=1)), axis=0)




gwv.showCustomGraph(testCosts)
print testSequence[DICT_TRANSITION_COSTS_LOCATION]
# testSequence[DICT_TRANSITION_COSTS_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave2/transition_costs_TEST_-tara2.npy"
# np.save(testSequence[DICT_TRANSITION_COSTS_LOCATION], testCosts)
# np.save(testSequence[DICT_SEQUENCE_LOCATION], testSequence)

# <codecell>

from scipy import special

# <codecell>

gwv.showCustomGraph(tmp)

# <codecell>

gwv.showCustomGraph(np.load("/home/ilisescu/PhD/data/havana/transition_costs_no_normalization_-blue_car1.npy"))

# <codecell>

gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/toy/transition_costs-precomputed_loops-toy1.npy"))

# <codecell>

# np.save("waveSeqBadJames.npy", window.semanticLoopingTab.synthesisedSequence)
synthSequence = np.load("waveSeqBadJames.npy").item()
usedSequence = np.load(synthSequence[DICT_USED_SEQUENCES][synthSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_IDX]]).item()
print synthSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]
# print synthSequence[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS]
# print usedSequence[DICT_FRAME_SEMANTICS][synthSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES], :]
usedDistMat = np.load(usedSequence[DICT_TRANSITION_COSTS_LOCATION])
print usedDistMat[368, 510]
visCosts = np.copy(usedDistMat)
visCosts[visCosts == GRAPH_MAX_COST] = 50.0
# visCosts[visCosts != 0] = np.log(visCosts[visCosts != 0])
gwv.showCustomGraph(visCosts)

# <codecell>

def saveSequenceImages(outputDir, bgImage, synthesisedSequence, semanticSequences, preloadedPatches) :
    for frameIdx in arange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]))[0:] :
        ## go through all the semantic sequence instances
#         frame = np.zeros((bgImage.shape[0], bgImage.shape[1], 4), dtype=uint8)
#         frame[:, :, 0:3] = bgImage[:, :, 0:3]
        frame = np.copy(bgImage[:, :, 0:3])
        for s in xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
            ## index in semanticSequences of current instance
            seqIdx = int(synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SEQUENCE_IDX])
            ## if there's a frame to show and the requested frameIdx exists for current instance draw, else draw just first frame
            if frameIdx < len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SEQUENCE_FRAMES]) :
                sequenceFrameIdx = int(synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SEQUENCE_FRAMES][frameIdx])
                if sequenceFrameIdx >= 0 and sequenceFrameIdx < len(semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys()) :
                    frameToShowKey = np.sort(semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys())[sequenceFrameIdx]
                else :
                    frameToShowKey = -1
                    print "NOT OVERLAYING 1"
            else :
                frameToShowKey = -1 #np.sort(semanticSequences[seqIdx][DICT_FRAMES_LOCATIONS].keys())[0]
                print "NOT OVERLAYING 2"

            if frameToShowKey >= 0 and seqIdx >= 0 and seqIdx < len(semanticSequences) :
                if seqIdx in preloadedPatches.keys() :
                    frame = drawFrame(np.copy(frame), semanticSequences[seqIdx], frameToShowKey,
                                      synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_OFFSET],
                                      synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SCALE],
                                      preloadedPatches[seqIdx][frameToShowKey])
                else :
                    frame = drawFrame(np.copy(frame), semanticSequences[seqIdx], frameToShowKey,
                                      synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_OFFSET],
                                      synthesisedSequence[DICT_SEQUENCE_INSTANCES][s][DICT_SCALE])
        
#                 drawOverlay.save(outputDir+"frame-{0:05d}.png".format(frameIdx+1))
#                 figure(); imshow(frame)%%!
        Image.fromarray(frame.astype(np.uint8)).save(outputDir+"frame-{0:05d}.png".format(frameIdx+1))

        
def drawFrame(currentFrame, sequence, frameKey, offset, scale, spritePatch = None) :
    
#     frame = np.zeros((imgSize[0], imgSize[1], 3), dtype=np.uint8)

    scaleTransf = np.array([[scale[0], 0.0], [0.0, scale[1]]])
    offsetTransf = np.array([[offset[0]], [offset[1]]])

    ## draw sprite
    tl = np.min(sequence[DICT_BBOXES][frameKey], axis=0)
    br = np.max(sequence[DICT_BBOXES][frameKey], axis=0)
    w, h = br-tl
    aabb = np.array([tl, tl + [w, 0], br, tl + [0, h]])

    transformedAABB = (np.dot(scaleTransf, aabb.T) + offsetTransf)

    if spritePatch != None :
        transformedPatchTopLeftDelta = np.dot(scaleTransf, spritePatch['top_left_pos'][::-1].reshape((2, 1))-tl.reshape((2, 1)))

        image = np.ascontiguousarray(np.zeros((spritePatch['patch_size'][0], spritePatch['patch_size'][1], 4)), dtype=np.uint8)
        image[spritePatch['visible_indices'][:, 0], spritePatch['visible_indices'][:, 1], :] = spritePatch['sprite_colors']

    else :
        transformedPatchTopLeftDelta = np.zeros((2, 1))

        frameName = sequence[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
        if DICT_MASK_LOCATION in sequence.keys() :
            image = np.array(Image.open(sequence[DICT_MASK_LOCATION]+frameName))[:, :, [2, 1, 0, 3]]
        else :
            image = np.array(Image.open(sequence[DICT_FRAMES_LOCATIONS][frameKey]))
        image = np.ascontiguousarray(image[aabb[0, 1]:aabb[2, 1], aabb[0, 0]:aabb[2, 0], :])

    if image.shape[-1] == 3 :
        img = cv2.cvtColor(cv2.resize(image, dsize=(0, 0), fx = scale[0], fy=scale[1], interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    else :
        img = cv2.cvtColor(cv2.resize(image, dsize=(0, 0), fx = scale[0], fy=scale[1], interpolation=cv2.INTER_AREA), cv2.COLOR_BGRA2RGBA)
    topLeftPos = transformedAABB[:, :1] + transformedPatchTopLeftDelta
    
#     currentFrame[topLeftPos[1]:topLeftPos[1]+img.shape[0], 
#                  topLeftPos[0]:topLeftPos[0]+img.shape[1], :] = img
    
    if img.shape[-1] == 3 :
        currentFrame[topLeftPos[1]:topLeftPos[1]+img.shape[0], 
                     topLeftPos[0]:topLeftPos[0]+img.shape[1], :] = img
    else :
        currentFrame[topLeftPos[1]:topLeftPos[1]+img.shape[0], 
                     topLeftPos[0]:topLeftPos[0]+img.shape[1]] = (img[:, :, 0:3]*(img[:, :, 3].reshape((img.shape[0], img.shape[1], 1))/255.0)+
                                                             currentFrame[topLeftPos[1]:topLeftPos[1]+img.shape[0], 
                                                                          topLeftPos[0]:topLeftPos[0]+img.shape[1], 0:3]*(1.0-img[:, :, 3].reshape((img.shape[0], img.shape[1], 1))/255.0))


    return currentFrame
        
        
# location = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave-batch_ish_changed_slightly_where_asking_sems_too_long/"
# location = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave-same_as_batch_ish_but_costs_from_dists_no_normalization/"
location = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave-test_sequence/"
synthesisedSequence = np.load(location+"synthesised_sequence.npy").item()
## update background
bgImage = np.ascontiguousarray(np.array(Image.open(synthesisedSequence[DICT_SEQUENCE_BG]))[:, :, 0:3])
# bgImage = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);

## load used semantic sequences
semanticSequences = []
preloadedPatches = {}
for index, seq in enumerate(synthesisedSequence[DICT_USED_SEQUENCES]) :
    semanticSequences.append(np.load(seq).item())
    if DICT_PATCHES_LOCATION in semanticSequences[-1].keys() :
        preloadedPatches[index] = np.load(semanticSequences[-1][DICT_PATCHES_LOCATION]).item()
        
saveSequenceImages(location, bgImage, synthesisedSequence, semanticSequences, preloadedPatches)

# <codecell>

### resample sprites to a subset of the initial frames
subsetExtremes = np.array([[1122, 2674], [1252, 2588], [1394, 2610], [1346, 2522], [1216, 2496], [1806, 2702], [1624, 2880], [1478, 2614]]) ## wave1
# subsetExtremes = np.array([[682, 2002], [1194, 1938], [470, 1990], [896, 2088], [899, 2019], [1320, 2256], [546, 2050], [886, 2006]]) ## wave2
# subsetExtremes = np.array([[868, 1844], [1060, 1708], [372, 1804], [832, 1984], [1173, 1893], [1020, 2100], [806, 1902], [906, 1906]]) ## wave3
baseLoc = "/media/ilisescu/Data1/PhD/data/wave1/"
# for i, sprite in enumerate(np.sort(glob.glob(baseLoc+"sprite*.npy"))) :
#     print sprite, subsetExtremes[i, :], np.diff(subsetExtremes[i, :])
#     s = np.load(sprite).item()
    
#     allKeys = s[DICT_FRAMES_LOCATIONS].keys()
#     for key in allKeys :
#         if key < subsetExtremes[i, 0] or key >= subsetExtremes[i, 1] :
#             del s[DICT_BBOX_CENTERS][key]
#             del s[DICT_BBOX_ROTATIONS][key]
#             del s[DICT_BBOXES][key]
#             del s[DICT_FRAMES_LOCATIONS][key]
#     s[DICT_FRAME_SEMANTICS] = s[DICT_FRAME_SEMANTICS][subsetExtremes[i, 0]:subsetExtremes[i, 1], :]
#     s[DICT_ICON_FRAME_KEY] = subsetExtremes[i, 0]
    
# #     print s.keys()
# #     print s[DICT_ICON_FRAME_KEY]
# #     print len(s[DICT_BBOX_CENTERS].keys())
# #     print len(s[DICT_BBOX_ROTATIONS].keys())
# #     print len(s[DICT_BBOXES].keys())
# #     print len(s[DICT_FRAMES_LOCATIONS].keys())
# #     print s[DICT_FRAME_SEMANTICS].shape


#     np.save(sprite, s)
# #     print np.load(baseLoc+s[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy").shape
#     np.save(baseLoc+s[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy", 
#             np.load(baseLoc+s[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy")[subsetExtremes[i, 0]:subsetExtremes[i, 1], subsetExtremes[i, 0]:subsetExtremes[i, 1]])

# <codecell>

## puts together sound based on semantics of synthesised sequence
soundTracks = []
samplerate = 0
for track in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/toy/toy[1-8].wav")) :
    print track,
    sig, samplerate = sf.read(track)
    print sig.shape
    soundTracks.append(sig)

synthSequence = np.load(dataPath+"synthesisedSequences/lullaby/synthesised_sequence.npy").item()#window.semanticLoopingTab.synthesisedSequence
usedSequence = np.load(synthSequence[DICT_USED_SEQUENCES][0]).item()
frameSemantics = usedSequence[DICT_FRAME_SEMANTICS][:, 1:]
numSemantics = frameSemantics.shape[1]
usedFrames = synthSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]
videoFPS = 50.0
audioSamplesPerFrame = int(np.round(samplerate/videoFPS))
audioTrack = np.zeros((audioSamplesPerFrame*len(usedFrames), 2))

for sem in arange(numSemantics)[0:] :
#     figure(); plot(frameSemantics[usedFrames, sem])
    semLocs = np.argwhere(frameSemantics[usedFrames, sem] > 0.9).flatten()
    print semLocs
    if len(semLocs) > 0 :
        clusters = np.concatenate(([0], np.cumsum((np.abs(semLocs[:-1]-semLocs[1:]) != 1).astype(int))))
        for cluster in xrange(np.max(clusters)+1) :
            centerLoc = int(np.round(np.median(semLocs[np.argwhere(clusters == cluster).flatten()])))
            print centerLoc, centerLoc*audioSamplesPerFrame, sem, len(soundTracks[sem])
            audioTrack[centerLoc*audioSamplesPerFrame:centerLoc*audioSamplesPerFrame+len(soundTracks[sem]), :] += soundTracks[sem]
    #         scatter(centerLoc, 1)
sf.write(dataPath+"synthesisedSequences/lullaby/lullaby.wav", audioTrack, samplerate)

# <codecell>

def intervalOverlap(intervals) :
    ############# this doesn't really work with more than 2 intervals #############
    
    
    ## this returns a sorted array such that the numbers in the first column are the sorted interval start and end points
    ## and the second column are the their indices where even numbers denote starting points and odd ones denote end points
    ## the key makes sure that I put start points before end points if they have the same value
    sortedPoints = np.array(sorted(np.array([(x, y) for x, y in zip(intervals.flatten(), arange(len(intervals)*2))]), key=lambda t: (t[0], np.mod(t[1], 2))))
#     print sortedPoints
    startInterval = -1
    numEnteredIntervals = 0
    for i in sortedPoints[:, 1] :
        ## if even then I'm entering an interval
        if np.mod(i, 2) == 0 :
            startInterval = i
            numEnteredIntervals += 1
        else :
            if numEnteredIntervals > 1 :
                startIntervalIdx = startInterval/len(intervals.T)
                endIntervalIdx = i/len(intervals.T)
                overlapAmount = intervals[endIntervalIdx, 1]-intervals[startIntervalIdx, 0]+1
#                 print "overlap =", startIntervalIdx, endIntervalIdx, np.sum(np.abs(np.diff(intervals, axis=-1)))+1-overlapAmount, 
                return float(overlapAmount)/(np.sum(np.abs(np.diff(intervals, axis=-1))+1)-overlapAmount), overlapAmount
#                 return float(overlapAmount)/np.min(np.abs(np.diff(intervals, axis=-1))+1), overlapAmount
            else :
                return 0.0, 0
            numEnteredIntervals -= 1
    
    
# print intervalOverlap(np.array([[200, 250], [220, 270]]))
# print intervalOverlap(np.array([[220, 270], [200, 250]]))
# print intervalOverlap(np.array([[190, 270], [200, 250]]))
# print intervalOverlap(np.array([[200, 250], [190, 270]]))
# print intervalOverlap(np.array([[150, 195], [200, 250]]))
# print intervalOverlap(np.array([[200, 250], [150, 195]]))
# print intervalOverlap(np.array([[200, 250], [250, 320]]))
# print intervalOverlap(np.array([[447, 467], [446, 467]]))

# <codecell>

###### load synthesised sequence ######
# synthSequence = np.load("/home/ilisescu/PhD/data/synthesisedSequences/wave/synthesised_sequence.npy").item()
###### get used semantic sequence ######
# usedSequence = np.load(synthSequence[DICT_USED_SEQUENCES][0]).item()
def getPrecomputedLoops(semanticSequence) :
    print semanticSequence[DICT_SEQUENCE_NAME]
    # semanticSequence = np.load("/media/ilisescu/Data1/PhD/data/wave3/semantic_sequence-peter3.npy").item()

    numSemantics = semanticSequence[DICT_FRAME_SEMANTICS].shape[1]

    ###### get distance matrix and process for video textures ######
    distMat = np.load("/".join(semanticSequence[DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"+semanticSequence[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy")

    filterSize = correction = 4
    distMat = vtu.filterDistanceMatrix(distMat, filterSize, False)[1:, :-1]

    probs, cumProbs = vtu.getProbabilities(distMat, 0.05, np.zeros_like(distMat), True)
#     gwv.showCustomGraph(probs)

    precomputedLoops = []
    ###### for each possible semantics get the best loops ######
    for currentSemantics in arange(numSemantics)[0:] :
        ###### set desired semantics and get cost of each frame based on how far their label is to the desired one and how far in the timeline they are ######
        desiredSemantics = np.zeros((1, numSemantics))
        desiredSemantics[0, currentSemantics] = 1.0
        distVariance = 0.05
        distToSemantics = vectorisedMinusLogMultiNormal(semanticSequence[DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

        frameSemanticCost = distToSemantics[filterSize:-filterSize-1]
        ## the further away from a frame showing the desired semantics, the higher the cost
        frameSemanticCost[np.argwhere(frameSemanticCost >= 0.01*np.max(frameSemanticCost))] = np.max(frameSemanticCost)
        frameSemanticCost = frameSemanticCost+np.min(np.abs(arange(len(frameSemanticCost)).reshape((1, len(frameSemanticCost))) - np.argwhere(frameSemanticCost < 0.01*np.max(frameSemanticCost))), axis=0)

        ###### run stochastic video textures and store the jump locations ######
        totalJumps = 10000
        currentFrame = np.random.choice(np.argsort(frameSemanticCost)[:10])

        ## jumps in jumpLocations are already at the correct position (i.e. they account for the filtering of distMat)
        jumpLocations = np.zeros(np.array(distMat.shape)+filterSize*2, dtype=int)
        jumpCounter = 0
        prevFrame = vtu.getNewFrame(currentFrame, cumProbs)
        while jumpCounter < totalJumps :
            currentFrame = vtu.getNewFrame(prevFrame, cumProbs, 100)
            ## if there is a jump, record it
            if currentFrame != prevFrame + 1 :
                jumpCounter += 1
                jumpLocations[prevFrame+correction, currentFrame+correction] += 1

                ## randomly reinitialize the start frame to one of the best frames showing desiredSemantics
                if np.random.rand() < 0.5 :
                    currentFrame = vtu.getNewFrame(np.random.choice(np.argsort(frameSemanticCost)[:10]), cumProbs, 10)
            prevFrame = currentFrame

            if np.mod(jumpCounter, 1000) == 0 :
                sys.stdout.write('\r' + "Jumped " + np.string_(jumpCounter) + " times")
                sys.stdout.flush()

        print
#         gwv.showCustomGraph(jumpLocations)

        ###### sort jumps based on how often they've been used ######
        sortedJumps = np.argsort(np.ndarray.flatten(jumpLocations))[::-1]
        sortedJumps = np.array([np.array(sortedJumps/jumpLocations.shape[0], dtype=int), 
                                np.array(np.mod(sortedJumps, jumpLocations.shape[0]), dtype=int)]).T

        ###### find backward jumps among the top used jumps ######
        numJumpsToConsider = 5000
        jump = sortedJumps[2, :]
        backwardsJumps = [] ## (jump_from, jump_to, semantics_cost, jump_cost)
        for jump in sortedJumps[:numJumpsToConsider, :] :
            if jump[0] > jump[1] :
                backwardsJumps.append(np.array([jump[0], jump[1], 
                                                np.sum(frameSemanticCost[arange(jump[1], jump[0]+1)-correction]),
                                                distMat[jump[0]-correction, jump[1]-correction]]))

        ###### only keep loops that show the desired semantic ######
        validLoops = []
        for i, jump in enumerate(backwardsJumps) :
            newJump = np.sort(jump[:2])

            loopDistToSemantics = frameSemanticCost[arange(newJump[0]-correction, newJump[1]-correction, dtype=int)]
            thresh = np.max(distToSemantics)+1 ## it means I'm within 1 frames of the desired semantics
            complyingFrames = loopDistToSemantics < thresh

            if np.any(complyingFrames) :
                validLoops.append(i)    

        backwardsJumps = np.array(backwardsJumps)[validLoops, :]

        ###### only keep loops that do not overlap too much and that are not too costly ######
        maxLoopOverlap = 0.5
        maxDivergenceToThresholdCost = 3.0
        thresholdSemanticCost = backwardsJumps[np.argsort(np.sum(backwardsJumps[:, 2:], axis=1)).flatten(), :][int(np.round(backwardsJumps.shape[0]*0.05)), 2] ## takingthe value of the fifth percentile as threshold
        print "using threshold", thresholdSemanticCost


        keptLoops = []
        for jump in backwardsJumps[np.argsort(np.sum(backwardsJumps[:, 2:], axis=1)).flatten(), :] :

            newJump = np.sort(jump[:2])
            if len(keptLoops) == 0 :
                print newJump, "\t", jump[2:]
                keptLoops.append(newJump)
            else :
                doKeep = True
                for loop in keptLoops :
                    ## only keep loop if it is not overlapping with already kept loops too much
                    if intervalOverlap(np.array([loop, newJump]))[0] > maxLoopOverlap :
                        doKeep = False
                        break

        #         if doKeep and np.sum(jump[2:]) < np.min(np.sum(backwardsJumps[:, 2:4], axis=-1))*1.5 :# len(np.argwhere(complyingFrames))/float(len(complyingFrames)) > 0.8 :
    #             if doKeep :
    #                 print newJump, "\t", jump[2:]
                if doKeep and jump[2] < thresholdSemanticCost*maxDivergenceToThresholdCost :
                    print newJump, "\t", jump[2:]
                    keptLoops.append(newJump)

        precomputedLoops.append(keptLoops)
        print "done semantics", currentSemantics; sys.stdout.flush()
        
    return precomputedLoops

# print getPrecomputedLoops(usedSequence)

# <codecell>

def printPrecomputedLoops(precomputedLoops, semanticSequence) :
    for loops in precomputedLoops :
        for loop in loops :
            print loop
            print semanticSequence[DICT_FRAMES_LOCATIONS][np.sort(semanticSequence[DICT_FRAMES_LOCATIONS].keys())[loop[0]]]
            print semanticSequence[DICT_FRAMES_LOCATIONS][np.sort(semanticSequence[DICT_FRAMES_LOCATIONS].keys())[loop[1]]]
            print
        print "--------------------"
        print
# printPrecomputedLoops(getPrecomputedLoops(usedSequence), usedSequence)

# <codecell>

def getPrecomputedLoopsTransitionCosts(semanticSequence, precomputedLoops) :
    print
    print
    print precomputedLoops
    ## modifies the transitionCosts matrix to give low cost to the precomputed loops
    costLoc = "/".join(semanticSequence[DICT_SEQUENCE_LOCATION].split("/")[:-1]) + "/transition_costs-" + semanticSequence[DICT_SEQUENCE_NAME] + ".npy"
    transitionCosts = np.load(costLoc)
#     precomputedLoopPrior = np.ones_like(transitionCosts)*0.0
    transitionsToChange = np.zeros_like(transitionCosts)
#     gwv.showCustomGraph(np.copy(transitionCosts))
    for loops in precomputedLoops :
        for loop in loops :
    #         print loop
    #         print arange(loop[0], loop[1]+1)
            idxes = meshgrid(arange(loop[0], loop[1]+1, dtype=int), arange(loop[0], loop[1]+1, dtype=int))
    #         transitionCosts[idxes] = np.max(transitionCosts)#/2
            transitionsToChange[idxes[0].flatten()[transitionCosts[idxes[0].flatten(), idxes[1].flatten()] != np.max(transitionCosts)],
                                idxes[1].flatten()[transitionCosts[idxes[0].flatten(), idxes[1].flatten()] != np.max(transitionCosts)]] = 1
    #         precomputedLoopPrior[meshgrid(arange(loop[0], loop[1]+1, dtype=int), arange(loop[0], loop[1]+1, dtype=int))]  = 0.0
    transitionCosts[transitionsToChange == 1] = transitionCosts[transitionsToChange == 1]**2

    for loops in precomputedLoops :
        for loop in loops :
    #         print loop
    #         print arange(loop[0], loop[1]+1)
            transitionCosts[arange(loop[0], loop[1]+1, dtype=int), np.roll(arange(loop[0], loop[1]+1, dtype=int), -1)] = 0
    #         transitionCosts[loop[1], loop[0]] = 0
#             precomputedLoopPrior[arange(loop[0], loop[1]+1, dtype=int), np.roll(arange(loop[0], loop[1]+1, dtype=int), -1)] = 1.0
#     gwv.showCustomGraph(transitionCosts)
    return transitionCosts
# loopsToUse = getPrecomputedLoops(usedSequence)
# gwv.showCustomGraph(getPrecomputedLoopsTransitionCosts(usedSequence, loopsToUse))

# <codecell>

seqLoc = "/media/ilisescu/Data1/PhD/data/wave1/"
for semLoc in np.sort(glob.glob(seqLoc+"semantic_sequence-*.npy")) :
    currentSequence = np.load(semLoc).item()
    precomputedLoops = getPrecomputedLoops(currentSequence)
    
    transitionCosts = getPrecomputedLoopsTransitionCosts(currentSequence, precomputedLoops)
    
    currentSequence[DICT_TRANSITION_COSTS_LOCATION] = "/".join(currentSequence[DICT_SEQUENCE_LOCATION].split("/")[:-1]) + "/transition_costs-precomputed_loops-" + currentSequence[DICT_SEQUENCE_NAME] + ".npy"
    gwv.showCustomGraph(transitionCosts)
    np.save(currentSequence[DICT_TRANSITION_COSTS_LOCATION], transitionCosts)
    print "saved", currentSequence[DICT_TRANSITION_COSTS_LOCATION]
#     np.save(currentSequence[DICT_SEQUENCE_LOCATION], currentSequence)

# <codecell>

gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/wave1/transition_costs-precomputed_loops-aron1.npy"))

# <codecell>

print transitionCosts[idxes][transitionCosts[idxes] != np.max(transitionCosts)]

# <codecell>

gwv.showCustomGraph(precomputedLoopPrior)

# <codecell>

usedSequence = np.load("/media/ilisescu/Data1/PhD/data/wave3/semantic_sequence-james3.npy").item()
print usedSequence[DICT_TRANSITION_COSTS_LOCATION]
gwv.showCustomGraph(np.load(usedSequence[DICT_TRANSITION_COSTS_LOCATION]))
# gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/wave2/james2-vanilla_distMat.npy"))

# <codecell>

## renormalize the prior
# precomputedLoopPrior = precomputedLoopPrior / np.sum(precomputedLoopPrior, axis=1).reshape((len(precomputedLoopPrior), 1))
## get rid of first filterSize frames and the last ones as they get lost in the filtering and will be added back later in the sequenceTransitionCosts
precomputedLoopPrior = precomputedLoopPrior[filterSize:-filterSize-1, filterSize:-filterSize-1]

# <codecell>

gwv.showCustomGraph(-np.log(precomputedLoopPrior))

# <codecell>

distMat = np.load("/media/ilisescu/Data1/PhD/data/wave1/james1-vanilla_distMat.npy")
    
## filter ##
filterSize = 4
optimizedDistMat = vtu.filterDistanceMatrix(distMat, filterSize, True)
## if using vanilla
if True :
    optimizedDistMat = optimizedDistMat[1:optimizedDistMat.shape[1], 0:-1]
    correction = 1
else :
    correction = 0
    
# #### this is just normalizing to [0, 1] A ####
# optimizedDistMat = optimizedDistMat/np.max(optimizedDistMat)
# ########

## don't want to jump too close so increase costs in a window
minJumpLength = 20
tmp = (np.triu(np.ones(optimizedDistMat.shape), k=minJumpLength) +
       np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
       np.eye(optimizedDistMat.shape[0], k=1))
tmp[tmp == 0] = 10.0
optimizedDistMat *= tmp


#### this is just normalizing to [0, 1] B ####
sequenceTransitionCost = optimizedDistMat
########

#### this turns to a probability and then does the same as the unaries ####
# sequenceTransitionCost = vtu.dist2prob(optimizedDistMat, 0.1, True)
# # sequenceTransitionCost = sequenceTransitionCost*precomputedLoopPrior

# gwv.showCustomGraph(np.copy(sequenceTransitionCost))

# impossibleTransitions = sequenceTransitionCost <= 0.0
# ## cost is -log(prob)
# sequenceTransitionCost[np.negative(impossibleTransitions)] = -np.log(sequenceTransitionCost[np.negative(impossibleTransitions)])
# ## if prob == 0.0 then set maxCost
# sequenceTransitionCost[impossibleTransitions] = GRAPH_MAX_COST
########

## do the thresholding based on how many jumps I want to keep per frame
desiredPercentage = 0.1 ## desired percentage of transitions to keep
jumpsToKeep = int(sequenceTransitionCost.shape[0]*desiredPercentage)
sequenceTransitionCost[np.arange(sequenceTransitionCost.shape[0]).repeat(sequenceTransitionCost.shape[0]-jumpsToKeep),
                       np.argsort(sequenceTransitionCost, axis=-1)[:, jumpsToKeep:].flatten()] = GRAPH_MAX_COST


## adding extra rows and columns so that the optimized matrix has the same dimensions as distMat
## for the indices that were cut out I put zero cost for jumps to frames that can still be used after optimization
sequenceTransitionCost = np.concatenate((np.ones((sequenceTransitionCost.shape[0], filterSize))*np.max(sequenceTransitionCost),
                                         sequenceTransitionCost,
                                         np.ones((sequenceTransitionCost.shape[0], filterSize+correction))*np.max(sequenceTransitionCost)), axis=1)
sequenceTransitionCost = np.concatenate((np.roll(np.concatenate((np.zeros((filterSize, 1)),
                                                                 np.ones((filterSize, distMat.shape[0]-1))*np.max(sequenceTransitionCost)), axis=1), filterSize, axis=1),
                                         sequenceTransitionCost,
                                         np.roll(np.concatenate((np.zeros((filterSize+correction, 1)),
                                                                 np.ones((filterSize+correction, distMat.shape[0]-1))*np.max(sequenceTransitionCost)), axis=1), filterSize, axis=1)), axis=0)

# sequenceTransitionCost += (1.0 - precomputedLoopPrior)*10000.0

gwv.showCustomGraph(np.copy(sequenceTransitionCost))

# <codecell>

# gwv.showCustomGraph(-np.log(precomputedLoopPrior))

# <codecell>

visCosts = np.copy(sequenceTransitionCost)#np.load(usedSequence[DICT_TRANSITION_COSTS_LOCATION]))
visCosts[visCosts == GRAPH_MAX_COST] = 50.0
# visCosts[visCosts != 0] = np.log(visCosts[visCosts != 0])
gwv.showCustomGraph(visCosts)

# <codecell>

usedSequence[DICT_TRANSITION_COSTS_LOCATION] = "/".join(usedSequence[DICT_SEQUENCE_LOCATION].split("/")[:-1]) + "/transition_costs-" + usedSequence[DICT_SEQUENCE_NAME] + ".npy"
np.save(usedSequence[DICT_TRANSITION_COSTS_LOCATION], sequenceTransitionCost)
np.save(usedSequence[DICT_SEQUENCE_LOCATION], usedSequence)

# <codecell>

gwv.showCustomGraph(precomputedLoopPrior)
print precomputedLoopPrior.shape

# <codecell>

print sequenceTransitionCost[111, 10]
print sequenceTransitionCost[111, 9]

# <codecell>

currentSemantics = 0
desiredSemantics = np.zeros((1, numSemantics))
desiredSemantics[0, currentSemantics] = 1.0
distVariance = 0.05
distToSemantics = vectorisedMinusLogMultiNormal(usedSequence[DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

frameSemanticCost = distToSemantics[filterSize:-filterSize-1]
## the further away from a frame showing the desired semantics, the higher the cost
frameSemanticCost[np.argwhere(frameSemanticCost >= 0.01*np.max(frameSemanticCost))] = np.max(frameSemanticCost)
frameSemanticCost = frameSemanticCost+np.min(np.abs(arange(len(frameSemanticCost)).reshape((1, len(frameSemanticCost))) - np.argwhere(frameSemanticCost < 0.01*np.max(frameSemanticCost))), axis=0)

newJump = precomputedLoops[currentSemantics][2]
print newJump
loopDistToSemantics = frameSemanticCost[arange(newJump[0]-correction, newJump[1]-correction, dtype=int)]
thresh = np.max(distToSemantics) + 1 ## it means I'm within 6 frames of the desired semantics
complyingFrames = loopDistToSemantics < thresh
print complyingFrames

# <codecell>

pie = np.exp(-frameSemanticCost).reshape((len(frameSemanticCost), 1))
pie /= np.sum(pie)

oldPie = np.copy(pie)
for i in xrange(1000) :
    pie = np.dot(probs, oldPie)
    pie /= np.sum(pie)
    print np.sum((pie-oldPie)**2)
    oldPie = np.copy(pie)

# <codecell>

# figure(); plot(pie)
print np.sum(pie)
print np.min(pie), np.max(pie)

# <codecell>

## plots desired semantics
sequence = window.semanticLoopingTab.synthesisedSequence
# sequence = np.load("/home/ilisescu/PhD/data/synthesisedSequences/lullaby/synthesised_sequence.npy").item()
# sequence = np.load("tmpSeq.npy").item()
desiredSemantics = sequence[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS]#[:366, :]
figure()
clrs = np.arange(0.0, 1.0+1.0/(len(desiredSemantics.T)-1), 1.0/(len(desiredSemantics.T)-1)).astype(np.string_)
stack_coll = stackplot(np.arange(len(desiredSemantics)), np.row_stack(tuple([i for i in desiredSemantics.T])), colors=clrs)
proxy_rects = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack_coll]
legend(proxy_rects, ["0", "1", "2", "3", "4", "5", "6", "7", "8"])
## plots frames semantics
usedSequence = np.load(sequence[DICT_USED_SEQUENCES][sequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_IDX]]).item()
actualSemantics = usedSequence[DICT_FRAME_SEMANTICS][sequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES], :]
figure()
clrs = np.arange(0.0, 1.0+1.0/(len(actualSemantics.T)-1), 1.0/(len(actualSemantics.T)-1)).astype(np.string_)
stack_coll = stackplot(np.arange(len(actualSemantics)), np.row_stack(tuple([i for i in actualSemantics.T])), colors=clrs)
proxy_rects = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack_coll]
legend(proxy_rects, ["0", "1", "2", "3", "4", "5", "6", "7", "8"])

# <codecell>

actualSemantics = np.load("/media/ilisescu/Data1/PhD/data/wave2/semantic_sequence-tara2.npy").item()[DICT_FRAME_SEMANTICS]
figure()
clrs = np.arange(0.0, 1.0+1.0/(len(actualSemantics.T)-1), 1.0/(len(actualSemantics.T)-1)).astype(np.string_)
stack_coll = stackplot(np.arange(len(actualSemantics)), np.row_stack(tuple([i for i in actualSemantics.T])), colors=clrs)
proxy_rects = [Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in stack_coll]
legend(proxy_rects, ["0", "1", "2", "3", "4", "5", "6", "7", "8"])

# <codecell>

# print actualSemantics[800:850, :]
np.sort(np.load("/media/ilisescu/Data1/PhD/data/wave2/semantic_sequence-tara2.npy").item()[DICT_FRAMES_LOCATIONS].keys())[805]

# <codecell>

startSwitchValue = smoothstep(8)[0]
currentSemantics = -1

for i, sem in enumerate(desiredSemantics[:-1, :]) :
    print i, "\t", "{0:04.3f}".format(np.sum(sem)), "\t",
    for n in sem :
        print "{0:04.3f}".format(n), "\t",
        
    if len(np.argwhere(desiredSemantics[i+1, :] == startSwitchValue)) > 1 :
        print
        print "oh, oh"
        break
        
    if len(np.argwhere(desiredSemantics[i+1, :] == startSwitchValue)) == 1 :
        newSem = int(np.argwhere(desiredSemantics[i+1, :] == startSwitchValue).flatten())
        if currentSemantics != newSem :
            currentSemantics = newSem
            print "switched to", currentSemantics
        else :
            print
    else :
        print

# <codecell>

# dirLoc = "/media/ilisescu/Data1/PhD/toyLullaby/"
# os.mkdir(dirLoc)
# for i, f in enumerate(window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]) :
#     print window.semanticLoopingTab.semanticSequences[0][DICT_FRAMES_LOCATIONS][f]
#     copyanything(window.semanticLoopingTab.semanticSequences[0][DICT_FRAMES_LOCATIONS][f], dirLoc + "frame-{0:05d}.png".format(i+1))

# <codecell>

# synthSeq = window.semanticLoopingTab.synthesisedSequence
# synthSeq[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES] = synthSeq[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][:370]
# synthSeq[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS] = synthSeq[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS][:370, :]
# # np.save(window.semanticLoopingTab.loadedSynthesisedSequence, synthSeq)
# print window.semanticLoopingTab.loadedSynthesisedSequence

# <codecell>

# gwv.showCustomGraph(window.semanticLoopingTab.unaries[:, 1:])
print np.min(window.semanticLoopingTab.unaries[:, 1:]), np.max(window.semanticLoopingTab.unaries[:, 1:])
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS]
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]
print (window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][:-1]-
       window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][1:])

# <codecell>

# gwv.showCustomGraph(window.semanticLoopingTab.unaries[:, 1:])
print np.min(window.semanticLoopingTab.unaries[:, 1:]), np.max(window.semanticLoopingTab.unaries[:, 1:])
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS]
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]
print (window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][:-1]-
       window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][1:])

# <codecell>

gwv.showCustomGraph(window.semanticLoopingTab.unaries[:, 1:])
print np.min(window.semanticLoopingTab.unaries[:, 1:]), np.max(window.semanticLoopingTab.unaries[:, 1:])
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS]
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]
print (window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][:-1]-
       window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][1:])
print
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS]
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES]
print (window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][:-1]-
       window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][1:])

# <codecell>

gwv.showCustomGraph(window.semanticLoopingTab.unaries[:, 1:])
print np.min(window.semanticLoopingTab.unaries[:, 1:]), np.max(window.semanticLoopingTab.unaries[:, 1:])
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_DESIRED_SEMANTICS].T
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES]
print (window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][:-1]-
       window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES][1:])
print
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS].T
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES]
print (window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][:-1]-
       window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][1:])

# <codecell>

print window.semanticLoopingTab.unaries[arange(3315, 3408), arange(3408-3315)]
# otherFrames = arange(3315, 3408)
gwv.showCustomGraph(np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION]))

# <codecell>

# gwv.showCustomGraph(window.semanticLoopingTab.unaries)
print np.min(window.semanticLoopingTab.unaries[:, 1:]), np.max(window.semanticLoopingTab.unaries[:, 1:])
instanceIdx = 0
chosenFrames = window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][0:]
otherFrames = arange(3315)
print "chosen frames", chosenFrames
print "chosen frames unary", np.sum(window.semanticLoopingTab.unaries[chosenFrames, arange(101)])
print "chosen frames cost", np.sum(np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[chosenFrames[1:-1],
                                                                                                                                     chosenFrames[2:]])
print "other cost", np.sum(np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[arange(2, 101), arange(3, 102)])
# print np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[3321, 2024]
print "chosen frames all costs", np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[chosenFrames[1:-1],
                                                                                                        chosenFrames[2:]]
print "other all costs", np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[arange(2, 101), arange(3, 102)]

# <codecell>

gwv.showCustomGraph(window.semanticLoopingTab.unaries)
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES]
print np.sum(np.load(window.semanticLoopingTab.semanticSequences[1][DICT_TRANSITION_COSTS_LOCATION])[window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][1:-1],
                                                                                                     window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][2:]])
print np.sum(np.load(window.semanticLoopingTab.semanticSequences[1][DICT_TRANSITION_COSTS_LOCATION])[arange(2, 101), arange(3, 102)])

print np.load(window.semanticLoopingTab.semanticSequences[1][DICT_TRANSITION_COSTS_LOCATION])[window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][1:-1],
                                                                                              window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES][2:]]
print np.load(window.semanticLoopingTab.semanticSequences[1][DICT_TRANSITION_COSTS_LOCATION])[arange(2, 101), arange(3, 102)]

# <codecell>

print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES]
print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_DESIRED_SEMANTICS]
chosenFrames = window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][1][DICT_SEQUENCE_FRAMES]
print window.semanticLoopingTab.semanticSequences[1][DICT_FRAME_SEMANTICS][chosenFrames, :]

# <codecell>

window.semanticLoopingTab.loadSynthesisedSequenceAtLocation(window.semanticLoopingTab.loadedSynthesisedSequence)
basePath = "/".join(window.semanticLoopingTab.loadedSynthesisedSequence.split('/')[:-1])+"/"
img = QtGui.QImage(QtCore.QSize(1280, 720), QtGui.QImage.Format_ARGB32)
img.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
painter = QtGui.QPainter(img)

# for i in xrange(len(window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][0][DICT_SEQUENCE_FRAMES])) :
#     window.semanticLoopingTab.showFrame(i)
    
#     painter.drawImage(QtCore.QPoint(0, 0), window.semanticLoopingTab.frameLabel.qImage)
#     painter.drawImage(QtCore.QPoint(0, 0), window.semanticLoopingTab.overlayImg)
#     img.save(basePath+"frame-{0:05d}.png".format(i+1))

# <codecell>

gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/wave3/transition_costs-tara3.npy"))

# <codecell>

# print np.sort(np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()[DICT_BBOXES].keys())
# print np.sort(np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()[DICT_FRAMES_LOCATIONS].keys())
# print np.load("/home/ilisescu/PhD/data/havana/semantic_sequence-blue_car1.npy").item()[DICT_BBOX_CENTERS][2590]
for seqLoc in np.sort(glob.glob("/home/ilisescu/PhD/data/havana/semantic_sequence-*.npy")) :
    tmp = np.load(seqLoc).item()
    print np.max(tmp[DICT_BBOXES].keys()), np.max(tmp[DICT_BBOX_CENTERS].keys()), np.max(tmp[DICT_BBOX_ROTATIONS].keys()), 
    print np.max(tmp[DICT_FOOTPRINTS].keys()), np.max(tmp[DICT_FRAMES_LOCATIONS].keys())
    
#     tmp[DICT_BBOX_CENTERS][np.max(tmp[DICT_BBOXES].keys())] = tmp[DICT_BBOX_CENTERS][np.max(tmp[DICT_BBOX_CENTERS].keys())]
#     del tmp[DICT_BBOX_CENTERS][np.max(tmp[DICT_BBOX_CENTERS].keys())]
    
#     tmp[DICT_BBOX_ROTATIONS][np.max(tmp[DICT_BBOXES].keys())] = tmp[DICT_BBOX_ROTATIONS][np.max(tmp[DICT_BBOX_ROTATIONS].keys())]
#     del tmp[DICT_BBOX_ROTATIONS][np.max(tmp[DICT_BBOX_ROTATIONS].keys())]
    
#     tmp[DICT_FOOTPRINTS][np.max(tmp[DICT_BBOXES].keys())] = tmp[DICT_BBOXES][np.max(tmp[DICT_BBOXES].keys())]
    
#     np.save(seqLoc, tmp)

# <codecell>

## code to copy semantic labels from sprite
# seqLoc = "/media/ilisescu/Data1/PhD/data/wave3/"
seqLoc = "/media/ilisescu/Data1/PhD/data/digger/"
for spriteLoc in np.sort(glob.glob(seqLoc+"sprite-*.npy"))[0:1] :
    sprite = np.load(spriteLoc).item()
    seqName = sprite[DICT_SEQUENCE_NAME]
    sequence = np.load(seqLoc+"semantic_sequence-"+seqName+".npy").item()
#     print np.max(np.abs(sprite[DICT_FRAME_SEMANTICS]-sequence[DICT_FRAME_SEMANTICS]))
    
    fig1 = figure()
    clrs = np.arange(0.0, 1.0+1.0/(len(sequence[DICT_FRAME_SEMANTICS].T)-1), 1.0/(len(sequence[DICT_FRAME_SEMANTICS].T)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
    stackplot(np.arange(len(sequence[DICT_FRAME_SEMANTICS])), np.row_stack(tuple([i for i in sequence[DICT_FRAME_SEMANTICS].T])), colors=clrs)
    
    fig1 = figure()
    clrs = np.arange(0.0, 1.0+1.0/(len(sprite[DICT_FRAME_SEMANTICS].T)-1), 1.0/(len(sprite[DICT_FRAME_SEMANTICS].T)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
    stackplot(np.arange(len(sprite[DICT_FRAME_SEMANTICS])), np.row_stack(tuple([i for i in sprite[DICT_FRAME_SEMANTICS].T])), colors=clrs)
        
    sequence[DICT_FRAME_SEMANTICS][:len(sprite[DICT_FRAME_SEMANTICS]), :] = sprite[DICT_FRAME_SEMANTICS]
    
    fig1 = figure()
    clrs = np.arange(0.0, 1.0+1.0/(len(sequence[DICT_FRAME_SEMANTICS].T)-1), 1.0/(len(sequence[DICT_FRAME_SEMANTICS].T)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
    stackplot(np.arange(len(sequence[DICT_FRAME_SEMANTICS])), np.row_stack(tuple([i for i in sequence[DICT_FRAME_SEMANTICS].T])), colors=clrs)
#     np.save(sequence[DICT_SEQUENCE_LOCATION], sequence)
    print "saved", sequence[DICT_SEQUENCE_LOCATION]

# <codecell>

tmp = np.load("/media/ilisescu/Data1/PhD/data/wave3/fullLength-sprites/semantic_sequence-daniel3.npy").item()[DICT_FRAME_SEMANTICS]
fig1 = figure()
clrs = np.arange(0.0, 1.0+1.0/(len(tmp.T)-1), 1.0/(len(tmp.T)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
stackplot(np.arange(len(tmp)), np.row_stack(tuple([i for i in tmp.T])), colors=clrs)

# <codecell>

synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/wave-no_learning-L2_cost_normalized_thresholded/synthesised_sequence.npy").item()
instanceIdx = 1
print synthSeq[DICT_USED_SEQUENCES][synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]]
semSeq = np.load(synthSeq[DICT_USED_SEQUENCES][synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]]).item()
transitionCosts = np.load(semSeq[DICT_TRANSITION_COSTS_LOCATION])
chosenFrames = synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][0:]
otherFrames = arange(3315)
print "chosen frames", chosenFrames
print "jumps", chosenFrames[:-1] - chosenFrames[1:]
# print "chosen frames unary", np.sum(window.semanticLoopingTab.unaries[chosenFrames, arange(101)])
print "chosen frames cost", np.sum(transitionCosts[chosenFrames[0:-1], chosenFrames[1:]])
# print "other cost", np.sum(transitionCosts[arange(2, 101), arange(3, 102)])
# print np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[3321, 2024]
print "chosen frames all costs", transitionCosts[chosenFrames[0:-1], chosenFrames[1:]]
# print "other all costs", transitionCosts[arange(2, 101), arange(3, 102)]

# <codecell>

synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/wave-no_learning-L2_cost_normalized_thresholded/synthesised_sequence.npy").item()
instanceIdx = 1
print synthSeq[DICT_USED_SEQUENCES][synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]]
semSeq = np.load(synthSeq[DICT_USED_SEQUENCES][synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]]).item()
transitionCosts = sequenceTransitionCost#np.load(semSeq[DICT_TRANSITION_COSTS_LOCATION])
chosenFrames = synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][0:]
otherFrames = arange(3315)
print "chosen frames", chosenFrames
print "jumps", chosenFrames[:-1] - chosenFrames[1:]
# print "chosen frames unary", np.sum(window.semanticLoopingTab.unaries[chosenFrames, arange(101)])
print "chosen frames cost", np.sum(transitionCosts[chosenFrames[0:-1], chosenFrames[1:]])
# print "other cost", np.sum(transitionCosts[arange(2, 101), arange(3, 102)])
# print np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[3321, 2024]
print "chosen frames all costs", transitionCosts[chosenFrames[0:-1], chosenFrames[1:]]
# print "other all costs", transitionCosts[arange(2, 101), arange(3, 102)]

# <codecell>

synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/wave-no_learning-L2_cost_normalized_thresholded/synthesised_sequence.npy").item()
instanceIdx = 1
print synthSeq[DICT_USED_SEQUENCES][synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]]
semSeq = np.load(synthSeq[DICT_USED_SEQUENCES][synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]]).item()
transitionCosts = sequenceTransitionCost#np.load(semSeq[DICT_TRANSITION_COSTS_LOCATION])
chosenFrames = window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]#synthSeq[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][0:]
otherFrames = arange(3315)
print "chosen frames", chosenFrames
print "jumps", chosenFrames[:-1] - chosenFrames[1:]
# print "chosen frames unary", np.sum(window.semanticLoopingTab.unaries[chosenFrames, arange(101)])
print "chosen frames cost", np.sum(transitionCosts[chosenFrames[0:-1], chosenFrames[1:]])
# print "other cost", np.sum(transitionCosts[arange(2, 101), arange(3, 102)])
# print np.load(window.semanticLoopingTab.semanticSequences[instanceIdx][DICT_TRANSITION_COSTS_LOCATION])[3321, 2024]
print "chosen frames all costs", transitionCosts[chosenFrames[0:-1], chosenFrames[1:]]
# print "other all costs", transitionCosts[arange(2, 101), arange(3, 102)]

# <codecell>

## this turns distance matrix into transition costs and updates the sequence dict
distMatName = "vanilla_distMat"
# distMatName = "learned_distMat"
transCostName = "transition_costs"
# transCostName = "learned_transition_costs"
dataset = "wave3"#"toy"
for s in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/"+dataset+"/semantic_sequence*.npy"))[0:1]:
    sequence = np.load(s).item()
    seqLoc = "/media/ilisescu/Data1/PhD/data/"+dataset+"/"
    seqName = sequence[DICT_SEQUENCE_NAME]
    print sequence[DICT_TRANSITION_COSTS_LOCATION]
    distMat = np.load(seqLoc+seqName+"-"+distMatName+".npy")
    
    ## filter ##
    filterSize = 4
    optimizedDistMat = vtu.filterDistanceMatrix(distMat/np.max(distMat), filterSize, True)
    ## if using vanilla
    if True :
        optimizedDistMat = optimizedDistMat[1:optimizedDistMat.shape[1], 0:-1]
        correction = 1
    else :
        correction = 0

#     #### this is just normalizing to [0, 1] A ####
#     optimizedDistMat = optimizedDistMat/np.max(optimizedDistMat)
#     ########

    ## don't want to jump too close so increase costs in a window
    minJumpLength = 20
    tmp = (np.triu(np.ones(optimizedDistMat.shape), k=minJumpLength) +
           np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
           np.eye(optimizedDistMat.shape[0], k=1))
    tmp[tmp == 0] = 10.0
    optimizedDistMat *= tmp
    
    
#     #### this is just normalizing to [0, 1] B ####
#     sequenceTransitionCost = optimizedDistMat
#     ########
    
    #### this turns to a probability and then does the same as the unaries ####
    sequenceTransitionCost = vtu.dist2prob(optimizedDistMat, 0.1, True)
    
    impossibleTransitions = sequenceTransitionCost <= 0.0
    ## cost is -log(prob)
    sequenceTransitionCost[np.negative(impossibleTransitions)] = -np.log(sequenceTransitionCost[np.negative(impossibleTransitions)])
    ## if prob == 0.0 then set maxCost
    sequenceTransitionCost[impossibleTransitions] = GRAPH_MAX_COST
    ########
    
    ## do the thresholding based on how many jumps I want to keep per frame
    desiredPercentage = 0.1 ## desired percentage of transitions to keep
    jumpsToKeep = int(sequenceTransitionCost.shape[0]*desiredPercentage)
    sequenceTransitionCost[np.arange(sequenceTransitionCost.shape[0]).repeat(sequenceTransitionCost.shape[0]-jumpsToKeep),
                           np.argsort(sequenceTransitionCost, axis=-1)[:, jumpsToKeep:].flatten()] = GRAPH_MAX_COST


    ## adding extra rows and columns so that the optimized matrix has the same dimensions as distMat
    ## for the indices that were cut out I put zero cost for jumps to frames that can still be used after optimization
    sequenceTransitionCost = np.concatenate((np.ones((sequenceTransitionCost.shape[0], filterSize))*np.max(sequenceTransitionCost),
                                             sequenceTransitionCost,
                                             np.ones((sequenceTransitionCost.shape[0], filterSize+correction))*np.max(sequenceTransitionCost)), axis=1)
    sequenceTransitionCost = np.concatenate((np.roll(np.concatenate((np.zeros((filterSize, 1)),
                                                                     np.ones((filterSize, distMat.shape[0]-1))*np.max(sequenceTransitionCost)), axis=1), filterSize, axis=1),
                                             sequenceTransitionCost,
                                             np.roll(np.concatenate((np.zeros((filterSize+correction, 1)),
                                                                     np.ones((filterSize+correction, distMat.shape[0]-1))*np.max(sequenceTransitionCost)), axis=1), filterSize, axis=1)), axis=0)

    #### this finds a threshold based on how many transitions are kept ####
#     maxCost = GRAPH_MAX_COST
#     desiredPercentage = 0.1 ## desired percentage of transitions to keep
#     ## finding best threshold
# #     step = 0.025
# #     steps = arange(step, 1.0+step, step)
# #     percentages = np.array([np.argwhere(sequenceTransitionCost <= t).shape[0]/float(np.prod(sequenceTransitionCost.shape)) for t in steps])
# #     diffs = np.abs(percentages-desiredPercentage)
# #     thresh = steps[np.max(np.argwhere(diffs == np.min(diffs)))]
#     bestT = 0.0
#     bestDiff = 1.0
#     t = 0.5
#     for i in xrange(50) :
#         p = np.argwhere(sequenceTransitionCost <= t).shape[0]/float(np.prod(sequenceTransitionCost.shape))
#         if np.abs(p - desiredPercentage) < desiredPercentage*0.1 :
#             bestT = t
#             break

#         if bestDiff > np.abs(p - desiredPercentage) :
#             bestDiff = np.abs(p - desiredPercentage)
#             bestT = t

#         if p > desiredPercentage :
#             t *= 0.5
#         else :
#             t *= 1.5
#     thresh = bestT
#     sequenceTransitionCost[sequenceTransitionCost > thresh] = maxCost
    ########


#     ## add some number to the costs to give some cost to following the timeline, which should not influence following the timeline loop-wise but should reduce the length of
#     ## transition animations
# #     sequenceTransitionCost += 0.1

    gwv.showCustomGraph(sequenceTransitionCost)
#     gwv.showCustomGraph(np.load(seqLoc+transCostName+"-"+seqName+".npy"))

#     sequence[DICT_TRANSITION_COSTS_LOCATION] = seqLoc+transCostName+"-"+seqName+".npy"
#     np.save(sequence[DICT_TRANSITION_COSTS_LOCATION], sequenceTransitionCost)
#     np.save(sequence[DICT_SEQUENCE_LOCATION], sequence)
    print sequence[DICT_TRANSITION_COSTS_LOCATION]
# #     print sequence.keys()

# <codecell>


# <codecell>

gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/wave3/transition_costs-aron3.npy"))
# jumpsToKeep = int(sequenceTransitionCost.shape[0]*desiredPercentage)
# print np.argsort(sequenceTransitionCost, axis=-1)[:, jumpsToKeep:].flatten()
# print np.arange(sequenceTransitionCost.shape[0]).repeat(sequenceTransitionCost.shape[0]-jumpsToKeep)

# <codecell>

for s in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/"+dataset+"/semantic_sequence*.npy"))[0:]:
    sequence = np.load(s).item()
    print sequence[DICT_SEQUENCE_NAME]
    print np.argwhere(np.load(sequence[DICT_TRANSITION_COSTS_LOCATION]) == maxCost+0.1).shape[0]/float(np.prod(sequenceTransitionCost.shape))

# <codecell>

## this removes frames from each sprite (so that the number of frames is divisible by 8 or 16 for computing blocked distance matrix)
dataset = "wave3"
frameLocs = np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/"+dataset+"/frame-0*.png"))
desiredNumFrames = 3040
print frameLocs.shape
for s in np.sort(glob.glob("/media/ilisescu/Data1/PhD/data/"+dataset+"/sprite*.npy"))[0:] :
    print s
    sprite = np.load(s).item()
    for i in xrange(desiredNumFrames, len(frameLocs)) :
        del sprite[DICT_BBOXES][i]
        del sprite[DICT_BBOX_ROTATIONS][i]
        del sprite[DICT_BBOX_CENTERS][i]
        del sprite[DICT_FRAMES_LOCATIONS][i]
        print "frame-{0:05d}.png".format(i+1)
#         os.remove("/media/ilisescu/Data1/PhD/data/"+dataset+"/"+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-{0:05d}.png".format(i+1))
#     np.save(s, sprite)

# <codecell>

# def estimateFutureCost(alpha, p, distanceMatrixFilt, weights) :

#     distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
#     distMat = distMatFilt ** p
    
#     last = np.copy(distMat)
#     current = np.zeros(distMat.shape)
    
#     ## while distance between last and current is larger than threshold
#     iterations = 0 
#     while np.linalg.norm(last - current) > 0.1 : 
#         for i in range(distMat.shape[0]-1, -1, -1) :
#             m = np.min(distMat*weights[1:distanceMatrixFilt.shape[1], 0:-1], axis=1)
#             distMat[i, :] = (distMatFilt[i, :] ** p) + alpha*m
            
#         last = np.copy(current)
#         current = np.copy(distMat)
        
#         sys.stdout.write('\r' + "Iteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
#         sys.stdout.flush()
#         iterations += 1
    
#     print
#     print 'finished in', iterations, 'iterations'
    
#     return distMat

# distMat = np.load("/media/ilisescu/Data1/PhD/data/candle_wind/vanilla_distMat.npy")
# seqLoc = "/media/ilisescu/Data1/PhD/data/candle_wind/"
# seqName = "candle_wind1"
# filterSize = 2
# if False :
#     optimizedDistMat = estimateFutureCost(0.999, 2.0, vtu.filterDistanceMatrix(distMat/np.max(distMat), filterSize, True),
#                                                 np.ones(np.array(distMat.shape)-(filterSize*2)))
# else :
#     optimizedDistMat = vtu.filterDistanceMatrix(distMat/np.max(distMat), filterSize, True)
#     optimizedDistMat = optimizedDistMat[1:optimizedDistMat.shape[1], 0:-1]

# sequenceTransitionCost = optimizedDistMat/np.max(optimizedDistMat)

# ## don't want to jump too close so increase costs in a window
# minJumpLength = 20
# tmp = (np.triu(np.ones(sequenceTransitionCost.shape), k=minJumpLength) +
#        np.tril(np.ones(sequenceTransitionCost.shape), k=-minJumpLength) +
#        np.eye(sequenceTransitionCost.shape[0], k=1))
# tmp[tmp == 0] = 10.0
# sequenceTransitionCost *= tmp


# ## adding extra rows and columns so that the optimized matrix has the same dimensions as distMat
# ## for the indices that were cut out I put zero cost for jumps to frames that can still be used after optimization
# sequenceTransitionCost = np.concatenate((np.ones((sequenceTransitionCost.shape[0], filterSize)),
#                                          sequenceTransitionCost,
#                                          np.concatenate((np.zeros((1, filterSize+1)),
#                                                          np.ones((sequenceTransitionCost.shape[0]-1, filterSize+1))), axis=0)), axis=1)
# sequenceTransitionCost = np.concatenate((np.roll(np.concatenate((np.zeros((filterSize, 1)),
#                                                                  np.ones((filterSize, distMat.shape[0]-1))), axis=1), filterSize, axis=1),
#                                          sequenceTransitionCost,
#                                          np.ones((filterSize+1, distMat.shape[0]))), axis=0)

# maxCost = GRAPH_MAX_COST
# ## threshold
# sequenceTransitionCost[sequenceTransitionCost > 0.25] = maxCost
# ## add some number to the costs to give some cost to following the timeline, which should not influence following the timeline loop-wise but should reduce the length of
# ## transition animations
# sequenceTransitionCost += 0.1

# # np.save("tmp.npy", sequenceTransitionCost)

# # sequenceTransitionCost = np.load("tmp.npy")
# # gwv.showCustomGraph(np.log(sequenceTransitionCost.T))

# sequence = np.load(seqLoc+"semantic_sequence-"+seqName+".npy").item()
# # sequence[DICT_TRANSITION_COSTS_LOCATION] = seqLoc+"transition_costs-"+seqName+".npy"
# np.save(sequence[DICT_TRANSITION_COSTS_LOCATION], sequenceTransitionCost)
# # np.save(seqLoc+"semantic_sequence-"+seqName+".npy", sequence)
# print sequence.keys()

# <codecell>

semanticSequence = window.semanticLoopingTab.semanticSequences[0]
desiredSemanticIdx = 1
## set starting frame
desiredStartFrame = frameIdx = int(np.argwhere(semanticSequence[DICT_FRAME_SEMANTICS][:, desiredSemanticIdx] >= 0.9)[0])
print "starting from", desiredStartFrame

distVariance = 0.0005

desiredSemantics = np.zeros((301, 3))
desiredSemantics[:, desiredSemanticIdx] = 1.0

unaries = vectorisedMinusLogMultiNormalMultipleMeans(semanticSequence[DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, True).T
unaries[:, 0] = GRAPH_MAX_COST
unaries[desiredStartFrame, 0] = 0.0

numNodes = len(desiredSemantics)
numLabels = sequenceTransitionCost.shape[0]
gm = opengm.gm(np.ones(numNodes,dtype=opengm.label_type)*numLabels)


fids = gm.addFunctions(unaries.T)
# add first order factors
gm.addFactors(fids, arange(numNodes))

pairIndices = np.array([np.arange(numNodes-1), np.arange(1, numNodes)]).T

## add function for row-nodes pairwise cost
#         fid = gm.addFunction(sequenceTransitionCost+np.random.rand(sequenceTransitionCost.shape[0], sequenceTransitionCost.shape[1])*0.01-0.005)

## randomize
bestTransitions = np.argsort(sequenceTransitionCost, axis=-1)
jumpLength = np.abs(bestTransitions-arange(sequenceTransitionCost.shape[0]).reshape((sequenceTransitionCost.shape[0], 1)))
minJumpLength = 10
numTop = 15
topBest = np.array([bestTransitions[i, jumpLength[i, :] >= minJumpLength][:numTop] for i in xrange(sequenceTransitionCost.shape[0])])
cost = np.copy(sequenceTransitionCost)
for i in xrange(sequenceTransitionCost.shape[0]) :
    cost[i, topBest[i, :]] += (np.random.rand(numTop)*0.1-0.05)
    

fid = gm.addFunction(cost)
## add second order factors
gm.addFactors(fid, pairIndices)
inferer = opengm.inference.DynamicProgramming(gm=gm)
inferer.infer()

minCostTraversal = np.array(inferer.arg(), dtype=int)
print minCostTraversal, gm.evaluate(inferer.arg())

