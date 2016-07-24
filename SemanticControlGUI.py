# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
# %pylab
import numpy as np
import sys
# from IPython.display import clear_output

import cv2

import time
import glob
import datetime

import os

from PIL import Image
from PySide import QtCore, QtGui


import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import SemanticsDefinitionTabGUI as sdt
import SemanticLoopingTabGUI as slt

# import shutil, errno

# def copyanything(src, dst):
#     try:
#         shutil.copytree(src, dst)
#     except OSError as exc: # python >2.5
#         if exc.errno == errno.ENOTDIR:
#             shutil.copy(src, dst)
#         else: raise

app = QtGui.QApplication(sys.argv)

DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_NUM_SEMANTICS = "number_of_semantic_classes"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"
DICT_FRAME_COMPATIBILITY_LABELS = 'compatibiliy_labels_per_frame'
DICT_LABELLED_FRAMES = 'labelled_frames' ## includes the frames labelled for the semantic labels (the first [DICT_FRAME_SEMANTICS].shape[1])
DICT_NUM_EXTRA_FRAMES = 'num_extra_frames' ## same len as DICT_LABELLED_FRAMES
DICT_CONFLICTING_SEQUENCES = 'conflicting_sequences'
DICT_COMPATIBLE_SEQUENCES = 'compatible_sequences'
DICT_DISTANCE_MATRIX_LOCATION = 'sequence_precomputed_distance_matrix_location' ## for label propagation
DICT_SEQUENCE_LOCATION = "sequence_location"

DICT_USED_SEQUENCES = "used_semantic_sequences"
DICT_SEQUENCE_INSTANCES = "sequence_instances"
DICT_SEQUENCE_BG = "sequence_background_image"

DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SEQUENCE_IDX = 'semantic_sequence_idx' # index of the instantiated sem sequence in the list of all used sem sequences for a synthesised sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_OFFSET = "instance_offset"
DICT_SCALE = "instance_scale"

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

class FormattedKeystrokeLabel(QtGui.QLabel):
    def __init__(self, text="", parent=None):
        super(FormattedKeystrokeLabel, self).__init__(text, parent)
        
        self.extraSpace = 3
        
        
        self.wordsToRender = []
        self.wordsAreBold = []
        self.wordsWidths = []
        self.spaceWidth = QtGui.QFontMetrics(QtGui.QFont()).width(" ")
        self.wordHeight = QtGui.QFontMetrics(QtGui.QFont()).height()
        totalWidth = 0
        
        ## getting words from input
        for word in text.split(" ") :
            if "<b>" in word :
                self.wordsAreBold.append(True)
                self.wordsToRender.append("".join(("".join(word.split("<b>"))).split("</b>")))
            else :
                self.wordsAreBold.append(False)
                self.wordsToRender.append(word)
                
            font = QtGui.QFont()
            if self.wordsAreBold[-1] :
                font.setWeight(QtGui.QFont.Bold)
            else :
                font.setWeight(QtGui.QFont.Normal)
            
            self.wordsWidths.append(QtGui.QFontMetrics(font).width(self.wordsToRender[-1]))
            totalWidth += self.wordsWidths[-1]
            
        totalWidth += self.spaceWidth*(len(self.wordsToRender)-1)
        
        ## resize label
        self.setFixedSize(totalWidth+self.extraSpace*2, self.wordHeight+self.extraSpace*2)
        
    def paintEvent(self, event) :
        painter = QtGui.QPainter(self)
        padding = 1
        
        currentX = self.extraSpace
        for word, isBold, wordWidth in zip(self.wordsToRender, self.wordsAreBold, self.wordsWidths) :
            wordRect = QtCore.QRect(currentX, self.extraSpace, wordWidth, self.wordHeight)
            if word != "or" and word != "to" :
                ## draw rectangle
                painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(225, 225, 225)))
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 0), 0, 
                                                  QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))

                painter.drawRect(QtCore.QRect(wordRect.left()-padding, wordRect.top()-padding,
                                              wordRect.width()+padding*2, wordRect.height()+padding*2))
                
            ## draw text
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0)))
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 255), 3, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))

            if isBold :
                font = QtGui.QFont()
                font.setWeight(QtGui.QFont.Bold)
                painter.setFont(font)
            else :
                painter.setFont(QtGui.QFont())

            painter.drawText(wordRect, word)
                
            currentX += (self.spaceWidth + wordWidth)
        painter.end()

# <codecell>

class HelpDialog(QtGui.QDialog):
    def __init__(self, parent=None, title=""):
        super(HelpDialog, self).__init__(parent)
        
        self.createGUI()
        
        self.setWindowTitle(title)
        
    def doneClicked(self):
        self.done(0)
    
    def createGUI(self):
        
        self.doneButton = QtGui.QPushButton("Done")
         
        ## SIGNALS ##
        
        self.doneButton.clicked.connect(self.doneClicked)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QGridLayout()
        
        idx = 0
        mainLayout.addWidget(QtGui.QLabel("<b>Definition</b>"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(QtGui.QLabel("<b>Synthesis</b>"), idx, 4, 1, 1, QtCore.Qt.AlignLeft);idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>Return</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Track Forward"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>u</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Move selected instance up in the list"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>Backspace</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Track Backwards"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>d</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Move selected instance down in the list"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Escape</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Stop tracking or batch segmentation"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>0</b> to <b>9</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Request given semantics for selected instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Delete</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Delete current frame's bounding box"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>r</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Refine synthesised sequence"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Enter</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Set bounding box for current frame"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Space</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Extend synthesised sequence from the <b>end</b>"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>c</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Copy current bounding box"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("Shift <b>Space</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Extend synthesised sequence from the <b>current</b> frame"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>v</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Paste current bounding box"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>Space</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Extend synthesised sequence of selected sequences from the <b>current</b> frame"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>s</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Save tracked sprites"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>t</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Tag conflict between <b>2<\b> instances or frame of <b>1<\b> selected instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>m</b>"), idx, 0, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Switch mode (<font color=\"red\"><b>bbox</b></font> vs <font color=\"blue\"><b>scribble</b></font>)"), idx, 1, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>Delete</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Delete currently selected instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("Ctrl <b>s</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Save synthesised sequence"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        mainLayout.addWidget(FormattedKeystrokeLabel("<b>a</b>"), idx, 3, 1, 1, QtCore.Qt.AlignRight)
        mainLayout.addWidget(QtGui.QLabel("Add new instance"), idx, 4, 1, 1, QtCore.Qt.AlignLeft)
        idx+=1
        
        verticalLine =  QtGui.QFrame()
        verticalLine.setFrameStyle(QtGui.QFrame.VLine)
        verticalLine.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        mainLayout.addWidget(verticalLine, 0, 2, idx, 1)
        
        horizontalLine =  QtGui.QFrame()
        horizontalLine.setFrameStyle(QtGui.QFrame.HLine)
        horizontalLine.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        mainLayout.addWidget(horizontalLine,idx, 0 , 1, 5)
        idx+=1
        
        mainLayout.addWidget(self.doneButton, idx, 0, 1, 5, QtCore.Qt.AlignCenter)
        idx+=1
        
        self.setLayout(mainLayout)

def showHelp(parent=None, title="Keyboard Shortcuts") :
    helpDialog = HelpDialog(parent, title)
    exitCode = helpDialog.exec_()
    
    return exitCode

# <codecell>

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        
        if not os.path.isdir("logFiles/") :
            os.mkdir("logFiles/")
            
        print "LOG:Starting", datetime.datetime.now()
        with open("logFiles/log-"+str(datetime.datetime.now()), "w+") as f:
            f.write("LOG:DEFINITION:Switch-&-" + str(datetime.datetime.now()) + "\n")
        
        if os.path.isfile("semantic_control_recent_loads.npy") :
            self.recentLoadedFiles = np.load("semantic_control_recent_loads.npy").item()
        else :
            self.recentLoadedFiles = {'raw_sequences':[], 'synthesised_sequences':[]}
        
        self.createGUI()
        
        self.showLoading(False)
        
        self.setWindowTitle("Action-based Video Synthesis")
        self.resize(1920, 950)
        
        self.readyForVT = False
        self.firstLoad = True
        self.dataLocation = ""
        self.semanticsDefinitionTab.setFocus()
    
    def openSequence(self) :
        return 
        
    def tabChanged(self, tabIdx) :
        if tabIdx == 0 :
            self.semanticsDefinitionTab.setFocus()
            
            ##
            with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                f.write("LOG:DEFINITION:Switch-&-" + str(datetime.datetime.now()) +"\n")
                
        elif tabIdx == 1 :
            self.semanticLoopingTab.setFocus()
            
            ##
            with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
                f.write("LOG:LOOPING:Switch-&-" + str(datetime.datetime.now()) +"\n")

    def closeEvent(self, event) :
        self.semanticsDefinitionTab.cleanup()
        self.semanticLoopingTab.cleanup()
        
        
        ##
        with open(np.sort(glob.glob("logFiles/log-*"))[-1], "a+") as f :
            f.write("LOG:Closing-&-" + str(datetime.datetime.now()) +"\n")
            
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

    def showHelpDialog(self) :
        showHelp(self)
        
    def loadRawSequencePressed(self, triggeredAction) :
        if triggeredAction.iconText() == "Find Location on Disk" :
            newLocation = self.semanticsDefinitionTab.loadFrameSequencePressed()
            if newLocation != "" :
                if len(self.recentLoadedFiles['raw_sequences']) > 9 :
                    del self.recentLoadedFiles['raw_sequences'][9]
                self.recentLoadedFiles['raw_sequences'].insert(0, newLocation)
                np.save("semantic_control_recent_loads.npy", self.recentLoadedFiles)
        else :
            self.semanticsDefinitionTab.loadFrameSequence(triggeredAction.iconText())
        
    def loadSynthesisedSequencePressed(self, triggeredAction) :
        if triggeredAction.iconText() == "Find Location on Disk" :
            newLocation = self.semanticLoopingTab.loadSynthesisedSequence()
            if newLocation != "" :
                if len(self.recentLoadedFiles['synthesised_sequences']) > 9 :
                    del self.recentLoadedFiles['synthesised_sequences'][9]
                self.recentLoadedFiles['synthesised_sequences'].insert(0, newLocation)
                np.save("semantic_control_recent_loads.npy", self.recentLoadedFiles)
        else :
            self.semanticLoopingTab.loadSynthesisedSequenceAtLocation(triggeredAction.iconText())
        
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
        
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/theme_park_sunny", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/windows", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/digger", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/toy", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/elevators", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/candle_wind", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/home/ilisescu/PhD/data/street", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/wave1", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/home/ilisescu/PhD/data/tutorial_sequence/", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/flowers/", self)#dataPath+dataSet)
#         self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("/media/ilisescu/Data1/PhD/data/plane_dep/", self)#dataPath+dataSet)
        self.semanticsDefinitionTab = sdt.SemanticsDefinitionTab("", self)#dataPath+dataSet)
        
#         self.semanticLoopingTab = SemanticLoopingTab(100, dataPath+"synthesisedSequences/waveFull/synthesised_sequence.npy", self)
#         self.semanticLoopingTab = SemanticLoopingTab(100, dataPath+"synthesisedSequences/waveFullBusier/synthesised_sequence.npy", self)
#         self.semanticLoopingTab = SemanticLoopingTab(250, dataPath+"synthesisedSequences/theme_park/synthesised_sequence.npy", self)
#         self.semanticLoopingTab = SemanticLoopingTab(250, dataPath+"synthesisedSequences/theme_park_mixedCompatibility/synthesised_sequence.npy", self)
#         self.semanticLoopingTab = SemanticLoopingTab(250, dataPath+"synthesisedSequences/tetris/synthesised_sequence.npy", self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(500, dataPath+"synthesisedSequences/havana_new_semantics/synthesised_sequence.npy", self)
#         self.semanticLoopingTab = SemanticLoopingTab(100, dataPath+"synthesisedSequences/multipleCandles/synthesised_sequence.npy", self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(200, dataPath+"synthesisedSequences/havana_semantic_compatiblity/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/plane_departures/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/wave_by_numbers/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/wave_by_numbers_fatterbar/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/wave_by_numbers_fattestbar/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/flowers/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/super_mario_planes/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/street_complex/synthesised_sequence.npy", True, self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(50, dataPath+"synthesisedSequences/street_complex_test/synthesised_sequence.npy", True, self)
        self.semanticLoopingTab = slt.SemanticLoopingTab(200, "", self)
#         self.semanticLoopingTab = slt.SemanticLoopingTab(800, "", self)

        self.tabWidget = QtGui.QTabWidget()
        self.tabWidget.addTab(self.semanticsDefinitionTab, self.tr("Define Input Sequences"))
        self.tabWidget.addTab(self.semanticLoopingTab, self.tr("Action-based Synthesis"))
        
        self.tabWidget.setCurrentIndex(0)
        self.semanticsDefinitionTab.setFocus()
        
#         self.tabWidget.setCurrentIndex(1)
#         self.semanticLoopingTab.setFocus()
        
        ## SIGNALS ##
        
        self.openSequenceButton.clicked.connect(self.openSequence)
        
        self.tabWidget.currentChanged.connect(self.tabChanged)
        
        ## LAYOUTS ##
        
        self.mainBox = QtGui.QGroupBox("Main Controls")
        self.mainBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
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
        self.mainBox.setLayout(mainBoxLayout)
        
#         mainLayout = QtGui.QVBoxLayout()
#         mainLayout.addWidget(self.tabWidget)
#         mainLayout.addWidget(mainBox)
        
        self.setCentralWidget(self.tabWidget)
        
        ## MENU ACTIONS ##
#         loadRawFrameSequenceAction = QtGui.QAction("Load &Raw Frame Sequence", self)
#         loadRawFrameSequenceAction.triggered.connect(self.loadRawSequencePressed)
        loadRawFrameSequenceMenu = QtGui.QMenu("Load &Raw Frame Sequence", self)
        loadRawFrameSequenceMenu.triggered.connect(self.loadRawSequencePressed)
        loadRawFrameSequenceMenu.addAction("Find Location on Disk")
        loadRawFrameSequenceMenu.addSeparator()
        for location in self.recentLoadedFiles['raw_sequences'] :
            loadRawFrameSequenceMenu.addAction(location)
        
        synthesiseNewSequenceAction = QtGui.QAction("Synthesise &New Sequence", self)
        synthesiseNewSequenceAction.triggered.connect(self.semanticLoopingTab.newSynthesisedSequence)
        loadSynthesisedSequenceMenu = QtGui.QMenu("Load &Synthesised Sequence", self)
        loadSynthesisedSequenceMenu.triggered.connect(self.loadSynthesisedSequencePressed)
        loadSynthesisedSequenceMenu.addAction("Find Location on Disk")
        loadSynthesisedSequenceMenu.addSeparator()
        for location in self.recentLoadedFiles['synthesised_sequences'] :
            loadSynthesisedSequenceMenu.addAction(location)
        
        
        loadInputSequenceAction = QtGui.QAction("Load &Input Sequence", self)
        loadInputSequenceAction.triggered.connect(self.semanticLoopingTab.loadSemanticSequence)
        setBackgroundImageAction = QtGui.QAction("Set &Background Image", self)
        setBackgroundImageAction.triggered.connect(self.semanticLoopingTab.setBgImage)
        
        helpAction = QtGui.QAction("&Help", self)
        helpAction.setShortcut('Ctrl+H')
        helpAction.triggered.connect(self.showHelpDialog)
    
        ## MENU BAR ##
        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addMenu(loadRawFrameSequenceMenu)
        fileMenu.addSeparator()
        fileMenu.addAction(synthesiseNewSequenceAction)
        fileMenu.addMenu(loadSynthesisedSequenceMenu)
        fileMenu.addAction(loadInputSequenceAction)
        fileMenu.addAction(setBackgroundImageAction)
        
        
        aboutMenu = self.menuBar().addMenu("&About")
        aboutMenu.addAction(helpAction)
        

# <codecell>

# %%capture
# def main():
#     window = Window()
#     window.show()
#     app.exec_()
#     del window

# if __name__ == "__main__":
#     main()
window = Window()
window.show()
app.exec_()

# <codecell>

# tmp = np.load("/media/ilisescu/Data1/PhD/data/toy/semantic_sequence-toy1.npy").item()
# print tmp.keys()
# del tmp[DICT_]
# tmp[DICT_TRANSITION_COSTS_LOCATION] = "/media/ilisescu/Data1/PhD/data/toy/transition_costs_no_normalization_toy1.npy"
# np.save(tmp[DICT_SEQUENCE_LOCATION], tmp)

# <codecell>

# tmp = np.load("/media/ilisescu/Data1/PhD/data/candle_wind/semantic_sequence-candle_wind1.npy").item()
# del tmp[DICT_BBOXES]
# del tmp[DICT_BBOX_CENTERS]
# del tmp[DICT_BBOX_ROTATIONS]
# del tmp[DICT_FOOTPRINTS]
# del tmp[DICT_MASK_LOCATION]
# del tmp[DICT_PATCHES_LOCATION]
# tmp[DICT_PATCHES_LOCATION] = "/media/ilisescu/Data1/PhD/data/candle_wind/preloaded_patches-candle_wind1.npy"
# del tmp[DICT_FRAME_SEMANTICS]
# tmp[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/candle_wind/candle_wind1-vanilla_distMat.npy"
# print tmp.keys()
# del tmp[DICT_USED_SEQUENCES][1]
# print tmp[DICT_USED_SEQUENCES]
# np.save(tmp[DICT_SEQUENCE_LOCATION], tmp)

# <codecell>

# for seqLoc in glob.glob("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-*") :
#     seq = np.load(seqLoc).item()
#     print seqLoc
#     if DICT_FRAME_SEMANTICS in seq.keys() :
#         del seq[DICT_FRAME_SEMANTICS]
#     if DICT_LABELLED_FRAMES in seq.keys() :
#         del seq[DICT_LABELLED_FRAMES]
#     if DICT_NUM_EXTRA_FRAMES in seq.keys() :
#         del seq[DICT_NUM_EXTRA_FRAMES]
#     if DICT_NUM_SEMANTICS in seq.keys() :
#         del seq[DICT_NUM_SEMANTICS]

#     np.save(seq[DICT_SEQUENCE_LOCATION], seq)

# <codecell>

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-daniel1.npy").item()
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/daniel1-vanilla_distMat.npy"
# seq[DICT_TRANSITION_COSTS_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/transition_costs-daniel1.npy"
# np.save(seq[DICT_SEQUENCE_LOCATION], seq)

# <codecell>

# gwv.showCustomGraph(np.load("/home/ilisescu/PhD/data/street/transition_costs_no_normalization-pidgeon1.npy"))

# <codecell>

# for patchesLoc in glob.glob("/media/ilisescu/Data1/PhD/data/wave1/preloaded_patches-*.npy") :
#     print patchesLoc; sys.stdout.flush()
#     patches = np.load(patchesgimpLoc).item()
#     for key in patches.keys() :
#         if "visible_indices" in patches[key] :
#             patches[key]["visible_indices"] = patches[key]["visible_indices"].astype(np.uint16)
#     np.save(patchesLoc, patches)
#     del patches

# <codecell>

# print np.load("/media/ilisescu/Data1/PhD/data/synthesisedSequences/waveFullBusier/synthesised_sequence.npy").item()[DICT_SEQUENCE_INSTANCES][0].keys()
# seq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/wave_by_numbers/synthesised_sequence.npy").item()
# seq[DICT_USED_SEQUENCES] = []
# np.save("/home/ilisescu/PhD/data/synthesisedSequences/wave_by_numbers/synthesised_sequence.npy", seq)

# <codecell>

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-aron1.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1881-1806, 2248-1806], [2346-1806, 2577-1806]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/aron1-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# seq[DICT_TRANSITION_COSTS_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/transition_costs_no_normalization_aron1.npy"
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-moos1.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1140-1122, 2660-1122], [2220-1122, 2527-1122]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/moos1-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave3/semantic_sequence-sara3.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[846-806, 1245-806], [1495-806, 1807-806]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave3/sara3-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave3/semantic_sequence-tara3.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1293-1060, 1530-1060], [1383-1060, 1596-1060]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave3/tara3-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-peter1.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1450-1252, 1780-1252], [2222-1252, 2472-1252]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/peter1-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave2/semantic_sequence-james2.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1268-1194, 1346-1194], [1640-1194, 1850-1194]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave2/james2-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave2/semantic_sequence-moos2.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[674-470, 1335-470], [1565-470, 1875-470]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave2/moos2-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-james1.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1375-1216, 1993-1216], [2151-1216, 2415-1216]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/james1-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-sara1.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1672-1394, 2430-1394], [2183-1394, 2519-1394]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/sara1-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-tara1.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1643-1346, 2491-1346], [2115-1346, 2438-1346]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/tara1-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave2/semantic_sequence-daniel2.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1434-1320, 1892-1320], [1742-1320, 2143-1320]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave2/daniel2-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave3/semantic_sequence-moos3.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[985-832, 1099-832], [1495-832, 1870-832]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave3/moos3-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave3/semantic_sequence-peter3.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[977-868, 1253-868], [1400-868, 1720-868]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave3/peter3-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave1/semantic_sequence-ferran1.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1770-1478, 1947-1478], [2290-1478, 2557-1478]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4], [4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave1/ferran1-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# seq = np.load("/media/ilisescu/Data1/PhD/data/wave2/semantic_sequence-aron2.npy").item()
# print seq.keys()
# seq[DICT_LABELLED_FRAMES] = [[1241-896, 1358-896, 1719-896], [1621-896, 1840-896, 1970-896]]
# seq[DICT_NUM_EXTRA_FRAMES] = [[4, 4, 4], [4, 4, 4]]
# seq[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/wave2/aron2-vanilla_distMat.npy"
# seq[DICT_NUM_SEMANTICS] = 2
# if DICT_FOOTPRINTS in seq.keys() :
#     del seq[DICT_FOOTPRINTS]
# print seq.keys()

# np.save(seq[DICT_SEQUENCE_LOCATION], seq)

# print np.load("/home/ilisescu/PhD/data/street/semantic_sequence-blue_car1.npy").item().keys()

# <codecell>

# print window.semanticLoopingTab.synthesisedSequence[DICT_SEQUENCE_INSTANCES][3][DICT_DESIRED_SEMANTICS]

# <codecell>

# # logLocation = np.sort(glob.glob("logFiles/log-*"))[-1]
# # logLocation = "/media/ilisescu/UUI/Semantic Control/logFiles/log-2016-04-05 14_06_44.540685"
# logLocation = "/home/ilisescu/PhD/data/synthesisedSequences/USER STUDIES SEQUENCES/aron/task_log"

# with open(logLocation) as f :
#     allLines = f.readlines()
    
#     timeSpentInTabs = [[], []]
#     listOfSpritesInDefinition = {}
#     isDoingTracking = True
#     currentSprite = ""
    
#     for line in allLines :
#         if "\n" in line :
#             line = line[:-2]
            
#         action, timestamp = line.split("-&-")
#         timeOfAction = np.array(timestamp.split(" ")[-1].split(":"), float)
#         print action, timestamp.split(" ")[-1]
#         if "DEFINITION:Switch" in action :
#             isDefinitionTab = True
#             timeSpentInTabs[0].append([timeOfAction, timeOfAction])
            
#             if len(timeSpentInTabs[1]) > 0 :
#                 timeSpentInTabs[1][-1][-1] = timeOfAction
                
#             if currentSprite != "" and currentSprite in listOfSpritesInDefinition.keys() :
#                 if isDoingTracking :
#                     listOfSpritesInDefinition[currentSprite]["tracking"].append([timeOfAction, timeOfAction])
#                 else :
#                     listOfSpritesInDefinition[currentSprite]["segmenting"].append([timeOfAction, timeOfAction])
                
#         elif "LOOPING:Switch" in action :
#             isDefinitionTab = False
#             timeSpentInTabs[1].append([timeOfAction, timeOfAction])
            
#             if len(timeSpentInTabs[0]) > 0 :
#                 timeSpentInTabs[0][-1][-1] = timeOfAction
                
#             if currentSprite != "" and currentSprite in listOfSpritesInDefinition.keys() :
#                 if isDoingTracking :
#                     listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                 else :
#                     listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
                
            
#         if isDefinitionTab :
#             if "Selecting" in action :
#                 if currentSprite != "" and currentSprite in listOfSpritesInDefinition.keys() :
#                     if isDoingTracking :
#                         listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                     else :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
                        
#                 currentSprite = action.split(":")[-1].split(" ")[-1]
                
#             if currentSprite != "" :
#                 if currentSprite not in listOfSpritesInDefinition :
#                     listOfSpritesInDefinition[currentSprite] = {}
                
#                 if "Selecting" in action :
#                     if isDoingTracking :
#                         if "tracking" not in listOfSpritesInDefinition[currentSprite] :
#                             listOfSpritesInDefinition[currentSprite]["tracking"] = []
                            
#                         listOfSpritesInDefinition[currentSprite]["tracking"].append([timeOfAction, timeOfAction])
#                     else :
#                         if "segmenting" not in listOfSpritesInDefinition[currentSprite] :
#                             listOfSpritesInDefinition[currentSprite]["segmenting"] = []
                            
#                         listOfSpritesInDefinition[currentSprite]["segmenting"].append([timeOfAction, timeOfAction])
                        
#                 if "Start Segmenting" in action :
#                     if "tracking" in listOfSpritesInDefinition[currentSprite].keys() :
#                         listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                     isDoingTracking = False
                    
#                     if "segmenting" not in listOfSpritesInDefinition[currentSprite] :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"] = []
                            
#                     listOfSpritesInDefinition[currentSprite]["segmenting"].append([timeOfAction, timeOfAction])
                    
#                 if "Start Tracking" in action :
#                     if "segmenting" in listOfSpritesInDefinition[currentSprite].keys() :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
#                     isDoingTracking = False
                    
#                     if "tracking" not in listOfSpritesInDefinition[currentSprite] :
#                         listOfSpritesInDefinition[currentSprite]["tracking"] = []
                    
#                     listOfSpritesInDefinition[currentSprite]["tracking"].append([timeOfAction, timeOfAction])
                
#         else :
#             print "nothing to do"
            
#         if "Closing" in action :
#             if isDefinitionTab :
#                 if currentSprite != "" :
#                     if isDoingTracking :
#                         listOfSpritesInDefinition[currentSprite]["tracking"][-1][-1] = timeOfAction
#                     else :
#                         listOfSpritesInDefinition[currentSprite]["segmenting"][-1][-1] = timeOfAction
                
#                 if len(timeSpentInTabs[0]) > 0 :
#                     timeSpentInTabs[0][-1][-1] = timeOfAction
#             else :
#                 if len(timeSpentInTabs[1]) > 0 :
#                     timeSpentInTabs[1][-1][-1] = timeOfAction
        
            
# print
# print "---------------------- STATISTICS ----------------------"
# for spriteKey in listOfSpritesInDefinition.keys() :
#     print "SPRITE:", spriteKey
#     if "tracking" in listOfSpritesInDefinition[spriteKey].keys() :
#         totalTime = np.zeros(3)
#         for instance in listOfSpritesInDefinition[spriteKey]["tracking"] :
#             tmp = instance[1]-instance[0]
#             if tmp[1] < 0.0 :
#                 tmp[1] += 60.0
#                 tmp[0] -= 1.0
#             if tmp[2] < 0.0 :
#                 tmp[2] += 60.0
#                 tmp[1] -= 1.0
#             totalTime += tmp
#             if totalTime[1] >= 60.0 :
#                 totalTime[1] -= 60.0
#                 totalTime[0] += 1.0
#             if totalTime[2] >= 60.0 :
#                 totalTime[2] -= 60.0
#                 totalTime[1] += 1.0
# #             print instance#, instance[1]-instance[0], tmp
#         print "TRACKING TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])
#     else :
#         print "NO TRACKING"
        
#     if "segmenting" in listOfSpritesInDefinition[spriteKey].keys() :
#         totalTime = np.zeros(3)
#         for instance in listOfSpritesInDefinition[spriteKey]["segmenting"] :
#             tmp = instance[1]-instance[0]
#             if tmp[1] < 0.0 :
#                 tmp[1] += 60.0
#                 tmp[0] -= 1.0
#             if tmp[2] < 0.0 :
#                 tmp[2] += 60.0
#                 tmp[1] -= 1.0
#             totalTime += tmp
#             if totalTime[1] >= 60.0 :
#                 totalTime[1] -= 60.0
#                 totalTime[0] += 1.0
#             if totalTime[2] >= 60.0 :
#                 totalTime[2] -= 60.0
#                 totalTime[1] += 1.0
# #             print instance#, instance[1]-instance[0], tmp
#         print "SEGMENTATION TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])
#     else :
#         print "NO SEGMENTATION"
#     print
        
# totalTime = np.zeros(3)
# for instance in timeSpentInTabs[0] :
#     tmp = instance[1]-instance[0]
#     if tmp[1] < 0.0 :
#         tmp[1] += 60.0
#         tmp[0] -= 1.0
#     if tmp[2] < 0.0 :
#         tmp[2] += 60.0
#         tmp[1] -= 1.0
#     totalTime += tmp
#     if totalTime[1] >= 60.0 :
#         totalTime[1] -= 60.0
#         totalTime[0] += 1.0
#     if totalTime[2] >= 60.0 :
#         totalTime[2] -= 60.0
#         totalTime[1] += 1.0
# #     print instance#, instance[1]-instance[0], tmp
# print "TOTAL DEFINITION TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])

# totalTime = np.zeros(3)
# for instance in timeSpentInTabs[1] :
#     tmp = instance[1]-instance[0]
#     if tmp[1] < 0.0 :
#         tmp[1] += 60.0
#         tmp[0] -= 1.0
#     elif tmp[1] >= 60.0 :
#         tmp[1] -= 60.0
#         tmp[0] += 1.0
#     if tmp[2] < 0.0 :
#         tmp[2] += 60.0
#         tmp[1] -= 1.0
#     elif tmp[2] >= 60.0 :
#         tmp[2] -= 60.0
#         tmp[1] += 1.0
#     totalTime += tmp
# #     print instance#, instance[1]-instance[0], tmp
# print "TOTAL LOOPING TIME: {0} hours, {1} minutes, {2} seconds".format(totalTime[0], totalTime[1], totalTime[2])

# <codecell>

# # gwv.showCustomGraph(window.semanticLoopingTab.preloadedTransitionCosts[0])
# for i in window.semanticLoopingTab.semanticSequences :
#     print i[DICT_SEQUENCE_NAME]
#     for key in i.keys() :
#         try :
#             print key, len(i[key])
#         except :
#             print
        
#     print

# <codecell>

# gwv.showCustomGraph(np.load(window.semanticsDefinitionTab.semanticSequences[2][DICT_DISTANCE_MATRIX_LOCATION]))
# # gwv.showCustomGraph(np.load("/home/ilisescu/PhD/data/havana/black_car1-new_overlap_norm_distMat.npy"))
# print np.load(window.semanticsDefinitionTab.semanticSequences[2][DICT_DISTANCE_MATRIX_LOCATION]).shape
# # for key in window.semanticsDefinitionTab.semanticSequences[2][DICT_FRAMES_LOCATIONS] :
# #     print key, window.semanticsDefinitionTab.semanticSequences[2][DICT_FRAMES_LOCATIONS][key], key in window.semanticsDefinitionTab.semanticSequences[2][DICT_BBOXES]
# print window.semanticsDefinitionTab.semanticSequences[2][DICT_LABELLED_FRAMES]
# # del window.semanticsDefinitionTab.semanticSequences[2][DICT_LABELLED_FRAMES][1][0]
# # del window.semanticsDefinitionTab.semanticSequences[2][DICT_NUM_EXTRA_FRAMES][1][0]
# # del window.semanticsDefinitionTab.semanticSequences[2][DICT_FRAMES_LOCATIONS][275]
# # np.save(window.semanticsDefinitionTab.semanticSequences[2][DICT_SEQUENCE_LOCATION], window.semanticsDefinitionTab.semanticSequences[2])

# <codecell>

# for i in window.semanticsDefinitionTab.semanticSequences :
#     if DICT_CONFLICTING_SEQUENCES in i.keys() :
#         print i[DICT_SEQUENCE_NAME], i[DICT_CONFLICTING_SEQUENCES], i[DICT_LABELLED_FRAMES]
# #     i["number_of_semantic_classes"] = 2
# #     np.save(i[DICT_SEQUENCE_LOCATION], i)

# <codecell>

# for key in window.semanticLoopingTab.semanticSequences[0] :
#     try :
#         print key, len(window.semanticLoopingTab.semanticSequences[0][key])
#     except :
#         print window.semanticLoopingTab.semanticSequences[0][key]

# <codecell>

# for i in window.semanticsDefinitionTab.semanticSequences :
# #     if DICT_CONFLICTING_SEQUENCES in i.keys() :
# #         print i[DICT_SEQUENCE_NAME], i[DICT_CONFLICTING_SEQUENCES], i[DICT_LABELLED_FRAMES]
# #     if DICT_NUM_SEMANTICS in i.keys() :
# #         print i[DICT_SEQUENCE_NAME], i[DICT_NUM_SEMANTICS]
#     print i[DICT_SEQUENCE_NAME]
# #     i[DICT_NUM_SEMANTICS] = 9
# #     np.save(i[DICT_SEQUENCE_LOCATION], i)

# <codecell>

# tmp = np.load("/media/ilisescu/Data1/PhD/data/elevators/semantic_sequence-elevator1.npy").item()
# print tmp.keys()
# # del tmp[DICT_MASK_LOCATION]
# del tmp[DICT_NUM_EXTRA_FRAMES], tmp[DICT_FRAME_SEMANTICS], tmp[DICT_LABELLED_FRAMES]
# np.save(tmp[DICT_SEQUENCE_LOCATION], tmp)

# <codecell>

# for seqLoc in np.sort(glob.glob("/home/ilisescu/PhD/data/havana/semantic_sequence-*.npy")) :
#     seq = np.load(seqLoc).item()
#     if DICT_LABELLED_FRAMES in seq.keys() :
#         print seq[DICT_SEQUENCE_NAME]
#         print seq[DICT_LABELLED_FRAMES], seq[DICT_NUM_EXTRA_FRAMES]
#         tmp = seq[DICT_LABELLED_FRAMES][0]
#         seq[DICT_LABELLED_FRAMES][0] = seq[DICT_LABELLED_FRAMES][1]
#         seq[DICT_LABELLED_FRAMES][1] = tmp
#         tmp = seq[DICT_NUM_EXTRA_FRAMES][0]
#         seq[DICT_NUM_EXTRA_FRAMES][0] = seq[DICT_NUM_EXTRA_FRAMES][1]
#         seq[DICT_NUM_EXTRA_FRAMES][1] = tmp
#         print seq[DICT_LABELLED_FRAMES], seq[DICT_NUM_EXTRA_FRAMES]
# #         np.save(seq[DICT_SEQUENCE_LOCATION], seq)
# #         print seq[DICT_SEQUENCE_LOCATION]

# <codecell>

# tmp = [[0, 9]]
# tmp2 = [[4, 2]]
# target = 4
# targetExtra = 4
# for classIdx in xrange(len(tmp)) :
#     c = tmp[classIdx]
#     e = tmp2[classIdx]
#     print np.array(c), np.abs(target-np.array(c)), np.abs(target-np.array(c)) <= np.array(e)/2
#     targetFrames = np.arange(target-targetExtra/2, target+targetExtra/2+1).reshape((1, targetExtra+1))
#     print np.abs(targetFrames - np.array(c).reshape((len(c), 1)))
#     print np.any(np.abs(targetFrames - np.array(c).reshape((len(c), 1))) <= (np.array(e)/2).reshape((len(e), 1)), axis=1)
#     found = np.abs(target-np.array(c)) <= np.array(e)/2
#     if np.any(found) :
#         tmp[classIdx] = [x for i, x in enumerate(c) if not found[i]]
#         tmp2[classIdx] = [x for i, x in enumerate(e) if not found[i]]
# #     if 17 in c :
# #         print 17 in c
# print tmp
# print tmp2

# <codecell>

# semanticSequence = window.semanticsDefinitionTab.semanticSequences[-1]

# def checkSemanticSequence(semanticSequence) :
    
#     if DICT_DISTANCE_MATRIX_LOCATION not in semanticSequence.keys() :
#         return 1, "Distance Matrix not computed"
#     if np.load(semanticSequence[DICT_DISTANCE_MATRIX_LOCATION]).shape[0] != len(semanticSequence[DICT_FRAMES_LOCATIONS].keys()) :
#         return 11, "Mismatch between distance matrix and number of frames"
    
#     if DICT_TRANSITION_COSTS_LOCATION not in semanticSequence.keys() :
#         return 2, "Transition costs not computed"
#     if np.load(semanticSequence[DICT_TRANSITION_COSTS_LOCATION]).shape[0] != len(semanticSequence[DICT_FRAMES_LOCATIONS].keys()) :
#         return 21, "Mismatch between transition matrix and number of frames"

#     ## only care about the stuff below if masks have been defined
#     if DICT_MASK_LOCATION in semanticSequence.keys() :
#         frameKeys = np.sort(semanticSequence[DICT_FRAMES_LOCATIONS].keys())
        
#         if DICT_PATCHES_LOCATION not in semanticSequence.keys() :
#             return 31, "Segmentation incomplete (patches not computed)"
#         patchKeys = np.load(semanticSequence[DICT_PATCHES_LOCATION]).item().keys()

#         for key in frameKeys :
#             if key not in semanticSequence[DICT_BBOXES].keys() :
#                 return 3, "BBox not defined for frame "+np.string_(key)
            
#             if key not in patchKeys :
#                 return 32, "Segmentation incomplete (patch for frame "+np.string_(key)+" not available)"
    
#     if DICT_FRAME_SEMANTICS not in semanticSequence.keys() :
#         return 4, "Semantics undefined"
#     if semanticSequence[DICT_FRAME_SEMANTICS].shape[0] != len(semanticSequence[DICT_FRAMES_LOCATIONS].keys()) :
#         return 41, "Mismatch between semantics and number of frames"
    
#     return 0, ""
# print checkSemanticSequence(semanticSequence)

# <codecell>

# print semanticSequence[DICT_FRAMES_LOCATIONS].keys()
# print semanticSequence[DICT_BBOXES].keys()
# print np.load(semanticSequence[DICT_TRANSITION_COSTS_LOCATION]).shape[0]
# print np.load(semanticSequence[DICT_DISTANCE_MATRIX_LOCATION]).shape[0]

# <codecell>

# print window.semanticsDefinitionTab.semanticSequences[0].keys()
# # print window.semanticsDefinitionTab.semanticSequences[0][DICT_SEQUENCE_LOCATION]
# print window.semanticsDefinitionTab.semanticSequences[1].keys()
# print window.semanticsDefinitionTab.semanticSequences[2].keys()
# print window.semanticsDefinitionTab.semanticSequences[3].keys()
# print window.semanticsDefinitionTab.semanticSequences[4].keys()

# # for seq in window.semanticsDefinitionTab.semanticSequences[1:2] :
# #     tmp = np.load(seq[DICT_PATCHES_LOCATION]).item()
# #     for key in tmp.keys() :
# #         tmp[key]['sprite_colors'] = tmp[key]['sprite_colors'][:, [2, 1, 0, 3]]
# #     np.save(seq[DICT_PATCHES_LOCATION], tmp)

