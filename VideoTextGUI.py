# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

from PIL import Image
from PySide import QtCore, QtGui

import sys
import numpy as np
import scipy as sp
import scipy.io as sio
import time
import cv2
import re
import glob
import os
import os.path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.sparsetools import _graph_tools
from sklearn.utils.sparsetools import _graph_validation
from sklearn.utils import lgamma

import VideoTexturesTabGUI as vttg
import PreProcessingTabGUI as pptg
import GraphWithValues as gwv
import scipy.stats as ss

import VideoTexturesUtils as vtu
import ComputeDistanceMatrix as cdm

dataFolder = ".." + os.sep + "data" + os.sep

app = QtGui.QApplication(sys.argv)

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
    def setPixmap(self, pixmap) :
        if pixmap.width() > self.width() :
            super(ImageLabel, self).setPixmap(pixmap.scaledToWidth(self.width()))
        else :
            super(ImageLabel, self).setPixmap(pixmap)
        
    def resizeEvent(self, event) :
        if self.pixmap() != None :
            if self.pixmap().width() > self.width() :
                self.setPixmap(self.pixmap().scaledToWidth(self.width()))
                
                
class CustomDockWidget(QtGui.QDockWidget) :
    
    def __init__(self, parent=None):
        super(CustomDockWidget, self).__init__(parent)
        
        self.setFeatures(~QtGui.QDockWidget.DockWidgetClosable & ~QtGui.QDockWidget.DockWidgetVerticalTitleBar)
        
        self.isMoving = False
        self.prevPos = None
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isFloating():
#             print "pressed"
            self.isMoving = True
            self.prevPos = event.globalPos()
        
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
#             print "released"
            self.isMoving = False
            self.prevPos = None
        
    def mouseMoveEvent(self, event):
        if self.prevPos != None and self.isFloating():
            # not sure why this is not working
#             if not self.isFloating():
#                 print "floated"
#                 self.setFloating(True)
#             print "moved"
            self.move(self.pos()+event.globalPos()-self.prevPos)
            self.prevPos = event.globalPos()
        

# <codecell>

class BarGraph(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(BarGraph, self).__init__(text, parent)
        
        self.marginSize = 10
        
        self.bars = []
        self.probs = None
        self.classClrs = [QtGui.QColor.fromRgb(255, 0, 0), QtGui.QColor.fromRgb(0, 255, 0), QtGui.QColor.fromRgb(0, 0, 255), QtGui.QColor.fromRgb(255, 0, 255)]
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paintBargraph(painter)
        
    def resizeEvent(self, event):
        self.marginSize = np.int(self.width()/25)
        self.computeBars()
    
    def setProbs(self, probs):
        self.probs = probs
        self.computeBars()
        
    def computeBars(self):
#         print "he", self.probs
        self.bars = []
        if self.probs != None and len(self.probs) > 0 :
            barWidth = np.int((self.width()-(self.marginSize*(len(self.probs)-1)))/len(self.probs))
            
            h = self.height()-self.marginSize
            for i in xrange(0, len(self.probs)) :
#                 print "tra", i*(barWidth+self.marginSize), self.height()-h*self.probs[i], barWidth, h
                bar = QtCore.QRect(i*(barWidth+self.marginSize), self.height()-h*self.probs[i], barWidth, h)
                self.bars.append(bar)
                
            self.update()
        
    def paintBargraph(self, painter):
        
        for b in xrange(0, len(self.bars)):
            ## paint midlines
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(192, 192, 192), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawLine(QtCore.QPoint(self.bars[b].center().x(), 0), QtCore.QPoint(self.bars[b].center().x(), self.height()))
            
            ## paint the bar rectangle
            painter.setPen(QtGui.QPen(self.classClrs[b], 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.setBrush(QtGui.QBrush(self.classClrs[b]))
            painter.drawRect(self.bars[b])
            
            ## paint the text
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            textRect = QtCore.QRect(self.bars[b].left(), 0, self.bars[b].width(), self.height())
            painter.drawText(textRect, QtCore.Qt.AlignCenter, np.string_(np.round(self.probs[b]*100, decimals=1)) + "%")
            
        if len(self.bars) < 1:
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.text())
            
        ## draw axes
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        ## x axis
        painter.drawLine(QtCore.QPoint(0, self.height()-1), QtCore.QPoint(self.width()-1, self.height()-1))
        ## y axis
        painter.drawLine(QtCore.QPoint(0, 0), QtCore.QPoint(0, self.height()-1))
        

# <codecell>

class LineGraph(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(LineGraph, self).__init__(text, parent)
        
        self.polylines = []
        self.numClasses = 0
        self.dataPoints = 0
        self.currentFrame = 0
        self.xPoints = 0
        self.yPoints = 100
        self.transform = QtGui.QMatrix()
        self.classClrs = [QtGui.QColor.fromRgb(255, 0, 0), QtGui.QColor.fromRgb(0, 255, 0), QtGui.QColor.fromRgb(0, 0, 255), QtGui.QColor.fromRgb(255, 0, 255)]
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paintLinegraph(painter)
    
    def setProbs(self, probs, idxCorrection):
        self.numClasses = probs.shape[-1]
        self.dataPoints = probs.shape[0]
        self.xPoints = self.dataPoints+idxCorrection*2
        self.polylines = []
        for i in xrange(0, self.numClasses) :
            if self.dataPoints > 0 :
                self.polylines.append(QtGui.QPolygonF())
                for j in xrange(0, self.dataPoints) :
                    ## need to do 1.0- for the y axis because in numpy y goes bottom to top and in qt it goes top to bottom
                    self.polylines[i].append(QtCore.QPointF(np.float(j+idxCorrection), (1.0-probs[j, i])*self.yPoints))
    
        self.updateTransform()
        self.update()
    
    def updateTransform(self) :
        if self.xPoints > 0 :
            self.transform = QtGui.QMatrix()
            self.transform = self.transform.scale(np.float(self.width())/np.float(self.xPoints), np.float(self.height())/np.float(self.yPoints))
    
    def setCurrentFrame(self, currentFrame):
        self.currentFrame = currentFrame
        self.update()
        
    def resizeEvent(self, event):
        self.updateTransform()
        
    def paintLinegraph(self, painter):
        if self.xPoints > 0 :
            ## draw currentFrame line
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(192, 192, 192), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            ratio = np.float(self.width())/np.float(self.xPoints)
            painter.drawLine(QtCore.QPoint(np.float(self.currentFrame)*ratio, 0), QtCore.QPoint(np.float(self.currentFrame)*ratio, self.height()))
            
        if self.dataPoints > 0 and self.numClasses > 0 :
            for p in xrange(0, self.numClasses):
                ## paint the polyline for current class probabilities
                painter.setPen(QtGui.QPen(self.classClrs[p], 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPolyline(self.transform.map(self.polylines[p]))
        else :
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.text())
            
        ## draw axes
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        ## x axis
        painter.drawLine(QtCore.QPoint(0, self.height()-1), QtCore.QPoint(self.width()-1, self.height()-1))
        ## y axis
        painter.drawLine(QtCore.QPoint(0, 0), QtCore.QPoint(0, self.height()-1))
        

# <codecell>

class ScatterGraph(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ScatterGraph, self).__init__(text, parent)
        
        self.points = []
        self.numClasses = 0
        self.currentFrame = 0
        self.dataPoints = 0
        self.idxCorrection = 0
        self.transform = QtGui.QMatrix()
        self.classClrs = [QtGui.QColor.fromRgb(255, 0, 0), QtGui.QColor.fromRgb(0, 255, 0), QtGui.QColor.fromRgb(0, 0, 255), QtGui.QColor.fromRgb(255, 0, 255)]
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paintScattergraph(painter)
    
    def setPoints(self, scatterPoints, labels, idxCorrection):
        self.scatterPoints = np.copy(scatterPoints)
        ## normalize the scatterPoints to fit the full label canvas
        self.scatterPoints[0, :] = (self.scatterPoints[0, :]-np.min(self.scatterPoints[0, :]))/(np.max(self.scatterPoints[0, :])-np.min(self.scatterPoints[0, :]))
        ## need to do 1.0- for the y axis because in numpy y goes bottom to top and in qt it goes top to bottom
        self.scatterPoints[1, :] = 1.0-((self.scatterPoints[1, :]-np.min(self.scatterPoints[1, :]))/(np.max(self.scatterPoints[1, :])-np.min(self.scatterPoints[1, :])))
        self.labels = labels
        self.dataPoints = self.scatterPoints.shape[-1]
        self.numClasses = len(self.labels)
        self.idxCorrection = idxCorrection
        self.points = []
        for i in xrange(0, self.numClasses) :
            if len(self.labels[i]) > 0 :
                self.points.append(QtGui.QPolygonF())
                for j in xrange(0, len(self.labels[i])) :
                    self.points[i].append(QtCore.QPointF(np.float(self.scatterPoints[0, self.labels[i][j]]), np.float(self.scatterPoints[1, self.labels[i][j]])))
    
        self.updateTransform()
        self.update()
    
    def updateTransform(self) :
        if self.dataPoints > 0 :
            self.transform = QtGui.QMatrix()
            self.transform = self.transform.scale(np.float(self.width()), np.float(self.height()))
            sys.stdout.flush()
    
    def setCurrentFrame(self, currentFrame):
        self.currentFrame = currentFrame - self.idxCorrection
        self.update()
        
    def resizeEvent(self, event):
        self.updateTransform()
        
    def paintScattergraph(self, painter):
            
        if self.dataPoints > 0 and self.numClasses > 0 :
            for p in xrange(0, self.numClasses):
                ## paint the points for current class
                painter.setPen(QtGui.QPen(self.classClrs[p], len(np.string_(self.width())), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPoints(self.transform.map(self.points[p]))
        else :
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.text())
            
        if self.currentFrame >= 0 and self.currentFrame < self.dataPoints:
#             ## draw currentFrame circle
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawEllipse(self.transform.map(QtCore.QPointF(np.float(self.scatterPoints[0, self.currentFrame]), 
                                                                  np.float(self.scatterPoints[1, self.currentFrame]))), 
                                len(np.string_(self.width())), len(np.string_(self.width())))
            
        ## draw axes
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        ## x axis
        painter.drawLine(QtCore.QPoint(0, self.height()-1), QtCore.QPoint(self.width()-1, self.height()-1))
        ## y axis
        painter.drawLine(QtCore.QPoint(0, 0), QtCore.QPoint(0, self.height()-1))
        

# <codecell>

class LabelViewerGroup(QtGui.QGroupBox):

    def __init__(self, title, parent=None):
        super(LabelViewerGroup, self).__init__(title, parent)
        
        self.createGUI()
        
    def setCurrentFrame(self, frame):
        self.imageLabel.setFixedSize(frame.shape[1], frame.shape[0])
        
        im = np.ascontiguousarray(frame)
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(qim))

    def setValue(self, value):
        self.slider.setValue(value)
        self.classesProbsGraph.setCurrentFrame(value)
        self.labelsMapGraph.setCurrentFrame(value)

    def setMinimum(self, value):
        self.slider.setMinimum(value)
        self.frameSpinBox.setMinimum(value)

    def setMaximum(self, value):
        self.slider.setMaximum(value)
        self.frameSpinBox.setMaximum(value)
        
    def setFrameLabel(self, label):
        if label == 0 :
            self.imageGroup.setStyleSheet("background-color: red")
        elif label == 1 :
            self.imageGroup.setStyleSheet("background-color: green")
        elif label == 2 :
            self.imageGroup.setStyleSheet("background-color: blue")
        elif label == 3 :
            self.imageGroup.setStyleSheet("background-color: magenta")
        else :
            self.imageGroup.setStyleSheet("")
    
    def setFrameLabelProbs(self, labelProbs):
        self.labelProbsGraph.setProbs(labelProbs)
    
    def setClassesProbs(self, classesProbs, idxCorrection):
        self.classesProbsGraph.setProbs(classesProbs, idxCorrection)
#         self.classesProbsGraph.setCurrentFrame(self.slider.value())
    
    def setLabelMap(self, mapPoints, labeledPoints, idxCorrection):
        self.labelsMapGraph.setPoints(mapPoints, labeledPoints, idxCorrection)
#         self.labelsMapGraph.setCurrentFrame(self.slider.value())
        
    def createGUI(self):
        
        self.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)

        ## WIDGETS ##
        self.imageLabel = ImageLabel("Current Frame")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageGroup = QtGui.QGroupBox()
        self.imageGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        imageLayout = QtGui.QVBoxLayout()
        imageLayout.addWidget(self.imageLabel)
        self.imageGroup.setLayout(imageLayout)
        self.imageGroup.setMinimumSize(200, 200)
        
        self.labelProbsGraph = BarGraph("Label probabilities")
        self.labelProbsGraph.setMinimumHeight(150)
        self.labelProbsGraph.setAlignment(QtCore.Qt.AlignCenter)
        
        self.classesProbsGraph = LineGraph("Class probabilities")
        self.classesProbsGraph.setMinimumHeight(150)
        self.classesProbsGraph.setAlignment(QtCore.Qt.AlignCenter)
        self.classesProbsGraph.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        
        self.labelsMapGraph = ScatterGraph("Labels Map")
        self.labelsMapGraph.setMinimumHeight(150)
        self.labelsMapGraph.setAlignment(QtCore.Qt.AlignCenter)
        self.labelsMapGraph.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider.setMaximum(0)
        self.slider.setMinimum(0)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        style = "QSlider::handle:horizontal { background: #000000; width: 4px; "
        style += "border-radius: 3px; } QSlider {border: 1px solid gray;}"
        style += "QSlider::groove:horizontal {background: #eeeeee;}"
        self.slider.setStyleSheet(style)
    
        self.frameSpinBox = QtGui.QSpinBox()
        self.frameSpinBox.setRange(0, 0)
        self.frameSpinBox.setSingleStep(1)
        
        
        ## SIGNALS ##
        self.slider.valueChanged[int].connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.setValue)
        
        
        ## LAYOUTS ##
        labelsLayout = QtGui.QHBoxLayout()
        labelsLayout.addStretch()
        vertLayout = QtGui.QVBoxLayout()
        vertLayout.addStretch()
        vertLayout.addWidget(self.imageGroup)
        vertLayout.addWidget(self.labelProbsGraph)
        vertLayout.addStretch()
        labelsLayout.addLayout(vertLayout)
        labelsLayout.addStretch()
        
        navLayout = QtGui.QHBoxLayout()
        navLayout.addWidget(self.slider)
        navLayout.addWidget(self.frameSpinBox)
        
        mainLayout = QtGui.QVBoxLayout()
#         mainLayout.addStretch()
        mainLayout.addLayout(labelsLayout)
#         mainLayout.addStretch()
        mainLayout.addWidget(self.classesProbsGraph)
        mainLayout.addWidget(self.labelsMapGraph)
        mainLayout.addLayout(navLayout)
        
        
        self.setLayout(mainLayout)

# <codecell>

class LabellingTab(QtGui.QWidget):
    def __init__(self, mainWindow, parent=None):
        super(LabellingTab, self).__init__(parent)
        
        self.mainWindow = mainWindow
        self.labeledPoints = None
        self.labelProbs = None
        self.mapPoints = None
        self.predictedLabels = None
        self.idxCorrection = None
        
        self.createGUI()
        
        self.lockGUI(True)
        
    def lockGUI(self, lock):
        
        self.labelViewerGroupLeft.slider.setEnabled(not lock)
        self.labelViewerGroupLeft.frameSpinBox.setEnabled(not lock)
        
        self.labelViewerGroupRight.slider.setEnabled(not lock)
        self.labelViewerGroupRight.frameSpinBox.setEnabled(not lock)
        
        self.loadLabelsButton.setEnabled(not lock)
        self.loadMapButton.setEnabled(not lock)
        
    def setLoadedMovie(self, movie, frameNames, dataLocation, downsampleFactor) :
        self.movie = movie
        self.frameNames = frameNames
        self.dataLocation = dataLocation
        self.downsampleFactor = downsampleFactor
        
        self.labelViewerGroupLeft.setMaximum(self.movie.shape[-1]-1)
        self.labelViewerGroupLeft.setValue(0)
        
        self.labelViewerGroupRight.setMaximum(self.movie.shape[-1]-1)
        self.labelViewerGroupRight.setValue(0)
            
        self.showFrameLeft(0)
        self.showFrameRight(0)
    
    def loadLabels(self) :
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Load Propagated Labels", 
            QtCore.QDir.currentPath()+os.sep+dataFolder, "Labels(*labels.npy)")
        if fileName :
            
            labelData = np.load(fileName)[()]
            if "labeledPoints" in labelData.keys() and "labelProbs" in labelData.keys() :
                self.loadedLabelsLabel.setText(fileName.split(os.sep)[-1])
                
                self.labeledPoints = labelData["labeledPoints"]
                self.labelProbs = labelData["labelProbs"]
                self.idxCorrection = np.int((self.movie.shape[-1]-self.labelProbs.shape[0])/2)
                # need to correct for idx due to filtering
                self.labeledPoints += self.idxCorrection
                
                self.labelViewerGroupLeft.setClassesProbs(self.labelProbs, self.idxCorrection)
                self.labelViewerGroupRight.setClassesProbs(self.labelProbs, self.idxCorrection)
                
                ## refresh viewers in case frame has been changed before loading labels
                self.showFrameLeft(self.labelViewerGroupLeft.slider.value())
                self.showFrameRight(self.labelViewerGroupRight.slider.value())
            else :
                QtGui.QMessageBox.warning(self, "The selected file is not valid",
                            "<p align='center'>The propagated labels file does not contain the necessary data.<br>"
                            "Please select a valid file.</p>")
    
    def loadMap(self) :
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Load Labels Map", 
            QtCore.QDir.currentPath()+os.sep+dataFolder, "Maps(*map.npy)")
        if fileName :
            
            mapData = np.load(fileName)[()]
            if "mapPoints" in mapData.keys() and "predictedLabels" in mapData.keys() :
                self.loadedMapLabel.setText(fileName.split(os.sep)[-1])
                
                self.mapPoints = mapData["mapPoints"]
                self.predictedLabels = mapData["predictedLabels"]
                self.idxCorrection = np.int((self.movie.shape[-1]-self.mapPoints.shape[-1])/2)
                
                self.labelViewerGroupLeft.setLabelMap(self.mapPoints, self.predictedLabels, self.idxCorrection)
                self.labelViewerGroupRight.setLabelMap(self.mapPoints, self.predictedLabels, self.idxCorrection)
                
                ## refresh viewers in case frame has been changed before loading map
                self.showFrameLeft(self.labelViewerGroupLeft.slider.value())
                self.showFrameRight(self.labelViewerGroupRight.slider.value())
                
            else :
                QtGui.QMessageBox.warning(self, "The selected file is not valid",
                            "<p align='center'>The labels map file does not contain the necessary data.<br>"
                            "Please select a valid file.</p>")
    
    def showFrameLeft(self, idx) :
        self.showFrame(idx, 1)
        
    def showFrameRight(self, idx) :
        self.showFrame(idx, 2)
        
    def showFrame(self, idx, which) :
        if idx >= 0 and idx < self.movie.shape[-1] :
            label = -1
            if self.labeledPoints != None and idx in self.labeledPoints :
                label = np.ndarray.flatten(np.argwhere(self.labeledPoints == idx))[0]
                
            labelProbs = None
            if self.labelProbs != None and idx-self.idxCorrection >=0 and idx-self.idxCorrection < self.labelProbs.shape[0] :
                labelProbs = self.labelProbs[idx-self.idxCorrection, :]
                
            if which == 1 :
                self.labelViewerGroupLeft.setCurrentFrame(self.movie[:, :, :, idx])
                self.labelViewerGroupLeft.setFrameLabel(label)
                self.labelViewerGroupLeft.setFrameLabelProbs(labelProbs)
            elif which == 2 :
                self.labelViewerGroupRight.setCurrentFrame(self.movie[:, :, :, idx])
                self.labelViewerGroupRight.setFrameLabel(label)
                self.labelViewerGroupRight.setFrameLabelProbs(labelProbs)
        
    def createGUI(self) :
                
        ## WIDGETS ##

        self.labelViewerGroupLeft = LabelViewerGroup("Labelling Area 1")
        self.labelViewerGroupRight = LabelViewerGroup("Labelling Area 2")
        
        labelViewerGroupLeftDock = CustomDockWidget()
        labelViewerGroupLeftDock.setWidget(self.labelViewerGroupLeft)
        labelViewerGroupRightDock = CustomDockWidget()
        labelViewerGroupRightDock.setWidget(self.labelViewerGroupRight)
        
        self.loadedLabelsLabel = QtGui.QLabel("No Loaded Labels")
        self.loadedMapLabel = QtGui.QLabel("No Loaded Map")
        
        self.loadLabelsButton = QtGui.QPushButton("Load Labels")
        self.loadMapButton = QtGui.QPushButton("Load Map")
        
        labellingControlsBox = QtGui.QGroupBox("Labelling Controls")
        labellingControlsBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        
        loadLabelsControlsBox = QtGui.QGroupBox("Loaded Labels")
        loadLabelsControlsBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        
        loadMapControlsBox = QtGui.QGroupBox("Loaded Map")
        loadMapControlsBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
                
        ## SIGNALS ##
        
        self.labelViewerGroupLeft.frameSpinBox.valueChanged[int].connect(self.showFrameLeft)
        self.labelViewerGroupRight.frameSpinBox.valueChanged[int].connect(self.showFrameRight)
        
        self.loadLabelsButton.clicked.connect(self.loadLabels)
        self.loadMapButton.clicked.connect(self.loadMap)
        
        ## LAYOUTS ##
        labellingControlsBoxLayout = QtGui.QVBoxLayout()

        loadLabelsControlsBoxLayout = QtGui.QVBoxLayout()
        loadLabelsControlsBoxLayout.addWidget(self.loadedLabelsLabel)
        loadLabelsControlsBoxLayout.addWidget(self.loadLabelsButton)
        loadLabelsControlsBox.setLayout(loadLabelsControlsBoxLayout)
        
        loadMapControlsBoxLayout = QtGui.QVBoxLayout()
        loadMapControlsBoxLayout.addWidget(self.loadedMapLabel)
        loadMapControlsBoxLayout.addWidget(self.loadMapButton)
        loadMapControlsBox.setLayout(loadMapControlsBoxLayout)
        
        labellingControlsBoxLayout.addStretch()
        labellingControlsBoxLayout.addWidget(loadLabelsControlsBox)
        labellingControlsBoxLayout.addWidget(loadMapControlsBox)
        labellingControlsBoxLayout.addStretch()
        labellingControlsBox.setLayout(labellingControlsBoxLayout)
        
        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(labellingControlsBox)
#         self.mainWindow.addDockWidget(QtCore.Qt.LeftDockWidgetArea, labelViewerGroupLeftDock)
        mainLayout.addWidget(labelViewerGroupLeftDock)
        mainLayout.addWidget(labelViewerGroupRightDock)
        self.setLayout(mainLayout)

# <codecell>

class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.showLoading(False)
        
        self.setWindowTitle("Video Textures GUI")
        self.resize(1920, 750)
        
        self.readyForVT = False
        self.firstLoad = True
        self.dataLocation = ""
        
    def openVideo(self):
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Open Video File",
                QtCore.QDir.currentPath()+os.sep+dataFolder)
        if fileName :
            self.dataLocation = fileName
            self.lockGUI(True)
            self.showLoading(True)

            self.loadingLabel.setText("Loading video...")
            self.loadingText = self.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()
            
            cap = cv2.VideoCapture(fileName)
            
            numFrames = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret==True:
                    numFrames += 1
                    frameSize = frame.shape
                    k = cv2.waitKey(30)
                    
                    sys.stdout.write('\r' + "Frame count: " + np.string_(numFrames))
                    sys.stdout.flush()
                    self.loadingLabel.setText(self.loadingText+"\nframe count: " + np.string_(numFrames))
                    QtCore.QCoreApplication.processEvents()
                else:
                    break
            
            # Release everything if job is finished
            cap.release()
            print
            self.loadingText = self.loadingLabel.text()
            cap = cv2.VideoCapture(fileName)
            self.movie = np.zeros(np.hstack([frameSize, numFrames]))
            i = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret==True:
                    self.movie[:, :, :, i] = frame
                    k = cv2.waitKey(30)
                    
                    sys.stdout.write('\r' + "Frames read: " + np.string_(i) + " of " + np.string_(numFrames))
                    sys.stdout.flush()
                    self.loadingLabel.setText(self.loadingText+"\nframes read: " + np.string_(i) + " of " + np.string_(numFrames))
                    QtCore.QCoreApplication.processEvents() 
                    
                    i += 1
                else:
                    break
            
            # Release everything if job is finished
            cap.release()
            
            self.videoTexturesTab.setLoadedMovie(self.movie)
            
            self.lockGUI(False)
            self.showLoading(False)
    
    def openSequence(self):
        self.frameNames = np.array(QtGui.QFileDialog.getOpenFileNames(self, "Open Sequence", 
                    QtCore.QDir.currentPath()+os.sep+dataFolder, "Images(*.png)")[0])
        if len(self.frameNames) > 1 :
#             self.dataLocation = '\\'.join(filter(None, self.frameNames[0].split('\\'))[0:-1])
            self.dataLocation = os.sep.join(filter(None, self.frameNames[0].split(os.sep))[0:-1])
            if os.sep == "/" :
                self.dataLocation = os.sep + self.dataLocation 
            self.lockGUI(True)
            self.showLoading(True)

            self.loadingLabel.setText("Loading set of frames...")
            self.loadingText = self.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()

            ## sort frame names
            self.frameNames = np.sort(self.frameNames)
            self.downsampleFactor = 1
            if len(self.frameNames) > 0 :
                frameSize = cv2.imread(self.frameNames[0]).shape
                requiredSpace = 2.0*(np.prod(frameSize)*float(len(self.frameNames)))/(1024**3) ## GB
                maxSpace = 2
                if requiredSpace > maxSpace :
                    self.downsampleFactor = 2 + np.argmin(np.abs(maxSpace-requiredSpace/np.arange(2.0, 6.0)))
                    print "Downsample Factor of", self.downsampleFactor
                    
                if self.downsampleFactor < 2 and frameSize[0]*3 > 1920 :
                    self.downsampleFactor = 3
                    
                if self.downsampleFactor > 1 :
                    self.movie = np.zeros(np.hstack([frameSize[0]/self.downsampleFactor, frameSize[1]/self.downsampleFactor, frameSize[2], len(self.frameNames)]), dtype=np.uint8)
                else :
                    self.movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], len(self.frameNames)]), dtype=np.uint8)
                
                for i in range(0, len(self.frameNames)) :
                    im = Image.open(self.frameNames[i])
                    if self.downsampleFactor > 1 :
                        im = im.resize(np.array([frameSize[1], frameSize[0]])/self.downsampleFactor)
                    self.movie[:, :, :, i] = np.array(im, dtype=np.uint8)[:, :, 0:3]
                    sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(self.frameNames)))
                    sys.stdout.flush()
                    self.loadingLabel.setText(self.loadingText+"\nframe " + np.string_(i) + " of " + np.string_(len(self.frameNames)))
                    QtCore.QCoreApplication.processEvents() 
                print
                
            self.setLoadedMovie()
            
            self.lockGUI(False)
            self.showLoading(False)
            
    def setLoadedMovie(self) :
        self.firstLoad = False
        frameSize = cv2.imread(self.frameNames[0])
        infoText = "Loaded video:\n\t"+self.dataLocation+"\nNumber of Frames:\n\t"+np.string_(len(self.frameNames))
        infoText += "\nFrame size:\n\t"+np.string_(frameSize.shape[1])+"x"+np.string_(frameSize.shape[0])
        if self.downsampleFactor > 1 :
            infoText += "\tdownscaled to: "+np.string_(self.movie.shape[1])+"x"+np.string_(self.movie.shape[0])
        self.infoLabel.setText(infoText)
        
        self.preProcessingTab.setLoadedMovie(self.movie, self.frameNames, self.dataLocation, self.downsampleFactor)        
        self.videoTexturesTab.setLoadedMovie(self.movie, self.frameNames, self.dataLocation, self.downsampleFactor)
        self.labellingTab.setLoadedMovie(self.movie, self.frameNames, self.dataLocation, self.downsampleFactor)
        
        distMatName = "proc_distMat.npy"
#         distMatName = "proc_hist2demd_32x48_distMat.npy"
        if os.path.isfile(self.dataLocation + os.sep + distMatName) :
            self.readyForVT = True
            self.videoTexturesTab.setDistMat(np.load(self.dataLocation + os.sep + distMatName))
        else :
            self.readyForVT = False
    
    def tabChanged(self, tabIdx) :
        try :
            self.movie
            self.frameNames
            self.dataLocation
        except:
            return
        
        if tabIdx == 1:
            if self.readyForVT == False :#and not os.path.isfile(self.dataLocation + os.sep + "proc_distMat.npy") :
                QtGui.QMessageBox.warning(self, "Pre-processing not ready",
                            "<p align='center'>The pre-processing step has not been completed<br>"
                            "Please return to the pre-processing tab and compute a distance matrix</p>")
                self.tabWidget.setCurrentIndex(0)
                #self.readyForVT = False
            #else : 
            #    self.readyForVT = True
            #    self.videoTexturesTab.setLoadedMovie(self.movie, self.frameNames, self.dataLocation, np.load(self.dataLocation + os.sep + "proc_distMat.npy"))
            #    self.lockGUI(False)
                
    def updateDistMat(self) :
        self.readyForVT = True
        self.videoTexturesTab.setDistMat(np.load(self.dataLocation + os.sep + "proc_distMat.npy"))
        
    def showLoading(self, show) :
        if show :
            self.loadingLabel.setText("Loading... Please wait")
            self.loadingWidget.setVisible(True)
            self.infoLabel.setVisible(False)
        else :
            self.loadingWidget.setVisible(False)
            self.infoLabel.setVisible(True)
            
    def lockGUI(self, lock):
        
        self.openVideoButton.setEnabled(False)#not lock)
        self.openSequenceButton.setEnabled(not lock)
        
        if self.readyForVT :
            self.videoTexturesTab.lockGUI(lock)
        else :
            self.videoTexturesTab.lockGUI(True)
            if self.tabWidget.currentIndex() == 1 and not self.firstLoad :
                QtGui.QMessageBox.warning(self, "Pre-processing not ready",
                        "<p align='center'>The pre-processing step has not been completed<br>"
                        "Please return to the pre-processing tab and compute a distance matrix</p>")
                self.tabWidget.setCurrentIndex(0)
            
        self.preProcessingTab.lockGUI(lock)
        self.labellingTab.lockGUI(lock)
        
    def createGUI(self) :
        
        ## WIDGETS ##

        self.infoLabel = QtGui.QLabel("No data loaded")
        self.infoLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.infoLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        self.openVideoButton = QtGui.QPushButton("Open &Video")
        self.openVideoButton.setEnabled(False)
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
        
        self.preProcessingTab = pptg.PreProcessingTab(self)
        self.videoTexturesTab = vttg.VideoTexturesTab(self)
        self.labellingTab = LabellingTab(self)

        self.tabWidget = QtGui.QTabWidget()
        self.tabWidget.addTab(self.preProcessingTab, self.tr("Pre-Processing"))
        self.tabWidget.addTab(self.videoTexturesTab, self.tr("Video-Textures"))
        self.tabWidget.addTab(self.labellingTab, self.tr("Labelling"))
        
        ## SIGNALS ##
        
        self.openVideoButton.clicked.connect(self.openVideo)
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
        buttonLayout.addWidget(self.openVideoButton)
        
        mainBoxLayout.addLayout(buttonLayout)
        mainBox.setLayout(mainBoxLayout)
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.tabWidget)
        mainLayout.addWidget(mainBox)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

from scipy import special
print special.binom(window.preProcessingTab.numFilterFrames*2, range(0, window.preProcessingTab.numFilterFrames*2 +1))

# <codecell>

figure(); imshow(np.load(dataFolder + "palm_tree1/proc_distMat.npy"), interpolation='nearest')

# <codecell>

gwv.showCustomGraph(np.load(dataFolder + "flag_blender/proc_distMat.npy"))

# <codecell>

print window.frameNames[189]
print window.frameNames

# <codecell>

matteNames = np.sort(glob.glob(window.dataLocation + os.sep + "matte-*.png"))
img = np.array(cv2.cvtColor(cv2.imread(window.frameNames[189]), cv2.COLOR_BGR2RGB))
alpha = np.array(cv2.cvtColor(cv2.imread(matteNames[189]), cv2.COLOR_BGR2GRAY))/255.0
f1 = img*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
figure(); imshow(f1)

# <codecell>

data_frames = []
trimaps = []
for frameName in window.preProcessingTab.trimaps.keys() :
    idx = np.ndarray.flatten(np.argwhere(window.preProcessingTab.frameNames == window.preProcessingTab.dataLocation + os.sep + frameName))[0]
    data_frames.append(cv2.cvtColor(cv2.imread(window.preProcessingTab.frameNames[idx]), cv2.COLOR_BGR2RGB))
    trimaps.append(window.preProcessingTab.trimaps[frameName])

# classifier = self.trainClassifier(data, tmaps)

# <codecell>

idxs = np.indices(data_frames[0].shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
#         print idxs.shape, data_frames[0].shape
data = np.concatenate((data_frames[0], idxs), axis=-1)

# extract training data
background = data[trimaps[0] == 0]
foreground = data[trimaps[0] == 2]

for data_frame, trimap in zip(data_frames[1:], trimaps[1:]) :
#             print background.shape, foreground.shape
    
    idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
#             print idxs.shape, data_frame.shape
    data = np.concatenate((data_frame, idxs), axis=-1)

    # extract training data
    background = np.vstack((background, data[trimap == 0]))
    foreground = np.vstack((foreground, data[trimap == 2]))
    
#         print background.shape, foreground.shape

X = np.vstack((background, foreground))
y = np.repeat([0.0, 1.0], [background.shape[0], foreground.shape[0]])

# <codecell>

del clf

# <codecell>

clf = ExtraTreesClassifier(verbose=1)
clf.fit(X, y)

# <codecell>

figure(); imshow(window.videoTexturesTab.probabilities, interpolation='nearest')

# <codecell>

figure(); imshow(window.videoTexturesTab.cumProb, interpolation='nearest')
print window.videoTexturesTab.frameRange

# <codecell>

figure(); imshow(np.cumsum(ss.t.pdf(window.videoTexturesTab.distMat, 1.0), axis=-1))

# <codecell>

# testDistMat = np.load(dataFolder+"eu_flag_ph_left/proc_distMat.npy")
testDistMat = np.load(dataFolder+"eu_flag_ph_left/proc_hist2demd_32x48_distMat.npy")

# <codecell>

frameRange = np.array([180, 350, 520])#window.videoTexturesTab.frameRange
rangeDistances = window.videoTexturesTab.getRangeDistances(0.7, frameRange, 0.5, testDistMat.shape)
print rangeDistances.shape

# <codecell>

sigma = 0.05
# tdistProbs = np.zeros((1199, 1199))
# # t = (window.videoTexturesTab.distMat - np.repeat(np.mean(window.videoTexturesTab.distMat, axis=-1).reshape((1199, 1)), 1199, axis=-1))/sigma
# for i in xrange(1199) :
#     t = window.videoTexturesTab.distMat[i, :]/sigma
#     tdistProbs[i, :] = ss.t.pdf(t, 10.0)/sigma
#     tdistProbs[i, :] /= np.sum(tdistProbs[i, :])

t = testDistMat/sigma + rangeDistances
tProbs = ss.t.pdf(t, 10.0)/sigma
## normalize
tProbs /= np.repeat(np.sum(tProbs, axis=-1).reshape((tProbs.shape[0], 1)), tProbs.shape[1], axis=-1)

# figure(); imshow(tProbs, interpolation='nearest')
figure(); imshow(np.cumsum(tProbs, axis=-1), interpolation='nearest')

# <codecell>

print window

# <codecell>

print arange(startFrame-5, startFrame+6)
print cumProb[startFrame, 280]

# <codecell>

# frameRange = window.videoTexturesTab.frameRange
probs, cumProb = vtu.getProbabilities(testDistMat, 0.005, rangeDistances, True)# window.videoTexturesTab.cumProb
startFrame = frameRange[0]+np.argmin(np.round(np.sum(cumProb[:, frameRange[0]:frameRange[2]+1] < 0.5, axis=0))) 
# startFrameProbs = np.max(rangeDistances[0, frameRange[0]:frameRange[-1]])-rangeDistances[0, frameRange[0]:frameRange[-1]]
# startFrameProbs = startFrameProbs / float(np.sum(startFrameProbs))
# startFrameProbs = np.concatenate((np.zeros(frameRange[0]), startFrameProbs, np.zeros(tProbs.shape[0]-frameRange[-1])))
# startFrame = np.random.choice(arange(tProbs.shape[0]), p=startFrameProbs)
print frameRange, startFrame
# print gwv.showCustomGraph(cumProb[startFrame-5:startFrame+6, :])
# finalFrames = vtu.getFinalFrames(cumProb, 100, 5, startFrame, True, True)
finalFrames = vtu.getFinalFrames(np.cumsum(tProbs, axis=-1), 100, 5, startFrame, True, True)

# <codecell>

rangeDistances = window.videoTexturesTab.getRangeDistances(0.7, window.videoTexturesTab.frameRange, window.videoTexturesTab.featherLevel, window.videoTexturesTab.distMat.shape)
probabilities, cumProb = vtu.getProbabilities(window.videoTexturesTab.distMat, 0.005, rangeDistances, True)

# <codecell>

import VideoTexturesUtils as vtu

# <codecell>

prev = finalFrames[0]-1
count = 0
for f in finalFrames :
    print f, 
    if f != prev + 1 :
        print "jump"
        count += 1
    else :
        print
    prev = f
print "jumps", count

# <codecell>

## find best transitions given the processed distance matrix
testDistMat = np.load(dataFolder+"flag_blender/proc_distMat.npy")
# testDistMat = np.load(dataFolder+"eu_flag_ph_left/proc_hist2demd_32x48_distMat.npy")
figure(); imshow(testDistMat, interpolation='nearest')

sortedBestFirst = np.argsort(testDistMat, axis=-1)

shortestJumpLength = 100
longestJumpLength = 400
numBestTransitionPerFrame = 10
numTotalTransitions = 50
bestTransitions = np.zeros((len(testDistMat), numBestTransitionPerFrame), dtype=int)
bestTransitionsCosts = np.zeros((len(testDistMat), numBestTransitionPerFrame))

for i in xrange(len(testDistMat)) :
    condition = np.all((np.abs(sortedBestFirst[i, :]-i) > shortestJumpLength, np.abs(sortedBestFirst[i, :]-i) <= longestJumpLength), axis = 0)
#     print sortedBestFirst[i, 0:100]
    bestTransitions[i, :] = np.ndarray.flatten(sortedBestFirst[i, np.argwhere(condition)][0:numBestTransitionPerFrame])
    bestTransitionsCosts[i, :] = testDistMat[i, bestTransitions[i, :]]
#     print testDistMat[i, np.ndarray.flatten(sortedBestFirst[0, np.argwhere(condition)][0:numBestTransitionPerFrame])]

## find best transitions
finalBestTransitions = np.zeros((numTotalTransitions, 2), dtype=int)
for i, bT in zip(arange(numTotalTransitions), np.argsort(np.reshape(bestTransitionsCosts, (np.prod(bestTransitionsCosts.shape))))[0:numTotalTransitions]) :
    finalBestTransitions[i, 0] = bT/numBestTransitionPerFrame
    finalBestTransitions[i, 1] = bestTransitions[bT/numBestTransitionPerFrame, np.mod(bT, numBestTransitionPerFrame)]

# <codecell>

# print testDistMat[0, 1]
# print testDistMat[137, 219]
# print testDistMat[217, 137]
# print np.sort(np.ndarray.flatten(testDistMat))[0:10]
print finalBestTransitions

# <codecell>

print bestTransitionsCosts[298, :]

# <codecell>

print window.labellingTab.labelViewerGroupLeft.labelsMapGraph.transform
maxi = 0
mini = 300
for point in window.labellingTab.labelViewerGroupLeft.labelsMapGraph.points[1].toList() :
    if maxi <= window.labellingTab.labelViewerGroupLeft.labelsMapGraph.transform.map(point).x() :
        maxi = window.labellingTab.labelViewerGroupLeft.labelsMapGraph.transform.map(point).x()
    if mini >= window.labellingTab.labelViewerGroupLeft.labelsMapGraph.transform.map(point).x() :
        mini = window.labellingTab.labelViewerGroupLeft.labelsMapGraph.transform.map(point).x()
        
print mini, maxi
print np.max(window.labellingTab.labelViewerGroupLeft.labelsMapGraph.scatterPoints[1, :])
print 264/np.max(window.labellingTab.labelViewerGroupLeft.labelsMapGraph.scatterPoints[1, :])

# <codecell>

def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
    
if  __name__ =='__main__':main()

# <codecell>

import scipy.io as sio
predictedLabels = sio.loadmat("predictedLabels.mat")["predictedLabels"]
mapPoints = sio.loadmat("mapPoints.mat")["mapPoints"]
print predictedLabels.shape
print mapPoints.shape
print np.ndarray.flatten(np.ndarray.flatten(predictedLabels)[0][0])

# <codecell>

class W(QtGui.QTableWidget):
    def __init__(self):
        super(W,self).__init__(3, 3)

        self.currentCellChanged.connect(self.onSectionClicked)

    def onSectionClicked(self, curr):
        print "onSectionClicked:", curr
        sys.stdout.flush()


#     app = QtGui.QApplication(sys.argv)
w = ScatterGraph("scatter")
# w.setFixedSize(100, 100)
w.setPoints(mapPoints, predictedLabels, 4)
w.show()
app.exec_()

# <codecell>

print len("1234")

# <codecell>

tmp = sio.loadmat("../data/ribbon2"+os.sep+"predictedLabels.mat")["predictedLabels"]
mapPoints = sio.loadmat("../data/ribbon2"+os.sep+"mapPoints.mat")["mapPoints"]
predictedLabels = []
for i in xrange(0,len(np.ndarray.flatten(tmp)[0])) :
    predictedLabels.append(np.ndarray.flatten(np.ndarray.flatten(tmp)[0][i])-1)
    
del tmp
print mapPoints.shape, predictedLabels[0].shape
print mapPoints
print predictedLabels
# idxCorrection = np.int((self.movie.shape[-1]-self.mapPoints.shape[-1])/2)
# labelViewerGroupLeft.setLabelMap(mapPoints, predictedLabels, 4)

# <codecell>

print np.round(window.labellingTab.labelProbs[0, 0]*100, decimals=1)

# <codecell>

print os.sep


# print window.preProcessingTab.trimaps[np.argsort(window.preProcessingTab.trimaps.keys())]
# print window.preProcessingTab.trimaps[np.sort(window.preProcessingTab.trimaps.keys())]
print sorted(window.preProcessingTab.trimaps)

# <codecell>

a = cv2.imread(window.dataLocation + "/000014.png")
b = np.reshape(cv2.cvtColor(cv2.imread(window.dataLocation + "/matte-000014.png"), cv2.COLOR_BGR2GRAY), np.hstack((a.shape[0:-1], 1)))

png = np.concatenate((a, b), axis=-1)
print png.shape, png.strides
# figure(); imshow(tmpImg)
#b = 255 << 24 | a[:,:,0] << 16 | a[:,:,1] << 8 | a[:,:,2] # pack RGB values
im = QtGui.QImage(png, png.shape[1], png.shape[0], png.strides[0], QtGui.QImage.Format_ARGB32)
label = QtGui.QLabel("Hello")
label.resize(300, 300)
label.setPixmap(QtGui.QPixmap(im))
label.show()
app.exec_()

# <codecell>

tmpImg = cv2.cvtColor(cv2.imread(window.dataLocation + "/000014.png"), cv2.COLOR_BGR2RGB)
tmpAlpha = np.reshape(cv2.cvtColor(cv2.imread(window.dataLocation + "/matte-000014.png"), cv2.COLOR_BGR2GRAY), np.hstack((tmpImg.shape[0:-1], 1)))
# figure(); imshow(tmpImg)
# figure(); imshow(tmpAlpha, interpolation='nearest')
print tmpImg.shape, tmpAlpha.shape
print np.concatenate((tmpImg, tmpAlpha), axis=-1).shape

# <codecell>

tmp = window.preProcessingTab.trimaps

print tmp.keys()

print np.argwhere(window.preProcessingTab.frameNames == window.dataLocation + "/" + tmp.keys()[0])
print tmp[0].shape

# tmp.pop("000001.png", None)

# print tmp.keys()



# tmp2 = cv2.resize(window.preProcessingTab.trimaps["000001.png"], (120, 120))
# print tmp2.shape
# figure(); imshow(window.preProcessingTab.trimaps["000001.png"], interpolation='nearest')

# <codecell>

# tmp = window.dataLocation + "/trimap-" + window.frameNames[0].split('/')[-1]
tmp = "/asd/asda/asd/asd/ad-asd-ad.png"
print tmp.split('/')[-1].split('-')[1:]
print filter(None, tmp.split(os.sep)[-1].split('-')[1:])
print os.sep.join(filter(None, tmp.split('/')[-1].split('-')[1:]))
# tmpTrimap = np.array(cv2.cvtColor(cv2.imread(tmp), cv2.COLOR_BGR2GRAY))
# figure(); imshow(tmpTrimap, interpolation='nearest')
# tmp = np.zeros(shape.tmpTrimap)
# print tmpTrimap.shape
# import GraphWithValues as gwv
# gwv.showCustomGraph(tmpTrimap)
# gwv.showCustomGraph(window.preProcessingTab.trimap)

# <codecell>

expandedScribble = np.zeros(scribble.shape, dtype=int32)
expandedScribble[scribble == 0] = 1
expandedScribble[scribble == 255] = 2
cv2.watershed(img, expandedScribble)
mask = np.zeros(expandedScribble.shape)
mask[expandedScribble == 2] = 1
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1,1),np.uint8))
figure(); imshow(mask, interpolation='nearest')

edges = cv2.Canny(np.array(mask, dtype=np.uint8), 1, 2)
edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=3)
figure(); imshow(edges, interpolation='nearest')

trimap = np.zeros(mask.shape)
trimap[mask == 1] = 2
trimap[edges == np.max(edges)] = 1
figure(); imshow(trimap, interpolation='nearest')

# <codecell>

scribble = np.copy(cv2.cvtColor(window.preProcessingTab.scribbleImg, cv2.COLOR_BGR2RGB))
print scribble.shape
figure(); imshow(scribble, interpolation='nearest')
matte = np.zeros(scribble.shape[0:2])
## foreground is red
fgIdx = np.argwhere(scribble[:, :, 0] == 255)
## background is blue
bgIdx = np.argwhere(scribble[:, :, 2] == 255)
matte[fgIdx[:, 0], fgIdx[:, 1]] = 2
matte[bgIdx[:, 0], bgIdx[:, 1]] = 1
# matte[np.argwhere(scribble == np.array([255, 0, 0]))] = 1
figure(); imshow(matte, interpolation='nearest')

# <codecell>

# distMat = np.copy(window.distMat)
figure(); imshow(distMat, interpolation='nearest')
def getRangeDistances(self, l, frameRange, featherLevel, weightShape) :

    kernelLength = np.float(frameRange[1]-frameRange[0])
    print kernelLength
    kernelRange = np.arange(1.0, 0.0, -(1.0)/np.floor(kernelLength*featherLevel))
    print kernelRange
    kernel = np.zeros(kernelLength+1)
    kernel[0:len(kernelRange)] = kernelRange#+np.ones(kernelRange.shape)
    kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
    
    rangeDistances = np.arange(0, weightShape[0])
    rangeDistances = np.abs(rangeDistances - frameRange[1])
    rangeDistances[frameRange[0]:frameRange[2]+1] = rangeDistances[frameRange[0]:frameRange[2]+1]*kernel
    print rangeDistances
    
    rangeDistances = np.repeat(np.reshape(rangeDistances, (1, len(rangeDistances))), len(rangeDistances), axis=0)
    
    return rangeDistances
rangeDistances = getRangeDistances(0.7, [24, 35, 46], 0.5, distMat.shape)
figure(); imshow(rangeDistances, interpolation='nearest')
print rangeWeights

# <codecell>

## Turn distances to probabilities
def rangedist2prob(dM, sigmaMult, rangeDist, normalize) :
    sigma = sigmaMult*np.mean(dM[np.nonzero(dM)])
    print 'sigma', sigma
    pM = np.exp(-(dM/sigma+ rangeDist))
## normalize probabilities row-wise
    if normalize :
        normTerm = np.sum(pM, axis=1)
        normTerm = cv2.repeat(normTerm, 1, dM.shape[1])
        pM = pM / normTerm
    return pM

def getProbabilities(distMat, sigmaMult, rangeDist, normalizeRows) :
    ## compute probabilities from distanceMatrix and the cumulative probabilities
    if normalizeRows = None :
        probabilities = dist2prob(distMat, sigmaMult, normalizeRows)
    else :
        probabilities = rangedist2prob(distMat, sigmaMult, rangeDist, normalizeRows)
    # since the probabilities are normalized on each row, the right most column will be all ones
    cumProb = np.cumsum(probabilities, axis=1)
    print probabilities.shape, cumProb.shape
    
    return probabilities, cumProb

# <codecell>

probs, cumProb = getProbabilities(distMat, 0.005, rangeWeights, True)
figure(); imshow(probs, interpolation='nearest')
figure(); imshow(cumProb, interpolation='nearest')

