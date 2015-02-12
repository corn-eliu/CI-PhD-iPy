# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from PIL import Image
from PySide import QtCore, QtGui

import sys
import numpy as np
import time
import cv2
import re
import glob
import os

import VideoTexturesUtils as vtu

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

# <codecell>

class TextureViewerGroup(QtGui.QGroupBox):
    
    ## don't remember putting this here ???
#     valueChanged = QtCore.Signal(int)

    def __init__(self, title, parent=None):
        super(TextureViewerGroup, self).__init__(title, parent)
        
        ## left-side labels
        self.imageLabel = ImageLabel("Original Video")
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.imageLow = ImageLabel("Lower boundary")
        self.imageLow.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLow.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLow.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.imageHigh = ImageLabel("Higher boundary")
        self.imageHigh.setBackgroundRole(QtGui.QPalette.Base)
        self.imageHigh.setAlignment(QtCore.Qt.AlignCenter)
        self.imageHigh.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        ## place labels for frame boundaries
        boundaryLayout = QtGui.QHBoxLayout()
        boundaryLayout.addWidget(self.imageLow)
        boundaryLayout.addStretch()
        boundaryLayout.addWidget(self.imageHigh)
        
        ## place imageLabel and boundary labels 
        framesLayout = QtGui.QVBoxLayout()
        framesLayout.addWidget(self.imageLabel)
        framesLayout.addStretch()
        framesLayout.addLayout(boundaryLayout)
        
        self.textureLabel = ImageLabel("Video Texture")
        self.textureLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.textureLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.textureLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        labelLayout = QtGui.QHBoxLayout()
        labelLayout.addStretch()
        labelLayout.addLayout(framesLayout)
        labelLayout.addStretch()
        labelLayout.addWidget(self.textureLabel)
        labelLayout.addStretch()

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
    
        self.renderTextureButton = QtGui.QPushButton("&Render Texture")
        
        self.recomputeFramesCheckBox = QtGui.QCheckBox("Re-sample")
        self.recomputeFramesCheckBox.setChecked(True)
        
        self.frameNumLabel = QtGui.QLabel("No rendered vt")
        self.frameNumLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.labelJumpStyle = "QLabel {border: 1px solid black; background: #aa0000; color: white; padding-left: 5px; padding-right: 5px;}"
        self.labelNoJumpStyle = "QLabel {border: 1px solid gray; background: #eeeeee; color: black; padding-left: 5px; padding-right: 5px;}"
        self.frameNumLabel.setStyleSheet(self.labelNoJumpStyle)
        frameNumLayout = QtGui.QHBoxLayout()
        frameNumLayout.addStretch()
        frameNumLayout.addWidget(self.frameNumLabel)
        frameNumLayout.addStretch()
    
        comandLayout = QtGui.QHBoxLayout()
        comandLayout.addWidget(self.slider)
        comandLayout.addWidget(self.renderTextureButton)
        comandLayout.addWidget(self.recomputeFramesCheckBox)

        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addLayout(frameNumLayout)
        mainLayout.addStretch()
        mainLayout.addLayout(labelLayout)
        mainLayout.addStretch()
        mainLayout.addLayout(comandLayout)
        self.setLayout(mainLayout)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        
    def setFrame(self, im):
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
                
    def setFrameLow(self, im):
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.imageLow.setPixmap(QtGui.QPixmap.fromImage(qim))
        
    def setFrameHigh(self, im):
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.imageHigh.setPixmap(QtGui.QPixmap.fromImage(qim))
        
    def setTextureFrame(self, im, alpha):
        im = np.ascontiguousarray(im)
        
        if alpha :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
        else :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            
        self.textureLabel.setPixmap(QtGui.QPixmap.fromImage(qim))

    def setValue(self, value):
        self.slider.setValue(value)

    def setMinimum(self, value):
        self.slider.setMinimum(value)  

    def setMaximum(self, value):
        self.slider.setMaximum(value)
        
    def setFrameInterval(self, interval, feather):
        s2 = np.float(interval[0])/np.float(self.slider.maximum()-self.slider.minimum())
        s4 = np.float(interval[1])/np.float(self.slider.maximum()-self.slider.minimum())
        s3 = s2+((s4-s2)*feather)
        s6 = np.float(interval[2])/np.float(self.slider.maximum()-self.slider.minimum())
        s5 = s6-((s6-s4)*feather)
        style = self.slider.styleSheet()
        style += "QSlider::groove:horizontal {background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0.0 #eeeeee, "
        style += "stop:"+np.str_(s2)+" #eeeeee, stop:"+np.str_(s3)+" #ff0000, stop:"+np.str_(s4)+" #ff0000, "
        style += "stop:"+np.str_(s5)+" #ff0000, stop:"+np.str_(s6)+" #eeeeee, stop:1.0 #eeeeee);} "
        self.slider.setStyleSheet(style)

# <codecell>

class VideoTexturesTab(QtGui.QWidget):
    def __init__(self, mainWindow, parent=None):
        super(VideoTexturesTab, self).__init__(parent)
        
        self.mainWindow = mainWindow
        self.frameRange = np.zeros(3)
        self.featherLevel = 0.5

        self.textureViewerGroup = TextureViewerGroup("Video Texture")
        
        self.numFilterFrames = 4
        self.numInterpolationFrames = 4
        self.textureLength = 100
        self.loopTexture = True
        self.showMatted = True
        self.availableFrames = 0
        self.requestedTextureUpdate = False
        self.lastVideoTextureUpdate = time.time()
        self.updateDelay = 2000 ## msecs

        self.createGUI()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.controlsGroup)
        layout.addWidget(self.textureViewerGroup)
        self.setLayout(layout)
        
        self.frameSpinBox.setValue(0)
        
        self.textureTimer = QtCore.QTimer(self)
        self.textureTimer.setInterval(1000/30)
        self.textureTimer.start()
        self.textureTimer.timeout.connect(self.renderOneFrame)
        self.visualizeTexture = False
        self.currentVisFrame = 0
        
        
        self.textureViewerGroup.imageLabel.setFixedWidth(1920/3)
        self.textureViewerGroup.textureLabel.setFixedWidth(1920/3)
        
        self.lockGUI(True)
        
    def renderOneFrame(self):
        if self.visualizeTexture:
            try:
                self.finalMovie
                self.currentVisFrame
            except AttributeError:
                return
            else:
                if self.currentVisFrame < 0 or self.currentVisFrame >= len(self.finalMovie) :
                    self.currentVisFrame = 0
                    
                frameName = self.frameNames[self.finalFrames[self.currentVisFrame]].split(os.sep)[-1]
                
                if self.showMatted and os.path.isfile(self.dataLocation + os.sep + "matte-" + frameName) :
                    alphaMatte = cv2.cvtColor(cv2.imread(self.dataLocation + os.sep + "matte-" + frameName), cv2.COLOR_BGR2GRAY)
                    if self.downsampleFactor > 1 :
                        alphaMatte = cv2.resize(alphaMatte, (alphaMatte.shape[1]/self.downsampleFactor, alphaMatte.shape[0]/self.downsampleFactor))
                    alphaMatte = np.reshape(alphaMatte, np.hstack((self.movie.shape[0:2], 1)))
                    self.textureViewerGroup.setTextureFrame(np.concatenate((cv2.cvtColor(self.finalMovie[self.currentVisFrame], cv2.COLOR_RGB2BGR), alphaMatte), axis=-1), True)
                else :
                    self.textureViewerGroup.setTextureFrame(self.finalMovie[self.currentVisFrame], False)
                    
                if self.currentVisFrame in self.finalJumps :
                    self.textureViewerGroup.frameNumLabel.setStyleSheet(self.textureViewerGroup.labelJumpStyle)
#                     self.textureViewerGroup.frameNumLabel.setText("jump---" + np.str_(self.currentVisFrame) + " from " + np.str_(self.finalFrames[self.currentVisFrame]) + "---jump")
                else :
                    self.textureViewerGroup.frameNumLabel.setStyleSheet(self.textureViewerGroup.labelNoJumpStyle)
#                     self.textureViewerGroup.frameNumLabel.setText(np.str_(self.currentVisFrame) + " from " + np.str_(self.finalFrames[self.currentVisFrame]))
                self.textureViewerGroup.frameNumLabel.setText(np.str_(self.currentVisFrame) + " from " + np.str_(self.finalFrames[self.currentVisFrame]))
                self.currentVisFrame = np.mod(self.currentVisFrame+1, len(self.finalMovie))
            
    def setRenderFps(self, value) :
        self.textureTimer.setInterval(1000/value)
        
    def setLoadedMovie(self, movie, frameNames, dataLocation, downsampleFactor):#, distMat):
        self.movie = movie
        self.frameNames = frameNames
        self.dataLocation = dataLocation
        self.downsampleFactor = downsampleFactor
        self.distMat = None
        
        ## set size of boundary frames
        self.textureViewerGroup.imageLow.setFixedSize(self.movie.shape[1]/2, self.movie.shape[0]/2)
        self.textureViewerGroup.imageHigh.setFixedSize(self.movie.shape[1]/2, self.movie.shape[0]/2)
        
        self.availableFrames = self.movie.shape[-1]-(self.numFilterFrames+1)*2
            
        if self.frameSpinBox.value() < self.availableFrames :
            self.textureViewerGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, self.frameSpinBox.value()+(self.numFilterFrames+1)]))
        else :
            self.textureViewerGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, 0+(self.numFilterFrames+1)]))
            self.frameSpinBox.setValue(0)
            
        self.frameSpinBox.setRange(0, self.availableFrames-1)
        self.textureViewerGroup.setMaximum(self.availableFrames-1)
        self.setFrameRange(self.frameRangeSpinBox.value())
        self.frameRangeSpinBox.setRange(1, self.availableFrames*0.75)
        
    def setDistMat(self, distMat):
        self.distMat = distMat
        
        
        ############# BIIIIG HAAAAACKKKK #############
        sortedBestFirst = np.argsort(distMat, axis=-1)

        shortestJumpLength = 100
        longestJumpLength = 400
        numBestTransitionPerFrame = 10
        numTotalTransitions = 50
        bestTransitions = np.zeros((len(distMat), numBestTransitionPerFrame), dtype=int)
        bestTransitionsCosts = np.zeros((len(distMat), numBestTransitionPerFrame))
        
        for i in xrange(len(distMat)) :
            condition = np.all((np.abs(sortedBestFirst[i, :]-i) > shortestJumpLength, np.abs(sortedBestFirst[i, :]-i) <= longestJumpLength), axis = 0)
        #     print sortedBestFirst[i, 0:100]
            bestTransitions[i, :] = np.ndarray.flatten(sortedBestFirst[i, np.argwhere(condition)][0:numBestTransitionPerFrame])
            bestTransitionsCosts[i, :] = distMat[i, bestTransitions[i, :]]
        #     print distMat[i, np.ndarray.flatten(sortedBestFirst[0, np.argwhere(condition)][0:numBestTransitionPerFrame])]
        
        ## find best transitions
        self.finalBestTransitions = np.zeros((numTotalTransitions, 2), dtype=int)
        for i, bT in zip(np.arange(numTotalTransitions), np.argsort(np.reshape(bestTransitionsCosts, (np.prod(bestTransitionsCosts.shape))))[0:numTotalTransitions]) :
            self.finalBestTransitions[i, 0] = bT/numBestTransitionPerFrame
            self.finalBestTransitions[i, 1] = bestTransitions[bT/numBestTransitionPerFrame, np.mod(bT, numBestTransitionPerFrame)]
            
        self.currentTransition = 0
        ############# BIIIIG HAAAAACKKKK #############
    
    def saveVideoTexture(self):
        try :
            self.finalFrames
            self.finalMovie
            self.frameNames
        except AttributeError:
            QtGui.QMessageBox.critical(self, "No video-texture", "There is no available video-texture.\nAborting...")
            return
        
        destFolder = QtGui.QFileDialog.getExistingDirectory(self, "Save Video Texture",
                QtCore.QDir.homePath())
        if destFolder :
            self.lockGUI(True)
            self.mainWindow.showLoading(True)
            
            self.mainWindow.loadingLabel.setText("Saving video texture...")
            self.loadingText = self.mainWindow.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()
            
            baseFolder = os.sep.join(filter(None, self.frameNames[0].split(os.sep))[0:-1])
            if os.sep == '/' :
                baseFolder = os.sep+baseFolder
                
#             useMatte = os.path.isfile(baseFolder + os.sep + "matte-" + self.frameNames[0].split(os.sep)[-1])
            matteNames = glob.glob(baseFolder + os.sep + "matte-*.png")
            matteNames = np.sort(matteNames)
            
            for i in range(0, len(self.finalMovie)) :
                
                digits = len(np.string_(i+1))
                if digits == 1 :
                    zeroes = "0000"
                elif digits == 2 :
                    zeroes = "000"
                elif digits == 3 :
                    zeroes = "00"
                elif digits == 4 :
                    zeroes = "0"
                else :
                    zeroes = ""
                
                
                sys.stdout.write('\r' + "Saving frame " + np.string_(i) + " of " + np.string_(len(self.finalMovie)))
                sys.stdout.flush()
                self.mainWindow.loadingLabel.setText(self.loadingText+"\nframe " + np.string_(i) + " of " + np.string_(len(self.finalMovie)))
                QtCore.QCoreApplication.processEvents() 
                
                if self.finalFrames[i] >= 0 and self.finalFrames[i] < len(matteNames):
                    ## HACK:: assume mattes are available and are good(i.e. finalMovie has no interpolated frames)
                    ##mattename = baseFolder + "\\matte-" + self.frameNames[self.finalFrames[i]].split('\\')[-1]
#                     mattename = baseFolder + os.sep + "matte-" + self.frameNames[self.finalFrames[i]].split(os.sep)[-1]
                    matte = Image.open(matteNames[self.finalFrames[i]])
                    matte = np.reshape(np.array(matte, dtype=np.uint8), (matte.size[1], matte.size[0], 1))
                    
                    fullResoFrame = Image.open(self.frameNames[self.finalFrames[i]])
                    fullResoFrame = np.array(fullResoFrame, dtype=np.uint8)[:, :, 0:self.finalMovie[0].shape[-1]]
                    
                    Image.frombytes("RGBA", (fullResoFrame.shape[1], fullResoFrame.shape[0]), np.concatenate((fullResoFrame, matte), axis=-1).tostring()).save(destFolder+os.sep+"vt."+zeroes+np.string_(i+1)+".png")
                else :
                    fullResoFrame = Image.open(self.frameNames[self.finalFrames[i]])
                    fullResoFrame = np.array(fullResoFrame, dtype=np.uint8)[:, :, 0:self.finalMovie[0].shape[-1]]
                    
                    Image.frombytes("RGB", (fullResoFrame.shape[1], fullResoFrame.shape[0]), fullResoFrame.tostring()).save(destFolder+os.sep+"vt."+zeroes+np.string_(i+1)+".png")
                
            print
            
            self.lockGUI(False)
            self.mainWindow.showLoading(False)
        
    def setFrameRange(self, value) :
        currentFrame = self.textureViewerGroup.slider.value()
        if currentFrame-(value-1)/2 >= 0 and currentFrame+(value-1)/2 < self.availableFrames :
            self.frameRange = np.array([currentFrame-((value-1)/2), currentFrame, currentFrame+((value-1)/2)])
            self.textureViewerGroup.setFrameInterval(self.frameRange, self.featherLevel)

            if self.frameRange[0] >= 0 and self.frameRange[0] < self.availableFrames :
                self.textureViewerGroup.setFrameLow(np.ascontiguousarray(self.movie[:, :, :, self.frameRange[0]+(self.numFilterFrames+1)]))
            if self.frameRange[-1] >= 0 and self.frameRange[-1] < self.availableFrames :
                self.textureViewerGroup.setFrameHigh(np.ascontiguousarray(self.movie[:, :, :, self.frameRange[-1]+(self.numFilterFrames+1)]))
        else :
            self.frameRangeSpinBox.setValue((self.frameRange[1]-self.frameRange[0])*2+1)
        
            
        self.requestTextureUpdate()
        
    def setFeatherLevel(self, value) :
        self.featherLevel = value
        self.textureViewerGroup.setFrameInterval(self.frameRange, self.featherLevel)
    
    def setNumInterpolationFrames(self, value) :
        self.numInterpolationFrames = value
    
    def setTextureLength(self, value) :
        self.textureLength = value
        
    def setLoopTexture(self, value) :
        self.loopTexture = (value == QtCore.Qt.Checked)
        
    def setShowMatted(self, value) :
        self.showMatted = (value == QtCore.Qt.Checked)
        
    def changeFrame(self, idx):
        if idx >= 0 and idx < self.availableFrames:
            if idx >= (self.frameRangeSpinBox.value()-1)/2 and idx < self.availableFrames-(self.frameRangeSpinBox.value()-1)/2 :
                self.textureViewerGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, idx+(self.numFilterFrames+1)]))
                self.setFrameRange(self.frameRangeSpinBox.value())
            else :
                self.textureViewerGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, self.frameRange[1]+(self.numFilterFrames+1)]))
                self.textureViewerGroup.slider.setValue(self.frameRange[1])
                self.frameSpinBox.setValue(self.frameRange[1])
            
            self.requestTextureUpdate()
    
    def lockGUI(self, lock):
        if lock :
            self.bfVisText = self.visualizeTexture
            self.visualizeTexture = False
        else :
            self.visualizeTexture = self.bfVisText
        self.textureViewerGroup.slider.setEnabled(not lock)
        self.textureViewerGroup.renderTextureButton.setEnabled(not lock)
        self.textureViewerGroup.recomputeFramesCheckBox.setEnabled(not lock)
        
        self.saveVTButton.setEnabled(not lock)
        
        self.frameSpinBox.setEnabled(not lock)
        self.textureLengthSpinBox.setEnabled(not lock)
        self.loopTextureCheckBox.setEnabled(not lock)
        self.showMattedCheckBox.setEnabled(not lock)
        self.renderFpsSpinBox.setEnabled(not lock)
        
        self.frameRangeSpinBox.setEnabled(not lock)
        self.featherLevelSpinBox.setEnabled(not lock)
        self.numInterpolationFramesSpinBox.setEnabled(not lock)
        
        
    def computeRangedProbs(self, distMat, rangeDistances):
        self.probabilities, self.cumProb = vtu.getProbabilities(distMat, 0.005, rangeDistances, True)
    
    def getRangeDistances(self, l, frameRange, featherLevel, weightShape) :

        kernelLength = np.float(frameRange[1]-frameRange[0])
#         print kernelLength
        kernelRange = np.arange(1.0, 0.0, -(1.0)/np.floor(kernelLength*featherLevel))
#         print kernelRange
        kernel = np.zeros(np.int(kernelLength+1))
        kernel[0:len(kernelRange)] = kernelRange
        kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
        
        rangeDistances = np.arange(0, weightShape[0])
        rangeDistances = np.abs(rangeDistances - frameRange[1])
        rangeDistances[frameRange[0]:frameRange[2]+1] = rangeDistances[frameRange[0]:frameRange[2]+1]*kernel
#         print rangeDistances
        
        rangeDistances = np.repeat(np.reshape(rangeDistances, (1, len(rangeDistances))), len(rangeDistances), axis=0)
        
        return rangeDistances
    
    def requestTextureUpdate(self) :
        if not self.requestedTextureUpdate :
            if time.time() - self.lastVideoTextureUpdate > self.updateDelay/1000 :
                print "immediate"
                self.updateTexture()
            else :
                print "delayed"
                self.requestedTextureUpdate = True
                QtCore.QTimer.singleShot(self.updateDelay, self.updateTexture)
            self.lastVideoTextureUpdate = time.time()
    
    def updateTexture(self):
        try:
            self.distMat
        except AttributeError:
            print "tried updating rendering before future cost estimation"
        
        if self.distMat != None :
        
            self.lockGUI(True)
            
            if self.frameRangeSpinBox.value() > 1 :
                frameRange = self.frameRange
                rangeDistances = self.getRangeDistances(0.7, frameRange, self.featherLevel, self.distMat.shape)
                self.computeRangedProbs(self.distMat, rangeDistances)
                startFrame = frameRange[0]+np.argmin(np.round(np.sum(self.cumProb[:, frameRange[0]:frameRange[2]+1] < 0.5, axis=0)))
            else :
                self.computeRangedProbs(self.distMat, None)
                startFrame = np.argmin(np.round(np.sum(self.cumProb < 0.5, axis=0)))
    
            print "starting frame=", startFrame
        
            self.loadingText = self.mainWindow.loadingLabel.text()
            try:
                self.finalFrames
            except AttributeError: 
                self.mainWindow.loadingLabel.setText(self.loadingText+"\nFinding best frames...")
                self.loadingText = self.mainWindow.loadingLabel.text()
                QtCore.QCoreApplication.processEvents()
                self.finalFrames = vtu.getFinalFrames(self.cumProb, self.textureLength, self.numFilterFrames+1, startFrame, self.loopTexture, False)
            else:
                if self.textureViewerGroup.recomputeFramesCheckBox.isChecked() :
                    self.mainWindow.loadingLabel.setText(self.loadingText+"\nFinding best frames...")
                    self.loadingText = self.mainWindow.loadingLabel.text()
                    QtCore.QCoreApplication.processEvents()
                    self.finalFrames = vtu.getFinalFrames(self.cumProb, 100, self.numFilterFrames+1, startFrame, self.loopTexture, False)
            
            ############# BIIIIG HAAAAACKKKK #############
#             self.finalFrames = np.arange(np.min(self.finalBestTransitions[self.currentTransition, :]), np.max(self.finalBestTransitions[self.currentTransition, :])+1)
#             print "rendering transition", self.currentTransition, self.finalBestTransitions[self.currentTransition, :]
#             self.currentTransition = np.mod(self.currentTransition+1, len(self.finalBestTransitions))
            ############# BIIIIG HAAAAACKKKK #############
            
            
            self.finalMovie, self.finalJumps = vtu.renderFinalFrames(self.movie, self.finalFrames, self.numInterpolationFrames)
                        
            print "total final frames =", len(self.finalFrames), "- jumps =", len(self.finalJumps)
            
            self.lockGUI(False)
            self.visualizeTexture = True
            
            
            self.requestedTextureUpdate = False

    def createGUI(self):
        
        ## WIDGETS ##
        
        self.controlsGroup = QtGui.QGroupBox("Controls")
        self.controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        self.controlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        frameLabel = QtGui.QLabel("Current frame:")
        renderFpsLabel = QtGui.QLabel("Texture render fps:")
        loopTextureLabel = QtGui.QLabel("Loop texture:")
        showMattedLabel = QtGui.QLabel("Show matted:")
        textureLengthLabel = QtGui.QLabel("Video texture length:")
        frameRangeLabel = QtGui.QLabel("Frame range:")
        featherLevelLabel = QtGui.QLabel("Feather level:")
        numInterpolationFramesLabel = QtGui.QLabel("# Frames to interpolate:")
        
#         self.tabBar = QtGui.QTabBar(
        self.saveVTButton = QtGui.QPushButton("Save Video-&Texture")

        self.frameSpinBox = QtGui.QSpinBox()
        self.frameSpinBox.setRange(0, 0)
        self.frameSpinBox.setSingleStep(1)
        
        self.textureLengthSpinBox = QtGui.QSpinBox()
        self.textureLengthSpinBox.setRange(30, 1000)
        self.textureLengthSpinBox.setSingleStep(1)
        self.textureLengthSpinBox.setValue(self.textureLength)
        
        self.loopTextureCheckBox = QtGui.QCheckBox()
        self.loopTextureCheckBox.setChecked(self.loopTexture)
        
        self.showMattedCheckBox = QtGui.QCheckBox()
        self.showMattedCheckBox.setChecked(self.showMatted)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 60)
        self.renderFpsSpinBox.setSingleStep(1)
        self.renderFpsSpinBox.setValue(30)
        
        self.frameRangeSpinBox = QtGui.QSpinBox()
        self.frameRangeSpinBox.setRange(1, 21)
        self.frameRangeSpinBox.setSingleStep(2)
        
        self.featherLevelSpinBox = QtGui.QDoubleSpinBox()
        self.featherLevelSpinBox.setRange(0.01, 1.0)
        self.featherLevelSpinBox.setSingleStep(0.01)
        self.featherLevelSpinBox.setValue(self.featherLevel)
        
        self.numInterpolationFramesSpinBox = QtGui.QSpinBox()
        self.numInterpolationFramesSpinBox.setRange(0, 10)
        self.numInterpolationFramesSpinBox.setSingleStep(1)
        self.numInterpolationFramesSpinBox.setValue(self.numInterpolationFrames)
        
        ## SIGNALS ##
        
        self.saveVTButton.clicked.connect(self.saveVideoTexture)
        
        self.textureViewerGroup.slider.valueChanged[int].connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.textureViewerGroup.slider.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.changeFrame)
        self.textureLengthSpinBox.valueChanged[int].connect(self.setTextureLength)
        self.loopTextureCheckBox.stateChanged[int].connect(self.setLoopTexture)
        self.showMattedCheckBox.stateChanged[int].connect(self.setShowMatted)
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        
        self.frameRangeSpinBox.valueChanged[int].connect(self.setFrameRange)
        self.featherLevelSpinBox.valueChanged[float].connect(self.setFeatherLevel)
        self.numInterpolationFramesSpinBox.valueChanged[int].connect(self.setNumInterpolationFrames)
        
        self.textureViewerGroup.renderTextureButton.clicked.connect(self.updateTexture)
        
        ## LAYOUTS ##

        renderingControls = QtGui.QGroupBox("Texture Rendering Controls")
        renderingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        renderingControlsLayout = QtGui.QGridLayout()
        renderingControlsLayout.addWidget(frameLabel, 0, 0)
        renderingControlsLayout.addWidget(self.frameSpinBox, 0, 1)
        renderingControlsLayout.addWidget(loopTextureLabel, 1, 0)
        renderingControlsLayout.addWidget(self.loopTextureCheckBox, 1, 1)
        renderingControlsLayout.addWidget(showMattedLabel, 2, 0)
        renderingControlsLayout.addWidget(self.showMattedCheckBox, 2, 1)
        renderingControlsLayout.addWidget(renderFpsLabel, 3, 0)
        renderingControlsLayout.addWidget(self.renderFpsSpinBox, 3, 1)
        renderingControlsLayout.addWidget(self.saveVTButton, 4, 0, 1, 2)
        renderingControls.setLayout(renderingControlsLayout)
        
        processingControls = QtGui.QGroupBox("Processing Controls")
        processingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        processingControlsLayout = QtGui.QGridLayout()
        processingControlsLayout.addWidget(textureLengthLabel, 0, 0)
        processingControlsLayout.addWidget(self.textureLengthSpinBox, 0, 1)
        processingControlsLayout.addWidget(frameRangeLabel, 1, 0)
        processingControlsLayout.addWidget(self.frameRangeSpinBox, 1, 1)
        processingControlsLayout.addWidget(featherLevelLabel, 2, 0)
        processingControlsLayout.addWidget(self.featherLevelSpinBox, 2, 1)
        processingControlsLayout.addWidget(numInterpolationFramesLabel, 3, 0)
        processingControlsLayout.addWidget(self.numInterpolationFramesSpinBox, 3, 1)
        processingControls.setLayout(processingControlsLayout)
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addStretch()
        controlsLayout.addWidget(renderingControls)
        controlsLayout.addWidget(processingControls)
        controlsLayout.addStretch()
        self.controlsGroup.setLayout(controlsLayout)

