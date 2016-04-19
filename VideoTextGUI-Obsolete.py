# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
from PySide import QtCore, QtGui
import sys
import numpy as np
import scipy as sp
import cv2
import glob
import re
from PIL import Image

import VideoTexturesUtils as vtu

dataFolder = "../data/"

app = QtGui.QApplication(sys.argv)

# <codecell>

class MainWidgetGroup(QtGui.QGroupBox):

    valueChanged = QtCore.Signal(int)

    def __init__(self, orientation, title, parent=None):
        super(MainWidgetGroup, self).__init__(title, parent)
        
        self.imageLabel = QtGui.QLabel("Original Video")
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.textureLabel = QtGui.QLabel("Video Texture")
        self.textureLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.textureLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
#         self.imageLabel.setScaledContents(True)
        labelLayout = QtGui.QHBoxLayout()
        labelLayout.addStretch()
        labelLayout.addWidget(self.imageLabel)
        labelLayout.addStretch()
        labelLayout.addWidget(self.textureLabel)
        labelLayout.addStretch()

        self.slider = QtGui.QSlider(orientation)
        self.slider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider.setMaximum(0)
        self.slider.setMinimum(0)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        style = "QSlider::handle:horizontal { background: #000000; width: 4px; "
        style += "border-radius: 3px; } "#QSlider::handle::horizontal:hover {width: 8px; margin: -2px 0px} "
        style += "QSlider::groove:horizontal {background: white} "
#         style += "QSlider {border: 2px solid grey;}"
        self.slider.setStyleSheet(style)
    
        self.renderTextureButton = QtGui.QPushButton("&Render Texture")
        
        self.recomputeFramesCheckBox = QtGui.QCheckBox("Re-sample")
        self.recomputeFramesCheckBox.setChecked(True)
        
        self.frameNumLabel = QtGui.QLabel()
        self.frameNumLabel.setAlignment(QtCore.Qt.AlignCenter)
    
        comandLayout = QtGui.QHBoxLayout()
        comandLayout.addWidget(self.slider)
        comandLayout.addWidget(self.renderTextureButton)
        comandLayout.addWidget(self.recomputeFramesCheckBox)

        if orientation == QtCore.Qt.Horizontal:
            direction = QtGui.QBoxLayout.TopToBottom
        else:
            direction = QtGui.QBoxLayout.LeftToRight

        mainLayout = QtGui.QBoxLayout(direction)
        mainLayout.addWidget(self.frameNumLabel)
        mainLayout.addLayout(labelLayout)
        mainLayout.addLayout(comandLayout)
        self.setLayout(mainLayout)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -10px;}")
        
    def setFrame(self, im):
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
    def setTextureFrame(self, im):
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
#         print "stops", 0.0, s2, s3, s4, s5, s6, 1.0
        style = self.slider.styleSheet()
        style += "QSlider::groove:horizontal {background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0.0 #ffffff, "
        style += "stop:"+np.str_(s2)+" #ffffff, stop:"+np.str_(s3)+" #ff0000, stop:"+np.str_(s4)+" #ff0000, "
        style += "stop:"+np.str_(s5)+" #ff0000, stop:"+np.str_(s6)+" #ffffff, stop:1.0 #ffffff);} "
        self.slider.setStyleSheet(style)


class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.frameRange = np.zeros(3)
        self.featherLevel = 0.5

        self.mainWidgetGroup = MainWidgetGroup(QtCore.Qt.Horizontal, "Video Texture")
        
        self.numInterpolationFrames = 4
        self.textureLength = 100
        self.loopTexture = True

        self.createControls("Controls")

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.controlsGroup)
        layout.addWidget(self.mainWidgetGroup)
        self.setLayout(layout)
        
        self.frameSpinBox.setValue(0)

        self.setWindowTitle("Video Textures GUI")
        self.resize(1500, 750)
        
        ## lock gui elements until video or sequence is loaded
        
        self.frameSpinBox.setEnabled(False)
        self.textureLengthSpinBox.setEnabled(False)
        self.loopTextureCheckBox.setEnabled(False)
        self.renderFpsSpinBox.setEnabled(False)
        self.frameRangeSpinBox.setEnabled(False)
        self.featherLevelSpinBox.setEnabled(False)
        self.numInterpolationFramesSpinBox.setEnabled(False)
        
        self.textureTimer = QtCore.QTimer(self)
        self.textureTimer.setInterval(1000/30)
        self.textureTimer.start()
        self.textureTimer.timeout.connect(self.renderOneFrame)
        self.visualizeTexture = False
        self.currentVisFrame = 0
        
    def renderOneFrame(self):
        if self.visualizeTexture:
            try:
                self.finalMovie
                self.currentVisFrame
            except AttributeError:
                return
            else:
                self.mainWidgetGroup.setTextureFrame(self.finalMovie[self.currentVisFrame])
                if self.currentVisFrame in self.finalJumps :
                    self.mainWidgetGroup.frameNumLabel.setText("jump---" + np.str_(self.currentVisFrame) + " from " + np.str_(self.finalFrames[self.currentVisFrame]) + "---jump")
                else :
                    self.mainWidgetGroup.frameNumLabel.setText(np.str_(self.currentVisFrame) + " from " + np.str_(self.finalFrames[self.currentVisFrame]))
                self.currentVisFrame = np.mod(self.currentVisFrame+1, len(self.finalMovie))
        
    def showLoading(self, show) :
        if show :
            self.loadingLabel.setText("Loading... Please wait")
            self.loadingLabel.setVisible(True)
            self.loadingSpinner.setVisible(True)
            self.infoLabel.setVisible(False)
        else :
            self.loadingLabel.setVisible(False)
            self.loadingSpinner.setVisible(False)
            self.infoLabel.setVisible(True)
            
    def setRenderFps(self, value) :
        self.textureTimer.setInterval(1000/value)
        
    def openVideo(self):
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                QtCore.QDir.currentPath()+"/"+dataFolder)
        if fileName :
            self.outputData = fileName
            self.lockGUI(True)
            self.showLoading(True)
#             im = np.array(cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2RGB))
#             self.mainWidgetGroup.setFrame(im)

            self.loadingLabel.setText("Loading set of frames...")
            
            cap = cv2.VideoCapture(fileName)
            
            self.frames = []
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret==True:
                    self.frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    k = cv2.waitKey(30)
                    
                    sys.stdout.write('\r' + "Frames read: " + np.string_(len(self.frames)))
                    sys.stdout.flush()
                else:
                    break
            
            # Release everything if job is finished
            cap.release()
            print
            ## init movie array
            if len(self.frames) > 0 :
                self.movie = np.zeros(np.hstack([self.frames[0].shape, len(self.frames)]))
                for i in xrange(0, len(self.frames)) :
                    self.movie[:, :, :, i] = self.frames[i]
                    
                    sys.stdout.write('\r' + "Frames saved: " + np.string_(i) + " of " + np.string_(len(self.frames)))
                    sys.stdout.flush()
            print        
#             del frames
            
            if self.frameSpinBox.value() < len(self.frames) :
                self.mainWidgetGroup.setFrame(self.frames[self.frameSpinBox.value()])
            else :
                self.mainWidgetGroup.setFrame(self.frames[0])
                self.frameSpinBox.setValue(0)
                
            self.frameSpinBox.setRange(0, len(self.frames)-1)
            self.mainWidgetGroup.setMaximum(len(self.frames)-1)
            self.setFrameRange(self.frameRangeSpinBox.value())
            self.frameRangeSpinBox.setRange(1, len(self.frames)*0.6)
            infoText = "Loaded video:\n\t"+fileName+"\nNumber of Frames:\n\t"+np.string_(len(self.frames))
            infoText += "\nFrame size:\n\t"+np.string_(self.movie.shape[1])+"x"+np.string_(self.movie.shape[0])
            self.infoLabel.setText(infoText)
            
            self.loadingLabel.setText("Preprocessing movie sequence...")
            
            self.preProcessMovie(self.movie, self.outputData)
            
            self.lockGUI(False)
            self.showLoading(False)
    
    def openSequence(self):
        frameNames = np.array(QtGui.QFileDialog.getOpenFileNames(self, "Open Sequence", 
                    QtCore.QDir.currentPath()+"/"+dataFolder, "Images(*.png)")[0])
        if len(frameNames) > 1 :
            self.outputData = "/"+ '/'.join(filter(None, re.split('/',frameNames[0]))[0:-1])
            self.lockGUI(True)
            self.showLoading(True)

            self.loadingLabel.setText("Loading set of frames...")

            ## sort frame names
            frameNames = np.sort(frameNames)
            if len(frameNames) > 0 :
                frameSize = cv2.imread(frameNames[0]).shape
                requiredSpace = 2.0*(np.prod(frameSize)*len(frameNames))/(1024**3) ## GB
                downsampleFactor = 1
                if requiredSpace > 8 :
                    downsampleFactor = 2 + np.argmin(np.abs(8-requiredSpace/np.arange(2.0, 6.0)))
                    print "Downsample Factor of", downsampleFactor
                    
                if downsampleFactor > 1 :
                    self.movie = np.zeros(np.hstack([frameSize[0]/downsampleFactor, frameSize[1]/downsampleFactor, frameSize[2], len(frameNames)]), dtype=uint8)
                else :
                    self.movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], len(frameNames)]), dtype=uint8)
                    
                self.frames = []
                for i in range(0, len(frameNames)) :
                    im = Image.open(frameNames[i])
                    if downsampleFactor > 1 :
                        im = im.resize(np.array(frameSize[0:2])/downsampleFactor)
                    im = np.ascontiguousarray(np.array(im, dtype=uint8)[:, :, 0:3])
                    self.movie[:, :, :, i] = im#np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
                    self.frames.append(im)
                    sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(frameNames)))
                    sys.stdout.flush()
                print
#                 frameSize = cv2.imread(frameNames[0]).shape
#                 self.movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], len(frameNames)]))
#                 self.frames = []
#                 for i in range(0, len(frameNames)) :
#                     im = np.array(cv2.cvtColor(cv2.imread(frameNames[i]), cv2.COLOR_BGR2RGB))#/255.0
#                     self.movie[:, :, :, i] = im#np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
#                     self.frames.append(im)
            
            if self.frameSpinBox.value() < len(self.frames) :
                self.mainWidgetGroup.setFrame(self.frames[self.frameSpinBox.value()])
            else :
                self.mainWidgetGroup.setFrame(self.frames[0])
                self.frameSpinBox.setValue(0)
                
            self.frameSpinBox.setRange(0, len(self.frames)-1)
            self.mainWidgetGroup.setMaximum(len(self.frames)-1)
            self.setFrameRange(self.frameRangeSpinBox.value())
            self.frameRangeSpinBox.setRange(1, len(self.frames)*0.6)
            infoText = "Loaded sequence:\n\t"+self.outputData+"\nNumber of Frames:\n\t"+np.string_(len(self.frames))
            infoText += "\nFrame size:\n\t"+np.string_(self.movie.shape[1])+"x"+np.string_(self.movie.shape[0])
            self.infoLabel.setText(infoText)
            
            self.loadingLabel.setText("Preprocessing movie sequence...")
            
            self.preProcessMovie(self.movie, self.outputData)
            
            self.lockGUI(False)
            self.showLoading(False)
        
    def setFrameRange(self, value) :
        currentFrame = self.mainWidgetGroup.slider.value()
        self.frameRange = np.array([currentFrame-((value-1)/2), currentFrame, currentFrame+((value-1)/2)])
        self.mainWidgetGroup.setFrameInterval(self.frameRange, self.featherLevel)
        
    def setFeatherLevel(self, value) :
        self.featherLevel = value
        self.mainWidgetGroup.setFrameInterval(self.frameRange, self.featherLevel)
    
    def setNumInterpolationFrames(self, value) :
        self.numInterpolationFrames = value
    
    def setTextureLength(self, value) :
        self.textureLength = value
        
    def setLoopTexture(self, value) :
        self.loopTexture = (value == QtCore.Qt.Checked)
        
    def changeFrame(self, idx):
        if idx >= 0 and idx < len(self.frames):
            self.mainWidgetGroup.setFrame(self.frames[idx])
            self.setFrameRange(self.frameRangeSpinBox.value())
    
    def lockGUI(self, lock):
        self.visualizeTexture = not lock
        self.mainWidgetGroup.slider.setEnabled(not lock)
        
        self.openVideoButton.setEnabled(not lock)
        self.openSequenceButton.setEnabled(not lock)
        
        self.frameSpinBox.setEnabled(not lock)
        self.textureLengthSpinBox.setEnabled(not lock)
        self.loopTextureCheckBox.setEnabled(not lock)
        self.renderFpsSpinBox.setEnabled(not lock)
        
        self.frameRangeSpinBox.setEnabled(not lock)
        self.featherLevelSpinBox.setEnabled(not lock)
        self.numInterpolationFramesSpinBox.setEnabled(not lock)
    
    def preProcessMovie(self, movie, outputData):
        self.numFilterFrames = 4
        distanceMatrix = vtu.computeDistanceMatrix(movie, outputData + "distMat.npy")
        self.distanceMatrixFilt = vtu.filterDistanceMatrix(distanceMatrix, self.numFilterFrames, False)
        
#         if self.frameRangeSpinBox.value() > 1 :
#             ## create the filter for distance matrix to only consider user given frame range
#             kernelLength = np.float(self.frameRange[1]-self.frameRange[0])
#             interval = 1.0/(np.floor(kernelLength*self.featherLevel)+1.0)
#             kernelRange = np.arange(interval, 1.0+interval, interval)
#             kernel = np.hstack((kernelRange, np.ones(np.ceil(kernelLength-kernelLength*self.featherLevel))))
#             kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
#             kernel = np.reshape(kernel, [len(kernel), 1])
            
#             kernel = kernel+kernel.T
#             kernel = kernel/np.max(kernel)
# #             kernel = np.repeat(kernel, kernel.shape[0], axis=1)
            
#             filterMat = np.zeros(distanceMatrixFilt.shape)
#             filterMat[self.frameRange[0]:self.frameRange[2]+1, self.frameRange[0]:self.frameRange[2]+1] = kernel
            
#             distanceMatrixFilt = distanceMatrixFilt+((1.0-filterMat)*50)
            
#         self.distMat = vtu.estimateFutureCost(0.999, 2.0, distanceMatrixFilt)
    
    
    
#         fig = plt.figure()
#         a=fig.add_subplot(1,2,1)
#         imgplot = plt.imshow(distanceMatrix, cmap = cm.Greys_r, interpolation='nearest')
#         a.set_title('distance matrix grey')
#         a=fig.add_subplot(1,2,2)
#         imgplot = plt.imshow(distanceMatrix, interpolation='nearest')
#         maxDist = np.max(distanceMatrix)
#         plt.colorbar(ticks=[0.1*maxDist,0.3*maxDist,0.5*maxDist,0.7*maxDist,0.9*maxDist], orientation ='horizontal')
#         a.set_title('distance matrix color')
    
#         fig = plt.figure()
#         a=fig.add_subplot(1,2,1)
#         imgplot = plt.imshow(distanceMatrixFilt, cmap = cm.Greys_r, interpolation='nearest')
#         a.set_title('filtered matrix grey')
#         a=fig.add_subplot(1,2,2)
#         imgplot = plt.imshow(distanceMatrixFilt, interpolation='nearest')
#         maxDist = np.max(distanceMatrixFilt)
#         plt.colorbar(ticks=[0.1*maxDist,0.3*maxDist,0.5*maxDist,0.7*maxDist,0.9*maxDist], orientation ='horizontal')
#         a.set_title('filtered matrix color')
        
#         fig = plt.figure()
#         a=fig.add_subplot(1,2,1)
#         imgplot = plt.imshow(self.distMat, cmap = cm.Greys_r, interpolation='nearest')
#         a.set_title('future cost matrix grey')
#         a=fig.add_subplot(1,2,2)
#         imgplot = plt.imshow(self.distMat, interpolation='nearest')
#         maxDist = np.max(self.distMat)
#         plt.colorbar(ticks=[0.1*maxDist,0.3*maxDist,0.5*maxDist,0.7*maxDist,0.9*maxDist], orientation ='horizontal')
#         a.set_title('future cost matrix color')
        
    def computeRangedProbs(self, distMat):
        
        self.probabilities, self.cumProb = vtu.getProbabilities(distMat, 0.005, True)
        
#         if False : #self.frameRangeSpinBox.value() > 1 :
#             ## create the filter for distance matrix to only consider user given frame range
#             kernelLength = np.float(self.frameRange[1]-self.frameRange[0])
#             interval = 1.0/(np.floor(kernelLength*self.featherLevel)+1.0)
#             kernelRange = np.arange(interval, 1.0+interval, interval)
#             kernel = np.hstack((kernelRange, np.ones(np.ceil(kernelLength-kernelLength*self.featherLevel))))
#             kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
#             kernel = np.reshape(kernel, [len(kernel), 1])
            
#             kernel = kernel+kernel.T
#             kernel = kernel/np.max(kernel)
# #             kernel = np.repeat(kernel, kernel.shape[0], axis=1)
            
#             self.filterMat = np.zeros(self.probabilities.shape)
#             self.filterMat[self.frameRange[0]:self.frameRange[2]+1, self.frameRange[0]:self.frameRange[2]+1] = kernel
#             self.filterMat = self.filterMat+0.01 ## still leave some chance for frames outside range to be chosen
#             self.filterMat = self.filterMat/np.max(self.filterMat)
# #             print kernel
            
#             self.probabilities = self.probabilities*self.filterMat
            
#             ## renormalize probabilities
#             normTerm = np.sum(self.probabilities, axis=1)
#             normTerm = cv2.repeat(normTerm, 1, self.probabilities.shape[1])
#             self.probabilities = self.probabilities / normTerm
            
#             self.cumProb = np.cumsum(self.probabilities, axis=1)
        
#     def renderTexture(self):
#         try:
#             self.distMat
#         except AttributeError:
#             print "tried rendering texture before computing distance matrix"
#         else:
#             print "rendering texture"
#             self.visualizeTexture = False
            
#             self.preProcessMovie(self.movie, self.outputData)
            
#             distMat = self.distMat#+((1.0-filterMat)*1000) ## this needs to be filtered based on frame range and feather
# #             figure(); imshow(distMat, interpolation='nearest')            
            
#             self.computeRangedProbs(distMat)
#             figure(); imshow(self.probabilities, interpolation='nearest'); plt.draw()
#             startFrame = self.frameRange[1]
#             ## check if frames have been computed before, if not compute them, otherwise only recompute if checkbox is checked
#             try:
#                 self.finalFrames
#             except AttributeError:
#                 self.finalFrames = vtu.getFinalFrames(self.cumProb, self.textureLength, self.numFilterFrames+1, startFrame)
#             else:
#                 if self.mainWidgetGroup.recomputeFramesCheckBox.isChecked() :
#                     self.finalFrames = vtu.getFinalFrames(self.cumProb, 100, self.numFilterFrames+1, startFrame)
#             self.finalMovie, self.finalJumps = vtu.renderFinalFrames(self.movie, self.finalFrames, self.numInterpolationFrames)
#             self.visualizeTexture = True

    ## Avoid dead ends: estimate future costs
    def estimateFutureCost(self, alpha, p, distanceMatrixFilt, weights) :
        # alpha = 0.999
        # p = 2.0
        
        distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
        distMat = distMatFilt ** p
        
        last = np.copy(distMat)
        current = np.zeros(distMat.shape)
        
        ## while distance between last and current is larger than threshold
        iterations = 0 
        while np.linalg.norm(last - current) > 0.1 : 
            for i in range(distMat.shape[0]-1, -1, -1) :
                m = np.min(distMat*weights[1:distanceMatrixFilt.shape[1], 0:-1], axis=1)
                distMat[i, :] = (distMatFilt[i, :] ** p) + alpha*m
                
            last = np.copy(current)
            current = np.copy(distMat)
            
            sys.stdout.write('\r' + "Iteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
            sys.stdout.flush()
            
            iterations += 1
        
        print
        print 'finished in', iterations, 'iterations'
        
        return distMat
    
    def getRangeWeights(self, l, frameRange, featherLevel, weightShape) :
#         kernelLength = np.float(frameRange[1]-frameRange[0])
#         interval = l/(np.floor(kernelLength*featherLevel)+1.0)
#         kernelRange = np.arange(1.0+l, 1.0, -interval)
#         kernel = np.hstack((kernelRange, np.ones(np.ceil(kernelLength-kernelLength*featherLevel))))
#         kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
#         kernel = np.reshape(kernel, [len(kernel), 1])

#         kernelLength = np.float(frameRange[1]-frameRange[0])
#         interval = l/np.floor(kernelLength*featherLevel)
#         kernelRange = interval*np.arange(np.floor(kernelLength*featherLevel), 0.0, -1.0)
#         kernel = np.ones(kernelLength+1)
#         kernel[0:len(kernelRange)] = kernelRange+np.ones(kernelRange.shape)
#         kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
#         kernel = np.reshape(kernel, [len(kernel), 1])

        kernelLength = np.float(frameRange[1]-frameRange[0])
        kernelRange = np.arange(1.0, l, -(1.0-l)/np.floor(kernelLength*featherLevel))
        kernel = np.ones(kernelLength+1)*l
        kernel[0:len(kernelRange)] = kernelRange#+np.ones(kernelRange.shape)
        kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
        kernel = np.reshape(kernel, [len(kernel), 1])
        
#         print kernel
        
        kernel = kernel+kernel.T
        kernel = kernel/np.max(kernel)#np.mean(kernel) 
        
        rangeWeights = np.ones(weightShape)+l*0.5
        rangeWeights[frameRange[0]:frameRange[2]+1, frameRange[0]:frameRange[2]+1] = kernel
#         filterMat = self.filterMat+0.01 ## still leave some chance for frames outside range to be chosen
#         filterMat = self.filterMat/np.max(self.filterMat)
#         filterMat = (1.7-self.filterMat)#*10.0 +1.0
        return rangeWeights
     
    def renderTextureNew(self):
        try:
            self.distanceMatrixFilt
        except AttributeError:
            print "tried rendering texture before computing distance matrix"
        else:
            print "rendering texture"
#             self.visualizeTexture = False
            
#             self.numFilterFrames = 4
#             distanceMatrix = vtu.computeDistanceMatrix(self.movie, self.outputData + "distMat.npy")
#             distanceMatrixFilt = vtu.filterDistanceMatrix(distanceMatrix, self.numFilterFrames, False)
                
#             figure(); imshow(distanceMatrixFilt, interpolation='nearest'); draw()
            
            if self.frameRangeSpinBox.value() > 1 :
                frameRange = self.frameRange-(self.numFilterFrames+1)
                print frameRange
                self.rangeWeights = self.getRangeWeights(0.7, frameRange, self.featherLevel, self.distanceMatrixFilt.shape)
#                 figure(); imshow(self.rangeWeights, interpolation='nearest'); draw()
                
                self.distMat = self.estimateFutureCost(0.999, 2.0, self.distanceMatrixFilt, self.rangeWeights)
                startFrame = frameRange[1]
            else :
                self.distMat = self.estimateFutureCost(0.999, 2.0, self.distanceMatrixFilt, np.ones(self.distanceMatrixFilt.shape))
                startFrame = self.frameSpinBox.value()+self.numFilterFrames+1
                
            figure(); imshow(self.distMat, interpolation='nearest'); draw()
#                 self.cumProb = np.cumsum(self.probabilities, axis=1)
            
#             distMat = self.distMat#+((1.0-filterMat)*1000) ## this needs to be filtered based on frame range and feather
# #             figure(); imshow(distMat, interpolation='nearest')
            
            self.computeRangedProbs(self.distMat)
            figure(); imshow(self.cumProb, interpolation='nearest'); plt.draw()
            ## check if frames have been computed before, if not compute them, otherwise only recompute if checkbox is checked
            try:
                self.finalFrames
            except AttributeError:
                self.finalFrames = vtu.getFinalFrames(self.cumProb, self.textureLength, self.numFilterFrames+1, startFrame, self.loopTexture)
            else:
                if self.mainWidgetGroup.recomputeFramesCheckBox.isChecked() :
                    self.finalFrames = vtu.getFinalFrames(self.cumProb, 100, self.numFilterFrames+1, startFrame, self.loopTexture)
            self.finalMovie, self.finalJumps = vtu.renderFinalFrames(self.movie, self.finalFrames, self.numInterpolationFrames)
            self.visualizeTexture = True

    def renderTexture(self):
#         self.renderTextureOld()
        self.renderTextureNew()

#     def renderTextureOld(self):
#         try:
#             self.distMat
#         except AttributeError:
#             print "tried rendering texture before computing distance matrix"
#         else:
#             print "rendering texture"
#             self.visualizeTexture = False
            
#             self.numFilterFrames = 4
#             distanceMatrix = vtu.computeDistanceMatrix(self.movie, self.outputData + "distMat.npy")
#             distanceMatrixFilt = vtu.filterDistanceMatrix(distanceMatrix, self.numFilterFrames, False)
            
#             if self.frameRangeSpinBox.value() > 1 :
#                 frameRange = self.frameRange-(self.numFilterFrames+1)
#                 ## create the filter for distance matrix to only consider user given frame range
#                 kernelLength = np.float(frameRange[1]-frameRange[0])
#                 interval = 1.0/(np.floor(kernelLength*self.featherLevel)+1.0)
#                 kernelRange = np.arange(interval, 1.0+interval, interval)
#                 kernel = np.hstack((kernelRange, np.ones(np.ceil(kernelLength-kernelLength*self.featherLevel))))
#                 kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
#                 kernel = np.reshape(kernel, [len(kernel), 1])
                
#                 print kernel
                
#                 kernel = kernel+kernel.T
#                 self.kernel = kernel/np.max(kernel)
#     #             kernel = np.repeat(kernel, kernel.shape[0], axis=1)

                
#                 self.filterMat = np.zeros(distanceMatrixFilt.shape)
#                 self.filterMat[frameRange[0]:frameRange[2]+1, frameRange[0]:frameRange[2]+1] = self.kernel
#                 self.filterMat = self.filterMat+0.01 ## still leave some chance for frames outside range to be chosen
#                 self.filterMat = self.filterMat/np.max(self.filterMat)
#                 self.filterMat = (1.5-self.filterMat)#*10.0 +1.0
#     #             print kernel
#                 figure(); imshow(self.filterMat, interpolation='nearest'); draw()
# #                 self.probabilities = self.probabilities*self.filterMat
                
# #                 ## renormalize probabilities
# #                 normTerm = np.sum(self.probabilities, axis=1)
# #                 normTerm = cv2.repeat(normTerm, 1, self.probabilities.shape[1])
# #                 self.probabilities = self.probabilities / normTerm

#             figure(); imshow(distanceMatrixFilt, interpolation='nearest'); draw()
#             self.distMat = self.estimateFutureCost(0.999, 2.0, distanceMatrixFilt, self.filterMat)
#             figure(); imshow(self.distMat, interpolation='nearest'); draw()
# #                 self.cumProb = np.cumsum(self.probabilities, axis=1)
            
# #             distMat = self.distMat#+((1.0-filterMat)*1000) ## this needs to be filtered based on frame range and feather
# # #             figure(); imshow(distMat, interpolation='nearest')            
            
#             self.computeRangedProbs(self.distMat)
#             figure(); imshow(self.probabilities, interpolation='nearest'); plt.draw()
#             figure(); imshow(self.cumProb, interpolation='nearest'); plt.draw()
#             startFrame = frameRange[1]
#             ## check if frames have been computed before, if not compute them, otherwise only recompute if checkbox is checked
#             try:
#                 self.finalFrames
#             except AttributeError:
#                 self.finalFrames = vtu.getFinalFrames(self.cumProb, self.textureLength, self.numFilterFrames+1, startFrame, self.loopTexture)
#             else:
#                 if self.mainWidgetGroup.recomputeFramesCheckBox.isChecked() :
#                     self.finalFrames = vtu.getFinalFrames(self.cumProb, 100, self.numFilterFrames+1, startFrame, self.loopTexture)
#             self.finalMovie, self.finalJumps = vtu.renderFinalFrames(self.movie, self.finalFrames, self.numInterpolationFrames)
#             self.visualizeTexture = True


    def createControls(self, title):
        self.controlsGroup = QtGui.QGroupBox(title)
        self.controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -10px;}")
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        self.infoLabel = QtGui.QLabel("No video loaded")
        self.infoLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.infoLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        frameLabel = QtGui.QLabel("Current frame:")
        renderFpsLabel = QtGui.QLabel("Texture render fps:")
        loopTextureLabel = QtGui.QLabel("Loop texture:")
        textureLengthLabel = QtGui.QLabel("Video texture length:")
        frameRangeLabel = QtGui.QLabel("Frame range:")
        featherLevelLabel = QtGui.QLabel("Feather level:")
        numInterpolationFramesLabel = QtGui.QLabel("# Frames to interpolate:")
        
        
        self.loadingLabel = QtGui.QLabel("Loading... Please wait!")
        self.loadingLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.loadingLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
        self.loadingLabel.setVisible(False)
        movie = QtGui.QMovie("loader.gif")
        self.loadingSpinner = QtGui.QLabel()
        self.loadingSpinner.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.MinimumExpanding)
        self.loadingSpinner.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.loadingSpinner.setMovie(movie)
        movie.start()
        self.loadingSpinner.setVisible(False)
        
        self.openVideoButton = QtGui.QPushButton("Open &Video")
        self.openSequenceButton = QtGui.QPushButton("Open &Sequence")

        self.frameSpinBox = QtGui.QSpinBox()
        self.frameSpinBox.setRange(0, 0)
        self.frameSpinBox.setSingleStep(1)
        
        self.textureLengthSpinBox = QtGui.QSpinBox()
        self.textureLengthSpinBox.setRange(30, 1000)
        self.textureLengthSpinBox.setSingleStep(1)
        self.textureLengthSpinBox.setValue(self.textureLength)
        
        self.loopTextureCheckBox = QtGui.QCheckBox()
        self.loopTextureCheckBox.setChecked(self.loopTexture)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 30)
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
        
        
        self.openVideoButton.clicked.connect(self.openVideo)
        self.openSequenceButton.clicked.connect(self.openSequence)
        
        self.mainWidgetGroup.slider.valueChanged[int].connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.mainWidgetGroup.slider.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.changeFrame)
        self.textureLengthSpinBox.valueChanged[int].connect(self.setTextureLength)
        self.loopTextureCheckBox.stateChanged[int].connect(self.setLoopTexture)
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        
        self.frameRangeSpinBox.valueChanged[int].connect(self.setFrameRange)
        self.featherLevelSpinBox.valueChanged[float].connect(self.setFeatherLevel)
        self.numInterpolationFramesSpinBox.valueChanged[int].connect(self.setNumInterpolationFrames)
        
        self.mainWidgetGroup.renderTextureButton.clicked.connect(self.renderTexture)
        
        
        mainBox = QtGui.QGroupBox("Main Controls")
        mainBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -10px;}")
        mainBoxLayout = QtGui.QGridLayout()
        mainBoxLayout.addWidget(self.loadingSpinner, 0, 0, 1, 2)
        mainBoxLayout.addWidget(self.loadingLabel, 1, 0, 1, 2)
        mainBoxLayout.addWidget(self.infoLabel, 1, 0, 1, 2)
        mainBoxLayout.addWidget(self.openVideoButton, 2, 0)
        mainBoxLayout.addWidget(self.openSequenceButton, 2, 1)
        mainBox.setLayout(mainBoxLayout)
        
        renderingControls = QtGui.QGroupBox("Texture Rendering Controls")
        renderingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -10px;}")
        renderingControlsLayout = QtGui.QGridLayout()
        renderingControlsLayout.addWidget(frameLabel, 0, 0)
        renderingControlsLayout.addWidget(self.frameSpinBox, 0, 1)
        renderingControlsLayout.addWidget(textureLengthLabel, 1, 0)
        renderingControlsLayout.addWidget(self.textureLengthSpinBox, 1, 1)
        renderingControlsLayout.addWidget(loopTextureLabel, 2, 0)
        renderingControlsLayout.addWidget(self.loopTextureCheckBox, 2, 1)
        renderingControlsLayout.addWidget(renderFpsLabel, 3, 0)
        renderingControlsLayout.addWidget(self.renderFpsSpinBox, 3, 1)
        renderingControls.setLayout(renderingControlsLayout)
        
        processingControls = QtGui.QGroupBox("Processing Controls")
        processingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -10px;}")
        processingControlsLayout = QtGui.QGridLayout()
        processingControlsLayout.addWidget(frameRangeLabel, 0, 0)
        processingControlsLayout.addWidget(self.frameRangeSpinBox, 0, 1)
        processingControlsLayout.addWidget(featherLevelLabel, 1, 0)
        processingControlsLayout.addWidget(self.featherLevelSpinBox, 1, 1)
        processingControlsLayout.addWidget(numInterpolationFramesLabel, 2, 0)
        processingControlsLayout.addWidget(self.numInterpolationFramesSpinBox, 2, 1)
        processingControls.setLayout(processingControlsLayout)
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(mainBox)
        controlsLayout.addWidget(renderingControls)
        controlsLayout.addWidget(processingControls)
        self.controlsGroup.setLayout(controlsLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

img = np.array(cv2.cvtColor(cv2.imread("../data/ribbon1_matte/frame-00001.png"), cv2.COLOR_BGR2RGB))
figure(); imshow(img)

# <codecell>

def downsample(a, shape):
    print shape
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

figure(); imshow(downsample(img, np.hstack((np.array(img.shape[0:2])/2, img.shape[-1]))))

# <codecell>

from PIL import Image
im = Image.open("../data/ribbon1_matte/frame-00001.png")
figure(); imshow(np.array(im))
figure(); imshow(np.array(im.resize(np.array(im.size)/2)))

# <codecell>

np.array(im.resize(np.array(im.size)/2)).shape

# <codecell>

tmp = []
tmp.append(0)
tmp.append(0)
tmp.append(0)
tmp.append(0)
print tmp
print np.concatenate((tmp, list(np.arange(0, 5)))).shape

# <codecell>

frameRange = window.frameRange-(window.numFilterFrames+1)
featherLevel = window.featherLevel
print frameRange, featherLevel
l = 0.5
weightShape = np.array(window.distMat.shape)+1
kernelLength = np.float(frameRange[1]-frameRange[0])
interval = (1-l)/(np.floor(kernelLength*featherLevel))
print interval
# kernelRange = interval*np.arange(np.floor(kernelLength*featherLevel), 0.0, -1.0)
kernelRange = np.arange(1.0, l, -(1.0-l)/np.floor(kernelLength*featherLevel))
print kernelRange
kernel = np.ones(kernelLength+1)*l
kernel[0:len(kernelRange)] = kernelRange#+np.ones(kernelRange.shape)
kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
kernel = np.reshape(kernel, [len(kernel), 1])

print kernel
print window.frameRange
print frameRange

# interval = l/(np.floor(kernelLength*featherLevel))
# print interval
# kernelRange = np.arange(1.0+l, 1.0, -interval)
# print kernelRange
# kernel = np.hstack((kernelRange, np.ones(np.floor(kernelLength-kernelLength*featherLevel-1))))
# kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
# kernel = np.reshape(kernel, [len(kernel), 1])

# print kernel

kernel = kernel+kernel.T
kernel = kernel/np.min(kernel)
print kernel.shape

rangeWeights = np.ones(weightShape)*np.max(kernel)
print rangeWeights[frameRange[0]:frameRange[2]+1, frameRange[0]:frameRange[2]+1].shape
rangeWeights[frameRange[0]:frameRange[2]+1, frameRange[0]:frameRange[2]+1] = kernel
figure(); imshow(kernel, interpolation='nearest')

# <codecell>

# print window.kernel.shape
print window.frameRange
print window.rangeWeights.shape
print window.rangeWeights[window.frameRange[0]:window.frameRange[2]+1, window.frameRange[0]:window.frameRange[2]+1].shape

# <codecell>

print window.frameRange
print np.max(window.rangeWeights)

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(window.filterMat, cmap=cm.jet, interpolation='nearest')

numrows, numcols = window.filterMat.shape
def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = window.filterMat[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

ax.format_coord = format_coord
show()
print np.min(window.filterMat), np.max(window.filterMat)

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(window.rangeWeights, cmap=cm.jet, interpolation='nearest')

numrows, numcols = window.rangeWeights.shape
def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = window.rangeWeights[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

ax.format_coord = format_coord
show()
print np.min(window.rangeWeights), np.max(window.rangeWeights)

# <codecell>

featherLevel = window.featherLevel
frameRange = window.frameRange
frameRange
print featherLevel, frameRange
distMat = window.distMat

# <codecell>

filterMat = np.zeros(distMat.shape)
filterMat[frameRange[0]:frameRange[2]+1, frameRange[0]:frameRange[2]+1] = finalKernel
print np.max(distMat)

# <codecell>

kernelLength = np.float(frameRange[1]-frameRange[0])
interval = 1.0/(np.floor(kernelLength*featherLevel)+1.0)
kernelRange = np.arange(interval, 1.0+interval, interval)
kernel = np.hstack((kernelRange, np.ones(np.ceil(kernelLength-kernelLength*featherLevel))))
kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))
kernel = np.reshape(kernel, [len(kernel), 1])
print kernel, kernel.shape

# <codecell>

filterMat[frameRange[0]:frameRange[2]+1, frameRange[0]:frameRange[2]+1] = finalKernel

# <codecell>

finalKernel = kernel+kernel.T
finalKernel = finalKernel/np.max(finalKernel)
figure(); imshow(finalKernel, cmap=cm.Greys_r, interpolation='nearest')
figure(); imshow(1.0-finalKernel, cmap=cm.Greys_r, interpolation='nearest')

# <codecell>

figure(); imshow(window.cumProb, interpolation='nearest')

# <codecell>

print window.cumProb[18, :]

# <codecell>

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

finalMovie = window.finalMovie

## visualize frames automatically

def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw

fig=plt.figure()
img = plt.imshow(np.array(finalMovie[0], dtype=uint8))
# img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
    img.set_data(np.array(finalMovie[0], dtype=uint8))
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f) + ' taken from ')
    img.set_data(np.array(finalMovie[f], dtype=uint8))
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalMovie),interval=33,blit=True)

# writer = animation.writers['ffmpeg'](fps=30)
# ani.save('demoa.mp4', writer=writer,dpi=160, bitrate=100)

plt.show()

# <codecell>

tmp = np.ones((2, 2, 2, 1))
tmp[0, :, :, :] = np.reshape([[0, 1], [2, 3]], [2, 2, 1])
tmp[1, :, :, :] = np.reshape([[4, 5], [6, 7]], [2, 2, 1])

# <codecell>

print tmp[0, :, :, :]
print tmp[1, :, :, :]

# <codecell>

tmp2 = np.reshape(tmp, (2, 2, 1, 2))
tmp2 = np.swapaxes(tmp, 0, 3)

# <codecell>

print tmp2[:, :, :, 0]
print tmp2[:, :, :, 1]

# <codecell>

requiredSpace = 19
downsampleFactor = 1
if requiredSpace > 8 :
    print np.argmin(np.abs(8-requiredSpace/np.arange(2.0, 6.0))), np.abs(8-(requiredSpace/np.arange(2.0, 6.0)))

# <codecell>

frameNames = glob.glob("../data/ribbon2_matted/frame*.png")
frameNames = np.sort(frameNames)
if len(frameNames) > 0 :
    frameSize = cv2.imread(frameNames[0]).shape
    print frameSize, len(frameNames)
    print 2.0*(np.prod(frameSize)*len(frameNames))/(1024**3), "GB"
#     movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], len(frameNames)]), dtype=uint8)
#     frames = []
#     for i in range(0, len(frameNames)) :
#         im = Image.open(frameNames[i])#.resize(np.array(frameSize[0:2])/3)
# #         im = np.array(im.resize(np.array(im.size)/3))
#         movie[:, :, :, i] = np.array(im, dtype=uint8)[:, :, 0:-1]#np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
#         frames.append(im)
#         sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(frameNames)))
#         sys.stdout.flush()

