# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import sys
import numpy as np
import scipy as sp
import cv2
import re
import os.path
from PIL import Image
from PySide import QtCore, QtGui

import VideoTexturesUtils as vtu

dataFolder = "../data/"

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
#             self.setPixmap(self.pixmap().scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio))
                self.setPixmap(self.pixmap().scaledToWidth(self.width()))
    
class MainWidgetGroup(QtGui.QGroupBox):

    valueChanged = QtCore.Signal(int)

    def __init__(self, orientation, title, parent=None):
        super(MainWidgetGroup, self).__init__(title, parent)
        
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
        
#         self.imageLabel.setScaledContents(True)
        labelLayout = QtGui.QHBoxLayout()
        labelLayout.addStretch()
        labelLayout.addLayout(framesLayout)
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
        
        self.numFilterFrames = 4
        self.numInterpolationFrames = 4
        self.textureLength = 100
        self.loopTexture = True
        self.availableFrames = 0

        self.createControls("Controls")

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.controlsGroup)
        layout.addWidget(self.mainWidgetGroup)
        self.setLayout(layout)
        
        self.frameSpinBox.setValue(0)

        self.setWindowTitle("Video Textures GUI")
        self.resize(1920, 750)
        
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
        
        print QtGui.QApplication.desktop().screen().rect()
        self.mainWidgetGroup.imageLabel.setFixedWidth(1920/3)
        self.mainWidgetGroup.textureLabel.setFixedWidth(1920/3)
        
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
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Open Video File",
                QtCore.QDir.currentPath()+"/"+dataFolder)
        if fileName :
            self.outputData = fileName
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
            
            self.availableFrames = self.movie.shape[-1]-(self.numFilterFrames+1)*2
            
            if self.frameSpinBox.value() < self.availableFrames :
                self.mainWidgetGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, self.frameSpinBox.value()+(self.numFilterFrames+1)]))
            else :
                self.mainWidgetGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, 0+(self.numFilterFrames+1)]))
                self.frameSpinBox.setValue(0)
                
            self.frameSpinBox.setRange(0, self.availableFrames-1)
            self.mainWidgetGroup.setMaximum(self.availableFrames-1)
            self.setFrameRange(self.frameRangeSpinBox.value())
            self.frameRangeSpinBox.setRange(1, self.availableFrames*0.75)
            infoText = "Loaded video:\n\t"+fileName+"\nNumber of Usable Frames:\n\t"+np.string_(self.availableFrames)
            infoText += "\nFrame size:\n\t"+np.string_(self.movie.shape[1])+"x"+np.string_(self.movie.shape[0])
            self.infoLabel.setText(infoText)
            
            self.loadingLabel.setText("Preprocessing movie sequence...")
            QtCore.QCoreApplication.processEvents()
            self.preProcessMovie(self.movie, self.outputData)
            
            self.lockGUI(False)
            self.showLoading(False)
    
    def openSequence(self):
        self.frameNames = np.array(QtGui.QFileDialog.getOpenFileNames(self, "Open Sequence", 
                    QtCore.QDir.currentPath()+"/"+dataFolder, "Images(*.png)")[0])
        if len(self.frameNames) > 1 :
#             self.outputData = '\\'.join(filter(None, self.frameNames[0].split('\\'))[0:-1])
            self.outputData = "/"+ '/'.join(filter(None, self.frameNames[0].split('/'))[0:-1])
            self.lockGUI(True)
            self.showLoading(True)

            self.loadingLabel.setText("Loading set of frames...")
            self.loadingText = self.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()

            ## sort frame names
            self.frameNames = np.sort(self.frameNames)
            if len(self.frameNames) > 0 :
                frameSize = cv2.imread(self.frameNames[0]).shape
                requiredSpace = 2.0*(np.prod(frameSize)*float(len(self.frameNames)))/(1024**3) ## GB
                downsampleFactor = 1
                maxSpace = 2
                if requiredSpace > maxSpace :
                    downsampleFactor = 2 + np.argmin(np.abs(maxSpace-requiredSpace/np.arange(2.0, 6.0)))
                    print "Downsample Factor of", downsampleFactor
                    
                if downsampleFactor > 1 :
                    self.movie = np.zeros(np.hstack([frameSize[0]/downsampleFactor, frameSize[1]/downsampleFactor, frameSize[2], len(self.frameNames)]), dtype=np.uint8)
                else :
                    self.movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], len(self.frameNames)]), dtype=np.uint8)
                
                for i in range(0, len(self.frameNames)) :
                    im = Image.open(self.frameNames[i])
                    if downsampleFactor > 1 :
                        im = im.resize(np.array([frameSize[1], frameSize[0]])/downsampleFactor)
#                     im = np.ascontiguousarray(np.array(im, dtype=uint8)[:, :, 0:3])
                    self.movie[:, :, :, i] = np.array(im, dtype=np.uint8)[:, :, 0:3]#im#np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
                    sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(self.frameNames)))
                    sys.stdout.flush()
                    self.loadingLabel.setText(self.loadingText+"\nframe " + np.string_(i) + " of " + np.string_(len(self.frameNames)))
                    QtCore.QCoreApplication.processEvents() 
                print
                
            self.availableFrames = self.movie.shape[-1]-(self.numFilterFrames+1)*2
            
            if self.frameSpinBox.value() < self.availableFrames :
                self.mainWidgetGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, self.frameSpinBox.value()+(self.numFilterFrames+1)]))
            else :
                self.mainWidgetGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, 0+(self.numFilterFrames+1)]))
                self.frameSpinBox.setValue(0)
                
            self.frameSpinBox.setRange(0, self.availableFrames-1)
            self.mainWidgetGroup.setMaximum(self.availableFrames-1)
            self.setFrameRange(self.frameRangeSpinBox.value())
            self.frameRangeSpinBox.setRange(1, self.availableFrames*0.75)
            infoText = "Loaded sequence:\n\t"+self.outputData+"\nNumber of Usable Frames:\n\t"+np.string_(self.availableFrames)
            infoText += "\nFrame size:\n\t"+np.string_(self.movie.shape[1])+"x"+np.string_(self.movie.shape[0])
            self.infoLabel.setText(infoText)
            
            self.loadingLabel.setText("Preprocessing movie sequence...")
            QtCore.QCoreApplication.processEvents()
            self.preProcessMovie(self.movie, self.outputData)
            
            self.lockGUI(False)
            self.showLoading(False)
    
    def saveVideoTexture(self):
        
        try :
            self.finalFrames
            self.finalMovie
            self.frameNames
        except AttributeError:
            QtGui.QMessageBox.critical(self, "No video-texture", "There is no available video-texture.\nAborting...")
            return
        
        destFolder = QtGui.QFileDialog.getExistingDirectory(self, "Save Video Texture",
                QtCore.QDir.currentPath()+"/"+dataFolder)
        if destFolder :
            self.lockGUI(True)
            self.showLoading(True)
            
            self.loadingLabel.setText("Saving video texture...")
            self.loadingText = self.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()
            
            # baseFolder = '\\'.join(filter(None, self.frameNames[0].split('\\'))[0:-1])
            baseFolder = "/"+'/'.join(filter(None, self.frameNames[0].split('/'))[0:-1])
            #useMatte = os.path.isfile(baseFolder + "\\matte-" + self.frameNames[0].split('\\')[-1])
            useMatte = os.path.isfile(baseFolder + "/matte-" + self.frameNames[0].split('/')[-1])
            
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
                self.loadingLabel.setText(self.loadingText+"\nframe " + np.string_(i) + " of " + np.string_(len(self.finalMovie)))
                QtCore.QCoreApplication.processEvents() 
                
                if useMatte :
                    ## HACK:: assume mattes are available and are good(i.e. finalMovie has no interpolated frames)
                    ##mattename = baseFolder + "\\matte-" + self.frameNames[self.finalFrames[i]].split('\\')[-1]
                    mattename = baseFolder + "/matte-" + self.frameNames[self.finalFrames[i]].split('/')[-1]
                    matte = Image.open(mattename)
                    matte = np.reshape(np.array(matte, dtype=np.uint8)[:, :, 0], np.hstack((matte.size[1], matte.size[0], 1)))
                    
                    fullResoFrame = Image.open(self.frameNames[self.finalFrames[i]])
                    fullResoFrame = np.array(fullResoFrame, dtype=np.uint8)[:, :, 0:self.finalMovie[0].shape[-1]]
                    
                    Image.fromstring("RGBA", (fullResoFrame.shape[1], fullResoFrame.shape[0]), np.concatenate((fullResoFrame, matte), axis=-1).tostring()).save(destFolder+"/vt."+zeroes+np.string_(i+1)+".png")
                else :
                    Image.fromstring("RGB", (self.finalMovie[0].shape[1], self.finalMovie[0].shape[0]), self.finalMovie[i].tostring()).save(destFolder+"/vt."+zeroes+np.string_(i+1)+".png")
                
            print
            
            self.lockGUI(False)
            self.showLoading(False)
        
    def setFrameRange(self, value) :
        currentFrame = self.mainWidgetGroup.slider.value()
        if currentFrame-(value-1)/2 >= 0 and currentFrame+(value-1)/2 < self.availableFrames :
            self.frameRange = np.array([currentFrame-((value-1)/2), currentFrame, currentFrame+((value-1)/2)])
            self.mainWidgetGroup.setFrameInterval(self.frameRange, self.featherLevel)

            if self.frameRange[0] >= 0 and self.frameRange[0] < self.availableFrames :
                self.mainWidgetGroup.setFrameLow(np.ascontiguousarray(self.movie[:, :, :, self.frameRange[0]+(self.numFilterFrames+1)]))
            if self.frameRange[-1] >= 0 and self.frameRange[-1] < self.availableFrames :
                self.mainWidgetGroup.setFrameHigh(np.ascontiguousarray(self.movie[:, :, :, self.frameRange[-1]+(self.numFilterFrames+1)]))
        else :
            self.frameRangeSpinBox.setValue((self.frameRange[1]-self.frameRange[0])*2+1)
        
#         self.updateTexture()
        
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
        if idx >= 0 and idx < self.availableFrames:
            if idx >= (self.frameRangeSpinBox.value()-1)/2 and idx < self.availableFrames-(self.frameRangeSpinBox.value()-1)/2 :
                self.mainWidgetGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, idx+(self.numFilterFrames+1)]))
                self.setFrameRange(self.frameRangeSpinBox.value())
            else :
                self.mainWidgetGroup.setFrame(np.ascontiguousarray(self.movie[:, :, :, self.frameRange[1]+(self.numFilterFrames+1)]))
                self.mainWidgetGroup.slider.setValue(self.frameRange[1])
                self.frameSpinBox.setValue(self.frameRange[1])
            self.updateTexture()
    
    def lockGUI(self, lock):
        if lock :
            self.bfVisText = self.visualizeTexture
            self.visualizeTexture = False
        else :
            self.visualizeTexture = self.bfVisText
        self.mainWidgetGroup.slider.setEnabled(not lock)
        self.mainWidgetGroup.renderTextureButton.setEnabled(not lock)
        self.mainWidgetGroup.recomputeFramesCheckBox.setEnabled(not lock)
        
        self.openVideoButton.setEnabled(not lock)
        self.openSequenceButton.setEnabled(not lock)
        self.saveVTButton.setEnabled(not lock)
        
        self.frameSpinBox.setEnabled(not lock)
        self.textureLengthSpinBox.setEnabled(not lock)
        self.loopTextureCheckBox.setEnabled(not lock)
        self.renderFpsSpinBox.setEnabled(not lock)
        
        self.frameRangeSpinBox.setEnabled(not lock)
        self.featherLevelSpinBox.setEnabled(not lock)
        self.numInterpolationFramesSpinBox.setEnabled(not lock)
    
    def preProcessMovie(self, movie, outputData):
        distanceMatrix = vtu.computeDistanceMatrix(movie, outputData + "/distMat.npy")
        self.distanceMatrixFilt = vtu.filterDistanceMatrix(distanceMatrix, self.numFilterFrames, False)
        self.distMat = self.estimateFutureCost(0.999, 2.0, self.distanceMatrixFilt, np.ones(self.distanceMatrixFilt.shape))
        
    def computeRangedProbs(self, distMat, rangeDistances):
        self.probabilities, self.cumProb = vtu.getProbabilities(distMat, 0.005, rangeDistances, True)

    ## Avoid dead ends: estimate future costs
    def estimateFutureCost(self, alpha, p, distanceMatrixFilt, weights) :
        
        distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
        distMat = distMatFilt ** p
        
        last = np.copy(distMat)
        current = np.zeros(distMat.shape)
        
        self.loadingLabel.setText(self.loadingText+"\nOptimizing transitions...")
        self.loadingText = self.loadingLabel.text()
        QtCore.QCoreApplication.processEvents()
        
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
            self.loadingLabel.setText(self.loadingText+"\niteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
            QtCore.QCoreApplication.processEvents()
            
            iterations += 1
        
        print
        print 'finished in', iterations, 'iterations'
        
        self.loadingLabel.setText(self.loadingText+"\nTransitions optimized in " + np.string_(iterations)+ " iterations")
        self.loadingText = self.loadingLabel.text()
        QtCore.QCoreApplication.processEvents()
        
        return distMat
    
    def getRangeWeights(self, l, frameRange, featherLevel, weightShape) :

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
        return rangeWeights
    
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
    
    def updateTexture(self):
        print "entering"
        try:
            self.distMat
        except AttributeError:
            print "tried updating rendering before future cost estimation"
        else:
            
            self.lockGUI(True)
            
            if self.frameRangeSpinBox.value() > 1 :
                frameRange = self.frameRange
                rangeDistances = self.getRangeDistances(0.7, frameRange, self.featherLevel, self.distMat.shape)
                self.computeRangedProbs(self.distMat, rangeDistances)
#                 print frameRange
#                 print "first frame comparison range", startFrame, frameRange[0]+np.argmin(np.round(np.sum(self.cumProb[:, frameRange[0]:frameRange[2]+1] < 0.5, axis=0)))
                startFrame = frameRange[0]+np.argmin(np.round(np.sum(self.cumProb[:, frameRange[0]:frameRange[2]+1] < 0.5, axis=0)))
            else :
                self.computeRangedProbs(self.distMat, None)
#                 print "first frame comparison", startFrame, np.argmin(np.round(np.sum(self.cumProb < 0.5, axis=0)))
                startFrame = np.argmin(np.round(np.sum(self.cumProb < 0.5, axis=0)))
    
            print "starting frame=", startFrame
        
            try:
                self.finalFrames
            except AttributeError: 
                self.loadingText = self.loadingLabel.text()
                self.loadingLabel.setText(self.loadingText+"\nFinding best frames...")
                self.loadingText = self.loadingLabel.text()
                QtCore.QCoreApplication.processEvents()
                self.finalFrames = vtu.getFinalFrames(self.cumProb, self.textureLength, self.numFilterFrames+1, startFrame, self.loopTexture, False)
            else:
                if self.mainWidgetGroup.recomputeFramesCheckBox.isChecked() :
                    self.loadingLabel.setText(self.loadingText+"\nFinding best frames...")
                    self.loadingText = self.loadingLabel.text()
                    QtCore.QCoreApplication.processEvents()
                    self.finalFrames = vtu.getFinalFrames(self.cumProb, 100, self.numFilterFrames+1, startFrame, self.loopTexture, False)
            
            print "total final frames =", len(self.finalFrames)
            
            self.finalMovie, self.finalJumps = vtu.renderFinalFrames(self.movie, self.finalFrames, self.numInterpolationFrames)
            
            self.lockGUI(False)
            self.visualizeTexture = True
     
    def renderTexture(self):
        print "balala"
        try:
            self.distanceMatrixFilt
        except AttributeError:
            print "tried rendering texture before computing distance matrix"
        else:
#             print "rendering texture"
            self.lockGUI(True)
            self.showLoading(True)
        
            self.loadingLabel.setText("Rendering video texture...")
            self.loadingText = self.loadingLabel.text()
            self.loadingLabel.setText(self.loadingText+"\nProcessing user input...")
            self.loadingText = self.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()
            
            if self.frameRangeSpinBox.value() > 1 :
                frameRange = self.frameRange#-(self.numFilterFrames+1)
#                 print frameRange
                self.rangeWeights = self.getRangeWeights(0.7, frameRange, self.featherLevel, self.distanceMatrixFilt.shape)
        
                self.distMat = self.estimateFutureCost(0.999, 2.0, self.distanceMatrixFilt, self.rangeWeights)
#                 startFrame = frameRange[1]
            else :
                self.distMat = self.estimateFutureCost(0.999, 2.0, self.distanceMatrixFilt, np.ones(self.distanceMatrixFilt.shape))
#                 startFrame = self.frameSpinBox.value()+self.numFilterFrames+1
                    
#             figure(); imshow(self.distMat, interpolation='nearest'); draw()
            
            self.loadingLabel.setText(self.loadingText+"\nComputing probabilities...")
            self.loadingText = self.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()
            
            self.computeRangedProbs(self.distMat, None)
#             figure(); imshow(self.cumProb, interpolation='nearest'); plt.draw()
            
            if self.frameRangeSpinBox.value() > 1 :
#                 print frameRange
#                 print "first frame comparison range", startFrame, frameRange[0]+np.argmin(np.round(np.sum(self.cumProb[:, frameRange[0]:frameRange[2]+1] < 0.5, axis=0)))
                startFrame = frameRange[0]+np.argmin(np.round(np.sum(self.cumProb[:, frameRange[0]:frameRange[2]+1] < 0.5, axis=0)))
            else :
#                 print "first frame comparison", startFrame, np.argmin(np.round(np.sum(self.cumProb < 0.5, axis=0)))
                startFrame = np.argmin(np.round(np.sum(self.cumProb < 0.5, axis=0)))
            
            ## check if frames have been computed before, if not compute them, otherwise only recompute if checkbox is checked
            
            try:
                self.finalFrames
            except AttributeError: 
                self.loadingLabel.setText(self.loadingText+"\nFinding best frames...")
                self.loadingText = self.loadingLabel.text()
                QtCore.QCoreApplication.processEvents()
                self.finalFrames = vtu.getFinalFrames(self.cumProb, self.textureLength, self.numFilterFrames+1, startFrame, self.loopTexture, False)
            else:
                if self.mainWidgetGroup.recomputeFramesCheckBox.isChecked() :
                    self.loadingLabel.setText(self.loadingText+"\nFinding best frames...")
                    self.loadingText = self.loadingLabel.text()
                    QtCore.QCoreApplication.processEvents()
                    self.finalFrames = vtu.getFinalFrames(self.cumProb, 100, self.numFilterFrames+1, startFrame, self.loopTexture, False)
            
            print "total final frames =", len(self.finalFrames)
            self.loadingLabel.setText(self.loadingText+"\nProducing final render...")
            self.loadingText = self.loadingLabel.text()
            QtCore.QCoreApplication.processEvents()
            
            self.finalMovie, self.finalJumps = vtu.renderFinalFrames(self.movie, self.finalFrames, self.numInterpolationFrames)
            self.visualizeTexture = True
            
            self.lockGUI(False)
            self.showLoading(False)

    def createControls(self, title):
        self.controlsGroup = QtGui.QGroupBox(title)
        self.controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        self.infoLabel = QtGui.QLabel("No data loaded")
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
        self.openVideoButton.setVisible(False)
        self.openSequenceButton = QtGui.QPushButton("Open &Sequence")
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
        
        
        self.openVideoButton.clicked.connect(self.openVideo)
        self.openSequenceButton.clicked.connect(self.openSequence)
        self.saveVTButton.clicked.connect(self.saveVideoTexture)
        
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
        mainBox.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        mainBoxLayout = QtGui.QGridLayout()
        mainBoxLayout.addWidget(self.loadingSpinner, 0, 0, 1, 2)
        mainBoxLayout.addWidget(self.loadingLabel, 1, 0, 1, 2)
        mainBoxLayout.addWidget(self.infoLabel, 1, 0, 1, 2)
        mainBoxLayout.addWidget(self.openVideoButton, 2, 0)
        mainBoxLayout.addWidget(self.openSequenceButton, 2, 0, 1, 2)
        mainBoxLayout.addWidget(self.saveVTButton, 3, 0, 1, 2)
        mainBox.setLayout(mainBoxLayout)
        
        renderingControls = QtGui.QGroupBox("Texture Rendering Controls")
        renderingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
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
        processingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
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

