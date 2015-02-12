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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.sparsetools import _graph_tools
from sklearn.utils.sparsetools import _graph_validation
from sklearn.utils import lgamma

import VideoTexturesUtils as vtu
import ComputeDistanceMatrix as cdm

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

class ScribbleArea(QtGui.QLabel):
    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__("Scribble Area", parent)
        
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.canScribble = False
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 1
        self.myPenColor = QtGui.QColor.fromRgb(0, 0, 255)
        self.scribble = QtGui.QImage(QtCore.QSize(200, 200), QtGui.QImage.Format_RGB32)
        self.lastPoint = QtCore.QPoint()
        self.scribbleAlpha = 0.5
        
        self.clearScribble()
 
    def openScribble(self, fileName):
        loadedScribble = QtGui.QImage()
        if not loadedScribble.load(fileName):
            return False
 
        newSize = loadedScribble.size().expandedTo(size())
        self.resizeScribble(loadedScribble, newSize)
        self.scribble = loadedScribble
        self.modified = False
        self.update()
        return True
 
    def saveScribble(self, fileName, fileFormat):
        visibleScribble = self.scribble
        self.resizeScribble(visibleScribble, size())
 
        if visibleScribble.save(fileName, fileFormat):
            self.modified = False
            return True
        else:
            return False
        
    def setScribble(self, scribble) :
        painter = QtGui.QPainter(self.scribble)
        painter.drawImage(QtCore.QPoint(0, 0), scribble)
        self.modified = False
        self.update()
        
    def setCanScribble(self, canScribble):
        self.canScribble = canScribble
 
    def setPenColor(self, newColor):
        self.myPenColor = newColor
 
    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth
        
    def setScribbleAlphaInt(self, newAlpha) :
        self.setScribbleAlpha(float(newAlpha)/10.0)
        
    def setScribbleAlpha(self, newAlpha) :
        self.scribbleAlpha = newAlpha
        self.update()
 
    def setBgImage(self, bgImage):
        self.clearScribble()
        self.setPixmap(QtGui.QPixmap(bgImage))
        self.resizeScribble(self.scribble, bgImage.size())
        
    def clearScribble(self):
        self.scribble.fill(QtGui.qRgb(250, 250, 250))
        self.modified = True
        self.update()
 
    def mousePressEvent(self, event):
        if self.canScribble and event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True
 
    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())
 
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False

    def paintEvent(self, event):
        super(ScribbleArea, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setOpacity(self.scribbleAlpha)
        painter.drawImage(QtCore.QPoint(0, 0), self.scribble)
 
    def resizeEvent(self, event):
        if self.width() > self.scribble.width() or self.height() > self.scribble.height():
            newWidth = max(self.width() + 128, self.scribble.width())
            newHeight = max(self.height() + 128, self.scribble.height())
            self.resizeScribble(self.scribble, QtCore.QSize(newWidth, newHeight))
            self.update()
 
        super(ScribbleArea, self).resizeEvent(event)
 
    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.scribble)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth,
                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True
 
        rad = self.myPenWidth / 2 + 2
        self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QtCore.QPoint(endPoint)
 
    def resizeScribble(self, scribble, newSize):
        if scribble.size() == newSize:
            return
 
        newScribble = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
        newScribble.fill(QtGui.qRgb(250, 250, 250))
        painter = QtGui.QPainter(newScribble)
        painter.drawImage(QtCore.QPoint(0, 0), scribble)
        self.scribble = newScribble
        
    def isModified(self):
        return self.modified
 
    def penColor(self):
        return self.myPenColor
 
    def penWidth(self):
        return self.myPenWidth

# <codecell>

class MatteViewerGroup(QtGui.QGroupBox):

    def __init__(self, title, parent=None):
        super(MatteViewerGroup, self).__init__(title, parent)
        
        self.createGUI()
        
        
    def setScribble(self, scribble):
        im = np.ascontiguousarray(cv2.cvtColor(scribble, cv2.COLOR_RGB2BGR))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.scribbleArea.setScribble(qim)
        
    def setCurrentFrame(self, frame, alpha):
        self.scribbleArea.setFixedSize(frame.shape[1], frame.shape[0])
        self.imageLabel.setFixedSize(frame.shape[1], frame.shape[0])
        
        im = np.ascontiguousarray(frame)
        
        if alpha :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
        else :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            self.scribbleArea.setBgImage(qim)
            
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
    def getScribble(self):
        return self.scribbleArea.scribble

    def setValue(self, value):
        self.slider.setValue(value)

    def setMinimum(self, value):
        self.slider.setMinimum(value)
        self.frameSpinBox.setMinimum(value)

    def setMaximum(self, value):
        self.slider.setMaximum(value)
        self.frameSpinBox.setMaximum(value)
        
    def createGUI(self):

        ## WIDGETS ##
        self.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        self.scribbleArea = ScribbleArea()
        scribbleGroup = QtGui.QGroupBox()
        scribbleGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        scribbleLayout = QtGui.QVBoxLayout()
        scribbleLayout.addWidget(self.scribbleArea)
        scribbleGroup.setLayout(scribbleLayout)
        scribbleGroup.setMinimumSize(200, 200)
        
        
        self.imageLabel = ImageLabel("Matte Frame")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        imageGroup = QtGui.QGroupBox()
        imageGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        imageLayout = QtGui.QVBoxLayout()
        imageLayout.addWidget(self.imageLabel)
        imageGroup.setLayout(imageLayout)
        imageGroup.setMinimumSize(200, 200)
        
        
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
        self.frameSpinBox.valueChanged[int].connect(self.slider.setValue)
        
        
        ## LAYOUTS ##
        labelsLayout = QtGui.QHBoxLayout()
        labelsLayout.addStretch()
        labelsLayout.addWidget(scribbleGroup)
        labelsLayout.addStretch()
        labelsLayout.addWidget(imageGroup)
        labelsLayout.addStretch()
        
        navLayout = QtGui.QHBoxLayout()
        navLayout.addWidget(self.slider)
        navLayout.addWidget(self.frameSpinBox)
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addStretch()
        mainLayout.addLayout(labelsLayout)
        mainLayout.addStretch()
        mainLayout.addLayout(navLayout)
        
        
        self.setLayout(mainLayout)

# <codecell>

class PreProcessingTab(QtGui.QWidget):
    def __init__(self, mainWindow, parent=None):
        super(PreProcessingTab, self).__init__(parent)
        
        self.mainWindow = mainWindow
        
        self.setProcessingState("SCRIBBLE")
        self.borderSize = 5
        self.trimaps = {}
        
        self.numBlocks = 8
        self.useMattes = True
        self.numFilterFrames = 4
        
        self.createGUI()
        
        self.scribbleAlphaSlider.setValue(5)
        self.scribbleSizeSlider.setValue(10)
        self.backgroundToolButton.toggle()
    
        self.scribbleBorderSlider.setValue(self.borderSize)
        self.numBlocksComboBox.setCurrentIndex(3)
        self.useMattesCheckBox.setChecked(self.useMattes)
        
        self.installEventFilter(self);
        
        self.lockGUI(True)
        
    def setProcessingState(self, newState) :
        ## SCRIBBLE is defalt state
        if newState == "SCRIBBLE":
            self.state = 0
        ## TRIMAP is state after scribble has been confirmed and border size for trimap needs to be chosen
        elif newState == "TRIMAP" and self.state < 1:
            try :
                self.trimap
            except :
                return
            
            self.state = 1
        
    def processScribble(self):
        
        ## load currentFrame from disc as it may have been downloaded
        currentFrame = cv2.cvtColor(cv2.imread(self.frameNames[self.matteViewerGroup.frameSpinBox.value()]), cv2.COLOR_BGR2RGB)
        
        ## get scribble from scribbleArea
        img = self.matteViewerGroup.getScribble().constBits()
        scribble = np.array(img).reshape(self.matteViewerGroup.getScribble().height(), self.matteViewerGroup.getScribble().width(), 4)[:, :, 0:-1]
        
        ## perform the expansion at full res
        if self.downsampleFactor > 1:
            scribble = cv2.resize(scribble, (currentFrame.shape[1], currentFrame.shape[0]), interpolation = cv2.INTER_NEAREST)
        ## prepare scribble and expand using watershed
        expandedScribble = np.zeros(scribble.shape[0:-1], dtype=np.int32)
        ## images in qt are BGR ???
        ## foreground is red
        fgIdx = np.argwhere(scribble[:, :, 2] == 255)
        expandedScribble[fgIdx[:, 0], fgIdx[:, 1]] = 2
        ## background is blue
        bgIdx = np.argwhere(scribble[:, :, 0] == 255)
        expandedScribble[bgIdx[:, 0], bgIdx[:, 1]] = 1
        
        cv2.watershed(currentFrame, expandedScribble)
        
        ## get mask for foreground and close holes
        mask = np.zeros(expandedScribble.shape)
        mask[expandedScribble == 2] = 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1,1),np.uint8))
        
        ## get edges and dilate
        edges = cv2.Canny(np.array(mask, dtype=np.uint8), 1, 2)
        edges = cv2.dilate(edges, np.ones((self.borderSize, self.borderSize),np.uint8), iterations=1)
        
        ## use dilated edges and mask to build trimap
        self.trimap = np.zeros(mask.shape)
        self.trimap[mask == 1] = 2
        self.trimap[edges == np.max(edges)] = 1
        
        
        ## get rgb image for trimap and visualize in scribbling area
        self.matteViewerGroup.setScribble(self.toRgbTrimap(self.trimap))
        
        self.setProcessingState("TRIMAP")
        
    def trainClassifier(self, data_frames, trimaps) :
        # train on first frame
        # augment with x-y positional data
        clf = ExtraTreesClassifier()
        
#         print len(data_frames), len(trimaps)
        
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
        clf.fit(X, y)
        return clf
        
    def computeMattes(self):
        if len(self.trimaps) <= 0 :
            return
        
        self.mainWindow.lockGUI(True)
        self.mainWindow.showLoading(True)
        
        self.mainWindow.loadingLabel.setText("Matting the frames...")
        self.loadingText = self.mainWindow.loadingLabel.text()
        
        self.mainWindow.loadingLabel.setText(self.loadingText+"\nTraining classifier")
        QtCore.QCoreApplication.processEvents()
        
        data = []
        tmaps = []
        for frameName in self.trimaps.keys() :
            idx = np.ndarray.flatten(np.argwhere(self.frameNames == self.dataLocation + os.sep + frameName))[0]
            data.append(cv2.cvtColor(cv2.imread(self.frameNames[idx]), cv2.COLOR_BGR2RGB))
            tmaps.append(self.trimaps[frameName])
    
        classifier = self.trainClassifier(data, tmaps)
        
        ## computing mattes
        for i in xrange(0, len(self.frameNames)) :
            img = cv2.cvtColor(cv2.imread(self.frameNames[i]), cv2.COLOR_BGR2RGB)
            
            indices = np.indices(img.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
            data = np.concatenate((img, indices), axis=-1)
            
            probabilities = classifier.predict_proba(data.reshape((-1, 5)))
            
#             alphaMatte = np.copy(probabilities[:, 1])
#             alphaMatte[probabilities[:, 1]>0.15] = 1
#             alphaMatte[probabilities[:, 0]>0.85] = 0

            alphaMatte = np.copy(probabilities[:, 1])
            alphaMatte[probabilities[:, 1]>0.5] = 1
            alphaMatte[probabilities[:, 0]>0.5] = 0
            
            filtAlphaMatte = cv2.GaussianBlur(np.array(alphaMatte*255, dtype=np.float32), (5, 5), 2.5)
            
            filtAlphaMatte = np.array(filtAlphaMatte.reshape(img.shape[:2]), dtype=np.uint8)
            
            frameName = self.frameNames[i].split(os.sep)[-1]
            
            cv2.imwrite(self.dataLocation + os.sep + "matte-" + frameName, filtAlphaMatte)
    
            self.mainWindow.loadingLabel.setText(self.loadingText+"\nmattes created: " + np.string_(i) + " of " + np.string_(self.movie.shape[-1]))
            QtCore.QCoreApplication.processEvents() 
        
        self.showFrame(self.matteViewerGroup.frameSpinBox.value())
        
        self.mainWindow.lockGUI(False)
        self.mainWindow.showLoading(False)
        
    def computeEucDistMat(self, numBlocks, frameNames, matteNames) :
        blockSize = np.int(len(frameNames)/numBlocks)
        
        counter = 0.0
        totComputes = np.float(numBlocks*(numBlocks+1))/2.0
        
        distanceMatrix = np.zeros([len(frameNames), len(frameNames)])
        if len(frameNames) >= 0 :
            frameSize = np.array(cv2.cvtColor(cv2.imread(frameNames[0]), cv2.COLOR_BGR2RGB)).shape
            for i in xrange(0, numBlocks) :
                
                ##load row frames
                f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]), dtype=np.uint8)
                for f, idx in zip(xrange(i*blockSize, i*blockSize+blockSize), xrange(0, blockSize)) :
                    img = np.array(cv2.cvtColor(cv2.imread(frameNames[f]), cv2.COLOR_BGR2RGB))
                    if self.useMattes and len(matteNames) == len(frameNames) and os.path.isfile(matteNames[f]) :
                        alpha = np.array(cv2.cvtColor(cv2.imread(matteNames[f]), cv2.COLOR_BGR2GRAY))/255.0
                        #f1s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
                        f1s[:, :, :, idx] = img*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
                    else :
                        f1s[:, :, :, idx] = img#/255.0
            
                ##compute distance between every pair of row frames
                data1 = np.array(np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T, dtype=np.float32)/255.0
                #data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
                #print data1
                distanceMatrix[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = cdm.distEuc(data1)
                
                counter += 1
                
                self.mainWindow.loadingLabel.setText(self.loadingText+"\n" + np.string_(np.int(100*counter/totComputes)) + "% computing done")
                QtCore.QCoreApplication.processEvents()
                sys.stdout.write('\r' + "Row Frames " + np.string_(i*blockSize) + " to " + np.string_(i*blockSize+blockSize-1) + " - " + np.string_(100*counter/totComputes))
                sys.stdout.flush()
                print
                
                for j in xrange(i+1, numBlocks) :
                    
                    ##load column frames
                    f2s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]), dtype=np.uint8)
                    for f, idx in zip(xrange(j*blockSize, j*blockSize+blockSize), xrange(0, blockSize)) :
                        img = np.array(cv2.cvtColor(cv2.imread(frameNames[f]), cv2.COLOR_BGR2RGB))
                        if self.useMattes and len(matteNames) == len(frameNames) and os.path.isfile(matteNames[f]) :
                            alpha = np.array(cv2.cvtColor(cv2.imread(matteNames[f]), cv2.COLOR_BGR2GRAY))/255.0
                            #f2s[:, :, :, idx] = (img/255.0)*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
                            f2s[:, :, :, idx] = img*np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
                        else :
                            f2s[:, :, :, idx] = img#/255.0
                        
                    ##compute distance between every pair of row-column frames
                    data2 = np.array(np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T, dtype=np.float32)/255.0
                    distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize] = cdm.distEuc2(data1, data2)
                    distanceMatrix[j*blockSize:j*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize].T
                                    
                    counter += 1
                    
                    self.mainWindow.loadingLabel.setText(self.loadingText+"\n" + np.string_(np.int(100*counter/totComputes)) + "% computing done")
                    QtCore.QCoreApplication.processEvents()
                    sys.stdout.write('\r' + "Column Frames " + np.string_(j*blockSize) + " to " + np.string_(j*blockSize+blockSize-1) + " - " + np.string_(100*counter/totComputes))
                    sys.stdout.flush()
                    print
        
        ## seems that for the diagonal there is some numerical instability and sometimes the result is nan because when doing sqrt(diff), the diff turns out to be negative
        distanceMatrix[np.where(np.eye(distanceMatrix.shape[0])==1)] = 0
        return distanceMatrix    

    ## Avoid dead ends: estimate future costs
    def estimateFutureCost(self, alpha, p, distanceMatrixFilt, weights) :

        distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
        distMat = distMatFilt ** p
        
        last = np.copy(distMat)
        current = np.zeros(distMat.shape)
        
        self.mainWindow.loadingLabel.setText(self.loadingText+"\nOptimizing transitions...")
        self.loadingText = self.mainWindow.loadingLabel.text()
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
            self.mainWindow.loadingLabel.setText(self.loadingText+"\niteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
            QtCore.QCoreApplication.processEvents()
            
            iterations += 1
        
        print
        print 'finished in', iterations, 'iterations'
        
        self.mainWindow.loadingLabel.setText(self.loadingText+"\nTransitions optimized in " + np.string_(iterations)+ " iterations")
        self.loadingText = self.mainWindow.loadingLabel.text()
        QtCore.QCoreApplication.processEvents()
        
        return distMat
    
    def computeDistMat(self) :
        
        self.mainWindow.lockGUI(True)
        self.mainWindow.showLoading(True)
        
        self.mainWindow.loadingLabel.setText("Computing Distance Matrix...")
        self.loadingText = self.mainWindow.loadingLabel.text()
        
        self.mainWindow.loadingLabel.setText(self.loadingText+"\n0% computing done")
        QtCore.QCoreApplication.processEvents()
        
        matteNames = np.sort(glob.glob(self.dataLocation + os.sep + "matte-*.png"))
        distanceMatrix = self.computeEucDistMat(self.numBlocks, self.frameNames, matteNames)
        
        np.save(self.dataLocation + os.sep + "vanilla_distMat.npy", distanceMatrix)
            
        distanceMatrixFilt = vtu.filterDistanceMatrix(distanceMatrix, self.numFilterFrames, False)
        distMat = self.estimateFutureCost(0.999, 2.0, distanceMatrixFilt, np.ones(distanceMatrixFilt.shape))
        
        np.save(self.dataLocation + os.sep + "proc_distMat.npy", distMat)
        
        self.mainWindow.updateDistMat()
        
        self.mainWindow.lockGUI(False)
        self.mainWindow.showLoading(False)
        
    def toRgbTrimap(self, trimap) :
        if self.downsampleFactor > 1:
            trimap = cv2.resize(trimap, (trimap.shape[1]/self.downsampleFactor, trimap.shape[0]/self.downsampleFactor), interpolation = cv2.INTER_NEAREST)
            
        colorTrimap = np.zeros(np.hstack((trimap.shape, 3)), dtype=np.uint8)
        bgIdx = np.argwhere(trimap == 0)
        colorTrimap[bgIdx[:, 0], bgIdx[:, 1], 2] = 255
        mgIdx = np.argwhere(trimap == 1)
        colorTrimap[mgIdx[:, 0], mgIdx[:, 1], 1] = 255
        fgIdx = np.argwhere(trimap == 2)
        colorTrimap[fgIdx[:, 0], fgIdx[:, 1], 0] = 255
        
        return cv2.cvtColor(colorTrimap, cv2.COLOR_RGB2BGR)
        
    def changeBorderSize(self, size):
        self.borderSize = size
        if self.state > 0 :
            self.processScribble()
                
    def lockGUI(self, lock):
        
        self.matteViewerGroup.scribbleArea.setCanScribble(not lock)
        self.matteViewerGroup.slider.setEnabled(not lock)
        self.matteViewerGroup.frameSpinBox.setEnabled(not lock)
        
        self.scribbleSizeSlider.setEnabled(not lock)
        self.backgroundToolButton.setEnabled(not lock)
        self.foregroundToolButton.setEnabled(not lock)
        self.deleteToolButton.setEnabled(not lock)
        self.clearScribbleButton.setEnabled(not lock)
        self.scribbleAlphaSlider.setEnabled(not lock)
        
        self.acceptScribbleButton.setEnabled(not lock)
        self.scribbleBorderSlider.setEnabled(not lock)
        self.acceptTrimapButton.setEnabled(not lock)
        self.trimapListTable.setEnabled(not lock)
        self.computeMattesButton.setEnabled(not lock)
        
        self.computeDistMatButton.setEnabled(not lock)
        self.numBlocksComboBox.setEnabled(not lock)
        self.useMattesCheckBox.setEnabled(not lock)
        
    def setLoadedMovie(self, movie, frameNames, dataLocation, downsampleFactor) :
        self.movie = movie
        self.frameNames = frameNames
        self.dataLocation = dataLocation
        self.downsampleFactor = downsampleFactor
        self.showFrame(0)
        self.matteViewerGroup.scribbleArea.setCanScribble(True)
        self.matteViewerGroup.setMaximum(self.movie.shape[-1]-1)
        self.matteViewerGroup.setValue(0)
        
        self.loadTrimaps()
        
    def loadTrimaps(self) :
        trimapNames = glob.glob(self.dataLocation + os.sep + "trimap*.png")
        self.trimaps = {}
        if len(trimapNames) > 0 :
            frameSize = cv2.imread(trimapNames[0]).shape
                
            self.loadingText = self.mainWindow.loadingLabel.text() + "\nLoading trimaps..."
            
            for i in range(0, len(trimapNames)) :
                
                im = cv2.imread(trimapNames[i])
                
                frameName = '-'.join(filter(None, trimapNames[i].split(os.sep)[-1].split('-')[1:]))
                self.trimaps[frameName] = np.array(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
                
                sys.stdout.write('\r' + "Loading trimap " + np.string_(i) + " of " + np.string_(len(trimapNames)))
                sys.stdout.flush()
                self.mainWindow.loadingLabel.setText(self.loadingText+"\ntrimap " + np.string_(i) + " of " + np.string_(len(trimapNames)))
                QtCore.QCoreApplication.processEvents()
            print
            
        self.setTrimapList()
            
    def setTrimapList(self) :
        self.trimapListTable.setRowCount(0)
        if len(self.trimaps) > 0 :
            self.trimapListTable.setRowCount(len(self.trimaps))
            
            for frameName, i in zip(np.sort(self.trimaps.keys()), xrange(0, len(self.trimaps))):
                self.trimapListTable.setItem(i, 0, QtGui.QTableWidgetItem(frameName))
                
                if self.dataLocation + os.sep + frameName == self.frameNames[self.matteViewerGroup.frameSpinBox.value()] :
                    self.matteViewerGroup.setScribble(self.toRgbTrimap(self.trimaps[frameName]))
        else :
            self.trimapListTable.setRowCount(1)
            self.trimapListTable.setItem(0, 0, QtGui.QTableWidgetItem("No confirmed trimaps"))
            
    def confirmTrimap(self):
        try :
            self.trimap
        except : 
            return
        
        frameName = self.frameNames[self.matteViewerGroup.frameSpinBox.value()].split(os.sep)[-1]
        
        doSave = QtGui.QMessageBox.Yes
        if frameName in self.trimaps.keys() :
            doSave = QtGui.QMessageBox.question(self, 'Override Trimap',
                                    "A trimap for this frame already exists.\nDo you want to override?", 
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            
        if doSave == QtGui.QMessageBox.Yes :
            trimapName = self.dataLocation + os.sep + "trimap-" + frameName
            toSave = cv2.cvtColor(np.array(self.trimap, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(trimapName, toSave)
            self.trimaps[frameName] = self.trimap
            self.setTrimapList()
            
            
    def deleteTrimap(self) :
        try :
            self.trimaps
        except :
            return
        
        if len(self.trimaps) > 0 and len(self.trimapListTable.selectionModel().selectedRows()) :
            doDel = QtGui.QMessageBox.question(self, 'Delete Trimap',
                                    "This will delete the trimap from disc.\nDo you want to delete?", 
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            if doDel == QtGui.QMessageBox.Yes :
                print "deleting", self.trimapListTable.selectionModel().selectedRows()[0].row()
                frameName = self.trimapListTable.item(self.trimapListTable.selectionModel().selectedRows()[0].row(), 0).text()
                ## deleting from disc
                os.remove(self.dataLocation + os.sep + "trimap-" + frameName)
                ## deleting from internal data structure
                self.trimaps.pop(frameName, None)
                ## deleting from table list
                self.trimapListTable.removeRow(self.trimapListTable.selectionModel().selectedRows()[0].row())
    
    def showFrame(self, idx) :
        if idx >= 0 and idx < self.movie.shape[-1] :
            self.matteViewerGroup.setCurrentFrame(self.movie[:, :, :, idx], False)
            frameName = self.frameNames[idx].split(os.sep)[-1]
            
            ## show trimap on scribble area if it exists
            if frameName in self.trimaps.keys() :
                self.matteViewerGroup.setScribble(self.toRgbTrimap(self.trimaps[frameName]))
            
            try :
                self.dataLocation
            except :
                return
            
            ## show the matted version of the frame it exists
            if os.path.isfile(self.dataLocation + os.sep + "matte-" + frameName) :
                alphaMatte = cv2.cvtColor(cv2.imread(self.dataLocation + os.sep + "matte-" + frameName), cv2.COLOR_BGR2GRAY)
                if self.downsampleFactor > 1 :
                    alphaMatte = cv2.resize(alphaMatte, (alphaMatte.shape[1]/self.downsampleFactor, alphaMatte.shape[0]/self.downsampleFactor))
                alphaMatte = np.reshape(alphaMatte, np.hstack((self.movie.shape[0:2], 1)))
                self.matteViewerGroup.setCurrentFrame(np.concatenate((cv2.cvtColor(self.movie[:, :, :, idx], cv2.COLOR_RGB2BGR), alphaMatte), axis=-1), True)
        
    def setScribbleSizeLabel(self, size) :
        self.scribbleSizeLabel.setText("Scribble Size [" + np.string_(size) + "]")
        
    def setScribbleAlphaLabel(self, alpha) :
        self.scribbleAlphaLabel.setText("Scribble Opacity [" + np.string_(float(alpha)/10.0) + "]")
        
    def setScribbleBorderLabel(self, size) :
        self.scribbleBorderLabel.setText("Border Size [" + np.string_(size) + "]")
        
    def bgScribbleToggled(self, checked) :
        if checked :
            self.foregroundToolButton.setChecked(False)
            self.deleteToolButton.setChecked(False)
            self.matteViewerGroup.scribbleArea.setPenColor(QtGui.QColor.fromRgb(0, 0, 255))
            self.matteViewerGroup.scribbleArea.setCanScribble(True)
        else :
            if not self.deleteToolButton.isChecked() and not self.foregroundToolButton.isChecked() :
                self.matteViewerGroup.scribbleArea.setCanScribble(False)
        
    def fgScribbleToggled(self, checked) :
        if checked :
            self.backgroundToolButton.setChecked(False)
            self.deleteToolButton.setChecked(False)
            self.matteViewerGroup.scribbleArea.setPenColor(QtGui.QColor.fromRgb(255, 0, 0))
            self.matteViewerGroup.scribbleArea.setCanScribble(True)
        else :
            if not self.deleteToolButton.isChecked() and not self.backgroundToolButton.isChecked() :
                self.matteViewerGroup.scribbleArea.setCanScribble(False)
            
    def delScribbleToggled(self, checked) :
        if checked :
            self.backgroundToolButton.setChecked(False)
            self.foregroundToolButton.setChecked(False)
            self.matteViewerGroup.scribbleArea.setPenColor(QtGui.QColor.fromRgb(250, 250, 250))
            self.matteViewerGroup.scribbleArea.setCanScribble(True)
        else :
            if not self.backgroundToolButton.isChecked() and not self.foregroundToolButton.isChecked() :
                self.matteViewerGroup.scribbleArea.setCanScribble(False)
                
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Delete :
                if self.trimapListTable.hasFocus() :
                    self.deleteTrimap()
                return True
        return QtGui.QWidget.eventFilter(self, obj, event)
            
    def goToTrimap(self, row) :
        try :
            frameName = self.trimapListTable.item(row, 0).text()
            idx = np.ndarray.flatten(np.argwhere(self.frameNames == self.dataLocation + os.sep + frameName))[0]
        except :
            return
        
        if idx >= 0 and idx <= self.matteViewerGroup.frameSpinBox.maximum() :
            self.matteViewerGroup.frameSpinBox.setValue(idx)
            
    def setNumBlocks(self, value) :
        self.numBlocks = 2**value
                    
    def setUseMattes(self, value) :
        self.useMattes = value
        
    def createGUI(self) :
                
        ## WIDGETS ##
        
        controlsGroup = QtGui.QGroupBox("Controls")
        controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        controlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        scribbleControlsGroup = QtGui.QGroupBox("Scribble Controls")
        scribbleControlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        scribbleControlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        mattingControlsGroup = QtGui.QGroupBox("Matting Controls")
        mattingControlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        mattingControlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        distanceControlsGroup = QtGui.QGroupBox("Distance Controls")
        distanceControlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        distanceControlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        self.matteViewerGroup = MatteViewerGroup("Work Area")

        self.scribbleSizeLabel = QtGui.QLabel("Scribble Size [1]")
        
        self.scribbleSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scribbleSizeSlider.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.scribbleSizeSlider.setMinimum(1)
        self.scribbleSizeSlider.setMaximum(50)
        self.scribbleSizeSlider.setSingleStep(1)
        
        self.backgroundToolButton = QtGui.QToolButton()
        self.backgroundToolButton.setToolTip("Scribble Background")
        self.backgroundToolButton.setCheckable(True)
        self.backgroundToolButton.setIcon(QtGui.QIcon("bg.png"))
        self.foregroundToolButton = QtGui.QToolButton()
        self.foregroundToolButton.setToolTip("Scribble Foreground")
        self.foregroundToolButton.setCheckable(True)
        self.foregroundToolButton.setIcon(QtGui.QIcon("fg.png"))
        self.deleteToolButton = QtGui.QToolButton()
        self.deleteToolButton.setToolTip("Delete Scribble")
        self.deleteToolButton.setCheckable(True)
        self.deleteToolButton.setIcon(QtGui.QIcon("del.png"))
        
        self.clearScribbleButton = QtGui.QPushButton("Clear Scribble")
        
        self.scribbleAlphaLabel = QtGui.QLabel("Scribble Opacity [0.0]")
        self.scribbleAlphaSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scribbleAlphaSlider.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.scribbleAlphaSlider.setMinimum(0)
        self.scribbleAlphaSlider.setMaximum(10)
        self.scribbleAlphaSlider.setSingleStep(1)
        
        self.acceptScribbleButton = QtGui.QPushButton("Confirm Scribble")
        
        self.scribbleBorderLabel = QtGui.QLabel("Border Size [1]")
        self.scribbleBorderSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.scribbleBorderSlider.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.scribbleBorderSlider.setMinimum(1)
        self.scribbleBorderSlider.setMaximum(50)
        self.scribbleBorderSlider.setSingleStep(1)
        
        self.acceptTrimapButton = QtGui.QPushButton("Confirm Trimap")
        
        self.trimapListTable = QtGui.QTableWidget(1, 1)
        self.trimapListTable.horizontalHeader().setStretchLastSection(True)
        self.trimapListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Trimaps for loaded data"))
        self.trimapListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Fixed)
        self.trimapListTable.setItem(0, 0, QtGui.QTableWidgetItem("No Loaded Data"))
        self.trimapListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.trimapListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.trimapListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.trimapListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        
        
        self.computeMattesButton = QtGui.QPushButton("Compute Mattes")
        
        self.computeDistMatButton = QtGui.QPushButton("Compute Distance Matrix")
        self.useMattesCheckBox = QtGui.QCheckBox("Use Mattes")
        
        numBlocksLabel = QtGui.QLabel("Number of Blocks")
        self.numBlocksComboBox = QtGui.QComboBox()
        self.numBlocksComboBox.addItem("1")
        self.numBlocksComboBox.addItem("2")
        self.numBlocksComboBox.addItem("4")
        self.numBlocksComboBox.addItem("8")
        self.numBlocksComboBox.addItem("16")
                
        ## SIGNALS ##
        
        self.matteViewerGroup.frameSpinBox.valueChanged[int].connect(self.showFrame)
        
        self.scribbleSizeSlider.valueChanged[int].connect(self.setScribbleSizeLabel)
        self.scribbleSizeSlider.valueChanged[int].connect(self.matteViewerGroup.scribbleArea.setPenWidth)
        self.scribbleAlphaSlider.valueChanged[int].connect(self.setScribbleAlphaLabel)
        self.scribbleAlphaSlider.valueChanged[int].connect(self.matteViewerGroup.scribbleArea.setScribbleAlphaInt)
        self.backgroundToolButton.toggled[bool].connect(self.bgScribbleToggled)
        self.foregroundToolButton.toggled[bool].connect(self.fgScribbleToggled)
        self.deleteToolButton.toggled[bool].connect(self.delScribbleToggled)
        self.clearScribbleButton.clicked.connect(self.matteViewerGroup.scribbleArea.clearScribble)
        self.acceptScribbleButton.clicked.connect(self.processScribble)
        self.scribbleBorderSlider.valueChanged[int].connect(self.setScribbleBorderLabel)
        self.scribbleBorderSlider.valueChanged[int].connect(self.changeBorderSize)
        
        self.acceptTrimapButton.clicked.connect(self.confirmTrimap)
        self.trimapListTable.currentCellChanged.connect(self.goToTrimap)
        self.computeMattesButton.clicked.connect(self.computeMattes)
        
        self.computeDistMatButton.clicked.connect(self.computeDistMat)
        self.numBlocksComboBox.currentIndexChanged[int].connect(self.setNumBlocks)
        self.useMattesCheckBox.stateChanged[int].connect(self.setUseMattes)
        
        ## LAYOUTS ##
        
        scribbleControlsLayout = QtGui.QGridLayout()
        scribbleControlsLayout.addWidget(self.scribbleSizeLabel, 0, 0, 1, 2, QtCore.Qt.AlignCenter)
        scribbleControlsLayout.addWidget(self.scribbleSizeSlider, 1, 0, 1, 2, QtCore.Qt.AlignCenter)
        scribbleControlsLayout.addWidget(self.scribbleAlphaLabel, 2, 0, 1, 2, QtCore.Qt.AlignCenter)
        scribbleControlsLayout.addWidget(self.scribbleAlphaSlider, 3, 0, 1, 2, QtCore.Qt.AlignCenter)
        scribbleControlsLayout.addWidget(self.backgroundToolButton, 4, 0, QtCore.Qt.AlignRight)
        scribbleControlsLayout.addWidget(self.foregroundToolButton, 4, 1, QtCore.Qt.AlignLeft)
        scribbleControlsLayout.addWidget(self.deleteToolButton, 5, 0, QtCore.Qt.AlignRight)
        scribbleControlsLayout.addWidget(self.clearScribbleButton, 6, 0, 1, 2, QtCore.Qt.AlignCenter)
        scribbleControlsGroup.setLayout(scribbleControlsLayout)
        
        mattingControlsLayout = QtGui.QGridLayout()
        mattingControlsLayout.addWidget(self.acceptScribbleButton, 0, 0, QtCore.Qt.AlignCenter)
        mattingControlsLayout.addWidget(self.scribbleBorderLabel, 1, 0, QtCore.Qt.AlignCenter)
        mattingControlsLayout.addWidget(self.scribbleBorderSlider, 2, 0, QtCore.Qt.AlignCenter)
        mattingControlsLayout.addWidget(self.acceptTrimapButton, 3, 0, QtCore.Qt.AlignCenter)
        mattingControlsLayout.addWidget(self.trimapListTable, 4, 0, QtCore.Qt.AlignCenter)
        mattingControlsLayout.addWidget(self.computeMattesButton, 5, 0, QtCore.Qt.AlignCenter)
        mattingControlsGroup.setLayout(mattingControlsLayout)
        
        distanceControlsLayout = QtGui.QGridLayout()
        distanceControlsLayout.addWidget(self.useMattesCheckBox, 0, 0, 1, 2, QtCore.Qt.AlignCenter)
        distanceControlsLayout.addWidget(numBlocksLabel, 1, 0, QtCore.Qt.AlignRight)
        distanceControlsLayout.addWidget(self.numBlocksComboBox, 1, 1, QtCore.Qt.AlignLeft)
        distanceControlsLayout.addWidget(self.computeDistMatButton, 2, 0, 1, 2, QtCore.Qt.AlignCenter)
        distanceControlsGroup.setLayout(distanceControlsLayout)
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addStretch()
        controlsLayout.addWidget(scribbleControlsGroup)
        controlsLayout.addWidget(mattingControlsGroup)
        controlsLayout.addWidget(distanceControlsGroup)
        controlsLayout.addStretch()
        controlsGroup.setLayout(controlsLayout)
        
        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(controlsGroup)
        mainLayout.addWidget(self.matteViewerGroup)
#         mainLayout.addWidget(workAreaGroup)
        self.setLayout(mainLayout)

