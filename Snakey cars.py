# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import sys
import scipy as sp

import cv2
import time
import os
import scipy.io as sio
import glob
from scipy import ndimage as spimg
import shutil

from PIL import Image
from PySide import QtCore, QtGui

import GraphWithValues as gwv

app = QtGui.QApplication(sys.argv)

# <codecell>

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

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"

# <codecell>

## load
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
    print trackedSprites[-1][DICT_SEQUENCE_NAME], DICT_FOOTPRINTS in trackedSprites[-1].keys()
    
spriteIdx = 0
currentSprite = trackedSprites[spriteIdx]
print "using", currentSprite[DICT_SEQUENCE_NAME]

bgImage = np.array(Image.open(dataPath+dataSet+"median.png"))
preloadedSpritePatches = list(np.load("/media/ilisescu/Data1/PhD/data/" + dataSet + "preloadedSpritePatches.npy"))[spriteIdx]

# <codecell>

DIRECTION_VECTOR_LENGTH = 50
class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text="", parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.image = None
        self.spritePatch = None
        self.trajectory = None
        self.currentFrame = 0
        self.currentPosition = None
        self.currentDirection = None
        self.patchOffset = np.array([0.0, 0.0])
        self.transparency = 128
        
    def setImage(self, image) : 
        self.image = image.copy()
        self.setMinimumSize(self.image.size())
        self.update()
        
    def setSpritePatch(self, spritePatch, patchOffset = np.array([0.0, 0.0])) : 
        self.spritePatch = spritePatch
        self.patchOffset = patchOffset
        self.update()
        
    def setTrajectory(self, trajectory):
        self.trajectory = trajectory
        self.update()
    
    def setCurrentFrame(self, currentFrame):
        self.currentFrame = currentFrame
        self.update()
    
    def setCurrentControlVector(self, currentPosition, currentDirection):
        self.currentPosition = currentPosition
        self.currentDirection = currentDirection
        self.update()
    
    def setTransparency(self, transparency):
        self.transparency = transparency
        self.update()
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        
        if self.image != None :
            upperLeft = ((self.width()-self.image.width())/2, (self.height()-self.image.height())/2)
            
            ## draw image
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.image)
            
            ## draw sprite patch
            if self.spritePatch != None :
                reconstructedImg = np.ascontiguousarray(np.zeros((self.spritePatch['patch_size'][0], self.spritePatch['patch_size'][1], 4)), dtype=np.uint8)
                reconstructedImg[self.spritePatch['visible_indices'][:, 0], self.spritePatch['visible_indices'][:, 1], :] = self.spritePatch['sprite_colors']
                reconstructedQImage = QtGui.QImage(reconstructedImg.data, reconstructedImg.shape[1], reconstructedImg.shape[0], 
                                                   reconstructedImg.strides[0], QtGui.QImage.Format_ARGB32)

                painter.drawImage(QtCore.QRect(self.spritePatch['top_left_pos'][1]+self.patchOffset[1]+upperLeft[0], self.spritePatch['top_left_pos'][0]+self.patchOffset[0]+upperLeft[1],
                                   self.spritePatch['patch_size'][1], self.spritePatch['patch_size'][0]), reconstructedQImage)
                
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                    
                            
            ## draw trajectory
            if self.trajectory != None :
                trajectory = self.trajectory + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, self.transparency), 2, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                for i, j in zip(trajectory[0:-1, :], trajectory[1:, :]) :
                    
                    painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                     QtCore.QPointF(j[0], j[1]))
            
                ## draw current frame direction vector
                if self.currentFrame >= 0 and self.currentFrame < len(self.trajectory) :
                    ## draw direction vector
                    if self.currentFrame < len(self.trajectory)-1 :
                        
                        normal = self.trajectory[self.currentFrame+1, :] - self.trajectory[self.currentFrame, :]
                        normal /= np.linalg.norm(normal)
                        
                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, self.transparency), 3, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawLine(QtCore.QPointF(trajectory[self.currentFrame, 0], trajectory[self.currentFrame, 1]),
                                         QtCore.QPointF(trajectory[self.currentFrame, 0], trajectory[self.currentFrame, 1])+(QtCore.QPointF(normal[0], normal[1])*DIRECTION_VECTOR_LENGTH))
                        
                        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 128, 128, self.transparency), 1, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        painter.drawLine(QtCore.QPointF(trajectory[self.currentFrame, 0], trajectory[self.currentFrame, 1]),
                                         QtCore.QPointF(trajectory[self.currentFrame, 0], trajectory[self.currentFrame, 1])+(QtCore.QPointF(normal[0], normal[1])*DIRECTION_VECTOR_LENGTH))
                        
                    ## draw point
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, self.transparency), 5, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawPoint(QtCore.QPointF(trajectory[self.currentFrame, 0], trajectory[self.currentFrame, 1]))
                    
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 128, 128, self.transparency), 3, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    
                    painter.drawPoint(QtCore.QPointF(trajectory[self.currentFrame, 0], trajectory[self.currentFrame, 1]))
                    
            
                            
            ## draw control vector
            if self.currentPosition != None and self.currentDirection  != None :
                currentPosition = self.currentPosition + np.array([upperLeft[0], upperLeft[1]])
                ## draw line
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, self.transparency), 3, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawLine(QtCore.QPointF(currentPosition[0], currentPosition[1]),
                                 QtCore.QPointF(currentPosition[0], currentPosition[1])+(QtCore.QPointF(self.currentDirection[0], self.currentDirection[1])*DIRECTION_VECTOR_LENGTH))

                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(128, 128, 255, self.transparency), 1, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawLine(QtCore.QPointF(currentPosition[0], currentPosition[1]),
                                 QtCore.QPointF(currentPosition[0], currentPosition[1])+(QtCore.QPointF(self.currentDirection[0], self.currentDirection[1])*DIRECTION_VECTOR_LENGTH))

                ## draw point
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, self.transparency), 5, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                painter.drawPoint(QtCore.QPointF(currentPosition[0], currentPosition[1]))

                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(128, 128, 255, self.transparency), 3, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

                painter.drawPoint(QtCore.QPointF(currentPosition[0], currentPosition[1]))
                        

# <codecell>

MAX_VELOCITY = 150.0 ## pixels/second
MAX_ANGULAR_VELOCITY = 45.0 ## degrees/second
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        ## parameters changed through UI
        self.frameIdx = 0
        self.trajectorySmoothness = 7
        self.orientationsSmoothness = 1
        self.doPlayControl = False
        self.desiredDirection = np.array([0.0, 0.0])
        self.currentPosition = np.array([0.0, 0.0])
        
        self.velocity = 0.0
        self.acceleration = 120.0
        
        self.angularVelocity = 0.0
        self.angularAcceleration = 150.0
        
        self.lastControlCommandTime = None
        self.doAccelerate = 0.0 ## if 0 it means no accelerate, otherwise it's forward using 1.0 and backwards using -1.0
        self.doTurn = 0.0 ## if 0 it means no turn, otherwise it's ccw using 1.0 and cw using -1.0
        
        ## create all widgets and layout
        self.createGUI()
        
        self.setWindowTitle("Snakey cars")
        self.resize(1800, 950)
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.advancePlayState)
        self.lastRenderTime = time.time()
        
        ## data computed using the manually adjusted parameters
        self.originalTrajectory = None
        self.originalTrajectoryDirections = None
        self.topDownTrajectory = None
        self.homography = np.eye(3)
        
        self.bgImage = np.ascontiguousarray(Image.open(dataPath+dataSet+"median.png"))[:, :, 0:3]
        qim = QtGui.QImage(self.bgImage.data, self.bgImage.shape[1], 
                           self.bgImage.shape[0], self.bgImage.strides[0], QtGui.QImage.Format_RGB888);
        self.originalImageLabel.setImage(qim)
        self.topDownBgImage = None
        
        self.changeTrajectoryFiltering()
        
        self.setFocus()
            
    def advancePlayState(self) :
        if self.doPlayControl :
            currentTime = time.time()
            deltaTime = currentTime - self.lastRenderTime

            ####################### change velocity based on acceleration #######################
            if self.doAccelerate == 1.0 :
                acceleration = self.acceleration
                ## if I want to go forwards while going backwards, accelerate faster
                if self.velocity < 0.0 :
                    acceleration = self.acceleration*2
                    
                self.velocity = np.min([MAX_VELOCITY, self.velocity + acceleration*deltaTime*self.doAccelerate])
            elif self.doAccelerate == -1.0 :
                acceleration = self.acceleration
                ## if I want to go backwards while going forwards, accelerate faster
                if self.velocity > 0.0 :
                    acceleration = self.acceleration*2
                self.velocity = np.max([-MAX_VELOCITY, self.velocity + acceleration*deltaTime*self.doAccelerate])
            else :
                ## decrease velocity based on direction
                if self.velocity < 0.0 :
                    self.velocity = np.min([0.0, self.velocity + self.acceleration*deltaTime])
                else :
                    self.velocity = np.max([0.0, self.velocity - self.acceleration*deltaTime])

            ####################### change angular velocity based on angular acceleration #######################
            if self.doTurn == 1.0 :
                self.angularVelocity = np.min([MAX_ANGULAR_VELOCITY, self.angularVelocity + self.angularAcceleration*deltaTime*self.doTurn])
            elif self.doTurn == -1.0 :
                self.angularVelocity = np.max([-MAX_ANGULAR_VELOCITY, self.angularVelocity + self.angularAcceleration*deltaTime*self.doTurn])
            else :
                ## decrease angular velocity based on direction
                if self.angularVelocity < 0.0 :
                    self.angularVelocity = np.min([0.0, self.angularVelocity + self.angularAcceleration*deltaTime])
                else :
                    self.angularVelocity = np.max([0.0, self.angularVelocity - self.angularAcceleration*deltaTime])
            
            ####################### update direction based on angular velocity #######################
#             angle = self.angularVelocity*np.pi/180.0*deltaTime*np.abs(self.velocity)/MAX_VELOCITY ## not sure how to get it to only turn if I'm moving but this works ish
            angle = self.angularVelocity*np.pi/180.0*deltaTime ## not sure how to get it to only turn if I'm moving but this works ish
            rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            newDirection = np.dot(self.desiredDirection.reshape([1, 2]), rotMatrix).flatten()
            directionAngleDistances = np.abs(np.arccos(np.dot(newDirection, self.originalTrajectoryDirections.T))*180.0/np.pi)
            if np.min(directionAngleDistances) < self.minAngleDistanceSlider.value()/10.0 :
                self.desiredDirection = np.copy(newDirection)
            
            ####################### update position based on direction and velocity #######################
            self.currentPosition = self.currentPosition+self.desiredDirection*self.velocity*deltaTime ## here there should be a -0.5at**2 but the particle acceleration a is 0.0 while self.acceleration refers to the velocity
            
            self.originalImageLabel.setCurrentControlVector(self.currentPosition, self.desiredDirection)
            
            ####################### get sprite frame patch and render it offset based on current current position #######################
            centerDistances = np.linalg.norm(self.originalTrajectory[:-1, :]-self.currentPosition, axis=1)
            frameIdxToUse = np.argmin(self.dirVsPosBalanceAlphaSpinBox.value()*centerDistances + (1.0-self.dirVsPosBalanceAlphaSpinBox.value())*directionAngleDistances)
            patchOffset = self.currentPosition - self.originalTrajectory[frameIdxToUse]
            self.originalImageLabel.setSpritePatch(preloadedSpritePatches[frameIdxToUse], patchOffset[::-1])
            
            self.originalImageInfo.setText("Rendering at {0} FPS, pos: {1}, velocity: {2}, angle: {3}".format(int(1.0/(deltaTime)), self.currentPosition, self.velocity, self.angularVelocity))
            self.lastRenderTime = np.copy(currentTime)
        
    def showOriginalFrame(self, idx) :
        if idx >= 0 and idx < len(currentSprite[DICT_BBOXES]) :
            self.frameIdx = idx
            ## get the sprite patch and give it to original image label
            self.originalImageLabel.setSpritePatch(preloadedSpritePatches[self.frameIdx])
            
            self.originalImageLabel.setCurrentFrame(self.frameIdx)
        self.setFocus()
        
    def changeTrajectoryFiltering(self) :
        self.trajectorySmoothness = self.trajectorySmoothnessSpinBox.value()
        self.orientationsSmoothness = self.orientationsSmoothnessSpinBox.value()

        xs, ys = np.array([currentSprite[DICT_BBOX_CENTERS][i] for i in np.sort(currentSprite[DICT_BBOX_CENTERS].keys())]).T
        smoothedPath = np.array([xs, ys]).T

        ## smooth path
        smoothedPath = np.array([spimg.filters.gaussian_filter1d(smoothedPath[:, 0], self.trajectorySmoothness, axis=0), 
                                 spimg.filters.gaussian_filter1d(smoothedPath[:, 1], self.trajectorySmoothness, axis=0)]).T
        
        self.originalTrajectory = np.copy(smoothedPath)
        self.originalTrajectoryDirections = np.array([self.originalTrajectory[i, :]-self.originalTrajectory[j, :] for i, j in zip(xrange(1, len(self.originalTrajectory)), xrange(0, len(self.originalTrajectory)-1))])
        self.originalTrajectoryDirections /= np.linalg.norm(self.originalTrajectoryDirections, axis=1).reshape([len(self.originalTrajectoryDirections), 1])
        self.originalImageLabel.setTrajectory(self.originalTrajectory)
        self.setFocus()
    
    def changeUITransparency(self, transparency) :
        self.originalImageLabel.setTransparency(transparency)
        self.setFocus()
        
    def wheelEvent(self, e) :
        if not self.doPlayControl :
            if e.delta() < 0 :
                self.frameSpinBox.setValue(self.frameIdx-1)
            else :
                self.frameSpinBox.setValue(self.frameIdx+1)
            time.sleep(0.01)
    
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_Space :
            if not self.doPlayControl or self.frameIdx < len(currentSprite[DICT_BBOXES]) - 1 :
                self.doPlayControl = not self.doPlayControl
                if self.doPlayControl :
                    ## init control direction
                    self.desiredDirection = self.originalTrajectory[self.frameIdx+1, :] - self.originalTrajectory[self.frameIdx, :]
                    self.desiredDirection /= np.linalg.norm(self.desiredDirection)
                    self.currentPosition = self.originalTrajectory[self.frameIdx, :]
                    
                    self.lastRenderTime = time.time()
                    self.playTimer.start()
                    
                    self.frameSpinBox.setEnabled(False)
                    self.frameSlider.setEnabled(False)
                else :
                    self.desiredDirection = np.array([0.0, 0.0])
                    self.velocity = 0.0
                    self.playTimer.stop()
                    self.originalImageInfo.setText("Original")
                    self.originalImageLabel.setCurrentControlVector(None, None)
                    
                    self.frameSpinBox.setEnabled(True)
                    self.frameSlider.setEnabled(True)
                    self.frameSpinBox.setValue(self.frameIdx)
                    
        if e.key() == QtCore.Qt.Key_W or e.key() == QtCore.Qt.Key_Up :
            self.doAccelerate += 1.0
        if e.key() == QtCore.Qt.Key_S or e.key() == QtCore.Qt.Key_Down :
            self.doAccelerate -= 1.0
        if e.key() == QtCore.Qt.Key_A or e.key() == QtCore.Qt.Key_Left :
            self.doTurn += 1.0
        if e.key() == QtCore.Qt.Key_D or e.key() == QtCore.Qt.Key_Right :
            self.doTurn -= 1.0
    
    def keyReleaseEvent(self, e) :
        if e.key() == QtCore.Qt.Key_W or e.key() == QtCore.Qt.Key_Up :
            self.doAccelerate -= 1.0
        if e.key() == QtCore.Qt.Key_S or e.key() == QtCore.Qt.Key_Down :
            self.doAccelerate += 1.0
        if e.key() == QtCore.Qt.Key_A or e.key() == QtCore.Qt.Key_Left :
            self.doTurn -= 1.0
        if e.key() == QtCore.Qt.Key_D or e.key() == QtCore.Qt.Key_Right :
            self.doTurn += 1.0
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.originalImageLabel = ImageLabel()
        self.originalImageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.originalImageLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.originalImageInfo = QtGui.QLabel("Original")
        self.originalImageInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        
        self.frameSpinBox = QtGui.QSpinBox()
        self.frameSpinBox.setRange(0, len(currentSprite[DICT_BBOXES])-1)
        self.frameSpinBox.setSingleStep(1)
        
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(len(currentSprite[DICT_BBOXES])-1)
        
        
        controlsGroup = QtGui.QGroupBox("Controls")
        controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
                                             "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        controlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.dirVsPosBalanceAlphaSpinBox = QtGui.QDoubleSpinBox()
        self.dirVsPosBalanceAlphaSpinBox.setRange(0, 1.0)
        self.dirVsPosBalanceAlphaSpinBox.setSingleStep(0.01)
        self.dirVsPosBalanceAlphaSpinBox.setValue(0.0)
        
        self.trajectorySmoothnessSpinBox = QtGui.QSpinBox()
        self.trajectorySmoothnessSpinBox.setRange(1, 200)
        self.trajectorySmoothnessSpinBox.setValue(self.trajectorySmoothness)
        
        self.orientationsSmoothnessSpinBox = QtGui.QSpinBox()
        self.orientationsSmoothnessSpinBox.setRange(1, 200)
        self.orientationsSmoothnessSpinBox.setValue(self.orientationsSmoothness)
        
        self.uiTransparencySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.uiTransparencySlider.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.uiTransparencySlider.setMinimum(0)
        self.uiTransparencySlider.setMaximum(255)
        self.uiTransparencySlider.setValue(128)
        
        self.minAngleDistanceSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.minAngleDistanceSlider.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.minAngleDistanceSlider.setMinimum(1)
        self.minAngleDistanceSlider.setMaximum(1800)
        self.minAngleDistanceSlider.setValue(10)
        
        
        ## SIGNALS ##
        
        self.frameSpinBox.valueChanged[int].connect(self.frameSlider.setValue)
        self.frameSlider.valueChanged[int].connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.showOriginalFrame)
        
        self.trajectorySmoothnessSpinBox.valueChanged[int].connect(self.changeTrajectoryFiltering)
        self.orientationsSmoothnessSpinBox.valueChanged[int].connect(self.changeTrajectoryFiltering)
        self.uiTransparencySlider.valueChanged[int].connect(self.changeUITransparency)
        self.dirVsPosBalanceAlphaSpinBox.editingFinished.connect(self.setFocus)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        
        sliderLayout = QtGui.QHBoxLayout()
        sliderLayout.addWidget(self.frameSlider)
        sliderLayout.addWidget(self.frameSpinBox)
        
        originalImageLayout = QtGui.QVBoxLayout()
        originalImageLayout.addWidget(self.originalImageLabel)
        originalImageLayout.addWidget(self.originalImageInfo)
        originalImageLayout.addLayout(sliderLayout)
        
        controlsLayout = QtGui.QGridLayout()
        idx = 0
        controlsLayout.addWidget(QtGui.QLabel("Dir(low) vs Pos(high) Balance"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.dirVsPosBalanceAlphaSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        controlsLayout.addWidget(QtGui.QLabel("Trajectory Smoothness"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.trajectorySmoothnessSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        controlsLayout.addWidget(QtGui.QLabel("Orientations Smoothness"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.orientationsSmoothnessSpinBox, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        controlsLayout.addWidget(QtGui.QLabel("Min Trajectory Angle Deviation"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.minAngleDistanceSlider, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        controlsLayout.addWidget(QtGui.QLabel("UI Transparency"), idx, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.uiTransparencySlider, idx, 1, 1, 1, QtCore.Qt.AlignLeft); idx+=1
        controlsGroup.setLayout(controlsLayout)
        
        topDownImageLayout = QtGui.QVBoxLayout()
        topDownImageLayout.addWidget(controlsGroup)
        topDownImageLayout.addStretch()
        
        mainLayout.addLayout(originalImageLayout)
        mainLayout.addLayout(topDownImageLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

