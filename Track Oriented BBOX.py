# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import sys
sys.path.append('CMT tracker/')

import CMT
import CMT_utils

import cv2
import time
import os
import scipy.io as sio
import glob


from PIL import Image
from PySide import QtCore, QtGui


app = QtGui.QApplication(sys.argv)

# <codecell>

DICT_SPRITE_NAME = 'sprite_name'
# DICT_BBOX_AFFINES = 'bbox_affines'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
# DICT_NUM_FRAMES = 'num_frames'
# DICT_START_FRAME = 'start_frame'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MEDIAN_COLOR = 'median_color'

DRAW_FIRST_FRAME = 'first_frame'
DRAW_LAST_FRAME = 'last_frame'
DRAW_COLOR = 'color'

<<<<<<< HEAD
dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"#"clouds/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
# dataSet = "candle2/"
=======
# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"#"clouds/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
dataSet = "candle2/"
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

# <codecell>

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
print numOfFrames

# <codecell>

###### THIS CAN BE USED TO TRANSFER TRACKING INFO FROM ONE SPRITE TO ANOTHER
# oldTrack = np.load(np.sort(glob.glob(dataPath + "theme_park_cloudy/sprite*.npy"))[0]).item()
# print oldTrack.keys()
# newTrack = {
#                DICT_SPRITE_NAME:u'roller_coaster1',
#                DICT_BBOXES:{},
#                DICT_BBOX_CENTERS:{},
#                DICT_BBOX_ROTATIONS:{},
#                DICT_FRAMES_LOCATIONS:{}
#            }

# delta = 507 - 1112
# for frameIdx in np.sort(oldTrack[DICT_FRAMES_LOCATIONS].keys()) :
#     newTrack[DICT_BBOXES][frameIdx+delta] = oldTrack[DICT_BBOXES][frameIdx]
#     newTrack[DICT_BBOX_CENTERS][frameIdx+delta] = oldTrack[DICT_BBOX_CENTERS][frameIdx]
#     newTrack[DICT_BBOX_ROTATIONS][frameIdx+delta] = oldTrack[DICT_BBOX_ROTATIONS][frameIdx]
#     newTrack[DICT_FRAMES_LOCATIONS][frameIdx+delta] = frameLocs[frameIdx+delta]
    
# np.save(dataPath + dataSet + "/sprite-" + newTrack[DICT_SPRITE_NAME] + ".npy", newTrack)

# <codecell>

## load the tracked sprites
sprites = [{
               DICT_SPRITE_NAME:'havana_red_car_plusrot', 
               # DICT_BBOX_AFFINES:sio.loadmat("../ASLA tracker/result/havana_red_car_plusrot/Result/result.mat")['result'], 
               DICT_NUM_FRAMES:0, 
               DICT_FRAMES_LOCATIONS:[]
           }, 
           {
               DICT_SPRITE_NAME:'havana_bus', 
               # DICT_BBOX_AFFINES:sio.loadmat("../ASLA tracker/result/havana_bus/Result/result.mat")['result'], 
               DICT_NUM_FRAMES:0, 
               DICT_FRAMES_LOCATIONS:[]
           }
          ]

numOfSprites = len(sprites)
## setting number of frames from the number of tracked bboxes
for i in arange(numOfSprites) :
    trackedSprites[i][DICT_NUM_FRAMES] = len(trackedSprites[i][DICT_BBOX_AFFINES])
## setting frame locations for the tracked sprites
for i in arange(numOfSprites) :
    trackedSprites[i][DICT_FRAMES_LOCATIONS] = np.sort(glob.glob(dataPath + trackedSprites[i][DICT_SPRITE_NAME] + "/*.png"))

# <codecell>

print (dataPath+dataSet+formatString).format(1)

# <codecell>

cap = CMT_utils.FileVideoCapture(dataPath + dataSet + formatString)

if not cap.isOpened():
	print 'Unable to open video input.'
	sys.exit(1)
    
status, im0 = cap.read()
im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
im_draw = np.copy(im0)

# (tl, br) = CMT_utils.get_rect(im_draw)

# <codecell>

tl = (1064, 410) # (1068, 418) # (1058, 426)
br = (1167, 506) # (1164, 496) # (1058+114, 426+67)
tr = (1183, 439) # (1176, 442) # None
bl = (1048, 478) # (1054, 471) # None

tracker = CMT.CMT()
tracker.estimate_scale = True
tracker.estimate_rotation = True
pause_time = 10

tracker.initialise(im_gray0, tl, br, tr, bl)

# <codecell>

cv2.destroyAllWindows()
output = None#"CMT tracker/results/" + dataSet
if output != None and not os.path.isdir(output):
    os.makedirs(output)

frame = 1
while True: #frame < 2:
    # Read image
    status, im = cap.read()
    if not status:
        break
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_draw = np.copy(im)

    tic = time.time()
    tracker.process_frame(im_gray)
    toc = time.time()

    # Display results

    # Draw updated estimate
    if tracker.has_result:

        cv2.line(im_draw, tracker.tl, tracker.tr, (255, 0, 0), 4)
        cv2.line(im_draw, tracker.tr, tracker.br, (255, 0, 0), 4)
        cv2.line(im_draw, tracker.br, tracker.bl, (255, 0, 0), 4)
        cv2.line(im_draw, tracker.bl, tracker.tl, (255, 0, 0), 4)

    CMT_utils.draw_keypoints(tracker.tracked_keypoints, im_draw, (255, 255, 255))
    # this is from simplescale
    CMT_utils.draw_keypoints(tracker.votes[:, :2], im_draw)  # blue
    CMT_utils.draw_keypoints(tracker.outliers[:, :2], im_draw, (0, 0, 255))

    if output is not None:
        # Original image
        cv2.imwrite('{0}/input_{1:08d}.png'.format(output, frame), im)
        # Output image
        cv2.imwrite('{0}/output_{1:08d}.png'.format(output, frame), im_draw)

        # Keypoints
        with open('{0}/keypoints_{1:08d}.csv'.format(output, frame), 'w') as f:
            f.write('x y\n')
            np.savetxt(f, tracker.tracked_keypoints[:, :2], fmt='%.2f')

        # Outlier
        with open('{0}/outliers_{1:08d}.csv'.format(output, frame), 'w') as f:
            f.write('x y\n')
            np.savetxt(f, tracker.outliers, fmt='%.2f')

        # Votes
        with open('{0}/votes_{1:08d}.csv'.format(output, frame), 'w') as f:
            f.write('x y\n')
            np.savetxt(f, tracker.votes, fmt='%.2f')

        # Bounding box
        with open('{0}/bbox_{1:08d}.csv'.format(output, frame), 'w') as f:
            f.write('x y\n')
            # Duplicate entry tl is not a mistake, as it is used as a drawing instruction
            np.savetxt(f, np.array((tracker.tl, tracker.tr, tracker.br, tracker.bl, tracker.tl)), fmt='%.2f') 

    if True : #not args.quiet:
        cv2.imshow('main', im_draw)

        # Check key input
        k = cv2.waitKey(pause_time)
        key = chr(k & 255)
        if key == 'q':
            break
        if key == 'd':
            import ipdb; ipdb.set_trace()

    # Remember image
    im_prev = im_gray

    # Advance frame number
    frame += 1

    print '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(tracker.center[0], 
                    tracker.center[1], tracker.scale_estimate, tracker.active_keypoints.shape[0], 1000 * (toc - tic), frame)

# <codecell>

# Get initial keypoints in whole image
keypoints_cv = tracker.detector.detect(im_gray0)

# Remember keypoints that are in the rectangle as selected keypoints
ind = CMT_utils.in_rect(keypoints_cv, tl, br)

figure(); imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))

points = CMT_utils.keypoints_cv_to_np(keypoints_cv)
scatter(points[:, 0], points[:, 1])
scatter(points[ind, 0], points[ind, 1], marker='x', c='r')

p = Path([tl, tr, br, bl, tl])
insideBBox = p.contains_points(points)
# cv2.minAreaRect(

scatter(points[insideBBox, 0], points[insideBBox, 1], marker='.', c='g')

# <codecell>

from matplotlib.path import Path
p = Path([tl, tr, br, bl, tl])
print len(np.argwhere(p.contains_points(points)))

# <codecell>

print "{0:04}".format(4)

# <codecell>

class SemanticsSlider(QtGui.QSlider) :
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None) :
        super(SemanticsSlider, self).__init__(orientation, parent)
        style = "QSlider::handle:horizontal { background: #cccccc; width: 25px; border-radius: 0px; } "
        style += "QSlider::groove:horizontal { background: #dddddd; } "
        self.setStyleSheet(style)
        
        self.semanticsToDraw = []
        self.numOfFrames = 1
        self.selectedSemantics = 0
        
    def setSelectedSemantics(self, selectedSemantics) :
        self.selectedSemantics = selectedSemantics
        
    def setSemanticsToDraw(self, semanticsToDraw, numOfFrames) :
        self.semanticsToDraw = semanticsToDraw
        self.numOfFrames = float(numOfFrames)
        
        desiredHeight = len(self.semanticsToDraw)*7
        self.setFixedHeight(desiredHeight)
        
        self.resize(self.width(), self.height())
        self.update()
        
    def paintEvent(self, event) :
        super(SemanticsSlider, self).paintEvent(event)
        
        painter = QtGui.QPainter(self)
        
        ## draw semantics
        
        yCoord = 0.0
        for i in xrange(len(self.semanticsToDraw)) :
            col = self.semanticsToDraw[i][DRAW_COLOR]

            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(col[0], col[1], col[2], 255)))
            startX =  self.semanticsToDraw[i][DRAW_FIRST_FRAME]/self.numOfFrames*self.width()
            endX =  self.semanticsToDraw[i][DRAW_LAST_FRAME]/self.numOfFrames*self.width()

            if self.selectedSemantics == i :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 127), 1, 
                                              QtCore.Qt.DashLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+0.5, endX-startX, 5)

            else :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255 - col[0], 255 - col[1], 255 - col[2], 63), 1, 
                                              QtCore.Qt.SolidLine, QtCore.Qt.SquareCap, QtCore.Qt.MiterJoin))
                painter.drawRect(startX, yCoord+1.5, endX-startX, 3)


            yCoord += 7


        ## draw slider

        ## the slider is 2 pixels wide so remove 1.0 from X coord
        sliderXCoord = np.max((self.sliderPosition()/self.numOfFrames*self.width()-1.0, 0.0))
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 0, 0), 0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(0, 0, 0, 128)))
        painter.drawRect(sliderXCoord, 0, 2, self.height())

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
        
        self.setWindowTitle("CMT Tracking")
        self.resize(1280, 720)
        
        self.settingBBox = False
        self.movingBBox = False
        self.rotatingBBox = False
        self.bboxIsSet = False
        
        self.bbox = np.array([QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF(), 
                              QtCore.QPointF(), QtCore.QPointF()])
        self.centerPoint = QtCore.QPointF()
        
        self.prevPoint = None
        self.copiedBBox = np.array([QtCore.QPointF(), QtCore.QPointF(), QtCore.QPointF(), 
                              QtCore.QPointF(), QtCore.QPointF()])
        self.copiedCenter = QtCore.QPointF()
        
        self.tracker = None
        self.tracking = False
        
        self.trackedSprites = []
        self.currentSpriteIdx = -1
        
        self.frameIdx = 0
        self.frameImg = None
        self.overlayImg = QtGui.QImage(QtCore.QSize(100, 100), QtGui.QImage.Format_ARGB32)
        self.showFrame(self.frameIdx)
        
        self.loadTrackedSprites()
        
        self.semanticsToDraw = []
        
        self.setFocus()
        
    def initTracker(self) :
        im0 = cv2.imread(frameLocs[self.frameIdx])
        im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        im_draw = np.copy(im0)
        
        self.tracker = CMT.CMT()
        self.tracker.estimate_scale = True
        self.tracker.estimate_rotation = True
        pause_time = 10
        
        self.tracker.initialise(im_gray0, (self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()), 
                                          (self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()), 
                                          (self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()), 
                                          (self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()))
        
        rot = np.mod(np.arctan2(-(self.bbox[TL_IDX]-self.bbox[TR_IDX]).y(), 
                                 (self.bbox[TL_IDX]-self.bbox[TR_IDX]).x()),2*np.pi)
        
        ## update sprite dict
        if self.currentSpriteIdx < len(self.trackedSprites) and self.currentSpriteIdx >= 0 :
            self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx] = np.array([[self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()], 
                                                                                                [self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()], 
                                                                                                [self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()], 
                                                                                                [self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()]])
            self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx] = np.array([self.centerPoint.x(), self.centerPoint.y()])
            self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_ROTATIONS][self.frameIdx] = np.copy(rot)
            self.trackedSprites[self.currentSpriteIdx][DICT_FRAMES_LOCATIONS][self.frameIdx] = frameLocs[self.frameIdx]
        
    def trackInFrame(self) :
        im = cv2.imread(frameLocs[self.frameIdx])
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_draw = np.copy(im)
    
        tic = time.time()
        self.tracker.process_frame(im_gray)
        toc = time.time()
    
        # Display results
    
        # Draw updated estimate
        if self.tracker.has_result:
            ## update bbox
            self.bbox[TL_IDX].setX(self.tracker.tl[0])
            self.bbox[TL_IDX].setY(self.tracker.tl[1])
            self.bbox[TR_IDX].setX(self.tracker.tr[0])
            self.bbox[TR_IDX].setY(self.tracker.tr[1])
            self.bbox[BR_IDX].setX(self.tracker.br[0])
            self.bbox[BR_IDX].setY(self.tracker.br[1])
            self.bbox[BL_IDX].setX(self.tracker.bl[0])
            self.bbox[BL_IDX].setY(self.tracker.bl[1])
            self.bbox[-1] = self.bbox[TL_IDX]
            
            ## update center point. NOTE: bbox center point is != self.tracker.center(= center of feature points)
            minX = np.min((self.bbox[TL_IDX].x(), self.bbox[TR_IDX].x(), self.bbox[BR_IDX].x(), self.bbox[BL_IDX].x()))
            maxX = np.max((self.bbox[TL_IDX].x(), self.bbox[TR_IDX].x(), self.bbox[BR_IDX].x(), self.bbox[BL_IDX].x()))
            minY = np.min((self.bbox[TL_IDX].y(), self.bbox[TR_IDX].y(), self.bbox[BR_IDX].y(), self.bbox[BL_IDX].y()))
            maxY = np.max((self.bbox[TL_IDX].y(), self.bbox[TR_IDX].y(), self.bbox[BR_IDX].y(), self.bbox[BL_IDX].y()))
            self.centerPoint.setX(minX + (maxX - minX)/2.0)
            self.centerPoint.setY(minY + (maxY - minY)/2.0)
            
            ## compute rotation in radians
            rot = np.mod(np.arctan2(-(self.bbox[TL_IDX]-self.bbox[TR_IDX]).y(), 
                                     (self.bbox[TL_IDX]-self.bbox[TR_IDX]).x()),2*np.pi)
#             print "rotation", rot*180.0/np.pi
            
            ## draw
            if self.drawOverlay() :
                self.frameLabel.setOverlay(self.overlayImg)
            
            ## update sprite dict
            if self.currentSpriteIdx < len(self.trackedSprites) and self.currentSpriteIdx >= 0 :
                self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx] = np.array([[self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()], 
                                                                                                    [self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()], 
                                                                                                    [self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()], 
                                                                                                    [self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()]])
                self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx] = np.array([self.centerPoint.x(), self.centerPoint.y()])
                self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_ROTATIONS][self.frameIdx] = rot
                self.trackedSprites[self.currentSpriteIdx][DICT_FRAMES_LOCATIONS][self.frameIdx] = frameLocs[self.frameIdx]
                
        else :
            print "tracker failed!!!"
        
        
    def trackInVideo(self, goForward) :        
        if goForward :
            self.frameIdxSpinBox.setValue(self.frameIdx+1)
        else :
            self.frameIdxSpinBox.setValue(self.frameIdx-1)
        
        while self.frameIdx >= 0 and self.frameIdx < numOfFrames and self.tracking :
            self.trackInFrame()
        
            # Advance frame number
            if goForward :
                self.frameIdxSpinBox.setValue(self.frameIdx+1)
            else :
                self.frameIdxSpinBox.setValue(self.frameIdx-1)
        
#             print '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(self.tracker.center[0], 
#                             self.tracker.center[1], self.tracker.scale_estimate, self.tracker.active_keypoints.shape[0], 1000 * (toc - tic), self.frameIdx)
            
            QtCore.QCoreApplication.processEvents()
    
    def showFrame(self, idx) :
        self.frameIdx = idx
        ## HACK ##
#         im = np.ascontiguousarray(Image.open((dataPath+dataSet+formatString).format(self.frameIdx+1)))
        im = np.ascontiguousarray(Image.open(frameLocs[self.frameIdx]))
        self.frameImg = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        
        self.frameLabel.setFixedSize(self.frameImg.width(), self.frameImg.height())
        self.frameLabel.setImage(self.frameImg)
        
        self.frameInfo.setText(frameLocs[self.frameIdx])
        
        if self.currentSpriteIdx < len(self.trackedSprites) and self.currentSpriteIdx >= 0 :
            ## set self.bbox to bbox computed for current frame if it exists
            if not self.tracking :
                if self.frameIdx in self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES].keys() :
                    self.bbox[TL_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TL_IDX, 0])
                    self.bbox[TL_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TL_IDX, 1])
                    self.bbox[TR_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TR_IDX, 0])
                    self.bbox[TR_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][TR_IDX, 1])
                    self.bbox[BR_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BR_IDX, 0])
                    self.bbox[BR_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BR_IDX, 1])
                    self.bbox[BL_IDX].setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BL_IDX, 0])
                    self.bbox[BL_IDX].setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx][BL_IDX, 1])
                    self.bbox[-1] = self.bbox[TL_IDX]
                    
                    self.centerPoint.setX(self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx][0])
                    self.centerPoint.setY(self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx][1])
                    
                    if self.drawOverlay(False) :
                        self.frameLabel.setOverlay(self.overlayImg)
                        
                    self.bboxIsSet = True
                else :
                    if self.drawOverlay(False, False, False) :
                        self.frameLabel.setOverlay(self.overlayImg)
            
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
    
    def createNewSprite(self) :
        spriteName, status = QtGui.QInputDialog.getText(self, "Create New Tracked Sprite", "Please name the new tracked sprite")
        if status :
            proceed = True
            for i in xrange(len(self.trackedSprites)) :
                if spriteName == self.trackedSprites[i][DICT_SPRITE_NAME] :
                    proceed = QtGui.QMessageBox.question(self, 'Override Tracked Sprite',
                                    "A tracked sprite named \"" + spriteName + "\" already exists.\nDo you want to override?", 
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
                    if proceed :
                        del self.trackedSprites[i]
                    break
            if proceed :
                print "adding sprite:", spriteName
                self.trackedSprites.append({
                                               DICT_SPRITE_NAME:spriteName,
                                               # DICT_BBOX_AFFINES:sio.loadmat("../ASLA tracker/result/havana_red_car_plusrot/Result/result.mat")['result'],
                                               DICT_BBOXES:{},
                                               DICT_BBOX_CENTERS:{},
                                               DICT_BBOX_ROTATIONS:{},
                                               DICT_FRAMES_LOCATIONS:{}
                                           })
#                 self.currentSpriteIdx = 
                self.setSemanticsToDraw()
                self.setSpriteList()
                self.spriteListTable.selectRow(len(self.trackedSprites)-1)
                self.showFrame(self.frameIdx)
                sys.stdout.flush()
            
        self.setFocus()
        
    def setSemanticsToDraw(self) :
        if len(self.trackedSprites) > 0  :
            self.semanticsToDraw = []
            for i in xrange(0, len(self.trackedSprites)):
                if DICT_MEDIAN_COLOR in self.trackedSprites[i].keys() :
                    col = self.trackedSprites[i][DICT_MEDIAN_COLOR]
                else :
                    col = np.array([0, 0, 0])
                
                if len(self.trackedSprites[i][DICT_BBOXES].keys()) > 0 :
                    self.semanticsToDraw.append({
                                                    DRAW_COLOR:col,
                                                    DRAW_FIRST_FRAME:np.min(self.trackedSprites[i][DICT_BBOXES].keys()),
                                                    DRAW_LAST_FRAME:np.max(self.trackedSprites[i][DICT_BBOXES].keys())
                                                })
                
            self.frameIdxSlider.setSemanticsToDraw(self.semanticsToDraw, numOfFrames)
            
    def changeSprite(self, row) :
        print "changingSprite"
        if len(self.trackedSprites) > row :
            self.currentSpriteIdx = row
            print "sprite: ", self.trackedSprites[self.currentSpriteIdx][DICT_SPRITE_NAME]
            sys.stdout.flush()
            ## go to the first frame that has been tracked
            if len(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES].keys()) > 0 :
                self.frameIdxSpinBox.setValue(np.min(self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES].keys()))
            ## go to the first frame in video
            else :
                self.frameIdxSpinBox.setValue(0)
            
            self.frameIdxSlider.setSelectedSemantics(self.currentSpriteIdx)
            
        self.setFocus()
            
    def loadTrackedSprites(self) :
        ## going to first frame of first sprite if there were no sprites before loading
        goToNewSprite = len(self.trackedSprites) == 0
        for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
            self.trackedSprites.append(np.load(sprite).item())
        
        self.setSpriteList()
        if len(self.trackedSprites) > 0 and goToNewSprite :
            self.spriteListTable.selectRow(0)
            
        self.setSemanticsToDraw()
            
    def setSpriteList(self) :
        self.spriteListTable.setRowCount(0)
        if len(self.trackedSprites) > 0 :
            self.spriteListTable.setRowCount(len(self.trackedSprites))
            
            for i in xrange(0, len(self.trackedSprites)):
                self.spriteListTable.setItem(i, 0, QtGui.QTableWidgetItem(self.trackedSprites[i][DICT_SPRITE_NAME]))
        else :
            self.spriteListTable.setRowCount(1)
            self.spriteListTable.setItem(0, 0, QtGui.QTableWidgetItem("No tracked sprites"))
        
    def deleteCurrentSpriteFrameBBox(self) :
        if self.currentSpriteIdx < len(self.trackedSprites) and self.currentSpriteIdx >= 0 and self.frameIdx in self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES].keys() :
            del self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx]
            del self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx]
            del self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_ROTATIONS][self.frameIdx]
            del self.trackedSprites[self.currentSpriteIdx][DICT_FRAMES_LOCATIONS][self.frameIdx]
            ## refresh current frame so that new bbox gets drawn
            self.showFrame(self.frameIdx)
            
    def setCurrentSpriteFrameBBox(self) :
        if self.currentSpriteIdx < len(self.trackedSprites) and self.currentSpriteIdx >= 0:
            ## compute rotation in radians
            rot = np.mod(np.arctan2(-(self.bbox[TL_IDX]-self.bbox[TR_IDX]).y(), 
                                     (self.bbox[TL_IDX]-self.bbox[TR_IDX]).x()),2*np.pi)
            
            self.trackedSprites[self.currentSpriteIdx][DICT_BBOXES][self.frameIdx] = np.array([[self.bbox[TL_IDX].x(), self.bbox[TL_IDX].y()], 
                                                                                                [self.bbox[TR_IDX].x(), self.bbox[TR_IDX].y()], 
                                                                                                [self.bbox[BR_IDX].x(), self.bbox[BR_IDX].y()], 
                                                                                                [self.bbox[BL_IDX].x(), self.bbox[BL_IDX].y()]])
            self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_CENTERS][self.frameIdx] = np.array([self.centerPoint.x(), self.centerPoint.y()])
            self.trackedSprites[self.currentSpriteIdx][DICT_BBOX_ROTATIONS][self.frameIdx] = rot
            self.trackedSprites[self.currentSpriteIdx][DICT_FRAMES_LOCATIONS][self.frameIdx] = frameLocs[self.frameIdx]
            ## refresh current frame so that new bbox gets drawn
            self.showFrame(self.frameIdx)
        
    def saveTrackedSprites(self) :
        for sprite, i in zip(self.trackedSprites, xrange(len(self.trackedSprites))) :
            np.save(dataPath + dataSet + "/sprite-" + "{0:04}".format(i) + "-" + sprite[DICT_SPRITE_NAME] + ".npy", sprite)
            print sprite[DICT_SPRITE_NAME], "saved"
            sys.stdout.flush()
        
    def isInsideBBox(self, point) :
        bboxPoly = QtGui.QPolygonF()
        for p in self.bbox :
            bboxPoly.append(p)
            
        return bboxPoly.containsPoint(point, QtCore.Qt.WindingFill)
            
    def closeEvent(self, event) :
        print "closing"
        sys.stdout.flush()
        self.saveTrackedSprites()
    
    def mousePressed(self, event):
#         print event.pos()
#         sys.stdout.flush()
        if event.button() == QtCore.Qt.LeftButton :
            if not self.settingBBox and self.isInsideBBox(event.posF()) :
#                 print "movingBBox"
                self.movingBBox = True
                self.bboxIsSet = False
                self.prevPoint = event.posF()
            else :
                if not self.settingBBox :
                    self.bbox[:] = event.posF()
                    self.settingBBox = True
                    self.bboxIsSet = False
                    self.updateBBox()
                else :
                    self.bbox[BR_IDX] = event.posF()
                    self.updateBBox()
                    self.settingBBox = False
                    self.bboxIsSet = True
        elif event.button() == QtCore.Qt.RightButton :
            if not self.settingBBox and self.bboxIsSet :
#                 print "rotatingBBox"
                self.rotatingBBox = True
                self.bboxIsSet = False
                self.prevPoint = event.posF()
                    
                
        sys.stdout.flush()
                
    def mouseMoved(self, event):
        if self.settingBBox :
            self.bbox[BR_IDX] = event.posF()
            self.updateBBox()
        elif self.movingBBox and self.prevPoint != None :
            self.bbox = self.bbox - self.prevPoint + event.posF()
            self.centerPoint = self.centerPoint - self.prevPoint + event.posF()
            self.prevPoint = event.posF()
            if self.drawOverlay(False) :
                self.frameLabel.setOverlay(self.overlayImg)
        elif self.rotatingBBox and self.prevPoint != None :
            deltaR = 0.5*(event.posF().y() - self.prevPoint.y())
            
            t = QtGui.QTransform()            
            t.rotateRadians(deltaR*(np.pi/180.0), QtCore.Qt.Axis.ZAxis)
            self.bbox = np.array(t.map(self.bbox-self.centerPoint))+self.centerPoint
            
            self.prevPoint = event.posF()
            if self.drawOverlay(False) :
                self.frameLabel.setOverlay(self.overlayImg)
            
    def mouseReleased(self, event):
        if self.movingBBox and self.prevPoint != None :
            self.bbox = self.bbox - self.prevPoint + event.posF()
            self.prevPoint = event.posF()
            if self.drawOverlay(False) :
                self.frameLabel.setOverlay(self.overlayImg)
            self.movingBBox = False
            self.bboxIsSet = True
        elif self.rotatingBBox and self.prevPoint != None :
            deltaR = 0.5*(event.posF().y() - self.prevPoint.y())
            
            t = QtGui.QTransform()            
            t.rotateRadians(deltaR*(np.pi/180.0), QtCore.Qt.Axis.ZAxis)
            self.bbox = np.array(t.map(self.bbox-self.centerPoint))+self.centerPoint
            
            self.prevPoint = event.posF()
            if self.drawOverlay(False) :
                self.frameLabel.setOverlay(self.overlayImg)
            self.rotatingBBox = False
            self.bboxIsSet = True
            
    def keyPressEvent(self, e) :
        if self.bboxIsSet and e.key() == QtCore.Qt.Key_Return : ## Track forward
            self.tracking = True
            self.initTracker()
            self.trackInVideo(True)
#             print "tracking forward", self.bbox, self.rotation
            self.setSemanticsToDraw()
            self.tracking = False
        elif self.bboxIsSet and e.key() == QtCore.Qt.Key_Backspace : ## Track backward
            self.tracking = True
            self.initTracker()
            self.trackInVideo(False)
#             print "tracking backward", self.bbox, self.rotation
            self.setSemanticsToDraw()
            self.tracking = False
        elif e.key() == QtCore.Qt.Key_Space : ## stop tracking
            self.setSemanticsToDraw()
            self.tracking = False
        elif self.bboxIsSet and self.tracker != None and e.key() == QtCore.Qt.Key_Right :
            self.tracking = True
            self.frameIdxSpinBox.setValue(self.frameIdx+1)
            self.trackInFrame()
            self.setSemanticsToDraw()
#             print "tracking next frame", self.bbox, self.rotation
            self.tracking = False
        elif self.bboxIsSet and self.tracker != None and e.key() == QtCore.Qt.Key_Left :
            self.tracking = True
            self.frameIdxSpinBox.setValue(self.frameIdx-1)
            self.trackInFrame()
            self.setSemanticsToDraw()
#             print "tracking previous frame", self.bbox, self.rotation
            self.tracking = False
        elif e.key() == QtCore.Qt.Key_Delete :
            self.deleteCurrentSpriteFrameBBox()
        elif e.key() == QtCore.Qt.Key_Enter :
            self.setCurrentSpriteFrameBBox()
        elif e.key() == QtCore.Qt.Key_C and e.modifiers() & QtCore.Qt.Modifier.CTRL :
#             print "copying bbox"
            self.copiedBBox[TL_IDX].setX(self.bbox[TL_IDX].x())
            self.copiedBBox[TL_IDX].setY(self.bbox[TL_IDX].y())
            self.copiedBBox[TR_IDX].setX(self.bbox[TR_IDX].x())
            self.copiedBBox[TR_IDX].setY(self.bbox[TR_IDX].y())
            self.copiedBBox[BR_IDX].setX(self.bbox[BR_IDX].x())
            self.copiedBBox[BR_IDX].setY(self.bbox[BR_IDX].y())
            self.copiedBBox[BL_IDX].setX(self.bbox[BL_IDX].x())
            self.copiedBBox[BL_IDX].setY(self.bbox[BL_IDX].y())
            self.copiedBBox[-1] = self.copiedBBox[TL_IDX]
            
            self.copiedCenter.setX(self.centerPoint.x())
            self.copiedCenter.setY(self.centerPoint.y())
        elif e.key() == QtCore.Qt.Key_V and e.modifiers() & QtCore.Qt.Modifier.CTRL :
            if self.copiedBBox != None and self.bbox != None :
#                 print "pasting bbox"
                self.bbox[TL_IDX].setX(self.copiedBBox[TL_IDX].x())
                self.bbox[TL_IDX].setY(self.copiedBBox[TL_IDX].y())
                self.bbox[TR_IDX].setX(self.copiedBBox[TR_IDX].x())
                self.bbox[TR_IDX].setY(self.copiedBBox[TR_IDX].y())
                self.bbox[BR_IDX].setX(self.copiedBBox[BR_IDX].x())
                self.bbox[BR_IDX].setY(self.copiedBBox[BR_IDX].y())
                self.bbox[BL_IDX].setX(self.copiedBBox[BL_IDX].x())
                self.bbox[BL_IDX].setY(self.copiedBBox[BL_IDX].y())
                self.bbox[-1] = self.bbox[TL_IDX]
                
                self.centerPoint.setX(self.copiedCenter.x())
                self.centerPoint.setY(self.copiedCenter.y())
                
                if self.drawOverlay(False) :
                    self.frameLabel.setOverlay(self.overlayImg)
        elif e.key() == QtCore.Qt.Key_S and e.modifiers() & QtCore.Qt.Modifier.CTRL :
            self.saveTrackedSprites()
            
        sys.stdout.flush()
        
        
    def wheelEvent(self, e) :
        if e.delta() < 0 :
            self.frameIdxSpinBox.setValue(self.frameIdx-1)
        else :
            self.frameIdxSpinBox.setValue(self.frameIdx+1)
        
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
        
        self.frameIdxSlider = SemanticsSlider(QtCore.Qt.Horizontal)
        self.frameIdxSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameIdxSlider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.frameIdxSlider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.frameIdxSlider.setMinimum(0)
        self.frameIdxSlider.setMaximum(numOfFrames-1)
        self.frameIdxSlider.setTickInterval(100)
        self.frameIdxSlider.setSingleStep(1)
        self.frameIdxSlider.setPageStep(100)
        self.frameIdxSlider.installEventFilter(self)
    
        self.frameIdxSpinBox = QtGui.QSpinBox()
        self.frameIdxSpinBox.setRange(0, numOfFrames-1)
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
        
        self.newSpriteButton = QtGui.QPushButton("&New Sprite")
        
        self.deleteCurrentSpriteBBoxButton = QtGui.QPushButton("Delete BBox")
        self.setCurrentSpriteBBoxButton = QtGui.QPushButton("Set BBox")
        
        
        ## SIGNALS ##
        
        self.frameIdxSlider.valueChanged[int].connect(self.frameIdxSpinBox.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.frameIdxSlider.setValue)
        self.frameIdxSpinBox.valueChanged[int].connect(self.showFrame)
        
        self.spriteListTable.currentCellChanged.connect(self.changeSprite)
        self.spriteListTable.cellPressed.connect(self.changeSprite)
        
        self.newSpriteButton.clicked.connect(self.createNewSprite)
        
        self.deleteCurrentSpriteBBoxButton.clicked.connect(self.deleteCurrentSpriteFrameBBox)
        self.setCurrentSpriteBBoxButton.clicked.connect(self.setCurrentSpriteFrameBBox)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.spriteListTable)
        controlsLayout.addWidget(self.newSpriteButton)
        controlsLayout.addWidget(self.deleteCurrentSpriteBBoxButton)
        controlsLayout.addWidget(self.setCurrentSpriteBBoxButton)
        
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

tmp = np.load(dataPath+dataSet+"sprite-0000-blue_car1.npy").item()
print tmp.keys()

# <codecell>

print window.bbox

# <codecell>

print min(window.trackedSprites[0][DICT_BBOXES].keys())

# <codecell>

window.overlayImg.save("lalala", "png")

# <codecell>

tmp = []
tmp.append(np.array([[window.bbox[window.TL_IDX].x(), window.bbox[window.TL_IDX].y()], 
                     [window.bbox[window.TR_IDX].x(), window.bbox[window.TR_IDX].y()], 
                     [window.bbox[window.BR_IDX].x(), window.bbox[window.BR_IDX].y()], 
                     [window.bbox[window.BL_IDX].x(), window.bbox[window.BL_IDX].y()]]))
print np.array(tmp).shape

# <codecell>

tmp = {1:"lalala", 
       2:"asihd", 
       3:"sdlfh"}
print tmp
tmp[5] = "sdkfh"
print tmp
tmp[2] = "lbv"
print tmp
tmp[0] = "akdbf"
print tmp

# <codecell>

img = QtGui.QImage(window.frameImg.size(), QtGui.QImage.Format_ARGB32)
img.fill(QtGui.QColor.fromRgb(255, 255, 255, 255))
painter = QtGui.QPainter(img)
# painter.drawImage(QtCore.QPoint(0, 0), window.frameImg)
    
painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255), 1,
QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
transformedBBox = np.copy(transformedBBox)#window.bbox)

transformation = QtGui.QTransform()
# tl = transformedBBox[window.TL_IDX]
# br = transformedBBox[window.BR_IDX]
# centerPoint = QtCore.QPoint(min((tl.x(), br.x())) + (max((tl.x(), br.x())) - min((tl.x(), br.x())))/2.0, 
#                             min((tl.y(), br.y())) + (max((tl.y(), br.y())) - min((tl.y(), br.y())))/2.0)

transformation.rotateRadians(0.5*(np.pi/180.0), QtCore.Qt.Axis.ZAxis)

transformedBBox = np.array(transformation.map(transformedBBox-centerPoint))+centerPoint

for p1, p2 in zip(transformedBBox[0:-1], transformedBBox[1:]) :
    painter.drawLine(p1, p2)
    
painter.end()
del painter
img.save("0", "png")

# <codecell>

print transformedBBox

# <codecell>

bboxPoly = QtGui.QPolygon()
for p in transformedBBox :
    bboxPoly << p
    
t = QtGui.QTransform()
t = t.translate(-centerPoint.x(), -centerPoint.y())
transformation = transformation.rotateRadians(45.0*(np.pi/180.0), )
transformation.translate(centerPoint.x(), centerPoint.y())

print t
print
print bboxPoly
print
print t.map(bboxPoly)

# <codecell>

trans = np.array([[t.m11(), t.m12(), t.m13()],
                  [t.m21(), t.m22(), t.m23()],
                  [t.m31(), t.m32(), t.m33()]])
print trans*

# <codecell>

t = QtGui.QMatrix4x4()

t.rotate(45.0, 0.0, 0.0, 1.0)
print t*QtCore.QPoint(0.0, 231)

# <codecell>

print transformedBBox
transformation = QtGui.QMatrix()
tl = transformedBBox[window.TL_IDX]
br = transformedBBox[window.BR_IDX]
centerPoint = QtCore.QPoint(min((tl.x(), br.x())) + (max((tl.x(), br.x())) - min((tl.x(), br.x())))/2.0, 
                            min((tl.y(), br.y())) + (max((tl.y(), br.y())) - min((tl.y(), br.y())))/2.0)
print centerPoint
transformation.translate(-centerPoint.x(), -centerPoint.y())
# transformation.rotate(45.0)
# transformation.translate(centerPoint.x(), centerPoint.y())
print transformation
print np.array(transformation.map(transformedBBox))

# <codecell>

print window.bbox - window.pos()
print window.frameLabel.pos()

# <codecell>

np.zeros(5, dtype=QtCore.QPoint)
print np.array([QtCore.QPoint(), QtCore.QPoint()])

