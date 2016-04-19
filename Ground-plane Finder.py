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
<<<<<<< HEAD
import shutil
=======
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

from PIL import Image
from PySide import QtCore, QtGui

import GraphWithValues as gwv

app = QtGui.QApplication(sys.argv)

# <codecell>

<<<<<<< HEAD
DICT_SEQUENCE_NAME = 'semantic_sequence_name'
=======
DICT_SPRITE_NAME = 'sprite_name'
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
<<<<<<< HEAD
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
=======
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SPRITE_IDX = 'sprite_idx' # stores the index in the self.trackedSprites array of the sprite used in the generated sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_MEDIAN_COLOR = 'median_color'
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"

# <codecell>

## load 
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
<<<<<<< HEAD
    print trackedSprites[-1][DICT_SEQUENCE_NAME], DICT_FOOTPRINTS in trackedSprites[-1].keys()

# <codecell>

=======
    print trackedSprites[-1][DICT_SPRITE_NAME], DICT_FOOTPRINTS in trackedSprites[-1].keys()

# <codecell>

im = cv2.cvtColor(cv2.imread(dataPath+dataSet+"median.png", cv2.CV_LOAD_IMAGE_UNCHANGED), cv2.COLOR_BGRA2RGBA)
figure(); imshow(im)
undistorted = cv2.undistort(im, np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]]), np.array([-0.18, -0.18, 0.0, 0.0, 0.0]))
figure(); imshow(undistorted)
blurred = cv2.GaussianBlur(undistorted, (0, 0), 1)
sharpened = cv2.addWeighted(undistorted, 1.5, blurred, -0.5, 0)
figure(); imshow(sharpened)

# <codecell>

# im = cv2.cvtColor(cv2.imread(dataPath+dataSet+"white_bus1-masked-blended/frame-01078.png", cv2.CV_LOAD_IMAGE_UNCHANGED), cv2.COLOR_BGRA2RGBA)
spriteName = trackedSprites[8][DICT_SPRITE_NAME]
spriteFrames = np.sort(glob.glob(dataPath+dataSet+spriteName+"-masked-blended/frame*.png"))
for path in spriteFrames[0:1] :
    im = np.array(Image.open(path))
    rgb = im[:, :, :-1]
    mask = im[:, :, -1]
#     figure(); imshow(rgb)
#     figure(); imshow(mask, interpolation='nearest')
    ## undistort mask
    undistorted = cv2.undistort(mask, np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]]), np.array([-0.18, -0.18, 0.0, 0.0, 0.0]))
    figure(); imshow(undistorted)
    sharpened = cv2.addWeighted(undistorted, 1.5, cv2.GaussianBlur(undistorted, (0, 0), 1), -0.5, 0)
    figure(); imshow(sharpened)
    figure(); imshow(sharpened == 255)

# <codecell>

print projectedPoints.shape

# <codecell>

Image.fromarray(np.array(sharpened, dtype=np.uint8)).save("tralalala.png")

# <codecell>

bgImage = cv2.cvtColor(cv2.imread(dataPath+dataSet+"undistorted_manual_median.png"), cv2.COLOR_BGR2RGB)
# fourCorners = np.array([[421, 316], [466, 305], [500, 315], [453, 328]], dtype=float)
fourCorners = np.array([[715.6, 288.5], [953, 331], [877.8, 400.4], [641.5, 337.5]], dtype=float)
figure(); imshow(bgImage)
plot(fourCorners[[0, 1, 2, 3, 0], 0], fourCorners[[0, 1, 2, 3, 0], 1])
hom = cv2.findHomography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float), fourCorners)[0]

gridIdxs = meshgrid(arange(-4, 10, 0.5), arange(-4, 10, 0.5))
xs = np.ndarray.flatten(gridIdxs[0])
ys = np.ndarray.flatten(gridIdxs[1])
gridPoints = np.array([xs, ys, np.ones(len(xs))])

projectedPoints = np.dot(hom, gridPoints)
projectedPoints /= projectedPoints[-1, :]
scatter(projectedPoints[0, :], projectedPoints[1, :], c='r')

# <codecell>

edges = cv2.Canny(bgImage, 50, 200, 3)
# edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# figure(); imshow(edges)
lines = cv2.HoughLinesP(edges, 1, cv2.cv.CV_PI/180.0, 150, 50, 10)
print lines.shape
figure(); imshow(edges)
for i in xrange(lines.shape[1]) :
    plot(lines[0, i, [0, 2]], lines[0, i, [1, 3]])

# <codecell>

## manually defined pairs of points per line
xAxisLines = np.array([[299.3, 488.0, 63.0, 238.7], 
                       [508.7, 307.7, 213.7, 217.3], 
                       [701.3, 286.0, 338.0, 219.7]], dtype=float)
yAxisLines = np.array([[401.0, 461.0, 938.5, 138.0], 
                       [1313.7, 602.0, 1087.7, 162.0]], dtype=float)

figure(); imshow(bgImage)
for line in xAxisLines[0:2, :] :
    plot(line[[0, 2]], line[[1, 3]], c='b')
for line in yAxisLines :
    plot(line[[0, 2]], line[[1, 3]], c='r')
    
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
def line2lineIntersection(line1, line2) :
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if denominator != 0 :
        Px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denominator
        Py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denominator
        return np.array([Px, Py])
    else :
        raise Exception("lines are parallel")

<<<<<<< HEAD
# <codecell>

# im = cv2.cvtColor(cv2.imread(dataPath+dataSet+"median.png", cv2.CV_LOAD_IMAGE_UNCHANGED), cv2.COLOR_BGRA2RGBA)
# figure(); imshow(im)
# undistorted = cv2.undistort(im, np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]]), np.array([-0.18, -0.18, 0.0, 0.0, 0.0]))
# figure(); imshow(undistorted)
# blurred = cv2.GaussianBlur(undistorted, (0, 0), 1)
# sharpened = cv2.addWeighted(undistorted, 1.5, blurred, -0.5, 0)
# figure(); imshow(sharpened)

# <codecell>

# edges = cv2.Canny(bgImage, 50, 200, 3)
# # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# # figure(); imshow(edges)
# lines = cv2.HoughLinesP(edges, 1, cv2.cv.CV_PI/180.0, 150, 50, 10)
# print lines.shape
# figure(); imshow(edges)
# for i in xrange(lines.shape[1]) :
#     plot(lines[0, i, [0, 2]], lines[0, i, [1, 3]])

# <codecell>

# ## manually defined pairs of points per line
# xAxisLines = np.array([[299.3, 488.0, 63.0, 238.7], 
#                        [508.7, 307.7, 213.7, 217.3], 
#                        [701.3, 286.0, 338.0, 219.7]], dtype=float)
# yAxisLines = np.array([[401.0, 461.0, 938.5, 138.0], 
#                        [1313.7, 602.0, 1087.7, 162.0]], dtype=float)

# figure(); imshow(bgImage)
# for line in xAxisLines[0:2, :] :
#     plot(line[[0, 2]], line[[1, 3]], c='b')
# for line in yAxisLines :
#     plot(line[[0, 2]], line[[1, 3]], c='r')
    
# def line2lineIntersection(line1, line2) :
#     x1, y1, x2, y2 = line1
#     x3, y3, x4, y4 = line2
#     denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
#     if denominator != 0 :
#         Px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denominator
#         Py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denominator
#         return np.array([Px, Py])
#     else :
#         raise Exception("lines are parallel")

# intersectionPoints = []        
# for line1 in xAxisLines[0:2, :] :
#     for line2 in yAxisLines :
#         intersectionPoints.append(line2lineIntersection(line1, line2))
#         scatter(intersectionPoints[-1][0], intersectionPoints[-1][1], c='g')
        
# vanishingPointX = line2lineIntersection(xAxisLines[0, :], xAxisLines[1, :])
# scatter(vanishingPointX[0], vanishingPointX[1], c='b')

# vanishingPointY = line2lineIntersection(yAxisLines[0, :], yAxisLines[1, :])
# scatter(vanishingPointY[0], vanishingPointY[1], c='r')

# plot([vanishingPointX[0], vanishingPointY[0]], [vanishingPointX[1], vanishingPointY[1]], c='c')

# <codecell>

# spriteIdx = 8
# xs, ys = np.array([trackedSprites[spriteIdx][DICT_BBOX_CENTERS][i] for i in np.sort(trackedSprites[spriteIdx][DICT_BBOX_CENTERS].keys())]).T
# ## compute bbox areas
# areas = np.zeros((len(trackedSprites[spriteIdx][DICT_BBOXES]),))
# for key, idx in zip(np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys()), xrange(len(areas))) :
#     areas[idx] = np.linalg.norm(trackedSprites[spriteIdx][DICT_BBOXES][key][TL_IDX, :] - trackedSprites[spriteIdx][DICT_BBOXES][key][TR_IDX, :])
#     areas[idx] *= np.linalg.norm(trackedSprites[spriteIdx][DICT_BBOXES][key][TR_IDX, :] - trackedSprites[spriteIdx][DICT_BBOXES][key][BR_IDX, :])

# filteredAreas = spimg.filters.gaussian_filter1d(areas, 30, axis=0)

# smoothedPath = np.array([spimg.filters.gaussian_filter1d(xs, 15, axis=0), 
#                          spimg.filters.gaussian_filter1d(ys, 15, axis=0)]).T#+filteredAreas*0.0005]).T

# # ## adjust ys to account for height of car
# # if True :
# #     interpolated = interpolate_polyline(np.array([xs, ys+filteredAreas*0.0009]).T, 20)
    
# # interpolated = interpolate_polyline(interpolated, 1000)

# figure(); imshow(cv2.cvtColor(cv2.imread(dataPath+dataSet+"median.png"), cv2.COLOR_BGR2RGB))
# plot(xs, ys)
# # plot(interpolated[:, 0], interpolated[:, 1])
# plot(smoothedPath[:, 0], smoothedPath[:, 1])

# undistortedPoints = cv2.undistortPoints(smoothedPath.reshape((1, len(smoothedPath), 2)),#projectedPoints.T[:, 0:2].reshape((1, len(projectedPoints.T), 2)), 
#                                         np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]]), undistortParameters,
#                                         P=np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]]))
# figure(); imshow(bgImage)
# # plot(interpolated[:, 0], interpolated[:, 1])
# plot(undistortedPoints[0, :, 0], undistortedPoints[0, :, 1], 'o-')
=======
intersectionPoints = []        
for line1 in xAxisLines[0:2, :] :
    for line2 in yAxisLines :
        intersectionPoints.append(line2lineIntersection(line1, line2))
        scatter(intersectionPoints[-1][0], intersectionPoints[-1][1], c='g')
        
vanishingPointX = line2lineIntersection(xAxisLines[0, :], xAxisLines[1, :])
scatter(vanishingPointX[0], vanishingPointX[1], c='b')

vanishingPointY = line2lineIntersection(yAxisLines[0, :], yAxisLines[1, :])
scatter(vanishingPointY[0], vanishingPointY[1], c='r')

plot([vanishingPointX[0], vanishingPointY[0]], [vanishingPointX[1], vanishingPointY[1]], c='c')

# <codecell>

fourCorners = np.array(intersectionPoints)[[0, 2, 3, 1], :]
figure(); imshow(bgImage)
plot(fourCorners[[0, 1, 2, 3, 0], 0], fourCorners[[0, 1, 2, 3, 0], 1])
hom = cv2.findHomography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float), fourCorners)[0]

gridIdxs = meshgrid(arange(-0.4, 8, 0.1), arange(-5, 1.2, 0.2))
xs = np.ndarray.flatten(gridIdxs[0])
ys = np.ndarray.flatten(gridIdxs[1])
gridPoints = np.array([xs, ys, np.ones(len(xs))])

projectedPoints = np.dot(hom, gridPoints)
projectedPoints /= projectedPoints[-1, :]
scatter(projectedPoints[0, :], projectedPoints[1, :], c='r')

# <codecell>

undistortParameters = np.array([-0.18, -0.18, 0.0, 0.0, 0.0])
cameraMatrix = np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]])
mapX, mapY = cv2.initUndistortRectifyMap(cameraMatrix, undistortParameters, None, cameraMatrix,
                                         bgImage.shape[0:2][::-1], cv2.CV_32FC1)
# gwv.showCustomGraph(mapX)
# gwv.showCustomGraph(mapY)

# <codecell>

pointsUndistorted = projectedPoints[0:2, np.all((projectedPoints[0, :] > 0,
                                                 projectedPoints[0, :] < bgImage.shape[1],
                                                 projectedPoints[1, :] > 0,
                                                 projectedPoints[1, :] < bgImage.shape[0]), axis=0)]
figure(); imshow(bgImage)
scatter(pointsUndistorted[0, :], pointsUndistorted[1, :])

pointsDistorted = np.array(pointsUndistorted, dtype=int)
figure(); imshow(cv2.cvtColor(cv2.imread(dataPath+dataSet+"median.png"), cv2.COLOR_BGR2RGB))
pointsDistorted = np.array([mapX[pointsDistorted[1, :], pointsDistorted[0, :]], mapY[pointsDistorted[1, :], pointsDistorted[0, :]]])
scatter(pointsDistorted[0, :], pointsDistorted[1, :])

# <codecell>

print projectedPoints.shape

# <codecell>

print gridPoints.shape
print 0.0 - gridPoints[1, :] < 1e-10
print gridPoints[1, 25*len(arange(-0.4, 8, 0.1))]
print gridPoints[:, 2104]
print gridPoints[:, 2524]

# <codecell>

print np.argwhere(np.all((np.abs(gridPoints[0, :]) < 1e-10, np.abs(gridPoints[1, :]) < 1e-10), axis=0))
print np.argwhere(np.all((np.abs(gridPoints[0, :]) < 1e-10, np.abs(gridPoints[1, :]-1.0) < 1e-10), axis=0))
print np.linalg.norm(projectedPoints[:2, 2104]-projectedPoints[:2, 2524])
print np.linalg.norm(undistortedPoints[0, 2104, :2]-undistortedPoints[0, 2524, :2])
print np.linalg.norm(undistortedPoints[0, 2104, :2]-undistortedPoints[0, 2524, :2])*np.linalg.norm(projectedPoints[:2, 2104]-projectedPoints[:2, 2524])

# <codecell>

spriteIdx = 8
xs, ys = np.array([trackedSprites[spriteIdx][DICT_BBOX_CENTERS][i] for i in np.sort(trackedSprites[spriteIdx][DICT_BBOX_CENTERS].keys())]).T
## compute bbox areas
areas = np.zeros((len(trackedSprites[spriteIdx][DICT_BBOXES]),))
for key, idx in zip(np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys()), xrange(len(areas))) :
    areas[idx] = np.linalg.norm(trackedSprites[spriteIdx][DICT_BBOXES][key][TL_IDX, :] - trackedSprites[spriteIdx][DICT_BBOXES][key][TR_IDX, :])
    areas[idx] *= np.linalg.norm(trackedSprites[spriteIdx][DICT_BBOXES][key][TR_IDX, :] - trackedSprites[spriteIdx][DICT_BBOXES][key][BR_IDX, :])

filteredAreas = spimg.filters.gaussian_filter1d(areas, 30, axis=0)

smoothedPath = np.array([spimg.filters.gaussian_filter1d(xs, 15, axis=0), 
                         spimg.filters.gaussian_filter1d(ys, 15, axis=0)]).T#+filteredAreas*0.0005]).T

# ## adjust ys to account for height of car
# if True :
#     interpolated = interpolate_polyline(np.array([xs, ys+filteredAreas*0.0009]).T, 20)
    
# interpolated = interpolate_polyline(interpolated, 1000)

figure(); imshow(cv2.cvtColor(cv2.imread(dataPath+dataSet+"median.png"), cv2.COLOR_BGR2RGB))
plot(xs, ys)
# plot(interpolated[:, 0], interpolated[:, 1])
plot(smoothedPath[:, 0], smoothedPath[:, 1])

undistortedPoints = cv2.undistortPoints(smoothedPath.reshape((1, len(smoothedPath), 2)),#projectedPoints.T[:, 0:2].reshape((1, len(projectedPoints.T), 2)), 
                                        np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]]), undistortParameters,
                                        P=np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]]))
figure(); imshow(bgImage)
# plot(interpolated[:, 0], interpolated[:, 1])
plot(undistortedPoints[0, :, 0], undistortedPoints[0, :, 1], 'o-')

# <codecell>

print undistortedPoints[0, :, :].shape
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

# <codecell>

preloadedSpritePatches = list(np.load(dataPath + dataSet + "preloadedSpritePatches.npy"))

# <codecell>

POINT_SELECTION_RADIUS = 30

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text="", parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.image = None
        self.spritePatch = None
        self.lines = None
        self.selectedPoint = None
        self.intersectionRectanglePoints = None
        self.planeGrid = None
        self.trajectory = None
        self.bboxRectangle = None
        
    def setImage(self, image) : 
        self.image = image.copy()
        self.setMinimumSize(self.image.size())
        self.update()
        
    def setSpritePatch(self, spritePatch) : 
        self.spritePatch = spritePatch
        self.update()
        
    def setSelectedPoint(self, selectedPoint) :
        self.selectedPoint = selectedPoint
        self.update()
        
    def setLines(self, lines) :
        self.lines = lines
        self.update()
    
    def setIntersectionRectangle(self, intersectionRectanglePoints):
        self.intersectionRectanglePoints = intersectionRectanglePoints
        self.update()
        
    def setPlaneGrid(self, planeGrid):
        self.planeGrid = planeGrid
        self.update()
        
    def setTrajectory(self, trajectory):
        self.trajectory = trajectory
        self.update()
        
    def setBBoxRectangle(self, bboxRectangle):
        self.bboxRectangle = bboxRectangle
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

                painter.drawImage(QtCore.QRect(self.spritePatch['top_left_pos'][1]+upperLeft[0], self.spritePatch['top_left_pos'][0]+upperLeft[1],
                                   self.spritePatch['patch_size'][1], self.spritePatch['patch_size'][0]), reconstructedQImage)
            
            ## draw point grid to show depth
            if self.planeGrid != None :
                planeGrid = self.planeGrid + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 128), 5, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                
                for point in planeGrid :
                    painter.drawPoint(QtCore.QPointF(point[0], point[1]))
            
            ## draw lines
            if self.lines != None :
                lines = self.lines + repeat(np.array([[upperLeft[0], upperLeft[1]]]), 2, axis=0).flatten().reshape((1, 4))
                for i in xrange(len(lines)) :
                    
                    ## draw line
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 2, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    
                    painter.drawLine(QtCore.QPointF(lines[i, 0], lines[i, 1]),
                                     QtCore.QPointF(lines[i, 2], lines[i, 3]))
                    
                    ## draw points
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 7, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    
                    painter.drawPoint(QtCore.QPointF(lines[i, 0], lines[i, 1]))
                    painter.drawPoint(QtCore.QPointF(lines[i, 2], lines[i, 3]))
                    
                    ## draw circle around points to know where to click to select
                    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 127), 2, 
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    
                    painter.drawEllipse(QtCore.QPointF(lines[i, 0], lines[i, 1]), POINT_SELECTION_RADIUS, POINT_SELECTION_RADIUS)
                    painter.drawEllipse(QtCore.QPointF(lines[i, 2], lines[i, 3]), POINT_SELECTION_RADIUS, POINT_SELECTION_RADIUS)
                    
            ## draw selected point        
            if self.selectedPoint != None :
                selectedPoint = self.selectedPoint + upperLeft
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 0, 0, 255), 11, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                
                painter.drawPoint(QtCore.QPointF(selectedPoint[0], selectedPoint[1]))
                
            ## draw intersection rectangle
            if self.intersectionRectanglePoints != None :
                intersectionRectanglePoints = self.intersectionRectanglePoints + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 0, 255, 255), 3, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                
                for i, j in zip(intersectionRectanglePoints, intersectionRectanglePoints[np.roll(np.arange(len(intersectionRectanglePoints)), -1), :]) :
                    painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                     QtCore.QPointF(j[0], j[1]))
                    
                            
            ## draw trajectory
            if self.trajectory != None :
                trajectory = self.trajectory + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 255, 0, 255), 2, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                for i, j in zip(trajectory[0:-1, :], trajectory[1:, :]) :
                    
                    painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                     QtCore.QPointF(j[0], j[1]))
                    
            ## draw bbox rectangle
            if self.bboxRectangle != None :
                bboxRectangle = self.bboxRectangle + np.array([upperLeft[0], upperLeft[1]]).reshape((1, 2))
                
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 255, 255), 2, 
                              QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                for i, j in zip(bboxRectangle, bboxRectangle[np.roll(np.arange(len(bboxRectangle)), -1), :]) :
                    
                    painter.drawLine(QtCore.QPointF(i[0], i[1]),
                                     QtCore.QPointF(j[0], j[1]))

# <codecell>

class LineGraph(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(LineGraph, self).__init__(text, parent)
        
        self.polyline = QtGui.QPolygonF()
        self.currentFrame = 0
        self.xPoints = 0
        self.yPoints = 360
        self.transform = QtGui.QMatrix()
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paintLinegraph(painter)
        
    def setYs(self, ys) :
        self.xPoints = ys.shape[0]
#         print ys.shape
#         print ys
        self.polyline = QtGui.QPolygonF()
        if self.xPoints > 0 :
            for i in xrange(0, self.xPoints) :
                self.polyline.append(QtCore.QPointF(np.float(i), self.yPoints - ys[i]))
            
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
            
        if self.xPoints > 0 : #and self.numClasses > 0 :
#             for p in xrange(0, self.numClasses):
#                 ## paint the polyline for current class probabilities
#                 painter.setPen(QtGui.QPen(self.classClrs[p], 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
#                 painter.drawPolyline(self.transform.map(self.polylines[p]))
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255), 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            painter.drawPolyline(self.transform.map(self.polyline))
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

class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        ## parameters changed through UI
        self.spriteIdx = 8
        self.currentFrame = 0
        self.trajectorySmoothness = 15
        self.orientationsSmoothness = 60
        
        self.bboxSize = np.array([180, 60], dtype=float)
        self.unitSquareSize = np.array([[500, 300]])
        self.unitSquarePos = np.array([[4000, 2500]])
        
        self.trajectorySizeDelta = np.array([[0, 0]])
        self.trajectoryPositionDelta = np.array([[0, 0]])
        
        self.distortionParameter = -0.18
        
        self.lines = np.array([[  281.3,   472. ,    66. ,   251.7],
                               [  458.7,   329.7,   191.7,   224.3],
                               [  401. ,   461. ,   905.5,   161. ],
                               [ 1269.7,   595. ,  1113.7,   186. ]], dtype=float)
        
        self.defaultSettings = {'bboxWidthSpinBox':180,
                                'bboxHeightSpinBox':60, 
                                'unitSquareWidthSpinBox':500,
                                'unitSquareHeightSpinBox':300,
                                'unitSquareXSpinBox':4000,
                                'unitSquareYSpinBox':2500,
                                'trajectorySmoothnessSpinBox':15,
                                'orientationsSmoothnessSpinBox':60,
                                'trajectoryWidthSpinBox':0,
                                'trajectoryHeightSpinBox':0,
                                'trajectoryXSpinBox':0,
                                'trajectoryYSpinBox':0,
                                'distortionParameterSpinBox':-0.18, 
                                'lines':np.array([[  281.3,   472. ,    66. ,   251.7],
                                                  [  458.7,   329.7,   191.7,   224.3],
                                                  [  401. ,   461. ,   905.5,   161. ],
                                                  [ 1269.7,   595. ,  1113.7,   186. ]], dtype=float)}
        
        ## create all widgets and layout
        self.createGUI()
        
        self.setWindowTitle("Adjust Ground Plane")
        self.resize(1700, 900)

        ## manually adjusted parameters        
        
        self.topDownScaling = np.eye(2)*0.1
        
        self.cameraMatrix = np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]])
        self.undistortParameters = np.array([self.distortionParameter, self.distortionParameter, 0.0, 0.0, 0.0])
        
        ## plane grid points
        gridIdxs = meshgrid(arange(-8, 1.4, 0.1), arange(-8, 2.6, 0.2))
        xs = np.ndarray.flatten(gridIdxs[0])*self.unitSquareSize[0, 0]+self.unitSquarePos[0, 0]
        ys = np.ndarray.flatten(gridIdxs[1])*self.unitSquareSize[0, 1]+self.unitSquarePos[0, 1]
        self.gridPoints = np.array([xs, ys, np.ones(len(xs))])
        
        ## data computed using the manually adjusted parameters
        self.originalTrajectory = None
        self.undistortedTrajectory = None
        self.topDownTrajectory = None
        self.orientations = None #np.zeros(len(self.topDownTrajectory))
        
        self.originalGridPoints = None
        self.undistortedGridPoints  = None
        
        self.currentFrameTrajectoryPoint = np.array([[0, 0]], dtype=float)
        
        self.rectangleCorners = None
        
        self.homography = np.eye(3)
        
        self.originalBgImage = np.ascontiguousarray(Image.open(dataPath+dataSet+"median.png"))
        self.undistortedBgImage = None
        self.topDownBgImage = None
        
        ## UI bookeeping
        self.movingPoint = None
        self.prevMousePosition = QtCore.QPoint(0, 0)
        
        
        ## update the data and views
<<<<<<< HEAD
        self.updateTrajectory()
        self.updateIntersectionRectangle()
        self.updateData()
=======
#         self.updateData(True, True, True, True, True, True, True)
        
#         self.updateData
#         self.updateViewsBasedOnData()
#         self.updateIntersectionRectangle()
#         self.updateTrajectory()
#         self.changeFrame(self.frameSpinBox.value())
        self.updateTrajectory()
        self.updateIntersectionRectangle()
        self.updateData()
#         self.updateOrientations()
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

        self.loadSettings()
        
        self.setFocus()
            
    def updateData(self) :
        ## update undistorted bg image
        self.undistortedBgImage = np.ascontiguousarray(cv2.undistort(self.originalBgImage, self.cameraMatrix, self.undistortParameters))
        
        ## update undistorted trajectory
        self.undistortedTrajectory = cv2.undistortPoints(self.originalTrajectory.reshape((1, len(self.originalTrajectory), 2)),
                                                         self.cameraMatrix, self.undistortParameters, P=self.cameraMatrix)[0, :, :]
        
        ## update homography (update rectangleCorners in a different method called before this one and after lines have been changed)
        self.homography = cv2.findHomography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)*self.unitSquareSize+self.unitSquarePos, 
                                             self.rectangleCorners[[1, 2, 3, 0], :])[0]
        
        
        ## update grid points for undistorted view
        self.undistortedGridPoints = np.dot(self.homography, self.gridPoints)
        self.undistortedGridPoints /= self.undistortedGridPoints[-1, :]
        self.undistortedGridPoints = self.undistortedGridPoints[0:2, :].T
        
        ## update grid points for original view
        self.originalGridPoints = cv2.undistortPoints(self.undistortedGridPoints.reshape((1, len(self.undistortedGridPoints), 2)),
                                                      self.cameraMatrix, -self.undistortParameters, P=self.cameraMatrix)[0, :, :]
        
        ## update top down background image
        self.topDownBgImage = np.ascontiguousarray(cv2.resize(cv2.warpPerspective(self.undistortedBgImage, np.linalg.inv(self.homography), (5000, 3000)), 
                                                        (0, 0), fx=self.topDownScaling[0, 0], fy=self.topDownScaling[1, 1]))
        
        ## update top down trajectory
        self.topDownTrajectory = np.dot(np.linalg.inv(self.homography), np.array([self.undistortedTrajectory[:, 0], 
                                                                                    self.undistortedTrajectory[:, 1], 
                                                                                    np.ones(len(self.undistortedTrajectory))]))
        self.topDownTrajectory /= self.topDownTrajectory[-1, :]
        self.topDownTrajectory = self.topDownTrajectory[0:2, :].T
        
        self.updateOrientations()
        
        ## update top down intersection rectangle
        self.topDownRectangle = np.dot(np.linalg.inv(self.homography), 
                                       np.array([self.rectangleCorners[:, 0], self.rectangleCorners[:, 1], np.ones(len(self.rectangleCorners))]))
        self.topDownRectangle /= self.topDownRectangle[-1, :]
        self.topDownRectangle = self.topDownRectangle[0:2, :].T
        
        ## update top down bbox
        self.currentFrameTrajectoryPoint = self.topDownTrajectory[self.currentFrame, :]
        self.topDownBBox = np.array([-self.bboxSize/2, [-self.bboxSize[0]/2, self.bboxSize[1]/2], 
                                self.bboxSize/2, [self.bboxSize[0]/2, -self.bboxSize[1]/2]], dtype=float)

        rotation = np.array([[np.cos(self.orientations[self.currentFrame]), -np.sin(self.orientations[self.currentFrame])], 
                             [np.sin(self.orientations[self.currentFrame]), np.cos(self.orientations[self.currentFrame])]])
        self.topDownBBox = np.dot(self.topDownBBox, rotation)
        
        ## update undistorted bbox
        self.undistortedBBox = np.dot(self.homography, np.array([self.topDownBBox[:, 0]+self.currentFrameTrajectoryPoint[0], 
                                                                 self.topDownBBox[:, 1]+self.currentFrameTrajectoryPoint[1], np.ones(len(self.topDownBBox))]))
        self.undistortedBBox /= self.undistortedBBox[-1, :]
        self.undistortedBBox = self.undistortedBBox[0:2, :].T
        
        ## update original bbox
        self.originalBBox = cv2.undistortPoints(self.undistortedBBox.reshape((1, len(self.undistortedBBox), 2)),
                                                self.cameraMatrix, -self.undistortParameters, P=self.cameraMatrix)[0, :, :]
        
        self.updateOriginalView()
        self.updateTopDownView()
        
         
    def mousePressEvent(self, event):
        sizeDiff = (self.originalImageLabel.size() - self.originalImageLabel.image.size())/2
        mousePos = event.pos() - self.originalImageLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
        mousePos = np.array([mousePos.x(), mousePos.y()])
        
        for l in xrange(len(self.lines)) :
            if np.sqrt(np.sum((self.lines[l, 0:2]-mousePos)**2)) < POINT_SELECTION_RADIUS :
                self.movingPoint = l*2
            
            if np.sqrt(np.sum((self.lines[l, 2:]-mousePos)**2)) < POINT_SELECTION_RADIUS :
                self.movingPoint = l*2+1
                
        self.prevMousePosition = event.pos()            
        
    def mouseReleaseEvent(self, event) :
        self.movingPoint = None
        self.prevMousePosition = QtCore.QPoint(0, 0)
        
        self.originalImageLabel.setSelectedPoint(None)
        
    def mouseMoveEvent(self, event) :
        if self.movingPoint != None :
            sizeDiff = (self.originalImageLabel.size() - self.originalImageLabel.image.size())/2
            mousePos = event.pos() - self.originalImageLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
            if (event.x() >= 0 and event.y() >= 0 and 
                    event.x() < self.originalImageLabel.width() and 
                    event.y() < self.originalImageLabel.height()) :
                
                deltaMove = event.pos() - self.prevMousePosition
                
                pointRow = int(np.floor(self.movingPoint/2.0))
                pointCol = np.mod(self.movingPoint, 2)*2
                
                self.lines[pointRow, pointCol] += deltaMove.x()
                self.lines[pointRow, pointCol+1] += deltaMove.y()
                
                self.originalImageLabel.setSelectedPoint(self.lines[pointRow, pointCol:pointCol+2])
                
                self.updateIntersectionRectangle()
                self.updateData()
#                 self.updateData(True, True, True, True, True, False, True)
        
            self.prevMousePosition = event.pos()
    
    def updateIntersectionRectangle(self) :
        intersectionPoints = []
        for line1 in self.lines[0:2, :] :
            for line2 in self.lines[2:, :] :
                intersectionPoints.append(line2lineIntersection(line1, line2))
                
        self.rectangleCorners = np.array(intersectionPoints)[[0, 2, 3, 1], :]
        
                
    def updateTrajectory(self) :
        xs, ys = np.array([trackedSprites[self.spriteIdx][DICT_BBOX_CENTERS][i] for i in np.sort(trackedSprites[self.spriteIdx][DICT_BBOX_CENTERS].keys())]).T

        smoothedPath = np.array([xs, ys]).T
        
        ## find topLeft of path bbox
        trajTopLeft = np.array([[np.min(smoothedPath[:, 0]), np.min(smoothedPath[:, 1])]])
        trajSize = np.array([[np.max(smoothedPath[:, 0]), np.max(smoothedPath[:, 1])]]) - trajTopLeft
        ## move to origin
        smoothedPath = smoothedPath - trajTopLeft
        ## resize
        sizeRatio = (trajSize+self.trajectorySizeDelta)/trajSize
        smoothedPath = np.dot(smoothedPath, np.array([[sizeRatio[0, 0], 0], [0, sizeRatio[0, 1]]]))
        ## translate by delta
        smoothedPath = smoothedPath + self.trajectoryPositionDelta
        ## move back to original top left
        smoothedPath = smoothedPath + trajTopLeft
        
        ## now do the filtering
        smoothedPath = np.array([spimg.filters.gaussian_filter1d(smoothedPath[:, 0], self.trajectorySmoothness, axis=0), 
                                 spimg.filters.gaussian_filter1d(smoothedPath[:, 1], self.trajectorySmoothness, axis=0)]).T#+filteredAreas*0.0005]).T
        
        ## reinitialize the trajectory
        self.originalTrajectory = smoothedPath
            
        
    def updateOrientations(self) :
        if self.topDownTrajectory != None :
            self.orientations = np.zeros(len(self.topDownTrajectory))
            
            tmp = self.topDownTrajectory[1:, :] - self.topDownTrajectory[0:-1, :]
            tmp /= np.linalg.norm(tmp, axis=-1).reshape((len(tmp), 1))
            self.orientations[:-1] = np.arctan2(tmp[:, 0], tmp[:, 1])

            ## deal with last one
            tmp = -(self.topDownTrajectory[-1, :] - self.topDownTrajectory[-2, :])
            tmp /= np.linalg.norm(tmp, axis=-1)
            self.orientations[-1] = np.arctan2(tmp[0], tmp[1])

            self.orientations = spimg.filters.gaussian_filter1d(np.pi/2 +self.orientations, self.orientationsSmoothness, axis=0)

            self.orientationsGraph.setYs(np.mod(360.0+self.orientations*180.0/np.pi, 360.0))
        
    def updateOriginalView(self) :
        
        ## show the UI needed to adjust the straight lines and intersection rectangle
        if self.doShowUndistortedCheckBox.isChecked() :
            ## don't show the sprite patch
            self.originalImageLabel.setSpritePatch(None)
            
            ## show the undistorted bg image
            if self.undistortedBgImage != None :
                qim = QtGui.QImage(self.undistortedBgImage.data, self.undistortedBgImage.shape[1], 
                                   self.undistortedBgImage.shape[0], self.undistortedBgImage.strides[0], QtGui.QImage.Format_RGB888);
                self.originalImageLabel.setImage(qim)
            
            ## show the intersection rectangle
            if self.rectangleCorners != None :
                self.originalImageLabel.setIntersectionRectangle(self.rectangleCorners)

            ## show the undistorted trajectory
            if self.undistortedTrajectory != None :
                self.originalImageLabel.setTrajectory(self.undistortedTrajectory)
            
            ## show the undistorted grid
            if self.undistortedGridPoints != None :
                self.originalImageLabel.setPlaneGrid(self.undistortedGridPoints)
                
            ## show the undistorted bbox
            if self.undistortedBBox != None :
                self.originalImageLabel.setBBoxRectangle(self.undistortedBBox)
            
            ## show the user defined straight lines
            self.originalImageLabel.setLines(self.lines)
            
        ## show the ultimate output from all of this 
        else :
            ## get the sprite patch and give it to original image label
            self.originalImageLabel.setSpritePatch(preloadedSpritePatches[self.spriteIdx][self.currentFrame])
            
            ## show the distorted bg image
            qim = QtGui.QImage(self.originalBgImage.data, self.originalBgImage.shape[1], 
                               self.originalBgImage.shape[0], self.originalBgImage.strides[0], QtGui.QImage.Format_RGB888);
            self.originalImageLabel.setImage(qim)
            
            ## do not show the intersection rectangle
            self.originalImageLabel.setIntersectionRectangle(None)

            ## show the distorted trajectory
            if self.originalTrajectory != None :
                self.originalImageLabel.setTrajectory(self.originalTrajectory)
                
            ## show the grid in the original (distorted) space
            if self.originalGridPoints != None :
                self.originalImageLabel.setPlaneGrid(self.originalGridPoints)
            
            ## show the distorted bbox
            if self.originalBBox != None :
                self.originalImageLabel.setBBoxRectangle(self.originalBBox)
            
            ## do not show the user defined straight lines
            self.originalImageLabel.setLines(None)
            
        
    def updateTopDownView(self) :
        
        ## show the top down bg image
        if self.topDownBgImage != None :
            qim = QtGui.QImage(self.topDownBgImage.data, self.topDownBgImage.shape[1], 
                               self.topDownBgImage.shape[0], self.topDownBgImage.strides[0], QtGui.QImage.Format_RGB888);
            self.topDownImageLabel.setImage(qim)
        
        ## show top down trajectory
        if self.topDownTrajectory != None :
            self.topDownImageLabel.setTrajectory(np.dot(self.topDownTrajectory, self.topDownScaling))
        
        ## get morphed intersection rectangle
        if self.topDownRectangle != None :
            self.topDownImageLabel.setIntersectionRectangle(np.dot(self.topDownRectangle, self.topDownScaling))
            
        ## show the top down bbox
        if self.topDownBBox != None :
            self.topDownImageLabel.setBBoxRectangle(np.dot(self.topDownBBox + self.currentFrameTrajectoryPoint, self.topDownScaling))
        
        
    def changeFrame(self, idx) :
        self.currentFrame = idx
        self.orientationsGraph.setCurrentFrame(self.currentFrame)
        
        self.updateData()
        
    def changeBBox(self) :
        self.bboxSize = np.array([self.bboxWidthSpinBox.value(), self.bboxHeightSpinBox.value()], dtype=float)
        
        self.updateData()
        
    def changeUnitSquare(self) :
        self.unitSquareSize = np.array([[self.unitSquareWidthSpinBox.value(), self.unitSquareHeightSpinBox.value()]])
        self.unitSquarePos = np.array([[self.unitSquareXSpinBox.value(), self.unitSquareYSpinBox.value()]])
        
        self.updateIntersectionRectangle()
        self.updateData()
        
    def changeTrajectoryDeltas(self) :
        self.trajectorySizeDelta = np.array([[self.trajectoryWidthSpinBox.value(), self.trajectoryHeightSpinBox.value()]])
        self.trajectoryPositionDelta = np.array([[self.trajectoryXSpinBox.value(), self.trajectoryYSpinBox.value()]])
        
        self.updateTrajectory()
        self.updateData()
        
    def changeSprite(self, idx) :
        self.spriteIdx = idx
        
        ## set slider limits
        self.frameSpinBox.setRange(0, len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
        self.frameSlider.setMaximum(len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
        

        self.loadSettings()
        self.updateTrajectory()
        self.updateData()
        
    def changeTrajectoryFiltering(self) :
        self.trajectorySmoothness = self.trajectorySmoothnessSpinBox.value()
        self.orientationsSmoothness = self.orientationsSmoothnessSpinBox.value()
        
        self.updateTrajectory()
        self.updateData()
        
    def changeDistortionParameter(self) :
        self.distortionParameter = self.distortionParameterSpinBox.value()
        self.undistortParameters = np.array([self.distortionParameter, self.distortionParameter, 0.0, 0.0, 0.0])
        
        self.updateData()
        
    def doShowUndistortedChanged(self) :
        self.updateOriginalView()
        
    def saveFootprints(self) :
<<<<<<< HEAD
        self.saveSettings()
        
        spriteName = trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]
=======
        spriteName = trackedSprites[self.spriteIdx][DICT_SPRITE_NAME]
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        sprite = np.load(dataPath+dataSet+"sprite-{0:04d}-".format(self.spriteIdx)+spriteName+".npy").item(0)
        sortedKeys = np.sort(sprite[DICT_BBOXES].keys())
        sprite[DICT_FOOTPRINTS] = {}

        for i in arange(len(sortedKeys)) :

            ## get top down bbox
            currentFrameTrajectoryPoint = self.topDownTrajectory[i, :]
            topDownBBox = np.array([-self.bboxSize/2, [-self.bboxSize[0]/2, self.bboxSize[1]/2], 
                                    self.bboxSize/2, [self.bboxSize[0]/2, -self.bboxSize[1]/2]], dtype=float)

            rotation = np.array([[np.cos(self.orientations[i]), -np.sin(self.orientations[i])], 
                                 [np.sin(self.orientations[i]), np.cos(self.orientations[i])]])
            topDownBBox = np.dot(topDownBBox, rotation)

            ## get undistorted bbox
            undistortedBBox = np.dot(self.homography, np.array([topDownBBox[:, 0]+currentFrameTrajectoryPoint[0], 
                                                                topDownBBox[:, 1]+currentFrameTrajectoryPoint[1], np.ones(len(topDownBBox))]))
            undistortedBBox /= undistortedBBox[-1, :]
            undistortedBBox = undistortedBBox[0:2, :].T

            ## get original bbox (i.e. undistort)
            originalBBox = cv2.undistortPoints(undistortedBBox.reshape((1, len(undistortedBBox), 2)),
                                               self.cameraMatrix, -self.undistortParameters, P=self.cameraMatrix)[0, :, :]


            sprite[DICT_FOOTPRINTS][sortedKeys[i]] = originalBBox
        np.save(dataPath+dataSet+"sprite-{0:04d}-".format(self.spriteIdx)+spriteName+".npy", sprite)

        print "Saved footprints in", dataPath+dataSet+"sprite-{0:04d}-".format(self.spriteIdx)+spriteName+".npy"
        
    def saveSettings(self) :
        
        settings = {DICT_SPRITE_IDX:self.spriteIdx,
<<<<<<< HEAD
                    DICT_SEQUENCE_NAME:trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME],
=======
                    DICT_SPRITE_NAME:trackedSprites[self.spriteIdx][DICT_SPRITE_NAME],
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
                    'bboxWidthSpinBox':self.bboxWidthSpinBox.value(),
                    'bboxHeightSpinBox':self.bboxHeightSpinBox.value(), 
                    'unitSquareWidthSpinBox':self.unitSquareWidthSpinBox.value(),
                    'unitSquareHeightSpinBox':self.unitSquareHeightSpinBox.value(),
                    'unitSquareXSpinBox':self.unitSquareXSpinBox.value(),
                    'unitSquareYSpinBox':self.unitSquareYSpinBox.value(),
                    'trajectorySmoothnessSpinBox':self.trajectorySmoothnessSpinBox.value(),
                    'orientationsSmoothnessSpinBox':self.orientationsSmoothnessSpinBox.value(),
                    'trajectoryWidthSpinBox':self.trajectoryWidthSpinBox.value(),
                    'trajectoryHeightSpinBox':self.trajectoryHeightSpinBox.value(),
                    'trajectoryXSpinBox':self.trajectoryXSpinBox.value(),
                    'trajectoryYSpinBox':self.trajectoryYSpinBox.value(),
                    'distortionParameterSpinBox':self.distortionParameterSpinBox.value(), 
                    'lines':self.lines}
        
<<<<<<< HEAD
        np.save(dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]+"-ground-plane-settings.npy", settings)
        
        print "Saved", dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]+"-ground-plane-settings.npy"
        
    def loadSettings(self) :
        
        if os.path.isfile(dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]+"-ground-plane-settings.npy") :
            settings = np.load(dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]+"-ground-plane-settings.npy").item(0)
=======
        np.save(dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SPRITE_NAME]+"-ground-plane-settings.npy", settings)
        
        print "Saved", dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SPRITE_NAME]+"-ground-plane-settings.npy"
        
    def loadSettings(self) :
        
        if os.path.isfile(dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SPRITE_NAME]+"-ground-plane-settings.npy") :
            settings = np.load(dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SPRITE_NAME]+"-ground-plane-settings.npy").item(0)
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
            
            self.bboxWidthSpinBox.setValue(settings['bboxWidthSpinBox'])
            self.bboxHeightSpinBox.setValue(settings['bboxHeightSpinBox'])
            self.unitSquareWidthSpinBox.setValue(settings['unitSquareWidthSpinBox'])
            self.unitSquareHeightSpinBox.setValue(settings['unitSquareHeightSpinBox'])
            self.unitSquareXSpinBox.setValue(settings['unitSquareXSpinBox'])
            self.unitSquareYSpinBox.setValue(settings['unitSquareYSpinBox'])
            self.trajectorySmoothnessSpinBox.setValue(settings['trajectorySmoothnessSpinBox'])
            self.orientationsSmoothnessSpinBox.setValue(settings['orientationsSmoothnessSpinBox'])
            self.trajectoryWidthSpinBox.setValue(settings['trajectoryWidthSpinBox'])
            self.trajectoryHeightSpinBox.setValue(settings['trajectoryHeightSpinBox'])
            self.trajectoryXSpinBox.setValue(settings['trajectoryXSpinBox'])
            self.trajectoryYSpinBox.setValue(settings['trajectoryYSpinBox'])
            self.distortionParameterSpinBox.setValue(settings['distortionParameterSpinBox'])
            self.lines = settings['lines']
            
<<<<<<< HEAD
            print "Loaded", dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]+"-ground-plane-settings.npy"
=======
            print "Loaded", dataPath+dataSet+trackedSprites[self.spriteIdx][DICT_SPRITE_NAME]+"-ground-plane-settings.npy"
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        else :
            self.bboxWidthSpinBox.setValue(self.defaultSettings['bboxWidthSpinBox'])
            self.bboxHeightSpinBox.setValue(self.defaultSettings['bboxHeightSpinBox'])
            self.unitSquareWidthSpinBox.setValue(self.defaultSettings['unitSquareWidthSpinBox'])
            self.unitSquareHeightSpinBox.setValue(self.defaultSettings['unitSquareHeightSpinBox'])
            self.unitSquareXSpinBox.setValue(self.defaultSettings['unitSquareXSpinBox'])
            self.unitSquareYSpinBox.setValue(self.defaultSettings['unitSquareYSpinBox'])
            self.trajectorySmoothnessSpinBox.setValue(self.defaultSettings['trajectorySmoothnessSpinBox'])
            self.orientationsSmoothnessSpinBox.setValue(self.defaultSettings['orientationsSmoothnessSpinBox'])
            self.trajectoryWidthSpinBox.setValue(self.defaultSettings['trajectoryWidthSpinBox'])
            self.trajectoryHeightSpinBox.setValue(self.defaultSettings['trajectoryHeightSpinBox'])
            self.trajectoryXSpinBox.setValue(self.defaultSettings['trajectoryXSpinBox'])
            self.trajectoryYSpinBox.setValue(self.defaultSettings['trajectoryYSpinBox'])
            self.distortionParameterSpinBox.setValue(self.defaultSettings['distortionParameterSpinBox'])
            self.lines = self.defaultSettings['lines']
            
<<<<<<< HEAD
            print "Loaded defaults for", trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]
=======
            print "Loaded defaults for", trackedSprites[self.spriteIdx][DICT_SPRITE_NAME]
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.originalImageLabel = ImageLabel()
        self.originalImageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.originalImageLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.originalImageInfo = QtGui.QLabel("Original")
        self.originalImageInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.topDownImageLabel = ImageLabel()
        self.topDownImageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.topDownImageLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.topDownImageInfo = QtGui.QLabel("Top Down")
        self.topDownImageInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        
        self.frameSpinBox = QtGui.QSpinBox()
        self.frameSpinBox.setRange(0, len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
        self.frameSpinBox.setSingleStep(1)
        
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
        
        
        controlsGroup = QtGui.QGroupBox("Controls")
        controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
                                             "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        controlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.bboxWidthSpinBox = QtGui.QSpinBox()
        self.bboxWidthSpinBox.setRange(1, 500)
        self.bboxWidthSpinBox.setValue(self.bboxSize[0])
        
        self.bboxHeightSpinBox = QtGui.QSpinBox()
        self.bboxHeightSpinBox.setRange(1, 500)
        self.bboxHeightSpinBox.setValue(self.bboxSize[1])
        
        unitSquareSizeLabel = QtGui.QLabel("Unit square size (w, h)")
        unitSquareSizeLabel.setToolTip("Set the size of the square to map to the intersection rectangle")
        self.unitSquareWidthSpinBox = QtGui.QSpinBox()
        self.unitSquareWidthSpinBox.setRange(1, 2000)
        self.unitSquareWidthSpinBox.setSingleStep(10)
        self.unitSquareWidthSpinBox.setValue(self.unitSquareSize[0, 0])
        
        self.unitSquareHeightSpinBox = QtGui.QSpinBox()
        self.unitSquareHeightSpinBox.setRange(1, 1500)
        self.unitSquareHeightSpinBox.setSingleStep(10)
        self.unitSquareHeightSpinBox.setValue(self.unitSquareSize[0, 1])
        
        unitSquarePosLabel = QtGui.QLabel("Unit square position (x, y)")
        unitSquarePosLabel.setToolTip("Set the position of the unit square in the top down morphed space")
        self.unitSquareXSpinBox = QtGui.QSpinBox()
        self.unitSquareXSpinBox.setRange(1, 10000)
        self.unitSquareXSpinBox.setValue(self.unitSquarePos[0, 0])
        
        self.unitSquareYSpinBox = QtGui.QSpinBox()
        self.unitSquareYSpinBox.setRange(1, 5000)
        self.unitSquareYSpinBox.setValue(self.unitSquarePos[0, 1])
        
        
        self.trajectoryWidthSpinBox = QtGui.QSpinBox()
        self.trajectoryWidthSpinBox.setRange(-500, 500)
        self.trajectoryWidthSpinBox.setValue(self.trajectorySizeDelta[0, 0])
        
        self.trajectoryHeightSpinBox = QtGui.QSpinBox()
        self.trajectoryHeightSpinBox.setRange(-500, 500)
        self.trajectoryHeightSpinBox.setValue(self.trajectorySizeDelta[0, 1])
        
        self.trajectoryXSpinBox = QtGui.QSpinBox()
        self.trajectoryXSpinBox.setRange(-500, 500)
        self.trajectoryXSpinBox.setValue(self.trajectoryPositionDelta[0, 0])
        
        self.trajectoryYSpinBox = QtGui.QSpinBox()
        self.trajectoryYSpinBox.setRange(-500, 500)
        self.trajectoryYSpinBox.setValue(self.trajectoryPositionDelta[0, 1])
        
        
        self.spriteIdxSpinBox = QtGui.QSpinBox()
        self.spriteIdxSpinBox.setRange(0, len(trackedSprites)-1)
        self.spriteIdxSpinBox.setValue(self.spriteIdx)
        
        
        self.trajectorySmoothnessSpinBox = QtGui.QSpinBox()
        self.trajectorySmoothnessSpinBox.setRange(1, 200)
        self.trajectorySmoothnessSpinBox.setValue(self.trajectorySmoothness)
        
        self.orientationsSmoothnessSpinBox = QtGui.QSpinBox()
        self.orientationsSmoothnessSpinBox.setRange(1, 200)
        self.orientationsSmoothnessSpinBox.setValue(self.orientationsSmoothness)
        
        
        self.orientationsGraph = LineGraph("Trajectory orientations")
        self.orientationsGraph.setMinimumHeight(150)
        self.orientationsGraph.setAlignment(QtCore.Qt.AlignCenter)
        self.orientationsGraph.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        
        
        self.distortionParameterSpinBox = QtGui.QDoubleSpinBox()
        self.distortionParameterSpinBox.setRange(-1.0, 1.0)
        self.distortionParameterSpinBox.setSingleStep(0.01)
        self.distortionParameterSpinBox.setValue(self.distortionParameter)
        
        
        self.doShowUndistortedCheckBox = QtGui.QCheckBox("")
        self.doShowUndistortedCheckBox.setChecked(True)
        
        
        self.saveSettingsForSpriteButton = QtGui.QPushButton("&Save settings")
        
        
        self.saveFootprintsForSpriteButton = QtGui.QPushButton("Save &footprints")
        
        
        ## SIGNALS ##
        
        self.frameSpinBox.valueChanged[int].connect(self.frameSlider.setValue)
        self.frameSlider.valueChanged[int].connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.changeFrame)
        
        self.bboxWidthSpinBox.valueChanged[int].connect(self.changeBBox)
        self.bboxHeightSpinBox.valueChanged[int].connect(self.changeBBox)
        
        self.unitSquareWidthSpinBox.valueChanged[int].connect(self.changeUnitSquare)
        self.unitSquareHeightSpinBox.valueChanged[int].connect(self.changeUnitSquare)
        self.unitSquareXSpinBox.valueChanged[int].connect(self.changeUnitSquare)
        self.unitSquareYSpinBox.valueChanged[int].connect(self.changeUnitSquare)
        
        
        self.trajectoryWidthSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
        self.trajectoryHeightSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
        self.trajectoryXSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
        self.trajectoryYSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
        
        self.spriteIdxSpinBox.valueChanged[int].connect(self.changeSprite)
        
        self.trajectorySmoothnessSpinBox.valueChanged[int].connect(self.changeTrajectoryFiltering)
        self.orientationsSmoothnessSpinBox.valueChanged[int].connect(self.changeTrajectoryFiltering)
        
        self.distortionParameterSpinBox.valueChanged[float].connect(self.changeDistortionParameter)
        
        
        self.doShowUndistortedCheckBox.stateChanged.connect(self.doShowUndistortedChanged)
        
        self.saveSettingsForSpriteButton.clicked.connect(self.saveSettings)
        
        self.saveFootprintsForSpriteButton.clicked.connect(self.saveFootprints)
        
        
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
        controlsLayout.addWidget(QtGui.QLabel("Sprite box size (w, h)"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.bboxWidthSpinBox, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.bboxHeightSpinBox, 0, 2, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(unitSquareSizeLabel, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.unitSquareWidthSpinBox, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.unitSquareHeightSpinBox, 1, 2, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(unitSquarePosLabel, 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.unitSquareXSpinBox, 2, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.unitSquareYSpinBox, 2, 2, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Sprite"), 3, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.spriteIdxSpinBox, 3, 1, 1, 2, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Trajectory Smoothness"), 4, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.trajectorySmoothnessSpinBox, 4, 1, 1, 2, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Orientations Smoothness"), 5, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.orientationsSmoothnessSpinBox, 5, 1, 1, 2, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Trajectory size delta (w, h)"), 6, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.trajectoryWidthSpinBox, 6, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.trajectoryHeightSpinBox, 6, 2, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Trajectory position delta (x, y)"), 7, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.trajectoryXSpinBox, 7, 1, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.trajectoryYSpinBox, 7, 2, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Distortion Parameter"), 8, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.distortionParameterSpinBox, 8, 1, 1, 2, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(QtGui.QLabel("Show Undistorted"), 9, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.doShowUndistortedCheckBox, 9, 1, 1, 2, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.saveSettingsForSpriteButton, 10, 0, 1, 1, QtCore.Qt.AlignLeft)
        controlsLayout.addWidget(self.saveFootprintsForSpriteButton, 10, 1, 1, 2, QtCore.Qt.AlignLeft)
        controlsGroup.setLayout(controlsLayout)
        
        topDownImageLayout = QtGui.QVBoxLayout()
        topDownImageLayout.addStretch()
        topDownImageLayout.addWidget(self.topDownImageLabel)
        topDownImageLayout.addWidget(self.topDownImageInfo)
        topDownImageLayout.addStretch()
        topDownImageLayout.addWidget(self.orientationsGraph)
        topDownImageLayout.addWidget(controlsGroup)
        topDownImageLayout.addStretch()
        
        mainLayout.addLayout(originalImageLayout)
        mainLayout.addLayout(topDownImageLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

<<<<<<< HEAD
distances = np.concatenate((np.linalg.norm(window.topDownTrajectory[1:, :] - window.topDownTrajectory[0:-1, :], axis=1), [0]))
# directions = (window.topDownTrajectory[1:, :] - window.topDownTrajectory[0:-1, :])/distances[0:-1].reshape((len(distances)-1, 1))

figure()
ax = gca()
scatter(window.topDownTrajectory[:, 0], window.topDownTrajectory[:, 1], c=cm.jet(distances), edgecolor='none', marker='.')
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_aspect('equal')

# <codecell>

print "max distance", np.max(distances)
print
desiredDistance = 0.5 ## distance to travel between frames
cumDistances = np.cumsum(distances)
print cumDistances.shape

corrected = True

resampledTopDownTrajectoryIdxs = [0]
while (len(resampledTopDownTrajectoryIdxs)-1)*desiredDistance < cumDistances[-1] :
    idx = len(np.argwhere(cumDistances<desiredDistance*len(resampledTopDownTrajectoryIdxs)))-1
    if corrected :
        if (idx < len(cumDistances)-1 and 
            np.abs(desiredDistance*len(resampledTopDownTrajectoryIdxs)-cumDistances[idx]) >
            np.abs(desiredDistance*len(resampledTopDownTrajectoryIdxs)-cumDistances[idx+1])) :
            idx += 1
#     print np.abs(desiredDistance*len(resampledTopDownTrajectoryIdxs)-cumDistances[idx]),
    resampledTopDownTrajectoryIdxs.append(idx)
print
print len(resampledTopDownTrajectoryIdxs)

# <codecell>

resampledTopDownTrajectory = window.topDownTrajectory[resampledTopDownTrajectoryIdxs, :]
resampledDistances = np.concatenate((np.linalg.norm(resampledTopDownTrajectory[1:, :] - resampledTopDownTrajectory[0:-1, :], axis=1), [0]))

if corrected :
    scatter(resampledTopDownTrajectory[:, 0], resampledTopDownTrajectory[:, 1], c='green', edgecolor='none', marker='.')
else :
    figure()
    ax = gca()
    # scatter(resampledTopDownTrajectory[:, 0], resampledTopDownTrajectory[:, 1], c=cm.jet(resampledDistances), edgecolor='none', marker='.')
    scatter(resampledTopDownTrajectory[:, 0], resampledTopDownTrajectory[:, 1], c='none', edgecolor='red', marker='o')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_aspect('equal')

# <codecell>

# ## no interpolation, just closest frame
# sortedKeys = np.sort(trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS].keys())
# bgImage = np.array(Image.open(dataPath+dataSet+"median.png"))
# for i in xrange(len(resampledTopDownTrajectoryIdxs)) :
#     frameName = trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[resampledTopDownTrajectoryIdxs[i]]].split('/')[-1]
# #     shutil.copyfile(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+"-masked-blended/"+frameName, 
# #                     "/media/ilisescu/Data1/hello4/frame-{0:05d}.png".format(i+1))
#     currentFrame = np.array(Image.open(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+"-masked-blended/"+frameName))
#     idxs = np.argwhere(currentFrame[:, :, -1] != 0)
#     img = np.copy(bgImage)
#     img[idxs[:, 0], idxs[:, 1], :] = currentFrame[idxs[:, 0], idxs[:, 1], :-1]
#     Image.fromarray(np.array(img, dtype=np.uint8)).save("/media/ilisescu/Data1/hello1/frame-{0:05d}.png".format(i+1))
#     del img
# #     shutil.copyfile(trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[resampledTopDownTrajectoryIdxs[i]]], 
# #                     "/media/ilisescu/Data1/hello2/frame-{0:05d}.png".format(i))
#     sys.stdout.write('\r' + "Image " + np.string_(i) + " of " + np.string_(len(resampledTopDownTrajectoryIdxs)) + " done")
#     sys.stdout.flush()

# <codecell>

## find what interpolations need to be done
frameIdxDelta = np.array(resampledTopDownTrajectoryIdxs)[1:]-np.array(resampledTopDownTrajectoryIdxs)[:-1]
incrementLocations = np.concatenate(([-1], np.argwhere(frameIdxDelta != 0).flatten()))
numToInterpolate = incrementLocations[1:]-incrementLocations[:-1]-1
idxFramesToInterpolateBetween = incrementLocations+1
## interpolate n (col 2) frames between a (col 0) and b (col 1)
necessaryInterpolations = np.array([np.array(resampledTopDownTrajectoryIdxs)[idxFramesToInterpolateBetween][:-1],
                                    np.array(resampledTopDownTrajectoryIdxs)[idxFramesToInterpolateBetween][1:],
                                    numToInterpolate]).T

## these are the weights for interpolation if I need to create k new frames through interpolation
k = 5
# print np.concatenate((np.arange(0.0, 1.0, 1.0/(2+k-1))[1:], [1.0]))
print necessaryInterpolations[np.argmax(necessaryInterpolations[:, -1]), :]
# print necessaryInterpolations[:100, :]

# <codecell>

# ## does interpolation using alpha blending
# saveLocation = "/media/ilisescu/Data1/bus_interpolated/"
# sortedKeys = np.sort(trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS].keys())
# bgImage = np.array(Image.open(dataPath+dataSet+"median.png"))
# i = 0

# ## merge first frame with bg and save
# frame1Name = trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[necessaryInterpolations[0, 0]]].split('/')[-1]
# frame1 = np.array(Image.open(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+"-masked-blended/"+frame1Name))
# alpha1 = frame1[:, :, -1].reshape((frame1.shape[0], frame1.shape[1], 1))/255.0
# compositeImg = np.array(frame1[:, :, :-1]*alpha1 + bgImage*(1.0-alpha1), dtype=uint8)
# Image.fromarray(compositeImg).save(saveLocation + "frame-{0:05d}.png".format(i+1))
# i += 1

# ## do all the rest
# for pair in necessaryInterpolations :
#     frame1Idx, frame2Idx, k = pair
# #     print frame1Idx, frame2Idx, k
#     frame1Name = trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frame1Idx]].split('/')[-1]
#     frame2Name = trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frame2Idx]].split('/')[-1]
    
#     frame1 = np.array(Image.open(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+"-masked-blended/"+frame1Name))
#     frame2 = np.array(Image.open(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+"-masked-blended/"+frame2Name))
    
#     alpha1 = frame1[:, :, -1].reshape((frame1.shape[0], frame1.shape[1], 1))/255.0
#     alpha2 = frame2[:, :, -1].reshape((frame2.shape[0], frame2.shape[1], 1))/255.0
    
#     for ratio in np.concatenate((np.arange(0.0, 1.0, 1.0/(2+k-1))[1:], [1.0])) :
#         globalAlpha = alpha1*(1.0-ratio) + alpha2*ratio
#         compositeImg = np.array(frame1[:, :, :-1]*alpha1*(1.0-ratio) + frame2[:, :, :-1]*alpha2*ratio + bgImage*(1.0-globalAlpha), dtype=uint8)
#         Image.fromarray(compositeImg).save(saveLocation + "frame-{0:05d}.png".format(i+1))
        
#         i += 1
    
#     sys.stdout.write('\r' + "Image " + np.string_(i) + " of " + np.string_(len(resampledTopDownTrajectoryIdxs)) + " done")
#     sys.stdout.flush()

# <codecell>

def getSpritePatch(im) :
    visiblePixelsGlobalIndices = np.argwhere(im[:, :, -1] != 0)
    topLeftPos = np.min(visiblePixelsGlobalIndices, axis=0)
    patchSize = np.max(visiblePixelsGlobalIndices, axis=0) - topLeftPos + 1
    imgSize = np.array(im.shape[0:2])

#     currentSpriteImages.append({'top_left_pos':topLeft, 'sprite_colors':im[visiblePixels[:, 0], visiblePixels[:, 1], :], 
#                                 'visible_indices': visiblePixels-topLeft, 'patch_size': patchSize})
    
#     topLeftPos = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['top_left_pos'])
#     patchSize = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['patch_size'])
#     visiblePixelsGlobalIndices = preloadedSpritePatches[spriteIdx][frameIdx]['visible_indices']+topLeftPos

    ## when the mask touches the border of the patch there's some weird white halos going on so I enlarge the patch slightly
    ## not sure what happens when the patch goes outside of the bounds of the original image...
    topLeftPos -= 1
    patchSize += 2
    ## make sure we're within bounds
    topLeftPos[np.argwhere(topLeftPos < 0)] = 0
    patchSize[(topLeftPos+patchSize) > imgSize] += (imgSize-(topLeftPos+patchSize))[(topLeftPos+patchSize) > imgSize]

#     if True :
#         spriteData.append(np.ndarray.flatten(np.vstack((topLeftPos[::-1], trackedSprites[spriteIdx][DICT_FOOTPRINTS][sortedKeys[frameIdx]]))))
#     else :
#         spriteData.append(np.ndarray.flatten(np.vstack((topLeftPos[::-1], trackedSprites[spriteIdx][DICT_BBOXES][sortedKeys[frameIdx]]))))

        
#     patch = im[topLeftPos[0]:topLeftPos[0]+patchSize[0], topLeftPos[1]:topLeftPos[1]+patchSize[1], :]

    ## save patch
#     PIL.Image.fromarray(np.uint8(patch)).save(saveLoc+spriteName+"/frame-{:05d}.png".format(frameIdx+1))
    
    return topLeftPos[::-1], im[topLeftPos[0]:topLeftPos[0]+patchSize[0], topLeftPos[1]:topLeftPos[1]+patchSize[1], :]

# <codecell>

## does interpolation using optical flow
saveLocation = dataPath+dataSet+"resampledSprites/"#+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+"-resampled/"
spriteName = trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]
if not os.path.isdir(saveLocation+spriteName) :
    os.mkdir(saveLocation+spriteName)
# frameLocSuffix = "-masked-blended/"
frameLocSuffix = "-maskedFlow-blended/"
sortedKeys = np.sort(trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS].keys())
bgImage = np.array(Image.open(dataPath+dataSet+"median.png"))
spriteData = []

allXs = arange(bgImage.shape[1], dtype=float32).reshape((1, bgImage.shape[1])).repeat(bgImage.shape[0], axis=0)
allYs = arange(bgImage.shape[0], dtype=float32).reshape((bgImage.shape[0], 1)).repeat(bgImage.shape[1], axis=1)

i = 0

## merge first frame with bg and save
frame1Name = trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[necessaryInterpolations[0, 0]]].split('/')[-1]
frame1 = np.array(Image.open(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+frameLocSuffix+frame1Name))

topLeft, patch = getSpritePatch(frame1)
spriteData.append(np.ndarray.flatten(np.vstack((topLeft, trackedSprites[window.spriteIdx][DICT_FOOTPRINTS][sortedKeys[necessaryInterpolations[0, 0]]]))))
Image.fromarray(patch).save(saveLocation + spriteName + "/frame-{0:05d}.png".format(i+1))
# alpha1 = frame1[:, :, -1].reshape((frame1.shape[0], frame1.shape[1], 1))/255.0
# compositeImg = np.array(frame1[:, :, :-1]*alpha1 + bgImage*(1.0-alpha1), dtype=uint8)
# Image.fromarray(compositeImg).save(saveLocation + spriteName + "/frame-{0:05d}.png".format(i+1))
i += 1

## do all the rest
for pair in necessaryInterpolations :
    frame1Idx, frame2Idx, k = pair
#     print frame1Idx, frame2Idx, k
    frame1Name = trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frame1Idx]].split('/')[-1]
    frame2Name = trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frame2Idx]].split('/')[-1]
    
    frame1 = np.array(Image.open(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+frameLocSuffix+frame1Name))
    frame2 = np.array(Image.open(dataPath+dataSet+trackedSprites[window.spriteIdx][DICT_SEQUENCE_NAME]+frameLocSuffix+frame2Name))
    
    alpha1 = frame1[:, :, -1].reshape((frame1.shape[0], frame1.shape[1], 1))/255.0
    alpha2 = frame2[:, :, -1].reshape((frame2.shape[0], frame2.shape[1], 1))/255.0
    
    if k > 0 :
#         flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame2, cv2.COLOR_RGBA2GRAY), 
#                                             cv2.cvtColor(frame1, cv2.COLOR_RGBA2GRAY), 
#                                             0.5, 3, 15, 3, 5, 1.1, 0)
        
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.array(Image.open(trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frame2Idx]])), cv2.COLOR_RGB2GRAY), 
                                            cv2.cvtColor(np.array(Image.open(trackedSprites[window.spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frame1Idx]])), cv2.COLOR_RGB2GRAY), 
                                            0.5, 3, 15, 3, 5, 1.1, 0)
        
        for ratio in np.arange(0.0, 1.0, 1.0/(2+k-1), dtype=float32)[1:] :
#             print ratio
            interpolated = cv2.remap(frame1, flow[:, :, 0]*ratio+allXs, flow[:, :, 1]*ratio+allYs, cv2.INTER_LINEAR)
            alpha = (interpolated[:, :, -1] == np.max(interpolated[:, :, -1])).reshape((bgImage.shape[0], bgImage.shape[1], 1))
            interpolated[:, :, -1] = np.array(alpha[:, :, 0], dtype=int)*255
#             compositeImg = np.array(interpolated[:, :, :-1]*alpha + bgImage*(1.0-alpha), dtype=uint8)
            
# #             globalAlpha = alpha1*(1.0-ratio) + alpha2*ratio
# #             compositeImg = np.array(frame1[:, :, :-1]*alpha1*(1.0-ratio) + frame2[:, :, :-1]*alpha2*ratio + bgImage*(1.0-globalAlpha), dtype=uint8)
#             Image.fromarray(compositeImg).save(saveLocation + spriteName + "/frame-{0:05d}.png".format(i+1))

            topLeft, patch = getSpritePatch(interpolated)
            spriteData.append(np.ndarray.flatten(np.vstack((topLeft, trackedSprites[window.spriteIdx][DICT_FOOTPRINTS][sortedKeys[frame1Idx]]))))
            Image.fromarray(patch).save(saveLocation + spriteName + "/frame-{0:05d}.png".format(i+1))

            i += 1
    ## merge second frame with bg and save
#     compositeImg = np.array(frame2[:, :, :-1]*alpha2 + bgImage*(1.0-alpha2), dtype=uint8)
#     Image.fromarray(compositeImg).save(saveLocation + spriteName + "/frame-{0:05d}.png".format(i+1))
    
    topLeft, patch = getSpritePatch(frame2)
    spriteData.append(np.ndarray.flatten(np.vstack((topLeft, trackedSprites[window.spriteIdx][DICT_FOOTPRINTS][sortedKeys[frame2Idx]]))))
    Image.fromarray(patch).save(saveLocation + spriteName + "/frame-{0:05d}.png".format(i+1))
    
    i += 1
    
    sys.stdout.write('\r' + "Image " + np.string_(i) + " of " + np.string_(len(resampledTopDownTrajectoryIdxs)) + " done")
    sys.stdout.flush()

numpy.savetxt(saveLocation + spriteName + ".csv", numpy.asarray(spriteData), delimiter=",")

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text="", parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.image = None
        self.footprint = None
        
    def setImage(self, image) : 
        self.image = image.copy()
        self.setMinimumSize(self.image.size())
        self.update()
        
    def setFootprint(self, footprint) :
        self.footprint = footprint
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        
        if self.image != None :
            upperLeft = (self.width()/2-self.image.width()/2, self.height()-self.image.height())
            
            ## draw image
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.image)
            
        if self.footprint != None :
            ## draw footprint
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 3, 
                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            ## draw bbox
                    
            for p1, p2 in zip(np.mod(arange(4), 4), np.mod(arange(1, 5), 4)) :
                painter.drawLine(QtCore.QPointF(self.footprint[p1, 0], self.footprint[p1, 1]), 
                                 QtCore.QPointF(self.footprint[p2, 0], self.footprint[p2, 1]))
                
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.setWindowTitle("Resampled Sprites")
        self.resize(1280, 720)
        
        self.playIcon = QtGui.QIcon("play.png")
        self.pauseIcon = QtGui.QIcon("pause.png")
        self.doPlaySequence = False
        
        self.playTimer = QtCore.QTimer(self)
        self.playTimer.setInterval(1000/30)
        self.playTimer.timeout.connect(self.renderOneFrame)
        self.lastRenderTime = time.time()
        self.oldInfoText = ""
        
        self.createGUI()
        
        self.frameIdx = 0
        self.spriteIdx = 8
        self.frameLocs = np.sort(glob.glob(saveLocation + trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]+ "/frame-*.png"))
        self.savedData = numpy.loadtxt(saveLocation+trackedSprites[self.spriteIdx][DICT_SEQUENCE_NAME]+".csv", delimiter=",")
        self.renderOneFrame()
        
        self.setFocus()
        
    def renderOneFrame(self) :
        ## get background image
        if self.frameIdx >= 0 and self.frameIdx < len(self.frameLocs) :
            sprite = cv2.imread(self.frameLocs[self.frameIdx], cv2.CV_LOAD_IMAGE_UNCHANGED)[:, :, [2, 1, 0, 3]]
            im = np.copy(bgImage)
            fgPixels = np.argwhere(sprite[:, :, -1] != 0)
            im[fgPixels[:, 0]+int(self.savedData[self.frameIdx, 1]), fgPixels[:, 1]+int(self.savedData[self.frameIdx, 0]), :] = sprite[fgPixels[:, 0], fgPixels[:, 1], :-1]
            
            self.im = np.ascontiguousarray(im)
            img = QtGui.QImage(self.im.data, self.im.shape[1], self.im.shape[0], self.im.strides[0], QtGui.QImage.Format_RGB888);
            self.frameLabel.setImage(img)
            self.frameLabel.setFootprint(np.array([self.savedData[self.frameIdx, [2, 4, 6, 8, 2]], self.savedData[self.frameIdx, [3, 5, 7, 9, 3]]]).T)

            self.frameInfo.setText("Rendering at " + np.string_(int(1.0/(time.time() - self.lastRenderTime))) + " FPS\n" + 
                                   self.frameLocs[self.frameIdx])
            self.lastRenderTime = time.time()
            self.frameIdx = np.mod(self.frameIdx+1, len(self.frameLocs))
            
    def closeEvent(self, event) :
        print "closing"
        sys.stdout.flush()
    
    def playSequenceButtonPressed(self) :
        if self.doPlaySequence :
            self.doPlaySequence = False
            self.playSequenceButton.setIcon(self.playIcon)
            self.playTimer.stop()
            
            self.frameInfo.setText(self.oldInfoText)
        else :
            self.lastRenderTime = time.time()
            self.doPlaySequence = True
            self.playSequenceButton.setIcon(self.pauseIcon)
            self.playTimer.start()
            
            self.oldInfoText = self.frameInfo.text()
            
    def setRenderFps(self, value) :
        self.playTimer.setInterval(1000/value)
        
    def setBackgroundColor(self) :
        newBgColor = QtGui.QColorDialog.getColor(QtCore.Qt.black, self, "Choose Background Color")
        self.frameLabel.setBackgroundColor(newBgColor)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frameLabel.installEventFilter(self)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 60)
        self.renderFpsSpinBox.setSingleStep(1)
        self.renderFpsSpinBox.setValue(30)
        
        self.playSequenceButton = QtGui.QToolButton()
        self.playSequenceButton.setToolTip("Play Generated Sequence")
        self.playSequenceButton.setCheckable(False)
        self.playSequenceButton.setShortcut(QtGui.QKeySequence("Alt+P"))
        self.playSequenceButton.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        self.playSequenceButton.setIcon(self.playIcon)
        
        self.setBackgroundColorButton = QtGui.QPushButton("&Background Color")
        
        
        ## SIGNALS ##
        
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        
        self.playSequenceButton.clicked.connect(self.playSequenceButtonPressed)
        self.setBackgroundColorButton.clicked.connect(self.setBackgroundColor)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        
        renderingControls = QtGui.QGroupBox("Rendering Controls")
        renderingControls.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        renderingControls.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        renderingControlsLayout = QtGui.QVBoxLayout()
        renderingControlsLayout.addWidget(self.playSequenceButton)
        renderingControlsLayout.addWidget(self.renderFpsSpinBox)
        renderingControlsLayout.addWidget(self.setBackgroundColorButton)
        renderingControls.setLayout(renderingControlsLayout)        
        
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(renderingControls)
        
        frameHLayout = QtGui.QHBoxLayout()
        frameHLayout.addStretch()
        frameHLayout.addWidget(self.frameLabel)
        frameHLayout.addStretch()
        
        frameVLayout = QtGui.QVBoxLayout()
        frameVLayout.addStretch()
        frameVLayout.addLayout(frameHLayout)
        frameVLayout.addStretch()
        frameVLayout.addWidget(self.frameInfo)
        
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameVLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()
=======
trackedSprites[window.spriteIdx].keys()

# <codecell>

figure()
plot(window.orientations)
# plot(spimg.filters.gaussian_filter1d(window.orientations, 60, axis=0))

# <codecell>

#print window.trajectoryInTopDown[]
print arange(10)
print arange(10)[0:-1]
print arange(10)[1:]

def computeOrientations(self) :
    orientations = np.zeros(len(self.trajectoryInTopDown))
    tmp = self.trajectoryInTopDown[1:, :] - self.trajectoryInTopDown[0:-1, :]
    tmp /= np.linalg.norm(tmp, axis=-1).reshape((len(tmp), 1))
    
    ## deal with last one
    orientations[:-1] = np.arctan2(tmp[:, 1], tmp[:, 0])
    tmp = -(self.trajectoryInTopDown[-1, :] - self.trajectoryInTopDown[-2, :])
    tmp /= np.linalg.norm(tmp, axis=-1)
    orientations[-1] = np.arctan2(tmp[1], tmp[0])
    
    orientations = np.pi - orientations
    
computeOrientations(window)

# <codecell>

print window.lines

# <codecell>

import scipy.interpolate as interp

def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i-1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))

# <codecell>

print filteredAreas.shape
print np.array([xs, ys]).T
print np.array([xs, ys+filteredAreas.flatten()*0.1]).T

# <codecell>

print ys.shape

# <codecell>

print interpolatedZAdjusted
print np.array([xs, ys]).T+filteredAreas

# <codecell>

figure()
plot(areas) 
plot(spimg.filters.gaussian_filter1d(areas, 30, axis=0))

# <codecell>

unitSquare = np.array([[0, 0], [500, 0], [500, 300], [0, 300]], dtype=float)+np.array([[4000, 2500]], dtype=float)
## changed the order of the corners to make the bus come from top to bottom
# hom = cv2.findHomography(unitSquare, window.rectangleCorners[[0, 3, 2, 1], :])[0]
hom = cv2.findHomography(unitSquare, window.rectangleCorners[[1, 2, 3, 0], :])[0]
# hom = cv2.findHomography(unitSquare, window.rectangleCorners[[2, 3, 0, 1], :])[0]
morphedBackground = cv2.warpPerspective(bgImage, np.linalg.inv(hom), (5000, 3000))

figure(); 
ax = plt.subplot(111)

ax.imshow(morphedBackground)#, origin='lower')
ax.set_autoscale_on(False)

ax.plot(unitSquare[[0, 1, 2, 3, 0], 0], unitSquare[[0, 1, 2, 3, 0], 1])

trajectoryInTopDown = np.dot(np.linalg.inv(hom), np.array([interpolated[:, 0], interpolated[:, 1], np.ones(len(interpolated))]))
trajectoryInTopDown /= trajectoryInTopDown[-1, :]
trajectoryInTopDown = trajectoryInTopDown[0:2, :].T
ax.plot(trajectoryInTopDown[:, 0], trajectoryInTopDown[:, 1])

# figure()
# ax = plt.subplot(111)
# ax.plot(unitSquare[[0, 1, 2, 3, 0], 0], unitSquare[[0, 1, 2, 3, 0], 1])

# ax.set_xlim([-5000, 5000])
# ax.set_ylim([-5000, 5000])

# <codecell>

print window.rectangleCorners
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

# <codecell>

# class Window(QtGui.QWidget):
#     def __init__(self):
#         super(Window, self).__init__()
        
#         ## parameters cahnged through UI
#         self.spriteIdx = 8
#         self.trajectorySmoothness = 15
#         self.orientationsSmoothness = 60
        
#         self.bboxSize = np.array([180, 60], dtype=float)
#         self.unitSquareSize = np.array([[500, 300]])
#         self.unitSquarePos = np.array([[4000, 2500]])
        
#         self.trajectorySizeDelta = np.array([[0, 0]])
#         self.trajectoryPositionDelta = np.array([[0, 0]])
        
#         self.distortionParameter = -0.18
        
#         ## create all widgets and layout
#         self.createGUI()
        
#         self.setWindowTitle("Adjust Ground Plane")
#         self.resize(1700, 900)
        
# #         self.lines = np.array([[299.3, 488.0, 63.0, 238.7], 
# #                                [508.7, 307.7, 213.7, 217.3], 
# #                                [401.0, 461.0, 938.5, 138.0], 
# #                                [1313.7, 602.0, 1087.7, 162.0]], dtype=float)
# #         ## manually adjusted
# #         self.lines = np.array([[  281.3,   472. ,    46. ,   230.7],
# #                                [  458.7,   329.7,   217.7,   220.3],
# #                                [  401. ,   461. ,   960.5,   124. ],
# #                                [ 1269.7,   595. ,  1069.7,   128. ]], dtype=float)

#         ## manually adjusted parameters
#         self.lines = np.array([[  281.3,   472. ,    66. ,   251.7],
#                                [  458.7,   329.7,   191.7,   224.3],
#                                [  401. ,   461. ,   905.5,   161. ],
#                                [ 1269.7,   595. ,  1113.7,   186. ]], dtype=float)
        
        
#         self.topDownScaling = np.eye(2)*0.1
        
#         self.cameraMatrix = np.array([[1280, 0, 640], [0, 1280, 320], [0, 0, 1]])
#         self.undistortParameters = np.array([self.distortionParameter, self.distortionParameter, 0.0, 0.0, 0.0])
        
#         ## plane grid points
#         gridIdxs = meshgrid(arange(-8, 1.4, 0.1), arange(-8, 2.6, 0.2))
#         xs = np.ndarray.flatten(gridIdxs[0])*self.unitSquareSize[0, 0]+self.unitSquarePos[0, 0]
#         ys = np.ndarray.flatten(gridIdxs[1])*self.unitSquareSize[0, 1]+self.unitSquarePos[0, 1]
#         self.gridPoints = np.array([xs, ys, np.ones(len(xs))])
        
#         ## data computed using the manually adjusted parameters
#         self.originalTrajectory = None
#         self.undistortedTrajectory = None
#         self.topDownTrajectory = None
#         self.orientations = None #np.zeros(len(self.topDownTrajectory))
        
#         self.originalGridPoints = None
#         self.undistortedGridPoints  = None
        
#         self.rectangleCorners = None
        
#         self.homography = np.eye(3)
        
#         self.originalBgImage = np.ascontiguousarray(Image.open(dataPath+dataSet+"median.png"))
#         self.undistortedBgImage = None
        
#         ## UI bookeeping
#         self.movingPoint = None
#         self.prevMousePosition = QtCore.QPoint(0, 0)
        
#         ## update the data and views
#         self.updateData(True, True, True, True, True, True, True)
        
# #         self.updateData
# #         self.updateViewsBasedOnData()
# #         self.updateRectangleHomography()
# #         self.updateTrajectory()
# #         self.changeFrame(self.frameSpinBox.value())
        
        
#         self.setFocus()
            
#     def updateData(self, updateRectangleHomography, updateTrajectory, updateDistortions, updateOrientations, updateOriginalView, updateTopDownView, updateFrameView) :
#         if updateRectangleHomography :
#             self.updateRectangleHomography()
#         if updateTrajectory :
#             self.updateTrajectory()
#         if updateDistortions :
#             self.updateDistortions()
#         if updateOrientations :
#             self.updateOrientations()
#         if updateOriginalView :
#             self.updateOriginalView()
#         if updateTopDownView :
#             self.updateTopDownView()
#         if updateFrameView :
#             self.changeFrame(self.frameSpinBox.value())
         
#     def mousePressEvent(self, event):
#         sizeDiff = (self.originalImageLabel.size() - self.originalImageLabel.image.size())/2
#         mousePos = event.pos() - self.originalImageLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
#         mousePos = np.array([mousePos.x(), mousePos.y()])
        
#         for l in xrange(len(self.lines)) :
#             if np.sqrt(np.sum((self.lines[l, 0:2]-mousePos)**2)) < POINT_SELECTION_RADIUS :
#                 self.movingPoint = l*2
            
#             if np.sqrt(np.sum((self.lines[l, 2:]-mousePos)**2)) < POINT_SELECTION_RADIUS :
#                 self.movingPoint = l*2+1
                
#         self.prevMousePosition = event.pos()            
        
#     def mouseReleaseEvent(self, event) :
#         self.movingPoint = None
#         self.prevMousePosition = QtCore.QPoint(0, 0)
        
#         self.updateData(True, True, True, True, True, False, True)
        
# #         self.originalImageLabel.setLines(self.lines)
# #         self.updateRectangleHomography()
# #         self.updateTrajectory()
# #         self.changeFrame(self.frameSpinBox.value())
        
#     def mouseMoveEvent(self, event) :
#         if self.movingPoint != None :
#             sizeDiff = (self.originalImageLabel.size() - self.originalImageLabel.image.size())/2
#             mousePos = event.pos() - self.originalImageLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
#             if (event.x() >= 0 and event.y() >= 0 and 
#                     event.x() < self.originalImageLabel.width() and 
#                     event.y() < self.originalImageLabel.height()) :
                
#                 deltaMove = event.pos() - self.prevMousePosition
                
#                 pointRow = int(np.floor(self.movingPoint/2.0))
#                 pointCol = np.mod(self.movingPoint, 2)*2
                
#                 self.lines[pointRow, pointCol] += deltaMove.x()
#                 self.lines[pointRow, pointCol+1] += deltaMove.y()
                
#                 self.originalImageLabel.setSelectedPoint(self.lines[pointRow, pointCol:pointCol+2])
                
#                 self.updateData(True, True, True, True, True, False, True)
                
# #                 self.originalImageLabel.setLines(self.lines, self.lines[pointRow, pointCol:pointCol+2])
                
# #                 self.updateRectangleHomography()
# #                 self.updateTrajectory()
# #                 self.changeFrame(self.frameSpinBox.value())
        
#             self.prevMousePosition = event.pos()
    
#     def updateRectangleHomography(self) :
#         intersectionPoints = []
#         for line1 in self.lines[0:2, :] :
#             for line2 in self.lines[2:, :] :
#                 intersectionPoints.append(line2lineIntersection(line1, line2))
                
#         self.rectangleCorners = np.array(intersectionPoints)[[0, 2, 3, 1], :]
        
#         ## get transformation and transform grid points to show in the original image
#         self.homography = cv2.findHomography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)*self.unitSquareSize+self.unitSquarePos, self.rectangleCorners[[1, 2, 3, 0], :])[0]
# #         self.homography = cv2.findHomography(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)*self.unitSquareSize+self.unitSquarePos, self.rectangleCorners[[0, 3, 2, 1], :])[0]
        
# #         ## update the views
# #         self.updateOriginalView()
# #         self.updateTopDownView()
                
#     def updateTrajectory(self) :
#         xs, ys = np.array([trackedSprites[self.spriteIdx][DICT_BBOX_CENTERS][i] for i in np.sort(trackedSprites[self.spriteIdx][DICT_BBOX_CENTERS].keys())]).T
#         ## compute bbox areas
# #         areas = np.zeros((len(trackedSprites[self.spriteIdx][DICT_BBOXES]),))
# #         for key, idx in zip(np.sort(trackedSprites[self.spriteIdx][DICT_BBOXES].keys()), xrange(len(areas))) :
# #             areas[idx] = np.linalg.norm(trackedSprites[self.spriteIdx][DICT_BBOXES][key][TL_IDX, :] - trackedSprites[self.spriteIdx][DICT_BBOXES][key][TR_IDX, :])
# #             areas[idx] *= np.linalg.norm(trackedSprites[self.spriteIdx][DICT_BBOXES][key][TR_IDX, :] - trackedSprites[self.spriteIdx][DICT_BBOXES][key][BR_IDX, :])

# #         filteredAreas = spimg.filters.gaussian_filter1d(areas, 30, axis=0)

#         smoothedPath = np.array([xs, ys]).T#+filteredAreas*0.0005]).T
        
#         ## find topLeft of path bbox
#         trajTopLeft = np.array([[np.min(smoothedPath[:, 0]), np.min(smoothedPath[:, 1])]])
#         trajSize = np.array([[np.max(smoothedPath[:, 0]), np.max(smoothedPath[:, 1])]]) - trajTopLeft
# #         print trajSize, trajSize+self.trajectorySizeDelta
#         ## move to origin
#         smoothedPath = smoothedPath - trajTopLeft
#         ## resize
#         sizeRatio = (trajSize+self.trajectorySizeDelta)/trajSize
#         smoothedPath = np.dot(smoothedPath, np.array([[sizeRatio[0, 0], 0], [0, sizeRatio[0, 1]]]))
#         ## translate by delta
#         smoothedPath = smoothedPath + self.trajectoryPositionDelta
#         ## move back to original top left
#         smoothedPath = smoothedPath + trajTopLeft
        
#         ## now do the filtering
#         smoothedPath = np.array([spimg.filters.gaussian_filter1d(smoothedPath[:, 0], self.trajectorySmoothness, axis=0), 
#                                  spimg.filters.gaussian_filter1d(smoothedPath[:, 1], self.trajectorySmoothness, axis=0)]).T#+filteredAreas*0.0005]).T
        
#         ## reinitialize the trajectory
#         self.originalTrajectory = smoothedPath
# #         ## and get it's top down projection
# #         self.topDownTrajectory = np.dot(np.linalg.inv(self.homography), np.array([self.originalTrajectory[:, 0], self.originalTrajectory[:, 1], np.ones(len(self.originalTrajectory))]))
# #         self.topDownTrajectory /= self.topDownTrajectory[-1, :]
# #         self.topDownTrajectory = self.topDownTrajectory[0:2, :].T
#         ## update the orientations now
# #         self.updateOrientations()
        
# #         ## update the views
# #         self.updateOriginalView()
# #         self.updateTopDownView()

#     def updateDistortions(self) :
        
#         ## update undistorted bg image
#         self.undistortedBgImage = np.ascontiguousarray(cv2.undistort(self.originalBgImage, self.cameraMatrix, self.undistortParameters))
        
#         ## update undistorted trajectory
#         if self.originalTrajectory != None :
#             self.undistortedTrajectory = cv2.undistortPoints(self.originalTrajectory.reshape((1, len(self.originalTrajectory), 2)),
#                                                              self.cameraMatrix, self.undistortParameters, P=self.cameraMatrix)[0, :, :]
            
#             ## and get it's top down projection
#             self.topDownTrajectory = np.dot(np.linalg.inv(self.homography), np.array([self.undistortedTrajectory[:, 0], 
#                                                                                         self.undistortedTrajectory[:, 1], 
#                                                                                         np.ones(len(self.undistortedTrajectory))]))
#             self.topDownTrajectory /= self.topDownTrajectory[-1, :]
#             self.topDownTrajectory = self.topDownTrajectory[0:2, :].T
            
        
#     def updateOrientations(self) :
#         if self.topDownTrajectory != None :
#             self.orientations = np.zeros(len(self.topDownTrajectory))
            
#             tmp = self.topDownTrajectory[1:, :] - self.topDownTrajectory[0:-1, :]
#             tmp /= np.linalg.norm(tmp, axis=-1).reshape((len(tmp), 1))
#             self.orientations[:-1] = np.arctan2(tmp[:, 0], tmp[:, 1])

#             ## deal with last one
#             tmp = -(self.topDownTrajectory[-1, :] - self.topDownTrajectory[-2, :])
#             tmp /= np.linalg.norm(tmp, axis=-1)
#             self.orientations[-1] = np.arctan2(tmp[0], tmp[1])

#             self.orientations = spimg.filters.gaussian_filter1d(np.pi/2 +self.orientations, self.orientationsSmoothness, axis=0)

#             self.orientationsGraph.setYs(np.mod(360.0+self.orientations*180.0/np.pi, 360.0))
        
#     def updateOriginalView(self) :

#         self.undistortedGridPoints = np.dot(self.homography, self.gridPoints)
#         self.undistortedGridPoints /= self.undistortedGridPoints[-1, :]
#         self.undistortedGridPoints = self.undistortedGridPoints[0:2, :].T
        
#         ## show the UI needed to adjust the straight lines and intersection rectangle
#         if self.doSetGroundPlaneCheckBox.isChecked() :
#             ## show the undistorted bg image
#             if self.undistortedBgImage != None :
#                 qim = QtGui.QImage(self.undistortedBgImage.data, self.undistortedBgImage.shape[1], 
#                                    self.undistortedBgImage.shape[0], self.undistortedBgImage.strides[0], QtGui.QImage.Format_RGB888);
#                 self.originalImageLabel.setImage(qim)
            
#             ## show the intersection rectangle
#             if self.rectangleCorners != None :
#                 self.originalImageLabel.setIntersectionRectangle(self.rectangleCorners)

#             ## show the undistorted trajectory
#             if self.undistortedTrajectory != None :
#                 self.originalImageLabel.setTrajectory(self.undistortedTrajectory)
            
#             ## show the undistorted grid
#             if self.undistortedGridPoints != None :
#                 self.originalImageLabel.setPlaneGrid(self.undistortedGridPoints)
            
#             ## show the user defined straight lines
#             self.originalImageLabel.setLines(self.lines)
            
#         ## show the ultimate output from all of this 
#         else :
#             ## show the distorted bg image
#             qim = QtGui.QImage(self.originalBgImage.data, self.originalBgImage.shape[1], 
#                                self.originalBgImage.shape[0], self.originalBgImage.strides[0], QtGui.QImage.Format_RGB888);
#             self.originalImageLabel.setImage(qim)
            
#             ## do not show the intersection rectangle
#             self.originalImageLabel.setIntersectionRectangle(None)

#             ## show the distorted trajectory
#             if self.originalTrajectory != None :
#                 self.originalImageLabel.setTrajectory(self.originalTrajectory)
                
#             ## show the grid in the original (distorted) space
#             self.originalGridPoints = cv2.undistortPoints(self.undistortedGridPoints.reshape((1, len(self.undistortedGridPoints), 2)), 
#                                                                      self.cameraMatrix, -self.undistortParameters, P=self.cameraMatrix)[0, :, :]
#             if self.originalGridPoints != None :
#                 self.originalImageLabel.setPlaneGrid(self.originalGridPoints)
            
#             ## do not show the user defined straight lines
#             self.originalImageLabel.setLines(None)
            
        
#     def updateTopDownView(self) :
        
#         ## get morphed background using the inverse of the homography and show it in the top down label
#         if self.undistortedBgImage != None :
#             morphedBackground = np.ascontiguousarray(cv2.resize(cv2.warpPerspective(self.undistortedBgImage, np.linalg.inv(self.homography), (5000, 3000)), 
#                                                             (0, 0), fx=self.topDownScaling[0, 0], fy=self.topDownScaling[1, 1]))
#             qim = QtGui.QImage(morphedBackground.data, morphedBackground.shape[1], morphedBackground.shape[0], morphedBackground.strides[0], QtGui.QImage.Format_RGB888);
#             self.topDownImageLabel.setImage(qim)
        
#         ## set morphed trajectory
#         if self.topDownTrajectory != None :
#             self.topDownImageLabel.setTrajectory(np.dot(self.topDownTrajectory, self.topDownScaling))
        
#         ## get morphed intersection rectangle
#         if self.rectangleCorners != None :
#             morphedRectangle = np.dot(np.linalg.inv(self.homography), np.array([self.rectangleCorners[:, 0], self.rectangleCorners[:, 1], np.ones(len(self.rectangleCorners))]))
#             morphedRectangle /= morphedRectangle[-1, :]
#             morphedRectangle = np.dot(morphedRectangle[0:2, :].T, self.topDownScaling)
#             self.topDownImageLabel.setIntersectionRectangle(morphedRectangle)
        
        
#     def changeFrame(self, idx) :
#         ## get the sprite patch and give it to original image label
#         self.originalImageLabel.setSpritePatch(preloadedSpritePatches[self.spriteIdx][idx])
#         self.orientationsGraph.setCurrentFrame(idx)
        
#         if self.orientations != None and self.topDownTrajectory != None :
#             bboxCenter = self.topDownTrajectory[idx, :]
#             bboxInTopDown = np.array([-self.bboxSize/2, 
#                                       [-self.bboxSize[0]/2, self.bboxSize[1]/2], 
#                                       self.bboxSize/2, 
#                                       [self.bboxSize[0]/2, -self.bboxSize[1]/2]], dtype=float)
            
#             rotation = np.array([[np.cos(self.orientations[idx]), -np.sin(self.orientations[idx])], [np.sin(self.orientations[idx]), np.cos(self.orientations[idx])]])
#             bboxInTopDown = np.dot(bboxInTopDown, rotation)
            
#             ## bbox in top down aligned to trajectory
#             self.topDownImageLabel.setBBoxRectangle(np.dot(bboxInTopDown+bboxCenter, self.topDownScaling))
            
#             ## bbox in original view morphed using homography
#             bbox = np.dot(self.homography, np.array([bboxInTopDown[:, 0]+bboxCenter[0], bboxInTopDown[:, 1]+bboxCenter[1], np.ones(len(bboxInTopDown))]))
#             bbox /= bbox[-1, :]
#             bbox = bbox[0:2, :].T
#             self.originalImageLabel.setBBoxRectangle(bbox)
        
#     def changeBBox(self) :
#         self.bboxSize = np.array([self.bboxWidthSpinBox.value(), self.bboxHeightSpinBox.value()], dtype=float)
        
# #         self.changeFrame(self.frameSpinBox.value())
        
#     def changeUnitSquare(self) :
#         self.unitSquareSize = np.array([[self.unitSquareWidthSpinBox.value(), self.unitSquareHeightSpinBox.value()]])
#         self.unitSquarePos = np.array([[self.unitSquareXSpinBox.value(), self.unitSquareYSpinBox.value()]])
        
# #         self.updateRectangleHomography()
# #         self.updateTrajectory()
# #         self.changeFrame(self.frameSpinBox.value())
        
#     def changeTrajectoryDeltas(self) :
#         self.trajectorySizeDelta = np.array([[self.trajectoryWidthSpinBox.value(), self.trajectoryHeightSpinBox.value()]])
#         self.trajectoryPositionDelta = np.array([[self.trajectoryXSpinBox.value(), self.trajectoryYSpinBox.value()]])
        
# #         self.updateTrajectory()
# #         self.changeFrame(self.frameSpinBox.value())
        
#     def changeSprite(self, idx) :
#         self.spriteIdx = idx
        
#         ## set slider limits
#         self.frameSpinBox.setRange(0, len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
#         self.frameSlider.setMaximum(len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
        
# #         self.updateTrajectory()
# #         self.changeFrame(self.frameSpinBox.value())
        
#     def changeTrajectoryFiltering(self) :
#         self.trajectorySmoothness = self.trajectorySmoothnessSpinBox.value()
#         self.orientationsSmoothness = self.orientationsSmoothnessSpinBox.value()
        
# #         self.updateTrajectory()
# #         self.changeFrame(self.frameSpinBox.value())
        
#     def changeDistortionParameter(self) :
#         self.distortionParameter = self.distortionParameterSpinBox.value()
#         self.undistortParameters = np.array([self.distortionParameter, self.distortionParameter, 0.0, 0.0, 0.0])
        
        
#         self.updateData(False, False, True, True, True, True, True)
#         ## update undistorted bg image
# #         self.undistortedBgImage = np.ascontiguousarray(cv2.undistort(self.originalBgImage, self.cameraMatrix, self.undistortParameters))
#         ## update undistorted trajectory
        
# #         self.updateViewsBasedOnData()
        
#     def doSetGroundPlaneChanged(self) :
#         print 
        
# #         ## show the undistorted background where a ground plane can be defined
# #         if self.doSetGroundPlaneCheckBox.isChecked() :
# #             self.undistortedBgImage = np.ascontiguousarray(cv2.undistort(self.originalBgImage, self.cameraMatrix, self.undistortParameters))

# #             ## HACK ##
# #             qim = QtGui.QImage(self.undistortedBgImage.data, self.undistortedBgImage.shape[1], 
# #                                self.undistortedBgImage.shape[0], self.undistortedBgImage.strides[0], QtGui.QImage.Format_RGB888);
            
# #             ## show undistorted bg image
# #             self.originalImageLabel.setImage(qim)
# #             ## show user defined straight lines
# #             self.originalImageLabel.setLines(self.lines)
# #             ## show undistorted trajectory
# # #             self.originalImageLabel.setTrajectory(cv2.undistortPoints(self.originalTrajectory.reshape((1, len(self.originalTrajectory), 2)),
# # #                                                                       self.cameraMatrix, self.undistortParameters, P=self.cameraMatrix))
# #         else :
# #             ## HACK ##
# #             qim = QtGui.QImage(self.originalBgImage.data, self.originalBgImage.shape[1], 
# #                                self.originalBgImage.shape[0], self.originalBgImage.strides[0], QtGui.QImage.Format_RGB888);
            
# #             ## show distorted bg image
# #             self.originalImageLabel.setImage(qim)
# #             ## do not show the user defiend lines
# #             self.originalImageLabel.setLines(None)
# #             ## show distorted trajectory
# #             self.originalImageLabel.setTrajectory(self.originalTrajectory)
            
        
#     def createGUI(self) :
        
#         ## WIDGETS ##
        
#         self.originalImageLabel = ImageLabel()
#         self.originalImageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
#         self.originalImageLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
#         self.originalImageInfo = QtGui.QLabel("Original")
#         self.originalImageInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
#         self.topDownImageLabel = ImageLabel()
#         self.topDownImageLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
#         self.topDownImageLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
#         self.topDownImageInfo = QtGui.QLabel("Top Down")
#         self.topDownImageInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        
#         self.frameSpinBox = QtGui.QSpinBox()
#         self.frameSpinBox.setRange(0, len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
#         self.frameSpinBox.setSingleStep(1)
        
#         self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
#         self.frameSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
#         self.frameSlider.setMinimum(0)
#         self.frameSlider.setMaximum(len(trackedSprites[self.spriteIdx][DICT_BBOXES])-1)
        
        
#         controlsGroup = QtGui.QGroupBox("Controls")
#         controlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
#                                              "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
#         controlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
#         self.bboxWidthSpinBox = QtGui.QSpinBox()
#         self.bboxWidthSpinBox.setRange(1, 500)
#         self.bboxWidthSpinBox.setValue(self.bboxSize[0])
        
#         self.bboxHeightSpinBox = QtGui.QSpinBox()
#         self.bboxHeightSpinBox.setRange(1, 500)
#         self.bboxHeightSpinBox.setValue(self.bboxSize[1])
        
#         unitSquareSizeLabel = QtGui.QLabel("Unit square size (w, h)")
#         unitSquareSizeLabel.setToolTip("Set the size of the square to map to the intersection rectangle")
#         self.unitSquareWidthSpinBox = QtGui.QSpinBox()
#         self.unitSquareWidthSpinBox.setRange(1, 2000)
#         self.unitSquareWidthSpinBox.setSingleStep(10)
#         self.unitSquareWidthSpinBox.setValue(self.unitSquareSize[0, 0])
        
#         self.unitSquareHeightSpinBox = QtGui.QSpinBox()
#         self.unitSquareHeightSpinBox.setRange(1, 1500)
#         self.unitSquareHeightSpinBox.setSingleStep(10)
#         self.unitSquareHeightSpinBox.setValue(self.unitSquareSize[0, 1])
        
#         unitSquarePosLabel = QtGui.QLabel("Unit square position (x, y)")
#         unitSquarePosLabel.setToolTip("Set the position of the unit square in the top down morphed space")
#         self.unitSquareXSpinBox = QtGui.QSpinBox()
#         self.unitSquareXSpinBox.setRange(1, 10000)
#         self.unitSquareXSpinBox.setValue(self.unitSquarePos[0, 0])
        
#         self.unitSquareYSpinBox = QtGui.QSpinBox()
#         self.unitSquareYSpinBox.setRange(1, 5000)
#         self.unitSquareYSpinBox.setValue(self.unitSquarePos[0, 1])
        
        
#         self.trajectoryWidthSpinBox = QtGui.QSpinBox()
#         self.trajectoryWidthSpinBox.setRange(-500, 500)
#         self.trajectoryWidthSpinBox.setValue(self.trajectorySizeDelta[0, 0])
        
#         self.trajectoryHeightSpinBox = QtGui.QSpinBox()
#         self.trajectoryHeightSpinBox.setRange(-500, 500)
#         self.trajectoryHeightSpinBox.setValue(self.trajectorySizeDelta[0, 1])
        
#         self.trajectoryXSpinBox = QtGui.QSpinBox()
#         self.trajectoryXSpinBox.setRange(-500, 500)
#         self.trajectoryXSpinBox.setValue(self.trajectoryPositionDelta[0, 0])
        
#         self.trajectoryYSpinBox = QtGui.QSpinBox()
#         self.trajectoryYSpinBox.setRange(-500, 500)
#         self.trajectoryYSpinBox.setValue(self.trajectoryPositionDelta[0, 1])
        
        
#         self.spriteIdxSpinBox = QtGui.QSpinBox()
#         self.spriteIdxSpinBox.setRange(0, len(trackedSprites)-1)
#         self.spriteIdxSpinBox.setValue(self.spriteIdx)
        
        
#         self.trajectorySmoothnessSpinBox = QtGui.QSpinBox()
#         self.trajectorySmoothnessSpinBox.setRange(1, 200)
#         self.trajectorySmoothnessSpinBox.setValue(self.trajectorySmoothness)
        
#         self.orientationsSmoothnessSpinBox = QtGui.QSpinBox()
#         self.orientationsSmoothnessSpinBox.setRange(1, 200)
#         self.orientationsSmoothnessSpinBox.setValue(self.orientationsSmoothness)
        
        
#         self.orientationsGraph = LineGraph("Trajectory orientations")
#         self.orientationsGraph.setMinimumHeight(150)
#         self.orientationsGraph.setAlignment(QtCore.Qt.AlignCenter)
#         self.orientationsGraph.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        
        
#         self.distortionParameterSpinBox = QtGui.QDoubleSpinBox()
#         self.distortionParameterSpinBox.setRange(-1.0, 1.0)
#         self.distortionParameterSpinBox.setSingleStep(0.01)
#         self.distortionParameterSpinBox.setValue(self.distortionParameter)
        
        
#         self.doSetGroundPlaneCheckBox = QtGui.QCheckBox("")
#         self.doSetGroundPlaneCheckBox.setChecked(True)
        
        
#         ## SIGNALS ##
        
#         self.frameSpinBox.valueChanged[int].connect(self.frameSlider.setValue)
#         self.frameSlider.valueChanged[int].connect(self.frameSpinBox.setValue)
#         self.frameSpinBox.valueChanged[int].connect(self.changeFrame)
        
#         self.bboxWidthSpinBox.valueChanged[int].connect(self.changeBBox)
#         self.bboxHeightSpinBox.valueChanged[int].connect(self.changeBBox)
        
#         self.unitSquareWidthSpinBox.valueChanged[int].connect(self.changeUnitSquare)
#         self.unitSquareHeightSpinBox.valueChanged[int].connect(self.changeUnitSquare)
#         self.unitSquareXSpinBox.valueChanged[int].connect(self.changeUnitSquare)
#         self.unitSquareYSpinBox.valueChanged[int].connect(self.changeUnitSquare)
        
        
#         self.trajectoryWidthSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
#         self.trajectoryHeightSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
#         self.trajectoryXSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
#         self.trajectoryYSpinBox.valueChanged[int].connect(self.changeTrajectoryDeltas)
        
#         self.spriteIdxSpinBox.valueChanged[int].connect(self.changeSprite)
        
#         self.trajectorySmoothnessSpinBox.valueChanged[int].connect(self.changeTrajectoryFiltering)
#         self.orientationsSmoothnessSpinBox.valueChanged[int].connect(self.changeTrajectoryFiltering)
        
        
#         self.distortionParameterSpinBox.valueChanged[float].connect(self.changeDistortionParameter)
        
#         self.doSetGroundPlaneCheckBox.stateChanged.connect(self.doSetGroundPlaneChanged)
        
        
#         ## LAYOUTS ##
        
#         mainLayout = QtGui.QHBoxLayout()
        
#         sliderLayout = QtGui.QHBoxLayout()
#         sliderLayout.addWidget(self.frameSlider)
#         sliderLayout.addWidget(self.frameSpinBox)
        
#         originalImageLayout = QtGui.QVBoxLayout()
#         originalImageLayout.addWidget(self.originalImageLabel)
#         originalImageLayout.addWidget(self.originalImageInfo)
#         originalImageLayout.addLayout(sliderLayout)
        
#         controlsLayout = QtGui.QGridLayout()
#         controlsLayout.addWidget(QtGui.QLabel("Sprite box size (w, h)"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.bboxWidthSpinBox, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.bboxHeightSpinBox, 0, 2, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(unitSquareSizeLabel, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.unitSquareWidthSpinBox, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.unitSquareHeightSpinBox, 1, 2, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(unitSquarePosLabel, 2, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.unitSquareXSpinBox, 2, 1, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.unitSquareYSpinBox, 2, 2, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(QtGui.QLabel("Sprite"), 3, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.spriteIdxSpinBox, 3, 1, 1, 2, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(QtGui.QLabel("Trajectory Smoothness"), 4, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.trajectorySmoothnessSpinBox, 4, 1, 1, 2, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(QtGui.QLabel("Orientations Smoothness"), 5, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.orientationsSmoothnessSpinBox, 5, 1, 1, 2, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(QtGui.QLabel("Trajectory size delta (w, h)"), 6, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.trajectoryWidthSpinBox, 6, 1, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.trajectoryHeightSpinBox, 6, 2, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(QtGui.QLabel("Trajectory position delta (x, y)"), 7, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.trajectoryXSpinBox, 7, 1, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.trajectoryYSpinBox, 7, 2, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(QtGui.QLabel("Distortion Parameter"), 8, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.distortionParameterSpinBox, 8, 1, 1, 2, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(QtGui.QLabel("Set ground plane"), 9, 0, 1, 1, QtCore.Qt.AlignLeft)
#         controlsLayout.addWidget(self.doSetGroundPlaneCheckBox, 9, 1, 1, 2, QtCore.Qt.AlignLeft)
#         controlsGroup.setLayout(controlsLayout)
        
#         topDownImageLayout = QtGui.QVBoxLayout()
#         topDownImageLayout.addStretch()
#         topDownImageLayout.addWidget(self.topDownImageLabel)
#         topDownImageLayout.addWidget(self.topDownImageInfo)
#         topDownImageLayout.addStretch()
#         topDownImageLayout.addWidget(self.orientationsGraph)
#         topDownImageLayout.addWidget(controlsGroup)
#         topDownImageLayout.addStretch()
        
#         mainLayout.addLayout(originalImageLayout)
#         mainLayout.addLayout(topDownImageLayout)
#         self.setLayout(mainLayout)

