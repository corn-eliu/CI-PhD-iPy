# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

from PIL import Image

import numpy as np

import scipy as sp
import scipy.io as sio
import cv2
import cv
import glob
import time
import gc
import re

import sys
import os

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

basePath = "/media/ilisescu/Data1/PhD/data/havana/"
bgImage = np.array(Image.open(basePath+"median.png"))
for i in np.arange(800, 800+476) :
    currentFrame = np.array(Image.open(basePath+"bus1/bus1-frame-{0:05d}.png".format(i)))
    spriteLoc = np.argwhere(currentFrame[:, :, -1] != 0)
    alphas = currentFrame[spriteLoc[:, 0], spriteLoc[:, 1], -1].reshape((len(spriteLoc), 1)).repeat(3, axis=-1)
    
    finalFrame = np.copy(bgImage)
    finalFrame[spriteLoc[:, 0], spriteLoc[:, 1], :] = (currentFrame[spriteLoc[:, 0], spriteLoc[:, 1], 0:-1]*(alphas/255.0) + 
                                                       bgImage[spriteLoc[:, 0], spriteLoc[:, 1], :]*((255-alphas)/255.0))

    
    Image.fromarray((finalFrame).astype(numpy.uint8)).save(basePath+"bus1OnMedian/bus1-frame-{0:05d}.png".format(i))

# <codecell>

img = np.array(Image.open(dataFolder+"testImage.png"))
figure(); imshow(img)

# <codecell>

square = np.vstack((np.array([[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]]).T, ones(5)))
print square
intrinsics = np.array([[640.0, 0.0, 320.0, 0.0], [0.0, 360.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 4.0], [0.0, 0.0, 0.0, 1.0]])


theta = 10.0*(np.pi/180.0)
R = np.array([[np.cos(theta), 0.0, np.sin(theta), 0.], [0.0, 1.0, 0.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta), 0.0], [0.0, 0.0, 0.0, 1.0]])

squareImg = np.dot(np.dot(intrinsics, extrinsics), np.dot(R, square))
squareImg = squareImg/squareImg[-1, :]
print squareImg[0:2, :]

figure(); 
xlim(0, 640); ylim(0, 360)
plot(squareImg[0, :], squareImg[1, :])

# <codecell>

origSquare = np.copy(squareImg)

# <codecell>

H = cv2.findHomography(squareImg[0:2, 0:-1].T, origSquare[0:2, 0:-1].T)[0]
xs = np.ndarray.flatten(np.arange(img.shape[1], dtype=float).reshape((img.shape[1], 1)).repeat(img.shape[0], axis=-1))
ys = np.ndarray.flatten(np.arange(img.shape[0], dtype=float).reshape((1, img.shape[0])).repeat(img.shape[1], axis=0))
data = np.array(np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys))), np.ones(len(ys)))), dtype=int)
imgWarped = np.zeros(img.shape, dtype=np.uint8)

warpedCoords = np.dot(H, data)
warpedCoords /= warpedCoords[-1, :]

for warpedCoord, coord in zip(np.array(warpedCoords.T, dtype=int), data.T) :
    if warpedCoord[0] < img.shape[1] and warpedCoord[0] > 0 and warpedCoord[1] < img.shape[0] and warpedCoord[1] > 0 :
        imgWarped[coord[1], coord[0], :] = img[warpedCoord[1], warpedCoord[0], :]
        
figure(); imshow(imgWarped)
figure(); imshow(img)

# <codecell>

square = np.vstack((np.array([[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]]).T, ones(5)))
print square
intrinsics = np.eye(3, 4)#np.array([[640.0, 0.0, 320.0, 0.0], [0.0, 360.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]])
print intrinsics
print extrinsics
print np.dot(intrinsics, extrinsics)

# square = np.dot(np.dot(intrinsics, extrinsics), square)
# square = square/square[-1, :]
print square

theta = 10.0*(np.pi/180.0)
R = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
t = np.array([0.0, 0.0, 0.0])
n = np.array([0.0, 0.0, 1.0])
d = 2.0

# H = R-np.dot(t, n.T)/d
# H = np.dot(np.dot(R, np.eye(3)-np.dot(extrinsics[0:3, -1]-t, n.T)/d), extrinsics[0:3, 0:3].T)
print H
# print np.dot(t, n.T)/d

H = np.hstack((np.vstack((R, np.zeros((1, 3)))), np.array([[0.0], [0], [0], [1]])))
extrinsicsCam2 = np.dot(H, extrinsics)
print H
print extrinsicsCam2

# square = np.dot(np.dot(np.dot(intrinsics[:, 0:-1], H), np.linalg.inv(intrinsics[:, 0:-1])), square)
# square = np.dot(np.linalg.inv(intrinsics[:, 0:-1]), square)
# square =  np.dot(np.dot(intrinsics, np.dot(extrinsics, H)), square)

# cam2 = np.dot(np.dot(np.dot(intrinsics, extrinsicsCam2), 
#                      np.hstack((np.linalg.inv(intrinsics[:, 0:-1]), np.zeros((3, 1))))), 
#               np.linalg.inv(extrinsicsCam2))
print cam2.shape
# square = np.dot(np.dot(intrinsics[:, 0:3], np.dot(np.linalg.inv(extrinsics), extrinsicsCam2)[0:3, :]), square)
# square = np.dot(np.dot(intrinsics, extrinsicsCam2), square)

square = square/square[-1, :]
print square[0:2, :]

figure(); 
xlim(0, 640); ylim(0, 360)
plot(square[0, :], square[1, :], )

# <codecell>

xs = np.ndarray.flatten(np.arange(img.shape[1], dtype=float).reshape((img.shape[1], 1)).repeat(img.shape[0], axis=-1))
ys = np.ndarray.flatten(np.arange(img.shape[0], dtype=float).reshape((1, img.shape[0])).repeat(img.shape[1], axis=0))
data = np.array(np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys))), np.ones(len(ys)))), dtype=int)
imgWarped = np.zeros(img.shape, dtype=np.uint8)


theta = 45.0*(np.pi/180.0)
R = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
originT = (np.array([[img.shape[1]], [img.shape[0]], [0]])/2).repeat(data.shape[-1], axis=-1)

warpedCoords = np.dot(R, data-originT)+originT
warpedCoords /= warpedCoords[-1, :]

for warpedCoord, coord in zip(np.array(warpedCoords.T, dtype=int), data.T) :
    if warpedCoord[0] < img.shape[1] and warpedCoord[0] > 0 and warpedCoord[1] < img.shape[0] and warpedCoord[1] > 0 :
        imgWarped[coord[1], coord[0], :] = img[warpedCoord[1], warpedCoord[0], :]
        
figure(); imshow(imgWarped)
figure(); imshow(img)

# <codecell>

print warpedCoords
print data

# <codecell>

print np.max(warpedCoords[0, :]), np.min(warpedCoords[0, :]), np.max(warpedCoords[1, :]), np.min(warpedCoords[1, :])
print R
print np.linalg.inv(R)

# <codecell>

print np.dot(intrinsics, extrinsicsCam2)
print np.hstack((np.linalg.inv(intrinsics[:, 0:-1]), np.zeros((3, 1))))

# <codecell>

print np.dot(extrinsicsCam2, np.linalg.inv(extrinsics))
print extrinsics
print extrinsicsCam2

# <codecell>

print np.dot(extrinsics[0:3, -1]-t, n.T)/d

# <codecell>

print np.dot(extrinsics, H).shape
print extrinsics.shape

# <codecell>

print np.dot(t, n.T)/d

# <codecell>

import sift

# <codecell>

sift.process_image(dataFolder + '/mopeds/frame-00001.png', 'tmp.key')
l1,d1 = sift.read_features_from_file('tmp.key')
im = array(Image.open(dataFolder + '/mopeds/frame-00001.png'))
sift.plot_features(im,l1)

# <codecell>

f = loadtxt('tmp2.key')
l = f[:,:4]
d = f[:,4:]

# <codecell>

def draw_circle(c,r):
    t = arange(0,1.01,.01)*2*pi
    x = r*cos(t) + c[0]
    y = r*sin(t) + c[1]
    plot(x,y,'b',linewidth=2)

figure(); imshow(im)
if False:
    [draw_circle([p[0],p[1]],p[2]) for p in l]
else:
    plot(l[:,0],l[:,1],'ob')
axis('off')

# <codecell>

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray,2,3,0.04)
# dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
corners  = np.zeros(gray.shape)
corners[dst>0.01*dst.max()]=255

# <codecell>

figure(); imshow(im)
scatter(np.argwhere(corners == 255)[:, 1], np.argwhere(corners == 255)[:, 0])

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
sampleData = "ribbon2/"
# sampleData = "ribbon1_matted/"
# sampleData = "little_palm1_cropped/"
# sampleData = "ballAnimation/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame-*.png")
mattes = glob.glob(dataFolder + sampleData + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes)

lowThresh = 96
highThresh = lowThresh*2

# <codecell>

img = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
matte = cv2.cvtColor(cv2.imread(mattes[0]), cv2.COLOR_BGR2GRAY)

imgEdges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), lowThresh, highThresh)
matteEdges = cv2.Canny(matte, lowThresh, highThresh)
matteEdges = cv2.dilate(matteEdges, np.ones((6,6),dtype=np.uint8))
matteEdges = cv2.erode(matteEdges, np.ones((6,6),dtype=np.uint8))

# figure(); imshow(imgEdges*(matte/255.0), interpolation='nearest')
figure(); imshow(matteEdges, interpolation='nearest')

# <codecell>

## find points on matte edges
edgePoints = np.argwhere(matteEdges == np.max(matteEdges))
## closes point to top-left (i.e.) [0, 0]
startPoint = edgePoints[np.argmin(np.sum(edgePoints, axis=1)), :]
print startPoint
scatter(startPoint[1], startPoint[0])

# <codecell>

edgePoints

