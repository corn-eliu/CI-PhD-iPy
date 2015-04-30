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

import sift
import GraphWithValues as gwv

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

sampleData = "mopeds/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame-*.png")
mattes = glob.glob(dataFolder + sampleData + "matte-*.png")
scribbles = glob.glob(dataFolder + sampleData + "scribble-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
scribbles = np.sort(scribbles)
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes), len(scribbles)

# <codecell>

sift.process_image(frames[0], 'tmp.key')
l1,d1 = sift.read_features_from_file('tmp.key')
im1 = array(Image.open(frames[0]))
figure(); sift.plot_features(im1,l1)

sift.process_image(frames[14], 'tmp.key')
l2,d2 = sift.read_features_from_file('tmp.key')
im2 = array(Image.open(frames[14]))
figure(); sift.plot_features(im2,l2)

# <codecell>

# m1 = sift.match(d1,d2)
# figure(); sift.plot_matches(im1,im2,l1,l2,m1)#, 1000)
# figure(); sift.plot_match_displacement(im1,l1,l2,m1)#, 1000)
# figure(); sift.plot_match_displacement(im2,l1,l2,m1)#, 1000)

m2 = sift.match_twosided(d1,d2)
figure(); sift.plot_matches(im1,im2,l1,l2,m2)#, 1000)
figure(); sift.plot_match_displacement(im1,l1,l2,m2)#, 1000)
figure(); sift.plot_match_displacement(im2,l1,l2,m2)#, 1000)

# <codecell>

scribble = array(Image.open(scribbles[0]))[:, :, 0]
figure(); imshow(scribble)

distTransOut = cv2.distanceTransform(uint8(np.max(scribble)-scribble), cv2.cv.CV_DIST_L2, 3)
distTransIn = cv2.distanceTransform(uint8(scribble), cv2.cv.CV_DIST_L2, 3)
distTrans = distTransOut-distTransIn
# gwv.showCustomGraph(distTrans)
figure(); imshow(distTrans, interpolation='nearest')

# <codecell>

## indices of points in im1 that got matched to some point in im2
matchedPointsIdxs = np.argwhere(np.ndarray.flatten(m2) > 0)
## locations of matched points in im1
matchedPoints1 = np.reshape(l1[matchedPointsIdxs, :], (len(matchedPointsIdxs), 4))
scatter(matchedPoints1[:, 1], matchedPoints1[:, 0])

# <codecell>

distTransProbs = np.exp(-distTrans/(0.5*np.mean(distTrans)))
## probabilities of matched points in im1 based on closeness to scribble
matchedPointsProbs = distTransProbs[np.array(matchedPoints1[:, 0], dtype=int), np.array(matchedPoints1[:, 1], dtype=int)]
## sorted indices of matched points from high to low probability
matchedPointsSortedIdxs = np.argsort(matchedPointsProbs)[::-1]
numBestMatches = 4
scatter(matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 1], matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 0], c='r', marker='x')

# <codecell>

## get matching points in im2 of top best points in im1 based on closeness to scribble
matchingPointsBestIdxs = np.array(np.ndarray.flatten(m2[matchedPointsIdxs[matchedPointsSortedIdxs[0:numBestMatches]]]), dtype=int)
scatter(l2[matchingPointsBestIdxs, 1], l2[matchingPointsBestIdxs, 0], c='y', marker='x')

# <codecell>

homography = cv2.findHomography(np.array(matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 0:2], dtype=np.float32), 
                                         np.array(l2[matchingPointsBestIdxs, 0:2], dtype=np.float32))[0]

# <codecell>

print homography
print np.linalg.inv(homography)

# <codecell>

homography = cv2.getPerspectiveTransform(np.array(matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 0:2], dtype=np.float32), 
                                         np.array(l2[matchingPointsBestIdxs, 0:2], dtype=np.float32));

# <codecell>

print homography

# <codecell>

A = np.zeros((8, 8))
b = np.zeros((8, 1))
for x, w, i in zip(np.array(matchedPoints1[matchedPointsSortedIdxs[0:4], 0:2], dtype=np.float32), 
                                         np.array(l2[matchingPointsBestIdxs[0:4], 0:2], dtype=np.float32), 
                                         np.arange(4, dtype=int)):
    print x, w, i*2, i*2+1
    A[i, :]   = [x[0], x[1], 1, 0, 0, 0, -x[0]*w[0], -x[1]*w[0]]
    A[i+4, :] = [0, 0, 0, x[0], x[1], 1, -x[0]*w[1], -x[1]*w[1]]
    b[i] = w[0]
    b[i+4] = w[1]
    
# print A
# print b
x = np.linalg.solve(A, b)
# x = np.linalg.lstsq(A, b)[0]
# print x
hom = np.reshape(np.concatenate((x, [[1]])), (3, 3))
print hom

# <codecell>

print homography
print hom
print homography-hom

# <codecell>

A = np.zeros((8, 9))
print "x \t \t y \t \t u \t \t v"
for w, x, i in zip(np.array(matchedPoints1[matchedPointsSortedIdxs[0:4], 0:2], dtype=np.float32), 
                                         np.array(l2[matchingPointsBestIdxs[0:4], 0:2], dtype=np.float32), 
                                         np.arange(4, dtype=int)):
    print x, w, i*2, i*2+1
    A[i*2, :]   = [0, 0, 0, -w[0], -w[1], -1, x[1]*w[0], x[1]*w[1], x[1]]
    A[i*2+1, :] = [w[0], w[1], 1, 0, 0, 0, -x[0]*w[0], -x[0]*w[1], -x[0]]
    print np.array(A[i*2, :], dtype=int)
    print np.array(A[i*2+1, :], dtype=int)
    
U, s, V = np.linalg.svd(A)
hom = np.reshape(np.transpose(V)[:, -1], (3, 3))
print hom
print hom/hom[-1, -1]
print np.linalg.inv(hom.T)
print np.linalg.inv(hom)
print np.transpose(V)[:, -1]

# <codecell>

transScribble = np.zeros_like(scribble)
for i in xrange(transScribble.shape[0]) :
    for j in xrange(transScribble.shape[1]) :
        la = np.dot(np.linalg.inv(homography), np.array([i, j, 1]))
        la = np.array(la/la[-1], dtype=int)
        if la[0] >= 0 and la[1] >= 0 and la[0]<transScribble.shape[0] and la[1]<transScribble.shape[1] :
            transScribble[i, j] = scribble[la[0], la[1]]
        if i == 277 and j == 264 :
            print la
figure(); imshow(transScribble, interpolation='nearest')

# <codecell>

figure(); imshow(np.array(im1*np.repeat(np.reshape(1.0-0.8*(255-scribble)/255, (scribble.shape[0], scribble.shape[1], 1)), 3, axis=-1), dtype=uint8))
figure(); imshow(np.array(im2*np.repeat(np.reshape(1.0-0.8*(255-transScribble)/255, (transScribble.shape[0], transScribble.shape[1], 1)), 3, axis=-1), dtype=uint8))

# <codecell>

gwv.showCustomGraph(1.0-0.5*(255-scribble)/255)

# <codecell>

print hom/hom[-1, -1]
print np.linalg.inv(hom)
tmp = np.dot(hom, np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:4], 0:2], dtype=np.float32).T, np.ones((1, 4))), axis=0))
print np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:4], 0:2], dtype=np.float32).T, np.ones((1, 4))), axis=0)
print tmp
print (tmp/np.reshape(np.repeat(tmp[-1, :], 3, axis=0), (4, 3)).T).T

# <codecell>

tmp = np.dot(hom, np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 0:2], dtype=np.float32).T, np.ones((1, numBestMatches))), axis=0))
print np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 0:2], dtype=np.float32).T, np.ones((1, numBestMatches))), axis=0)
print tmp
print (tmp/np.reshape(np.repeat(tmp[-1, :], 3, axis=0), (numBestMatches, 3)).T).T

# <codecell>

tmp = np.dot(homography, np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 0:2], dtype=np.float32).T, np.ones((1, numBestMatches))), axis=0))
print np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:numBestMatches], 0:2], dtype=np.float32).T, np.ones((1, numBestMatches))), axis=0)
print tmp
print (tmp/np.reshape(np.repeat(tmp[-1, :], 3, axis=0), (numBestMatches, 3)).T).T

# <codecell>

bob = np.dot(homography, np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:4], 0:2], dtype=np.float32).T, np.ones((1, 4))), axis=0)[:, 0])
print bob/bob[-1], bob, bob/bob[0]

# <codecell>

bob = np.dot(hom, np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:4], 0:2], dtype=np.float32).T[::-1], np.ones((1, 4))), axis=0)[:, 0])
print bob/bob[-1], bob, bob/bob[0]

# <codecell>

tmp = np.dot(np.concatenate((np.array(matchedPoints1[matchedPointsSortedIdxs[0:4], 0:2], dtype=np.float32), np.ones((4, 1))), axis=-1), homography)
print np.reshape(np.repeat(tmp[:, -1], 2, axis=0), (4, 2))
print tmp/np.reshape(np.repeat(tmp[:, -1], 3, axis=0), (4, 3))

# <codecell>

gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray,2,3,0.04)
# dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
corners  = np.zeros(gray.shape)
corners[dst>0.05*dst.max()]=255

# <codecell>

figure(); imshow(im1)
scatter(np.argwhere(corners == 255)[:, 1], np.argwhere(corners == 255)[:, 0])

# <codecell>

M2 = icp(l1[:, 0:2].T, l2[:, 0:2].T, no_iterations=30)

#Plot the result
src = np.array([l1[:, 0:2]]).astype(np.float32)
res = cv2.transform(src, M2)
figure()
imshow(im2)
ax = gca()
ax.set_autoscale_on(False)
plot(l2[:, 1],l2[:, 0], 'b.') ## target point cloud
plot(res[0].T[1], res[0].T[0], 'r.') ## point cloud fitted with found transform
plot(l1[:, 1], l1[:, 0], 'g.') ## point cloud to fit to target

# <codecell>

ang = np.linspace(-np.pi/2, np.pi/2, 320)
a = np.array([ang, np.sin(ang)])
th = np.pi/2
rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
b = np.dot(rot, a) + np.array([[0.2], [0.3]])

#Run the icp
M2 = icp(a, b, [0.18,  0.32, np.pi/2.2], 100)

#Plot the result
src = np.array([a.T]).astype(np.float32)
res = cv2.transform(src, M2)
plt.figure()
plt.plot(b[0],b[1], 'b.')
plt.plot(res[0].T[0], res[0].T[1], 'r.')
plt.plot(a[0], a[1], 'g.')
plt.show()

# <codecell>

from sklearn.neighbors import NearestNeighbors

def icp(a, b, #init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)
    print src.shape

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])
    print Tr

    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]

