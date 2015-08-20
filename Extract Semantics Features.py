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

from PIL import Image

from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.datasets.samples_generator import make_blobs
from skimage.feature import hog
from skimage import color

import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import opengm

DICT_SPRITE_NAME = 'sprite_name'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SPRITE_IDX = 'sprite_idx' # stores the index in the self.trackedSprites array of the sprite used in the generated sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed

DATA_IMAGE = 'image_data'
DATA_MASK = 'mask_data'
DATA_TRACK_BBOX = 'track_bbox_data'
DATA_TRACK_BBOX_CENTERS = 'track_bbox_centers_data'

# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
# dataSet = "pendulum/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "splashes_water/"
# dataSet = "small_waterfall/"
dataSet = "eu_flag_ph_left/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3
hogOrientations = 9
pixelsPerCell = 16

# <codecell>

## load 
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
    print trackedSprites[-1][DICT_SPRITE_NAME]

# <codecell>

def getFisherEncoding(feats, featsShape, gmm) :
    """Take feats feature vector and encode it using Fisher encoding
    
           feats: feature vector
           featsShape: shape that feats had when the gmm was fitted (if e.g. feats are hog features and it has been flattened)
           gmm: a fitted sklearn.mixture.GMM model
           
        return: 1D encoded feature vector encodedFeats"""
    
    if gmm != None :
        try :
            modelMeans = gmm.means_
            numComponents = modelMeans.shape[0]
            featDim = featsShape[-1]
            ##get prior probability pi for each component
            priors = gmm.weights_
            ##get posterior probabilities q for each data point and each component
            posteriors = gmm.predict_proba(np.reshape(feats, featsShape))
            
            us = np.empty(0)
            vs = np.empty(0)
            ## this one uses the formulation given by vlfeat
            for k in xrange(numComponents) :
                ## get covariance matrix for component k
                kCompCov = gmm.covars_[k, :]
                
                kCompMeansRep = modelMeans[k, :].reshape((1, featDim)).repeat(featsShape[0], axis=0)
                kCompCovRep = kCompCov.reshape((1, featDim)).repeat(featsShape[0], axis=0)
                kCompPostRep = posteriors[:, k].reshape((featsShape[0], 1)).repeat(featDim, axis=-1)
                
                uk = np.sum(kCompPostRep*(np.reshape(feats, featsShape)-kCompMeansRep)/kCompCovRep, axis=0)
                uk /= (featsShape[0]*np.sqrt(priors[k]))
                us = np.concatenate((us, uk))
                
                vk = np.sum(kCompPostRep*((((np.reshape(feats, featsShape)-kCompMeansRep)/kCompCovRep)**2)-1), axis=0)
                vk /= (featsShape[0]*np.sqrt(2*priors[k]))
                vs = np.concatenate((vs, vk))
                
            encodedFeats = np.concatenate((us, vs))
            return encodedFeats
        except Exception :
            raise Exception, "Unfitted gmm"
    else :
        print "No gm model has been specified"
        return feats

# <codecell>

def fitGMM(trainingFeats, ncomp) :
    """Takes a set of features trainingFeats and returns Gaussian Mixture Model with ncomp components
        Descriptors of an image can be softly assinged to each K Gaussian component and used for Fisher encoding
    
           trainingFeats: training features
           ncomp: number of gaussians in the GMM
           
        return: gmm model"""
    
    model = GMM(n_components=ncomp, covariance_type='diag')
    model.fit(trainingFeats)
    return model

# <codecell>

def getHOGFeats(imageData, visHogs = False) :
    """Take RGB imageData and compute HOG features. The returned features are flattened to a 1D array from an array of shape
        (len(feats)/orientations, orientations) where the first orientations number of values represent the hog features of the
        top left cell and then counting cells right row-wise (i.e. do first row left to right, then second and so on)
    
           imageData: input RGB image data
           
        return: 1D vector feats"""
    
    feats = hog(color.rgb2gray(imageData), orientations=hogOrientations, 
                     pixels_per_cell=(pixelsPerCell, pixelsPerCell), 
                     cells_per_block=(1, 1), visualise=visHogs)
    if visHogs :
        gwv.showCustomGraph(feats[1])
        return feats[0]
    else :
        return feats

def getSemanticsData(rawSemanticData, frameIdx) :
    """Extracts the necessay data from rawSemanticData for the desired frame at frameIdx.
    
           semanticData: data attached to a semantic label (e.g. a tracked sprite)
           frameIdx: index of the frame in semanticData to compute features for          
           
        return: dictionary of extracted data which includes :
                        RGB data for desired patch imageData
                        mask for desired patch maskData"""
    

    frameKey = np.sort(rawSemanticData[DICT_FRAMES_LOCATIONS].keys())[frameIdx]
    frameName = rawSemanticData[DICT_FRAMES_LOCATIONS][frameKey].split(os.sep)[-1]
#     print frameName

    semanticsData = {}

    if DICT_SPRITE_NAME in rawSemanticData.keys() :
        maskDir = dataPath + dataSet + rawSemanticData[DICT_SPRITE_NAME] + "-masked-blended"
    
        ## get mask and find patch to deal with
        maskImg = np.array(cv2.imread(maskDir+"/"+frameName, cv2.CV_LOAD_IMAGE_UNCHANGED), dtype=np.uint8)[:, :, -1]
        imgSize = np.array(maskImg.shape)
        visiblePixels = np.argwhere(maskImg != 0)
        enlargePatchBy = 20
        topLeft = np.min(visiblePixels, axis=0) - enlargePatchBy
        patchSize = np.max(visiblePixels, axis=0) - topLeft + 1 + 2*enlargePatchBy
        
        ## make sure we're within bounds
        topLeft[np.argwhere(topLeft < 0)] = 0
        patchSize[(topLeft+patchSize) > imgSize] += (imgSize-(topLeft+patchSize))[(topLeft+patchSize) > imgSize]
        
        imageData = np.asarray(Image.open(rawSemanticData[DICT_FRAMES_LOCATIONS][frameKey]))[topLeft[0]:topLeft[0]+patchSize[0], topLeft[1]:topLeft[1]+patchSize[1], :]
        maskData = maskImg[topLeft[0]:topLeft[0]+patchSize[0], topLeft[1]:topLeft[1]+patchSize[1]]
        
        semanticsData[DATA_IMAGE] = imageData
        semanticsData[DATA_MASK] = maskData
    else :
        semanticsData[DATA_IMAGE] = np.asarray(Image.open(rawSemanticData[DICT_FRAMES_LOCATIONS][frameKey]))
    
    if DICT_BBOXES in rawSemanticData.keys() :
        semanticsData[DATA_TRACK_BBOX] = rawSemanticData[DICT_BBOXES][frameKey]
        
    if DICT_BBOX_CENTERS in rawSemanticData.keys() :
        semanticsData[DATA_TRACK_BBOX_CENTERS] = rawSemanticData[DICT_BBOX_CENTERS][frameKey]
    
    return semanticsData
    
    
def getSemanticsFeatures(semanticsData, gmModel, getHog = False, verbose = False) :
    """Computes relevant features given the data in semanticsData.
    
           semanticsData: data attached to a semantic label (e.g. a color patch )
           gmModel[sklearn.mixture.GMM]: trained gaussian mixture model used for computing fisher encoding of hog features           
           
        return: 1D vector features"""


    if DATA_IMAGE in semanticsData.keys() :
    
        hogFeats = getHOGFeats(semanticsData[DATA_IMAGE])
        
        if DATA_MASK in semanticsData.keys() :
            patchSize = np.array(semanticsData[DATA_MASK].shape)
            
            ########### THIS KEEPS FEATS WHO'S CELL CENTER IS INSIDE MASK ###########
#             ## get coords of cells centers
#             rowCoords = np.arange(pixelsPerCell/2, patchSize[0]-pixelsPerCell/2, pixelsPerCell)
#             colCoords = np.arange(pixelsPerCell/2, patchSize[1]-pixelsPerCell/2, pixelsPerCell)
            
#             cellGridRows = len(rowCoords)#int(np.round(float(patchSize[0]-pixelsPerCell/2)/pixelsPerCell))
#             cellGridCols = len(colCoords)#int(np.round(float(patchSize[1]-pixelsPerCell/2)/pixelsPerCell))
            
#             rowCoords = rowCoords.reshape((1, cellGridRows)).repeat(cellGridCols)
#             colCoords = np.ndarray.flatten(colCoords.reshape((1, cellGridCols)).repeat(cellGridRows, axis=0))
            
            ## check which centers are within the mask and only keep those
#             hogFeats = hogFeats[(semanticsData[DATA_MASK][rowCoords, colCoords] != 0).repeat(hogOrientations)]

            
            ########### THIS KEEPS FEATS WHO'S CELL'S EXTENT FALLS INSIDE THE MASK ###########
            
            ## make an image that contains the index of a hog cell for the extent of the cell (lots of colored squares :))
            allIdxs = -np.ones(patchSize, dtype=int)
            numRowIdxs = len(np.arange(0, patchSize[0], pixelsPerCell)) - 1
            numColIdxs = len(np.arange(0, patchSize[1], pixelsPerCell)) - 1
            gridIdxs = np.arange(numColIdxs, dtype=int).reshape((1, numColIdxs)).repeat(numRowIdxs*pixelsPerCell, axis=0).repeat(pixelsPerCell, axis=-1)
            gridIdxs += np.arange(numRowIdxs, dtype=int).reshape((numRowIdxs, 1)).repeat(numColIdxs*pixelsPerCell, axis=-1).repeat(pixelsPerCell, axis=0)*numColIdxs
            allIdxs[:gridIdxs.shape[0], :gridIdxs.shape[1]] = gridIdxs
            
            ## find which of the cells fall inside the mask and keep them
            visiblePixels = np.argwhere(semanticsData[DATA_MASK] != 0)
            hogsToKeep = np.zeros(numRowIdxs*numColIdxs, dtype=bool)
            hogsToKeep[np.unique(allIdxs[visiblePixels[:, 0], visiblePixels[:, 1]])] = True
            
            hogFeats = hogFeats[hogsToKeep.repeat(hogOrientations)]
            
        if getHog :
            if verbose :
                print "return hog feats only"
            return hogFeats
        
        features = np.empty(0)
        if gmModel != None :
            features = getFisherEncoding(hogFeats, (len(hogFeats)/hogOrientations, hogOrientations), gmModel)
            
            if verbose :
                print "added hog fisher encoding"
            
        if DATA_TRACK_BBOX in semanticsData.keys() :
            ## compute area of bbox
            area = np.linalg.norm(semanticsData[DATA_TRACK_BBOX][TL_IDX, :] - semanticsData[DATA_TRACK_BBOX][TR_IDX, :])
            area *= np.linalg.norm(semanticsData[DATA_TRACK_BBOX][TR_IDX, :] - semanticsData[DATA_TRACK_BBOX][BR_IDX, :])
            
            features = np.concatenate((features, [area]))
            
            if verbose :
                print "added bbox area"
            
        if DATA_TRACK_BBOX_CENTERS in semanticsData.keys() :
            features = np.concatenate((features, semanticsData[DATA_TRACK_BBOX_CENTERS]))
            
            if verbose :
                print "added bbox center"
            
        return features
    
    return None
#     return features

# <codecell>

def aabb2obbDist(aabb, obb, verbose = False) :
    if verbose :
        figure(); plot(aabb[:, 0], aabb[:, 1])
        plot(obb[:, 0], obb[:, 1])
    minDist = 100000000.0
    colors = ['r', 'g', 'b', 'y']
    for i, j in zip(arange(4), np.mod(arange(1, 5), 4)) :
        m = (obb[j, 1] - obb[i, 1]) / (obb[j, 0] - obb[i, 0])
        b = obb[i, 1] - (m * obb[i, 0]);
        ## project aabb points onto obb segment
        projPoints = np.dot(np.hstack((aabb, np.ones((len(aabb), 1)))), np.array([[1, m, -m*b], [m, m**2, b]]).T)/(m**2+1)
        if np.all(np.negative(np.isnan(projPoints))) :
            ## find distances
            dists = np.linalg.norm(projPoints-aabb, axis=-1)
#             dists = aabb2pointsDist(aabb, projPoints)
            ## find closest point
            closestPoint = np.argmin(dists)
            ## if rs is between 0 and 1 the point is on the segment
            rs = np.sum((obb[j, :]-obb[i, :])*(aabb-obb[i, :]), axis=1)/(np.linalg.norm(obb[j, :]-obb[i, :])**2)
            if verbose :
                print projPoints
                scatter(projPoints[:, 0], projPoints[:, 1], c=colors[i])
                print dists
                print closestPoint
                print rs
            ## if closestPoint is on the segment
            if rs[closestPoint] > 0.0 and rs[closestPoint] < 1.0 :
#                 print "in", aabb2pointDist(aabb, projPoints[closestPoint, :])
                minDist = np.min((minDist, aabb2pointDist(aabb, projPoints[closestPoint, :])))
            else :
#                 print "out", aabb2pointDist(aabb, obb[i, :]), aabb2pointDist(aabb, obb[j, :])
                minDist = np.min((minDist, aabb2pointDist(aabb, obb[i, :]), aabb2pointDist(aabb, obb[j, :])))

    return minDist


def aabb2pointDist(aabb, point) :
    dx = np.max((np.min(aabb[:, 0]) - point[0], 0, point[0] - np.max(aabb[:, 0])))
    dy = np.max((np.min(aabb[:, 1]) - point[1], 0, point[1] - np.max(aabb[:, 1])))
    return np.sqrt(dx**2 + dy**2);

def aabb2pointsDist(aabb, points) :
    dx = np.max(np.vstack((np.min(aabb[:, 0]) - points[:, 0], np.zeros(len(points)), points[:, 0] - np.max(aabb[:, 0]))), axis=0)
    dy = np.max(np.vstack((np.min(aabb[:, 1]) - points[:, 1], np.zeros(len(points)), points[:, 1] - np.max(aabb[:, 1]))), axis=0)
    return np.sqrt(dx**2 + dy**2);


def getShiftedSpriteTrackDist(firstSprite, secondSprite, shift) :
    
    spriteTotalLength = np.zeros(2, dtype=int)
    spriteTotalLength[0] = len(firstSprite[DICT_BBOX_CENTERS])
    spriteTotalLength[1] = len(secondSprite[DICT_BBOX_CENTERS])
    
    ## find the overlapping sprite subsequences
    ## length of overlap is the minimum between length of the second sequence and length of the first sequence - the advantage it has n the second sequence
    overlapLength = np.min((spriteTotalLength[0]-shift, spriteTotalLength[1]))
    
    frameRanges = np.zeros((2, overlapLength), dtype=int)
    frameRanges[0, :] = np.arange(shift, overlapLength + shift)
    frameRanges[1, :] = np.arange(overlapLength)
    
    totalDistance, distances = getOverlappingSpriteTracksDistance(firstSprite, secondSprite, frameRanges)
    
    return totalDistance, distances, frameRanges


def getOverlappingSpriteTracksDistance(firstSprite, secondSprite, frameRanges, doEarlyOut = True, verbose = False) :
#     ## for now the distance is only given by the distance between bbox center but can add later other things like bbox overlapping region
#     bboxCenters0 = np.array([firstSprite[DICT_BBOX_CENTERS][x] for x in np.sort(firstSprite[DICT_BBOX_CENTERS].keys())[frameRanges[0, :]]])
#     bboxCenters1 = np.array([secondSprite[DICT_BBOX_CENTERS][x] for x in np.sort(secondSprite[DICT_BBOX_CENTERS].keys())[frameRanges[1, :]]])
    
#     centerDistance = np.linalg.norm(bboxCenters0-bboxCenters1, axis=1)
    
#     totDist = np.min(centerDistance)
#     allDists = centerDistance
    
    firstSpriteKeys = np.sort(firstSprite[DICT_BBOX_CENTERS].keys())
    secondSpriteKeys = np.sort(secondSprite[DICT_BBOX_CENTERS].keys())
    allDists = np.zeros(frameRanges.shape[-1])
    for i in xrange(frameRanges.shape[-1]) :
        
        allDists[i] = getSpritesBBoxDist(firstSprite[DICT_BBOX_ROTATIONS][firstSpriteKeys[frameRanges[0, i]]],
                                          firstSprite[DICT_BBOXES][firstSpriteKeys[frameRanges[0, i]]], 
                                          secondSprite[DICT_BBOXES][secondSpriteKeys[frameRanges[1, i]]])
        
        if verbose and np.mod(i, frameRanges.shape[-1]/100) == 0 :
            sys.stdout.write('\r' + "Computed image pair " + np.string_(i) + " of " + np.string_(frameRanges.shape[-1]))
            sys.stdout.flush()
        
        ## early out since you can't get lower than 0
        if doEarlyOut and allDists[i] == 0.0 :
            break
            
    totDist = np.min(allDists)
#     return np.sum(centerDistance)/len(centerDistance), centerDistance    
    return totDist, allDists

def getSpritesBBoxDist(theta, bbox1, bbox2) :
    rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    bbox1 = np.dot(rotMat, bbox1.T).T
    bbox2 = np.dot(rotMat, bbox2.T).T
    ## if the bboxes coincide then the distance is set to 0
    if np.all(np.abs(bbox1 - bbox2) <= 10**-10) :
        return 0.0
    else :
        return aabb2obbDist(bbox1, bbox2)
    
# print getOverlappingSpriteTracksDistance(trackedSprites[3], trackedSprites[0], np.array([[1020], [400]]))

# <codecell>

# tmpData = getSemanticsData(trackedSprites[0], 155)
# tmp = getSemanticsFeatures(getSemanticsData(trackedSprites[0], 155), None, True)
print len(tmp)/hogOrientations
print tmpData
print getSemanticsData(trackedSprites[0], 155)

# <headingcell level=2>

# Hacking together a way to consider full frame features

# <codecell>

## HACK ##
## read the video frames and treat as a sprite that only has frame locations
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
print numOfFrames
semanticEntities = []
semanticEntity = {DICT_FRAMES_LOCATIONS:{}}
for f in xrange(numOfFrames) :
    semanticEntity[DICT_FRAMES_LOCATIONS][f] = frameLocs[f]
semanticEntities.append(semanticEntity)

# <codecell>

## get random frames from each tracked sprite and get their hog features to then train the GMM
numLeaveOut = 0
numFramesPerEntity = 20
gmmTrainingFeats = np.empty(0)
for entity in semanticEntities:
    for frameIdx in random.choice(arange(numLeaveOut, len(entity[DICT_FRAMES_LOCATIONS])-numLeaveOut), numFramesPerEntity, replace=False) :
        tmp = len(gmmTrainingFeats)
        gmmTrainingFeats = np.concatenate((gmmTrainingFeats, getSemanticsFeatures(getSemanticsData(entity, frameIdx), None, True)))
        print "added", (len(gmmTrainingFeats)-tmp), "features from", "semantic entity", "at frame", frameIdx
#     break
        
gmmTrainingFeats = gmmTrainingFeats.reshape((len(gmmTrainingFeats)/hogOrientations, hogOrientations))
gmModel = fitGMM(gmmTrainingFeats, 15)

# <codecell>

## extract features for all sprites
allFeats = {}
for entityIdx in [0] : #arange(len(trackedSprites)) :
    entityFeats = []
    for frameIdx in xrange(len(semanticEntities[entityIdx][DICT_FRAMES_LOCATIONS])) :
        feats = getSemanticsFeatures(getSemanticsData(semanticEntities[entityIdx], frameIdx), gmModel)#, False, True)
        entityFeats.append(feats)
        sys.stdout.write('\r' + "Done with frame " + np.string_(frameIdx) + " of " + np.string_(len(semanticEntities[entityIdx][DICT_FRAMES_LOCATIONS])))
        sys.stdout.flush()
    
    print
    print "done with semantic entity"
    allFeats[entityIdx] = entityFeats

# <codecell>

sio.savemat(dataPath + dataSet + "allFramesHogs", {"hogFeats":allFeats[0]})

# <codecell>

print np.sqrt(np.sum(np.power(allFeats[0][0]-allFeats[0][100], 2)))
featSize = len(allFeats[0][0])
diffVec = (allFeats[0][0]-allFeats[0][100]).reshape((featSize, 1))
print np.sqrt(np.dot(diffVec.T, diffVec))

# <codecell>

from scipy import optimize
## example optimization
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

iterNum = 0
def printStats(xk) :
    global iterNum
    global x0
    print iterNum, np.mean(np.abs(xk-x0)); sys.stdout.flush()
    iterNum += 1

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
xopt = optimize.fmin_ncg(rosen, x0, fprime=rosen_der, callback=printStats)

# <codecell>

hogFeats = sio.loadmat(dataPath + dataSet + "allFramesHogs.mat")["hogFeats"]
print hogFeats.shape

## get feats of subsequent frames
goodPairsIdxs = np.array([np.arange(len(hogFeats)-1, dtype=int), np.arange(1, len(hogFeats), dtype=int)])
print goodPairsIdxs
## ABS DIST
goodExamplesData = np.sqrt((hogFeats[goodPairsIdxs[0, :], :]-hogFeats[goodPairsIdxs[1, :], :])**2)
print goodExamplesData.shape

## get feats of random pairings that are considered bad
numBadExamples = 500
minIdxsDiff = 10
badPairsIdxs = np.sort(np.array([np.random.choice(np.arange(len(hogFeats)), numBadExamples), 
                                 np.random.choice(np.arange(len(hogFeats)), numBadExamples)]), axis=0)

print len(np.argwhere(np.abs(badPairsIdxs[0, :]-badPairsIdxs[1, :]) < minIdxsDiff)), "invalid pairs"
for pairIdx in xrange(numBadExamples) :
    idxDiff = np.abs(badPairsIdxs[0, pairIdx] - badPairsIdxs[1, pairIdx])
    tmp = idxDiff
    newPair = badPairsIdxs[:, pairIdx]
    while idxDiff < minIdxsDiff :
        newPair = np.sort(np.random.choice(np.arange(len(hogFeats)), 2))
        idxDiff = np.abs(newPair[0] - newPair[1])
#     print badPairsIdxs[:, pairIdx], newPair, tmp
    badPairsIdxs[:, pairIdx] = newPair
#     if badPairsIdxs[pairIdx, 0] - badPairsIdxs[pairIdx, 1] < minIdxsDiff

# print badPairsIdxs
print len(np.argwhere(np.abs(badPairsIdxs[0, :]-badPairsIdxs[1, :]) < minIdxsDiff)), "invalid pairs"
## ABS DIST
badExamplesData = np.sqrt((hogFeats[badPairsIdxs[0, :], :]-hogFeats[badPairsIdxs[1, :], :])**2)
print badExamplesData.shape

tic = time.time()
regressor = ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=4, verbose=0)
regressor.fit(list(np.concatenate((goodExamplesData, badExamplesData))), 
              list(np.concatenate((np.zeros(len(goodExamplesData)), np.ones(len(badExamplesData))))))
print "regressor trained in", time.time()-tic; sys.stdout.flush()

# <codecell>

allPairsHogs = []
for i in xrange(len(hogFeats)) :
    for j in xrange(i+1, len(hogFeats)) :
        ## ABS DIST
        allPairsHogs.append(np.sqrt((hogFeats[i, :]-hogFeats[j, :])**2))
        
    sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(len(hogFeats)))
    sys.stdout.flush()

# <codecell>

tic = time.time()
dists = regressor.predict(allPairsHogs)
print "distance regressed in", time.time()-tic; sys.stdout.flush()

# <codecell>

numFrames = len(hogFeats)
regressedDist = np.ones((numFrames, numFrames))
flatRegressedDist = list(np.copy(dists))
for i in xrange(numFrames-1) :
    regressedDist[i, i+1:] = flatRegressedDist[:numFrames-(i+1)]
    regressedDist[i+1:, i] = regressedDist[i, i+1:]
    del flatRegressedDist[:numFrames-(i+1)]
# print flatRegressedDist

# <codecell>

gwv.showCustomGraph(regressedDist)
## set all backwards jumps to 1 as well
regressedDist[np.arange(1, len(regressedDist)), np.arange(0, len(regressedDist)-1)] = 1.0
filteredRegressed = vtu.filterDistanceMatrix(regressedDist, 4, False)
gwv.showCustomGraph(filteredRegressed)
futureRegressed = vtu.estimateFutureCost(0.999, 2.0, filteredRegressed)

# <codecell>

probs, cumProbs = vtu.getProbabilities(filteredRegressed, 0.005, None, True)
gwv.showCustomGraph(probs)
gwv.showCustomGraph(cumProbs)

# <codecell>

bob = [638, 669]
filtSize = 4
gwv.showCustomGraph(regressedDist[bob[0]-filtSize:bob[1]-filtSize, bob[0]-filtSize:bob[1]-filtSize])
gwv.showCustomGraph(filteredRegressed[bob[0]:bob[1], bob[0]:bob[1]])
gwv.showCustomGraph(vtu.filterDistanceMatrix(regressedDist[bob[0]-filtSize*2:bob[1], bob[0]-filtSize*2:bob[1]], 4, False))

# <codecell>

# l2Dist = np.load(dataPath+"Videos/6489810.avi_distanceMatrix.npy")
gwv.showCustomGraph(l2Dist)
gwv.showCustomGraph(vtu.filterDistanceMatrix(l2Dist, 4, False))

# <codecell>

print len(np.argwhere(filteredRegressed == 0.0))
print filteredRegressed.shape
print (664-1)*2

# <headingcell level=2>

# Hack over

# <codecell>

## get random frames from each tracked sprite and get their hog features to then train the GMM
numLeaveOut = 30
numFramesPerSprite = 50
gmmTrainingFeats = np.empty(0)
for sprite in trackedSprites:
    for frameIdx in random.choice(arange(numLeaveOut, len(sprite[DICT_BBOX_CENTERS])-numLeaveOut), numFramesPerSprite, replace=False) :
        tmp = len(gmmTrainingFeats)
        gmmTrainingFeats = np.concatenate((gmmTrainingFeats, getSemanticsFeatures(getSemanticsData(sprite, frameIdx), None, True)))
        print "added", (len(gmmTrainingFeats)-tmp), "features from", sprite[DICT_SPRITE_NAME], "at frame", frameIdx
#     break
        
gmmTrainingFeats = gmmTrainingFeats.reshape((len(gmmTrainingFeats)/hogOrientations, hogOrientations))

gmModel = fitGMM(gmmTrainingFeats, 15)

# <codecell>

validationSetSize = int(len(gmmTrainingFeats)*0.1)
validationSet = random.choice(arange(len(gmmTrainingFeats)), validationSetSize, replace=False)
trainingSet = np.setdiff1d(arange(len(gmmTrainingFeats)), validationSet)

for numComponents in xrange(5, 128, 1) :
    print numComponents,
    gmModel = fitGMM(gmmTrainingFeats[trainingSet, :], numComponents)
    print gmModel.converged_, 
    print "finished training",
    sys.stdout.flush()
    gmScores = gmModel.score(gmmTrainingFeats[validationSet, :])
    print "\t->\tavg score is", np.mean(gmScores)
    sys.stdout.flush()

# <codecell>

## extract features for all sprites
allFeats = {}
for spriteIdx in [0, 3] : #arange(len(trackedSprites)) :
    spriteFeats = []
    for frameIdx in xrange(len(trackedSprites[spriteIdx][DICT_BBOXES])) :
        feats = getSemanticsFeatures(getSemanticsData(trackedSprites[spriteIdx], frameIdx), gmModel)#, False, True)
        spriteFeats.append(feats)
        sys.stdout.write('\r' + "Done with frame " + np.string_(frameIdx) + " of " + np.string_(len(trackedSprites[spriteIdx][DICT_BBOXES])))
        sys.stdout.flush()
    
    print
    print "done with sprite", spriteIdx, trackedSprites[spriteIdx][DICT_SPRITE_NAME]
    allFeats[spriteIdx] = spriteFeats

# <codecell>

## compute L2 distance between bbox centers for given sprites
spriteDistMats = {}
for spriteIdx in [0, 3] :
    bboxCenters = np.array([trackedSprites[spriteIdx][DICT_BBOX_CENTERS][x] for x in np.sort(trackedSprites[spriteIdx][DICT_BBOX_CENTERS].keys())])
    l2DistMat = np.zeros((len(bboxCenters), len(bboxCenters)))
    for c in xrange(len(bboxCenters)) :
        l2DistMat[c, c:] = np.linalg.norm(bboxCenters[c].reshape((1, 2)).repeat(len(bboxCenters)-c, axis=0) - bboxCenters[c:], axis=1)
        l2DistMat[c:, c] = l2DistMat[c, c:]
            
    spriteDistMats[spriteIdx] = vtu.filterDistanceMatrix(l2DistMat, 4, False)

# <codecell>

# ## precompute images on which the bbox of a sprite is rendered
# for spriteIdx in arange(len(trackedSprites))[0:1] :
#     allBBoxImages = []
#     imageSize = np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys()[0]])).shape[0:2]
#     for i in xrange(len(trackedSprites[spriteIdx][DICT_BBOXES])) :
#         img = np.zeros(imageSize, dtype=np.uint8)
#         cv2.drawContours(img, [int32(trackedSprites[spriteIdx][DICT_BBOXES][np.sort(trackedSprites[spriteIdx][DICT_BBOXES].keys())[i]]).reshape((4, 1, 2))], 0, 1, cv2.cv.CV_FILLED)
#         allBBoxImages.append(img)

# <codecell>

print np.argwhere(isnan(allPairFeatures))

# <codecell>

## compute pair feats for all frame pairs for each sprite individually
allSpritesPairFeats = {}
for spriteIdx in [0, 3] : ##arange(len(trackedSprites))[0:1] :
    allPairFeatures = []
    imageSize = np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys()[0]])).shape[0:2]
    for i in xrange(len(trackedSprites[spriteIdx][DICT_BBOXES])) :
        for j in xrange(i+1, len(trackedSprites[spriteIdx][DICT_BBOXES])) :
#             ## L2 DIST
#             pairFeats = np.sqrt(np.sum(np.power(allFeats[spriteIdx][i]-allFeats[spriteIdx][j],2), axis=-1))
            ## ABS DIST
            pairFeats = np.sqrt((allFeats[spriteIdx][i]-allFeats[spriteIdx][j])**2)
#             ## MEAN
#             pairFeats = np.mean(np.hstack((allFeats[spriteIdx][i], allFeats[spriteIdx][j])), axis=1)
            
            ## compute percentage of overlapping area by first drawing the bboxes onto images and then ANDing and ORing them to get instersection area and total area
#             intersection = np.logical_and(allBBoxImages[i],allBBoxImages[j])
#             totalArea = np.logical_or(allBBoxImages[i],allBBoxImages[j])
#             pairFeats = np.concatenate((pairFeats, [float(len(np.argwhere(intersection)))/float(len(np.argwhere(totalArea)))]))
            
            ## compute bbox distance
            pairFeats = np.concatenate((pairFeats, [getOverlappingSpriteTracksDistance(trackedSprites[spriteIdx], trackedSprites[spriteIdx], np.array([[i], [j]]))[0]]))

            allPairFeatures.append(pairFeats)
        sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(len(trackedSprites[spriteIdx][DICT_BBOXES])))
        sys.stdout.flush()
    
    print
    print "done with sprite", spriteIdx, trackedSprites[spriteIdx][DICT_SPRITE_NAME]
    allSpritesPairFeats[spriteIdx] = allPairFeatures

# <codecell>

print len(allSpritesPairFeats[0][1])

# <codecell>

numFrames = len(trackedSprites[spriteIdxs[0]][DICT_BBOXES])
print numFrames
visSpritePairFeats = np.zeros((numFrames, numFrames))
tmp = [allSpritesPairFeats[0][x][-5] for x in xrange(len(allSpritesPairFeats[0]))]
for i in xrange(numFrames-1) :
    visSpritePairFeats[i, i+1:] = tmp[:numFrames-(i+1)]
    visSpritePairFeats[i+1:, i] = visSpritePairFeats[i, i+1:]
    del tmp[:numFrames-(i+1)]
    sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()
    
gwv.showCustomGraph(visSpritePairFeats)

# <codecell>

numFrames = len(trackedSprites[spriteIdx][DICT_BBOXES])
regressedDist = np.zeros((numFrames, numFrames))
prevStop = 0
for i in xrange(numFrames-1) :
    regressedDist[i, i+1:] = flatRegressedDist[:numFrames-(i+1)]
    regressedDist[i+1:, i] = regressedDist[i, i+1:]
    del flatRegressedDist[:numFrames-(i+1)]
print flatRegressedDist

# <codecell>

## get pairs with smallest current distance to use as good examples of labelled pairs
spriteIdx = 0
spriteNumFrames = spriteDistMats[spriteIdx].shape[0]
upperTriangleIndices = np.argwhere(np.triu(np.ones((spriteNumFrames, spriteNumFrames)), k=1) == 1)
bestFirstPairs = np.argsort(spriteDistMats[spriteIdx][upperTriangleIndices[:, 0], upperTriangleIndices[:, 1]])
bestPercentageToUse = 0.1
pairsToUseAsLabelled = upperTriangleIndices[bestFirstPairs, :][:int(bestPercentageToUse*len(bestFirstPairs))]

pairsToUseAsLabelledDists = spriteDistMats[spriteIdx][pairsToUseAsLabelled[:, 0], pairsToUseAsLabelled[:, 1]].reshape((len(pairsToUseAsLabelled), 1))
pairsToUseAsLabelledDists /= np.max(spriteDistMats[spriteIdx])
# print np.concatenate((pairsToUseAsLabelled, np.ones((len(pairsToUseAsLabelled), 1))), axis=-1)
print np.concatenate((pairsToUseAsLabelled, pairsToUseAsLabelledDists), axis=-1)
print len(pairsToUseAsLabelled)

# <codecell>

gwv.showCustomGraph(spriteDistMats[0])

# <codecell>

## Get labelled pairs of compatible/not compatible frames and learn compatibility

# userLabelledExamples = np.array([[180, 185, 0.0], 
#                           [173, 164, 0.0], 
#                           [87, 368, 1.0], 
#                           [47, 182, 1.0], 
#                           [223, 304, 1.0], 
#                           [329, 418, 1.0], 
#                           [403, 468, 1.0], 
#                           [490, 72, 1.0]])


userLabelledExamples = np.array([[180, 185, 0.0], 
                          [173, 164, 0.0], 
                          [87, 368, 1.0]])

labelledPairs = np.concatenate((np.concatenate((pairsToUseAsLabelled+4, pairsToUseAsLabelledDists), axis=-1), 
                                userLabelledExamples), axis=0)

# labelledPairs = np.array([[180, 185, 0.0], 
#                           [173, 164, 0.0], 
#                           [87, 368, 1.0], 
#                           [47, 182, 1.0], 
#                           [223, 304, 1.0], 
#                           [329, 418, 1.0], 
#                           [403, 468, 1.0], 
#                           [490, 72, 1.0]])

labelledData = []
for pair in labelledPairs :
#     ## L2 DIST
#     pairFeats = np.sqrt(np.sum(np.power(allFeats[0][int(pair[0])]-allFeats[0][int(pair[1])],2), axis=-1))
    ## ABS DIST
    pairFeats = np.sqrt((allFeats[0][int(pair[0])]-allFeats[0][int(pair[1])])**2)
#     ## MEAN
#     pairFeats = np.mean(np.hstack((allFeats[0][int(pair[0])], allFeats[0][int(pair[1])])), axis=1)
    labelledData.append(pairFeats)

tic = time.time()
try :
    del regressor
except :
    print "no regressor"
regressor = ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=4, verbose=0)
regressor.fit(labelledData, list(labelledPairs[:, -1]))
print "regressor trained in", time.time()-tic

tic = time.time()
dists = regressor.predict(allSpritesPairFeats[spriteIdx])
print "distance regressed in", time.time()-tic
flatRegressedDist = list(np.copy(dists))

# <codecell>

numFrames = len(trackedSprites[spriteIdx][DICT_BBOXES])
regressedDist = np.zeros((numFrames, numFrames))
prevStop = 0
for i in xrange(numFrames-1) :
    regressedDist[i, i+1:] = flatRegressedDist[:numFrames-(i+1)]
    regressedDist[i+1:, i] = regressedDist[i, i+1:]
    del flatRegressedDist[:numFrames-(i+1)]
print flatRegressedDist

# <codecell>

gwv.showCustomGraph(regressedDist)

# <headingcell level=2>

# From here on it's about compatibility learning

# <codecell>

for spriteIdxs in [[0, 3]] :#arange(len(trackedSprites))[0:1] :
    allSpritePairsPairFeatures = []
#     imageSize = np.array(Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys()[0]])).shape[0:2]
    for i in xrange(len(trackedSprites[spriteIdxs[0]][DICT_BBOXES])) :
        tmp = len(allSpritePairsPairFeatures)
        for j in xrange(len(trackedSprites[spriteIdxs[1]][DICT_BBOXES])) :
#             ## L2 DIST
#             pairFeats = np.sqrt(np.sum(np.power(allFeats[spriteIdx][i]-allFeats[spriteIdx][j],2), axis=-1))
            ## ABS DIST
            pairFeats = np.sqrt((allFeats[spriteIdxs[0]][i]-allFeats[spriteIdxs[1]][j])**2)
#             ## MEAN
#             pairFeats = np.mean(np.hstack((allFeats[spriteIdx][i], allFeats[spriteIdx][j])), axis=1)
            
            ## compute percentage of overlapping area by first drawing the bboxes onto images and then ANDing and ORing them to get instersection area and total area
#             intersection = np.logical_and(allBBoxImages[i],allBBoxImages[j])
#             totalArea = np.logical_or(allBBoxImages[i],allBBoxImages[j])
#             pairFeats = np.concatenate((pairFeats, [float(len(np.argwhere(intersection)))/float(len(np.argwhere(totalArea)))]))

            ## compute bbox distance
            pairFeats = np.concatenate((pairFeats, [getOverlappingSpriteTracksDistance(trackedSprites[spriteIdxs[0]], trackedSprites[spriteIdxs[1]], np.array([[i], [j]]))[0]]))

            allSpritePairsPairFeatures.append(pairFeats)
        sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(len(trackedSprites[spriteIdxs[0]][DICT_BBOXES])) + " added " + np.string_(len(allSpritePairsPairFeatures)-tmp))
        sys.stdout.flush()
    
    print
    print "done with sprite pair", spriteIdxs, trackedSprites[spriteIdxs[0]][DICT_SPRITE_NAME], trackedSprites[spriteIdxs[1]][DICT_SPRITE_NAME]

# <codecell>

print len(allSpritePairsPairFeatures[0])

# <codecell>

numFrames = np.array([len(trackedSprites[spriteIdxs[0]][DICT_BBOXES]), len(trackedSprites[spriteIdxs[1]][DICT_BBOXES])])
print numFrames
visSpritePairsPairFeats = np.zeros((numFrames[0], numFrames[1]))
# tmp = list(np.copy(allSpritePairsPairFeatures))
# print len(tmp); sys.stdout.flush()
# prevStop = 0 
for i in xrange(numFrames[0]) :
#     print i, numFrames[1], prevStop, prevStop+numFrames[1]
#     prevStop += numFrames[1]
#     visSpritePairsPairFeats[i, :] = np.sum(np.array(tmp[:numFrames[1]]), axis=-1)
#     visSpritePairsPairFeats[i, :] = np.sum(np.array(allSpritePairsPairFeatures[i*numFrames[1]:(i+1)*numFrames[1]]), axis=-1)

#     visSpritePairsPairFeats[i, :] = np.sum(np.array(allSpritePairsPairFeatures[i*numFrames[1]:(i+1)*numFrames[1]])[:, :-4], axis=-1)
    visSpritePairsPairFeats[i, :] = np.array(allSpritePairsPairFeatures[i*numFrames[1]:(i+1)*numFrames[1]])[:, -1]
    
#     classifiedCompatibilities[i+1:, i] = regressedDist[i, i+1:]
#     del tmp[:numFrames[1]]
    sys.stdout.write('\r' + "Done with row " + np.string_(i) + " of " + np.string_(numFrames[0]))
    sys.stdout.flush()
# print len(tmp)

# <codecell>

gwv.showCustomGraph(visSpritePairsPairFeats)

# <codecell>

## fictitious generated sequence such that white_bus1 and red_car1 clash at frames 1011 and 338 respectively
sequenceLength = 1350
generatedSequence = {}
generatedSequence[0] = np.zeros(sequenceLength, dtype=int)
generatedSequence[0][679:679+numFrames[0]+1] = arange(numFrames[0]+1)
generatedSequence[3] = np.zeros(sequenceLength, dtype=int)
generatedSequence[3][6:6+numFrames[1]+1] = arange(numFrames[1]+1)

## user marks pair 338 - 1011 as incompatible but do -1 as frame 0 is the invisible frame
incompatiblePair = np.array([338, 1011]) - 1
incompatiblePairDist = visSpritePairsPairFeats[incompatiblePair[0], incompatiblePair[1]]
pairingsInSequence = np.concatenate((generatedSequence[0].reshape((1, sequenceLength)), generatedSequence[3].reshape((1, sequenceLength))), axis=0)-1
pairingsInSequence = pairingsInSequence[:, np.all(pairingsInSequence >= 0, axis=0)]
print pairingsInSequence
# figure(); plot(visSpritePairsPairFeats[pairingsInSequence[0, :], pairingsInSequence[1, :]] - incompatiblePairDist)
# plot(arange(len(pairingsInSequence.T)), zeros(len(pairingsInSequence.T)))

## since the distance matrix is filtered I need to get rid of some frames
# get rid of first 4 frames
validPairingsForDist = np.all(pairingsInSequence >= 4, axis=0) ## 4 is the size of the filter
# get rid of last 4 frames
validPairingsForDist = np.all((validPairingsForDist, np.all(pairingsInSequence < (numFrames-4).reshape((2, 1)), axis=0)), axis=0) ## 4 is the size of the filter
validPairingsIdxs = np.ndarray.flatten(np.argwhere(validPairingsForDist))
# print validPairingsForDist
pairingsIHaveDistFor = pairingsInSequence[:, validPairingsForDist]-4
print pairingsIHaveDistFor

## now get distances of overlapping frames to the labelled frames according to their respective distance matrices
distsToLabelledFrames = np.zeros(pairingsIHaveDistFor.shape)
distsToLabelledFrames[0, :] = spriteDistMats[0][incompatiblePair[0], pairingsIHaveDistFor[0, :]]
distsToLabelledFrames[1, :] = spriteDistMats[3][incompatiblePair[1], pairingsIHaveDistFor[1, :]]

# figure(); plot(distsToLabelledFrames[0, :])
# figure(); plot(distsToLabelledFrames[1, :])
# figure(); plot(np.sum(distsToLabelledFrames, axis=0))
totalDists = np.sum(distsToLabelledFrames, axis=0)
averageDist = np.mean(totalDists)
## pairs who's total distance to their respective labelled frames are smaller than some percentage of the mean
percentage = 0.9
print averageDist*percentage
pairsToUseAsCompatible = pairingsInSequence[:, validPairingsIdxs[np.negative(totalDists < averageDist*percentage)]]
print pairsToUseAsCompatible.shape
# print validPairingsIdxs

# <codecell>

## Get labelled pairs and try to learn a distance
## 0 = incompatible, 1 = compatible
# userLabelledExamples = np.array([[165, 73, 0, 3, 1],
#                           [475, 1104, 0, 3, 1],
#                           [377, 852, 0, 3, 1],
#                           [222, 1139, 0, 3, 1],
#                           [340, 1020, 0, 3, 0],
#                           [313, 1049, 0, 3, 0],
#                           [382, 1000, 0, 3, 0],
#                           [332, 1083, 0, 3, 0]])
# userLabelledExamples = np.array([[165, 73, 0, 3, 1],
#                           [470, 1154, 0, 3, 1],
#                           [73, 200, 0, 3, 1],
#                           [150, 500, 0, 3, 1],
#                           [360, 450, 0, 3, 1],
#                           [34, 1210, 0, 3, 1],
#                           [340, 1020, 0, 3, 0]])
userLabelledExamples = np.array([[337, 1010, 0, 3, 0],
                                 [340, 1020, 0, 3, 0]])

labelledPairs = np.concatenate((np.concatenate((pairsToUseAsCompatible.T, np.array([[0, 3, 1]]).repeat(len(pairsToUseAsCompatible.T), axis=0)), axis=-1), 
                                userLabelledExamples), axis=0)

labelledData = []
for pair in labelledPairs :
#     ## L2 DIST
#     pairFeats = np.sqrt(np.sum(np.power(allFeats[0][int(pair[0])]-allFeats[0][int(pair[1])],2), axis=-1))
    ## ABS DIST
    pairFeats = np.sqrt((allFeats[pair[2]][pair[0]]-allFeats[pair[3]][pair[1]])**2)
#     ## MEAN
#     pairFeats = np.mean(np.hstack((allFeats[0][int(pair[0])], allFeats[0][int(pair[1])])), axis=1)
    
    ## compute bbox distance
    pairFeats = np.concatenate((pairFeats, [getOverlappingSpriteTracksDistance(trackedSprites[pair[2]], trackedSprites[pair[3]], np.array([[pair[0]], [pair[1]]]))[0]]))
    
    labelledData.append(pairFeats)

tic = time.time()
try :
    del classifier
except :
    print "no classifier"
    
classifier = ensemble.ExtraTreesClassifier(n_estimators=100, n_jobs=4, verbose=0)#, max_features=None, criterion='entropy')
# classifier = svm.LinearSVC()
classifier.fit(np.array(labelledData)[:, -4:], list(labelledPairs[:, -1]))
print "classifier trained in", time.time()-tic; sys.stdout.flush()

tic = time.time()
compatibilities = classifier.predict(np.array(allSpritePairsPairFeatures)[:, -4:])
print "compatibility found in", time.time()-tic; sys.stdout.flush()
flatCompatibilities = list(np.copy(compatibilities))

# <codecell>

numFrames = np.array([len(trackedSprites[spriteIdxs[0]][DICT_BBOXES]), len(trackedSprites[spriteIdxs[1]][DICT_BBOXES])])
print numFrames
classifiedCompatibilities = np.zeros((numFrames[0], numFrames[1]))
# prevStop = 0 
for i in xrange(numFrames[0]) :
#     print i, numFrames[1], prevStop, prevStop+numFrames[1]
#     prevStop += numFrames[1]
    classifiedCompatibilities[i, :] = flatCompatibilities[:numFrames[1]]
#     classifiedCompatibilities[i+1:, i] = regressedDist[i, i+1:]
    del flatCompatibilities[:numFrames[1]]
print len(flatCompatibilities)

# <codecell>

gwv.showCustomGraph(classifiedCompatibilities)

# <codecell>

img = cv2.cvtColor(cv2.imread(trackedSprites[0][DICT_FRAMES_LOCATIONS][2094]), cv2.COLOR_BGR2GRAY)
contours, h = cv2.findContours(img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

# <codecell>

print [int32(trackedSprites[0][DICT_BBOXES][np.sort(trackedSprites[0][DICT_BBOXES].keys())[180]]).reshape((4, 1, 2))]
print contours

# <codecell>

blank = np.zeros(img.shape[0:2])
img1 = blank.copy()
cv2.drawContours(img1, [int32(trackedSprites[0][DICT_BBOXES][np.sort(trackedSprites[0][DICT_BBOXES].keys())[180]]).reshape((4, 1, 2))], 0, 1, cv2.cv.CV_FILLED)
img2 = blank.copy()
cv2.drawContours(img2, [int32(trackedSprites[0][DICT_BBOXES][np.sort(trackedSprites[0][DICT_BBOXES].keys())[450]]).reshape((4, 1, 2))], 0, 1, cv2.cv.CV_FILLED)
intersection = np.logical_and(img1,img2)
totalArea = np.logical_or(img1,img2)
print len(np.argwhere(intersection)), len(np.argwhere(totalArea))
# img2 = blank.copy()
# cv2.drawContours(img2,contours, 0, cv2.cv.Scalar(255, 255, 255), cv2.cv.CV_FILLED)
# cv2.findContours(
# figure(); imshow(totalArea)
# print img2
# cv2.drawContours(

# <codecell>

semData = getSemanticsData(sprite, frameIdx) #getSemanticsData(trackedSprites[0], 450)
patchSize = np.array(semData[DATA_MASK].shape)
print patchSize
allIdxs = -np.ones(patchSize, dtype=int)
numRowIdxs = len(np.arange(0, patchSize[0], pixelsPerCell)) - 1
numColIdxs = len(np.arange(0, patchSize[1], pixelsPerCell)) - 1
print numRowIdxs
print numColIdxs
gridIdxs = np.arange(numColIdxs, dtype=int).reshape((1, numColIdxs)).repeat(numRowIdxs*pixelsPerCell, axis=0).repeat(pixelsPerCell, axis=-1)
gridIdxs += np.arange(numRowIdxs, dtype=int).reshape((numRowIdxs, 1)).repeat(numColIdxs*pixelsPerCell, axis=-1).repeat(pixelsPerCell, axis=0)*numColIdxs
allIdxs[:gridIdxs.shape[0], :gridIdxs.shape[1]] = gridIdxs
gwv.showCustomGraph(allIdxs)

visiblePixels = np.argwhere(semData[DATA_MASK] != 0)
print np.unique(allIdxs[visiblePixels[:, 0], visiblePixels[:, 1]])
hogsToKeep = np.zeros(numRowIdxs*numColIdxs, dtype=bool)
hogsToKeep[np.unique(allIdxs[visiblePixels[:, 0], visiblePixels[:, 1]])] = True
print hogsToKeep
gwv.showCustomGraph(semData[DATA_MASK])
gca().set_autoscale_on(False)
scatter(colCoords, rowCoords)
scatter(colCoords[hogsToKeep], rowCoords[hogsToKeep], c='r')

# <codecell>

print visiblePixels.shape

# <codecell>

rowCoords = np.arange(pixelsPerCell/2, patchSize[0]-pixelsPerCell/2, pixelsPerCell)
colCoords = np.arange(pixelsPerCell/2, patchSize[1]-pixelsPerCell/2, pixelsPerCell)

cellGridRows = len(rowCoords)#int(np.round(float(patchSize[0]-pixelsPerCell/2)/pixelsPerCell))
cellGridCols = len(colCoords)#int(np.round(float(patchSize[1]-pixelsPerCell/2)/pixelsPerCell))

rowCoords = rowCoords.reshape((1, cellGridRows)).repeat(cellGridCols)
colCoords = np.ndarray.flatten(colCoords.reshape((1, cellGridCols)).repeat(cellGridRows, axis=0))

## check which centers are within the mask and only keep those
gwv.showCustomGraph(semData[DATA_MASK])
gca().set_autoscale_on(False)
scatter(colCoords, rowCoords)
scatter(colCoords[semData[DATA_MASK][rowCoords, colCoords] != 0], rowCoords[semData[DATA_MASK][rowCoords, colCoords] != 0], c='r')
# hogFeats = hogFeats[(semanticsData[DATA_MASK][rowCoords, colCoords] != 0).repeat(hogOrientations)]

# <codecell>

print feats.shape[0]/hogOrientations
patchSize = np.array([132, 194])
cellGridRows = int(np.round(float(patchSize[0]-pixelsPerCell/2)/pixelsPerCell))
cellGridCols = int(np.round(float(patchSize[1]-pixelsPerCell/2)/pixelsPerCell))
rowCoords = np.arange(pixelsPerCell/2, patchSize[0], pixelsPerCell).reshape((1, cellGridRows)).repeat(cellGridCols)
colCoords = np.ndarray.flatten(np.arange(pixelsPerCell/2, patchSize[1], pixelsPerCell).reshape((1, cellGridCols)).repeat(cellGridRows, axis=0))
print rowCoords
print colCoords

figure(); imshow(vis, interpolation='nearest'); gca().set_autoscale_on(False); scatter(colCoords, rowCoords, c='r')
figure(); imshow(mask, interpolation='nearest')
# ylim(131, 0)
# xlim(0, 193)

