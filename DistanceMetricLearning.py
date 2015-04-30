# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab

from PIL import Image
from PySide import QtCore, QtGui

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
from scipy import ndimage
from scipy import stats

from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.datasets.samples_generator import make_blobs
from skimage.feature import hog
from skimage import color

import ComputeGridFeatures as cgf
import GraphWithValues as gwv

dataFolder = "/home/ilisescu/PhD/data/"
supportedFrameExt = [".png"]

app = QtGui.QApplication(sys.argv)

# <codecell>

## define feature types
# ALL = 0 ## compute all possible features
# ALL_FRAME = 1 ## compute all possible frame features
# ALL_PAIR = 2 ## compute all possible pairwise features
idx = 0
## single frame feature
STD_RGB = idx; idx +=1 ## compute standard rgb frame features (i.e. each variable is a 3D RGB vector)
FULL_RGB = idx; idx +=1 ## compute full rgb frame features (i.e. each RGB component is a variable)
HOG_FEAT = idx; idx +=1 ## compute HOG features

## frame pair feature
L2_DIST = idx; idx +=1 ## compute l2 distance between RGB pixel pairs
ABS_DIFF = idx; idx +=1 ## compute pairwise absolute difference feature
AVG = idx; idx +=1 ## compute average

## encoding
K_MEANS_HIST = idx; idx +=1
FISHER = idx; idx +=1

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
hogOrientations = 8

# <codecell>

frameLocation = frames[570]
matteLocation = mattes[570]

frameData = cv2.cvtColor(cv2.imread(frameLocation), cv2.COLOR_BGR2RGB)
frameData = np.array(frameData, dtype=np.float32)
#     figure(); imshow(frameData)
if matteLocation != None and os.path.splitext(frameLocation)[-1] in supportedFrameExt:
    matte = cv2.cvtColor(cv2.imread(matteLocation), cv2.COLOR_BGR2GRAY)
    matte = np.array(matte)/255.0
    frameData = frameData*np.repeat(np.reshape(matte, (matte.shape[0], matte.shape[1], 1)), 3, axis=-1)

feats, featsImg = hog(color.rgb2gray(frameData), orientations=hogOrientations, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)

# <codecell>

# figure(); imshow(featsImg, interpolation='nearest')
figure(); imshow(np.array(frameData, dtype=np.uint8))

# <codecell>

############# GETTING FEATURES FROM FRAMES ###############

## compute features for image
blocksPerWidth = 32
blocksPerHeight = 48
subDivisions = blocksPerWidth*blocksPerHeight

## given block sizes and img sizes build indices representing each block
imageSize = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)).shape
stencils = cgf.stencil2D(blocksPerWidth, blocksPerHeight, imageSize)

features = cgf.histFgFeatures(stencils, subDivisions, frames, mattes)
# features = np.load("tmpfeat.npy")
figure(); imshow(features.T, interpolation='nearest')

# np.save("tmpfeat.npy", features)

# <codecell>

def getKMeansHistEncoding(feats, featsShape, kmeansClusterer) :
    """Take feats feature vector and encode it using K-Means Histograms
    
           feats: feature vector
           featsShape: shape that feats had when the clusterer was trained (if e.g. feats are hog features and it has been flattened)
           kmeansClusterer: a trained sklearn.cluster.KMeans clusterer
           
        return: 1D encoded feature vector encodedFeats"""
    
    if kmeansClusterer != None :
        try :
            bins = len(kmeansClusterer.cluster_centers_)
            encodedFeats = np.histogram(kmeansClusterer.predict(np.reshape(feats, featsShape)), bins=bins)[0]
            return encodedFeats
        except Exception :
            raise Exception, "Untrained kmeansClusterer"
    else :
        print "No kmeansClusterer has been specified"
        return feats
    
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

def getStdRGBFeats(imageData) :
    """Take RGB imageData and reshape it as a W*Hx3 vector
    
           imageData: input RGB image data
           
        return: 3D vector feats"""
    
    feats = np.reshape(imageData, (np.prod(imageData.shape[0:-1]), 3))
    return feats

def getFullRGBFeats(imageData) :
    """Take RGB imageData and reshape it as a 1D vector
    
           imageData: input RGB image data
           
        return: 1D vector feats"""
    
    feats = np.reshape(imageData, (np.prod(imageData.shape)))
    return feats

def getHOGFeats(imageData) :
    """Take RGB imageData and compute HOG features
    
           imageData: input RGB image data
           
        return: 1D vector feats"""
    
    feats = hog(color.rgb2gray(imageData), orientations=hogOrientations, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=False)
    return feats
    
def getFrameFeatures(frameLocation, featureTypes, **kwargs) :
    """Reads image at frameLocation and returns formatted features specified in featureTypes.
    
           frameLocation: location on disk of an image
           featureTypes: array containing the types of features that need to be computed for the image
           
           accepted kwargs:
               downsampleFactor[float]: if > 1.0 the frame and relative matte will be downsampled
               matteLocation[string]: location on disk of its background substracting matte
               kmeansClusterer[sklearn.cluster.KMeans]: trained kmeans clusterer
               gmModel[sklearn.mixture.GMM]: trained gaussian mixture model
           
           
        return: 1D vector features"""
    
    if os.path.splitext(frameLocation)[-1] not in supportedFrameExt :
        raise Exception("Image type not supported")
        
    if STD_RGB in featureTypes and (FULL_RGB in featureTypes or HOG_FEAT in featureTypes) :
        raise Exception("Can't use STD_RGB and FULL_RGB together")
    
    ## accepted kwargs
    downsampleFactor = 1.0
    matteLocation = None
    kmeansClusterer = None
    gmModel = None
    ## parse kwargs
    if "downsampleFactor" in kwargs.keys() :
        downsampleFactor = kwargs["downsampleFactor"]
    if "matteLocation" in kwargs.keys() :
        matteLocation = kwargs["matteLocation"]
    if "kmeansClusterer" in kwargs.keys() :
        kmeansClusterer = kwargs["kmeansClusterer"]
    if "gmModel" in kwargs.keys() :
        gmModel = kwargs["gmModel"]
        
    frameData = cv2.cvtColor(cv2.imread(frameLocation), cv2.COLOR_BGR2RGB)
    if downsampleFactor > 1.0 :
        frameData = cv2.resize(frameData, (int(frameData.shape[1]/downsampleFactor), int(frameData.shape[0]/downsampleFactor)), interpolation=cv2.INTER_AREA)
    frameData = np.array(frameData, dtype=np.float32)
#     figure(); imshow(frameData)
    if matteLocation != None and os.path.splitext(frameLocation)[-1] in supportedFrameExt:
        matte = cv2.cvtColor(cv2.imread(matteLocation), cv2.COLOR_BGR2GRAY)
        if downsampleFactor > 1.0 :
            matte = cv2.resize(matte, (int(matte.shape[1]/downsampleFactor), int(matte.shape[0]/downsampleFactor)), interpolation=cv2.INTER_AREA)
        matte = np.array(matte)/255.0
        
        frameData = frameData*np.repeat(np.reshape(matte, (matte.shape[0], matte.shape[1], 1)), 3, axis=-1)
#     figure(); imshow(frameData)
    features = np.empty(0)
    ## init feature vecotr
    if FULL_RGB in featureTypes or HOG_FEAT in featureTypes :
        features = np.empty(0)
    elif STD_RGB in featureTypes :
        features = np.empty((0, 3))
          
    if STD_RGB in featureTypes :#or ALL_FRAME in featureTypes or ALL in featureTypes :
        features = np.concatenate((features, getStdRGBFeats(frameData)))
        
    if FULL_RGB in featureTypes :#or ALL_FRAME in featureTypes or ALL in featureTypes :
        features = np.concatenate((features, getFullRGBFeats(frameData)))
        
    if HOG_FEAT in featureTypes :#or ALL_FRAME in featureTypes or ALL in featureTypes :
        features = np.concatenate((features, getHOGFeats(frameData)))
        
    if K_MEANS_HIST in featureTypes :
        if HOG_FEAT in frameFeatTypes :
            features = getKMeansHistEncoding(features, (len(features)/hogOrientations, hogOrientations), kmeansClusterer)
        else :
            features = getKMeansHistEncoding(features, features.shape, kmeansClusterer)
        
    if FISHER in featureTypes :
        if HOG_FEAT in frameFeatTypes :
            features = getFisherEncoding(features, (len(features)/hogOrientations, hogOrientations), gmModel)
        else :
            features = getFisherEncoding(features, features.shape, gmModel)
        
    return features

# <codecell>

def getL2Dist(image1Data, image2Data) :
    """Take image1Data and image2Data and compute l2 distance between every pair of corresponding N-D data points (e.g. RGB pixels)
    
           image1Data: input RGB image data
           image1Data: input RGB image data
           
        return: 1D vector feats"""
    
    feats = np.sqrt(np.sum(np.power(image1Data-image2Data,2), axis=-1))
    return feats

def getAbsDiff(image1Data, image2Data) :
    """Take image1Data and image2Data and compute absolute difference between every pair of corresponding data points
    
           image1Data: input RGB image data
           image1Data: input RGB image data
           
        return: 1D vector feats"""
    
    feats = np.sqrt((image1Data-image2Data)**2)
    return feats

def getAvg(image1Data, image2Data) :
    """Take image1Data and image2Data and compute feature average between every pair of corresponding data points
    
           image1Data: input RGB image data
           image1Data: input RGB image data
           
        return: 1D vector feats"""
    
    if len(image1Data.shape) == 1 :
        feats = np.mean(np.hstack((image1Data.reshape((len(image1Data), 1)),image2Data.reshape((len(image2Data), 1)))), axis=1)
    else :
        feats = np.mean(np.hstack((image1Data,image2Data)), axis=1)
    return feats
    
def getFramesPairFeatures(frame1Features, frame2Features, featureTypes, **kwargs) :
    """Takes the feature vectors of 2 frames and computes the pairwise features specified in featureTypes
    
           frame1Features: features of frame 1
           frame2Features: features of frame 2
           featureTypes: array containing the types of pairwise features to compute for the 2 given frames
           
           accepted kwargs:
               kmeansClusterer[sklearn.cluster.KMeans]: trained kmeans clusterer
               gmModel[sklearn.mixture.GMM]: trained gaussian mixture model
           
        return: 1D vector features"""
    
    ## accepted kwargs
    kmeansClusterer = None
    gmModel = None
    ## parse kwargs
    if "kmeansClusterer" in kwargs.keys() :
        kmeansClusterer = kwargs["kmeansClusterer"]
    if "gmModel" in kwargs.keys() :
        gmModel = kwargs["gmModel"]
    
    if L2_DIST in featureTypes and ABS_DIFF in featureTypes :
        raise Exception("Can't use L2_DIST and ABS_DIFF together")
    
    features = np.empty(0)
    if L2_DIST in featureTypes :#or ALL_PAIR in featureTypes or ALL in featureTypes :
        features = np.concatenate((features, getL2Dist(frame1Features, frame2Features)))
        
    if ABS_DIFF in featureTypes :#or ALL_PAIR in featureTypes or ALL in featureTypes :
        features = np.concatenate((features, getAbsDiff(frame1Features, frame2Features)))
        
    if AVG in featureTypes :#or ALL_PAIR in featureTypes or ALL in featureTypes :
        features = np.concatenate((features, getAvg(frame1Features, frame2Features)))
        
    if K_MEANS_HIST in featureTypes :
        if HOG_FEAT in frameFeatTypes :
            features = getKMeansHistEncoding(features, (len(features)/hogOrientations, hogOrientations), kmeansClusterer)
        else :
            features = getKMeansHistEncoding(features, features.shape, kmeansClusterer)
        
    if FISHER in featureTypes :
        if HOG_FEAT in frameFeatTypes :
            features = getFisherEncoding(features, (len(features)/hogOrientations, hogOrientations), gmModel)
        else :
            features = getFisherEncoding(features, features.shape, gmModel)
        
    return features

# <codecell>

## example encoding
print getFrameFeatures(frames[0], [HOG_FEAT, K_MEANS_HIST], matteLocation=mattes[0], kmeansClusterer=clusterer)
print getFramesPairFeatures(frameFeatures[0], frameFeatures[1], [ABS_DIFF, K_MEANS_HIST], kmeansClusterer=clusterer)

# <codecell>

def fitKmeansClusterer(trainingFeats, bins) :
    """Takes a set of features trainingFeats and returns a kmeans clusterer with bins clusters
        Descriptors of an image can be assigned to one cluster center for the purpose of doing histogram encoding
    
           trainingFeats: training features
           bins: number of k-means centers to find and use as words
           
        return: kmeans clusterer"""
    
    clusterer = KMeans(n_clusters=bins, n_jobs=6)
    clusterer.fit(trainingFeats)
    return clusterer

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

frameFeatTypes = [HOG_FEAT]
pairFeatTypes = [ABS_DIFF, FISHER]#[L2_DIST]#[L2_DIST, AVG]

# <codecell>

## compute HOG features of labelled frames and transform it to be (N*M)xb where N is number of individual frames
## and M and b are number of cells and number of bins in HOG respectively
# labelledPairs = np.load("labelledPairs.npy")
labelledPairs = np.load("labelledPairs_better.npy")
labelledFramesFeats = []

## compute the cluster centers on the single frame features
if False :
    alreadyComputed = []
    for labelledPair in labelledPairs :
        f1 = int(labelledPair[0])
        f2 = int(labelledPair[1])
        if f1 not in alreadyComputed :
            print f1
            alreadyComputed.append(f1)
            labelledFramesFeats.append(getFrameFeatures(frames[f1], frameFeatTypes, mattes[f1]))
        else :
            print "skipping", f1
            
        if f2 not in alreadyComputed :
            print f2
            alreadyComputed.append(f2)
            labelledFramesFeats.append(getFrameFeatures(frames[f2], frameFeatTypes, mattes[f2]))
        else :
            print "skipping", f2
## compute the cluster centers on the pairwise features
else :
    for labelledPair, i in zip(labelledPairs, arange(len(labelledPairs))) :
        frame1Loc = frames[int(labelledPair[0])]
        frame1Matte = mattes[int(labelledPair[0])]
        frame2Loc = frames[int(labelledPair[1])]
        frame2Matte = mattes[int(labelledPair[1])]
        labelledFramesFeats.append(getFramesPairFeatures(getFrameFeatures(frame1Loc, frameFeatTypes, matteLocation=frame1Matte), getFrameFeatures(frame2Loc, frameFeatTypes, matteLocation=frame2Matte), pairFeatTypes))
        
        sys.stdout.write('\r' + "Done " + np.string_(i+1) + " pairs: " + np.string_(int(labelledPair[0])) + " - " + np.string_(int(labelledPair[1])))
        sys.stdout.flush()
        
flatLabelledData = np.ndarray.flatten(np.array(labelledFramesFeats))
flatLabelledData = flatLabelledData.reshape((len(flatLabelledData)/hogOrientations, hogOrientations))
print flatLabelledData.shape
tic = time.time()
clusterer = fitKmeansClusterer(flatLabelledData, 256)
print time.time()-tic

# <codecell>

tic = time.time()
clusterer = fitKmeansClusterer(flatLabelledData, 256)
print time.time()-tic

# <codecell>

tmp = KMeans(n_clusters=256, n_jobs=6)
print len(clusterer.cluster_centers_)

try :
    tmp.cluster_centers_ 
    print len(tmp.cluster_centers_)
except Exception :
    raise Exception, "la"

# <codecell>

## compute frame features for all frames
frameFeatures = []
downsampleFactor = 1.0
for i in xrange(0, len(frames)) :
    frameFeatures.append(getFrameFeatures(frames[i], frameFeatTypes, downsampleFactor=downsampleFactor, matteLocation=mattes[i]))
    print i, 

# <codecell>

## try finding best number of gaussians for gmm
labelledPairs = np.load("labelledPairs_better.npy")
# labelledPairs = np.load("appearanceLabelledPairs_setchoice.npy")
labelledData = []
for labelledPair in labelledPairs :
    labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], pairFeatTypes))

labelledData = np.ndarray.flatten(np.array(labelledData))
labelledData = labelledData.reshape((len(labelledData)/hogOrientations, hogOrientations))
print labelledData.shape

# <codecell>

print gmModel.converged_
GMM(n_components=3).converged_

# <codecell>

validationSetSize = int(len(labelledData)*0.1)
validationSet = random.choice(arange(len(labelledData)), validationSetSize, replace=False)
trainingSet = np.setdiff1d(arange(len(labelledData)), validationSet)

for numComponents in xrange(5, 64, 1) :
    print numComponents,
    gmModel = fitGMM(labelledData[trainingSet, :], numComponents)
    print gmModel.converged_, 
    print "finished training",
    sys.stdout.flush()
    gmScores = gmModel.score(labelledData[validationSet, :])
    print "\t->\tavg score is", np.mean(gmScores)
    sys.stdout.flush()

# <codecell>

numComponents = 10
# gmModel = fitGMM(labelledData, numComponents)

# <codecell>

tic = time.time()
modelMeans = gmModel.means_
print modelMeans.shape
imgData = labelledData[0:frameFeatures[0].shape[0]/hogOrientations, :]
featDim = imgData.shape[-1]
print imgData.shape
##get prior probability pi for each component
priors = gmModel.weights_
print priors.shape
##get posterior probabilities q for each data point and each component
posteriors = gmModel.predict_proba(imgData)
print posteriors.shape

us = np.empty(0)
vs = np.empty(0)
## this one uses the formulation given by vlfeat
for k in xrange(numComponents) :
    ## get covariance matrix for component k
    kCompCov = gmModel.covars_[k, :]
    
    kCompMeansRep = modelMeans[k, :].reshape((1, featDim)).repeat(imgData.shape[0], axis=0)
    kCompCovRep = kCompCov.reshape((1, featDim)).repeat(imgData.shape[0], axis=0)
    kCompPostRep = posteriors[:, k].reshape((imgData.shape[0], 1)).repeat(featDim, axis=-1)
    
    uk = np.sum(kCompPostRep*(imgData-kCompMeansRep)/kCompCovRep, axis=0)
    uk /= (imgData.shape[0]*np.sqrt(priors[k]))
    us = np.concatenate((us, uk))
    
    vk = np.sum(kCompPostRep*((((imgData-kCompMeansRep)/kCompCovRep)**2)-1), axis=0)
    vk /= (imgData.shape[0]*np.sqrt(2*priors[k]))
    vs = np.concatenate((vs, vk))
    
encodedFeats = np.concatenate((us, vs))

print time.time()-tic
### this one uses the formulation given by the chatfield evaluation but vk doesn't really make sense
# for k in xrange(1):#numComponents) :
#     ## get covariance matrix for component k
#     cov = gmModel.covars_[k, :]*np.eye(featDim)
#     invSqrtCov = np.linalg.inv(np.sqrt(cov))
#     invCov = np.linalg.inv(cov)
    
#     invSqrtCovRep = invSqrtCov.reshape((1, featDim, featDim)).repeat(imgData.shape[0], axis=0)
#     invCovRep = invCov.reshape((1, featDim, featDim)).repeat(imgData.shape[0], axis=0)
#     kCompMeansRep = modelMeans[k, :].reshape((1, featDim)).repeat(imgData.shape[0], axis=0)
#     kCompPostRep = posteriors[:, k].reshape((imgData.shape[0], 1, 1)).repeat(featDim, axis=1).repeat(featDim, axis=-1)
#     dataMinusMeanRep = (imgData-kCompMeansRep).reshape((imgData.shape[0], 1, featDim)).repeat(featDim, axis=1)
    
#     uk = np.sum(kCompPostRep*invSqrtCovRep*dataMinusMeanRep, axis=-1)
#     uk = np.sum(uk, axis=0)/(imgData.shape[0]*np.sqrt(priors[k]))
    
#     vk = 

# <codecell>

########### LOAD LABELLED PAIRS, FIT K-MEANS OR GMM IF USING ENCODING AND COMPUTE LABELLED PAIRS FEATURES ###########

# labelledPairs = np.copy(window.labelledPairs)
# labelledPairs = np.load("labelledPairs.npy")
# labelledPairs = np.load("labelledPairs_better.npy")
# labelledPairs = np.load("labelledPairs_better_x3.npy")
# labelledPairs = np.load("appearanceLabelledPairs.npy")
labelledPairs = np.load("appearanceLabelledPairs_setchoice.npy")
print len(labelledPairs), "labelled pairs"
labelledData = []
histBins = 256

## check if encoding is being used for frame features
if K_MEANS_HIST in frameFeatTypes or FISHER in frameFeatTypes :
    print "encoding frame features", 
    
    ## get the single frame features from set of labelled frames
    labelledFrameFeats = [ frameFeatures[frameIdx] for frameIdx in np.array(np.unique(np.ndarray.flatten(labelledPairs[:, 0:2])), dtype=int)]
    
    ## flatten the single frame features and reshape them to fit k-means clusterer or gmm 
    flatLabelledFeats = np.ndarray.flatten(np.array(labelledFrameFeats))
    if HOG_FEAT in frameFeatTypes :
        flatLabelledFeats = flatLabelledFeats.reshape((len(flatLabelledFeats)/hogOrientations, hogOrientations))
    
    if K_MEANS_HIST in frameFeatTypes :
        print "using K-means histogram"
        clusterer = fitKmeansClusterer(flatLabelledFeats, histBins)
        for i in xrange(len(frameFeatures)) :
            if HOG_FEAT in frameFeatTypes :
                frameFeatures[i] = getKMeansHistEncoding(frameFeatures[i], (len(frameFeatures[i])/hogOrientations, hogOrientations), clusterer)
            else :
                frameFeatures[i] = getKMeansHistEncoding(frameFeatures[i], frameFeatures[i].shape, clusterer)
                
            sys.stdout.write('\r' + "Encoded frame " + np.string_(i) + " of " + np.string_(int(numFrames)))
            sys.stdout.flush()
        print
        
    elif FISHER in frameFeatTypes :
        print "using fisher"
        
elif K_MEANS_HIST in pairFeatTypes or FISHER in pairFeatTypes :
    print "encoding pair features", 
    
    ## get the pair frame features from set of labelled frames
    labelledPairFeats = []
    for labelledPair in labelledPairs :
        labelledPairFeats.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], pairFeatTypes))
    
    ## flatten the single frame features and reshape them to fit k-means clusterer or gmm 
    flatLabelledFeats = np.ndarray.flatten(np.array(labelledPairFeats))
    if HOG_FEAT in frameFeatTypes :
        flatLabelledFeats = flatLabelledFeats.reshape((len(flatLabelledFeats)/hogOrientations, hogOrientations))
    
    if K_MEANS_HIST in pairFeatTypes :
        print "using K-means histogram"
        clusterer = fitKmeansClusterer(flatLabelledFeats, histBins)
            
    elif FISHER in pairFeatTypes :
        print "using fisher"
        gmModel = fitGMM(flatLabelledFeats, 10)
        
for labelledPair, i in zip(labelledPairs, arange(len(labelledPairs))) :
    if K_MEANS_HIST in pairFeatTypes :
        labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], pairFeatTypes, kmeansClusterer=clusterer))
    elif FISHER in pairFeatTypes :
        labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], pairFeatTypes, gmModel=gmModel))
    elif L2_DIST in pairFeatTypes :
        labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), frameFeatures[int(labelledPair[1])].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), pairFeatTypes))
#     frame1Loc = frames[int(labelledPair[0])]
#     frame1Matte = mattes[int(labelledPair[0])]
#     frame2Loc = frames[int(labelledPair[1])]
#     frame2Matte = mattes[int(labelledPair[1])]
#     labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], pairFeatTypes, kmeansClusterer=clusterer))
    ## with grid features
#     labelledData.append(getFramesPairFeatures(features[int(labelledPair[0])], features[int(labelledPair[1])], pairFeatTypes))
    ## no encoding
#     labelledData.append(getFramesPairFeatures(getFrameFeatures(frame1Loc, frameFeatTypes, frame1Matte), getFrameFeatures(frame2Loc, frameFeatTypes, frame2Matte), pairFeatTypes))
#     labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], pairFeatTypes))
    ## no encoding and taking L2_DIST of orientations-size hog vectors
#     labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), frameFeatures[int(labelledPair[1])].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), pairFeatTypes))
    ## encoding single frames' features
#     labelledData.append(getFramesPairFeatures(encodedFrameFeatures[int(labelledPair[0])][0], encodedFrameFeatures[int(labelledPair[1])][0], pairFeatTypes))
    ## encoding pairwise hog features
#     pairwiseFeats = getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], pairFeatTypes)
#     labelledData.append(np.histogram(clusterer.predict(np.reshape(pairwiseFeats, (len(pairwiseFeats)/hogOrientations, hogOrientations))), bins=256)[0])
#     labelledData.append(getFramesPairFeatures(frameFeatures[int(labelledPair[0])], frameFeatures[int(labelledPair[1])], [ABS_DIFF, K_MEANS_HIST], kmeansClusterer=clusterer))
    
    sys.stdout.write('\r' + "Done " + np.string_(i+1) + " pairs: " + np.string_(int(labelledPair[0])) + " - " + np.string_(int(labelledPair[1])))
    sys.stdout.flush()

# <codecell>

getKMeansHistEncoding(frameFeatures[i], (len(frameFeatures[i])/hogOrientations, hogOrientations), clusterer)

# <codecell>

print labelledData[0].shape
print len(labelledData)

# <codecell>

print labelledData[0].shape

# <codecell>

figure(); imshow(np.load(dataFolder + sampleData + "sematics_hog_set150_distMat.npy"), interpolation='nearest')

# <codecell>

print len(list(np.concatenate((labelledData, [pairFeatures[0]]))))
print len(list(np.concatenate((1.0-np.array(labelledPairs)[:, 2], [1.0]))))

# <codecell>

tic = time.time()
classifier = ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=4, verbose=4)
classifier.fit(list(np.concatenate((labelledData, [pairFeatures[1]]))), list(np.concatenate((1.0-np.array(labelledPairs)[:, 2], [1.0]))))
print time.time()-tic

# <codecell>

learnedDistMat = np.zeros((len(frames), len(frames)))
    
pairFeatures = []

for i in xrange(len(frames)) :
    tic = time.time()
#     pairFeatures = []
    for j in xrange(i, len(frames)) :
    ## with grid features
#         pairFeatures.append(getFramesPairFeatures(features[i], features[j], pairFeatTypes))
    ## no encoding
#         pairFeatures.append(getFramesPairFeatures(frameFeatures[i], frameFeatures[j], pairFeatTypes))
    ## no encoding and taking L2_DIST of orientations-size hog vectors
#         pairFeatures.append(getFramesPairFeatures(frameFeatures[i].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), frameFeatures[j].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), pairFeatTypes))
        pairFeatures.append(getFramesPairFeatures(frameFeatures[i].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), frameFeatures[j].reshape((len(frameFeatures[0])/hogOrientations, hogOrientations)), pairFeatTypes, gmModel=gmModel))
    ## encoding single frames' features
#         pairFeatures.append(getFramesPairFeatures(encodedFrameFeatures[i][0], encodedFrameFeatures[j][0], pairFeatTypes))
    ## encoding pairwise hog features
#         pairwiseFeats = getFramesPairFeatures(frameFeatures[i], frameFeatures[j], pairFeatTypes)
#         pairFeatures.append(np.histogram(clusterer.predict(np.reshape(pairwiseFeats, (len(pairwiseFeats)/hogOrientations, hogOrientations))), bins=256)[0])
#         pairFeatures.append(getFramesPairFeatures(frameFeatures[i], frameFeatures[j], pairFeatTypes, kmeansClusterer=clusterer))
#         pairFeatures.append(getFramesPairFeatures(frameFeatures[i], frameFeatures[j], pairFeatTypes, gmModel=gmModel))
    print i, "pairFeatures time", time.time() - tic
#     tic = time.time()
#     learnedDistMat[i, i:] = classifier.predict(pairFeatures)
#     print "predict time", time.time() - tic
#     del pairFeatures
#     gc.collect()
#     print i,

# <codecell>

# np.save("fisherFeatsAll.npy", np.array(pairFeatures))
pairFeatures = list(np.load("fisherFeatsAll.npy"))

# <codecell>

## take labelled pairs from precomputed pairFeatures
labelledPairs = np.load("appearanceLabelledPairs_setchoice.npy")
labelledData = []
for labelledPair in labelledPairs :
    firstIdx = int(np.min(labelledPair[0:2]))
    secondIdx = int(np.max(labelledPair[0:2]))
    labelledData.append(pairFeatures[np.sum(arange(numFrames, numFrames-firstIdx, -1)) + secondIdx - firstIdx])
#     tmp = pairFeatures[np.sum(arange(numFrames, numFrames-firstIdx, -1)) + secondIdx - firstIdx]

# <codecell>

tic = time.time()
dists = classifier.predict(pairFeatures)
print "predict time", time.time() - tic

# <codecell>

print labelledPairs

# <codecell>

learnedDistMat = np.zeros((len(frames), len(frames)))
idx = 0
for i, length in zip(xrange(len(frames)), xrange(len(frames), 0, -1)) :
    print idx, length
#     print i*len(frames), ":", i*len(frames)
    learnedDistMat[i, i:] = dists[idx:idx+length]
    idx += length

# <codecell>

print np.argsort(normedDistMat, axis=1)
print normedDistMat[0, 99], normedDistMat[0, 134], np.max(normedDistMat[0, :])

# <codecell>

## make distMat symmetric and save
distMat = np.copy(learnedDistMat)
distMat += np.triu(distMat, k=1).T
normedDistMat = np.copy(distMat)/np.sum(distMat, axis=0).reshape((len(distMat), 1)).repeat(len(distMat), axis=-1)
# figure(); imshow(distMat, interpolation='nearest')
gwv.showCustomGraph(-np.log(np.maximum(normedDistMat, np.zeros(distMat.shape)+0.0001)))
# np.save(dataFolder + sampleData + "appearance_hog_set150_fisher_distMat.npy", distMat)

# <codecell>

import GraphWithValues as gwv
# distMat = np.load(dataFolder + sampleData + "semantics_hog_L2_set150_distMat.npy")
# gwv.showCustomGraph(np.load(dataFolder + sampleData + "semantics_hog_set50_distMat.npy"))
# gwv.showCustomGraph(np.load(dataFolder + sampleData + "appearance_hog_set150_distMat.npy"))
gwv.showCustomGraph(np.load(dataFolder + sampleData + "appearance_hog_L2_set150_distMat.npy"))
# gwv.showCustomGraph(np.load(dataFolder + sampleData + "appearance_hog_set150_fisher_distMat.npy"))
# gwv.showCustomGraph(np.load(dataFolder + sampleData + "semantics_hog_rand50_encodedlast_distMat.npy"))
# gwv.showCustomGraph(np.load(dataFolder + sampleData + "semantics_hog_rand50_encodedfirst_distMat.npy"))

# <codecell>

tmpFeats = getFramesPairFeatures(frameFeatures[345], frameFeatures[786], pairFeatTypes, kmeansClusterer=clusterer)
print tmpFeats
pairwiseFeats = getFramesPairFeatures(frameFeatures[345], frameFeatures[786], pairFeatTypes)
tmpFeats2 = np.histogram(clusterer.predict(np.reshape(pairwiseFeats, (len(pairwiseFeats)/hogOrientations, hogOrientations))), bins=256)[0]
print tmpFeats2
print tmpFeats2-tmpFeats

# <codecell>

print distMat[463, 400:500]

# <codecell>

figure(); imshow(np.load(dataFolder + sampleData + "hog_betterexamples_x3_distMat.npy"), interpolation='nearest')

# <codecell>

# figure(); imshow(learnedDistMat, interpolation='nearest')
print learnedDistMat[list(np.where(np.eye(len(frames))==1))]

# <codecell>

frame1Idx = 0
frame2Idx = 1
frame1Loc = frames[frame1Idx]
frame1Matte = mattes[frame1Idx]
frame2Loc = frames[frame2Idx]
frame2Matte = mattes[frame2Idx]

# figure(); imshow(cv2.cvtColor(cv2.imread(frame1Loc), cv2.COLOR_BGR2RGB))
# figure(); imshow(cv2.cvtColor(cv2.imread(frame2Loc), cv2.COLOR_BGR2RGB))

tic = time.time()
framesFeats = [getFramesPairFeatures(getFrameFeatures(frame1Loc, frameFeatTypes, frame1Matte), getFrameFeatures(frame2Loc, frameFeatTypes, frame2Matte), pairFeatTypes)]

for i in xrange(2, 30):#len(frames)) :
    frame2Loc = frames[i]
    frame2Matte = mattes[i]
    framesFeats.append(getFramesPairFeatures(getFrameFeatures(frame1Loc, frameFeatTypes, frame1Matte), getFrameFeatures(frame2Loc, frameFeatTypes, frame2Matte), pairFeatTypes))
    print i, 

print time.time() - tic


tic = time.time()
print classifier.predict(framesFeats)
print time.time() - tic

# <codecell>

# from skimage.feature import hog
# from skimage import color
frame1 = np.array(cv2.cvtColor(cv2.imread(frames[1]), cv2.COLOR_BGR2RGB), dtype=np.float32)
matte = np.array(cv2.cvtColor(cv2.imread(mattes[1]), cv2.COLOR_BGR2GRAY))/255.0
frame1 = frame1*np.repeat(np.reshape(matte, (matte.shape[0], matte.shape[1], 1)), 3, axis=-1)

image = color.rgb2gray(frame1)
fd1, hog_image0 = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
figure(); imshow(hog_image0, interpolation='nearest')

# <codecell>

# print fd0
# print fd1
# print np.sum(getAbsDiff(fd0, fd1))
print labelledPairs[3, :]
print np.sum(labelledData[3])

# <codecell>

tic = time.time()
print classifier.predict(framesFeats[0:10])
print time.time() - tic

# <codecell>

############# GETTING FEATURES FROM FRAMES ###############

## compute features for image
blocksPerWidth = 4#16#32
blocksPerHeight = 4#16#48
subDivisions = blocksPerWidth*blocksPerHeight

## given block sizes and img sizes build indices representing each block
imageSize = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)).shape
stencils = cgf.stencil2D(blocksPerWidth, blocksPerHeight, imageSize)

# features = cgf.histFgFeatures(stencils, subDivisions, frames, mattes)
features = np.load("tmpfeat.npy")
figure(); imshow(features.T, interpolation='nearest')

# np.save("tmpfeat.npy", features)

# <codecell>

data_frame = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB), dtype=np.float32)
tmp = data_frame.reshape((np.prod(data_frame.shape[0:-1]), 3))
print data_frame[400, 500, :]
print tmp[400*1280+500, :]
features = np.empty((0, 3))
print np.concatenate((features, tmp))
print np.sqrt(np.sum(np.power(tmp[0:-1, :]-tmp[1:, :],2), axis=-1))

# <codecell>

figure(); imshow(labelledData[0].reshape((720, 1280)))
scatter(700, 200)
frame1 = np.array(cv2.cvtColor(cv2.imread(frames[labelledPairs[0, 0]]), cv2.COLOR_BGR2RGB), dtype=np.float32)
matte = np.array(cv2.cvtColor(cv2.imread(mattes[labelledPairs[0, 0]]), cv2.COLOR_BGR2GRAY))/255.0
frame1 = frame1*np.repeat(np.reshape(matte, (matte.shape[0], matte.shape[1], 1)), 3, axis=-1)
frame2 = np.array(cv2.cvtColor(cv2.imread(frames[labelledPairs[0, 1]]), cv2.COLOR_BGR2RGB), dtype=np.float32)
matte = np.array(cv2.cvtColor(cv2.imread(mattes[labelledPairs[0, 1]]), cv2.COLOR_BGR2GRAY))/255.0
frame2 = frame2*np.repeat(np.reshape(matte, (matte.shape[0], matte.shape[1], 1)), 3, axis=-1)
figure(); imshow(frame1)
figure(); imshow(frame2)
print labelledData[0][200*1280+700]
print frame1[200, 700, :]
print frame2[200, 700, :]
np.sqrt(np.sum(np.power(frame1[200, 700, :]-frame2[200, 700, :],2)))

# <codecell>

clf = ensemble.ExtraTreesClassifier()

#         print len(data_frames), len(trimaps)
data_frame = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
#         print idxs.shape, data_frames[0].shape
data = np.concatenate((data_frame, idxs), axis=-1)
print idxs.shape, data.shape

trimap = np.array(cv2.cvtColor(cv2.imread(dataFolder + sampleData + "trimap-frame-00001.png"), cv2.COLOR_BGR2GRAY), dtype=np.uint8)
print trimap.shape

# # extract training data
background = data[trimap == 0]
foreground = data[trimap == 2]

print background.shape, foreground.shape

background = np.vstack((background, data[trimap == 0]))
foreground = np.vstack((foreground, data[trimap == 2]))

print background.shape, foreground.shape

# for data_frame, trimap in zip(data_frames[1:], trimaps[1:]) :
# #             print background.shape, foreground.shape
    
#     idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
# #             print idxs.shape, data_frame.shape
#     data = np.concatenate((data_frame, idxs), axis=-1)

#     # extract training data
#     background = np.vstack((background, data[trimap == 0]))
#     foreground = np.vstack((foreground, data[trimap == 2]))

# <codecell>

print np.array(X).shape

# <codecell>

X = [[0], [10], [5], [0], [10], [5], [0], [10], [5], [0], [10], [5]]
y = [0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5]
clf = ensemble.RandomForestRegressor()
clf = clf.fit(X, y)
clf.predict([7])

# <codecell>

X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                  random_state=0)
# print X
# print y

clf = ensemble.RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
clf.fit(X, y)

# <codecell>

figure()
scatter(X[np.argwhere(y==0), 0], X[np.argwhere(y==0), 1], c='b', marker='x')
scatter(X[np.argwhere(y==1), 0], X[np.argwhere(y==1), 1], c='r', marker='x')
scatter(X[np.argwhere(y==2), 0], X[np.argwhere(y==2), 1], c='g', marker='x')

# <codecell>

minX = np.floor(np.min(X[:, 0]))
maxX = np.ceil(np.max(X[:, 0]))
minY = np.floor(np.min(X[:, 1]))
maxY = np.ceil(np.max(X[:, 1]))
xs = np.linspace(minX, maxX, (np.abs(minX-maxX)+1)*1)
ys = np.linspace(minY, maxY, (np.abs(minY-maxY)+1)*1)
xv, yv = np.meshgrid(xs, ys)
newData = np.array([np.ndarray.flatten(xv), np.ndarray.flatten(yv)]).T
predictedClasses = clf.predict(newData)
print predictedClasses
figure()
scatter(newData[np.argwhere(predictedClasses==0), 0], newData[np.argwhere(predictedClasses==0), 1], c='b', marker='x')
scatter(newData[np.argwhere(predictedClasses==1), 0], newData[np.argwhere(predictedClasses==1), 1], c='r', marker='x')
scatter(newData[np.argwhere(predictedClasses==2), 0], newData[np.argwhere(predictedClasses==2), 1], c='g', marker='x')

# <codecell>

print ((np.abs(minX-maxX)+1)*10, (np.abs(minY-maxY)+1)*10)
predictedClasses2D = np.zeros(np.array((((np.abs(minY-maxY)+1)*1, np.abs(minX-maxX)+1)*1), dtype=int))
for i, j in zip(predictedClasses2D.shape[0]-1-arange(predictedClasses2D.shape[0]), arange(predictedClasses2D.shape[0])) :
    print i, j*predictedClasses2D.shape[1], (j+1)*predictedClasses2D.shape[1]
    predictedClasses2D[i, :] = predictedClasses[j*predictedClasses2D.shape[1]:(j+1)*predictedClasses2D.shape[1]]
figure(); imshow(predictedClasses2D, interpolation='nearest')

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
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.setWindowTitle("Semantic Distance Metric Learning")
        self.resize(1280, 720)
        
        self.labelledPairs = []
#         self.labelledPairs.append([1234, 234, 0.5])
#         self.setLabelledFramesListTable()

        self.RANDOM_CHOICE = 0
        self.SET_CHOICE = 1
        self.choiceMode = self.RANDOM_CHOICE
        
        self.choiceSet1 = arange(100)
        self.choiceSet2 = arange(100)
        
        self.getNewPair()
        
        self.setFocus()
        
    def setLabelledFramesListTable(self) :
        
        if len(self.labelledPairs) > 0 :
            self.labelledFramesListTable.setRowCount(len(self.labelledPairs))
            
            for pairing, i in zip(self.labelledPairs, arange(len(self.labelledPairs))) :
                self.labelledFramesListTable.setItem(i, 0, QtGui.QTableWidgetItem(np.string_(pairing[0])))
                self.labelledFramesListTable.setItem(i, 1, QtGui.QTableWidgetItem(np.string_(pairing[1])))
                self.labelledFramesListTable.setItem(i, 2, QtGui.QTableWidgetItem(np.string_(pairing[2])))
        else :
            self.labelledFramesListTable.setRowCount(0)
    
    def getNewPair(self) :
        if self.choiceMode == self.RANDOM_CHOICE :
            self.getNewRandomPair()
        elif self.choiceMode == self.SET_CHOICE :
            self.getNewPairFromSet()
    
    def getNewPairFromSet(self) :
        self.frame1Idx = np.random.choice(self.choiceSet1)
        self.frame2Idx = np.random.choice(self.choiceSet2) 
        while self.frame2Idx == self.frame1Idx :
            print "stuck"
            self.frame2Idx = np.random.choice(self.choiceSet2)
        
        self.setFrameImages()
            
    def getNewRandomPair(self) :
        self.frame1Idx = np.random.randint(0, len(frames))
        self.frame2Idx = np.random.randint(0, len(frames)) 
        while self.frame2Idx == self.frame1Idx :
            print "stuck 2"
            self.frame2Idx = np.random.randint(0, len(frames))
        
        self.setFrameImages()
        
    def setFrameImages(self) :
        ## HACK ##
        im = np.ascontiguousarray(Image.open(frames[self.frame1Idx]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frame1Label.setPixmap(QtGui.QPixmap.fromImage(qim))
        self.frame1Info.setText(frames[self.frame1Idx])
        
        im = np.ascontiguousarray(Image.open(frames[self.frame2Idx]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frame2Label.setPixmap(QtGui.QPixmap.fromImage(qim))
        self.frame2Info.setText(frames[self.frame2Idx])
        
    def existingPairSelected(self) :
        selectedRow = self.labelledFramesListTable.currentRow()
        if selectedRow >= 0 and selectedRow < len(self.labelledPairs):
            ## HACK ##
            im = np.ascontiguousarray(Image.open(frames[self.labelledPairs[selectedRow][0]]))
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            self.frame1Label.setPixmap(QtGui.QPixmap.fromImage(qim))
            self.frame1Info.setText(frames[self.labelledPairs[selectedRow][0]])
            
            im = np.ascontiguousarray(Image.open(frames[self.labelledPairs[selectedRow][1]]))
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            self.frame2Label.setPixmap(QtGui.QPixmap.fromImage(qim))
            self.frame2Info.setText(frames[self.labelledPairs[selectedRow][1]])
        
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_Up :
            return
#             self.labelledFramesListTable.clearSelection()
        elif e.key() == QtCore.Qt.Key_Down : ## Distance is in the middle
            self.setFramePairDistance(0.5)
        elif e.key() == QtCore.Qt.Key_Left : ## Distance is low
            self.setFramePairDistance(0.0)
        elif e.key() == QtCore.Qt.Key_Right : ## Distance is high
            self.setFramePairDistance(1.0)
        elif e.key() == QtCore.Qt.Key_Space : ## Get new pair
            self.getNewPair()
            
    def setFramePairDistance(self, distance) :
        if self.labelledFramesListTable.currentRow() >= 0 : ## I'm modifying distance of existing pair
            self.labelledPairs[self.labelledFramesListTable.currentRow()][2] = distance
        else :
            self.labelledPairs.append([self.frame1Idx, self.frame2Idx, distance])
            
        self.labelledFramesListTable.clearSelection()
        self.labelledFramesListTable.setCurrentCell(-1, -1)
        self.setLabelledFramesListTable()
        
        self.getNewPair()
        
    def changeChoiceMode(self, index) :
        if index == self.RANDOM_CHOICE :
            self.choiceMode = self.RANDOM_CHOICE
            self.choiceSetIntervals.setEnabled(False)
            self.choiceSetIntervals.setVisible(False)
        elif index == self.SET_CHOICE :
            self.choiceMode = self.SET_CHOICE
            self.choiceSetIntervals.setEnabled(True)
            self.choiceSetIntervals.setVisible(True)
        
        self.getNewPair()
        
    def changeChoiceSets(self) :
        choiceSetsText = self.choiceSetIntervals.text()
        intervals = np.array(re.split("-|:", choiceSetsText), dtype=int)
        if len(intervals) == 2 :
            self.choiceSet1 = arange(intervals[0], intervals[1]+1)
            self.choiceSet2 = arange(intervals[0], intervals[1]+1)
        elif len(intervals) == 4 :
            self.choiceSet1 = arange(intervals[0], intervals[1]+1)
            self.choiceSet2 = arange(intervals[2], intervals[3]+1)
            
        self.setFocus()
        self.getNewPair()
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frame1Label = ImageLabel("Frame 1")
        self.frame1Label.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frame1Label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frame2Label = ImageLabel("Frame 2")
        self.frame2Label.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frame2Label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frame1Info = QtGui.QLabel("Info text")
        self.frame1Info.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        self.frame2Info = QtGui.QLabel("Info text")
        self.frame2Info.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.labelledFramesListTable = QtGui.QTableWidget(0, 3)
        self.labelledFramesListTable.horizontalHeader().setStretchLastSection(True)
        self.labelledFramesListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Index f1"))
        self.labelledFramesListTable.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Index f2"))
        self.labelledFramesListTable.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem("Distance"))
        self.labelledFramesListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
#         self.labelledFramesListTable.setItem(0, 0, QtGui.QTableWidgetItem("No Labelled Frames"))
        self.labelledFramesListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.labelledFramesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.labelledFramesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.labelledFramesListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.labelledFramesListTable.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.choiceModeComboBox = QtGui.QComboBox()
        self.choiceModeComboBox.addItem("Random")
        self.choiceModeComboBox.addItem("Set")
        self.choiceModeComboBox.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.choiceSetIntervals = QtGui.QLineEdit()
        self.choiceSetIntervals.setEnabled(False)
        self.choiceSetIntervals.setVisible(False)
        
        
        ## SIGNALS ##
        
        self.labelledFramesListTable.cellPressed.connect(self.existingPairSelected)
        self.choiceModeComboBox.currentIndexChanged[int].connect(self.changeChoiceMode)
        self.choiceSetIntervals.returnPressed.connect(self.changeChoiceSets)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.labelledFramesListTable)
        controlsLayout.addWidget(self.choiceModeComboBox)
        controlsLayout.addWidget(self.choiceSetIntervals)
        frame1Layout = QtGui.QVBoxLayout()
        frame1Layout.addWidget(self.frame1Label)
        frame1Layout.addWidget(self.frame1Info)
        frame2Layout = QtGui.QVBoxLayout()
        frame2Layout.addWidget(self.frame2Label)
        frame2Layout.addWidget(self.frame2Info)
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frame1Layout)
        mainLayout.addLayout(frame2Layout)
        self.setLayout(mainLayout)

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
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.setWindowTitle("Appearance Distance Metric Learning")
        self.resize(1280, 720)
        
        self.textureTimer = QtCore.QTimer(self)
        self.textureTimer.setInterval(1000/30)
        self.textureTimer.start()
        self.textureTimer.timeout.connect(self.renderOneFrame)
        self.visualizeMovie = False
        self.currentVisFrame = 0
        self.showMatted = True
        
        self.labelJumpStyle = "QLabel {border: 1px solid black; background: #aa0000; color: white; padding-left: 5px; padding-right: 5px;}"
        self.labelNoJumpStyle = "QLabel {border: 1px solid gray; background: #eeeeee; color: black; padding-left: 5px; padding-right: 5px;}"
        
        self.additionalFrames = 5
        
        self.labelledPairs = []#np.copy(labelledPairs)#[]#list(np.load("appearanceLabelledPairs.npy"))
        self.setLabelledFramesListTable()
#         self.labelledPairs.append([1234, 234, 0.5])
#         self.setLabelledFramesListTable()

        self.RANDOM_CHOICE = 0
        self.SET_CHOICE = 1
        self.choiceMode = self.RANDOM_CHOICE
        
        self.choiceSet1 = arange(100)
        self.choiceSet2 = arange(100)
        
        self.getNewPair()
        
        self.setFocus()
        
    def setRenderFps(self, value) :
        self.textureTimer.setInterval(1000/value)
        
    def finishedSettingFps(self) :
        self.setFocus()
        
    def setTextureFrame(self, im, alpha):
        im = np.ascontiguousarray(im)
        
        if alpha :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
        else :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            
        self.movieLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
    def renderOneFrame(self):
        if self.visualizeMovie:
            try:
                self.movie
                self.currentVisFrame
            except AttributeError:
                return
            else:
                if self.currentVisFrame < 0 or self.currentVisFrame >= len(self.movie) :
                    self.currentVisFrame = 0
                    
                frameIdx = self.movie[self.currentVisFrame]
                
                if self.showMatted and os.path.isfile(mattes[frameIdx]) :
                    alphaMatte = cv2.cvtColor(cv2.imread(mattes[frameIdx]), cv2.COLOR_BGR2GRAY)
                    alphaMatte = np.reshape(alphaMatte, np.hstack((alphaMatte.shape[0:2], 1)))
                    self.setTextureFrame(np.concatenate((cv2.imread(frames[frameIdx]), alphaMatte), axis=-1), True)
                else :
                    self.textureViewerGroup.setTextureFrame(cv2.imread(frames[frameIdx], False))
                    
                if self.currentVisFrame == 6 :
                    self.movieInfo.setStyleSheet(self.labelJumpStyle)
                else :
                    self.movieInfo.setStyleSheet(self.labelNoJumpStyle)
                    
                self.movieInfo.setText(np.str_(self.currentVisFrame) + " from " + np.str_(frameIdx))
                
                self.currentVisFrame = np.mod(self.currentVisFrame+1, len(self.movie))
        
    def setLabelledFramesListTable(self) :
        
        if len(self.labelledPairs) > 0 :
            self.labelledFramesListTable.setRowCount(len(self.labelledPairs))
            
            for pairing, i in zip(self.labelledPairs, arange(len(self.labelledPairs))) :
                self.labelledFramesListTable.setItem(i, 0, QtGui.QTableWidgetItem(np.string_(pairing[0])))
                self.labelledFramesListTable.setItem(i, 1, QtGui.QTableWidgetItem(np.string_(pairing[1])))
                self.labelledFramesListTable.setItem(i, 2, QtGui.QTableWidgetItem(np.string_(pairing[2])))
        else :
            self.labelledFramesListTable.setRowCount(0)
    
    def getNewPair(self) :
        if self.choiceMode == self.RANDOM_CHOICE :
            self.getNewRandomPair()
        elif self.choiceMode == self.SET_CHOICE :
            self.getNewPairFromSet()
    
    def getNewPairFromSet(self) :
        self.frame1Idx = np.random.choice(self.choiceSet1)
        self.frame2Idx = np.random.choice(self.choiceSet2) 
        while self.frame2Idx == self.frame1Idx :
            print "stuck"
            self.frame2Idx = np.random.choice(self.choiceSet2)
        
        self.setMovie()
            
    def getNewRandomPair(self) :
        self.frame1Idx = np.random.randint(0, len(frames)-self.additionalFrames-3)
        self.frame2Idx = np.random.randint(0, len(frames)-self.additionalFrames-3) 
        while self.frame2Idx == self.frame1Idx :
            print "stuck 2"
            self.frame2Idx = np.random.randint(0, len(frames)-self.additionalFrames-3)
        
        self.setMovie()
        
    def setMovie(self) :
        self.visualizeMovie = False
        
        self.movie = np.arange(self.frame1Idx-self.additionalFrames, self.frame1Idx+1, dtype=int)
        self.movie = np.concatenate((self.movie, np.arange(self.frame2Idx+1, self.frame2Idx+self.additionalFrames+2, dtype=int)))
        
        print self.movie, self.frame1Idx, self.frame2Idx
        
        self.visualizeMovie = True
#         ## HACK ##
#         im = np.ascontiguousarray(Image.open(frames[self.frame1Idx]))
#         qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
#         self.frame1Label.setPixmap(QtGui.QPixmap.fromImage(qim))
#         self.frame1Info.setText(frames[self.frame1Idx])
        
#         im = np.ascontiguousarray(Image.open(frames[self.frame2Idx]))
#         qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
#         self.frame2Label.setPixmap(QtGui.QPixmap.fromImage(qim))
#         self.frame2Info.setText(frames[self.frame2Idx])
        
    def existingPairSelected(self) :
        selectedRow = self.labelledFramesListTable.currentRow()
        if selectedRow >= 0 and selectedRow < len(self.labelledPairs):
            self.frame1Idx = self.labelledPairs[selectedRow][0]
            self.frame2Idx = self.labelledPairs[selectedRow][1]
            
            self.setMovie()
#             ## HACK ##
#             im = np.ascontiguousarray(Image.open(frames[self.labelledPairs[selectedRow][0]]))
#             qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
#             self.frame1Label.setPixmap(QtGui.QPixmap.fromImage(qim))
#             self.frame1Info.setText(frames[self.labelledPairs[selectedRow][0]])
            
#             im = np.ascontiguousarray(Image.open(frames[self.labelledPairs[selectedRow][1]]))
#             qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
#             self.frame2Label.setPixmap(QtGui.QPixmap.fromImage(qim))
#             self.frame2Info.setText(frames[self.labelledPairs[selectedRow][1]])
        
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_Left : ## Distance is low
            self.setFramePairDistance(0.0)
        elif e.key() == QtCore.Qt.Key_Right : ## Distance is high
            self.setFramePairDistance(1.0)
        elif e.key() == QtCore.Qt.Key_Space : ## Get new pair
            self.getNewPair()
            
    def setFramePairDistance(self, distance) :
        if self.labelledFramesListTable.currentRow() >= 0 : ## I'm modifying distance of existing pair
            self.labelledPairs[self.labelledFramesListTable.currentRow()][2] = distance
        else :
            self.labelledPairs.append([self.frame1Idx, self.frame2Idx, distance])
            
        self.labelledFramesListTable.clearSelection()
        self.labelledFramesListTable.setCurrentCell(-1, -1)
        self.setLabelledFramesListTable()
        
        self.getNewPair()
        
    def changeChoiceMode(self, index) :
        if index == self.RANDOM_CHOICE :
            self.choiceMode = self.RANDOM_CHOICE
            self.choiceSetIntervals.setEnabled(False)
            self.choiceSetIntervals.setVisible(False)
        elif index == self.SET_CHOICE :
            self.choiceMode = self.SET_CHOICE
            self.choiceSetIntervals.setEnabled(True)
            self.choiceSetIntervals.setVisible(True)
        
        self.getNewPair()
        
    def changeChoiceSets(self) :
        choiceSetsText = self.choiceSetIntervals.text()
        intervals = np.array(re.split("-|:", choiceSetsText), dtype=int)
        if len(intervals) == 2 :
            self.choiceSet1 = arange(intervals[0], intervals[1]+1)
            self.choiceSet2 = arange(intervals[0], intervals[1]+1)
        elif len(intervals) == 4 :
            self.choiceSet1 = arange(intervals[0], intervals[1]+1)
            self.choiceSet2 = arange(intervals[2], intervals[3]+1)
            
        self.setFocus()
        self.getNewPair()
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.movieLabel = ImageLabel("Movie")
        self.movieLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.movieInfo = QtGui.QLabel("Info text")
        self.movieInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.labelledFramesListTable = QtGui.QTableWidget(0, 3)
        self.labelledFramesListTable.horizontalHeader().setStretchLastSection(True)
        self.labelledFramesListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Index f1"))
        self.labelledFramesListTable.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Index f2"))
        self.labelledFramesListTable.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem("Distance"))
        self.labelledFramesListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
#         self.labelledFramesListTable.setItem(0, 0, QtGui.QTableWidgetItem("No Labelled Frames"))
        self.labelledFramesListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.labelledFramesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.labelledFramesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.labelledFramesListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.labelledFramesListTable.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.choiceModeComboBox = QtGui.QComboBox()
        self.choiceModeComboBox.addItem("Random")
        self.choiceModeComboBox.addItem("Set")
        self.choiceModeComboBox.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.choiceSetIntervals = QtGui.QLineEdit()
        self.choiceSetIntervals.setEnabled(False)
        self.choiceSetIntervals.setVisible(False)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 60)
        self.renderFpsSpinBox.setSingleStep(1)
        self.renderFpsSpinBox.setValue(30)
        
        
        ## SIGNALS ##
        
        self.labelledFramesListTable.cellPressed.connect(self.existingPairSelected)
        self.choiceModeComboBox.currentIndexChanged[int].connect(self.changeChoiceMode)
        self.choiceSetIntervals.returnPressed.connect(self.changeChoiceSets)
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        self.renderFpsSpinBox.editingFinished.connect(self.finishedSettingFps)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.labelledFramesListTable)
        controlsLayout.addWidget(self.choiceModeComboBox)
        controlsLayout.addWidget(self.choiceSetIntervals)
        controlsLayout.addWidget(self.renderFpsSpinBox)
        movieLayout = QtGui.QVBoxLayout()
        movieLayout.addWidget(self.movieLabel)
        movieLayout.addWidget(self.movieInfo)
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(movieLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

print np.array(window.labelledPairs)
labelledPairs = np.copy(np.array(window.labelledPairs))
# np.save("tmp.npy", labelledPairs)

# <codecell>

labelledPairs = np.copy(np.array(window.labelledPairs))
# np.save("appearanceLabelledPairs_setchoice.npy", labelledPairs)
print labelledPairs

# <codecell>

np.save("appeareanceLabelledPairs_setchoice_oneeach.npy", labelledPairs)

# <codecell>

# print labelledPairs
print classifier.predict(getFramesPairFeatures(frameFeatures[901].reshape((len(frameFeatures[0])/8, 8)), frameFeatures[1126].reshape((len(frameFeatures[0])/8, 8)), pairFeatTypes))

# <codecell>

print np.load(dataFolder + sampleData + "appearance_hog_L2_set150_distMat.npy")[901, 1126]

# <codecell>

labelledPairs = np.load("appearanceLabelledPairs_setchoice.npy")

# <codecell>

0:279-280:756
0:279-757:936
0:279-937:1270
280:756-757:936
280:756-937:1270
757:936-937:1270
0:279
280:756
757:936
937:1270

# <codecell>

np.set_printoptions(suppress=True)
# labelledPairs = np.load("labelledPairs.npy")
# labelledPairs = np.load("labelledPairs_better.npy")
# labelledPairs = np.array(window.labelledPairs)
label1Pairs = any((all((labelledPairs[:, 0] <= 279,labelledPairs[:, 0] >= 0), axis=0), all((labelledPairs[:, 1] <= 279,labelledPairs[:, 1] >= 0), axis=0)), axis=0)
label2Pairs = any((all((labelledPairs[:, 0] <= 756,labelledPairs[:, 0] >= 280), axis=0), all((labelledPairs[:, 1] <= 756,labelledPairs[:, 1] >= 280), axis=0)), axis=0)
label3Pairs = any((all((labelledPairs[:, 0] <= 936,labelledPairs[:, 0] >= 757), axis=0), all((labelledPairs[:, 1] <= 936,labelledPairs[:, 1] >= 757), axis=0)), axis=0)
label4Pairs = any((all((labelledPairs[:, 0] <= 1279,labelledPairs[:, 0] >= 937), axis=0), all((labelledPairs[:, 1] <= 1279,labelledPairs[:, 1] >= 937), axis=0)), axis=0)
print "label 1 and 2"
print labelledPairs[all((label1Pairs, label2Pairs), axis=0)]
print "label 1 and 3"
print labelledPairs[all((label1Pairs, label3Pairs), axis=0)]
print "label 1 and 4"
print labelledPairs[all((label1Pairs, label4Pairs), axis=0)]
print "label 2 and 3"
print labelledPairs[all((label2Pairs, label3Pairs), axis=0)]
print "label 2 and 4"
print labelledPairs[all((label2Pairs, label4Pairs), axis=0)]
print "label 3 and 4"
print labelledPairs[all((label3Pairs, label4Pairs), axis=0)]
print "label 1"
print labelledPairs[all((all((labelledPairs[:, 0] <= 279,labelledPairs[:, 0] >= 0), axis=0), all((labelledPairs[:, 1] <= 279,labelledPairs[:, 1] >= 0), axis=0)), axis=0)]
print "label 2"
print labelledPairs[all((all((labelledPairs[:, 0] <= 756,labelledPairs[:, 0] >= 280), axis=0), all((labelledPairs[:, 1] <= 756,labelledPairs[:, 1] >= 280), axis=0)), axis=0)]
print "label 3"
print labelledPairs[all((all((labelledPairs[:, 0] <= 936,labelledPairs[:, 0] >= 757), axis=0), all((labelledPairs[:, 1] <= 936,labelledPairs[:, 1] >= 757), axis=0)), axis=0)]
print "label 4"
print labelledPairs[all((all((labelledPairs[:, 0] <= 1279,labelledPairs[:, 0] >= 937), axis=0), all((labelledPairs[:, 1] <= 1279,labelledPairs[:, 1] >= 937), axis=0)), axis=0)]
# print labelledPairs

