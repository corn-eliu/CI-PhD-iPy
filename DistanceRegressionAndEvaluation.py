# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
from IPython.display import clear_output
import numpy as np
import sys
import scipy as sp
from scipy import optimize

import time
import os

from sklearn import ensemble
import cv2

import scipy.io as sio
import glob
import commands
import shutil, errno

from PIL import Image

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

cudaFolder = "/home/ilisescu/PhD/cuda/"

import GraphWithValues as gwv
import VideoTexturesUtils as vtu

import ComputeGridFeatures as cgf

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
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

# <codecell>

# dataPath = "/home/ilisescu/PhD/data/"
dataPath = "/media/ilisescu/Data1/PhD/data/"

# dataSet = "pendulum/"
# dataSet = "tree/"
# dataSet = "splashes_water/"
# dataSet = "small_waterfall/"
# dataSet = "sisi_flag/"
# dataSet = "eu_flag_ph_left/"
# dataSet = "ribbon2/"
# dataSet = "candle1/segmentedAndCropped/"
# dataSet = "wave3/"
dataSet = "wave2/"
# dataSet = "wave1/"
framePaths = np.sort(glob.glob(dataPath + dataSet + "frame*.png"))
numFrames = len(framePaths)
print numFrames
imageSize = np.array(Image.open(framePaths[0])).shape[0:2]

# <codecell>

def loadSequenceData(framePaths, resizeRatio, doRGB, temporalRange = None, doNormalize = True) :
    
    baseDimensionality = int(np.prod(np.round(np.array(imageSize)*resizeRatio)))
    if temporalRange != None :
        rangeResizeRatios = resizeRatio/2**np.abs(arange(-np.floor(len(temporalRange)/2), np.floor(len(temporalRange)/2)+1))
    
    numChannels = np.array([1, 3])
    
    if temporalRange != None :
        sequenceData = np.zeros((np.sum(baseDimensionality/((resizeRatio/rangeResizeRatios)**2))*numChannels[int(doRGB)], numFrames), dtype=np.float32)
    else :
        sequenceData = np.zeros((baseDimensionality*numChannels[int(doRGB)], numFrames), dtype=np.float32)
        
    for i in xrange(len(framePaths)) :
        if temporalRange != None :
            feats = np.empty(0)
            for delta, ratio in zip(temporalRange, rangeResizeRatios) :
                if delta+i >= 0 and delta+i < numFrames :
                    feats = np.concatenate((feats, 
                                            np.ndarray.flatten(np.array(cv2.cvtColor(cv2.resize(cv2.imread(framePaths[delta+i]), 
                                                                                                (0, 0), fx=ratio, fy=ratio, 
                                                                                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)))))
                else :
                    feats = np.concatenate((feats, np.zeros(baseDimensionality/((resizeRatio/ratio)**2))))
            sequenceData[:, i] = feats
        else :
            sequenceData[:, i] = np.ndarray.flatten(np.array(cv2.cvtColor(cv2.resize(cv2.imread(framePaths[i]), 
                                                                                     (0, 0), fx=resizeRatio, fy=resizeRatio,
                                                                                     interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)))

        sys.stdout.write('\r' + "Loaded image " + np.string_(i) + " of " + np.string_(numFrames))
        sys.stdout.flush()
    
    if doNormalize :
        sequenceData /= np.float32(255)
        
    clear_output()
    print "loaded", sequenceData.shape[-1], "frames with", numChannels[int(doRGB)], "channels ( D =", sequenceData.shape[0], ")"
    
    return sequenceData

resizeRatio = 1.0#0.4#0.5#0.75
doRGB = False
temporalRange = np.array([-2, -1, 0, 1, 2])
# temporalRange = np.array([-1, 0, 1])

# sequenceData = loadSequenceData(framePaths, resizeRatio, doRGB)

# <codecell>

useValidatedJumps = True

def getTrainingExamples(dataLoc, numFrames, useValidatedJumps, useRandomBads, numBadExamples = 1000, minIdxsDiff = 10, verbose = True) :
    ## get feats of subsequent frames
    goodPairsIdxs = np.array([np.arange(numFrames-1, dtype=int), np.arange(1, numFrames, dtype=int)])
    if verbose : print "added", goodPairsIdxs.shape[-1], "subsequent pairs as GOOD PAIRS"
    badPairsIdxs = np.empty((2, 0), dtype=int)


    if useValidatedJumps and os.path.isfile(dataLoc+"validatedJumps.npy") :
        ## validatedJumps has indices of good jumps which means that it contains indices of distances between i and j+1
        ## so need to take (j+1)-1 to get indices of pairs whos distance has been labelled
        validatedJumps = np.load(dataLoc+"validatedJumps.npy")

        additionalGoodPairsIdxs = np.argwhere(validatedJumps == 1).T
        if additionalGoodPairsIdxs.shape[-1] > 0 :
            goodPairsIdxs = np.concatenate((goodPairsIdxs, additionalGoodPairsIdxs), axis=1)
            if verbose : print "added", additionalGoodPairsIdxs.shape[-1], "validated pairs as GOOD PAIRS"
        
        additionalBadPairsIdxs = np.argwhere(validatedJumps == 0).T
        if additionalBadPairsIdxs.shape[-1] > 0 :
            badPairsIdxs = np.concatenate((badPairsIdxs, additionalBadPairsIdxs), axis=1)
            if verbose : print "added", additionalBadPairsIdxs.shape[-1], "validated pairs as BAD PAIRS"
            
    if useRandomBads :
        ## get feats of random pairings that are considered bad
        additionalBadPairsIdxs = np.sort(np.array([np.random.choice(np.arange(numFrames), numBadExamples),
                                                   np.random.choice(np.arange(numFrames), numBadExamples)]), axis=0)

        if verbose : print len(np.argwhere(np.abs(additionalBadPairsIdxs[0, :]-additionalBadPairsIdxs[1, :]) < minIdxsDiff)), "invalid random pairs"
        for pairIdx in xrange(numBadExamples) :
            idxDiff = np.abs(additionalBadPairsIdxs[0, pairIdx] - additionalBadPairsIdxs[1, pairIdx])
            tmp = idxDiff
            newPair = additionalBadPairsIdxs[:, pairIdx]
            while idxDiff < minIdxsDiff :
                newPair = np.sort(np.random.choice(np.arange(numFrames), 2))
                idxDiff = np.abs(newPair[0] - newPair[1])
        #     print badPairsIdxs[:, pairIdx], newPair, tmp
            additionalBadPairsIdxs[:, pairIdx] = newPair
        #     if badPairsIdxs[pairIdx, 0] - badPairsIdxs[pairIdx, 1] < minIdxsDiff

        # print badPairsIdxs.T
        if verbose : print len(np.argwhere(np.abs(additionalBadPairsIdxs[0, :]-additionalBadPairsIdxs[1, :]) < minIdxsDiff)), "invalid random pairs"
        if additionalBadPairsIdxs.shape[-1] > 0 :
            badPairsIdxs = np.concatenate((badPairsIdxs, additionalBadPairsIdxs), axis=1)
            if verbose : print "added", additionalBadPairsIdxs.shape[-1], "random pairs as BAD PAIRS"
    
    if verbose : print
    print goodPairsIdxs.shape[-1], "GOOD PAIRS and", badPairsIdxs.shape[-1], "BAD PAIRS total"
    
    
    return goodPairsIdxs, badPairsIdxs
        
goodPairsIdxs, badPairsIdxs = getTrainingExamples(dataPath+dataSet, numFrames, useValidatedJumps, True, verbose=False)

# <codecell>

def getPairFeats(frame1, frame2, imageSize, framePaths, frame1PatchesLocs, frame2PatchesLocs, displacePatchSize, gridStencils,
                 usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid, k = 1.0, L = 10.0) :
    
    feats = np.empty(0)
    
    if usePatchDisplace :
        frame1Tmp = frame1/255.0
        frame2Tmp = frame2/255.0
        feats = np.concatenate((feats, np.array([np.sqrt(np.sum((frame2Tmp[l2[0]:l2[0]+displacePatchSize[0], l2[1]:l2[1]+displacePatchSize[1], :]-
                                                                 frame1Tmp[l1[0]:l1[0]+displacePatchSize[0], l1[1]:l1[1]+displacePatchSize[1], :])**2)) 
                                                 for l2, l1 in zip(frame2PatchesLocs, frame1PatchesLocs)])))
    if frame1.shape[-1] == 4 :
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGBA2GRAY)
    else :
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        
    if frame2.shape[-1] == 4 :
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGBA2GRAY)
    else :
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    if useFlow :
        flow = np.array(cv2.calcOpticalFlowFarneback(frame1, frame2, 0.5, 3, 15, 3, 5, 1.1, 0), np.float64)
        ## sigmoid
        if useSigmoid :
            flow = L/(1.0+np.exp(-k*flow))

    for j, stencil in enumerate(gridStencils) :
        ## add SSD to feature vector
        if useSSD :
            feats = np.concatenate((feats, [np.average((frame1[stencil]/255.0-frame2[stencil]/255.0)**2)]))
        ## add average flow intensity
        if useFlow :
            feats = np.concatenate((feats, [np.average(np.linalg.norm(np.array([flow[:, :, 0][stencil], flow[:, :, 1][stencil]]), axis=0))]))

        ## add augmentation to x
        if useAugment :
            feats = np.concatenate((feats, [np.average(((255.0-frame1[stencil])/255.0-frame2[stencil]/255.0)**2+
                                                       ((255.0-frame2[stencil])/255.0-frame1[stencil]/255.0)**2)]))
            
    return feats

# <codecell>

def getPairFeatsData(displacePatchSize, displaceSpacing, dirDisplacements, imageSize, pairsIdxs, gridSize, framePaths,
                     usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid, subPatchTopLeft=np.array([0, 0])) :

    displacements = np.concatenate((np.array([np.zeros(len(dirDisplacements)), dirDisplacements], int), ## EAST
                          np.array([dirDisplacements, dirDisplacements], int), ## SOUT-EAST
                          np.array([dirDisplacements, np.zeros(len(dirDisplacements))], int), ## SOUTH
                          np.array([dirDisplacements, -dirDisplacements], int), ## SOUTH-WEST
                          np.array([np.zeros(len(dirDisplacements)), -dirDisplacements], int), ## WEST
                          np.array([-dirDisplacements, -dirDisplacements], int), ## NORTH-WEST
                          np.array([-dirDisplacements, np.zeros(len(dirDisplacements))], int), ## NORTH
                          np.array([-dirDisplacements, dirDisplacements], int), ## NORTH-EAST
                          ), axis=-1)

    patchYs = np.arange(np.max(displacements), imageSize[0]-displacePatchSize[0]-np.max(displacements), displaceSpacing)
    patchYs = patchYs.reshape((1, len(patchYs)))
    patchXs = np.arange(np.max(displacements), imageSize[1]-displacePatchSize[1]-np.max(displacements), displaceSpacing)
    patchXs = patchXs.reshape((1, len(patchXs)))
    patchLocations = np.array(np.concatenate((patchYs.repeat(len(patchXs.T)),
                                              patchXs.repeat(len(patchYs.T), axis=0).flatten())).reshape((2, len(patchXs.T)*len(patchYs.T))), int)

    frame1Idxs = np.empty((0, 2), dtype=int)
    frame2Idxs = np.empty((0, 2), dtype=int)
    for i, locSlice in enumerate(np.arange(patchXs.shape[-1], patchLocations.shape[-1]+1, patchXs.shape[-1])) :
        frame1Idxs = np.concatenate((frame1Idxs, np.repeat(patchLocations[:, locSlice-patchXs.shape[-1]:locSlice], len(displacements.T), axis=1).T), axis=0)
        frame2Idxs = np.concatenate((frame2Idxs, np.array([disp + loc for loc in patchLocations[:, locSlice-patchXs.shape[-1]:locSlice].T for disp in displacements.T])), axis=0)

    stencils2D = cgf.stencil2D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])
    
#     displacePatchSize = displacePatchSize/2
    
    featsData = []
    avgTime = 0.0
    for j, pair in enumerate(pairsIdxs.T) :
        t = time.time()
        
        frame1 = np.array(Image.open(framePaths[pair[0]]))[subPatchTopLeft[0]:subPatchTopLeft[0]+imageSize[0], subPatchTopLeft[1]:subPatchTopLeft[1]+imageSize[1], :]
        frame2 = np.array(Image.open(framePaths[pair[1]]))[subPatchTopLeft[0]:subPatchTopLeft[0]+imageSize[0], subPatchTopLeft[1]:subPatchTopLeft[1]+imageSize[1], :]
        
        featsData.append(getPairFeats(frame1, frame2, imageSize, framePaths, frame1Idxs, frame2Idxs, displacePatchSize, stencils2D,
                                      usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid))

        avgTime = (avgTime*j + time.time()-t)/(j+1)
        remainingTime = avgTime*(len(pairsIdxs.T)-j-1)/60.0
        sys.stdout.write('\r' + "Done pair " + np.string_(j) + " of " + np.string_(len(pairsIdxs.T)) +
                         " (avg time: " + np.string_(avgTime) + " secs --- remaining: " +
                         np.string_(int(np.floor(remainingTime))) + ":" + np.string_(int((remainingTime - np.floor(remainingTime))*60)) + ")")
        sys.stdout.flush()
        
    return featsData#, frame1Idxs, frame2Idxs

patchSize = np.array([32, 32], int)
spacing = 32.0 ## how far away a patch is from the previous one

dirDisplacements = np.array([1, 2, 4, 8, 16])
gridSize = np.array((10, 10))

usePatchDisplace = True
useSSD = False
useFlow = False
useAugment = False
useSigmoid = False

# goodExamplesData = getPairFeatsData(patchSize, spacing, dirDisplacements, imageSize, goodPairsIdxs, gridSize, framePaths,
#                                     usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid)

# badExamplesData = getPairFeatsData(patchSize, spacing, dirDisplacements, imageSize, badPairsIdxs, gridSize, framePaths,
#                                    usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid)

# <codecell>

badExamplesData, frame1Idxs, frame2Idxs = getPairFeatsData(patchSize, spacing, dirDisplacements, imageSize, badPairsIdxs[:, 0:1], 
                                                            np.array((10, 10)), framePaths, usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid)

# <codecell>

## load a given semantic sequence
tmpFolder = "/home/ilisescu/"        
sequence = np.load(dataPath+dataSet+"semantic_sequence-tara2.npy").item()
tmpLoc = tmpFolder+"".join(sequence[DICT_MASK_LOCATION].split("/")[-2])+"/"
print "starting", sequence[DICT_SEQUENCE_NAME]; sys.stdout.flush()

t = time.time()
if not os.path.isdir(tmpLoc) :
    copyanything(sequence[DICT_MASK_LOCATION], tmpLoc)
    print "copied folder to", tmpLoc; sys.stdout.flush()
else :
    print tmpLoc, "already exists"; sys.stdout.flush()

## compute bounding box
    
topLeft = np.array([720, 1280])
bottomRight = np.array([0, 0])
for i, key in enumerate(np.sort(sequence[DICT_BBOXES].keys())) :
    alpha = np.array(Image.open(tmpLoc+"frame-{0:05d}.png".format(key+1)))[:, :, -1]
    vis = np.argwhere(alpha != 0)
    tl = np.min(vis, axis=0)
    topLeft[0] = np.min([topLeft[0], tl[0]])
    topLeft[1] = np.min([topLeft[1], tl[1]])

    br = np.max(vis, axis=0)
    bottomRight[0] = np.max([bottomRight[0], br[0]])
    bottomRight[1] = np.max([bottomRight[1], br[1]])

    sys.stdout.write('\r' + "Frames " + np.string_(i) + " of " + np.string_(len(sequence[DICT_BBOXES])) + " done")
    sys.stdout.flush()
print
print "computed bbox", topLeft, bottomRight, "need", np.prod(bottomRight-topLeft)*3*8/1000000.0*len(sequence[DICT_BBOXES]), "MBs"; sys.stdout.flush()

numFrames = len(sequence[DICT_BBOXES])

bgPatch = np.array(Image.open(dataPath+dataSet+"median.png"))[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], 0:3]/255.0

## load floating point pixel data

paddedFrames = np.zeros(np.hstack([numFrames, np.prod((bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1])), 4]), dtype=np.float32)
for idx, key in enumerate(np.sort(sequence[DICT_BBOXES].keys())) :
    img = np.array(Image.open(tmpLoc+"frame-{0:05d}.png".format(key+1)), dtype=np.float32)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
    alpha = img[:, :, -1]/255.0
    paddedFrames[idx, :, :3] = ((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)) + 
                                 bgPatch*(1.0-np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))))[:, :, :3].reshape((paddedFrames.shape[1], 3)).astype(np.float32)
    
imageSize = np.array((bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1]))

## initialize features parameters

displacements = np.concatenate((np.array([[0], [0]], int),
                                np.array([np.zeros(len(dirDisplacements)), dirDisplacements], int), ## EAST
                                np.array([dirDisplacements, dirDisplacements], int), ## SOUT-EAST
                                np.array([dirDisplacements, np.zeros(len(dirDisplacements))], int), ## SOUTH
                                np.array([dirDisplacements, -dirDisplacements], int), ## SOUTH-WEST
                                np.array([np.zeros(len(dirDisplacements)), -dirDisplacements], int), ## WEST
                                np.array([-dirDisplacements, -dirDisplacements], int), ## NORTH-WEST
                                np.array([-dirDisplacements, np.zeros(len(dirDisplacements))], int), ## NORTH
                                np.array([-dirDisplacements, dirDisplacements], int), ## NORTH-EAST
                                ), axis=-1)

patchYs = np.arange(np.max(displacements), imageSize[0]-patchSize[0]-np.max(displacements), spacing)
patchYs = patchYs.reshape((1, len(patchYs)))
patchXs = np.arange(np.max(displacements), imageSize[1]-patchSize[1]-np.max(displacements), spacing)
patchXs = patchXs.reshape((1, len(patchXs)))
patchLocations = np.array(np.concatenate((patchYs.repeat(len(patchXs.T)),
                                          patchXs.repeat(len(patchYs.T), axis=0).flatten())).reshape((2, len(patchXs.T)*len(patchYs.T))), int)

frame1Idxs = np.empty((0, 2), dtype=int)
frame2Idxs = np.empty((0, 2), dtype=int)
for i, locSlice in enumerate(np.arange(patchXs.shape[-1], patchLocations.shape[-1]+1, patchXs.shape[-1])) :
    frame1Idxs = np.concatenate((frame1Idxs, np.repeat(patchLocations[:, locSlice-patchXs.shape[-1]:locSlice], len(displacements.T), axis=1).T), axis=0)
    frame2Idxs = np.concatenate((frame2Idxs, np.array([disp + loc for loc in patchLocations[:, locSlice-patchXs.shape[-1]:locSlice].T for disp in displacements.T])), axis=0)
    
stencils2D = cgf.stencil2D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])

# <codecell>

## initializing parameters for GPU computing
displacementsForCuda = displacements.T.astype(np.int32)[:, ::-1]
offset = np.array(np.max(displacementsForCuda, axis=0)[::-1], np.int32)

cudaGridSize = np.array([np.arange(offset[0], imageSize[0]-patchSize[0]-offset[0], spacing).shape[0],
                         np.arange(offset[1], imageSize[1]-patchSize[1]-offset[1], spacing).shape[0]])

M, N = cudaGridSize.astype(np.int32)

Q = np.int32(np.prod(patchSize/[32, 32])) ## number of 32x32 quadrants
D = np.int32(len(displacementsForCuda))
quadrants = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32) ## hardcoded for now but whatevs

## loading kernel
module = drv.module_from_file(cudaFolder + "computeFeat/computeFeat.cubin")
computeFeatsGPU32x32 = module.get_function("computeFeats32x32")

def computeFeatsGPU(frame1, frame2) :
    ## sending data to GPU
    displacements_gpu = gpuarray.to_gpu(displacementsForCuda.flatten())

    d_img1 = gpuarray.to_gpu(frame1)
    d_img2 = gpuarray.to_gpu(frame2)
    d_feat = gpuarray.to_gpu(np.zeros((M*N*D, 1), dtype=np.float32))
    
    
    ## running kernel
    computeFeatsGPU32x32(d_feat, d_img1, d_img2, displacements_gpu, M, N, D, offset[1], offset[0], np.int32(spacing),
                         np.int32(imageSize[0]), np.int32(imageSize[1]), block=(32, 32, 1), grid=(int(N), int(M*D)))

    ## getting result
    h_feat = np.sqrt(d_feat.get()).flatten()
    return h_feat

# <codecell>

goodPairsIdxs, badPairsIdxs = getTrainingExamples(dataPath+dataSet, numFrames, useValidatedJumps, True, verbose=False)
taggedFramesLoc = "/home/ilisescu/PhD/data/synthesisedSequences/wave-tagging_bad_jumps/tagged_frames.npy"
taggedFrames = np.load(taggedFramesLoc).item()
for key in taggedFrames :
    if taggedFrames[key][DICT_SEQUENCE_NAME] == sequence[DICT_SEQUENCE_NAME] :
        taggedFrames = taggedFrames[key][DICT_SEQUENCE_FRAMES]
        break
# print "tagged frames", taggedFrames; sys.stdout.flush()

badPairsIdxs = np.concatenate((badPairsIdxs, taggedFrames.T), axis=1)
print goodPairsIdxs
print badPairsIdxs

# <codecell>

def getGPUFeatsForPairs(pairsIdxs) :
    featsData = np.zeros((M*N*D, len(pairsIdxs.T)))
    avgTime = 0.0
    for i, pair in enumerate(pairsIdxs.T) :
#         print pair
        t = time.time()
        ## getting features
        featsData[:, i] = computeFeatsGPU(paddedFrames[pair[0], :, :], paddedFrames[pair[1], :, :])

        avgTime = (avgTime*i + time.time()-t)/(i+1)
        remainingTime = avgTime*(len(pairsIdxs.T)-i-1)/60.0
        if np.mod(i, 200) == 0 :
            sys.stdout.write('\r' + "Done pair " + np.string_(i) + " of " + np.string_(len(pairsIdxs.T)) +
                             " (avg time: " + np.string_(avgTime) + " secs --- remaining: " +
                             np.string_(int(np.floor(remainingTime))) + ":" + np.string_(int((remainingTime - np.floor(remainingTime))*60)) + ")")
            sys.stdout.flush()
    print 
    return featsData
        
## compute feats for each pair
startTime = time.time()

goodExamplesData = getGPUFeatsForPairs(goodPairsIdxs)
badExamplesData = getGPUFeatsForPairs(badPairsIdxs)
    
print "done in", time.time()-startTime

# <codecell>

def getTrainingData(goodPairsToUse, goodExamplesData, badPairsToUse, badExamplesData, saveLoc, labelMultiplier = 1) : 
    ### X has shape [featSize, numPairs]
        
    X = np.concatenate((goodExamplesData[:, goodPairsToUse], badExamplesData[:, badPairsToUse]), axis=1)
    w = np.concatenate((np.zeros(len(goodPairsToUse)), labelMultiplier*np.ones(len(badPairsToUse)))).reshape((X.shape[-1], 1))
    sio.savemat(saveLoc, {"X":X, "w":w})
    
    return X, w

trainingExamplesLoc = dataPath + dataSet + "trainingExamplesForImageData"
featsGoodSubset = arange(len(goodExamplesData[0]))
featsBadSubset = arange(len(badExamplesData[0]))
X, w = getTrainingData(arange(len(goodPairsIdxs.T)), goodExamplesData[:, featsGoodSubset], arange(len(badPairsIdxs.T)),
                       badExamplesData[:, featsBadSubset], trainingExamplesLoc)

phiSaveLoc = dataPath + dataSet + "fittedPhiForImageDataUsingPsi.mat"

matlabCommand = "cd ~/PhD/MATLAB/; matlab -nosplash -nodesktop -nodisplay -r "
matlabCommand += "\"fitPsiForRegression '" + trainingExamplesLoc + "' '"
matlabCommand += phiSaveLoc + "'; exit;\""

stat, output = commands.getstatusoutput(matlabCommand)
stat /= 256

if stat == 10 :
    print "Error when saving result"
elif stat == 11 :
    print "Error when loading examples"
else :
    print "Optimization completed with status", stat
    
print output

# <codecell>


# # pairsToCompute = np.argwhere(np.triu(np.ones((numFrames, numFrames)), k=1))
# print pairsToCompute.shape[0], M*N*D, pairsToCompute.shape[0]*M*N*D*4.0/1000000

# pairsToCompute = np.argwhere(np.triu(np.ones((numFrames, numFrames)), k=1))
# numSubsets = 2 #1
# subsetSize = len(pairsToCompute)/numSubsets
# gridStencils = cgf.stencil2D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])

# for i in arange(numSubsets)[1:] :
#     startTime = time.time()
# #     feats = np.zeros((subsetSize, len(gridStencils)))
    
#     print pairsToCompute[subsetSize*i:subsetSize*(i+1), :].T.shape
    
#     subsetFeats = getGPUFeatsForPairs(pairsToCompute[subsetSize*i:subsetSize*(i+1), :].T)
        
#     np.save("/".join(sequence[DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"+"features-"+sequence[DICT_SEQUENCE_NAME]+"-subset"+np.string_(i)+".npy", subsetFeats)
#     del subsetFeats

#     # print (f1s[stencilRowIdxs, stencilColIdxs, pairsIdxs].reshape((subsetSize, len(stencil[0])))/255.0).shape
#     print 
#     print "done subset", i, "in", time.time() - startTime, "secs"

# <codecell>

distMat = np.zeros((numFrames, numFrames))
regressedWeights = sio.loadmat(phiSaveLoc)["phi_MAP"]
for i, subsetLoc in enumerate(np.sort(glob.glob("/".join(sequence[DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"+"features-*.npy"))) :
    distMat[pairsToCompute[subsetSize*i:subsetSize*(i+1), 0],
            pairsToCompute[subsetSize*i:subsetSize*(i+1), 1]] = np.dot(np.load(subsetLoc).T, regressedWeights).flatten()
    print subsetLoc
distMat[pairsToCompute[:, 1], pairsToCompute[:, 0]] = distMat[pairsToCompute[:, 0], pairsToCompute[:, 1]]
distMat[arange(numFrames), arange(numFrames)] = np.max(distMat)
gwv.showCustomGraph(distMat)

# <codecell>

np.sort(sequence[DICT_BBOXES].keys())[814]

# <codecell>

print np.max(np.abs(regressedWeights-sio.loadmat(phiSaveLoc)["phi_MAP"]))

# <codecell>

filterSize = 0
GRAPH_MAX_COST = 10000000.0
if filterSize > 0 :
    optimizedDistMat = vtu.filterDistanceMatrix(distMat, filterSize, True)
else :
    optimizedDistMat = np.copy(distMat)

## if using vanilla
if False :
    optimizedDistMat = optimizedDistMat[1:optimizedDistMat.shape[1], 0:-1]
    correction = 1
else :
    ## setting diagonal to 0 as the learned distance won't have it
#     optimizedDistMat[np.argwhere(np.eye(numFrames-filterSize*2, k=1))[:, 0],
#                      np.argwhere(np.eye(numFrames-filterSize*2, k=1))[:, 1]] = 0
    correction = 0

## don't want to jump too close so increase costs in a window
minJumpLength = 20
tmp = (np.triu(np.ones(optimizedDistMat.shape), k=minJumpLength) +
       np.tril(np.ones(optimizedDistMat.shape), k=-minJumpLength) +
       np.eye(optimizedDistMat.shape[0], k=1))
tmp[tmp == 0] = 10.0
optimizedDistMat *= tmp
#########################################
## exponential
testCosts = np.copy(optimizedDistMat) #np.exp(np.copy(optimizedDistMat)/(np.average(optimizedDistMat)*0.15)) ## the lower the multiplier the more the stretch

#########################################
## do the thresholding based on how many jumps I want to keep per frame
desiredPercentage = 0.1 ## desired percentage of transitions to keep
jumpsToKeep = int(testCosts.shape[0]*desiredPercentage)
testCosts[np.arange(testCosts.shape[0]).repeat(testCosts.shape[0]-jumpsToKeep),
                       np.argsort(testCosts, axis=-1)[:, jumpsToKeep:].flatten()] = GRAPH_MAX_COST


## adding extra rows and columns so that the optimized matrix has the same dimensions as distMat
## for the indices that were cut out I put zero cost for jumps to frames that can still be used after optimization
testCosts = np.concatenate((np.ones((testCosts.shape[0], filterSize))*np.max(testCosts),
                                         testCosts,
                                         np.ones((testCosts.shape[0], filterSize+correction))*np.max(testCosts)), axis=1)
testCosts = np.concatenate((np.roll(np.concatenate((np.zeros((filterSize, 1)),
                                                                 np.ones((filterSize, distMat.shape[0]-1))*np.max(testCosts)), axis=1), filterSize, axis=1),
                                         testCosts,
                                         np.roll(np.concatenate((np.zeros((filterSize+correction, 1)),
                                                                 np.ones((filterSize+correction, distMat.shape[0]-1))*np.max(testCosts)), axis=1), filterSize, axis=1)), axis=0)




gwv.showCustomGraph(testCosts)

# <codecell>

# gwv.showCustomGraph(testCosts-np.load(np.load(sequence[DICT_SEQUENCE_LOCATION]).item()[DICT_TRANSITION_COSTS_LOCATION]))
print np.min(testCosts-np.load(np.load(sequence[DICT_SEQUENCE_LOCATION]).item()[DICT_TRANSITION_COSTS_LOCATION]))

# <codecell>

sequence[DICT_TRANSITION_COSTS_LOCATION] = ("/".join(sequence[DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"+
                                            "transition_costs-learned_thresholded_filtered-"+sequence[DICT_SEQUENCE_NAME]+".npy")
print "using sequence", sequence[DICT_SEQUENCE_LOCATION]
print
np.save(sequence[DICT_TRANSITION_COSTS_LOCATION], testCosts)
# np.save(sequence[DICT_SEQUENCE_LOCATION], sequence)
print "sequence uses for costs", np.load(sequence[DICT_SEQUENCE_LOCATION]).item()[DICT_TRANSITION_COSTS_LOCATION]

# <codecell>

pair = np.array([828, 1256]) ## pair for tara2
pair = np.array([71, 574]) ## pair for tara2
print pair
frame1 = paddedFrames[pair[0], :, :3].reshape((imageSize[0], imageSize[1], 3))
frame2 = paddedFrames[pair[1], :, :3].reshape((imageSize[0], imageSize[1], 3))

t = time.time()
features = getPairFeats(frame1*255.0, frame2*255.0, imageSize, framePaths, frame1Idxs, frame2Idxs, patchSize, stencils2D,
                        usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid)
print time.time() - t

# <codecell>

## visualize features
tmp = np.zeros(imageSize)
tmp[frame2Idxs[:, 0], frame2Idxs[:, 1]] = features
gwv.showCustomGraph(tmp)

# <codecell>

# paddedImages = np.zeros((len(framePaths), np.prod(imageSize), 4), dtype=np.float32)
# for i, path in enumerate(framePaths) :
#     paddedImages[i, :, :3] = np.array(Image.open(path))[:, :, :3].reshape((paddedImages.shape[1], 3)).astype(np.float32) / 255.0
    
#     sys.stdout.write('\r' + "Loaded image " + np.string_(i) + " of " + np.string_(len(framePaths)))
#     sys.stdout.flush()
        
clear_output()
print "loaded", paddedImages.shape[0], "( D =", paddedImages.shape[1], ")"

displacements = np.concatenate((np.array([np.zeros(len(dirDisplacements)), dirDisplacements], int), ## EAST
                                np.array([dirDisplacements, dirDisplacements], int), ## SOUT-EAST
                                np.array([dirDisplacements, np.zeros(len(dirDisplacements))], int), ## SOUTH
                                np.array([dirDisplacements, -dirDisplacements], int), ## SOUTH-WEST
                                np.array([np.zeros(len(dirDisplacements)), -dirDisplacements], int), ## WEST
                                np.array([-dirDisplacements, -dirDisplacements], int), ## NORTH-WEST
                                np.array([-dirDisplacements, np.zeros(len(dirDisplacements))], int), ## NORTH
                                np.array([-dirDisplacements, dirDisplacements], int), ## NORTH-EAST
                                ), axis=-1).T.astype(np.int32)

## flipping the columns so I can get x, y coords
displacements = displacements[:, ::-1]

# N = np.int32(len(img1))
# spacing = np.int32(32)
offset = np.array(np.max(displacements, axis=0)[::-1], np.int32)
# patchSize = np.array([64, 64], np.int32)
# imageSize = np.array([h, w], np.int32)

gridSize = np.array([np.arange(offset[0], imageSize[0]-patchSize[0]-offset[0], spacing).shape[0],
                     np.arange(offset[1], imageSize[1]-patchSize[1]-offset[1], spacing).shape[0]])

maxGPUmem = 3*1024**3
print maxGPUmem

# <codecell>

module = drv.module_from_file(cudaFolder + "computeFeat/computeFeat.cubin")
computeFeat = module.get_function("computeFeat")

# <codecell>

displacements_gpu = gpuarray.to_gpu(displacements.flatten())
patchSize_gpu = gpuarray.to_gpu(patchSize.astype(np.int32))
offset_gpu = gpuarray.to_gpu(offset)
imageSize_gpu = gpuarray.to_gpu(np.array(imageSize, np.int32))

print drv.mem_get_info()[1]-maxGPUmem, drv.mem_get_info()[0]

pair = badPairsIdxs[:, 0]
gpuImg1 = gpuarray.to_gpu(paddedImages[pair[0], :, :])
gpuImg2 = gpuarray.to_gpu(paddedImages[pair[1], :, :])
gpuRes = gpuarray.to_gpu(numpy.zeros((np.prod(gridSize), displacements.shape[0])).astype(np.float32))

computeFeat(gpuImg1, gpuImg2, displacements_gpu, patchSize_gpu, offset_gpu, np.int32(spacing), 
            imageSize_gpu, gpuRes, block=(displacements.shape[0], 1, 1), grid=(gridSize[1], gridSize[0]))

# img1_gpu = gpuarray.to_gpu(img1)
# img2_gpu = gpuarray.to_gpu(img2)
# res_gpu = gpuarray.to_gpu(numpy.zeros((np.prod(gridSize), displacements.shape[0])).astype(np.float32))


# ## run kernel
# computeFeat(img1_gpu, img2_gpu, displacements_gpu, patchSize_gpu, offset_gpu, spacing, imageSize_gpu, res_gpu, block=(displacements.shape[0], 1, 1), grid=(gridSize[1], gridSize[0]))

## read result
# res = np.empty_like(res)
# drv.memcpy_dtoh(res, res_gpu)
res = gpuRes.get()

print drv.mem_get_info()

# del displacements_gpu, patchSize_gpu, offset_gpu, imageSize_gpu

# <codecell>

# displacements_gpu = gpuarray.to_gpu(displacements.flatten())
# patchSize_gpu = gpuarray.to_gpu(patchSize.astype(np.int32))
# offset_gpu = gpuarray.to_gpu(offset)
# imageSize_gpu = gpuarray.to_gpu(np.array(imageSize, np.int32))

# print drv.mem_get_info()[1]-maxGPUmem, drv.mem_get_info()[0]

# gpuImg1 = []
# gpuImg2 = []
# gpuRes = []
# streams = []

# memPerKernel = (np.prod(imageSize)*4*4*2 + np.prod(gridSize)*displacements.shape[0]*4 + 
#                 displacements.nbytes + patchSize.astype(np.int32).nbytes + offset.nbytes + np.array(imageSize, np.int32).nbytes)
# approxMemUsage = 0

# while approxMemUsage < 2500000000 : #drv.mem_get_info()[1]-maxGPUmem < drv.mem_get_info()[0] and len(gpuRes) < len(goodPairsIdxs.T) :
# # for i in xrange(8) :
#     pair = goodPairsIdxs[:, len(gpuRes)]
#     gpuImg1.append(gpuarray.to_gpu(paddedImages[pair[0], :, :]))
#     gpuImg2.append(gpuarray.to_gpu(paddedImages[pair[1], :, :]))
#     gpuRes.append(gpuarray.to_gpu(numpy.zeros((np.prod(gridSize), displacements.shape[0])).astype(np.float32)))
    
# #     streams.append(drv.Stream())
    
# #     computeFeat(gpuImg1[-1], gpuImg2[-1], displacements_gpu, patchSize_gpu, offset_gpu, np.int32(spacing), 
# #                 imageSize_gpu, gpuRes[-1], block=(displacements.shape[0], 1, 1), grid=(gridSize[1], gridSize[0]))#, stream=streams[-1])
    
#     approxMemUsage += (memPerKernel*1.1)

#     sys.stdout.write('\r' + "Ran pair " + np.string_(len(gpuRes)) + " of " + np.string_(len(goodPairsIdxs.T)) +
#                      " (" + np.string_(approxMemUsage/1000000) + " MB used)")
# #                      " (" + np.string_(drv.mem_get_info()[0]/1000000) + ")")
# #     sys.stdout.flush()
# # img1_gpu = gpuarray.to_gpu(img1)
# # img2_gpu = gpuarray.to_gpu(img2)
# # res_gpu = gpuarray.to_gpu(numpy.zeros((np.prod(gridSize), displacements.shape[0])).astype(np.float32))


# # ## run kernel
# # computeFeat(img1_gpu, img2_gpu, displacements_gpu, patchSize_gpu, offset_gpu, spacing, imageSize_gpu, res_gpu, block=(displacements.shape[0], 1, 1), grid=(gridSize[1], gridSize[0]))

# # ## read result
# # # res = np.empty_like(res)
# # # drv.memcpy_dtoh(res, res_gpu)
# # res = res_gpu.get()

# # print drv.mem_get_info()

# # del displacements_gpu, patchSize_gpu, offset_gpu, imageSize_gpu

# <codecell>

module = drv.module_from_file(cudaFolder + "computeFeat/computeFeat.cubin")
computeFeatsGPU64x64 = module.get_function("computeFeats64x64")

M, N = gridSize.astype(np.int32)
pair = badPairsIdxs[:, 0]

Q = np.int32(np.prod(patchSize/[32, 32])) ## number of 32x32 quadrants
D = np.int32(len(displacements))
quadrants = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32) ## hardcoded for now but whatevs

t = time.time()
# offset_gpu = gpuarray.to_gpu(offset)
# imageSize_gpu = gpuarray.to_gpu(np.array(imageSize, np.int32))
displacements_gpu = gpuarray.to_gpu(displacements.flatten())
quadrants_gpu = gpuarray.to_gpu(quadrants.flatten())

d_img1 = gpuarray.to_gpu(paddedImages[pair[0], :, :])
d_img2 = gpuarray.to_gpu(paddedImages[pair[1], :, :])
d_feat = gpuarray.to_gpu(np.zeros((M*N*D, 1), dtype=np.float32))
# d_poop = gpuarray.to_gpu(np.zeros((1024, 1), dtype=np.float32))

# offset_gpu = gpuarray.to_gpu(offset)
# imageSize_gpu = gpuarray.to_gpu(np.array(imageSize, np.int32))

for i in xrange(1) :
    computeFeatsGPU64x64(d_feat, d_img1, d_img2, displacements_gpu, quadrants_gpu, M, N, D, offset[1], offset[0], np.int32(spacing),
                         np.int32(imageSize[0]), np.int32(imageSize[1]), block=(32, 32, 1), grid=(int(N*Q), int(M*D)))

h_feat = d_feat.get()

print time.time() - t
# print h_feat

# <codecell>

print features.shape

# <codecell>

module = drv.module_from_file(cudaFolder + "computeFeat/computeFeat.cubin")
# computeFeatsGPU32x32 = module.get_function("computeFeats32x32")
# computeFeatsGPU32x32 = module.get_function("computeFeats32x32small")
computeFeatsGPU32x32 = module.get_function("computeFeats32x32dispLoop")

M, N = gridSize.astype(np.int32)
pair = badPairsIdxs[:, 0]

Q = np.int32(np.prod(patchSize/[32, 32])) ## number of 32x32 quadrants
D = np.int32(len(displacements))
quadrants = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32) ## hardcoded for now but whatevs

t = time.time()
# offset_gpu = gpuarray.to_gpu(offset)
# imageSize_gpu = gpuarray.to_gpu(np.array(imageSize, np.int32))
displacements_gpu = gpuarray.to_gpu(displacements.flatten())
# quadrants_gpu = gpuarray.to_gpu(quadrants.flatten())

d_img1 = gpuarray.to_gpu(paddedImages[pair[0], :, :])
d_img2 = gpuarray.to_gpu(paddedImages[pair[1], :, :])
d_feat = gpuarray.to_gpu(np.zeros((M*N*D, 1), dtype=np.float32))
# d_poop = gpuarray.to_gpu(np.zeros((1024, 1), dtype=np.float32))

# offset_gpu = gpuarray.to_gpu(offset)
# imageSize_gpu = gpuarray.to_gpu(np.array(imageSize, np.int32))

for i in xrange(1) :
#     computeFeatsGPU32x32(d_feat, d_img1, d_img2, displacements_gpu, M, N, D, offset[1], offset[0], np.int32(spacing),
#                          np.int32(imageSize[0]), np.int32(imageSize[1]), block=(32, 32, 1), grid=(int(N), int(M*D)))
#     computeFeatsGPU32x32(d_feat, d_img1, d_img2, displacements_gpu, M, N, D, offset[1], offset[0], np.int32(spacing),
#                          np.int32(imageSize[0]), np.int32(imageSize[1]), block=(32, 4, 1), grid=(int(N*8), int(M*D)))
    computeFeatsGPU32x32(d_feat, d_img1, d_img2, displacements_gpu, M, N, D, offset[1], offset[0], np.int32(spacing),
                         np.int32(imageSize[0]), np.int32(imageSize[1]), block=(32, 4, 1), grid=(int(N*8), int(M)))

h_feat = d_feat.get()

print time.time() - t
# print h_feat

# <codecell>

tmpFolder = "/home/ilisescu/"
# seqName = "james3"
seqName = "aron1"

taggedFramesLoc = "/home/ilisescu/PhD/data/synthesisedSequences/wave/tagged_frames.npy"
taggedFrames = np.load(taggedFramesLoc).item()
for key in taggedFrames :
    if taggedFrames[key][DICT_SEQUENCE_NAME] == seqName :
        taggedFrames = taggedFrames[key][DICT_SEQUENCE_FRAMES]
        break
print "tagged frames", taggedFrames; sys.stdout.flush()
        
sequence = np.load(dataPath+dataSet+"semantic_sequence-"+seqName+".npy").item()
tmpLoc = tmpFolder+"".join(sequence[DICT_MASK_LOCATION].split("/")[-2])+"/"
print "starting", sequence[DICT_SEQUENCE_NAME]; sys.stdout.flush()

t = time.time()
if not os.path.isdir(tmpLoc) :
    copyanything(sequence[DICT_MASK_LOCATION], tmpLoc)
    print "copied folder to", tmpLoc; sys.stdout.flush()
else :
    print tmpLoc, "already exists"; sys.stdout.flush()

topLeft = np.array([720, 1280])
bottomRight = np.array([0, 0])
for i in np.sort(sequence[DICT_BBOXES].keys()) :
    alpha = np.array(Image.open(tmpLoc+"frame-{0:05d}.png".format(i+1)))[:, :, -1]
    vis = np.argwhere(alpha != 0)
    tl = np.min(vis, axis=0)
    topLeft[0] = np.min([topLeft[0], tl[0]])
    topLeft[1] = np.min([topLeft[1], tl[1]])

    br = np.max(vis, axis=0)
    bottomRight[0] = np.max([bottomRight[0], br[0]])
    bottomRight[1] = np.max([bottomRight[1], br[1]])

    sys.stdout.write('\r' + "Frames " + np.string_(i) + " of " + np.string_(len(sequence[DICT_BBOXES])) + " done")
    sys.stdout.flush()
print
print "computed bbox", topLeft, bottomRight, "need", np.prod(bottomRight-topLeft)*3*8/1000000.0*len(sequence[DICT_BBOXES]), "MBs"; sys.stdout.flush()

goodPairsIdxs, badPairsIdxs = getTrainingExamples(dataPath+dataSet, False, False, verbose=False)
badPairsIdxs = np.concatenate((badPairsIdxs, taggedFrames.T), axis=1)

patchSize = np.array([64, 64], int)
spacing = 32.0 ## how far away a patch is from the previous one

dirDisplacements = np.array([1, 2, 4, 8, 16])
gridSize = np.array((20, 20))

usePatchDisplace = False
useSSD = True
useFlow = False
useAugment = False
useSigmoid = False

imageSize = bottomRight-topLeft
framePaths = np.sort(glob.glob(tmpLoc + "frame*.png"))

goodExamplesData = getPairFeatsData(patchSize, spacing, dirDisplacements, imageSize, goodPairsIdxs, gridSize, framePaths,
                                   usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid, subPatchTopLeft=topLeft)

badExamplesData = getPairFeatsData(patchSize, spacing, dirDisplacements, imageSize, badPairsIdxs, gridSize, framePaths,
                                   usePatchDisplace, useSSD, useFlow, useAugment, useSigmoid, subPatchTopLeft=topLeft)


# numFrames = len(sprite[DICT_BBOXES])
# if np.prod(bottomRight-topLeft)*3*8/1000000.0*len(sprite[DICT_BBOXES]) > 15000 :
#     numBlocks = 4
# else :
#     numBlocks = 1
# blockSize = numFrames/numBlocks
# distMat = np.zeros([numFrames, numFrames])

# for i in xrange(0, numBlocks) :

#     f1s = np.zeros(np.hstack([bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1], 3, blockSize]), dtype=np.float64)
#     for idx, frame in enumerate(np.sort(glob.glob(tmpFolder+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-0*.png"))[i*blockSize:i*blockSize+blockSize]) :
#         img = np.array(Image.open(frame), dtype=np.float64)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
#         alpha = img[:, :, -1]/255.0
#         f1s[:, :, :, idx] = (img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))

#     print "loaded frames, block i=", i; sys.stdout.flush()

#     data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
#     distMat[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = ssd(data1)
#     print "computed distance"; sys.stdout.flush()

#     for j in xrange(i+1, numBlocks) :

#         f2s = np.zeros(np.hstack([bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1], 3, blockSize]), dtype=np.float64)
#         for idx, frame in enumerate(np.sort(glob.glob(tmpFolder+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-0*.png"))[j*blockSize:j*blockSize+blockSize]) :
#             img = np.array(Image.open(frame), dtype=np.float64)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
#             alpha = img[:, :, -1]/255.0
#             f2s[:, :, :, idx] = (img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))

#         print "loaded frames, block j=", j; sys.stdout.flush()
#         data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
#         distMat[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize] = ssd2(data1, data2)
#         distMat[j*blockSize:j*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = distMat[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize].T
#         print "computed distance"; sys.stdout.flush()

#         del f2s, data2


# ## due to imprecision I need to make this check
# distMat[distMat > 0.0] = np.sqrt(distMat[distMat > 0.0])
# distMat[distMat <= 0.0] = 0.0

shutil.rmtree(tmpLoc)

# np.save(dataPath+dataSet+sprite[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy", distMat)
# del f1s, data1

# print "done", sprite[DICT_SEQUENCE_NAME], "in", time.time() - t; sys.stdout.flush()
# print 
# print

# <codecell>

def getTrainingData(goodPairsToUse, goodExamplesData, badPairsToUse, badExamplesData, saveLoc, labelMultiplier = 1) :  
    ### X has shape [featSize, numPairs]  
        
    X = np.concatenate((goodExamplesData[:, goodPairsToUse], badExamplesData[:, badPairsToUse]), axis=1)
    w = np.concatenate((np.zeros(len(goodPairsToUse)), labelMultiplier*np.ones(len(badPairsToUse)))).reshape((X.shape[-1], 1))
    sio.savemat(saveLoc, {"X":X, "w":w})
    
    return X, w

trainingExamplesLoc = dataPath + dataSet + "trainingExamplesForImageData"
featsSubset = arange(len(goodExamplesData[0]))
X, w = getTrainingData(arange(len(goodPairsIdxs.T)), np.array(goodExamplesData).T[featsSubset, :], arange(len(badPairsIdxs.T)),
                       np.array(badExamplesData).T[featsSubset, :], trainingExamplesLoc)

phiSaveLoc = dataPath + dataSet + "fittedPhiForImageDataUsingPsi.mat"

matlabCommand = "cd ~/PhD/MATLAB/; matlab -nosplash -nodesktop -nodisplay -r "
matlabCommand += "\"fitPsiForRegression '" + trainingExamplesLoc + "' '"
matlabCommand += phiSaveLoc + "'; exit;\""

stat, output = commands.getstatusoutput(matlabCommand)
stat /= 256

if stat == 10 :
    print "Error when saving result"
elif stat == 11 :
    print "Error when loading examples"
else :
    print "Optimization completed with status", stat
    
print output

# <codecell>

phiSaveLoc = dataPath + dataSet + "fittedPhiForImageDataUsingPsi.mat"
copyanything(sequence[DICT_MASK_LOCATION], tmpLoc)
regressedWeights = sio.loadmat(phiSaveLoc)['phi_MAP']

numFrames = len(sequence[DICT_BBOXES])
if np.prod(bottomRight-topLeft)*3*8/1000000.0*len(sequence[DICT_BBOXES]) > 15000 :
    numBlocks = 4
else :
    numBlocks = 1
blockSize = numFrames/numBlocks
distMat = np.zeros([numFrames, numFrames])
print "blocks", numBlocks

# for i in xrange(0, numBlocks) :
i = 0

# f1s = np.zeros(np.hstack([bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1], 3, blockSize]), dtype=np.float64)
# for idx, frame in enumerate(np.sort(glob.glob(tmpLoc+"frame-0*.png"))[i*blockSize:i*blockSize+blockSize]) :
#     img = np.array(Image.open(frame), dtype=np.float64)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
#     alpha = img[:, :, -1]/255.0
#     f1s[:, :, :, idx] = (img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))

f1s = np.zeros(np.hstack([bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1], blockSize]), dtype=np.uint8)
for idx, frame in enumerate(np.sort(glob.glob(tmpLoc+"frame-0*.png"))[i*blockSize:i*blockSize+blockSize]) :
    img = np.array(Image.open(frame), dtype=np.uint8)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
#     alpha = img[:, :, -1]/255.0
#     f1s[:, :, :, idx] = np.array((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), dtype=np.uint8)
    f1s[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    if np.mod(idx, 200) == 0 :
        print "loaded", idx; sys.stdout.flush()

print "loaded frames, block i=", i; sys.stdout.flush()

# data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
# distMat[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = ssd(data1)
print "computed distance"; sys.stdout.flush()

# <codecell>

### computes subsets of the full feature set
# frameSubset = f1s[:, :, pairsToCompute[:30000, 0]]
# pairsToCompute = goodPairsIdxs.T
# pairsToCompute = badPairsIdxs.T
pairsToCompute = np.argwhere(np.triu(np.ones((numFrames, numFrames)), k=1))
numSubsets = 8 #1
subsetSize = len(pairsToCompute)/numSubsets
gridStencils = cgf.stencil2D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])

for i in arange(numSubsets)[6:] :
    startTime = time.time()
    feats = np.zeros((subsetSize, len(gridStencils)))

    pairsIdxs1 = pairsToCompute[subsetSize*i:subsetSize*(i+1), 0].repeat(len(gridStencils[0][0]))
    pairsIdxs2 = pairsToCompute[subsetSize*i:subsetSize*(i+1), 1].repeat(len(gridStencils[0][0]))
    
    feats[:, :270] = np.load("tmpfeats.npy")

    avgTime = 0.0
    for j, stencil in enumerate(gridStencils[270:]) :
        t = time.time()
        ## add SSD to feature vector
        stencilRowIdxs = stencil[0].reshape((1, len(stencil[0]))).repeat(subsetSize, axis=0).flatten()
        stencilColIdxs = stencil[1].reshape((1, len(stencil[1]))).repeat(subsetSize, axis=0).flatten()
        
        
        feats[:, j] = np.average((f1s[stencilRowIdxs, stencilColIdxs, pairsIdxs1].reshape((subsetSize, len(stencil[0])))/255.0-
                                  f1s[stencilRowIdxs, stencilColIdxs, pairsIdxs2].reshape((subsetSize, len(stencil[0])))/255.0)**2, axis=1)
        
#         feats = np.concatenate((feats, [np.average((frame1[stencil]/255.0-frame2[stencil]/255.0)**2)]))

        avgTime = (avgTime*j + time.time()-t)/(j+1)
        remainingTime = avgTime*(len(gridStencils)-j-1)/60.0
        sys.stdout.write('\r' + "Done stencil " + np.string_(j) + " of " + np.string_(len(gridStencils)) +
                         " (avg time: " + np.string_(avgTime) + " secs --- remaining: " +
                         np.string_(int(np.floor(remainingTime))) + ":" + np.string_(int((remainingTime - np.floor(remainingTime))*60)) + ")")
        sys.stdout.flush()
        
    np.save("/".join(sequence[DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"+"features-"+sequence[DICT_SEQUENCE_NAME]+"-subset"+np.string_(i)+".npy", feats)

    # print (f1s[stencilRowIdxs, stencilColIdxs, pairsIdxs].reshape((subsetSize, len(stencil[0])))/255.0).shape
    print 
    print "done subset", i, "in", time.time() - startTime, "secs"

# <codecell>

np.save("tmpfeats.npy", feats[:, :270])

# <codecell>

distMat = np.zeros((numFrames, numFrames))
for i, subsetLoc in enumerate(np.sort(glob.glob("/".join(sequence[DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"+"features-*.npy"))) :
    distMat[pairsToCompute[subsetSize*i:subsetSize*(i+1), 0],
            pairsToCompute[subsetSize*i:subsetSize*(i+1), 1]] = np.dot(np.load(subsetLoc), regressedWeights).flatten()
    print subsetLoc
distMat[pairsToCompute[:, 1], pairsToCompute[:, 0]] = distMat[pairsToCompute[:, 0], pairsToCompute[:, 1]]
distMat[arange(numFrames), arange(numFrames)] = np.max(distMat)
gwv.showCustomGraph(distMat)

# <codecell>

np.save("/".join(sequence[DICT_SEQUENCE_LOCATION].split("/")[:-1])+"/"+sequence[DICT_SEQUENCE_NAME]+"-learned_distMat.npy", distMat)

# <codecell>

# print feats.shape
# print np.array(goodExamplesData).shape
# print np.argwhere(feats-np.array(goodExamplesData) != 0.0)
# print np.argwhere(feats != 0.0)
# print np.argwhere(np.array(goodExamplesData) != 0.0)
# print np.argwhere(np.argwhere(feats != 0.0) - np.argwhere(np.array(goodExamplesData) != 0.0) != 0.0)
print feats.shape
print np.array(badExamplesData).shape
print np.argwhere(feats-np.array(badExamplesData) != 0.0)
print np.argwhere(feats != 0.0)
print np.argwhere(np.array(badExamplesData) != 0.0)
print np.argwhere(np.argwhere(feats != 0.0) - np.argwhere(np.array(badExamplesData) != 0.0) != 0.0)

# <codecell>

subsetSize = 2
stencil = gridStencils[87]
print pairsToCompute[:2, :]
stencilRowIdxs = stencil[0].reshape((1, len(stencil[0]))).repeat(subsetSize, axis=0).flatten()
stencilColIdxs = stencil[1].reshape((1, len(stencil[1]))).repeat(subsetSize, axis=0).flatten()
pairsIdxs = pairsToCompute[:subsetSize, 1].repeat(len(stencil[0]))
print f1s[stencilRowIdxs, stencilColIdxs, pairsIdxs].reshape((subsetSize, len(stencil[0])))

# <codecell>

print frameSubset[stencil[0], stencil[1], :].shape
# figure(); imshow(f1s[:, :, 2400])

# <codecell>

gridStencils = cgf.stencil2D(gridSize[0], gridSize[1], [imageSize[0], imageSize[1], 3])
frame1 = f1s[:, :, :, 0]
for j, stencil in enumerate(gridStencils) :
    ## add SSD to feature vector
    print len(stencil[0])
    print frame1[:, :, 1][stencil].shape
#     feats = np.concatenate((feats, [np.average((frame1[stencil]/255.0-frame2[stencil]/255.0)**2)]))

# <codecell>

# gwv.showCustomGraph(sio.loadmat(phiSaveLoc)["phi_MAP"].reshape(gridSize))
print "Weighted", np.sqrt(np.dot(X[:, -1].reshape((1, np.prod(gridSize))), sio.loadmat(phiSaveLoc)['phi_MAP'])) 
print "L2", np.sqrt(np.dot(X[:, -1].reshape((1, np.prod(gridSize))), np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))

# <codecell>

print "Weighted", np.sqrt(np.dot(X[:, 1].reshape((1, np.prod(gridSize))), sio.loadmat(phiSaveLoc)['phi_MAP'])) 
print "L2", np.sqrt(np.dot(X[:, 1].reshape((1, np.prod(gridSize))), np.ones_like(sio.loadmat(phiSaveLoc)['phi_MAP'])))

