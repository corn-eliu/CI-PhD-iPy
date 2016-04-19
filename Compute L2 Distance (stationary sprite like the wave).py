# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab 
import numpy as np
import scipy as sp
import cv2
import glob
import time
import sys
import os
import shutil, errno
import psutil

from PIL import Image
import GraphWithValues as gwv

# <codecell>

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
            
## compute euclidean distance assuming f is an array where each row is a flattened image (1xN array, N=W*H*Channels)
## euclidean distance defined as the length of the the displacement vector:
## len(q-p) = sqrt(len(q)^2+len(p)^2 - 2*dot(p, q)) where p and q are two images in vector format and 1xN size
def ssd(f) :
    ## gives sum over squared intensity values for each image
    ff = np.sum(f*f, axis=1)
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
    d = np.reshape(ff, [len(ff),1])+ff.T - 2*np.dot(f, f.T)
    return d

def ssd2(f1, f2) :
    ## gives sum over squared intensity values for each image
    ff1 = np.sum(f1*f1, axis=1)
    ff2 = np.sum(f2*f2, axis=1)
#     print ff1.shape
#     print ff2.shape
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
#     print "askdfh", np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1).shape, np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0).shape
    d = np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1)+np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0) - 2*np.dot(f1, f2.T)
    return d

# <codecell>

## load the tracked sprites
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
DICT_FRAME_COMPATIBILITY_LABELS = 'compatibiliy_labels_per_frame'
DICT_LABELLED_FRAMES = 'labelled_frames' ## includes the frames labelled for the semantic labels (the first [DICT_FRAME_SEMANTICS].shape[1])
DICT_NUM_EXTRA_FRAMES = 'num_extra_frames' ## same len as DICT_LABELLED_FRAMES
DICT_CONFLICTING_SEQUENCES = 'conflicting_sequences'
DICT_DISTANCE_MATRIX_LOCATION = 'sequence_precomputed_distance_matrix_location' ## for label propagation

# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
# dataSet = "wave1/"
# dataSet = "wave2/"
# dataSet = "wave3/"
# dataSet = "windows/"
dataSet = "digger/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
bgImage = np.array(Image.open(dataPath + dataSet + "median.png"))

allXs = arange(bgImage.shape[1], dtype=float32).reshape((1, bgImage.shape[1])).repeat(bgImage.shape[0], axis=0)
allYs = arange(bgImage.shape[0], dtype=float32).reshape((bgImage.shape[0], 1)).repeat(bgImage.shape[1], axis=1)

trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
    print trackedSprites[-1][DICT_SEQUENCE_NAME]

## merge tracked sprite with bg
spriteIdx = 0
sequenceLength = len(trackedSprites[spriteIdx][DICT_BBOXES])

# <codecell>

# semanticSequence = np.load("/media/ilisescu/Data1/PhD/data/digger/semantic_sequence-digger_right1.npy").item()
# semanticSequence = np.load("/media/ilisescu/Data1/PhD/data/havana/semantic_sequence-black_car1.npy").item()
semanticSequence = np.load("/media/ilisescu/Data1/PhD/data/toy/semantic_sequence-toy1.npy").item()
# semanticSequence = np.load("/media/ilisescu/Data1/PhD/data/candle_wind/semantic_sequence-candle_wind1.npy").item()
print semanticSequence.keys()
print "/".join(semanticSequence[DICT_SEQUENCE_LOCATION].split("/")[:-1]) + "/"
# print semanticSequence[DICT_MASK_LOCATION]

# <codecell>

# semanticSequence[DICT_LABELLED_FRAMES] = [[17, 344],                                  ## None
#                                           [84, 574],                                  ## C
#                                           [116, 543],                                 ## D
#                                           [148, 516],                                 ## E
#                                           [180, 487],                                 ## F
#                                           [212, 457],                                 ## G
#                                           [243, 431],                                 ## A
#                                           [273, 399],                                 ## B
#                                           [308, 369]]                                 ## C
# semanticSequence[DICT_NUM_EXTRA_FRAMES] = [[4, 4],                                     ## None
#                                            [4, 4],                                     ## C
#                                            [4, 4],                                     ## D
#                                            [4, 4],                                     ## E
#                                            [4, 4],                                     ## F
#                                            [4, 4],                                     ## G
#                                            [4, 4],                                     ## A
#                                            [4, 4],                                     ## B
#                                            [4, 4]]                                     ## C

# semanticSequence[DICT_DISTANCE_MATRIX_LOCATION] = "/media/ilisescu/Data1/PhD/data/toy/overlap_normalization_distMat-toy1.npy"
# np.save(semanticSequence[DICT_SEQUENCE_LOCATION], semanticSequence)

# <codecell>

# gwv.showCustomGraph(np.load(semanticSequence[DICT_DISTANCE_MATRIX_LOCATION]))
gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/toy/toy1-vanilla_distMat.npy"))

# <codecell>

print semanticSequence.keys()
print semanticSequence[DICT_SEQUENCE_LOCATION]
# print semanticSequence[DICT_BBOXES]
# np.save(semanticSequence[DICT_SEQUENCE_LOCATION], semanticSequence)

# <codecell>

# del semanticSequence[DICT_BBOXES], semanticSequence[DICT_BBOX_CENTERS], semanticSequence[DICT_BBOX_ROTATIONS], semanticSequence[DICT_FOOTPRINTS]

# <codecell>

# tmpFolder = "/home/ilisescu/"

print "starting", semanticSequence[DICT_SEQUENCE_NAME]; sys.stdout.flush()
t = time.time()
numFrames = len(semanticSequence[DICT_FRAMES_LOCATIONS].keys())
if numFrames > 0 :
    progress = 0.0
    sequenceLocation = "/".join(semanticSequence[DICT_SEQUENCE_LOCATION].split("/")[:-1]) + "/"
    ## get keys of tracked frames and size of frame
    sortedKeys = np.sort(semanticSequence[DICT_FRAMES_LOCATIONS].keys())
    frameSize = np.array(Image.open(semanticSequence[DICT_FRAMES_LOCATIONS][sortedKeys[0]])).shape[:2]
    budget = 0.25
    
    ## find sub-patch if frames have been segmented
    if DICT_MASK_LOCATION in semanticSequence.keys() :
        topLeft = np.array([frameSize[0], frameSize[1]])
        bottomRight = np.array([0, 0])
        frameLocs = np.sort(glob.glob(semanticSequence[DICT_MASK_LOCATION]+"frame-0*.png"))
        for i, frameLoc in enumerate(frameLocs) :
            alpha = np.array(Image.open(frameLoc))[:, :, -1]
            vis = np.argwhere(alpha != 0)
            tl = np.min(vis, axis=0)
            topLeft[0] = np.min([topLeft[0], tl[0]])
            topLeft[1] = np.min([topLeft[1], tl[1]])

            br = np.max(vis, axis=0)
            bottomRight[0] = np.max([bottomRight[0], br[0]])
            bottomRight[1] = np.max([bottomRight[1], br[1]])

#             sys.stdout.write('\r' + "Frames " + np.string_(i) + " of " + np.string_(len(semanticSequence[DICT_BBOXES])) + " done")
#             sys.stdout.flush()
            
            ##
            progress += 1.0/len(frameLocs)*budget
            if np.mod(i, 10) == 0 :
                print "done", progress
                
        print
        print "computed bbox", topLeft, bottomRight; sys.stdout.flush()

        bgPatch = np.array(Image.open(sequenceLocation+"median.png"))[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], 0:3]/255.0
    else :
        topLeft = np.array([0, 0])
        bottomRight = np.array([frameSize[0], frameSize[1]])
        frameLocs = np.sort([semanticSequence[DICT_FRAMES_LOCATIONS][key] for key in sortedKeys])
        bgPatch = np.zeros([frameSize[0], frameSize[1], 3])
        
        ##
        progress += budget
        print "done", progress
        
    budget = 0.1
            
    ## render bboxes
    if DICT_BBOXES in semanticSequence.keys() :
        numVisibile = np.zeros(numFrames, int)
        renderedBBoxes = np.zeros((np.prod(frameSize), numFrames), np.uint8)
        for i, key in enumerate(sortedKeys) :
            img = np.zeros((frameSize[0], frameSize[1]), np.uint8)
            cv2.fillConvexPoly(img, semanticSequence[DICT_BBOXES][key].astype(int)[[0, 1, 2, 3, 0], :], 1)
            renderedBBoxes[:, i] = img.flatten()
            numVisibile[i] = len(np.argwhere(img.flatten() == 1))
            ##
            progress += 1.0/len(sortedKeys)*budget
            if np.mod(i, 10) == 0 :
                print "done", progress
    else :
        numVisibile = np.ones(numFrames, int)*np.prod(frameSize)
        renderedBBoxes = np.ones((1, numFrames), np.uint8)*np.sqrt(np.prod(frameSize))

        ##
        progress += budget
        print "done", progress
    
    ## figure out how to split the data to fit into memory
    memNeededPerFrame = np.prod(bgPatch.shape)*8/1000000.0#*len(semanticSequence[DICT_BBOXES])
    maxMemToUse = psutil.virtual_memory()[1]/1000000*0.4/2 ## use 0.4 of the available memory
    numFramesThatFit = np.round(maxMemToUse/memNeededPerFrame)
    numBlocks = int(np.ceil(numFrames/numFramesThatFit))
    blockSize = int(np.ceil(numFrames/float(numBlocks)))
    print "need", memNeededPerFrame*numFrames, "MBs: splitting into", numBlocks, "blocks of", blockSize, "frames (", blockSize*memNeededPerFrame, "MBs)"; sys.stdout.flush()

    frameIdxs = np.arange(numFrames)
    distMat = np.zeros([numFrames, numFrames])
    
    ##
    budget = 0.6
    totalBlocks = np.sum(arange(1, numBlocks+1))
    for i in xrange(numBlocks) :
        idxsToUse1 = frameIdxs[i*blockSize:(i+1)*blockSize]

        f1s = np.zeros(np.hstack([bgPatch.shape[0], bgPatch.shape[1], 3, len(idxsToUse1)]), dtype=float)
        for idx, frame in enumerate(frameLocs[idxsToUse1]) :
            if DICT_MASK_LOCATION in semanticSequence.keys() :
                img = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
                alpha = img[:, :, -1]/255.0
                f1s[:, :, :, idx] = ((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)) + 
                                     bgPatch*(1.0-np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))))
            else :
                f1s[:, :, :, idx] = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :3]/255.0

        print "loaded frames, block i=", i; sys.stdout.flush()
        data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
        distMat[i*blockSize:i*blockSize+len(idxsToUse1), i*blockSize:i*blockSize+len(idxsToUse1)] = ssd(data1)
        print "computed distance", distMat[i*blockSize:i*blockSize+len(idxsToUse1), i*blockSize:i*blockSize+len(idxsToUse1)].shape; sys.stdout.flush()
        
        ##
        progress += 1.0/totalBlocks*budget
        print "done", progress

        for j in xrange(i+1, numBlocks) :
            idxsToUse2 = frameIdxs[j*blockSize:(j+1)*blockSize]

            f2s = np.zeros(np.hstack([bgPatch.shape[0], bgPatch.shape[1], 3, len(idxsToUse2)]), dtype=float)
            for idx, frame in enumerate(frameLocs[idxsToUse2]) :
                if DICT_MASK_LOCATION in semanticSequence.keys() :
                    img = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
                    alpha = img[:, :, -1]/255.0
                    f2s[:, :, :, idx] = ((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)) + 
                                         bgPatch*(1.0-np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))))
                else :
                    f2s[:, :, :, idx] = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :3]/255.0

            print "loaded frames, block j=", j; sys.stdout.flush()
            data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
            distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)] = ssd2(data1, data2)
            distMat[j*blockSize:j*blockSize+len(idxsToUse2), i*blockSize:i*blockSize+len(idxsToUse1)] = distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)].T
            print "computed distance", distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)].shape; sys.stdout.flush()

            del f2s, data2
            
            ##
            progress += 1.0/totalBlocks*budget
            print "done", progress

        del f1s, data1


    ## due to imprecision I need to make this check
    distMat[distMat > 0.0] = np.sqrt(distMat[distMat > 0.0])
    distMat[distMat <= 0.0] = 0.0

    #     np.save(dataPath+dataSet+semanticSequence[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy", distMat/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize))))
    
    tmp = np.copy(renderedBBoxes.T).astype(float)
    numOverlapPixels = np.dot(tmp, tmp.T)
    del tmp
    print "done", 1.0

    print "done", semanticSequence[DICT_SEQUENCE_NAME], "in", time.time() - t; sys.stdout.flush()
    print
    print

# <codecell>

# gwv.showCustomGraph(renderedBBoxes[:, 0].reshape(frameSize[:2]))
# tmp = np.copy(renderedBBoxes.T).astype(float)
# lala = np.dot(tmp, tmp.T)
# del tmp
print lala
# print renderedBBoxes.shape
print frameSize, np.prod(frameSize)

# <codecell>

# gwv.showCustomGraph(renderedBBoxes[:, 0].reshape(frameSize[:2]))
# tmp = np.copy(renderedBBoxes.T).astype(float)
# lala = np.dot(tmp, tmp.T)
# del tmp
print lala
# print renderedBBoxes.shape
print frameSize, np.prod(frameSize)
print 921600*921600, np.sqrt(921600)

# <codecell>

# gwv.showCustomGraph(distMat)
print ((2.0*numOverlapPixels)/(numVisibile.reshape((numFrames, 1)) + numVisibile.reshape((1, numFrames)))+0.01)
# print numVisibile.reshape((numFrames, 1)) + numVisibile.reshape((1, numFrames))
print numOverlapPixels
# print renderedBBoxes

# <codecell>

# print frameLocs
print frameLocs.shape
print idxsToUse1

# <codecell>

print np.sum(arange(1, numBlocks+1)), numBlocks
totalBlocks = np.sum(arange(1, numBlocks+1))
budget = 0.6
progress = 0.0
for i in xrange(numBlocks) :
    progress += 1.0/totalBlocks*budget
    print progress
    for j in xrange(i+1, numBlocks) :
        progress += 1.0/totalBlocks*budget
        print progress

# <codecell>

gwv.showCustomGraph(distMat)
gwv.showCustomGraph(distMat/((2.0*dotOverlap)/(numVisibile.reshape((numFrames, 1)) + numVisibile.reshape((1, numFrames)))+0.01))
gwv.showCustomGraph(np.load("/media/ilisescu/Data1/PhD/data/toy/toy1-vanilla_distMat.npy"))

# <codecell>

print np.max(np.abs(np.load("/media/ilisescu/Data1/PhD/data/toy/toy1-vanilla_distMat.npy")-
                    (distMat/((2.0*dotOverlap)/(numVisibile.reshape((numFrames, 1)) + numVisibile.reshape((1, numFrames)))+0.01))))

# <codecell>

((2.0*dotOverlap)/(numVisibile.reshape((numFrames, 1)) + numVisibile.reshape((1, numFrames)))+0.01)

# <codecell>

tmp = np.copy(renderedBBoxes.reshape((np.prod(renderedBBoxes.shape[0:2]), renderedBBoxes.shape[2])).T).astype(float)
numOverlapPixels = np.dot(tmp, tmp.T)
del tmp

# <codecell>

gwv.showCustomGraph(numOverlapPixels)

# <codecell>

tmpFolder = "/home/ilisescu/"
for s in [2] :#arange(len(trackedSprites))[0:] : #np.roll(arange(len(trackedSprites)), -1) :
    sprite = trackedSprites[s]
    print "starting", sprite[DICT_SEQUENCE_NAME]; sys.stdout.flush()
    t = time.time()
    topLeft = np.array([720, 1280])
    bottomRight = np.array([0, 0])
    os.mkdir(tmpFolder+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/")
    for i in np.sort(sprite[DICT_BBOXES].keys()) :
        copyanything(dataPath+dataSet+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-{0:05d}.png".format(i+1), 
                     tmpFolder+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-{0:05d}.png".format(i+1))
        
        
    print "copied folder to tmp"; sys.stdout.flush()
    for i, frameLoc in enumerate(np.sort(glob.glob(tmpFolder+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-0*.png"))) :
        alpha = np.array(Image.open(frameLoc))[:, :, -1]
        vis = np.argwhere(alpha != 0)
        tl = np.min(vis, axis=0)
        topLeft[0] = np.min([topLeft[0], tl[0]])
        topLeft[1] = np.min([topLeft[1], tl[1]])

        br = np.max(vis, axis=0)
        bottomRight[0] = np.max([bottomRight[0], br[0]])
        bottomRight[1] = np.max([bottomRight[1], br[1]])

        sys.stdout.write('\r' + "Frames " + np.string_(i) + " of " + np.string_(len(sprite[DICT_BBOXES])) + " done")
        sys.stdout.flush()
    print
    print "computed bbox", topLeft, bottomRight, "need", np.prod(bottomRight-topLeft)*3*8/1000000.0*len(sprite[DICT_BBOXES]), "MBs"; sys.stdout.flush()
    
    bgPatch = np.array(Image.open(dataPath+dataSet+"median.png"))[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], 0:3]/255.0
    
    numFrames = len(sprite[DICT_BBOXES])
#     if True or np.prod(bottomRight-topLeft)*3*8/1000000.0*len(sprite[DICT_BBOXES]) > 15000 :
#         numBlocks = 3
#     else :
#         numBlocks = 1
    memNeededPerFrame = np.prod(bottomRight-topLeft)*3*8/1000000.0#*len(sprite[DICT_BBOXES])
    maxMemToUse = psutil.virtual_memory()[1]/1000000*0.5/2 ## use half of the available memory
    numFramesThatFit = np.round(maxMemToUse/memNeededPerFrame)
    numBlocks = int(np.ceil(numFrames/numFramesThatFit))
    
    frameIdxs = np.arange(numFrames)
    blockSize = int(np.ceil(numFrames/float(numBlocks)))
    frameLocs = np.sort(glob.glob(tmpFolder+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/frame-0*.png"))
    for i in xrange(numBlocks) :
        idxsToUse1 = frameIdxs[i*blockSize:(i+1)*blockSize]

        f1s = np.zeros(np.hstack([bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1], 3, len(idxsToUse1)]), dtype=float)
        numVisibile1 = np.zeros(blockSize, int)
        renderedBBoxes = np.zeros((f1s.shape[0], f1s.shape[1], blockSize), np.uint8)
        sortedKeys = np.sort(sprite[DICT_BBOXES].keys())[i*blockSize:i*blockSize+blockSize]
        for idx, frame in enumerate(frameLocs[idxsToUse1]) :
            img = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
            alpha = img[:, :, -1]/255.0
            f1s[:, :, :, idx] = ((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)) + 
                                 bgPatch*(1.0-np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))))
    #             numVisibile1[idx] = len(np.argwhere(alpha.flatten() == 1))
            img = np.zeros((f1s.shape[0], f1s.shape[1]), np.uint8)
            ## bbox coords are (x, y) but topLeft is (row, col)
            cv2.fillConvexPoly(img, sprite[DICT_BBOXES][sortedKeys[idx]].astype(int)[[0, 1, 2, 3, 0], :]-topLeft[::-1], 1)
            renderedBBoxes[:, :, idx] = img
            numVisibile1[idx] = len(np.argwhere(img.flatten() == 1))
        
        print "loaded frames, block i=", i; sys.stdout.flush()
        data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
        distMat[i*blockSize:i*blockSize+len(idxsToUse1), i*blockSize:i*blockSize+len(idxsToUse1)] = ssd(data1)
        print "computed distance", distMat[i*blockSize:i*blockSize+len(idxsToUse1), i*blockSize:i*blockSize+len(idxsToUse1)].shape; sys.stdout.flush()

        for j in xrange(i+1, numBlocks) :
            idxsToUse2 = frameIdxs[j*blockSize:(j+1)*blockSize]

            f2s = np.zeros(np.hstack([bottomRight[0]-topLeft[0], bottomRight[1]-topLeft[1], 3, len(idxsToUse2)]), dtype=float)
            for idx, frame in enumerate(frameLocs[idxsToUse2]) :
                img = np.array(Image.open(frame), dtype=float)[topLeft[0]:bottomRight[0], topLeft[1]:bottomRight[1], :]
                alpha = img[:, :, -1]/255.0
                f2s[:, :, :, idx] = ((img[:, :, :-1]/255.0)*np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)) + 
                                     bgPatch*(1.0-np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))))

            print "loaded frames, block j=", j; sys.stdout.flush()
            data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
            distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)] = ssd2(data1, data2)
            distMat[j*blockSize:j*blockSize+len(idxsToUse2), i*blockSize:i*blockSize+len(idxsToUse1)] = distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)].T
            print "computed distance", distMat[i*blockSize:i*blockSize+len(idxsToUse1), j*blockSize:j*blockSize+len(idxsToUse2)].shape; sys.stdout.flush()

            del f2s, data2

        del f1s, data1
        
        
    ## due to imprecision I need to make this check
    distMat[distMat > 0.0] = np.sqrt(distMat[distMat > 0.0])
    distMat[distMat <= 0.0] = 0.0

    shutil.rmtree(tmpFolder+sprite[DICT_SEQUENCE_NAME]+"-maskedFlow/")
    
#     np.save(dataPath+dataSet+sprite[DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy", distMat/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize))))

    
    print "done", sprite[DICT_SEQUENCE_NAME], "in", time.time() - t; sys.stdout.flush()
    print
    print

# <codecell>

memNeededPerFrame = np.prod(bottomRight-topLeft)*3*8/1000000.0#*len(sprite[DICT_BBOXES])
maxMemToUse = 1000/2 #psutil.virtual_memory()[1]/1000000*0.5/2 ## use half of the available memory
numFramesThatFit = np.round(maxMemToUse/memNeededPerFrame)
numBlocks = int(np.ceil(numFrames/numFramesThatFit))
print memNeededPerFrame, maxMemToUse
print numFramesThatFit
print numBlocks
print numFrames/numBlocks

# <codecell>

# oneBlockDistMat = np.copy(distMat)
print distMat.shape, oneBlockDistMat.shape
print np.max(np.abs(distMat-oneBlockDistMat))
gwv.showCustomGraph(np.abs(distMat-oneBlockDistMat))
gwv.showCustomGraph(distMat)
gwv.showCustomGraph(oneBlockDistMat)

# <codecell>

gwv.showCustomGraph(distMat)
gwv.showCustomGraph(distMat/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize))))
# gwv.showCustomGraph(distMat/(dotOverlap+1.0))
gwv.showCustomGraph(distMat/((2.0*dotOverlap)/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize)))+0.01))
# gwv.showCustomGraph(distMat/(cv2.filter2D(2.0*dotOverlap, -1, np.eye(4*2+1))/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize)))+0.01))
# print numVisibile1.shape

# <codecell>

gwv.showCustomGraph(np.load("/home/ilisescu/PhD/data/havana/bordeaux_car1-vanilla_distMat.npy"))

# <codecell>

tmp = np.copy(renderedBBoxes.reshape((np.prod(renderedBBoxes.shape[0:2]), renderedBBoxes.shape[2])).T).astype(float)
t = time.time()
dotOverlap = np.dot(tmp, tmp.T)
print time.time()-t

# <codecell>

print dataPath+dataSet+sprite[DICT_SEQUENCE_NAME]+"-new_overlap_norm_distMat.npy"
np.save(dataPath+dataSet+sprite[DICT_SEQUENCE_NAME]+"-new_overlap_norm_distMat.npy",
        distMat/((2.0*dotOverlap)/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize)))+0.01))

# <codecell>

gwv.showCustomGraph(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize)))
gwv.showCustomGraph(dotOverlap)
gwv.showCustomGraph(distMat)
gwv.showCustomGraph(distMat/(numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize))))

# <codecell>

print dataPath+dataSet+sprite[DICT_SEQUENCE_NAME]+"-num_overlapping_pixels.npy"
# np.save(dataPath+dataSet+sprite[DICT_SEQUENCE_NAME]+"-num_overlapping_pixels.npy", overlappingPixels)

# <codecell>

# np.save("white_car1-ditMat-overlapp_normalization.npy", distMat/(overlappingPixels+1.0))
gwv.showCustomGraph(np.load("white_car1-ditMat-overlapp_normalization.npy"))

# <codecell>

# gwv.showCustomGraph(renderedBBoxes[:, :, 1])
startTime = time.time()
overlappingPixels = np.zeros((renderedBBoxes.shape[-1], renderedBBoxes.shape[-1]))
for i in arange(renderedBBoxes.shape[-1]) :
    t = time.time()
    for j in xrange(i+1, renderedBBoxes.shape[-1]) :
        overlappingPixels[i, j] = len(np.argwhere(renderedBBoxes[:, :, i] & renderedBBoxes[:, :, j]))
        overlappingPixels[j, i] = overlappingPixels[i, j]
#     print "done", i, "in", time.time() - t
    sys.stdout.write('\r' + "Done " + np.string_(i) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()
        
overlappingPixels[arange(len(overlappingPixels)), arange(len(overlappingPixels))] = numVisibile1
print 
print time.time() - startTime
print renderedBBoxes.shape[-1]

# <codecell>

print renderedBBoxes.shape
print tmp.shape
# gwv.showCustomGraph(renderedBBoxes[:, :, 400])
print 560*853

# <codecell>

t = time.time()
np.dot(tmp[0, :], tmp.T)
print time.time() - t

# <codecell>

tmp = np.copy(renderedBBoxes.reshape((np.prod(renderedBBoxes.shape[0:2]), renderedBBoxes.shape[2])).T).astype(float)
dotOverlap = np.dot(tmp, tmp.T)

# <codecell>

print tmp.shape

# <codecell>

# tmp = np.copy(renderedBBoxes.reshape((np.prod(renderedBBoxes.shape[0:2]), renderedBBoxes.shape[2])).T)
t = time.time()
dotOverlap = np.dot(tmp, tmp.T)
print time.time()-t

# <codecell>

print dotOverlap.shape

# <codecell>

gwv.showCustomGraph(dotOverlap)
# gwv.showCustomGraph(overlappingPixels)

# <codecell>

from multiprocessing.pool import ThreadPool

def computePixelOverlap(ims1, ims2, i, j, threadIdx) :
    t = time.time()
#     time.sleep(1)#np.random.randint(2, 4))
    result = np.dot(ims1, ims2.T)
#     print "done thread", threadIdx
    return result, i, j#, [ims1.shape, ims2.shape, threadIdx]

numBlocks = 2
blockSize = int(np.ceil(tmp.shape[0]/float(numBlocks)))
numThreads = int(numBlocks*(numBlocks+1)/2)
print "need", numThreads, "threads"

pool = ThreadPool(processes=numThreads)

threadResults = []
offset = 0
for i in xrange(numBlocks) :
    offset += i
    for j in xrange(i, numBlocks) :
        threadResults.append(pool.apply_async(computePixelOverlap, (tmp[i*blockSize:i*blockSize+blockSize, :], 
                                                                    tmp[j*blockSize:j*blockSize+blockSize, :],
                                                                    i, j, i*numBlocks+j-offset)))
        
multiprocMat = np.zeros((tmp.shape[0], tmp.shape[0]))
t = time.time()
for res in threadResults : 
#     t.join()
#     i = int(n/numBlocks)
#     j = int(np.mod(n, numBlocks))+i
#     print i, j, i*blockSize,i*blockSize+blockSize, j*blockSize,j*blockSize+blockSize, multiprocMat[i*blockSize:i*blockSize+blockSize,
#                                                                                                    j*blockSize:j*blockSize+blockSize].shape
    blockResult, i, j = res.get()
    multiprocMat[i*blockSize:i*blockSize+blockSize,
                 j*blockSize:j*blockSize+blockSize] = np.copy(blockResult)
    multiprocMat[j*blockSize:j*blockSize+blockSize,
                 i*blockSize:i*blockSize+blockSize] = np.copy(blockResult.T)
#     print res.get()
print "Exiting Main Thread", time.time() -t

# <codecell>

gwv.showCustomGraph(multiprocMat)
gwv.showCustomGraph(overlappingPixels)

# <codecell>

print np.max(np.abs(multiprocMat-overlappingPixels))

# <codecell>

print tmp.shape, tmp.dtype, np.max(tmp), np.min(tmp)

# <codecell>

tmp = np.copy(overlappingPixels)

# <codecell>

gwv.showCustomGraph(overlappingPixels)

# <codecell>

gwv.showCustomGraph(np.load(dataPath+dataSet+trackedSprites[2][DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy"))

# <codecell>

gwv.showCustomGraph(np.load(dataPath+dataSet+"window_row2a-vanilla_distMat.npy"))
# print sprite[DICT_SEQUENCE_NAME]

# <codecell>

print sprite[DICT_BBOXES].keys()
print sprite[DICT_SEQUENCE_NAME]
print sprite[DICT_FRAMES_LOCATIONS]

# <codecell>

gwv.showCustomGraph(np.load(dataPath+dataSet+"black_car1-vanilla_distMat.npy"))

# <codecell>

print numVisibile1.reshape((blockSize, 1)) + numVisibile1.reshape((1, blockSize))
print distMat[0, 0]/n

# <codecell>

import VideoTexturesUtils as vtu
gwv.showCustomGraph(vtu.filterDistanceMatrix(distMat, 4, False))

