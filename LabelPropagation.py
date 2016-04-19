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
import sys
import os
import re
from scipy import ndimage
from scipy import stats

from tsne import tsne

# from _emd import emd

import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import ComputeGridFeatures as cgf

# dataPath = "/home/ilisescu/PhD/data/"
dataPath = "/media/ilisescu/Data1/PhD/data/"

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

app = QtGui.QApplication(sys.argv)

# <codecell>

## read frames from sequence of images
# dataSet = "pendulum/"
# dataSet = "ribbon2/"
# dataSet = "flag_blender/"
# dataSet = "ribbon1_matted/"
# dataSet = "little_palm1_cropped/"
# dataSet = "ballAnimation/"
# dataSet = "eu_flag_ph_left/"
dataSet = "candle_wind/"
outputData = dataPath+dataSet

## Find pngs in sample data
frames = glob.glob(dataPath + dataSet + "frame-*.png")
mattes = glob.glob(dataPath + dataSet + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes)

# <codecell>

## compute features for image
blocksPerWidth = 16
blocksPerHeight = 16
subDivisions = blocksPerWidth*blocksPerHeight

## given block sizes and img sizes build indices representing each block
imageSize = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)).shape
stencils = cgf.stencil2D(blocksPerWidth, blocksPerHeight, imageSize)

features = cgf.histFgFeatures(stencils, subDivisions, frames, mattes)
figure(); imshow(features.T, interpolation='nearest')

# <codecell>

figure(); imshow(features[462, :].reshape((blocksPerHeight, blocksPerWidth)), interpolation='nearest')

# <codecell>

figure(); 
xlim(0, blocksPerHeight*blocksPerWidth);
bar(xrange(blocksPerHeight*blocksPerWidth), features[462, :]/np.sum(features[462, :]));

# <codecell>

print features[0, :]
print np.sum(features[0, :]/np.linalg.norm(features[0, :]))
print np.sum(features[0, :]/((1280/16)*(720/16)))
np.sum(features[0, :])/np.sum(features[99, :])

# <codecell>

sio.savemat("features.mat", {"features":features})

# <codecell>

hist2demdDistMat = np.array(sio.loadmat(dataPath + dataSet + "hist2demd_32x48_distMat.mat")['distMat'], dtype=float)
print hist2demdDistMat

# <codecell>

figure(); imshow(np.array(sio.loadmat(dataPath + dataSet + "hist2demd_32x48_distMat.mat")['distMat'], dtype=float), interpolation='nearest')

# <codecell>

def estimateFutureCost(alpha, p, distanceMatrixFilt, weights) :

    distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
    distMat = distMatFilt ** p
    
    last = np.copy(distMat)
    current = np.zeros(distMat.shape)
    
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
        iterations += 1
    
    print
    print 'finished in', iterations, 'iterations'
    
    return distMat

# <codecell>

optimizedDistMat = estimateFutureCost(0.999, 2.0, vtu.filterDistanceMatrix(hist2demdDistMat/np.max(hist2demdDistMat), 4, True), np.ones((1200, 1200)))

# <codecell>

np.save(dataPath + dataSet + "proc_hist2demd_32x48_distMat.npy", optimizedDistMat)

# <codecell>

figure(); imshow(vtu.filterDistanceMatrix(hist2demdDistMat/np.max(hist2demdDistMat), 4, True), interpolation='nearest')
figure(); imshow(optimizedDistMat, interpolation='nearest')
figure(); imshow(np.load(dataPath + dataSet + "proc_distMat.npy"), interpolation='nearest')

# <codecell>

tmp3248 = np.load(dataPath + dataSet + "proc_hist2demd_32x48_distMat.npy")
tmp1616 = np.load(dataPath + dataSet + "proc_hist2demd_16x16_distMat.npy")
figure(); imshow((np.abs(tmp3248/np.max(tmp3248)-tmp1616/np.max(tmp1616)))[500:600, 500:600], interpolation='nearest')
figure(); imshow((tmp3248/np.max(tmp3248))[500:600, 500:600], interpolation='nearest')
figure(); imshow((tmp1616/np.max(tmp1616))[500:600, 500:600], interpolation='nearest')
print np.max((np.abs(tmp3248/np.max(tmp3248)-tmp1616/np.max(tmp1616))))

# <codecell>

## load precomputed distance matrix and filter for label propagation
name = "vanilla_distMat"
# name = "histcos_16x16_distMat"
# name = "hist2demd_32x48_distMat"
# name = "hist2demd_16x16_distMat"
# name = "semantics_hog_rand50_distMat"
# name = "semantics_hog_rand50_encodedfirst_distMat"
# name = "semantics_hog_rand50_encodedlast_distMat"
# name = "semantics_hog_set50_distMat"
# name = "semantics_hog_set50_encodedlast_distMat"
# name = "semantics_hog_set150_distMat"
# name = "semantics_hog_L2_set150_distMat"
# name = "appearance_hog_rand150_distMat"
# name = "appearance_hog_L2_rand150_distMat"
# name = "appearance_hog_set150_distMat"
# name = "appearance_hog_L2_set150_distMat"
distanceMatrix = np.array(np.load(outputData + name + ".npy"), dtype=np.float)
distanceMatrix /= np.max(distanceMatrix)
filterSize = 2
if True :
    distMat = vtu.filterDistanceMatrix(distanceMatrix, filterSize, True)
else :
    distMat = np.copy(distanceMatrix)
figure(); imshow(distMat, interpolation='nearest')
## save for matlab to compute isomap
# sio.savemat(name + ".mat", {"distMat":distMat})
distances = np.array(np.copy(distMat), dtype=float)

# <codecell>

print distMat.shape
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

close('all')

# <codecell>

# distances = np.copy(distanceMatrix)
#/np.max(distMat)
# distances = np.copy(distMat[1:distMat.shape[1], 0:-1])
# distances = np.copy(distMatFut)

if False :
    ## use dotstar
    numClasses = 2
    ## init labeled points
    labelledPoints = np.array([[9], [21]])-1
    fl = np.zeros((len(labelledPoints), numClasses))
    fl = np.eye(numClasses)
else :
    ## use ribbon2
    numClasses = 4
    ## init labeled points
    labelledPoints = np.array([[122], [501], [838], [1106]]) -4
    fl = np.eye(numClasses)
    
#     labelledPoints = np.array([[22, 122, 222], [281, 501, 721], [754, 838, 922], [956, 1106, 1256]]) -4
#     fl = np.zeros((np.prod(labelledPoints.shape), numClasses))
#     fl[0:3, 0] = 1
#     fl[3:6, 1] = 1
#     fl[6:9, 2] = 1
#     fl[9:, 3] = 1

    initPoints = np.array([122, 501, 838, 1106]) -4
    extraPoints = 16
    labelledPoints = np.zeros((numClasses, extraPoints+1), dtype=np.int)
    for i in xrange(0, len(initPoints)) :
        labelledPoints[i, :] = range(initPoints[i]-extraPoints/2, initPoints[i]+extraPoints/2+1)

    fl = np.zeros((np.prod(labelledPoints.shape), numClasses))
    for i in xrange(0, numClasses) :
        fl[i*(extraPoints+1):(i+1)*(extraPoints+1), i] = 1
    
print numClasses, labelledPoints
print fl

## order w to have labeled nodes at the top-left corner
flatLabelled = np.ndarray.flatten(labelledPoints)

# <codecell>

## code to make a sprite for a given dataset
dataSet = "toy/"
outputData = dataPath+dataSet

## Find pngs in sample data
frames = glob.glob(dataPath + dataSet + "frame-*.png")
mattes = glob.glob(dataPath + dataSet + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
h, w, _ = frameSize
print numFrames, len(mattes)

sprite = {DICT_SEQUENCE_NAME:"toy1",
          DICT_FRAMES_LOCATIONS:{},
          DICT_BBOXES:{},
          DICT_BBOX_CENTERS:{},
          DICT_BBOX_ROTATIONS:{},
          DICT_FOOTPRINTS:{},
          DICT_ICON_FRAME_KEY:int(24),
          DICT_ICON_TOP_LEFT:np.array([29, 307], dtype=int), ## (row, col)
          DICT_ICON_SIZE:int(640),
          DICT_REPRESENTATIVE_COLOR:np.array([153, 208, 54], dtype=int)}

for i in xrange(numFrames) :
    sprite[DICT_FRAMES_LOCATIONS][i] = frames[i]
    sprite[DICT_BBOXES][i] = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    sprite[DICT_BBOX_CENTERS][i] = np.array([w/2.0, h/2.0])
    sprite[DICT_BBOX_ROTATIONS][i] = 0.0
    sprite[DICT_FOOTPRINTS][i] = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    
np.save(outputData+"sprite-0000-"+sprite[DICT_SEQUENCE_NAME]+".npy", sprite)

# <codecell>

print outputData

# <codecell>

## read frames from sequence of images
# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"

dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "wave1/"
# dataSet = "wave2/"
# dataSet = "wave3/"
dataSet = "toy/"
# dataSet = "theme_park_sunny/"
# dataSet = "windows/"
# dataSet = "digger/"
# dataSet = "candle_wind/"
outputData = dataPath+dataSet

trackedSprites = []
# for i, sprite in enumerate(np.sort(glob.glob(dataPath + dataSet + "sprite*.npy"))[[0, 2]]) : ## this is for digger
for i, sprite in enumerate(np.sort(glob.glob(dataPath + dataSet + "sprite*.npy"))) : #[[0, 1, 5, 6, 9]]) :## this is for havana 
    trackedSprites.append(np.load(sprite).item())
    print i+1, trackedSprites[-1][DICT_SEQUENCE_NAME]

## Find pngs in sample data
frames = glob.glob(dataPath + dataSet + "frame-*.png")
mattes = glob.glob(dataPath + dataSet + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes)

 ## wave1
#  ## wave2
#  ## wave3
if dataSet == "wave1/" :
    print "using wave1"
    subsetExtremes = np.array([[1122, 2674], [1252, 2588], [1394, 2610], [1346, 2522], [1216, 2496], [1806, 2702], [1624, 2880], [1478, 2614]])
    ## wave1 ## full length sprites
    labelledFrames = [np.array([[1130, 2670, 3220],                         ## moos
                                [2220, 2527, 2793]]),
                      np.array([[865, 1780, 2960, 3510],                    ## peter
                                [2222, 2472, 2740, 3120]]),
                      np.array([[923, 2430, 3021],                          ## sara
                                [2183, 2519, 2842]]),
                      np.array([[1643, 1756, 2491, 3361],                   ## tara
                                [2115, 2438, 2626, 2815]]),
                      np.array([[601, 1375, 3159],                          ## james
                                [2151, 2415, 2694]]),
                      np.array([[390, 1383, 1881, 3219],                    ## aron
                                [2346, 2577, 2772, 2979]]),
                      np.array([[293, 1481, 1757],                          ## daniel
                                [2348, 2732, 3131]]),
                      np.array([[387, 875, 1338, 1770, 3520],               ## ferran
                                [2290, 2557, 2687, 2877, 3264]])]
    ## wave1 ## trimmed down sprites
    labelledFrames = [np.array([[1140, 2660],                               ## moos
                                [2220, 2527]])-subsetExtremes[0, 0],
                      np.array([[1450, 1780],                               ## peter
                                [2222, 2472]])-subsetExtremes[1, 0],
                      np.array([[1672, 2430],                               ## sara
                                [2183, 2519]])-subsetExtremes[2, 0],
                      np.array([[1643, 2491],                               ## tara
                                [2115, 2438]])-subsetExtremes[3, 0],
                      np.array([[1375, 1993],                               ## james
                                [2151, 2415]])-subsetExtremes[4, 0],
                      np.array([[1881, 2248],                               ## aron
                                [2346, 2577]])-subsetExtremes[5, 0],
                      np.array([[1757, 2500],                               ## daniel
                                [2348, 2732]])-subsetExtremes[6, 0],
                      np.array([[1770, 1947],                               ## ferran
                                [2290, 2557]])-subsetExtremes[7, 0]]
    
    for i, extremes in enumerate(subsetExtremes) :
        print i, extremes, np.diff(extremes), labelledFrames[i].flatten(),
        print np.all(np.array([labelledFrames[i].flatten() > 0, labelledFrames[i].flatten() < np.diff(extremes)]), axis=0)
    
    
elif dataSet == "wave2/" :
    print "using wave2"
    subsetExtremes = np.array([[682, 2002], [1194, 1938], [470, 1990], [896, 2088], [899, 2019], [1320, 2256], [546, 2050], [886, 2006]])
    ## wave2 ## full length sprites
    labelledFrames = [np.array([[50, 440, 617, 1733, 2540, 3075],           ## tara
                                [1505, 2095, 2350, 2680, 2845, 3301]]),
                      np.array([[380, 1080, 1950, 2303, 2845],              ## james
                                [1640, 1850, 2049, 2613, 3030]]),
                      np.array([[674, 1335, 1985, 2890],                    ## moos
                                [1565, 1875, 2186, 2513]]),
                      np.array([[40, 815, 1241, 2125, 2768, 3290],          ## aron
                                [1621, 1840, 1970, 2023, 2508, 2879]]),
                      np.array([[868, 1551, 1976, 2170, 2511],              ## peter
                                [1671, 1881, 2098, 2387, 2720]]),
                      np.array([[356, 749, 1249, 2781],                     ## daniel
                                [1742, 2143, 2559, 3048]]),
                      np.array([[1040, 1500, 1740, 2750, 3305],             ## sara
                                [1609, 1957, 2195, 2493, 2915]]),
                      np.array([[770, 1736, 2464, 2809, 3275],              ## ferran
                                [1618, 1888, 2191, 2601, 2930]])]
    
    
    ## wave2 ## trimmed down sprites
    labelledFrames = [np.array([[814, 1733],                                ## tara
                                [1505, 1942]])-subsetExtremes[0, 0],
                      np.array([[1268, 1346],                               ## james
                                [1640, 1850]])-subsetExtremes[1, 0],
                      np.array([[674, 1335],                                ## moos
                                [1565, 1875]])-subsetExtremes[2, 0],
                      np.array([[1241, 1358, 1719],                         ## aron
                                [1621, 1840, 1970]])-subsetExtremes[3, 0],
                      np.array([[1551, 1976],                               ## peter
                                [1671, 1881]])-subsetExtremes[4, 0],
                      np.array([[1434, 1892],                               ## daniel
                                [1742, 2143]])-subsetExtremes[5, 0],
                      np.array([[1040, 1500],                               ## sara
                                [1609, 1957]])-subsetExtremes[6, 0],
                      np.array([[950, 1736],                                ## ferran
                                [1618, 1888]])-subsetExtremes[7, 0]]
    
    
    for i, extremes in enumerate(subsetExtremes) :
        print i, extremes, np.diff(extremes), labelledFrames[i].flatten(),
        print np.all(np.array([labelledFrames[i].flatten() > 0, labelledFrames[i].flatten() < np.diff(extremes)]), axis=0)
        
elif dataSet == "wave3/" :
    print "using wave3"
    subsetExtremes = np.array([[868, 1844], [1060, 1708], [372, 1804], [832, 1984], [1173, 1893], [1020, 2100], [806, 1902], [906, 1906]])
    ## wave3 ## full length sprites
    labelledFrames = [np.array([[727, 1253, 1910, 2381, 2966],              ## peter
                                [1400, 1720, 2005, 2263, 2489]]),
                      np.array([[723, 1293, 1530, 2238, 2997],              ## tara
                                [1383, 1596, 1860, 2133, 2460]]),
                      np.array([[191, 749, 1400, 1590, 2028, 2666],         ## aron
                                [1511, 1727, 1901, 2189, 2477, 2822]]),
                      np.array([[604, 985, 1099, 2902],                     ## moos
                                [1495, 1870, 2209, 2536]]),
                      np.array([[144, 1050, 1326, 2898],                    ## james
                                [1569, 1728, 1998, 2505]]),
                      np.array([[366, 1297, 2525, 2739],                    ## daniel
                                [1643, 1973, 2405, 2915]]),
                      np.array([[294, 846, 2337, 2784],                     ## sara
                                [1495, 1807, 2122, 2452]]),
                      np.array([[288, 576, 708, 1044, 2838],                ## ferran
                                [1533, 1794, 2067, 2337, 2610]])]
    
    ## wave3 ## trimmed down sprites
    labelledFrames = [np.array([[977, 1253],                                ## peter
                                [1400, 1720]])-subsetExtremes[0, 0],
                      np.array([[1293, 1530],                               ## tara
                                [1383, 1596]])-subsetExtremes[1, 0],
                      np.array([[749, 1400],                                ## aron
                                [1511, 1727]])-subsetExtremes[2, 0],
                      np.array([[985, 1099],                                ## moos
                                [1495, 1870]])-subsetExtremes[3, 0],
                      np.array([[1198, 1326],                               ## james
                                [1569, 1728]])-subsetExtremes[4, 0],
                      np.array([[1297, 1767],                               ## daniel
                                [1643, 1973]])-subsetExtremes[5, 0],
                      np.array([[846, 1245],                                ## sara
                                [1495, 1807]])-subsetExtremes[6, 0],
                      np.array([[1044, 1384],                               ## ferran
                                [1533, 1794]])-subsetExtremes[7, 0]]
    
    
    for i, extremes in enumerate(subsetExtremes) :
        print i, extremes, np.diff(extremes), labelledFrames[i].flatten(),
        print np.all(np.array([labelledFrames[i].flatten() > 0, labelledFrames[i].flatten() < np.diff(extremes)]), axis=0)
        
# C, D, E, F, G, A, B, C
elif dataSet == "toy/" :
    print "using toy"
    ## toy
    labelledFrames = [np.array([[17, 344],                                  ## None
                                [84, 574],                                  ## C
                                [116, 543],                                 ## D
                                [148, 516],                                 ## E
                                [180, 487],                                 ## F
                                [212, 457],                                 ## G
                                [243, 431],                                 ## A
                                [273, 399],                                 ## B
                                [308, 369]])                                ## C
                      ]
# C, D, E, F, G, A, B, C
elif dataSet == "theme_park_sunny/" :
    print "using theme_park_sunny"
    ## toy
    labelledFrames = [np.array([[185, 256, 337, 407, 803, 1256],            ## Moving
                                [9, 39, 60, 2006, 2047, 2051]])             ## Not Moving
                      ]
    spriteSubset = np.array([2])
    trackedSprites = [trackedSprites[spriteSubset]]
    
elif dataSet == "windows/" :
    print "using windows" ## first label is off, second label is on
    labelledFrames = [np.array([[95],                                       ## 2a
                                [10]]),
                      np.array([[46],                                       ## 2c
                                [14]]),
                      np.array([[58],                                       ## 2d
                                [15]]),
                      np.array([[57],                                       ## 2e
                                [26]]),
                      np.array([[36],                                       ## 3b
                                [14]]),
                      np.array([[41],                                       ## 3c
                                [27]]),
                      np.array([[78],                                       ## 3d
                                [24]]),
                      np.array([[70],                                       ## 3e
                                [19]]),
                      np.array([[50],                                       ## 3f
                                [15]]),
                      np.array([[40],                                       ## 5b
                                [18]]),
                      np.array([[64],                                       ## 5c
                                [24]]),
                      np.array([[55],                                       ## 5e
                                [23]]),
                      np.array([[54],                                       ## 5f
                                [18]]),
                      np.array([[64],                                       ## 6a
                                [30]]),
                      np.array([[45],                                       ## 6f
                                [30]]),
                      np.array([[24],                                       ## 7a
                                [13]]),
                      np.array([[43],                                       ## 7d
                                [21]]),
                      np.array([[3],                                        ## 7f
                                [15]]),
                      np.array([[55],                                       ## 9a
                                [21]]),
                      np.array([[32],                                       ## 9d
                                [15]]),
                      np.array([[45],                                       ## 1b
                                [13]]),
                      np.array([[79],                                       ## 1c
                                [10]]),
                      np.array([[47],                                       ## 1e
                                [19]])]
elif dataSet == "digger/" :
    print "using digger"
#     labelledFrames = [np.array([[8, 17, 66],                                ## truck_right1 (moving, receiving dirt)
#                                 [33, 48, 54]]),
#                       np.array([[21, 82, 143, 200],                         ## digger_right1 (scooping, dropping)
#                                 [29, 94, 152, 215]])]
    labelledFrames = [[[13, 16, 19, 67],                                    ## truck_right1 (moving, receiving dirt)
                       [22, 30, 36, 42, 49, 55, 64]],
                      np.array([[21, 82, 143, 200],                         ## digger_right1 (scooping, dropping)
                                [29, 94, 152, 215]])]
elif dataSet == "havana/" :
    print "using havana"
    labelledFrames = [np.array([[8, 17, 45, 72],                            ## blue_car1
                                [110, 130, 174, 205],                            
                                [246, 251, 261, 267]]),
                               [[18, 53, 85, 134, 177, 201, 252, 285, 321, 384, 412, 467],                        ## blue_car2
                                [515, 528, 540, 553],                            
                                [573, 585, 603, 624]],
                      np.array([[15, 39, 53],                               ## pink_car1
                                [83, 101, 122],                            
                                [149, 206, 342]]),
                      np.array([[21, 57, 96, 268],                          ## red_car1
                                [326, 381, 424, 470],                            
                                [500, 510, 521, 542]]),
                               [[48, 148, 240, 320, 382, 419],                        ## white_car1
                                [498, 525, 550, 612, 650, 727, 781],                            
                                [811, 820, 830, 838]]]
elif dataSet == "candle_wind/" :
    print "using candle_wind"
    labelledFrames = [np.array([[16, 290, 580, 880, 1150, 1160],                                ## center
                                [72, 127, 201, 699, 733, 800],                                  ## right 
                                [392, 413, 501, 1007, 1051, 1079]])-4                           ## left
                      ]
    labelledFrames = [np.array([[290, 880],                                ## center
                                [127, 733],                                  ## right 
                                [413, 1051]])-4                           ## left
                      ]
else :
    try :
        del labelledFrames
        print "no labelledFrames"
    except :
        print "no labelledFrames"
        
#### candle_wind
# initPoints = np.array([[16, 290, 580, 880, 1150, 1160],
#                        [72, 127, 201, 699, 733, 800],
#                        [392, 413, 501, 1007, 1051, 1079]])-filterSize

# <codecell>

## load precomputed distance matrix and filter for label propagation
for spriteIdx in arange(len(trackedSprites)) :
# for spriteIdx in xrange(1) :
    
    ########## loading the distance matrix and filtering ###########
    
#     print "loading", "candle_wind1"
#     name = "candle_wind1-vanilla_distMat"
    print "loading", trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]
#     name = "4Sprites_distmats_normalized_overlap/"+trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]+"-vanilla_distMat"
#     name = "4sprites_BAK_visible-invisible-labels/"+trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]+"-vanilla_distMat"
#     distanceMatrix = np.array(np.load(outputData + name + ".npy"), dtype=np.float)
    
#     distanceMatrix[distanceMatrix <= 0.0] = 0.0
#     np.save(outputData + name + ".npy", distanceMatrix)
    
    
#     distanceMatrix = np.array(np.load("blue_car2-ditMat-overlapp_normalization.npy"), dtype=np.float)
#     distanceMatrix = np.array(np.load("pink_car1-ditMat-overlapp_normalization.npy"), dtype=np.float)
#     distanceMatrix = np.array(np.load("red_car1-ditMat-overlapp_normalization.npy"), dtype=np.float)
#     distanceMatrix = np.array(np.load("white_car1-ditMat-overlapp_normalization.npy"), dtype=np.float)
#     distanceMatrix = np.array(np.load(outputData+trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]+"-vanilla_distMat-norm_num_overlapping_pixels.npy"))
    distanceMatrix = np.array(np.load(outputData+trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]+"-vanilla_distMat.npy"))
    
    
    
    distanceMatrix /= np.max(distanceMatrix)
    filterSize = 4
    if True :
#         distMat = vtu.filterDistanceMatrix(distanceMatrix, filterSize, True)
        
        if False :
            coeff = special.binom(filterSize*2, range(0, filterSize*2 +1))
            kernel = np.eye(len(coeff))
            kernel = kernel*coeff/np.sum(coeff)
        else :
            kernel = np.eye(filterSize*2+1)

        distMat = cv2.filter2D(distanceMatrix, -1, kernel)
    else :
        distMat = np.copy(distanceMatrix)
#     gwv.showCustomGraph(distMat)

    distances = np.array(np.copy(distMat), dtype=float)
    
    ########## initializing the labelled frames #########
    
#     sprite = "candle_wind" 
    sprite = trackedSprites[spriteIdx][DICT_SEQUENCE_NAME]
    initPoints = labelledFrames[spriteIdx]#-filterSize
#     print initPoints
    
    print initPoints
    numClasses = len(initPoints)
    print numClasses
    extraPoints = 4
    labelledPoints = np.empty(0, dtype=np.int)
    numClassExamples = []
    for i in xrange(numClasses) :
        for j in xrange(len(initPoints[i])) :
            labelledPoints = np.concatenate((labelledPoints, range(initPoints[i][j]-extraPoints/2, initPoints[i][j]+extraPoints/2+1)))
        numClassExamples.append(len(initPoints[i])*(extraPoints+1))
    numClassExamples = np.array(numClassExamples)
    print labelledPoints, labelledPoints.shape, np.sum(np.array(numClassExamples))
    
    print numClassExamples
    print
    fl = np.zeros((len(labelledPoints), numClasses))
    for i in xrange(0, numClasses) :
        print i, np.sum(numClassExamples[:i]), np.sum(numClassExamples[:i+1])
        fl[np.sum(numClassExamples[:i]):np.sum(numClassExamples[:i+1]), i] = 1
    print
#     numClasses = initPoints.shape[0]

#     extraPoints = 4
#     labelledPoints = np.zeros((numClasses, initPoints.shape[1]*(extraPoints+1)), dtype=np.int)
#     for i in xrange(numClasses) :
#         for j in xrange(initPoints.shape[1]) :
#             labelledPoints[i, j*(extraPoints+1):(j+1)*(extraPoints+1)] = range(initPoints[i, j]-extraPoints/2, initPoints[i, j]+extraPoints/2+1)

#     fl = np.zeros((np.prod(labelledPoints.shape), numClasses))
#     for i in xrange(0, numClasses) :
#         fl[i*(initPoints.shape[1]*(extraPoints+1)):(i+1)*(initPoints.shape[1]*(extraPoints+1)), i] = 1

    print numClasses, labelledPoints
#     print fl

    ## order w to have labeled nodes at the top-left corner
    flatLabelled = np.ndarray.flatten(labelledPoints)
    
    ######### do label propagation as zhu 2003 #########
    
    orderedDist = np.copy(distances)
    sortedFlatLabelled = flatLabelled[np.argsort(flatLabelled)]
    sortedFl = fl[np.argsort(flatLabelled), :]
    print sortedFlatLabelled
    for i in xrange(0, len(sortedFlatLabelled)) :
        #shift sortedFlatLabelled[i]-th row up to i-th row and adapt remaining rows
        tmp = np.copy(orderedDist)
        orderedDist[i, :] = tmp[sortedFlatLabelled[i], :]
        orderedDist[i+1:, :] = np.vstack((tmp[i:sortedFlatLabelled[i], :], tmp[sortedFlatLabelled[i]+1:, :]))
        #shift sortedFlatLabelled[i]-th column left to i-th column and adapt remaining columns
        tmp = np.copy(orderedDist)
        orderedDist[:, i] = tmp[:, sortedFlatLabelled[i]]
        orderedDist[:, i+1:] = np.hstack((tmp[:, i:sortedFlatLabelled[i]], tmp[:, sortedFlatLabelled[i]+1:]))
    #     print len(sortedFlatLabelled)+sortedFlatLabelled[i]

    # gwv.showCustomGraph(distances)
    # gwv.showCustomGraph(orderedDist)

    ## compute weights
    w, cumW = vtu.getProbabilities(orderedDist, 0.06, None, False)
    # gwv.showCustomGraph(w)
    # gwv.showCustomGraph(cumW)

    l = len(sortedFlatLabelled)
    n = orderedDist.shape[0]
    ## compute graph laplacian
    L = np.diag(np.sum(w, axis=0)) - w
    # gwv.showCustomGraph(L)

    ## propagate labels
    fu = np.dot(np.dot(-np.linalg.inv(L[l:, l:]), L[l:, 0:l]), sortedFl)

    ## use class mass normalization to normalize label probabilities
    q = np.sum(sortedFl)+1
    fu_CMN = fu*(np.ones(fu.shape)*(q/np.sum(fu)))
    
    
    ########## get label probabilities and plot ##########
    
    ## add labeled points to propagated labels (as labelProbs) and plot
    print fu.shape
    # print fu_CMN

    print flatLabelled
    numClasses = fl.shape[-1]
    
    labelProbs = np.copy(np.array(fu))
    print labelProbs.shape
    for frame, i in zip(sortedFlatLabelled, np.arange(len(sortedFlatLabelled))) :
        labelProbs = np.vstack((labelProbs[0:frame, :], sortedFl[i, :], labelProbs[frame:, :]))
        
    print labelProbs.shape
    
    if True :
        fig1 = figure()
        clrs = np.arange(0.0, 1.0+1.0/(len(initPoints)-1), 1.0/(len(initPoints)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
        stackplot(np.arange(len(labelProbs)), np.row_stack(tuple([i for i in labelProbs.T])), colors=clrs)
#         for i in xrange(0, numClasses) :    
#             for node in labelledPoints[i] :
#                 figure(fig1.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])
                
    ########## save the semantic labels into the sprite ##########
    
    ## add labels for the first and last filterSize frames by copying the first and last labels
#     finalLabels = np.concatenate((labelProbs[0, :].reshape((1, numClasses)).repeat(filterSize, axis=0),
#                                   labelProbs,
#                                   labelProbs[-1, :].reshape((1, numClasses)).repeat(filterSize, axis=0)))
    finalLabels = np.copy(labelProbs)
    print "READ ME", finalLabels.shape, distances.shape
#     for spriteLoc in glob.glob(dataPath+dataSet+"sprite-*.npy") :
#         if sprite in spriteLoc :
#             print spriteLoc
#             spriteDict = np.load(spriteLoc).item()
#             spriteDict["semantics_per_frame"] = finalLabels
# #             print spriteDict.keys()
#             np.save(spriteLoc, spriteDict)

# <codecell>

        fig1 = figure()
        clrs = ['r', 'g', 'b'] #np.arange(0.0, 1.0+1.0/(len(initPoints)-1), 1.0/(len(initPoints)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
        stackplot(np.arange(len(labelProbs)), np.row_stack(tuple([i for i in labelProbs.T])), colors=clrs)

# <codecell>

print finalLabels[:, 0]

# <codecell>

sprite = trackedSprites[spriteIdx]
numVisibile1 = np.zeros(len(trackedSprites[spriteIdx][DICT_BBOXES]), int)
renderedBBoxes = np.zeros((frameSize[0], frameSize[1], len(trackedSprites[spriteIdx][DICT_BBOXES])), np.uint8)
sortedKeys = np.sort(sprite[DICT_BBOXES].keys())
for idx in xrange(len(sortedKeys)) :
    img = np.zeros((frameSize[0], frameSize[1]), np.uint8)
    ## bbox coords are (x, y) but topLeft is (row, col)
    cv2.fillConvexPoly(img, sprite[DICT_BBOXES][sortedKeys[idx]].astype(int)[[0, 1, 2, 3, 0], :], 1)
    renderedBBoxes[:, :, idx] = img
    numVisibile1[idx] = len(np.argwhere(img.flatten() == 1))
    
t = time.time()
overlappingPixels = np.zeros((renderedBBoxes.shape[-1], renderedBBoxes.shape[-1]))
for i in xrange(renderedBBoxes.shape[-1]) :
    t = time.time()
    for j in xrange(i+1, renderedBBoxes.shape[-1]) :
        overlappingPixels[i, j] = len(np.argwhere(renderedBBoxes[:, :, i] & renderedBBoxes[:, :, j]))
        overlappingPixels[j, i] = overlappingPixels[i, j]
#     print "done", i, "in", time.time() - t
    sys.stdout.write('\r' + "Done " + np.string_(i) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()
        
overlappingPixels[arange(len(overlappingPixels)), arange(len(overlappingPixels))] = numVisibile1
print time.time() - t
print renderedBBoxes.shape[-1]

# <codecell>

origDistMat = np.array(np.load(outputData + name + ".npy"), dtype=np.float)*(numVisibile1.reshape((len(trackedSprites[spriteIdx][DICT_BBOXES]), 1)) +
                                                                             numVisibile1.reshape((1, len(trackedSprites[spriteIdx][DICT_BBOXES]))))
gwv.showCustomGraph(origDistMat)
gwv.showCustomGraph(np.array(np.load(outputData + name + ".npy"), dtype=np.float))
gwv.showCustomGraph(origDistMat/(overlappingPixels+1.0))

# <codecell>

print outputData+sprite[DICT_SEQUENCE_NAME]+"-num_overlapping_pixels.npy"
print outputData+sprite[DICT_SEQUENCE_NAME]+"-vanilla_distMat-norm_num_overlapping_pixels.npy"
np.save(outputData+sprite[DICT_SEQUENCE_NAME]+"-num_overlapping_pixels.npy", overlappingPixels)
np.save(outputData+sprite[DICT_SEQUENCE_NAME]+"-vanilla_distMat-norm_num_overlapping_pixels.npy", origDistMat/(overlappingPixels+1.0))

# <codecell>


if False :
################ this bit doesn't work anymore since using #################
    ## add labeled frames to fu and plot
    labelProbs = np.array(fu[0:flatLabelled[0]])
    print labelProbs.shape
    for i in xrange(1, len(flatLabelled)) :
    #     print flatLabelled[i]+i, flatLabelled[i+1]-i
    #     print fu[flatLabelled[i]+i:flatLabelled[i+1]-i, :]
    
        labelProbs = np.vstack((labelProbs, fl[i-1, :]))
        print labelProbs.shape, flatLabelled[i-1]-(i-1), flatLabelled[i]-i
        labelProbs = np.vstack((labelProbs, fu[flatLabelled[i-1]-(i-1):flatLabelled[i]-i, :]))
        print labelProbs.shape
        
    
    
    labelProbs = np.vstack((labelProbs, fl[-1, :]))
    labelProbs = np.vstack((labelProbs, fu[flatLabelled[-1]-len(flatLabelled)+1:, :]))
    # labelProbs = labelProbs[1:, :]
    print labelProbs, labelProbs.shape
else :
    labelProbs = np.copy(np.array(fu))
    print labelProbs.shape
    for frame, i in zip(sortedFlatLabelled, np.arange(len(sortedFlatLabelled))) :
        labelProbs = np.vstack((labelProbs[0:frame, :], sortedFl[i, :], labelProbs[frame:, :]))
        
    print labelProbs.shape
    
    
fig1 = figure()
clrs = ['r', 'g', 'b', 'm', 'c']
stackplot(np.arange(len(labelProbs)), np.row_stack(tuple([i for i in labelProbs.T])), colors=clrs)
for i in xrange(0, numClasses) :    
    for node in labelledPoints[i] :
        figure(fig1.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])

# clrs = ['r', 'g', 'b', 'm']
# fig1 = figure()
# xlabel('all points')
# fig2 = figure()
# xlabel('only unlabeled')
# fig3 = figure()
# xlabel('only unlabeled + CMN')

# for i in xrange(0, numClasses) :
#     figure(fig1.number); plot(labelProbs[:, i], clrs[i])
#     figure(fig2.number); plot(fu[:, i], clrs[i])
#     figure(fig3.number); plot(fu_CMN[:, i], clrs[i])
    
#     for node in labelledPoints[i] :
#         figure(fig1.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])
#         figure(fig2.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])
#         figure(fig3.number); plot(np.repeat(node, 2), [0, np.max(fu_CMN)], clrs[i])

# <codecell>

## add labels for the first and last filterSize frames by copying the first and last labels
finalLabels = np.concatenate((labelProbs[0, :].reshape((1, numClasses)).repeat(filterSize, axis=0),
                              labelProbs,
                              labelProbs[-1, :].reshape((1, numClasses)).repeat(filterSize, axis=0)))
print finalLabels
### save as a semantic sequence
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


h, w, _ = frameSize
print frames.shape ## frame loc
print np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]) ## bbox
print np.array([w/2.0, h/2.0]) ## bbox center
print 0.0 ## bbox rotation
print np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]) ## footprint
print dataPath+dataSet

semanticSequence = {DICT_SEQUENCE_NAME:"candle_wind1",
                    DICT_SEQUENCE_LOCATION:dataPath+dataSet+"semantic_sequence-candle_wind1.npy",
                    DICT_MASK_LOCATION:dataPath+dataSet,
                    DICT_FRAME_SEMANTICS:finalLabels,
                    DICT_FRAMES_LOCATIONS:{},
                    DICT_BBOXES:{},
                    DICT_BBOX_CENTERS:{},
                    DICT_BBOX_ROTATIONS:{},
                    DICT_FOOTPRINTS:{},
                    DICT_ICON_FRAME_KEY:int(0),
                    DICT_ICON_TOP_LEFT:np.array([220, 310], dtype=int),
                    DICT_ICON_SIZE:int(500),
                    DICT_REPRESENTATIVE_COLOR:np.array([255, 225, 162], dtype=int)}

for i in xrange(numFrames) :
    semanticSequence[DICT_FRAMES_LOCATIONS][i] = frames[i]
    semanticSequence[DICT_BBOXES][i] = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    semanticSequence[DICT_BBOX_CENTERS][i] = np.array([w/2.0, h/2.0])
    semanticSequence[DICT_BBOX_ROTATIONS][i] = 0.0
    semanticSequence[DICT_FOOTPRINTS][i] = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
np.save(semanticSequence[DICT_SEQUENCE_LOCATION], semanticSequence)

# <codecell>

## save label propagation for visualizing within videotextgui
# np.save(outputData + "labeledPoints.npy", labeledPoints)
# np.save(outputData + "labelProbs.npy", labelProbs)
np.save(outputData + "l2dist_guiex60_mult0.045_labels.npy", {"labeledPoints": np.array(labelledPoints), "labelProbs": labelProbs})
## save label propagation for visualizing isomaps in matlab
# sio.savemat("labeledPoints.mat", {"labeledPoints":labeledPoints})
# sio.savemat("l2dist_mult0.05_labelProbs.mat", {"labelProbs":labelProbs})

# <codecell>

## save label propagation for visualizing within videotextgui
# np.save(outputData + "labeledPoints.npy", labeledPoints)
# np.save(outputData + "labelProbs.npy", labelProbs)
np.save(outputData + "learned_appereance_hog_set150_mult0.06_labels.npy", {"labeledPoints": np.array(labelledPoints), "labelProbs": labelProbs})
## save label propagation for visualizing isomaps in matlab
# sio.savemat("labeledPoints.mat", {"labeledPoints":labeledPoints})
# sio.savemat("l2dist_mult0.05_labelProbs.mat", {"labelProbs":labelProbs})

# <codecell>

## save label propagation for visualizing within videotextgui
# np.save(outputData + "labeledPoints.npy", labeledPoints)
# np.save(outputData + "labelProbs.npy", labelProbs)
np.save(outputData + "l2dist_mult0.05_labels.npy", {"labeledPoints": labeledPoints, "labelProbs": labelProbs})
## save label propagation for visualizing isomaps in matlab
sio.savemat("labeledPoints.mat", {"labeledPoints":labeledPoints})
sio.savemat("l2dist_mult0.05_labelProbs.mat", {"labelProbs":labelProbs})

# <codecell>

## load isomap from .mat and save as .npy
## it automatically loads the output from PhD/MATLAB so run this after running ComputeIsomap.m
mapPoints = sio.loadmat("../MATLAB/mapPoints.mat")["mapPoints"]
tmp = sio.loadmat("../MATLAB/predictedLabels.mat")["predictedLabels"]
predictedLabels = []
for i in xrange(0,len(np.ndarray.flatten(tmp)[0])) :
    predictedLabels.append(np.ndarray.flatten(np.ndarray.flatten(tmp)[0][i])-1)

np.save(outputData + "l2dist_mult0.05_isomap.npy", {"mapPoints": mapPoints, "predictedLabels": predictedLabels})

# <codecell>

## check that orderedDist is still symmetric
print np.sum(orderedDist[4, :] - orderedDist[:, 4])
print np.sum(orderedDist[10, :] - orderedDist[:, 10])
print np.sum(orderedDist[250, :] - orderedDist[:, 250])
## check that orderedDist has been ordered the right way
print orderedDist[0, 0:50]
print distances[117, list(flatLabelled)]
print distances[117, 0:50]
print 
print orderedDist[3, 0:50]
print distances[496, list(flatLabelled)]
print distances[496, 0:50]
print 
print orderedDist[10, 0:50]
print distances[1102, list(flatLabelled)]
print distances[1102, 0:50]

# <codecell>

## compute tsne representation for given distMat

Y = tsne(np.zeros((distMat.shape[0], 10)), distMat)

# <codecell>

## show result of tsne
print labelProbs.shape
labels = np.argmax(labelProbs, axis=-1)
reds = np.argwhere(labels == 0)
greens = np.argwhere(labels == 1)
blues = np.argwhere(labels == 2)
magentas = np.argwhere(labels == 3)
# labels = np.loadtxt("mnist2500_labels.txt");
print labels, labels.shape
print reds.shape, greens.shape, blues.shape, magentas.shape

## normalize and fit to interval [0, 1]
Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))

figure();
scatter(Y[reds,0], Y[reds,1], 20, '#ff0000');
scatter(Y[greens,0], Y[greens,1], 20, '#00ff00');
scatter(Y[blues,0], Y[blues,1], 20, '#0000ff');
scatter(Y[magentas,0], Y[magentas,1], 20, '#ff00ff');

## save results to use with videotextgui
np.save(outputData + "l2dist_mult0.05_tsnemap.npy", {"mapPoints": Y.T, "predictedLabels": [np.ndarray.flatten(reds), np.ndarray.flatten(greens), np.ndarray.flatten(blues), np.ndarray.flatten(magentas)]})

# <codecell>

## show stencil
im = np.zeros(img.shape)
im[stencils[5][0], stencils[5][1]] = 1
figure(); imshow(im)

# <codecell>

## show features
gwv.showCustomGraph(features[717, :].reshape((blocksPerDim, blocksPerDim)))
gwv.showCustomGraph(features[1165, :].reshape((blocksPerDim, blocksPerDim)))
gwv.showCustomGraph(features[1166, :].reshape((blocksPerDim, blocksPerDim)))
print np.dot(features[717, :], features[1166, :])
print np.dot(features[1165, :], features[1166, :])

# <codecell>

## compute the distance matrix where distance is dot product between feature vectors
distanceMatrix = np.ones((numFrames, numFrames))

for r in xrange(0, numFrames) :
    for c in xrange(r+1, numFrames) :
        distanceMatrix[r, c] = distanceMatrix[c, r] = np.dot(features[r, :], features[c, :])
    print r, 

distanceMatrix = 1 - distanceMatrix
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

## compute 1D emd
def distance(f1, f2):
#     return np.sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    return np.sqrt((f1 - f2)**2)#np.sqrt( (f1[0] - f2[0])**2  + (f1[1] - f2[1])**2 + (f1[2] - f2[2])**2 )

print emd((list(features[1165, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance)
print emd((list(features[717, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance)

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
        
        self.setWindowTitle("Semantic Labelling")
        self.resize(1280, 720)
        
        self.labelledFrames = []
#         self.labelledPairs.append([1234, 234, 0.5])
#         self.setLabelledFramesListTable()

        self.RANDOM_CHOICE = 0
        self.SET_CHOICE = 1
        self.choiceMode = self.RANDOM_CHOICE
        
        self.choiceSet = arange(100)
        
        self.getNewFrame()
        
        self.setFocus()
        
    def setLabelledFramesListTable(self) :
        
        if len(self.labelledFrames) > 0 :
            self.labelledFramesListTable.setRowCount(len(self.labelledFrames))
            
            for labelledFrame, i in zip(self.labelledFrames, arange(len(self.labelledFrames))) :
                self.labelledFramesListTable.setItem(i, 0, QtGui.QTableWidgetItem(np.string_(labelledFrame[0])))
                self.labelledFramesListTable.setItem(i, 1, QtGui.QTableWidgetItem(np.string_(labelledFrame[1])))
        else :
            self.labelledFramesListTable.setRowCount(0)
    
    def getNewFrame(self) :
        if self.choiceMode == self.RANDOM_CHOICE :
            self.getNewRandomFrame()
        elif self.choiceMode == self.SET_CHOICE :
            self.getNewFrameFromSet()
    
    def getNewFrameFromSet(self) :
        self.frameIdx = np.random.choice(self.choiceSet)
        
        while len(self.labelledFrames) > 0 and self.frameIdx in np.array(self.labelledFrames)[:, 0] :
            print "stuck"
            self.frameIdx = np.random.choice(self.choiceSet)
        
        self.setFrameImage()
            
    def getNewRandomFrame(self) :
        self.frameIdx = np.random.randint(0, len(frames))
        
        while len(self.labelledFrames) > 0 and self.frameIdx in np.array(self.labelledFrames)[:, 0] :
            print "stuck"
            self.frameIdx = np.random.randint(0, len(frames))
        
        self.setFrameImage()
        
    def setFrameImage(self) :
        ## HACK ##
        im = np.ascontiguousarray(Image.open(frames[self.frameIdx]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        self.frameInfo.setText(frames[self.frameIdx])
        
    def labelledFrameSelected(self) :
        selectedRow = self.labelledFramesListTable.currentRow()
        if selectedRow >= 0 and selectedRow < len(self.labelledFrames):
            ## HACK ##
            im = np.ascontiguousarray(Image.open(frames[self.labelledFrames[selectedRow][0]]))
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
            self.frameInfo.setText(frames[self.labelledFrames[selectedRow][0]])
        
    def keyPressEvent(self, e) :
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            self.setFrameLabel(np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9)))
        elif e.key() == QtCore.Qt.Key_Space : ## Get new frame
            self.getNewFrame()
            
    def setFrameLabel(self, label) :
        if self.labelledFramesListTable.currentRow() >= 0 : ## I'm modifying label of existing frame
            self.labelledFrames[self.labelledFramesListTable.currentRow()][1] = label
        else :
            self.labelledFrames.append([self.frameIdx, label])
            
        self.labelledFramesListTable.clearSelection()
        self.labelledFramesListTable.setCurrentCell(-1, -1)
        self.setLabelledFramesListTable()
        
        self.getNewFrame()
        
    def changeChoiceMode(self, index) :
        if index == self.RANDOM_CHOICE :
            self.choiceMode = self.RANDOM_CHOICE
            self.choiceSetInterval.setEnabled(False)
            self.choiceSetInterval.setVisible(False)
        elif index == self.SET_CHOICE :
            self.choiceMode = self.SET_CHOICE
            self.choiceSetInterval.setEnabled(True)
            self.choiceSetInterval.setVisible(True)
        
        self.getNewFrame()
        
    def changeChoiceSet(self) :
        choiceSetText = self.choiceSetInterval.text()
        interval = np.array(re.split("-|:", choiceSetText), dtype=int)
        if len(interval) == 2 :
            self.choiceSet = arange(interval[0], interval[1]+1)
            
        self.setFocus()
        self.getNewFrame()
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.labelledFramesListTable = QtGui.QTableWidget(0, 2)
        self.labelledFramesListTable.horizontalHeader().setStretchLastSection(True)
        self.labelledFramesListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Frame Index"))
        self.labelledFramesListTable.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Label"))
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
        
        self.choiceSetInterval = QtGui.QLineEdit()
        self.choiceSetInterval.setEnabled(False)
        self.choiceSetInterval.setVisible(False)
        
        
        ## SIGNALS ##
        
        self.labelledFramesListTable.cellPressed.connect(self.labelledFrameSelected)
        self.choiceModeComboBox.currentIndexChanged[int].connect(self.changeChoiceMode)
        self.choiceSetInterval.returnPressed.connect(self.changeChoiceSet)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.labelledFramesListTable)
        controlsLayout.addWidget(self.choiceModeComboBox)
        controlsLayout.addWidget(self.choiceSetInterval)
        frameLayout = QtGui.QVBoxLayout()
        frameLayout.addWidget(self.frameLabel)
        frameLayout.addWidget(self.frameInfo)
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

# labelledFramesRand = np.copy(labelledFrames)
labelledFramesSet = np.copy(labelledFrames)

# <codecell>

labelledFrames = np.copy(labelledFramesRand)

# <codecell>

# labelledFrames = np.array(window.labelledFrames)
print labelledFrames
# labelledFrames = labelledFrames[]
# print len(np.argwhere(labelledFrames[:, 1]==0))

# <codecell>

labelledFrames = np.array([[ 119,    0],
 [ 270,    0],
 [ 178,    0],
 [  82,   0],
 [  46,    0],
 [ 243,    0],
 [ 256,    0],
 [  36,    1],
 [ 242,    0],
 [  30,    1],
 [  21,    1],
 [ 257,    0],
 [ 104,    0],
 [ 240,    0],
 [ 138,    0],
 [ 265,    0],
 [ 254,    0],
 [ 106,    0],
 [ 208,    0],
 [ 125,    0],
 [  69,    0],
 [ 102,    0],
 [ 249,    0],
 [ 720,    1],
 [ 494,    1],
 [ 430,    1],
 [ 335,    1],
 [ 692,    1],
 [ 440,    1],
 [ 357,    1],
 [ 616,    1],
 [ 587,    1],
 [ 287,    1],
 [ 688,    1],
 [ 437,    1],
 [ 753,    1],
 [ 358,    1],
 [ 482,    1],
 [ 671,    1],
 [ 644,    1],
 [ 887,    2],
 [ 879,    2],
 [ 930,    2],
 [ 837,    2],
 [ 871,    2],
 [ 856,    2],
 [ 866,    2],
 [ 902,    2],
 [ 890,    2],
 [ 772,    2],
 [ 888,    2],
 [ 759,    2],
 [ 891,    2],
 [ 765,    2],
 [ 835,    2],
 [ 933,    2],
 [ 872,    2],
 [ 807,    2],
 [ 915,    2],
 [ 905,    2],
 [1268,    3],
 [ 954,    3],
 [1212,    3],
 [1206,    3],
 [1045,    3],
 [1128,    3],
 [ 987,    3],
 [ 970,    3],
 [ 965,    3],
 [1124,    3],
 [1092,    3],
 [1049,    3],
 [ 960,    3],
 [ 946,    3],
 [1048,    3],
 [1227,    3],
 [1253,    3],
 [1233,    3],
 [1019,    3],
 [ 953,    3]])

# <codecell>

np.save(dataPath + dataSet + "semantic_labels_gui_set60.npy", labelledFrames)

# <codecell>

labelledFrames = np.array(window.labelledFrames)
# np.save(dataPath + dataSet + "semantic_labels_prop_rand50.npy", labelledFrames)

# <codecell>

## get flatLabelled and fl
# labelledFrames = np.array(window.labelledFrames)
# labelledFrames = np.load(dataPath + dataSet + "semantic_labels_gui_set60.npy")
usedLabels = np.unique(np.ndarray.flatten(labelledFrames[:, 1]))
fl = np.zeros((len(labelledFrames), len(usedLabels)))
prevIdx = 0
flatLabelled = np.empty(0, dtype=int)
labelledPoints = []
print "used labels", usedLabels
for i in usedLabels :
    iLabels = labelledFrames[np.ndarray.flatten(np.argwhere(labelledFrames[:, 1] == i)), :][:, 0]
    labelledPoints.append(iLabels)
    flatLabelled = np.concatenate((flatLabelled, iLabels))
    fl[prevIdx:prevIdx+len(iLabels), i] = 1
    prevIdx += len(iLabels)
    print i, iLabels
print flatLabelled
print fl

