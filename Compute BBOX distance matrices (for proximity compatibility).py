# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import sys
import scipy as sp
from IPython.display import clear_output

import cv2
import time
import os
import scipy.io as sio
import glob
import itertools

from PIL import Image

import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import SemanticsDefinitionTabGUI as sdt
import opengm
import soundfile as sf

from matplotlib.patches import Rectangle

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise


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

GRAPH_MAX_COST = 10000000.0

dataPath = "/home/ilisescu/PhD/data/"

dataSet = "havana/"
# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
formatString = "{:05d}.png"

TL_IDX = 0
TR_IDX = 1
BR_IDX = 2
BL_IDX = 3

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
            dists = aabb2pointsDist(aabb, projPoints)#np.linalg.norm(projPoints-aabb, axis=-1)
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

def getSpritesBBoxDist(theta, bbox1, bbox2, verbose = False) :
    rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    bbox1 = np.dot(rotMat, bbox1.T).T
    bbox2 = np.dot(rotMat, bbox2.T).T
    ## if the bboxes coincide then the distance is set to 0
    if np.all(np.abs(bbox1 - bbox2) <= 10**-10) :
        return 0.0
    else :
        return aabb2obbDist(bbox1, bbox2, verbose)

# <codecell>

#### computes inter sprite compatibilities as bbox distances
# baseLoc = "/home/ilisescu/PhD/data/synthesisedSequences/newHavana/"
# synthSequence = np.load(baseLoc+"synthesised_sequence.npy").item()
# baseLoc = "/home/ilisescu/PhD/data/havana/"
baseLoc = "/media/ilisescu/Data1/PhD/data/theme_park_sunny/"

# for seq1Idx in xrange(len(synthSequence[DICT_USED_SEQUENCES])) :
#     for seq2Idx in xrange(seq1Idx, len(synthSequence[DICT_USED_SEQUENCES])) :
#         seq1 = np.load(synthSequence[DICT_USED_SEQUENCES][seq1Idx]).item()
#         seq2 = np.load(synthSequence[DICT_USED_SEQUENCES][seq2Idx]).item()

for s, seq1Loc in enumerate(np.sort(glob.glob(baseLoc+"semantic_sequence-*.npy"))) :
    for seq2Loc in np.sort(glob.glob(baseLoc+"semantic_sequence-*.npy"))[s:] :
        seq1 = np.load(seq1Loc).item()
        seq2 = np.load(seq2Loc).item()
        
#         print seq1[DICT_SEQUENCE_NAME], seq2[DICT_SEQUENCE_NAME]
        print "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy"
        
#         tmp = np.load(baseLoc + "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy")
#         print np.max(tmp[tmp!=10000.0])
#         if np.any(np.isnan(tmp)) :
#             print "oopsie"

        bboxDistance = np.zeros((len(seq1[DICT_BBOXES]), len(seq2[DICT_BBOXES])))
        avgTime = 0.0
        for i, iKey in enumerate(np.sort(seq1[DICT_BBOXES].keys())[0:]) :
            t = time.time()
            if iKey not in seq1[DICT_FRAMES_LOCATIONS] :
                bboxDistance[i, :] = 10000.0
            else :
                for j, jKey in enumerate(np.sort(seq2[DICT_BBOXES].keys())[0:]) :
                    if jKey not in seq2[DICT_FRAMES_LOCATIONS] :
                        bboxDistance[i, j] = 10000.0
                    else :
                        bboxDistance[i, j] = getSpritesBBoxDist(seq1[DICT_BBOX_ROTATIONS][iKey],
                                                                seq1[DICT_BBOXES][iKey],
                                                                seq2[DICT_BBOXES][jKey])

            avgTime = (avgTime*i + time.time()-t)/(i+1)
            remainingTime = avgTime*(len(seq1[DICT_BBOXES])-i-1)/60.0

            if np.mod(i, 5) == 0 :
                sys.stdout.write('\r' + "Done row " + np.string_(i) + " of " + np.string_(len(seq1[DICT_BBOXES])) +
                                 " (avg time: " + np.string_(avgTime) + " secs --- remaining: " +
                                 np.string_(int(np.floor(remainingTime))) + ":" + np.string_(int((remainingTime - np.floor(remainingTime))*60)) + ")")
                sys.stdout.flush()
        print        
        np.save(baseLoc + "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy", bboxDistance)

# <codecell>

for s, seq1Loc in enumerate(np.sort(glob.glob(baseLoc+"semantic_sequence-*.npy"))) :
    for seq2Loc in np.sort(glob.glob(baseLoc+"semantic_sequence-*.npy"))[s:] :
        seq1 = np.load(seq1Loc).item()
        seq2 = np.load(seq2Loc).item()
        
#         print seq1[DICT_SEQUENCE_NAME], seq2[DICT_SEQUENCE_NAME]
        print "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy"
        
        tmp = np.load(baseLoc + "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy")
#         print np.max(tmp[tmp!=10000.0])
#         if np.any(np.isnan(tmp)) :
#             print "oopsie"
        print tmp.shape, len(seq1[DICT_BBOXES]), len(seq2[DICT_BBOXES])
    
        if tmp.shape[0] < len(seq1[DICT_BBOXES]) :
            tmp = np.concatenate((tmp, np.ones((1, tmp.shape[1]))*10000.0), axis=0)
        
        if tmp.shape[1] < len(seq2[DICT_BBOXES]) :
            tmp = np.concatenate((tmp, np.ones((tmp.shape[0], 1))*10000.0), axis=1)
        print tmp.shape
        
        np.save(baseLoc + "inter_sequence_compatibility-bbox_dist-" + seq1[DICT_SEQUENCE_NAME] + "--" + seq2[DICT_SEQUENCE_NAME] + ".npy", tmp)

