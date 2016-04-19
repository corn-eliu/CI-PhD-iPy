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
from PySide import QtCore, QtGui

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

app = QtGui.QApplication(sys.argv)

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

def vectorisedMinusLogMultiNormal(dataPoints, means, var, normalized = True) :
    if (dataPoints.shape[1] != means.shape[1] or np.any(dataPoints.shape[1] != np.array(var.shape)) 
        or len(var.shape) != 2 or var.shape[0] != var.shape[1]) :
        raise Exception("Data shapes don't agree data(" + np.string_(dataPoints.shape) + ") mean(" + np.string_(means.shape) + 
                        ") var(" + np.string_(var.shape) + ")")
    
    D = float(dataPoints.shape[1])
    n = -0.5*np.log(np.linalg.det(var))-(D/2.0)*np.log(2.0*np.pi)
    
    ## this does 0.5*dot(dot(data-mean, varInv), data-mean)
    varInv = np.linalg.inv(var)
    dataMinusMean = dataPoints-means
    
    ps = []
    for i in xrange(int(D)) :
        ps.append(np.sum((dataMinusMean)*varInv[:, i], axis=-1))
    
    ps = np.array(ps).T
    
    ps = -0.5*np.sum(ps*(dataMinusMean), axis=-1)
    
    if normalized :
        return -(n+ps)
    else :
        return -ps
# s = time.time()
# vectorisedMinusLogMultiNormal(semanticDist.reshape((len(semanticDist), 1)), np.array([0.0]).reshape((1, 1)), np.array([0.0001]).reshape((1, 1)), True)
# print time.time() - s
# s = time.time()
# vectorisedMinusLogMultiNormal(semanticLabels, np.array(desiredLabel).reshape((1, 2)), np.eye(2)*0.0001, True)

def vectorisedMinusLogMultiNormalMultipleMeans(dataPoints, means, var, normalized = True) :
    D = float(dataPoints.shape[1])
    n = -0.5*np.log(np.linalg.det(var))-(D/2.0)*np.log(2.0*np.pi)

    ## this does 0.5*dot(dot(data-mean, varInv), data-mean)
    varInv = np.linalg.inv(var)
    dataMinusMean = dataPoints.reshape((1, len(dataPoints), dataPoints.shape[1]))-means.reshape((means.shape[0], 1, means.shape[1]))

    ps = np.zeros((means.shape[0], dataPoints.shape[0], int(D)))
    
    for i in xrange(int(D)) :
        ps[:, :, i] = np.sum(dataMinusMean*varInv[:, i], axis=-1)

    ps = -0.5*np.sum(ps*(dataMinusMean), axis=-1)
    
    if normalized :
        return -(n+ps)
    else :
        return -ps
    
def vectorisedMultiNormalMultipleMeans(dataPoints, means, var, normalized = True) :
    D = float(dataPoints.shape[1])
    n = (1/(np.power(2.0*np.pi, D/2.0)*np.sqrt(np.linalg.det(var))))

    ## this does 0.5*dot(dot(data-mean, varInv), data-mean)
    varInv = np.linalg.inv(var)
    dataMinusMean = dataPoints.reshape((1, len(dataPoints), dataPoints.shape[1]))-means.reshape((means.shape[0], 1, means.shape[1]))

    ps = np.zeros((means.shape[0], dataPoints.shape[0], int(D)))
    
    for i in xrange(int(D)) :
        ps[:, :, i] = np.sum(dataMinusMean*varInv[:, i], axis=-1)

    ps = np.exp(-0.5*np.sum(ps*(dataMinusMean), axis=-1))
    
    if normalized :
        return n*ps
    else :
        return ps

# <codecell>

compatibilityMats = {}
compatibilityMats["00"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--black_car1.npy")/50)*10.0
compatibilityMats["01"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-black_car1--blue_car1.npy")/50)*10.0
compatibilityMats["11"] = np.exp(-np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/inter_sequence_compatibility-bbox_dist-blue_car1--blue_car1.npy")/50)*10.0

# <codecell>

class TempWindow() :
    def __init__(self, synthesisedSequence):
        self.EXTEND_LENGTH = 301
        self.semanticSequences = []
        self.preloadedTransitionCosts = {}
        for index, seq in enumerate(synthesisedSequence[DICT_USED_SEQUENCES]) :
            self.semanticSequences.append(np.load(seq).item())
            if DICT_TRANSITION_COSTS_LOCATION in self.semanticSequences[-1].keys() :
                self.preloadedTransitionCosts[index] = np.load(self.semanticSequences[-1][DICT_TRANSITION_COSTS_LOCATION])#/GRAPH_MAX_COST*100.0
                print "loaded", self.semanticSequences[-1][DICT_TRANSITION_COSTS_LOCATION]

def getNewFramesForSequenceFull(self, synthesisedSequence, instancesToUse, instancesLengths, startingFrame, resolveCompatibility = True, numSteps=10, costsAlpha=0.1, compatibilityAlpha=0.65) :

    gm = opengm.gm(instancesLengths.repeat(self.EXTEND_LENGTH))

    self.allUnaries = []

    for i, instanceIdx in enumerate(instancesToUse) : # xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
        seqIdx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
        desiredSemantics = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS][startingFrame:startingFrame+self.EXTEND_LENGTH, :]

        if len(desiredSemantics) != self.EXTEND_LENGTH :
            raise Exception("desiredSemantics length is not the same as EXTEND_LENGTH")

        ################ FIND DESIRED START FRAME ################ 
        if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
            desiredStartFrame = 0
        else :
            desiredStartFrame = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]

        distVariance = 1.0/50.0 ##self.semanticsImportanceSpinBox.value() ##0.0005

        ################ GET UNARIES ################
        self.unaries = vectorisedMultiNormalMultipleMeans(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

        ## normalizing to turn into probabilities
        self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
        impossibleLabels = self.unaries <= 0.0
        ## cost is -log(prob)
        self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
        ## if prob == 0.0 then set maxCost
        self.unaries[impossibleLabels] = GRAPH_MAX_COST


        ## force desiredStartFrame to be the first frame of the new sequence
        self.unaries[:, 0] = GRAPH_MAX_COST
        self.unaries[desiredStartFrame, 0] = 0.0
        
        self.unaries = costsAlpha*self.unaries

        self.allUnaries.append(np.copy(self.unaries.T))

        ## add unaries to the graph
        fids = gm.addFunctions(self.unaries.T)
        # add first order factors
        gm.addFactors(fids, arange(self.EXTEND_LENGTH*i, self.EXTEND_LENGTH*i+self.EXTEND_LENGTH))


        ################ GET PAIRWISE ################
        pairIndices = np.array([np.arange(self.EXTEND_LENGTH-1), np.arange(1, self.EXTEND_LENGTH)]).T + self.EXTEND_LENGTH*i

        ## add function for row-nodes pairwise cost
        fid = gm.addFunction((1.0-costsAlpha)*(1.0-compatibilityAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        ## add second order factors
        gm.addFactors(fid, pairIndices)

    ################ ADD THE PAIRWISE BETWEEN ROWS ################
    if resolveCompatibility :
        for i, j in np.argwhere(np.triu(np.ones((len(instancesToUse), len(instancesToUse))), 1)) :
            seq1Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instancesToUse[i]][DICT_SEQUENCE_IDX]
            seq2Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instancesToUse[j]][DICT_SEQUENCE_IDX]
            pairIndices = np.array([np.arange(self.EXTEND_LENGTH*i, self.EXTEND_LENGTH*i+self.EXTEND_LENGTH), 
                                    np.arange(self.EXTEND_LENGTH*j, self.EXTEND_LENGTH*j+self.EXTEND_LENGTH)]).T
#             print pairIndices

            ## add function for column-nodes pairwise cost
            if seq1Idx <= seq2Idx :
                fid = gm.addFunction((1.0-costsAlpha)*compatibilityAlpha*np.copy(compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))]))
                print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx])),
                print compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].shape
            else :
                fid = gm.addFunction((1.0-costsAlpha)*compatibilityAlpha*np.copy(compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].T))
                print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used Transposed comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))
            ## add second order factors
            gm.addFactors(fid, pairIndices)
            
    print gm; sys.stdout.flush()

    t = time.time()
    inferer = opengm.inference.TrwsExternal(gm=gm, parameter=opengm.InfParam(steps=numSteps))#, useRandomStart=True))
    inferer.infer()
    print "solved in", time.time() - t

    return np.array(inferer.arg(), dtype=int), gm, inferer


def getNewFramesForSequenceIterative(self, synthesisedSequence, instancesToUse, instancesLengths, lockedInstances, startingFrame, resolveCompatibility = False, costsAlpha=0.5, compatibilityAlpha=0.5) :

    self.allUnaries = []
    
    self.synthesisedFrames = {}
    totalCost = 0.0
    for instanceIdx, instanceLength, lockedInstance in zip(instancesToUse, instancesLengths, lockedInstances) : # xrange(len(synthesisedSequence[DICT_SEQUENCE_INSTANCES])) :
        
        gm = opengm.gm(np.array([instanceLength]).repeat(self.EXTEND_LENGTH))
        
        seqIdx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
        desiredSemantics = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_DESIRED_SEMANTICS][startingFrame:startingFrame+self.EXTEND_LENGTH, :]
        
        if lockedInstance : 
            if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]) != self.EXTEND_LENGTH :
                raise Exception("not enough synthesised frames")
            else :
                self.synthesisedFrames[instanceIdx] = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame:startingFrame+self.EXTEND_LENGTH]
                print "locked instance", instanceIdx
                print self.synthesisedFrames[instanceIdx]
                continue

        if len(desiredSemantics) != self.EXTEND_LENGTH :
            raise Exception("desiredSemantics length is not the same as EXTEND_LENGTH")

        ################ FIND DESIRED START FRAME ################ 
        if len(synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES]) == 0 :
            desiredStartFrame = 0
        else :
            desiredStartFrame = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_FRAMES][startingFrame]

        distVariance = 1.0/2.0 ##self.semanticsImportanceSpinBox.value() ##0.0005

        ################ GET UNARIES ################
        self.unaries = vectorisedMultiNormalMultipleMeans(self.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS], desiredSemantics, np.eye(desiredSemantics.shape[1])*distVariance, False).T

        ## normalizing to turn into probabilities
        self.unaries = self.unaries / np.sum(self.unaries, axis=0).reshape((1, self.unaries.shape[1]))
        impossibleLabels = self.unaries <= 0.0
        ## cost is -log(prob)
        self.unaries[np.negative(impossibleLabels)] = -np.log(self.unaries[np.negative(impossibleLabels)])
        ## if prob == 0.0 then set maxCost
        self.unaries[impossibleLabels] = GRAPH_MAX_COST


        ## force desiredStartFrame to be the first frame of the new sequence
        self.unaries[:, 0] = GRAPH_MAX_COST
        self.unaries[desiredStartFrame, 0] = 0.0
        
        #### minimizing totalCost = a * unary + (1 - a) * (b * vert_link + (1-b)*horiz_link) = a*unary + (1-a)*b*sum(vert_link) + (1-a)*(1-b)*horiz_link
        #### where a = costsAlpha, b = compatibilityAlpha, 
        
        compatibilityCosts = np.zeros_like(self.unaries)
        if resolveCompatibility :
            seq1Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instanceIdx][DICT_SEQUENCE_IDX]
            for instance2Idx in np.sort(self.synthesisedFrames.keys()) :
                seq2Idx = synthesisedSequence[DICT_SEQUENCE_INSTANCES][instance2Idx][DICT_SEQUENCE_IDX]
                print "considering sequences", seq1Idx, seq2Idx, self.synthesisedFrames.keys()
                
#                 if instance2Idx != 1 :
#                     continue
                
                if seq1Idx <= seq2Idx :
#                     self.unaries = (1.0-compatibilityAlpha)*self.unaries + compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].T[self.synthesisedFrames[instance2Idx], :].T
                    compatibilityCosts += (1.0-costsAlpha)*compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].T[self.synthesisedFrames[instance2Idx], :].T
                    
                    print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx])),
                    print compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))].shape
                else :
#                     self.unaries = (1.0-compatibilityAlpha)*self.unaries + compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))][self.synthesisedFrames[instance2Idx], :].T
                    compatibilityCosts += (1.0-costsAlpha)*compatibilityAlpha*compatibilityMats[np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))][self.synthesisedFrames[instance2Idx], :].T
                    
                    print "added vertical pairwise between", seq1Idx, "and", seq2Idx, "   used Transposed comptabilityMat", np.string_(np.min([seq1Idx, seq2Idx])) + np.string_(np.max([seq1Idx, seq2Idx]))        
#         ## doing the alpha*unaries + (1-alpha)*pairwise thingy
#         self.unaries *= costsAlpha
        self.unaries = costsAlpha*self.unaries + compatibilityCosts
        

        self.allUnaries.append(np.copy(self.unaries.T))
        

        ## add unaries to the graph
        fids = gm.addFunctions(self.unaries.T)
        # add first order factors
        gm.addFactors(fids, arange(self.EXTEND_LENGTH))


        ################ GET PAIRWISE ################
        pairIndices = np.array([np.arange(self.EXTEND_LENGTH-1), np.arange(1, self.EXTEND_LENGTH)]).T

#         ## add function for row-nodes pairwise cost doing the alpha*unaries + (1-alpha)*pairwise thingy at the same time
#         fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        if resolveCompatibility :
            fid = gm.addFunction((1.0-costsAlpha)*(1.0-compatibilityAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        else :
            fid = gm.addFunction((1.0-costsAlpha)*(self.preloadedTransitionCosts[seqIdx]+0.1))##self.toggleSpeedDeltaSpinBox.value())
        ## add second order factors
        gm.addFactors(fid, pairIndices)        
            
        print gm; sys.stdout.flush()

        t = time.time()
        inferer = opengm.inference.DynamicProgramming(gm=gm)
        inferer.infer()
        print "solved in", time.time() - t, "cost", gm.evaluate(inferer.arg())
        print np.array(inferer.arg(), dtype=int)
        totalCost += gm.evaluate(inferer.arg())
        self.synthesisedFrames[instanceIdx] = np.array(inferer.arg(), dtype=int)
        
    return self.synthesisedFrames, totalCost
#     return np.array(inferer.arg(), dtype=int), gm

# <codecell>

synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/synthesised_sequence.npy").item()

frameIdx = 0

tempWindow = TempWindow(synthSeq)

#### NEW WAY ####

instancesToUse = []
instancesLengths = []
maxFrames = 0
t = time.time()
for i in xrange(len(synthSeq[DICT_SEQUENCE_INSTANCES])) :

#     availableDesiredSemantics = len(self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]) - self.frameIdx
#     if availableDesiredSemantics < self.EXTEND_LENGTH :
#         ## the required desired semantics by copying the last one
#         print "extended desired semantics for", i,
#         lastSemantics = self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][-1, :]
#         self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS] = np.concatenate((self.synthesisedSequence[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS],
#                                                                                                        lastSemantics.reshape((1, len(lastSemantics))).repeat(self.EXTEND_LENGTH-availableDesiredSemantics, axis=0)))
#     else :
#         print "didn't extend semantics for", i
    desiredSemantics = synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS][frameIdx:frameIdx+tempWindow.EXTEND_LENGTH, :]
    print "num of desired semantics =", desiredSemantics.shape[0], "(", len(synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_DESIRED_SEMANTICS]), ")",
    print 

    seqIdx = synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_IDX]

    if seqIdx in tempWindow.preloadedTransitionCosts.keys() :
        instancesToUse.append(i)
        instancesLengths.append(len(tempWindow.semanticSequences[seqIdx][DICT_FRAME_SEMANTICS]))
#                 newFrames = self.getNewFramesForSequenceInstanceQuick(i, self.semanticSequences[seqIdx],
#                                                                       self.preloadedTransitionCosts[seqIdx]+self.toggleSpeedDeltaSpinBox.value(),
#                                                                       desiredSemantics, self.frameIdx, framesToNotUse)
    else :
        print "ERROR: cannot extend instance", i, "because the semantic sequence", seqIdx, "does not have preloadedTransitionCosts"
        break
        
instancesToUse = np.array(instancesToUse)
instancesLengths = np.array(instancesLengths)
print instancesToUse, instancesLengths
newFramesNewWay, gm, inferer = getNewFramesForSequenceFull(tempWindow, synthSeq, np.array(instancesToUse), np.array(instancesLengths), frameIdx, numSteps=100)
# getNewFramesForSequenceIterative(tempWindow, synthSeq, np.array(instancesToUse), np.array(instancesLengths), np.ones(len(instancesToUse), bool), frameIdx, True)
# gm = window.semanticLoopingTab.getNewFramesForSequenceFull(synthSeq, np.array(instancesToUse), np.array(instancesLengths), frameIdx, True)

# selectedSequences = np.array([1, 2])
# ## using Peter's idea
# if True :
# #     print selectedSequences, instancesToUse, np.array([instancesToUse != selectedSequence for selectedSequence in selectedSequences]).all(axis=0)
#     notSelected = np.array([instancesToUse != selectedSequence for selectedSequence in selectedSequences]).all(axis=0)
#     notSelectedInstances = instancesToUse[notSelected]
#     selectedSequences = instancesToUse[np.negative(notSelected)]
#     for s in xrange(len(selectedSequences)) : #permutation in itertools.permutations(selectedSequences, len(selectedSequences)) :
# #         print np.concatenate((notSelectedInstances, permutation)), np.concatenate((np.ones(len(notSelectedInstances), bool), np.zeros(len(permutation), bool)))
#         reorderedInstances = np.concatenate((notSelectedInstances, np.roll(selectedSequences, s)))
#         reorderedLengths = np.concatenate((instancesLengths[notSelected], np.roll(instancesLengths[np.negative(notSelected)], s)))
#         lockedInstances = np.concatenate((np.ones(len(instancesToUse)-1, bool), [False]))
#         print reorderedInstances, reorderedLengths, lockedInstances
#         print 
# #         getNewFramesForSequenceIterative(tempWindow, synthSeq, reorderedInstances, reorderedLengths, lockedInstances, frameIdx-50, True, 0.3, 0.7)
#         print 


print "new way done in", time.time() - t



# print gm.evaluate(newFramesNewWay)
# print gm
# # print 757.411073682 + 827.424882297 + 805.571144802 + 717.196980651 + 739.054008251 + 765.178588429
# print newFramesNewWay.reshape((2, tempWindow.EXTEND_LENGTH))
# newFrames1 = newFramesNewWay.reshape((2, tempWindow.EXTEND_LENGTH))[0, :]
# newFrames2 = newFramesNewWay.reshape((2, tempWindow.EXTEND_LENGTH))[1, :]

# <codecell>

print gm.evaluate(newFramesNewWay)
print gm.evaluate(np.array(tmp).flatten())

# <codecell>


synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana/synthesised_sequence.npy").item()
# print synthSeq
tmp = []
for i in xrange(len(synthSeq[DICT_SEQUENCE_INSTANCES])) :
#     synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES] = newFramesNewWay.reshape((4, tempWindow.EXTEND_LENGTH))[i, :]
#     print synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES]
    tmp.append(synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES])
# np.save("/home/ilisescu/PhD/data/synthesisedSequences/newHavana-TRWS-100/synthesised_sequence.npy", synthSeq)
print np.array(tmp).flatten()

# <codecell>

print np.sort(np.load("/media/ilisescu/Data1/PhD/data/theme_park_sunny/sprite-0000-orange_flag1.npy").item()[DICT_BBOXES].keys())
print np.sort(np.load("/media/ilisescu/Data1/PhD/data/theme_park_sunny/sprite-0000-orange_flag1.npy").item()[DICT_ICON_FRAME_KEY])

# <codecell>

# figure(); plot(newFramesNewWay.reshape((4, tempWindow.EXTEND_LENGTH))[0, :], c="r")
# plot(newFramesNewWay.reshape((4, tempWindow.EXTEND_LENGTH))[1, :], c="g")
# plot(newFramesNewWay.reshape((4, tempWindow.EXTEND_LENGTH))[2, :], c="b")
# plot(newFramesNewWay.reshape((4, tempWindow.EXTEND_LENGTH))[3, :], c="m")
synthSeq = np.load("/home/ilisescu/PhD/data/synthesisedSequences/newHavana-TRWS-500-3356secs/synthesised_sequence.npy").item()
print synthSeq
for i in xrange(len(newFramesNewWay.reshape((4, tempWindow.EXTEND_LENGTH)))) :
    synthSeq[DICT_SEQUENCE_INSTANCES][i][DICT_SEQUENCE_FRAMES] = newFramesNewWay.reshape((4, tempWindow.EXTEND_LENGTH))[i, :]
    
np.save("/home/ilisescu/PhD/data/synthesisedSequences/newHavana-TRWS-500-3356secs/synthesised_sequence.npy", synthSeq)

