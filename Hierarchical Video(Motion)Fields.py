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
from scipy import ndimage
from scipy import stats

from tsne import tsne

from _emd import emd

import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import ComputeGridFeatures as cgf

from pygraph.classes.digraph import digraph
from pygraph.classes.graph import graph
from pygraph.readwrite.dot import write
from pygraph.algorithms.minmax import shortest_path

dataFolder = "/home/ilisescu/PhD/data/"
POSE = 0
VELOCITY = 1
FUT_VELOCITY = 2

app = QtGui.QApplication(sys.argv)

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
sampleData = "ribbon2/"
# sampleData = "flag_blender/"
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

np.save("tmpfeat.npy", features)
np.save("tmpv.npy", v)

# <codecell>

############# INITIALIZING FRAME STATES AND DISTANCE MATRIX ###############

## video field data
print "fg histogram features: ", features.shape
numStates = features.shape[0]-2
descriptorSize = features.shape[1]
## fs contains all the f's, i.e. frame states
## f = (x, v, y), x is pixel data/fg features, v = x'-x ('=i+1), y = x"-x'
## compute x's
fs = np.copy(np.reshape(features[0:numStates, :], (numStates, descriptorSize, 1)))
## compute v's
fs = np.concatenate((fs, (features[1:-1, :]-features[0:-2, :]).reshape((numStates, descriptorSize, 1))), axis=-1)
## compute y's
fs = np.concatenate((fs, (features[2:, :]-features[1:-1, :]).reshape((numStates, descriptorSize, 1))), axis=-1)
print fs.shape[0], "frame states(lost", features.shape[0]-numStates, "because I need extra frames at",
print "the end for computing v and y),", "features size is", fs.shape[1]

## load distance matrix as dissimilarity matrix d
## d(f, f') says how dissimilar states f and f' are from each other
# distanceMatrix = np.array(np.load(outputData + "hist2demd_16x16_distMat" + ".npy"), dtype=np.float)
# distanceMatrix = np.array(np.load(outputData + "hog_distMat" + ".npy"), dtype=np.float)
distanceMatrix = np.array(np.load(outputData + "appearance_hog_L2_set150_distMat" + ".npy"), dtype=np.float)
# distanceMatrix = np.array(np.load(outputData + "semantics_hog_set150_distMat" + ".npy"), dtype=np.float)
# distanceMatrix = np.array(np.load(outputData + "appearance_set150_grid16x16_ABS_distMat" + ".npy"), dtype=np.float)
# distanceMatrix = np.array(np.load(outputData + "appearance_hog_set150_fisher_distMat" + ".npy"), dtype=np.float)
# distanceMatrix = np.array(np.load(outputData + "vanilla_distMat" + ".npy"), dtype=np.float)
distanceMatrix /= np.max(distanceMatrix)
if True :
    filterSize = 4
    distMat = vtu.filterDistanceMatrix(distanceMatrix, filterSize, True)
    ## troncate fs to match indices in d to indices in fs
    fs = np.copy(fs[filterSize:numStates-(filterSize-2), :, :])
    print fs.shape[0], "frame states(lost", numStates-fs.shape[0], "because I had to filter 8 frames out for ",
    print "dynamicism preservation),", "features size is", fs.shape[1]
else :
    filterSize = 0
    distMat = np.copy(distanceMatrix[0:-2, 0:-2])
    
figure(); imshow(distMat, interpolation='nearest')
d = np.copy(distMat)
numStates = fs.shape[0]

# <codecell>

############# INITIALIZING TRANSITIONS COST MATRIX ###############

## transition cost based on appearance are the shifted up distance matrix
t = np.copy(d[1:d.shape[1], 0:-1])
## this is used to compute appearance reward
transProbs = vtu.dist2prob(t, 0.5, True)
d = np.copy(d[0:-1, 0:-1])
## troncate fs to match indices in d to indices in fs
fs = np.copy(fs[0:-1, :, :])
print fs.shape[0], "frame states(lost", numStates-fs.shape[0], "because transition cost does not",
print "have info about last frame),", "features size is", fs.shape[1]
numStates = fs.shape[0]

# <codecell>

############# COMPUTING FRAME STATES NEIGHBOURS AND SIMILARITY WEIGHTS ###############

## find k-nearest neighbors based on transition costs
k = 15
neighbours = np.argsort(t+(np.eye(d.shape[0])*np.max(d)+1))[:, 0:k] #np.argsort(d)[:, 0:k] # we don't want ourselves as neighbour
# figure(); imshow(neighbours)

# w are similarity weights
w = 1/(0.01 + d[np.reshape(range(d.shape[0]), (d.shape[0], 1)).repeat(k, axis=-1), neighbours]**2)#(np.sort(d)[:, 1:k+1]**2)
# normalize to make weights sum to 1
w /= np.repeat(np.reshape(np.sum(w, axis=-1), (numStates, 1)), k, axis=-1)

# <codecell>

print np.argwhere(d[np.reshape(range(d.shape[0]), (d.shape[0], 1)).repeat(k, axis=-1), neighbours]**2 == 0)
print w[23, :]
print neighbours[23, :]

# <codecell>

############# LOADING LABELS AND INITIALIZING VIDEO FIELD A ###############

# load labels to use as taskParamsValues
# labels = np.load(outputData + "hist2demd_mult0.02_16x16_labels.npy")[()]['labelProbs']
# labels = np.load(outputData + "appearance_hog_L2_set150_userex80_mult0.048_labels.npy")[()]['labelProbs']
labels = np.load(outputData + "semantics_hog_L2_set150_userex80_mult0.035_labels.npy")[()]['labelProbs']
# labels = np.load(outputData + "l2dist_guiex_set60_mult0.045_labels.npy")[()]['labelProbs']
possibleLabelTasks = np.eye(labels.shape[-1])

# compute video field A value at each state, i.e. compute possible actions at each state as defined in section 4.1
# A[i, :, :] gives KxK matrix where each row is an action that favors the neighbour corresponding to row idx
A = np.zeros((numStates, k , k))
for fIdx in range(numStates) :
    tmp = np.repeat(np.reshape(w[fIdx, :], (1, k)), k, axis=0)
    ## set action corresponding to ith neighbour to 1
    tmp[np.where(np.eye(k)==1)] = 1
    ## and renormalize
    tmp /= np.repeat(np.reshape(np.sum(tmp, axis=-1), (k, 1)), k, axis=-1) #np.repeat(np.reshape(np.sum(tmp, axis=0), (1, k)), k, axis=0)
    A[fIdx, :, :] = np.copy(tmp)

# <codecell>

clrs = ['r', 'g', 'b', 'm']
fig1 = figure()
xlabel('all points')

for i in xrange(0, 4) :
    figure(fig1.number); plot(labels[:, i], clrs[i])

# <codecell>

############# INITIALIZING THE SET OF TASK STATES ###############

## ss contains all the s's, i.e. task states
ss = []
for i in xrange(numStates) :
    ss.append({'frameState': fs[i, :, :], 'taskParams': (labels[i, :], i)})

# <codecell>

print len(ss)
print ss[0]
print np.argsort(t[0, :])[0:15]
print np.sort(t[0, :])[0:15]
print np.max(t[0, :]), np.max(t)
print 1/((np.sort(t[0, :])+1)**2)[0:15], np.min(1/((np.sort(t[0, :])+1)**2))
print distMat.shape
print neighbours[0, :]
print w[0, :]

# <codecell>

def integrateFrameState(frameState, action, neighbours, delta, neighboursIndices) :
    """Computes a new frame state from frameState given action weights and neighbours to interpolate from.
    
           frameState: Nx3 array containing (x, v, y), with N the descriptor size
           action: 1xK array of weights
           neighbours: 1xK frame states
           delta: tugging parameter in [0, 1]
           neighboursIndices: contains the indices in fs of the given neighbours
           
        return: newFrameState"""
    if len(action) != neighbours.shape[0] :
        raise Exception("Number of action weights and neighbours does not match")
    
    newFrameState = np.zeros(frameState.shape)
    
    descriptorSize = neighbours.shape[1]
    k = len(action)
    actionWeights = np.repeat(np.reshape(action, (k, 1)), descriptorSize, axis=-1)
    closestFrameState = neighbours[0, :, :]
    
    ## interpolate new pose based on velocities v (equation 5)
    newFrameState[:, POSE] = frameState[:, POSE] + (1-delta)*np.sum(actionWeights*neighbours[:, :, VELOCITY], axis=0) + delta*(closestFrameState[:, POSE]+closestFrameState[:, VELOCITY]-frameState[:, POSE])
    ## interpolate new velocity based on future velocities y (equation 6)
    newFrameState[:, VELOCITY] = (1-delta)*np.sum(actionWeights*neighbours[:, :, FUT_VELOCITY], axis=0) + delta*closestFrameState[:, FUT_VELOCITY]
    ## hack to avoid the interpolation thing
    return np.copy(neighbours[np.argmax(action), :, :]), neighboursIndices[np.argmax(action)] #newFrameState

def integrateTaskState(taskState, action, delta, neighboursIndices, taskStates, frameStates) :
    """Computes a new task state with a newly integrated frame state and update task parameters.
    
           taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
           action: 1xK array of weights
           neighbours: 1xK frame states
           delta: tugging parameter in [0, 1]
           neighboursIndices: contains the indices in fs of the given neighbours
           taskStates: contains the full set of N task states
           frameStates: contains the full set of N frame states
           
        return: newTaskState"""
    newFrameState, newIdx = integrateFrameState(taskState['frameState'], action, frameStates[neighboursIndices, :, :], delta, neighboursIndices)
#     newTaskState = {'frameState':newFrameState, 'taskParams':taskParamsValues[newIdx, :]}
    newTaskState = taskStates[newIdx]
    
    return newTaskState, newIdx
    

# <codecell>

## test integration function
fIdx = 234
newTaskState, newIdx = integrateTaskState(ss[fIdx], A[fIdx, 8, :], 1.0, neighbours[fIdx, :], ss, fs)
print neighbours[fIdx]
print labels[newIdx, :]
print newTaskState
print np.sum(newTaskState['taskParams'][0])
print newIdx

# <codecell>

def semanticLabelsTask(currentLabels, goalLabels) :
    """Computes the distance in [0.0, 1.0] between currentLabels and goalLabels
            
            currentLabels: the semantic labels of current state
            goalLabels: the semantic labels of goal task
        
        return: distance d"""
    d = (2.0-np.sum(np.abs(currentLabels-goalLabels)))/2.0
    return d

def idxTask(currentIdx, goalIdx, numFrames) :
    """Computes the distance in [0.0, 1.0] between currentIdx and goalIdx given the total number of frames
        Distance defined as 1-(current - goal)/total
            
            currentIdx: the index of the current state
            goalIdx: the index of the goal state
            numFrames: total number of frames
        
        return: distance d"""
    tot = np.max((goalIdx, np.abs(goalIdx-numFrames)))
    d = 1.0-(float(np.abs(currentIdx-goalIdx))/float(tot))
    return d

def idxStepTask(currentIdx, goalIdx, numFrames) : 
    """Computes the distance in [0.0, 1.0] between currentIdx and goalIdx given the total number of frames
        Distance defined as a smoothstep function
            
            currentIdx: the index of the current state
            goalIdx: the index of the goal state
            numFrames: total number of frames
        
        return: distance d"""
    stepReward = smoothStep(arange(0.0, numFrames), 0.0, 80.0*2, 0.3)
    d = stepReward[np.abs(currentIdx-goalIdx)]
    return d

def idxRangeTask(currentIdx, goalIdx, numFrames) : 
    """Computes the distance in [0.0, 1.0] between currentIdx and goalIdx given the total number of frames
        Distance defined as a smoothstep function
            
            currentIdx: the index of the current state
            goalIdx: the index of the goal state
            numFrames: total number of frames
        
        return: distance d"""
    rangeSize = 10.0
#     tot = np.max((goalIdx, np.abs(goalIdx-numFrames)))
#     d = 1.0-np.floor((float(np.abs(currentIdx-goalIdx))+rangeSize/2.0)/rangeSize)*float(rangeSize/tot)
    d = np.exp(-np.floor((np.abs(currentIdx-goalIdx)+rangeSize/2)/rangeSize)/(numFrames/(rangeSize*10)))
    return d


def r_t(taskState, neighboursTaskParamsValues, action, taskToPerform) :
    """Computes the task reward [0.0, 1.0] of performing action at taskState given the taskToPerform
            
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            neighboursTaskParamsValues: kxN array containing the N task labels of the k neighbours of taskState
            action: action performed at taskState
            taskToPerform: task that needs to be performed
        
        return: reward r
        
        NOTE: since the action is not actually used for interpolation, only the neighbour corresponding to the heighest action weight are considered"""
    
    r = (2.0-np.sum(np.abs(neighboursTaskParamsValues[np.argmax(action)]-taskToPerform)))/2.0
    return r

def r_ht(tiers, tierFunctions, tierImportance) :
    """Computes the hierarchical task reward [0.0, 1.0] given the state values at the various tiers, the functions to call for each tier and the importance of each tier
            
            tiers: for each tier contains the values of the parameters the corresponding tier function needs
            tierFunctions: pointers to functions for each tier
            tierImportance: how much each tier contributes to final reward
        
        return: reward r
        
        NOTE: since the action is not actually used for interpolation, only the neighbour corresponding to the heighest action weight are considered"""
    
    if len(tierImportance) != len(tiers) or len(tierImportance) != len(tierFunctions) :
        raise Exception("length mismatch in hierarchical task reward")
        
#     print "la", tiers
    r = 0
    for i in xrange(len(tierImportance)) :
#         print "la", tierImportance[i], tierFunctions[i](*tiers[i]), 
        r += tierImportance[i]*tierFunctions[i](*tiers[i])
#     print
    return r

def getTaskHierarchyTiers(taskState, neighboursIdxs, action, taskToPerform, taskStates) :
    """Returns the tiers used to compute the hierachical task reward
            
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            neighboursIdxs: indices of neighbours of taskState in the set of task states
            action: action performed at taskState
            taskToPerform: hierarchical task that needs to be performed
            taskStates: full list of task states
        
        return: tiers
        
        NOTE: since the action is not actually used for interpolation, only the neighbour corresponding to the heighest action weight are considered"""
    
    tiers = ((taskStates[neighboursIdxs[np.argmax(action)]]['taskParams'][0], taskToPerform[0]), 
             (taskStates[neighboursIdxs[np.argmax(action)]]['taskParams'][1], taskToPerform[1], len(taskStates)))
    return tiers
    

def r_a(taskState, stateIdx, neighboursIdxs, action, transitionsCost, p) :
    """Computes the apperance reward [0.0, 1.0] of performing action at taskState given the appearance-based transition cost
            
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            stateIdx: index of taskState in the set of task states
            neighboursIdxs: indices of neighbours of taskState in the set of task states
            action: action performed at taskState
            transitionsCost: NxN matrix continaing transition costs from frame i to frame j
            p: controls the tradeoff between taking multiple good transitions versus a single poorer one
        
        return: reward r
        
        NOTE: since the action is not actually used for interpolation, only the neighbour corresponding to the heighest action weight are considered"""
    
#     costs = 1.0/(transitionsCost[stateIdx, neighboursIdxs]**p)
#     costs /= np.max(costs)
    costs = transProbs[stateIdx, neighboursIdxs]/np.max(transProbs[stateIdx, :])
    r = costs[np.argmax(action)]
    return r
    
def pi_l(taskState, actions, valueFunction, stateIdx, neighboursIndices, taskToPerform, transitionsCost, l, p, tierFunctions, tierImportance, taskStates, frameStates, verbose=False):
    """Gives the next best action based on a lookahead policy to maximize reward.
    
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            actions: possible actions from state KxK matrix where each row is an action
            valueFunction: 1xN precomputed value function that represents future reward
            stateIdx: is the idx of state in states
            neighboursIndices: list of neighbour states indices for each state
            taskToPerform: task that needs to be performed
            transitionsCost: NxN matrix continaing transition costs from frame i to frame j
            l: controls the tradeoff between favouring good looking transitions versus task-wise good transitions
            p: controls the tradeoff between taking multiple good transitions versus a single poorer one
            tierFunctions: kxN array containing the N task labels of the k neighbours of taskState
            tierImportance: how much each tier contributes to final reward
            taskStates: contains the full set of N task states
            frameStates: contains the full set of N frame states
            verbose: print additional information
        
        return: bestAction, bestActionIdx"""
    
    rewards = np.zeros(len(actions))
    immediateRewards = np.zeros(len(actions))
    individualImmediateRewards = np.zeros((2, len(actions)))
    for idx in range(len(actions)) :
        newTaskState, newIdx = integrateTaskState(taskState, actions[idx, :], 1.0, neighboursIndices[stateIdx, :], taskStates, frameStates)
        
        tiers = getTaskHierarchyTiers(taskState, neighboursIndices[stateIdx, :], actions[idx, :], taskToPerform, taskStates)
        taskReward = r_ht(tiers, tierFunctions, tierImportance)
#         taskReward = r_t(taskState, taskParamsValues[neighboursIndices[stateIdx, :], :], actions[idx, :], taskToPerform)
        appearanceReward = r_a(taskState, stateIdx, neighboursIndices[stateIdx, :], actions[idx, :], transitionsCost, p)
        
        amount = l*taskReward + (1.0-l)*appearanceReward + valueFunction[newIdx]
        rewards[idx] = amount
        immediateRewards[idx] = l*taskReward + (1.0-l)*appearanceReward
        individualImmediateRewards[0, idx] = taskReward
        individualImmediateRewards[1, idx] = appearanceReward
        if verbose :
            print "current amount for action in pi_l", amount, np.sum(actions[idx, :]), immediateRewards[idx], taskReward, appearanceReward, valueFunction[newIdx], newIdx
    
    ## get best action index
    bestActionIdx = np.argmax(rewards)
    
    bestAction = np.copy(actions[bestActionIdx, :])
    return bestAction, bestActionIdx, immediateRewards, individualImmediateRewards

# <codecell>

def smoothStep(x, mean, interval, steepness) :
    a = mean-np.floor(interval/2.0)
    b = mean-np.floor((interval*steepness)/2.0)
    c = mean+np.ceil((interval*steepness)/2.0)
    d = mean+np.ceil(interval/2.0)
#     print a, b, c, d
    
    ## find step from 0 to 1
    step1 = np.clip((x - a)/(b - a), 0.0, 1.0);
    step1 = step1*step1*step1*(step1*(step1*6 - 15) + 10);
    
    ## find step from 1 to 0
    step2 = np.clip((x - d)/(c - d), 0.0, 1.0);
    step2 = step2*step2*step2*(step2*(step2*6 - 15) + 10);
    
    ## combine the two steps together
    result = np.zeros(x.shape)
    result += step1
    result[np.argwhere(step2 != 1.0)] = step2[np.argwhere(step2 != 1.0)]
    return result;

stepReward = smoothStep(arange(0.0, numFrames), 0.0, 30.0*2, 0.3)

# <codecell>

## test pi_l function
## amount, np.sum(actions[idx, :]), taskReward, appearanceReward, valueFunction[newIdx], newIdx
trackingState = 906
taskState = {'frameState': fs[trackingState, :, :], 'taskParams': labels[trackingState, :]}
neighboursTaskParamsValues = labels[neighbours[trackingState, :], :]
action = A[trackingState, 5, :]
task = 2
p = 2.0
l = 0.65
taskToPerform = possibleLabelTasks[task, :]
# v = np.zeros((fs.shape[0], possibleTasksToPerform.shape[0]))

print "neighbours", neighbours[trackingState, :]
print "neighbours' labels", neighboursTaskParamsValues
print "chosen action", action, np.sum(action)
print "task to perform", taskToPerform
print "rewards for given neighbours", (2.0-np.sum(np.abs(neighboursTaskParamsValues-np.reshape(taskToPerform, (1, 4)).repeat(k, axis=0)), axis = -1))/2.0

print "total task reward computed by method", r_t(taskState, neighboursTaskParamsValues, action, taskToPerform)
print "total apperance reward computed by method", r_a(taskState, trackingState, neighbours[trackingState, :], action, 1.0+t, 2.0)

tiers = getTaskHierarchyTiers(taskState, neighbours[trackingState, :], action, (taskToPerform, 0), ss)
print tiers
print r_ht(tiers, (semanticLabelsTask, idxTask), (1.0, 0.0))
print
bestA, bestIdx, immediateRewards, individualImmediateRewards = pi_l(taskState, A[trackingState, :, :], np.zeros(numFrames), trackingState, neighbours, (taskToPerform, 230), 1.0+t, l, p, (semanticLabelsTask, idxTask), (1.0, 0.0), ss, fs, True)
print "best A and idx", bestA, bestIdx
print 
bestA, bestIdx, immediateRewards, individualImmediateRewards = pi_l(taskState, A[trackingState, :, :], np.zeros(numFrames), trackingState, neighbours, (taskToPerform, 230), 1.0+t, l, p, (semanticLabelsTask, idxStepTask), (1.0, 0.0), ss, fs, True)
print "step best A and idx", bestA, bestIdx





# <codecell>

## let's try interpolation yay
print trackingState
print neighbours[trackingState, :]

stencils = cgf.stencil2D(4, 4, imageSize)
tmpAlpha = np.zeros(imageSize[0:2])
tmpImg = np.zeros(imageSize)
for i, n in zip(np.arange(len(neighbours)), neighbours[trackingState]+filterSize) :
    img = np.array(cv2.cvtColor(cv2.imread(frames[n]), cv2.COLOR_BGR2RGB))/255.0
    alpha = np.zeros(img.shape[0:-1])
    if os.path.isfile(mattes[i]) :
        alpha = np.array(cv2.cvtColor(cv2.imread(mattes[n]), cv2.COLOR_BGR2GRAY))/255.0
        img *= np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)

    ## use stencils to divide the image into blocks and count number of foreground pixels
    for s in xrange(0, len(stencils)) :
#         index = s + idx*len(stencils)
        print len(np.argwhere(alpha[stencils[s]] != 0)),
    print
    tmpAlpha += bestA[i]*alpha
    tmpImg += bestA[i]*img
    

# <codecell>

figure(); imshow(np.array(cv2.cvtColor(cv2.imread(frames[neighbours[trackingState, bestIdx]+filterSize]), cv2.COLOR_BGR2RGB))/255.0, interpolation='nearest')

# <codecell>

figure(); imshow(tmpAlpha/k, interpolation='nearest')

# <codecell>

matte = np.reshape(np.array(tmpAlpha*255, dtype=np.uint8), (tmpAlpha.shape[0], tmpAlpha.shape[1], 1))
print matte.shape
fullResoFrame = np.array(tmpImg, dtype=np.uint8)
print fullResoFrame.shape
Image.frombytes("RGBA", (fullResoFrame.shape[1], fullResoFrame.shape[0]), np.concatenate((fullResoFrame, matte), axis=-1).tostring()).save("interpolated3.png")

# <codecell>

print np.max(tmpAlpha)

# <codecell>

############# FIND THE BEST FRAMES FOR EACH LABEL S.T. THEY ARE THE FURTHEST FROM EACH OTHER ###############

framesPerTask = 6
bestFramesPerTask = []
for i in xrange(len(possibleLabelTasks)) :
    rewardsFromTask = (2.0-np.sum(np.abs(labels[0:numStates, :]-np.reshape(possibleLabelTasks[i, :], (1, len(possibleLabelTasks)), numStates)), axis=1))/2.0
#     print np.argsort(rewardsFromTask)[-50:]
    bestFrames = np.ndarray.flatten(np.argwhere(rewardsFromTask > 0.9))
    idxDiffs = np.abs(bestFrames[0:-1]-bestFrames[1:])
    print idxDiffs, len(bestFrames), np.sum(idxDiffs)
    chosenIdxs = np.arange(0, len(bestFrames), (len(bestFrames)-1)/(framesPerTask-1), dtype=int)
#     chosenIdxs = np.array([0, 41, 80, 87])
    print chosenIdxs, 
    
    frameDists = np.zeros(framesPerTask-1, dtype=int)
    for x in xrange(framesPerTask-1) :
        frameDists[x] = np.sum(idxDiffs[chosenIdxs[x]:chosenIdxs[x+1]])
    chosenScore = np.sum(np.abs(np.diff(frameDists))) + np.sum(idxDiffs)-np.sum(frameDists)
    terminationScore = np.copy(chosenScore/15)
    print chosenScore
    
    for j in xrange(100000):#while chosenScore > terminationScore :
        randIdxs = np.sort(np.random.randint(0, len(bestFrames), framesPerTask))
#         print randIdxs, 
        frameDists = np.zeros(framesPerTask-1, dtype=int)
        for x in xrange(framesPerTask-1) :
            frameDists[x] = np.sum(idxDiffs[randIdxs[x]:randIdxs[x+1]])
        
        ## score has to be minimized and is difference in frame idx difference between chosen indices and how
        ## high these frame index differences
        score = np.sum(np.abs(np.diff(frameDists))) + np.sum(idxDiffs)-np.sum(frameDists)
#         print score
        if score < chosenScore :
            print "better indices", randIdxs, score
            chosenScore = np.copy(score)
            chosenIdxs = np.copy(randIdxs)
            
#     print np.abs(chosenIdxs[0:-1]-chosenIdxs[1:])
#     for j in xrange(len(bestFrames))
    print "chosen frames for task", i, chosenIdxs, bestFrames[chosenIdxs]
    bestFramesPerTask.append(bestFrames[chosenIdxs])

# <codecell>

print bestFramesPerTask

# <codecell>

############# OPTIMIZING FUTURE REWARDS TO GET VALUE FUNCTION V ###############

v = np.zeros((fs.shape[0], possibleLabelTasks.shape[0]))
possibleIdxTasks = np.zeros(4)#np.array([95, 496, 906, 1158])
vTmp = np.copy(v)
prevAdditionalReward = 0
alpha = 0.999
p = 2.0
l = 0.75
gamma = 0.99 #1.0

maxIterations = 20
trackingState = 234
print "tracking progression of state", trackingState
print "neighbours", neighbours[trackingState, :]

# v = np.load("tmpv.npy")
for i in range(maxIterations) :
    ## for each possible task
    print "best actions",
    for task in range(possibleLabelTasks.shape[0]) :
        ## for each frame state
        for fIdx in range(numStates) :
#             si = {'frameState': fs[fIdx, :, :], 'taskParams': labels[fIdx, :]}

            bestA, bestIdx, immediateRewards, individualImmediateRewards = pi_l(ss[fIdx], A[fIdx, :, :], v[:, task], fIdx, neighbours, (possibleLabelTasks[task, :], 0), 1.0+t, l, p, (semanticLabelsTask, idxTask), (1.0, 0.0), ss, fs)
            
            newTaskState, newIdx = integrateTaskState(ss[fIdx], bestA, 1.0, neighbours[fIdx, :], ss, fs)
            
            tiers = getTaskHierarchyTiers(ss[fIdx], neighbours[fIdx, :], bestA, (possibleLabelTasks[task, :], possibleIdxTasks[task]), ss)
            taskReward = r_ht(tiers, (semanticLabelsTask, idxTask), (1.0, 0.0))
            appearanceReward = r_a(ss[fIdx], fIdx, neighbours[fIdx, :], bestA, 1.0+t, 2.0)
            
            vTmp[fIdx, task] = l*taskReward + (1.0-l)*appearanceReward + gamma*v[newIdx, task]
            
            if fIdx == trackingState :
                print bestIdx, "(", newIdx, "),",
    additionalReward = np.linalg.norm(vTmp-v)
    print
    print i, np.sum(np.linalg.norm(vTmp-v, axis=-1)), np.sum(np.abs(vTmp-v)), additionalReward, additionalReward-prevAdditionalReward, np.min(vTmp), np.max(vTmp), np.sum(vTmp)
    print vTmp[trackingState, :]
    sys.stdout.flush()
    v = np.copy(vTmp)
    if additionalReward-prevAdditionalReward == 0.0 :
        break
    prevAdditionalReward = additionalReward

    
# np.save("tmpv.npy", v)

# <codecell>

figure(); plot(np.cumsum(v[:, 1]/np.sum(v[:, 1])))
figure(); plot(v[:, 1]/np.max(v[:, 1]))
figure(); plot(np.cumsum(labels[0:-1, 1])/np.sum(np.cumsum(labels[0:-1, 1])))
figure(); plot(labels[0:-1, 1]/np.sum(labels[0:-1, 1]))

# <codecell>

print np.random.choice(arange(1271), size=10, p=labels[0:-1, 1]/np.sum(labels[0:-1, 1]))
print bestFramesPerTask[1]

# <codecell>

############# INITIALIZE GRAPH AS DEFINED BY MOTION FIELDS AND USE TO COMPUTE SHORTEST PATHS ###############

labelTaskToPerform = possibleLabelTasks[2, :]

gr = digraph()
gr.add_nodes(arange(numStates, dtype=int))
print neighbours.shape, len(gr.nodes())
for i in xrange(numStates) :
    for n in xrange(k):
        taskReward = semanticLabelsTask(labels[neighbours[i, n], :], labelTaskToPerform)
        appearanceReward = r_a(ss[i], i, neighbours[i, :], A[i, n, :], 1.0+t, 2.0)
#         tmp = 1.0-(l*taskReward + (1.0-l)*appearanceReward)+0.01
        tmp = 0.1 + (1.0-appearanceReward)
#         tmp = 0.05 + t[i, neighbours[i, n]]
#         tmp = 0.05 + eucT[i, neighbours[i, n]]
        gr.add_edge((i, neighbours[i, n]), wt=tmp)
    sys.stdout.write('\r' + "Added node " + np.string_(i) + " of " + np.string_(numStates) + " and its neighbours ")
    sys.stdout.flush()
print 
## precomputed shortest paths costs
precomputedSPCosts = np.zeros((numStates, numStates))
precomputedSPLengths = np.zeros((numStates, numStates), dtype=int)
for i in xrange(numStates) :
    shortestPaths = shortest_path(gr, i)
    ## costs
    if len(shortestPaths[1]) < numStates :
        print "state", i, "does not have a shortest path to all other states"
        precomputedSPCosts[i, :] = 200.0 ## high value for costs for path to all states
        precomputedSPCosts[i, shortestPaths[1].keys()] = np.array(shortestPaths[1].values()) ## actual value for eachable states
    else :
        precomputedSPCosts[i, :] = np.array(shortestPaths[1].values())
    ## lengths
    for j in xrange(len(shortestPaths[0])) :
        curr = j
        path = []
        
        ## no path from start to end
        if curr not in shortestPaths[0] :
            precomputedSPLengths[i, j] = -1
            continue
        
        path.append(curr)
        while curr != i :
            curr = shortestPaths[0][curr]
            path.append(curr)
            
        path = np.array(path)[::-1]
        precomputedSPLengths[i, j] = len(path)-1
        
    sys.stdout.write('\r' + "Computed shortest paths from node " + np.string_(i) + " of " + np.string_(numStates))
    sys.stdout.flush()

# <codecell>

print len(shortestPaths[1]), numStates

# <codecell>

print precomputedSPCosts[i, :]

# <codecell>

i = 400
shortestPaths = shortest_path(gr, i)[0]
shortestPathsLengths = np.zeros(len(shortestPaths))
for j in xrange(len(shortestPaths)) :
    curr = j
    path = []
    
    ## no path from start to end
    if curr not in shortestPaths :
        shortestPathsLengths[j] = -1
        continue
    
    path.append(curr)
    while curr != i :
        curr = shortestPaths[curr]
        path.append(curr)
        
    path = np.array(path)[::-1]
    shortestPathsLengths[j] = len(path)-1

# <codecell>

print shortestPathsLengths[340]
print len(getShortestPath(gr, 400, 340)[0])-1
print precomputedSPLengths

# <codecell>

print precomputedSPLengths[neighbours[449, :], 789]
print precomputedSPCosts[neighbours[449, :], 789]

# <codecell>

# figure(); imshow(precomputedSPCosts, interpolation='nearest')
print getShortestPath(gr, 450, 789)
print 1.1*(len(getShortestPath(gr, 450, 789)[0])-1)-getShortestPath(gr, 450, 789)[1]
print getShortestPath(gr, 601, 789)
print 1.1*(len(getShortestPath(gr, 601, 789)[0])-1)-getShortestPath(gr, 601, 789)[1]
print neighbours[449, :]
print 
currentIdx = 115
goalIdx = 130
for i in xrange(k) :
    print neighbours[currentIdx, i], getShortestPath(gr, neighbours[currentIdx, i], goalIdx)[1], len(getShortestPath(gr, neighbours[currentIdx, i], goalIdx)[0])-1, 1.2*(len(getShortestPath(gr, neighbours[currentIdx, i], goalIdx)[0])-1)-getShortestPath(gr, neighbours[currentIdx, i], goalIdx)[1]

# <codecell>

print getShortestPath(gr, 115, 130)
print getShortestPath(gr, 148, 145)

# <codecell>

## try and sample, see what happens....

taskIdx = 0
labelTaskToPerform = possibleLabelTasks[taskIdx, :]
idxTaskToPerform = 130#possibleIdxTasks[taskIdx]
# taskToPerform = np.array([1, 0, 0, 0])
print "task to perform:", labelTaskToPerform, idxTaskToPerform
startIdx = 115
print "starting from", startIdx, "with task params", labels[startIdx, :]
# extraRewards = np.zeros((numStates, numStates))

rewardsFromTask = (2.0-np.sum(np.abs(labels[0:numStates, :]-np.reshape(possibleLabelTasks[taskIdx, :], (1, len(possibleLabelTasks)), numStates)), axis=1))/2.0
bestFrames = np.ndarray.flatten(np.argwhere(rewardsFromTask > 0.9))#[1:]
bestProbs = rewardsFromTask[bestFrames]/np.sum(rewardsFromTask[bestFrames])

## these are to check for loops
framesSoFar = []
framesSoFar.append(startIdx)
loopDetected = False
idxTaskWhenLoopBroken = -1#np.copy(idxTaskToPerform)

# l = 0.35
currentVFunc = np.sum(np.repeat(labelTaskToPerform.reshape((1, len(labelTaskToPerform))), numStates, axis=0)*v, axis=-1)
totalFrames = 150
currentIdx = startIdx
for i in range(totalFrames) :
    bestA, bestAIdx, immediateRewards, individualImmediateRewards = pi_l(ss[currentIdx], A[currentIdx, :, :], currentVFunc, currentIdx, neighbours, (labelTaskToPerform, idxTaskToPerform), 1.0+t, l, p, (semanticLabelsTask, idxRangeTask), (1.0, 0.0), ss, fs)#, True)
#     print immediateRewards
    if loopDetected :
#         bestProbs = rewardsFromTask[bestFramesPerTask[taskIdx]]/np.sum(rewardsFromTask[bestFramesPerTask[taskIdx]])
        bestProbs = np.ones(len(bestFramesPerTask[taskIdx]))
        bestProbs[np.argwhere(bestFramesPerTask[taskIdx]==idxTaskToPerform)] = 0
        bestProbs /= np.sum(bestProbs)
#         print "gna", bestProbs
        idxTaskToPerform = np.random.choice(bestFramesPerTask[taskIdx], p=bestProbs)#bestFrames, p=bestProbs)
        framesSoFar = []
        print "[loop detected] new goal idx is", idxTaskToPerform
        loopDetected = False
    elif semanticLabelsTask(labels[currentIdx, :], labelTaskToPerform) >= 0.85 :
#         print currentIdx, semanticLabelsTask(labels[currentIdx, :], labelTaskToPerform)
        if idxTaskToPerform != currentIdx :#not in neighbours[currentIdx, :] :
    #         idxRewards = np.zeros(k)
    #         for neighbour, idx in zip(neighbours[currentIdx, :], xrange(k)) :
    #             taskReward = semanticLabelsTask(labels[neighbour, :], labelTaskToPerform)
    #             appearanceReward = r_a(ss[currentIdx], currentIdx, neighbours[currentIdx, :], A[currentIdx, idx, :], 1.0+t, 2.0)
    #             path, pathCost = getShortestPath(gr, neighbour, idxTaskToPerform)
    #             print neighbour, pathCost, np.exp(1/pathCost), idxRewards[idx], taskReward, appearanceReward, l*taskReward + (1.0-l)*appearanceReward
    #             idxRewards[idx] = l*taskReward + (1.0-l)*appearanceReward + np.exp(1/pathCost)
    #         print idxRewards, np.argmax(idxRewards), bestAIdx
    #         print "la", immediateRewards + np.exp(1/precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
            neighbourWeights = np.zeros(k)
            idxCosts = np.zeros(k)
            for neighbourIdx in xrange(k) :
                neighbourWeights[neighbourIdx] = gr.edge_weight((int(currentIdx), int(neighbours[currentIdx, neighbourIdx])))
                idxCosts[neighbourIdx] = (1.0-l)*(1.0-individualImmediateRewards[0, neighbourIdx]) + l*(1.0-individualImmediateRewards[1, neighbourIdx])
#             print neighbourWeights
            idxRewards = immediateRewards + np.exp(1/(neighbourWeights+precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform]))
#             idxRewards = immediateRewards + np.exp(1/(precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform]))
#             idxRewards = immediateRewards + np.exp((neighbourWeights+precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])/5)
#             idxRewards = immediateRewards + 3/(neighbourWeights+precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
#             idxRewards = immediateRewards + (precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform]+1)*1.1 - (neighbourWeights+precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
#             idxRewards = immediateRewards + (precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform])*1.1 - (precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
#             idxRewards = (precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform]+1)*1.1 - (neighbourWeights+precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
#             print immediateRewards
#             print idxCosts
            idxCosts = neighbourWeights + precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform]
#             idxCosts += precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform]
#             idxRewards = immediateRewards + precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform]
#             print precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform]
#             print idxCosts
#             print idxRewards
#             print idxCosts
#             print "idxRewards", currentIdx, idxRewards-immediateRewards
#             print "NEW-noneighs-idxRewards", currentIdx, (precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform])*1.1 - (precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
#             print "NEWidxRewards", currentIdx, (precomputedSPLengths[neighbours[currentIdx, :], idxTaskToPerform]+1)*1.1 - (neighbourWeights+precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
#             print neighbours[currentIdx, :]
#             print idxRewards-immediateRewards
#             print 3/(neighbourWeights+precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
#             print np.exp(-precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform]/4)
#             print immediateRewards
#             bestAIdx = np.argmin(idxCosts)
            bestAIdx = np.argmax(idxRewards)
            bestA = A[currentIdx, bestAIdx, :]
        else :
#             bestProbs = rewardsFromTask[bestFramesPerTask[taskIdx]]/np.sum(rewardsFromTask[bestFramesPerTask[taskIdx]])
            bestProbs = np.ones(len(bestFramesPerTask[taskIdx]))
            bestProbs[np.argwhere(bestFramesPerTask[taskIdx]==idxTaskToPerform)] = 0
            bestProbs /= np.sum(bestProbs)
#             print "gna", bestProbs
            idxTaskToPerform = np.random.choice(bestFramesPerTask[taskIdx], p=bestProbs)#bestFrames, p=bestProbs)
            framesSoFar = []
            print "[goal idx reached] new goal idx is", idxTaskToPerform
    else :
        print "semantic label reward is less than 0.9 for", currentIdx, semanticLabelsTask(labels[currentIdx, :], labelTaskToPerform)
    
    newTaskState, newIdx = integrateTaskState(ss[currentIdx], bestA, 1.0, neighbours[currentIdx, :], ss, fs)
    
    if newIdx in framesSoFar :
        loopDetected = False#True
        
    framesSoFar.append(newIdx)
    
    print newIdx, newTaskState['taskParams'][0]
    currentIdx = np.copy(newIdx)

# <codecell>

print getShortestPath(gr, 121, 190)
print gr.edge_weight((345, 346))
print neighbourIdx, currentIdx
print int(neighbours[currentIdx, neighbourIdx])
print gr.edge_weight((int(currentIdx), int(neighbours[currentIdx, neighbourIdx])))

# <codecell>

print "neighs of 345", neighbours[345, :]
print precomputedSPCosts[neighbours[345, :], 779]
print getShortestPath(gr, 345, 779)
print "neighs of 346", neighbours[346, :]
print precomputedSPCosts[neighbours[346, :], 779]
print getShortestPath(gr, 346, 779)
print getShortestPath(gr, 301, 779)
print
print gr.edge_weight((345, 346))
print gr.edge_weight((346, 301))
print gr.edge_weight((301, 302))
print gr.edge_weight((302, 303))
print gr.edge_weight((303, 778))

# <codecell>

currentIdx = 847
bestA, bestAIdx, immediateRewards = pi_l(ss[currentIdx], A[currentIdx, :, :], currentVFunc, currentIdx, neighbours, (labelTaskToPerform, idxTaskToPerform), 1.0+t, l, p, (semanticLabelsTask, idxRangeTask), (1.0, 0.0), ss, fs)#, True)
print immediateRewards + np.exp(1/precomputedSPCosts[neighbours[currentIdx, :], idxTaskToPerform])
print neighbours[currentIdx, :]

# <codecell>

print neighbours[840, :]
print tmp[neighbours[840, :]]
print tmp[820:920]
print idxRangeTask(841, 907, 1271)
print labels[840, :]
print tmp[841]
print np.exp(-np.floor((np.abs(841-907)+rangeSize/2)/rangeSize)/(1271/(rangeSize*10)))
print np.exp(-np.floor((np.abs(arange(numFrames)-907)+rangeSize/2)/rangeSize)/(numFrames/(rangeSize*10)))[841]
print numFrames, numStates
taskReward = 0.57651873443
appearanceReward = 1.0
print l*taskReward + (1.0-l)*appearanceReward

# <codecell>

goal = 907
tot = np.max((goal, np.abs(goal-numStates)))
rangeSize = 10.0
absDiff = 1.0-np.abs(arange(numStates)-goal)/float(tot)
rangeAbsDiff =1.0-np.floor((np.abs(arange(numStates)-goal)+rangeSize/2.0)/rangeSize)*float(rangeSize/tot)
tmp = np.exp(-np.floor((np.abs(arange(numStates)-907)+rangeSize/2)/rangeSize)/(numStates/(rangeSize*10)))
figure(); plot(xrange(numStates), absDiff, xrange(numStates), rangeAbsDiff, xrange(numStates), tmp)

# <codecell>

# amount, np.sum(actions[idx, :]), taskReward, appearanceReward, taskValueFunction[newIdx], appValueFunction[newIdx], newIdx
idx1 = 208
idx2 = 207
print taskToPerform
print
print v_a[idx1, :]
print v_t[idx1, :]
print labels[idx1, :], (2.0-np.sum(np.abs(labels[idx1, :]-taskToPerform)))/2.0
print neighbours[idx1, :]
print pi_l(s, A[idx1, :, :], np.sum(np.repeat(taskToPerform.reshape((1, len(taskToPerform))), numStates, axis=0)*v_t, axis=-1), v_a[:, task], idx1, fs, neighbours, labels, taskToPerform, 1.0+t, l, p, True)
print
print v_a[idx2, :]
print v_t[idx2, :]
print labels[idx2, :], (2.0-np.sum(np.abs(labels[idx2, :]-taskToPerform)))/2.0
print neighbours[idx2, :]
print pi_l(s, A[idx2, :, :], np.sum(np.repeat(taskToPerform.reshape((1, len(taskToPerform))), numStates, axis=0)*v_t, axis=-1), v_a[:, task], idx2, fs, neighbours, labels, taskToPerform, 1.0+t, l, p, True)
print
print v_a[idx1+1, :]
print v_t[idx1+1, :]
print labels[idx1+1, :], (2.0-np.sum(np.abs(labels[idx1+1, :]-taskToPerform)))/2.0

# <codecell>

eucDist = np.array(np.load(outputData + "vanilla_distMat" + ".npy"), dtype=np.float)
eucDist /= np.max(eucDist)
eucDist = np.copy(vtu.filterDistanceMatrix(np.copy(eucDist), 4, True))
eucT = np.copy(eucDist[1:eucDist.shape[1], 0:-1])
figure(); imshow(eucT, interpolation='nearest')

# <codecell>

print np.max(eucT), np.max(t)

# <codecell>

print neighbours[840, :]
tic =time.time()
for neighbour, i in zip(neighbours[840, :], xrange(k)) :
    taskReward = semanticLabelsTask(labels[neighbour, :], labelTaskToPerform)
    appearanceReward = r_a(ss[840], 840, neighbours[840, :], A[840, i, :], 1.0+t, 2.0)
    path, pathCost = getShortestPath(gr, neighbour, 906)
    print taskReward, appearanceReward, l*taskReward + (1.0-l)*appearanceReward
    print path, pathCost, np.exp(1/pathCost)
    print "la", l*taskReward + (1.0-l)*appearanceReward + np.exp(1/pathCost)
print time.time()-tic

# <codecell>

print gr.edge_weight((841, 842))

# <codecell>

def getShortestPath(graph, start, end) :
    paths = shortest_path(graph, start)[0]
#     print paths
    curr = end
    path = []
    
    ## no path from start to end
    if curr not in paths :
        return np.array(path), -1
    
    path.append(curr)
    while curr != start :
        curr = paths[curr]
        path.append(curr)
        
    path = np.array(path)[::-1]
    distance = 0
    for i in xrange(1, len(path)) :
        distance += graph.edge_weight((path[i-1], path[i]))
    return path, distance

print np.ndarray.flatten(getShortestPath(gr, 0, 1200)[0])
frameQueue = list(np.ndarray.flatten(getShortestPath(gr, 0, 1200)[0]))
print frameQueue.pop(0)
print frameQueue.pop(0)
# print list(np.ndarray.flatten(getShortestPath(gr, currentIdx, random)[0]))

# <codecell>

## use shortest path to get to traverse graph
startIdx = 835
totalFrames = 100
currentIdx = startIdx
taskIdx = 2

bestFramesPerTask = []
for i in xrange(len(possibleLabelTasks)) :
    bestFramesPerTask.append(np.ndarray.flatten(np.argwhere(labels[:, i] > 0.9)))
    
# print bestFramesPerTask
frameQueue = []
frameQueue.append(currentIdx)

for i in range(totalFrames) :
    while len(frameQueue) == 0 :
        randomChoice = random.choice(bestFramesPerTask[taskIdx])
        print "going to", randomChoice
        frameQueue = list(np.ndarray.flatten(getShortestPath(gr, currentIdx, randomChoice)[0][1:]))
        
    currentIdx = frameQueue.pop(0)
    print currentIdx

# print random.choice(bestFramesPerTask[taskIdx])

# <codecell>

possibleTasksToPerform = np.eye(4)

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
        
        self.setWindowTitle("Video Fields Visualizer")
        self.resize(1280, 720)
        
        self.textureTimer = QtCore.QTimer(self)
        self.textureTimer.setInterval(1000/30)
        self.textureTimer.start()
        self.textureTimer.timeout.connect(self.renderOneFrame)
        
        self.taskIdx = 0
        self.labelTaskToPerform = possibleLabelTasks[self.taskIdx, :]
        
        rewardsFromTask = (2.0-np.sum(np.abs(labels[0:numStates, :]-np.reshape(self.labelTaskToPerform, (1, len(possibleLabelTasks)), numStates)), axis=1))/2.0
        self.bestFrames = np.ndarray.flatten(np.argwhere(rewardsFromTask > 0.9))#[1:]
        self.bestProbs = rewardsFromTask[self.bestFrames]/np.sum(rewardsFromTask[self.bestFrames])
        print "task set to", self.taskIdx, self.bestFrames
        
#         self.taskToPerform = np.array([0, 0.5, 0.0, 0.5])
        self.frameRenderedBeforeIdx = 0
        self.currentIdx = 115
        self.renderedFrames = []; self.renderedFrames.append(self.currentIdx)
        self.frameQueue = [self.currentIdx]
        self.idxTaskToPerform = 130#np.random.choice(self.bestFrames, p=self.bestProbs)
        print "goal idx is", self.idxTaskToPerform
        
        
        ## these are to check for loops
        self.framesSoFar = []
        self.framesSoFar.append(self.currentIdx)
        self.loopDetected = False
        
        
        ## HACK ##
        im = np.ascontiguousarray(Image.open(frames[self.currentIdx+filterSize]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
        infoText = "Frame idx: "+np.string_(self.currentIdx)+"\t"+np.string_(labels[self.currentIdx, :])
        self.infoLabel.setText(infoText)
        print self.currentIdx, labels[self.currentIdx, :]
        
        self.labelJumpStyle = "QLabel {border: 1px solid black; background: #aa0000; color: white; padding-left: 5px; padding-right: 5px;}"
        self.labelNoJumpStyle = "QLabel {border: 1px solid gray; background: #eeeeee; color: black; padding-left: 5px; padding-right: 5px;}"
        self.infoLabel.setStyleSheet(self.labelNoJumpStyle)
        
    def changeTaskLabel(self, index) :
        self.taskIdx = index
        self.labelTaskToPerform = possibleLabelTasks[self.taskIdx, :]
        
        rewardsFromTask = (2.0-np.sum(np.abs(labels[0:numStates, :]-np.reshape(self.labelTaskToPerform, (1, len(possibleLabelTasks)), numStates)), axis=1))/2.0
        self.bestFrames = np.ndarray.flatten(np.argwhere(rewardsFromTask > 0.9))#[1:]
        self.bestProbs = rewardsFromTask[self.bestFrames]/np.sum(rewardsFromTask[self.bestFrames])
        self.idxTaskToPerform = np.random.choice(self.bestFrames, p=self.bestProbs)
        print "changed task to", self.taskIdx, self.bestFrames
        print "new goal idx is", self.idxTaskToPerform
        
    def renderOneFrame(self) :
        
        s = {'frameState': fs[self.currentIdx, :, :], 'taskParams': labels[self.currentIdx, :]}
        neighsProbs = np.zeros(k)#probs[self.currentIdx, neighbours[self.currentIdx, :]]/np.sum(probs[self.currentIdx, neighbours[self.currentIdx, :]])
#         bestA, bestAIdx = pi_l(s, A[self.currentIdx, :, :], v[:, self.taskIdx], self.currentIdx, fs, neighbours, labels, self.taskToPerform)
#         bestA, bestAIdx = pi_l(s, A[self.currentIdx, :, :], np.sum(np.repeat(self.taskToPerform.reshape((1, len(self.taskToPerform))), numStates, axis=0)*v, axis=-1), self.currentIdx, fs, neighbours, labels, self.taskToPerform, neighsProbs, w[self.currentIdx, :])
#         bestA, bestAIdx = pi_l(s, A[self.currentIdx, :, :], np.sum(np.repeat(self.taskToPerform.reshape((1, len(self.taskToPerform))), numStates, axis=0)*v, axis=-1), self.currentIdx, fs, neighbours, labels, self.taskToPerform, 1.0+t, l, p, 0.1)
        bestA, bestAIdx, immediateRewards, individualImmediateRewards = pi_l(ss[self.currentIdx], A[self.currentIdx, :, :], np.sum(np.repeat(self.labelTaskToPerform.reshape((1, len(self.labelTaskToPerform))), numStates, axis=0)*v, axis=-1), self.currentIdx, neighbours, (self.labelTaskToPerform, self.idxTaskToPerform), 1.0+t, l, p, (semanticLabelsTask, idxRangeTask), (1.0, 0.0), ss, fs)#, True)
        
        if self.loopDetected :
            bestProbs = np.ones(len(bestFramesPerTask[self.taskIdx]))
            bestProbs[np.argwhere(bestFramesPerTask[self.taskIdx]==self.idxTaskToPerform)] = 0
            bestProbs /= np.sum(bestProbs)
            self.idxTaskToPerform = np.random.choice(bestFramesPerTask[self.taskIdx], p=bestProbs)
#             self.idxTaskToPerform = np.random.choice(self.bestFrames, p=self.bestProbs)
            self.framesSoFar = []
            print "[loop detected] new goal idx is", self.idxTaskToPerform
            self.loopDetected = False
        elif semanticLabelsTask(labels[self.currentIdx, :], self.labelTaskToPerform) >= 0.9 :
            ## randomly try to reach a certain frame
            if self.idxTaskToPerform != self.currentIdx :# not in neighbours[self.currentIdx, :] :
#                 idxRewards = np.zeros(k)
#                 for neighbour, i in zip(neighbours[self.currentIdx, :], xrange(k)) :
#                     taskReward = semanticLabelsTask(labels[neighbour, :], self.labelTaskToPerform)
#                     appearanceReward = r_a(ss[self.currentIdx], self.currentIdx, neighbours[self.currentIdx, :], A[self.currentIdx, i, :], 1.0+t, 2.0)
#                     path, pathCost = getShortestPath(gr, neighbour, self.idxTaskToPerform)
#                     if pathCost > 0 :
#                         idxRewards[i] = l*taskReward + (1.0-l)*appearanceReward + np.exp(1/pathCost)
#     #                     print neighbour, pathCost, np.exp(1/pathCost), idxRewards[idx], taskReward, appearanceReward
#                     else :
#                         idxRewards[i] = 0.0
#     #                     print "zero path cost", neighbour, pathCost, idxRewards[idx], taskReward, appearanceReward
#     #             print idxRewards, np.argmax(idxRewards), bestAIdx

#                 idxRewards = immediateRewards + np.exp(1/precomputedSPCosts[neighbours[self.currentIdx, :], self.idxTaskToPerform])

                neighbourWeights = np.zeros(k)
                idxCosts = np.zeros(k)
                for neighbourIdx in xrange(k) :
                    neighbourWeights[neighbourIdx] = gr.edge_weight((int(self.currentIdx), int(neighbours[self.currentIdx, neighbourIdx])))
                    idxCosts[neighbourIdx] = (1.0-l)*(1.0-individualImmediateRewards[0, neighbourIdx]) + l*(1.0-individualImmediateRewards[1, neighbourIdx])
#                 idxRewards = immediateRewards + (precomputedSPLengths[neighbours[self.currentIdx, :], self.idxTaskToPerform]+1)*1.1 - (neighbourWeights+precomputedSPCosts[neighbours[self.currentIdx, :], self.idxTaskToPerform])
                idxRewards = immediateRewards + np.exp(1/(neighbourWeights+precomputedSPCosts[neighbours[self.currentIdx, :], self.idxTaskToPerform]))
                
                idxCosts = neighbourWeights + precomputedSPCosts[neighbours[self.currentIdx, :], self.idxTaskToPerform]
#                 print idxRewards
#                 bestAIdx = np.argmin(idxCosts)
                bestAIdx = np.argmax(idxRewards)
                bestA = A[self.currentIdx, bestAIdx, :]
            else :
                bestProbs = np.ones(len(bestFramesPerTask[self.taskIdx]))
                bestProbs[np.argwhere(bestFramesPerTask[self.taskIdx]==self.idxTaskToPerform)] = 0
                bestProbs /= np.sum(bestProbs)
                self.idxTaskToPerform = np.random.choice(bestFramesPerTask[self.taskIdx], p=bestProbs)
#                 self.idxTaskToPerform = np.random.choice(self.bestFrames, p=self.bestProbs)
                self.framesSoFar = []
                print "[goal idx reached] new goal idx is", self.idxTaskToPerform
        else :
            print "semantic label reward is less than 0.9 for", self.currentIdx, semanticLabelsTask(labels[self.currentIdx, :], self.labelTaskToPerform),
        
#         newTaskState, newIdx = integrateTaskState(s, bestA, fs[neighbours[self.currentIdx, :], :, :], 1.0, neighbours[self.currentIdx, :], labels)
        newTaskState, newIdx = integrateTaskState(ss[self.currentIdx], bestA, 1.0, neighbours[self.currentIdx, :], ss, fs)
        
            
        if newIdx in self.framesSoFar :
            self.loopDetected = True
        self.framesSoFar.append(newIdx)
        
        isJump = self.currentIdx != newIdx-1
        self.currentIdx = np.copy(newIdx)
        self.renderedFrames.append(self.currentIdx)
        
        print self.currentIdx
  
        ## randomly p[ick a frame and pick shortest path to it
#         while len(self.frameQueue) == 0 :
#             randomChoice = random.choice(bestFramesPerTask[self.taskIdx])
#             print "going to", randomChoice
#             self.frameQueue = list(np.ndarray.flatten(getShortestPath(gr, self.currentIdx, randomChoice)[0][1:]))
        
#         isJump = self.currentIdx != self.frameQueue[0]-1
#         self.currentIdx = self.frameQueue.pop(0)
        
#         self.currentIdx = totalFramesRenderedBefore[np.mod(self.frameRenderedBeforeIdx, len(totalFramesRenderedBefore))]
#         isJump = self.currentIdx != totalFramesRenderedBefore[np.mod(self.frameRenderedBeforeIdx-1, len(totalFramesRenderedBefore))]
#         self.frameRenderedBeforeIdx += 1
        
        ## HACK ##
        im = np.ascontiguousarray(Image.open(frames[self.currentIdx+filterSize]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
        infoText = "Frame idx: "+np.string_(self.currentIdx)+"\t"+np.string_(labels[self.currentIdx, :])
        if isJump :
            self.infoLabel.setStyleSheet(self.labelJumpStyle)
        else :
            self.infoLabel.setStyleSheet(self.labelNoJumpStyle)
        self.infoLabel.setText(infoText)
#         print newIdx, newTaskState['taskParams']
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.infoLabel = QtGui.QLabel("Info text")
        
        self.labelComboBox = QtGui.QComboBox()
        self.labelComboBox.addItem("Away")
        self.labelComboBox.addItem("Right")
        self.labelComboBox.addItem("Towards")
        self.labelComboBox.addItem("Left")
        
        ## SIGNALS ##
        
        self.labelComboBox.currentIndexChanged[int].connect(self.changeTaskLabel)
#         self.openSequenceButton.clicked.connect(self.openSequence)
        
#         self.tabWidget.currentChanged.connect(self.tabChanged)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.frameLabel)
        mainLayout.addWidget(self.infoLabel)
        mainLayout.addWidget(self.labelComboBox)
        self.setLayout(mainLayout)

# <codecell>

currentIdx = 537
labelTaskToPerform = np.array([0, 0, 0, 1], dtype=float)
idxTaskToPerform = 1228
bestA, bestAIdx, immediateRewards, individualImmediateRewards = pi_l(ss[currentIdx], A[currentIdx, :, :], np.sum(np.repeat(labelTaskToPerform.reshape((1, len(labelTaskToPerform))), numStates, axis=0)*v, axis=-1), currentIdx, neighbours, (labelTaskToPerform, idxTaskToPerform), 1.0+t, l, p, (semanticLabelsTask, idxRangeTask), (1.0, 0.0), ss, fs)#, True)

print l
print immediateRewards
print individualImmediateRewards
print v[neighbours[currentIdx, :], 3]
print neighbours[currentIdx, :]
print bestAIdx, bestA

# <codecell>

bob = 1126
print distMat[901, 1126], np.max(distMat[901, :])
print distanceMatrix[901, 1126], np.max(distanceMatrix[901, :])
print neighbours[bob, :]
print v[neighbours[bob, :], 2]

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

print window.currentIdx, window.labelTaskToPerform, window.idxTaskToPerform
print semanticLabelsTask(labels[852, :], window.labelTaskToPerform)
print np.array(window.renderedFrames)[-500:]
currentIdx = 882
print window.idxTaskToPerform not in neighbours[currentIdx, :]
bestA, bestAIdx, immediateRewards = pi_l(ss[currentIdx], A[currentIdx, :, :], np.sum(np.repeat(window.labelTaskToPerform.reshape((1, len(window.labelTaskToPerform))), numStates, axis=0)*v, axis=-1), currentIdx, neighbours, (window.labelTaskToPerform, window.idxTaskToPerform), 1.0+t, l, p, (semanticLabelsTask, idxRangeTask), (1.0, 0.0), ss, fs)#, True)
print immediateRewards + np.exp(1/precomputedSPCosts[neighbours[currentIdx, :], window.idxTaskToPerform])
print neighbours[currentIdx, :]
print window.framesSoFar

# <codecell>

# totalFramesRenderedBefore = np.copy(np.ndarray.flatten(np.array(window.renderedFrames)))
print len(totalFramesRenderedBefore)
print np.mod(-1, len(totalFramesRenderedBefore))

