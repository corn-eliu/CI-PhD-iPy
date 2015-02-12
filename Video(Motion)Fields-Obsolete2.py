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

dataFolder = "/home/ilisescu/PhD/data/"
POSE = 0
VELOCITY = 1
FUT_VELOCITY = 2

app = QtGui.QApplication(sys.argv)

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

# <codecell>

## compute features for image
blocksPerWidth = 4#16#32
blocksPerHeight = 4#16#48
subDivisions = blocksPerWidth*blocksPerHeight

## given block sizes and img sizes build indices representing each block
imageSize = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)).shape
stencils = cgf.stencil2D(blocksPerWidth, blocksPerHeight, imageSize)

features = cgf.histFgFeatures(stencils, subDivisions, frames, mattes)
figure(); imshow(features.T, interpolation='nearest')

# <codecell>

## video field data
print "fg histogram features: ", features.shape
numStates = features.shape[0]-2
descriptorSize = features.shape[1]
## fs contains all the fs, i.e. frame states
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
distanceMatrix = np.array(np.load(outputData + "hist2demd_16x16_distMat" + ".npy"), dtype=np.float)
distanceMatrix /= np.max(distanceMatrix)
distMat = vtu.filterDistanceMatrix(distanceMatrix, 4, True)
# figure(); imshow(distMat, interpolation='nearest')
d = np.copy(distMat)
## troncate fs to match indices in d to indices in fs
fs = np.copy(fs[4:numStates-2, :, :])
print fs.shape[0], "frame states(lost", numStates-fs.shape[0], "because I had to filter 8 frames out for ",
print "dynamicism preservation),", "features size is", fs.shape[1]
numStates = fs.shape[0]

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
    
dFut = estimateFutureCost(0.999, 2.0, np.copy(d), np.ones(d.shape))

# <codecell>

## normalize dFut so it goes from 0 to 1 to get the extraRewards
extraRewards = np.copy(dFut)
mins = np.repeat(np.reshape(np.min(extraRewards, axis=-1), (extraRewards.shape[0], 1)), extraRewards.shape[0], axis=-1)
diffs = np.repeat(np.reshape(np.max(extraRewards, axis=-1) - np.min(extraRewards, axis=-1), (extraRewards.shape[0], 1)), extraRewards.shape[0], axis=-1)
extraRewards = -(extraRewards - mins)/diffs
## remove data about last frame since I wouldn't have info about it in extraRewards
d = np.copy(d[0:-1, 0:-1])
## copy filtered d to dFilt
dFilt = np.copy(d)
## troncate fs to match indices in d to indices in fs
fs = np.copy(fs[0:-1, :, :])
print fs.shape[0], "frame states(lost", numStates-fs.shape[0], "because future cost estimation does not",
print "have info about last frame),", "features size is", fs.shape[1]
numStates = fs.shape[0]

# <codecell>

d = np.copy(dFut)
# d = np.copy(dFilt)

# <codecell>

## find k-nearest neighbors based on dissimilarity matrix
k = 15
neighbours = np.argsort(d+(np.eye(d.shape[0])*np.max(d)+1))[:, 0:k] #np.argsort(d)[:, 0:k] # we don't want ourselves as neighbour
# figure(); imshow(neighbours)

# w are similarity weights
w = 1/(d[np.reshape(range(d.shape[0]), (d.shape[0], 1)).repeat(k, axis=-1), neighbours]**2)#(np.sort(d)[:, 1:k+1]**2)
# normalize to make weights sum to 1
w /= np.repeat(np.reshape(np.sum(w, axis=-1), (numStates, 1)), k, axis=-1)

# <codecell>

print neighbours[234, :]
print w[234, :], np.sum(w[234, :])

# <codecell>

print neighbours[234, :]
print w[234, :]

# <codecell>

# figure();imshow(d*(np.eye(d.shape[0])*np.max(d)+1), interpolation='nearest')
print np.argsort(d+(np.eye(d.shape[0])*np.max(d)+1))[:, 0:k][106, :]
print neighbours[106, :]
print w[106, :]
print A[106, 0, :]
print np.concatenate(([1], w[106, 1:]))/np.sum(np.concatenate(([1], w[106, 1:])))
print A[106, 1, :]

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

def integrateTaskState(taskState, action, neighbours, delta, neighboursIndices, taskParamsValues) :
    """Computes a new task state with a newly integrated frame state and update task parameters.
    
           taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
           action: 1xK array of weights
           neighbours: 1xK frame states
           delta: tugging parameter in [0, 1]
           neighboursIndices: contains the indices in fs of the given neighbours
           taskParamsValues: contains values of task parameters for the full set of frame states
           
        return: newTaskState"""
    newFrameState, newIdx = integrateFrameState(taskState['frameState'], action, neighbours, delta, neighboursIndices)
    newTaskState = {'frameState':newFrameState, 'taskParams':taskParamsValues[newIdx, :]}
    
    return newTaskState, newIdx
    

# <codecell>

## test task state integration function
a = np.zeros(k)
a[k/2] = 1
fIdx = 300
f = fs[fIdx, :, :]
ns = fs[neighbours[fIdx, :], :, :]
labels = np.load(outputData + "hist2demd_mult0.02_16x16_labels.npy")[()]['labelProbs']
newT, newIdx = integrateTaskState({'frameState': f, 'taskParams': labels[fIdx, :]}, a, ns, 1.0, neighbours[fIdx, :], labels)
print newT, newIdx
print fs[newIdx, :, :]
print labels[newIdx, :]

# <codecell>

def reward_old(taskState, taskToPerform, transitionProb):#action) :
    """Computes the reward of being at taskState given the taskToPerform (taskState has been reached by performing some action already).
            
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            taskToPerform: task that needs to be performed
        
        return: reward r
        
        NOTE: sligtly different definition than paper because each taskState saves the state of a potential task which then gets compared to the desired one, rather than computing that state on the fly when the taskState is chosen.
        Also I compute the reward after I pick a potential next state in pi_l so I just need to evaluate that potential state."""
    
#     r = transitionProb + (2.0-np.sum(np.abs(taskState['taskParams']-taskToPerform)))
    r = (2.0-np.sum(np.abs(taskState['taskParams']-taskToPerform)))
#     r = -np.sum(np.abs(taskState['taskParams']-taskToPerform))
    return r

def reward_weighted(taskState, neighboursTaskParamsValues, action, taskToPerform) :
    """Computes the reward of performing action at taskState given the taskToPerform
            
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            neighboursTaskParamsValues: kxN array containing the N task labels of the k neighbours of taskState
            action: action performed at taskState
            taskToPerform: task that needs to be performed
        
        return: reward r"""
    
    r = 2.0-np.sum(np.abs(neighboursTaskParamsValues-np.reshape(taskToPerform, (1, len(taskToPerform))).repeat(len(neighboursTaskParamsValues), axis=0)), axis = -1)
    r = np.sum(action*r)
#     r = (2.0-np.sum(np.abs(neighboursTaskParamsValues[np.argmax(action)]-taskToPerform)))
#     r = 0
#     for i in range(len(action)) :
#         r += action[i]*(2.0-np.sum(np.abs(neighboursTaskParamsValues[i, :]-taskToPerform)))
    return r

def reward(taskState, neighboursTaskParamsValues, action, taskToPerform, weight) :
    """Computes the reward of performing action at taskState given the taskToPerform
            
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            neighboursTaskParamsValues: kxN array containing the N task labels of the k neighbours of taskState
            action: action performed at taskState
            taskToPerform: task that needs to be performed
            weight: similarity weights of neighbour favoured by action to taskState
        
        return: reward r"""
    
#     r = 2.0-np.sum(np.abs(neighboursTaskParamsValues-np.reshape(taskToPerform, (1, len(taskToPerform))).repeat(len(neighboursTaskParamsValues), axis=0)), axis = -1)
#     r = np.sum(action*r)
    l = 0.8
    
    r = (1.0-l)*weight + l*((2.0-np.sum(np.abs(neighboursTaskParamsValues[np.argmax(action)]-taskToPerform)))/2.0)
#     r = 0
#     for i in range(len(action)) :
#         r += action[i]*(2.0-np.sum(np.abs(neighboursTaskParamsValues[i, :]-taskToPerform)))
    return r
    
def pi_l(taskState, actions, valueFunction, stateIdx, states, neighboursIndices, taskParamsValues, taskToPerform, transitionProbs, weights, verbose=False):#, extraInfo) :
    """Gives the next best action based on a lookahead policy to maximize reward.
    
            taskState: dict {'frameState':a frame state, 'taskParams': theta task parameters}
            actions: possible actions from state KxK matrix where each row is an action
            valueFunction: precomputed value function that represents future reward
            stateIdx: is the idx of state in states
            states: list of all possible frame states
            neighboursIndices: list of neighbour states indices for each state
            taskParamsValues: contains values of task parameters for the full set of frame states
            taskToPerform: task that needs to be performed
            extraInfo: NxN matrix (N = numStates) extra rewards to add to reward function based on current and next frame
            weights: similarity weights of taskState to its neighbours
        
        return: bestAction"""
    currentMax = -sys.float_info.max
    if verbose :
        print "weights", weights
    for idx in range(len(actions)) :
        newTaskState, newIdx = integrateTaskState(taskState, actions[idx, :], states[neighboursIndices[stateIdx, :], :, :], 1.0, neighboursIndices[stateIdx, :], taskParamsValues)
#         amount = (extraInfo[stateIdx, newIdx])*(reward(newTaskState, taskToPerform) + valueFunction[newIdx])
#         print "amount", amount, "extra", extraRewards[stateIdx, newIdx]
        bob = 0.0 #10.0 if newIdx-stateIdx == 1 else 0.0
#         amount = reward(newTaskState, taskToPerform, transitionProbs[idx]) + valueFunction[newIdx] + bob# + transitionProbs[idx]
        amount = reward(taskState, taskParamsValues[neighboursIndices[stateIdx, :], :], actions[idx, :], taskToPerform, weights[idx]) + valueFunction[newIdx]
        if verbose :
            print "current amount for action in pi_l", amount, np.sum(actions[idx, :]), reward(taskState, taskParamsValues[neighboursIndices[stateIdx, :], :], actions[idx, :], taskToPerform, weights[idx]), valueFunction[newIdx], newIdx
#         print actions[idx, :]
        if amount > currentMax :
            currentMax = amount
            bestIdx = idx
    bestAction = np.copy(actions[bestIdx, :])
    return bestAction, bestIdx

# <codecell>

## this is to test that the new implementation does something plausible
taskState = {'frameState': fs[trackingState, :, :], 'taskParams': labels[trackingState, :]}
neighboursTaskParamsValues = labels[neighbours[trackingState, :], :]
action = A[trackingState, 4, :]
taskToPerform = possibleTasksToPerform[3, :]
# v = np.zeros((fs.shape[0], possibleTasksToPerform.shape[0]))

print "neighbours", neighbours[trackingState, :]
print "neighbours' labels", neighboursTaskParamsValues
print "chosen action", action, np.sum(action)
print "task to perform", taskToPerform
print "rewards for given neighbours", 2.0-np.sum(np.abs(neighboursTaskParamsValues-np.reshape(taskToPerform, (1, 4)).repeat(k, axis=0)), axis = -1)

print "total reward computed by method", reward(taskState, neighboursTaskParamsValues, action, taskToPerform)

tmp = 2.0-np.sum(np.abs(neighboursTaskParamsValues-np.reshape(taskToPerform, (1, 4)).repeat(k, axis=0)), axis = -1)
print "total reward based on normalized rewards for neighbours by action weights", np.sum(action*tmp)
print
print

bestA, bestIdx = pi_l(taskState, A[trackingState, :, :], v[:, t], trackingState, fs, neighbours, labels, taskToPerform, neighsProbs)#extraRewards)
print "best A and idx", bestA, bestIdx

# <codecell>

############# COMPUTE VALUE FUNCTION V ###############

# load labels to use as taskParamsValues
labels = np.load(outputData + "hist2demd_mult0.02_16x16_labels.npy")[()]['labelProbs']
possibleTasksToPerform = np.eye(labels.shape[-1])

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

## init v to zeros
# v = np.load("valuefunction.npy")
v = np.zeros((fs.shape[0], possibleTasksToPerform.shape[0]))
vTmp = np.copy(v)
prevAdditionalReward = 0

maxIterations = 20
trackingState = 300
print "tracking progression of state", trackingState
print "neighbours", neighbours[trackingState, :]
for i in range(maxIterations) :
    ## for each possible task
    print "best actions",
    for t in range(possibleTasksToPerform.shape[0]) :
        ## for each frame state
        for fIdx in range(numStates) :
            si = {'frameState': fs[fIdx, :, :], 'taskParams': labels[fIdx, :]}
            neighsProbs = (1.0/(np.abs(neighbours[fIdx, :]-fIdx-1.0)+1.0))/np.sum(1.0/(np.abs(neighbours[fIdx, :]-fIdx-1.0)+1.0))#probs[fIdx, neighbours[fIdx, :]]/np.sum(probs[fIdx, neighbours[fIdx, :]])
            bestA, bestIdx = pi_l(si, A[fIdx, :, :], v[:, t], fIdx, fs, neighbours, labels, possibleTasksToPerform[t, :], neighsProbs, w[fIdx, :])#extraRewards)
            
            newTaskState, newIdx = integrateTaskState(si, bestA, fs[neighbours[fIdx, :], :, :], 1.0, neighbours[fIdx, :], labels)
#             v[fIdx, t] = reward(newTaskState, possibleTasksToPerform[t, :]) + v[newIdx, t]
#             vTmp[fIdx, t] = reward(newTaskState, possibleTasksToPerform[t, :], neighsProbs[np.where(neighbours[fIdx, :] == newIdx)]) + v[newIdx, t]
            vTmp[fIdx, t] = reward(si, labels[neighbours[fIdx, :], :], bestA, possibleTasksToPerform[t, :], w[fIdx, bestIdx]) + v[newIdx, t]
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

# <codecell>

print labels[300, :]
print v[neighbours[300, :], :]
print labels[326, :]
print labels[575, :]
print labels[327, :]
print labels[58, :]
print v[575, :]
print neighbours[58, :]
print v[neighbours[58, :], :]
print neighbours[228, :]
print v[neighbours[228, :], :]
print neighbours[111, :]
print v[neighbours[111, :], :]
print neighbours[501, :]
print v[neighbours[501, :], :]
print neighbours[502, :]
print v[neighbours[502, :], :]
print neighbours[503, :]
print v[neighbours[503, :], :]

# <codecell>

print fIdx, newIdx
print neighsProbs
print neighbours[fIdx, :]
print np.where(neighbours[fIdx, :] == newIdx)
print (1.0/(np.abs(neighbours[fIdx, :]-fIdx-1.0)+1.0))/np.sum(1.0/(np.abs(neighbours[fIdx, :]-fIdx-1.0)+1.0))
print probs[fIdx, neighbours[fIdx, :]]/np.sum(probs[fIdx, neighbours[fIdx, :]])

# <codecell>

## plot v
paperV = False
fullV = True
# if paperV :
#     v = np.load("valuefunction_paper.npy")
# else :
#     if fullV :
#         v = np.load("valuefunction_mine_full.npy")
#     else :
#         v = np.load("valuefunction_mine.npy")
    
figure(); plot(range(numStates), v[:, 0], 'r', range(numStates), v[:, 1], 'g', range(numStates), v[:, 2], 'b', range(numStates), v[:, 3], 'm')

# if paperV :
#     np.save("valuefunction_paper.npy", v)
# else :
#     if fullV :
#         np.save("valuefunction_mine_full.npy", v)
# #         np.save("valuefunction_mine_extra.npy", v)
#     else :
#         np.save("valuefunction_mine.npy", v)

# <codecell>

 ## try and sample, see what happens....

taskIdx = 2
taskToPerform = possibleTasksToPerform[taskIdx, :]
# taskToPerform = np.array([1, 0, 0, 0])
print "task to perform:", taskToPerform
startIdx = 234
print "starting from", startIdx, "with task params", labels[startIdx, :]
# extraRewards = np.zeros((numStates, numStates))

totalFrames = 100
currentIdx = startIdx
for i in range(totalFrames) :
    s = {'frameState': fs[currentIdx, :, :], 'taskParams': labels[currentIdx, :]}
    neighsProbs = (1.0/(np.abs(neighbours[fIdx, :]-fIdx-1.0)+1.0))/np.sum(1.0/(np.abs(neighbours[fIdx, :]-fIdx-1.0)+1.0))#probs[currentIdx, neighbours[currentIdx, :]]/np.sum(probs[currentIdx, neighbours[currentIdx, :]])
    bestA, bestAIdx = pi_l(s, A[currentIdx, :, :], np.sum(np.repeat(taskToPerform.reshape((1, len(taskToPerform))), numStates, axis=0)*v, axis=-1), currentIdx, fs, neighbours, labels, taskToPerform, neighsProbs, w[currentIdx, :])
#     print bestA
#     print neighbours[currentIdx, :], neighbours[currentIdx, np.argmax(probs[currentIdx, neighbours[currentIdx, :]])]
#     print probs[currentIdx, neighbours[currentIdx, :]]
#     print probs[currentIdx, neighbours[currentIdx, :]]*bestA, np.argmax(probs[currentIdx, neighbours[currentIdx, :]]*bestA), neighbours[currentIdx, np.argmax(probs[currentIdx, neighbours[currentIdx, :]]*bestA)]
#     tmp = probs[currentIdx, neighbours[currentIdx, :]]
#     tmp /= np.sum(tmp)
#     bob = 0.8
#     newAction = ((1.0-bob)*tmp) * (bob*bestA)
#     newAction /= np.sum(newAction)
    newTaskState, newIdx = integrateTaskState(s, bestA, fs[neighbours[currentIdx, :], :, :], 1.0, neighbours[currentIdx, :], labels)
#     newTaskState, newIdx = integrateTaskState(s, newAction, fs[neighbours[currentIdx, :], :, :], 1.0, neighbours[currentIdx, :], labels)
    print newIdx, newTaskState['taskParams']
    currentIdx = np.copy(newIdx)
#     print newTaskState

# <codecell>

print np.argsort(v[:, 1])[-30:]

# <codecell>

idx1 = 848
idx2 = 847
print taskToPerform
print
print v[idx1, :]
print labels[idx1, :], (2.0-np.sum(np.abs(labels[idx1, :]-taskToPerform)))/2.0
print neighbours[idx1, :]
print pi_l(s, A[idx1, :, :], np.sum(np.repeat(taskToPerform.reshape((1, len(taskToPerform))), numStates, axis=0)*v, axis=-1), idx1, fs, neighbours, labels, taskToPerform, neighsProbs, w[idx1, :], True)
print
print v[idx2, :]
print labels[idx2, :], (2.0-np.sum(np.abs(labels[idx2, :]-taskToPerform)))/2.0
print neighbours[idx2, :]
print pi_l(s, A[idx2, :, :], np.sum(np.repeat(taskToPerform.reshape((1, len(taskToPerform))), numStates, axis=0)*v, axis=-1), idx2, fs, neighbours, labels, taskToPerform, neighsProbs, w[idx2, :], True)
print
print v[idx1+1, :]
print labels[idx1+1, :], (2.0-np.sum(np.abs(labels[idx1+1, :]-taskToPerform)))/2.0

# <codecell>

## distance of neighbours' labels to task to perform
print taskToPerform
print
print 2.0-np.sum(np.abs(labels[neighbours[106, :], :]-np.reshape(taskToPerform, (1, len(taskToPerform))).repeat(len(labels[neighbours[106, :], :]), axis=0)), axis = -1)
print neighbours[106, :]
print labels[neighbours[106, :], :]
print
print 2.0-np.sum(np.abs(labels[neighbours[122, :], :]-np.reshape(taskToPerform, (1, len(taskToPerform))).repeat(len(labels[neighbours[122, :], :]), axis=0)), axis = -1)
print neighbours[122, :]
print labels[neighbours[122, :], :]

# <codecell>

print "neighbours", neighbours[startIdx, :]
print "v values", np.sum(np.repeat(taskToPerform.reshape((1, len(taskToPerform))), numStates, axis=0)*v, axis=-1)[neighbours[startIdx, :]]
print "rewards", -np.sum(np.abs(labels[neighbours[startIdx, :]]-taskToPerform), axis=-1)
print "labels", labels[neighbours[startIdx, :], 2]
print "extra r", dFut[startIdx, neighbours[startIdx]], np.min(dFut), np.max(dFut)
print "probs", probs[startIdx, neighbours[startIdx]]

# <codecell>

def dist2prob(dM, sigmaMult, normalize) :
    sigma = sigmaMult*np.mean(dM[np.nonzero(dM)])
    print 'sigma', sigma
    pM = np.exp((-dM)/sigma)
## normalize probabilities row-wise
    if normalize :
        normTerm = np.sum(pM, axis=1)
        normTerm = cv2.repeat(normTerm, 1, dM.shape[1])
        pM = pM / normTerm
    return pM

probs = dist2prob(dFut, 0.01, True)
figure(); imshow(probs)

figure(); imshow(np.cumsum(probs, axis=1))

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
        self.taskToPerform = possibleTasksToPerform[self.taskIdx, :]
#         self.taskToPerform = np.array([0, 0.5, 0.0, 0.5])
        self.currentIdx = 570
        
        ## HACK ##
        im = np.ascontiguousarray(Image.open(frames[self.currentIdx+4]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
        infoText = "Frame idx: "+np.string_(self.currentIdx)+"\t"+np.string_(labels[self.currentIdx, :])
        self.infoLabel.setText(infoText)
        print self.currentIdx, labels[self.currentIdx, :]
        
    def changeTaskLabel(self, index) :
        self.taskIdx = index
        self.taskToPerform = possibleTasksToPerform[self.taskIdx, :]
        
    def renderOneFrame(self) :
        
        s = {'frameState': fs[self.currentIdx, :, :], 'taskParams': labels[self.currentIdx, :]}
        neighsProbs = np.zeros(k)#probs[self.currentIdx, neighbours[self.currentIdx, :]]/np.sum(probs[self.currentIdx, neighbours[self.currentIdx, :]])
#         bestA, bestAIdx = pi_l(s, A[self.currentIdx, :, :], v[:, self.taskIdx], self.currentIdx, fs, neighbours, labels, self.taskToPerform)
        bestA, bestAIdx = pi_l(s, A[self.currentIdx, :, :], np.sum(np.repeat(self.taskToPerform.reshape((1, len(self.taskToPerform))), numStates, axis=0)*v, axis=-1), self.currentIdx, fs, neighbours, labels, self.taskToPerform, neighsProbs, w[self.currentIdx, :])
        newTaskState, newIdx = integrateTaskState(s, bestA, fs[neighbours[self.currentIdx, :], :, :], 1.0, neighbours[self.currentIdx, :], labels)
        self.currentIdx = np.copy(newIdx)
        
        ## HACK ##
        im = np.ascontiguousarray(Image.open(frames[self.currentIdx+4]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
        infoText = "Frame idx: "+np.string_(newIdx)+"\t"+np.string_(newTaskState['taskParams'])
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

window = Window()
window.show()
app.exec_()

