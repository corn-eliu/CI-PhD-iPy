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
import ssim
import opengm

from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write
from pygraph.algorithms.cycles import find_cycle
import pygraph.algorithms.minmax as mm
import pygraph.algorithms.searching as search

import pygraphviz as gv

import VideoTexturesUtils as vtu

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
# sampleData = "ribbon1_matted/"
# sampleData = "little_palm1_cropped/"
sampleData = "ribbon2_matted/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame*.png")
maxFrames = len(frames)
frames = np.sort(frames)[0:maxFrames]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames
# if numFrames > 0 :
#     frameSize = cv2.imread(frames[0]).shape
#     movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=np.uint8)
#     for i in range(0, numFrames) :
#         movie[:, :, :, i] = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB), dtype=np.uint8)
        
#         sys.stdout.write('\r' + "Loaded frame " + np.string_(i) + " of " + np.string_(numFrames))
#         sys.stdout.flush()

# print        
# print 'Movie has shape', movie.shape

# <codecell>

## compute euclidean distance assuming f is an array where each row is a flattened image (1xN array, N=W*H*Channels)
## euclidean distance defined as the length of the the displacement vector:
## len(q-p) = sqrt(len(q)^2+len(p)^2 - 2*dot(p, q)) where p and q are two images in vector format and 1xN size
def distEuc(f) :
    ## gives sum over squared intensity values for each image
    ff = np.sum(f*f, axis=1)
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
    d = np.sqrt(np.reshape(ff, [len(ff),1])+ff.T - 2*np.dot(f, f.T))
    return d

def distEuc2(f1, f2) :
    ## gives sum over squared intensity values for each image
    ff1 = np.sum(f1*f1, axis=1)
    ff2 = np.sum(f2*f2, axis=1)
#     print ff1.shape
#     print ff2.shape
    ## first term is sum between each possible combination of frames
    ## second term is the the dot product between each frame as in the formula above
#     print "askdfh", np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1).shape, np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0).shape
    d = np.sqrt(np.repeat(np.reshape(ff1, [len(ff1),1]), len(ff2), axis=1)+np.repeat(np.reshape(ff2, [1, len(ff2)]), len(ff1), axis=0) - 2*np.dot(f1, f2.T))
    return d

# <codecell>

def computeDist(distType, f1, f2) :
    if distType == "Euc" :
        if f2 != None :
            return distEuc2(f1, f2)
        else :
            return distEuc(f1)
    elif distType == "SSIM" :
        return distSSIM(f1, f2)

# <codecell>

def areOverlapping(r1, r2) :
    if r1[0] > r1[1] or r2[0] > r2[1] :
        raise BaseException, np.string_("given ranges are not sorted: " + np.string_(r1) + ", " + np.string_(r2))
    return r1[1] >= r2[0] and r2[1] >= r1[0]

def mergeRanges(r1, r2) :
    united = union1d(r1, r2)
    return np.array((united[0], united[-1]))

# <codecell>

## divide data into subblocks
s = time.time()
numBlocks = 1
blockSize = numFrames/numBlocks
print numFrames, numBlocks, blockSize
distanceMatrix = np.zeros([numFrames, numFrames])

distanceType = "Euc"

for i in xrange(0, numBlocks) :
    
    t = time.time()
    
    ##load row frames
    f1s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
    for f, idx in zip(xrange(i*blockSize, i*blockSize+blockSize), xrange(0, blockSize)) :
        f1s[:, :, :, idx] = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0

    ##compute distance between every pair of row frames
    data1 = np.reshape(f1s, [np.prod(f1s.shape[0:-1]), f1s.shape[-1]]).T
    distanceMatrix[i*blockSize:i*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = computeDist(distanceType, data1, None)
    
    sys.stdout.write('\r' + "Row Frames " + np.string_(i*blockSize) + " to " + np.string_(i*blockSize+blockSize-1) + " in " + np.string_(time.time() - t))
    sys.stdout.flush()
    print
    
    for j in xrange(i+1, numBlocks) :
        
        t = time.time()
        
        ##load column frames
        f2s = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], blockSize]))
        for f, idx in zip(xrange(j*blockSize, j*blockSize+blockSize), xrange(0, blockSize)) :
            f2s[:, :, :, idx] = np.array(cv2.cvtColor(cv2.imread(frames[f]), cv2.COLOR_BGR2RGB))/255.0
            
        ##compute distance between every pair of row-column frames
        data2 = np.reshape(f2s, [np.prod(f2s.shape[0:-1]), f2s.shape[-1]]).T
        distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize] = computeDist(distanceType, data1, data2)
        distanceMatrix[j*blockSize:j*blockSize+blockSize, i*blockSize:i*blockSize+blockSize] = distanceMatrix[i*blockSize:i*blockSize+blockSize, j*blockSize:j*blockSize+blockSize].T
    
        sys.stdout.write('\r' + "Column Frames " + np.string_(j*blockSize) + " to " + np.string_(j*blockSize+blockSize-1) + " in " + np.string_(time.time() - t))
        sys.stdout.flush()
        print

figure(); imshow(distanceMatrix, interpolation='nearest')
print "finished in", time.time() - s

# <codecell>

## load ssim based dist matrix computed externally
distanceMatrix = np.load("ssimDist60-little_palm.npy")
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

## load full precomputed distMat for dataset
distanceMatrix = np.load(dataFolder + sampleData + "distMat.npy")
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

distMat = vtu.filterDistanceMatrix(distanceMatrix, 4, True)
figure(); imshow(distMat, interpolation='nearest')

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

distMatFut = estimateFutureCost(0.999, 2.0, distMat, np.ones(distMat.shape))

# <codecell>

print np.max(distMat)
print np.max(distMatFut)
figure(); imshow(distMat, interpolation='nearest')
figure(); imshow(distMatFut, interpolation='nearest')

# <codecell>

# Graph creation
gr = digraph()

size=distMatFut.shape[0]
numNeighbours = 2
weights = (distMatFut-np.min(distMatFut))/(np.max(distMatFut)-np.min(distMatFut))

gr.add_nodes(arange(size))
for i in xrange(size) :
    neighbours = np.argsort(distMatFut[i ,:])
#     neighbours = np.delete(neighbours, np.where(neighbours==i))
    if i > numNeighbours or True :
        for j in xrange(numNeighbours) :
            if (i, neighbours[j]) not in gr.edges():
                gr.add_edge((i, neighbours[j]), wt=weights[i, neighbours[j]], label=np.str(weights[i, neighbours[j]]))

# Draw as PNG
if True :
    dot = write(gr)
    gvv = gv.AGraph()
    gvv.from_string(dot)
    gvv.layout(prog='dot')
    gvv.draw(path='frameGraph.png', format='png')

# <codecell>

## find which of the nodes in distMat get added to graph
threshold = np.ones(distMatFut.shape)*np.max(weights)*2
threshold[np.repeat(arange(size), numNeighbours), np.reshape(np.argsort(distMatFut)[:, 0:numNeighbours], numNeighbours*size)] = 1
figure(); imshow(threshold, interpolation='nearest')
figure(); imshow(threshold*weights, interpolation='nearest')

# <codecell>

def strongly_connected_components(graph):
    """
    Tarjan's Algorithm (named for its discoverer, Robert Tarjan) is a graph theory algorithm
    for finding the strongly connected components of a graph.
    
    Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    """

    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    result = []
    
    def strongconnect(node):
        # set the depth index for this node to the smallest unused index
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
    
        # Consider successors of `node`
        try:
            successors = graph[node]
        except:
            successors = []
        for successor in successors:
            if successor not in lowlinks:
                # Successor has not yet been visited; recurse on it
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node],lowlinks[successor])
            elif successor in stack:
                # the successor is in the stack and hence in the current strongly connected component (SCC)
                lowlinks[node] = min(lowlinks[node],index[successor])
        
        # If `node` is a root node, pop the stack and generate an SCC
        if lowlinks[node] == index[node]:
            connected_component = []
            
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            component = tuple(connected_component)
            # storing the result
            result.append(component)
    
    for node in graph:
        if node not in lowlinks:
            strongconnect(node)
    
    return result

# <codecell>

gr.add_edge((15, 9), wt=gr.edge_weight((15, 0)))
gr.del_edge((15,0))

if True :
    dot = write(gr)
    gvv = gv.AGraph()
    gvv.from_string(dot)
    gvv.layout(prog='dot')
    gvv.draw(path='frameGraph.png', format='png')

# <codecell>

strongly_connected_components(gr)

# <codecell>

aCycle = np.array(find_cycle(gr))
print aCycle.shape, aCycle[-1]-aCycle[0]
print aCycle - arange(aCycle[0], aCycle[-1]+1)
print frames[aCycle[0]+5]
print frames[aCycle[-1]+5]
# print gr
print aCycle
# print
# tmp = mm.maximum_flow(gr, 0, 15)
# print tmp[0]
# print
# print tmp[1]
# for edge in gr.edges():
#     print edge, gr.edge_weight(edge)

# <codecell>

from pygraph.algorithms.cycles import find_cycle
import pygraph.algorithms.minmax as mm
import pygraph.algorithms.searching as search
print find_cycle(gr)
print mm.minimal_spanning_tree(gr)
print search.breadth_first_search(gr)

# <codecell>

print gr.edges()
print (i, neighbours[j])
print size

# <codecell>

## make a graph with 1 node for each frame
f1=numpy.ones([2])
# f2=numpy.ones([2,2])

size=distMatFut.shape[0]
numLabels=2 ## dunno what to set this to
numNeighbours = 5
gm=opengm.gm([numLabels]*size, operator='adder')

unaryId = gm.addFunction(np.ones(size))
pairwiseId=gm.addFunction(distMatFut)
for i in xrange(size) :
    gm.addFactor(unaryId,i)
    neighbours = np.argsort(distMatFut[i ,:])
    neighbours = np.delete(neighbours, np.where(neighbours==i))
    if i > numNeighbours :
        for j in xrange(numNeighbours) :
            gm.addFactor(pairwiseId,np.sort([i, neighbours[j]]))
#     print "unary factor f1", i
    
    

print gm
opengm.visualizeGm(gm,layout='neato',iterations=3000,
                    show=True,plotFunctions=True,
                    plotNonShared=True,relNodeSize=0.5)
# plt.savefig("graph.png",bbox_inches='tight',dpi=300) 
# plt.close()

# <codecell>

print np.min(distMatFut)

# <codecell>

chainLength=20
numLabels=1024
print numpy.ones(chainLength,dtype=opengm.label_type)*numLabels

# <codecell>

## example graph and printout
f1=numpy.ones([2])
f2=numpy.ones([2,2])

"""
Grid:
    - 4x4=16 variables
    - second order factors in 4-neigbourhood
      all connected to the same function
    - higher order functions are shared
"""

size=3
gm=opengm.gm([2]*size*size)

fid=gm.addFunction(f2)
print fid.functionIndex, fid.functionType
countU = 0
countP = 0
for y in range(size):   
    for x in range(size):
        fid2 = gm.addFunction(f1)
        gm.addFactor(fid2,x*size+y)
        print "unary factor f1", x, y, x*size+y, "la", fid2.functionIndex, fid2.functionType
        countU += 1
        if(x+1<size):
            gm.addFactor(fid,[x*size+y,(x+1)*size+y])
            print "pairwise factor f2 (x+1)", x, y, [x*size+y,(x+1)*size+y]
            countP += 1
        if(y+1<size):
            gm.addFactor(fid,[x*size+y,x*size+(y+1)])
            print "pairwise factor f2 (y+1)", x, y, [x*size+y,x*size+(y+1)]
            countP += 1

print countU, countP
print gm
opengm.visualizeGm( gm,layout='spring',iterations=3000,
                    show=True,plotFunctions=True,
                    plotNonShared=True,relNodeSize=0.4)
# plt.savefig("grid3.png",bbox_inches='tight',dpi=300) 
# plt.close()

# <codecell>

probs, cumprobs = vtu.getProbabilities(distMat, 0.105, True)
figure(); imshow(probs, interpolation='nearest')
figure(); imshow(cumprobs, interpolation='nearest')
finalFrames = vtu.getFinalFrames(cumprobs, 100, 5, 30, False, False)
print finalFrames

# <codecell>

## build dynamic programming table for finding optimal loops
# find a set of primitive loops
numOfBestTransitions = 1 # keep only numOfBestTransitions best transitions from frame i
primitiveLoops = np.zeros(((distMat.shape[0]-numOfBestTransitions)*numOfBestTransitions, 2), dtype=uint)

for i in xrange(numOfBestTransitions, distMat.shape[0]) :
    j = np.argsort(distMat[i, 0:i])[0:numOfBestTransitions]
    for t in xrange(0, numOfBestTransitions) :
        loopIdx = numOfBestTransitions*(i-numOfBestTransitions)+t
        primitiveLoops[loopIdx, :]= np.array([j[t], i])

numPrimitiveLoops = len(primitiveLoops) # distMat.shape[0]

## we'll consider loops of maximum maxLoopLength frames and each frame is a primitive loop
maxLoopLength = 40
# the costTable contains the cost of the cheapest loop in the corresponding cell in the loopsTable
costTable = np.ones((maxLoopLength+1, numPrimitiveLoops))*sys.float_info.max

# the loopsTable contains the possible primitive and compound loops
# there's one row for each considered loop lenght and one column for each considered primitive loop
# each cell, apart from the cells on first row, contains a set of indices that reference loops in the loopsTable
loopsTable = [] #-np.ones((maxLoopLength, numPrimitiveLoops, 2), dtype=int)

primitiveLoopsLength = np.ndarray.flatten(np.diff(primitiveLoops, axis=1)+1)
# row 0 contains info relative to primitive loops
costTable[0, :] = distMat[primitiveLoops[:, 0], primitiveLoops[:, 1]]
loopsTable.append(primitiveLoops)

# row 1 has no loops as I don't want loops with only 1 frame
loopsTable.append(np.reshape(-np.ones((2, numPrimitiveLoops), dtype=int), (numPrimitiveLoops, 2)))
## now compute lowest cost compound loops of row r-length
for r in xrange(2, 3):#len(costTable)) :
    print r
    if r in primitiveLoopsLength :
        print "lala"
    # find

# <codecell>

i = 50
j = costMat[i:i+windowSize, :]
figure(); imshow(j, interpolation='nearest')
sortIndices = np.argsort(j, axis=-1)
print sortIndices
print j
print sortIndices[:, 0]
print np.sort(j, axis=-1)[:, 0]
print [sortIndices[np.argmin(np.sort(j, axis=-1)[:, 0]), 0], i+np.argmin(np.sort(j, axis=-1)[:, 0])]

# <codecell>

## find primitive loops with sliding window and min loop length
windowSize = 20 ## maybe change this value based on average loop length after a preliminary first pass or numFrames
minLoopLength = 10
costMat = np.copy(distMat)
# only allow backwards transitions
costMat[np.triu_indices(len(costMat), k=-minLoopLength+1)] = np.max(costMat)
# figure(); imshow(costMat, interpolation='nearest')

primitiveLoops = []
currentMinCost = np.max(costMat)
currentBestTransition = [-1, -1]
primitiveLoopsCosts = []
for i in xrange(minLoopLength, costMat.shape[0], windowSize) :
#     print i, i+windowSize
    j = costMat[i:i+windowSize, :]
    sortIndices = np.argsort(j, axis=-1)
    if np.min(np.sort(j, axis=-1)[:, 0]) <= currentMinCost :
#         print "better from", currentMinCost,
        currentMinCost = np.min(np.sort(j, axis=-1)[:, 0])
        currentBestTransition = [sortIndices[np.argmin(np.sort(j, axis=-1)[:, 0]), 0], i+np.argmin(np.sort(j, axis=-1)[:, 0])]
#         print "to", currentMinCost
    else :
#         print "tralala", primitiveLoopsCosts, np.mean(primitiveLoopsCosts), np.median(primitiveLoopsCosts)
        print "new best", currentMinCost, currentBestTransition, np.min(np.sort(j, axis=-1)[:, 0])
        primitiveLoopsCosts.append(currentMinCost)
        primitiveLoops.append(currentBestTransition)
        currentMinCost = np.max(costMat)
# #     j = np.argsort(costMat[i, 0:i], )[0:numOfBestTransitions]
#     for t in xrange(0, numOfBestTransitions) :
#         loopIdx = numOfBestTransitions*(i-numOfBestTransitions)+t
#         primitiveLoops[loopIdx, :]= np.array([j[t], i])

## not sure if I should keep the last best transition 
if not np.all(primitiveLoops[-1] == currentBestTransition) and not np.all(primitiveLoopsCosts[-1] == currentMinCost):
    primitiveLoopsCosts.append(currentMinCost)
    primitiveLoops.append(currentBestTransition)
    

## only keep transitions with cost lower than median plus certain percentage
thresh = 0.1 ## maybe change this depending on how many potential transitions have been found 
             ## (the more, the higher the probability that there are more outliers so lower tresh should be)
medianCost = np.median(primitiveLoopsCosts)
goodTransitions = np.ndarray.flatten(np.argwhere(primitiveLoopsCosts <= medianCost+medianCost*thresh))
print "keeping", len(goodTransitions), "of", len(primitiveLoopsCosts)
primitiveLoopsCosts = np.array(primitiveLoopsCosts)[goodTransitions]
primitiveLoops = np.array(primitiveLoops)[goodTransitions, :]
# primitiveLoops = np.reshape(primitiveLoops, (len(primitiveLoops), 2))
print primitiveLoopsCosts
print primitiveLoops

print "stats"
print np.median(np.diff(primitiveLoops)), np.mean(np.diff(primitiveLoops))
print np.diff(primitiveLoops)

# <codecell>

## print the frame names fro qualitative comparison
correction = 5
print np.reshape(frames[np.ndarray.flatten(primitiveLoops+5)], primitiveLoops.shape)

# <codecell>

print currentMinCost, currentBestTransition

# <codecell>

# figure(); imshow(costMat, interpolation='nearest')
tmpCostMat = (costMat-np.min(costMat))/(np.max(costMat)-np.min(costMat))
print tmpCostMat[43, 0], tmpCostMat[26, 0], tmpCostMat[43, 0], np.max(tmpCostMat)
figure(); imshow(tmpCostMat, interpolation='nearest')

# <codecell>

U, s, Vh = sp.linalg.svd(distMat)
w, v = np.linalg.eig(distMat)
print w, np.linalg.det(distMat), np.prod(w), v.shape
print v[:, 0]
print np.dot(distMat, v[:, 0]), w[0]*v[:, 0]

# <codecell>

print np.argmin(costMat), np.min(costMat), costMat.shape, costMat[38, 17]
print currentBestTransition, currentMinCost
print primitiveLoopsCosts
print np.mean(primitiveLoopsCosts), np.median(primitiveLoopsCosts), np.median(primitiveLoopsCosts)+np.median(primitiveLoopsCosts)*thresh
print sp.constants.e, np.exp(1.0/len(primitiveLoopsCosts)), np.exp(1.0/1.0)

# <codecell>

# print minCosts
print primitiveLoops
print len(primitiveLoops)
# print np.array(minCosts)[np.where(minCosts <= np.median(minCosts)+np.median(minCosts)*0.1)]

# <codecell>

print costMat[20, 0]
print costMat[50, 8]

# <codecell>

# print primitiveLoops
tmpMat = np.copy(distMat)
tmpMat[np.triu_indices(len(tmpMat), k=-minLoopLength+1)] = np.max(tmpMat)
figure(); imshow(tmpMat, interpolation='nearest')
print np.argsort(tmpMat)[0, 1]
tmpSort = np.argsort(np.reshape(tmpMat, (np.prod(tmpMat.shape))))[0:10]
print tmpSort
tmpBest = np.reshape(np.mod(tmpSort, len(tmpMat)*np.ones(len(tmpSort))), (len(tmpSort), 1))
tmpBest = np.array(np.hstack((tmpBest, np.reshape(tmpSort/len(tmpMat)*np.ones(len(tmpSort)), (len(tmpSort), 1)))), dtype=int)
print tmpBest
print distMatFut[tmpBest[:, 0], tmpBest[:, 1]]
print np.reshape(tmpMat, (np.prod(tmpMat.shape)))[tmpSort]
print np.diff(tmpBest, axis=1)

# <codecell>

print costTable[0, :]
print loopsTable[1]
print primitiveLoops.shape

# <rawcell>

# ####################################### OBSOLETE #####################################
# 
# ## build dynamic programming table for finding optimal loops
# # find a set of primitive loops
# numOfBestTransitions = 1 # keep only numOfBestTransitions best transitions from frame i
# primitiveLoops = np.zeros(((distMat.shape[0]-numOfBestTransitions)*numOfBestTransitions, 2), dtype=uint)
# 
# for i in xrange(numOfBestTransitions, distMat.shape[0]) :
#     j = np.argsort(distMat[i, 0:i])[0:numOfBestTransitions]
#     for t in xrange(0, numOfBestTransitions) :
#         loopIdx = numOfBestTransitions*(i-numOfBestTransitions)+t
#         primitiveLoops[loopIdx, :]= np.array([j[t], i])
# 
# numPrimitiveLoops = len(primitiveLoops) # distMat.shape[0]
# 
# ## we'll consider loops of maximum maxLoopLength frames and each frame is a primitive loop
# maxLoopLength = 40
# # the costTable contains the cost of the cheapest loop in the corresponding cell in the loopsTable
# costTable = np.ones((maxLoopLength+1, numPrimitiveLoops))*sys.float_info.max
# 
# # the loopsTable contains the possible primitive and compound loops
# # there's one row for each considered loop lenght and one column for each considered primitive loop
# # each cell, apart from the cells on first row, contains a set of indices that reference loops in the loopsTable
# loopsTable = [] #-np.ones((maxLoopLength, numPrimitiveLoops, 2), dtype=int)
# 
# # row 0 is ignored and row 1 contains cost of primitive loops of length 1 (i.e. one transition from frame i to i-1 )
# singleFrameLoops = np.where(np.ndarray.flatten(np.diff(primitiveLoops, axis=1)+1) == 1)[0]
# costTable[0, singleFrameLoops] = distMat[primitiveLoops[singleFrameLoops, 0], primitiveLoops[singleFrameLoops, 1]]
# # the first row only contains 1-length loops from primitiveLoops
# loopsTable.append(np.reshape(-np.ones((1, numPrimitiveLoops), dtype=int), (numPrimitiveLoops, 1)))
# loopsTable[0][singleFrameLoops] = np.reshape(primitiveLoops[singleFrameLoops, 0], (len(singleFrameLoops), 1))
# # loopsTable[0, singleFrameLoops, :] = primitiveLoops[singleFrameLoops, :]
# 
# ## now compute lowest cost compound loops of row r+1 length
# for r in xrange(1, maxLoopLength) :
#     currentLoopLength = r+1
#     # find 

# <codecell>

print singleFrameLoops
print loopsTable[0][singleFrameLoops].shape
print primitiveLoops[singleFrameLoops, 0].shape

# <codecell>

# for loop, cost in zip(loopsTable[0, :, :], costTable[0, :]) :
#     print loop, cost
for loop, cost in zip(loopsTable[0], costTable[0, :]) :
    print loop, cost, primitiveLoops[loop]

# <codecell>

for loop, cost in zip(primitiveLoops, costTable[0, :]) :
    print areOverlapping(primitiveLoops[27, :], loop), primitiveLoops[27, :], loop, cost,
    if areOverlapping(primitiveLoops[27, :], loop) :
        print mergeRanges(primitiveLoops[27, :], loop)
    else :
        print

# <codecell>

print distMat[np.array(primitiveLoops)[:, 0], np.array(primitiveLoops)[:, 1]]
print np.argsort(distMat[np.array(primitiveLoops)[:, 0], np.array(primitiveLoops)[:, 1]])
print primitiveLoops[np.argsort(distMat[np.array(primitiveLoops)[:, 0], np.array(primitiveLoops)[:, 1]])[0]]

# <codecell>

## test finalFrames
# bestLoop = np.argsort(distMat[np.array(primitiveLoops)[:, 0], np.array(primitiveLoops)[:, 1]])[0]
# print bestLoop, primitiveLoops[bestLoop]
# finalFrames = np.arange(primitiveLoops[bestLoop][0], primitiveLoops[bestLoop][1])
finalFrames = np.arange(aCycle[0]+5, aCycle[-1]+5)
# print frames[finalFrames]
print

# <codecell>

idx = 1
print np.argsort(distMat[idx, 0:idx])
figure(); imshow(np.reshape(distMat[idx, 0:idx], (1, distMat[idx, 0:idx].shape[0])), interpolation='nearest')

# <codecell>

loopsTable = []
loopsTable.append([0, 1, 2])
loopsTable.append([3, 4, 5])
loopsTable.append([0, 1, 2])
# loopsTable[0].append([5, 1, 2])
loopsTable = list(reshape(loopsTable, (3, 1, 3)))
# loopsTable[0][0].append([5, 1, 2])
print loopsTable
print loopsTable[0][0][:]

# <codecell>

## visualize frames automatically

def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw

fig=plt.figure()
img = plt.imshow(np.array(cv2.cvtColor(cv2.imread(frames[finalFrames[0]]), cv2.COLOR_BGR2RGB), dtype=np.uint8))
img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
#     img.set_data(movie[:, :, :, finalFrames[0]])
    img.set_data(np.array(cv2.cvtColor(cv2.imread(frames[finalFrames[0]]), cv2.COLOR_BGR2RGB), dtype=np.uint8))
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f) + ' from ' + str(finalFrames[f]))
#     img.set_data(movie[:, :, :, finalFrames[f]])
    img.set_data(np.array(cv2.cvtColor(cv2.imread(frames[finalFrames[f]]), cv2.COLOR_BGR2RGB), dtype=np.uint8))
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalFrames),interval=33,blit=True)

plt.show()

# <codecell>


