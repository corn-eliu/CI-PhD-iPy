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

from Queue import Queue

from pygraph.classes.digraph import digraph
from pygraph.classes.graph import graph
from pygraph.readwrite.dot import write
from pygraph.algorithms.minmax import shortest_path

import pygraphviz as gv

import VideoTexturesUtils as vtu
import GraphWithValues as gwv

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

def gauss(x, mean, variance) :
    normTerm = 1.0#np.sqrt(2*variance*np.pi)
    return np.exp(-np.power(x-mean, 2.0)/(2*variance))/normTerm

def multiGauss(X, Y, theta, mean, variance) :
    normTerm = 1.0#np.sqrt(np.linalg.det(covar)*np.power(2*np.pi, x.shape[0]))
    if len(mean) != 2 or len(variance) != 2 :
        raise BaseException, "multiGauss needs as many means and variances as k, where k=2"
        
    a = (np.cos(theta)**2)/(2*variance[0])+(np.sin(theta)**2)/(2*variance[1])
    b = np.sin(2*theta)/(4*variance[0])+np.sin(2*theta)/(4*variance[1])
    c = (np.sin(theta)**2)/(2*variance[0])+(np.cos(theta)**2)/(2*variance[1])
    
    return np.exp(-(a*np.power(X-mean[0], 2)+2*b*(X-mean[0])*(Y-mean[1])+c*np.power(Y-mean[1], 2)))/normTerm

# <codecell>

def smoothStep(x, mean, interval, steepness) :
    a = mean-np.floor(interval/2.0)
    b = mean-np.floor((interval*steepness)/2.0)
    c = mean+np.ceil((interval*steepness)/2.0)
    d = mean+np.ceil(interval/2.0)
    print a, b, c, d
    
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

# <codecell>

# Draw graph as PNG
def drawGraph(graph, name, yes) :
    if yes :
        dot = write(graph)
        gvv = gv.AGraph()
        gvv.from_string(dot)
        gvv.layout(prog='circo')
        gvv.draw(path=name)

# <codecell>

## start from a set of frames (i.e. initNodes, 1 for each situation) and add frames that can be played together
def buildGraph(initNodes, weights, numBest, minLoopLength, minMaxIterations, drawOnPng, initGraph) : 
    if initGraph != None : 
        gr = initGraph
    else :
        gr = digraph()
    gr.add_nodes(initNodes)
    for i in xrange(0, minMaxIterations[1]) :
        ## for each node find the best node to go to next and the best node to get there from
        printout = '\r' + "Iteration " + np.string_(i) + "; "
        finished = False
        ## check if all initNodes have been connected by graph
        finished = True
        for p in xrange(-1, len(initNodes)-1) :
            if initNodes[p+1] in shortest_path(gr, initNodes[p])[0] :
                printout += "conn " + np.string_(initNodes[p+1]) + "-" + np.string_(initNodes[p]) + " "
            else :
                finished = False
        sys.stdout.write(printout)
        sys.stdout.flush()
        
        if finished and i >= minMaxIterations[0]: 
            print "finished"
            break
        for node in gr.nodes():
            
            bestBackward = []
            bestForward = []
            
            if node > 2 :
                bestBackward = np.argsort(weights[node, :])
                bestBackward = np.ndarray.flatten(bestBackward[np.argwhere(bestBackward < node-1)])
                bestBackward = np.ndarray.flatten(bestBackward[np.argwhere(weights[node, bestBackward] <= 9.0)])
                bestBackward = bestBackward[0:numBest]
            if node+1 < weights.shape[-1] :
                bestForward = np.argsort(weights[node, :])
                bestForward = np.ndarray.flatten(bestForward[np.where(bestForward > node+1)])
                bestForward = np.ndarray.flatten(bestForward[np.argwhere(weights[node, bestForward] <= 9.0)])
                bestForward = bestForward[0:numBest]
            
            # add prev and next frames
#             if node-1 >=0 :
#                 bestBackward = np.hstack((bestBackward, node-1))
            if node+1 < weights.shape[-1] :
                bestForward = np.hstack((bestForward, int(node+1)))
            print bestForward, bestBackward, node
            
            
            for edgeTo in np.array(np.hstack((bestBackward, bestForward)), dtype=int) :
                print "la", edgeTo
                if edgeTo not in gr.nodes() :
                    gr.add_node(edgeTo)
                    
                if (node, edgeTo) not in gr.edges() :
                    gr.add_edge((node, edgeTo), wt=weights[node, edgeTo], label=np.str(weights[node, edgeTo]))
                else :
                    gr.set_edge_weight((node, edgeTo), weights[node, edgeTo])
                    gr.set_edge_label((node, edgeTo), np.str(weights[node, edgeTo]))
    #         print edgeTo
            
    drawGraph(gr, "frameGraph.png", drawOnPng)
    return gr

# <codecell>

## start from a set of frames (i.e. initNodes, 1 for each situation) and add frames that can be played together
def buildGraph(initNodes, weights, numBest, minLoopLength, minIterations, drawOnPng, initGraph) : 
    if initGraph != None : 
        gr = initGraph
    else :
        gr = digraph()
    gr.add_nodes(initNodes)
    for i in xrange(0, 30) :
        ## for each node find the best node to go to next and the best node to get there from
        printout = '\r' + "Iteration " + np.string_(i) + "; "
        finished = False
        ## check if all initNodes have been connected by graph
        finished = True
        for p in xrange(-1, len(initNodes)-1) :
            if initNodes[p+1] in shortest_path(gr, initNodes[p])[0] :
                printout += "conn " + np.string_(initNodes[p+1]) + "-" + np.string_(initNodes[p]) + " "
            else :
                finished = False
        sys.stdout.write(printout)
        sys.stdout.flush()
        
        if finished and i >= minIterations: 
            print "finished"
            break
        for node in gr.nodes():
            bestPrevs = np.argsort(weights[:, node])#[-numBest:]
            bestNexts = np.argsort(weights[node, :])#[-numBest:]
            print bestPrevs
            print bestNexts
            # remove current node
            bestPrevs = np.delete(bestPrevs, np.where(bestPrevs == node))
            bestNexts = np.delete(bestNexts, np.where(bestNexts == node))
            # remove prev and next frames as they are specifically added later --> only useful if minLoopLength = 1
            if minLoopLength <= 1 :
                bestPrevs = np.delete(bestPrevs, np.where(bestPrevs == node-1))
                bestPrevs = np.delete(bestPrevs, np.where(bestPrevs == node+1))
                bestNexts = np.delete(bestNexts, np.where(bestNexts == node-1))
                bestNexts = np.delete(bestNexts, np.where(bestNexts == node+1))

            
            # take only transitions s.a. minimum loop length = minLoopLength and take first numBest transitions 
            # (lower weight is better)
            bestPrevs = bestPrevs[np.where(np.abs(bestPrevs-node) >= minLoopLength)][0:numBest]
            bestNexts = bestNexts[np.where(np.abs(bestNexts-node) >= minLoopLength)][0:numBest]
            
            # add prev and next frames
            if node-1 >=0 :
                bestPrevs = np.hstack((bestPrevs, node-1))
            if node+1 < weights.shape[-1] :
                bestNexts = np.hstack((bestNexts, node+1))
#             print bestNexts, bestPrevs, node
            
            
            for bestPrev, bestNext in zip(bestPrevs, bestNexts) :
                if bestPrev not in gr.nodes() :
                    gr.add_node(bestPrev)
                if bestNext not in gr.nodes() :
                    gr.add_node(bestNext)
            
                if (bestPrev, node) not in gr.edges() :
                    gr.add_edge((bestPrev, node), wt=weights[bestPrev, node], label=np.str(weights[bestPrev, node]))
                else :
                    gr.set_edge_weight((bestPrev, node), weights[bestPrev, node])
                    gr.set_edge_label((bestPrev, node), np.str(weights[bestPrev, node]))
                    
                if (node, bestNext) not in gr.edges() :
                    gr.add_edge((node, bestNext), wt=weights[node, bestNext], label=np.str(weights[node, bestNext]))
                else :
                    gr.set_edge_weight((node, bestNext), weights[node, bestNext])
                    gr.set_edge_label((node, bestNext), np.str(weights[node, bestNext]))
    #         print bestNext, bestPrev
            
    drawGraph(gr, "frameGraph.png", drawOnPng)
    return gr

# <codecell>

def findBestLoops(windowSize, minLoopLength, costMatrix, way) : 
    ## find best loops with sliding window and min loop length
    ## maybe change windowSize this value based on average loop length after a preliminary first pass or numFrames
    costMat = np.copy(costMatrix)
    # only allow backwards transitions
    if way == 'forward' :
        costMat[np.tril_indices(len(costMat), k=minLoopLength-1)] = np.max(costMat)
    elif way == 'backward' :
        costMat[np.triu_indices(len(costMat), k=-minLoopLength+1)] = np.max(costMat)
#     figure(); imshow(costMat, interpolation='nearest')
    
    bestLoops = []
    currentMinCost = np.max(costMat)
    currentBestTransition = [-1, -1]
    bestLoopsCosts = []
    for i in xrange(minLoopLength, costMat.shape[0], windowSize) :
        j = costMat[i:i+windowSize, :]
        sortIndices = np.argsort(j, axis=-1)
        if np.min(np.sort(j, axis=-1)[:, 0]) <= currentMinCost :
#             print "better from", currentMinCost,
            currentMinCost = np.min(np.sort(j, axis=-1)[:, 0])
            currentBestTransition = [sortIndices[np.argmin(np.sort(j, axis=-1)[:, 0]), 0], i+np.argmin(np.sort(j, axis=-1)[:, 0])]
        else :
#             print "new best", currentMinCost, currentBestTransition, np.min(np.sort(j, axis=-1)[:, 0])
            bestLoopsCosts.append(currentMinCost)
            bestLoops.append(currentBestTransition)
            currentMinCost = np.max(costMat)
    
    ## not sure if I should keep the last best transition 
    if not np.all(bestLoops[-1] == currentBestTransition) and not np.all(bestLoopsCosts[-1] == currentMinCost):
        bestLoopsCosts.append(currentMinCost)
        bestLoops.append(currentBestTransition)
        
    
    ## only keep transitions with cost lower than median plus certain percentage
    thresh = 0.1 ## maybe change this depending on how many potential transitions have been found 
                 ## (the more, the higher the probability that there are more outliers so lower tresh should be)
    medianCost = np.median(bestLoopsCosts)
    goodTransitions = np.ndarray.flatten(np.argwhere(bestLoopsCosts <= medianCost+medianCost*thresh))
    print "keeping", len(goodTransitions), "of", len(bestLoopsCosts)
    bestLoopsCosts = np.array(bestLoopsCosts)[goodTransitions]
    bestLoops = np.array(bestLoops)[goodTransitions, :]
    # bestLoops = np.reshape(bestLoops, (len(bestLoops), 2))
#     print bestLoopsCosts

    return bestLoops

# <codecell>

## made up data of a dot moving left to right and top to bottom
frameSize = np.array([7, 7, 3])
numFrames = 31
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=uint8)
movie[3, 0, :, 0] = movie[3, 1, :, 1] = movie[3, 2, :, 2] = movie[3, 3, :, 3] = movie[3, 4, :, 4] = 255
movie[3, 5, :, 5] = movie[3, 6, :, 6] = movie[3, 5, :, 7] = movie[3, 4, :, 8] = movie[3, 3, :, 9] = 255
movie[3, 2, :, 10] = movie[3, 1, :, 11] = movie[3, 0, :, 12] = movie[3, 1, :, 13] = movie[3, 2, :, 14] = 255
movie[3, 3, :, 15] = movie[2, 3, :, 16] = movie[1, 3, :, 17] = movie[0, 3, :, 18] = movie[1, 3, :, 19] = 255
movie[2, 3, :, 20] = movie[3, 3, :, 21] = movie[4, 3, :, 22] = movie[5, 3, :, 23] = movie[6, 3, :, 24] = 255
movie[5, 3, :, 25] = movie[4, 3, :, 26] = movie[3, 3, :, 27] = movie[2, 3, :, 28] = movie[1, 3, :, 29] = 255
movie[0, 3, :, 30] = 255

# numFrames = 26
# movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=uint8)
# movie[3, 0, :, 0] = movie[3, 1, :, 1] = movie[3, 2, :, 2] = movie[3, 3, :, 3] = movie[3, 4, :, 4] = 255
# movie[3, 5, :, 5] = movie[3, 6, :, 6] = movie[3, 5, :, 7] = movie[3, 4, :, 8] = movie[3, 3, :, 9] = 255
# movie[3, 2, :, 10] = movie[3, 1, :, 11] = movie[3, 0, :, 12] = movie[0, 3, :, 13] = movie[1, 3, :, 14] = 255
# movie[2, 3, :, 15] = movie[3, 3, :, 16] = movie[4, 3, :, 17] = movie[5, 3, :, 18] = movie[6, 3, :, 19] = 255
# movie[5, 3, :, 20] = movie[4, 3, :, 21] = movie[3, 3, :, 22] = movie[2, 3, :, 23] = movie[1, 3, :, 24] = 255
# movie[0, 3, :, 25] = 255

distanceMatrix = np.zeros((numFrames, numFrames))
for r in xrange(0, numFrames) :
    for c in xrange(r+1, numFrames) :
        rowLoc = np.argwhere(movie[:, :, 0, r] == 255)[0]
        colLoc = np.argwhere(movie[:, :, 0, c] == 255)[0]
        distanceMatrix[r, c] = distanceMatrix[c, r] = np.sum(np.abs(rowLoc-colLoc))

## add noise
# distanceMatrix += np.random.random(distanceMatrix.shape)*(1-np.eye(distanceMatrix.shape[0]))
figure(); imshow(distanceMatrix, interpolation='nearest')

distMat = vtu.filterDistanceMatrix(distanceMatrix, 1, True)
idxCorrection = 1
figure(); imshow(distMat, interpolation='nearest')

# <codecell>

## start from a set of frames (i.e. initNodes, 1 for each situation) and add frames that can be played together
def buildGraph(initNodes, weights, ranges, numBest, minLoopLength, minMaxIterations, drawOnPng, initGraph) : 
    if initGraph != None : 
        gr = initGraph
    else :
        gr = digraph()
    for initNode in initNodes :
        if initNode not in gr.nodes() :
            gr.add_node(initNode)
    for i in xrange(0, minMaxIterations[1]) :
        ## for each node find the best node to go to next and the best node to get there from
        printout = '\r' + "Iteration " + np.string_(i+1) + "; "
        ## check if all initNodes have been connected by graph
        finished = True
        ## check initNodes are connected
        for i1, i2 in zip(initNodes, np.roll(initNodes, 1)) :
#             print "t", i1, i2
            if i2 in shortest_path(gr, i1)[0] and i1 in shortest_path(gr, i2)[0] :
                printout += "conn " + np.string_(i1) + "-" + np.string_(i2) + " "
            else :
                finished = False
        sys.stdout.write(printout)
        sys.stdout.flush()
        
        if finished and i >= minMaxIterations[0]: 
            print "finished"
            break
        
        for node in gr.nodes():
            ## add node for frame node-1 and connect to node
            if node-1 >= 0 :
                if node-1 not in gr.nodes() :
                    gr.add_node(node-1)
                    
                if (node-1, node) not in gr.edges() :
                    gr.add_edge((node-1, node), wt=weights[node-1, node], label=np.str(weights[node-1, node]))
                else :
                    gr.set_edge_weight((node-1, node), weights[node-1, node])
                    gr.set_edge_label((node-1, node), np.str(weights[node-1, node]))
            
            ## add node for frame node+1 and connect to node
            if node+1 < weights.shape[-1] :
                if node+1 not in gr.nodes() :
                    gr.add_node(node+1)
                
                if (node, node+1) not in gr.edges() :
                    gr.add_edge((node, node+1), wt=weights[node, node+1], label=np.str(weights[node, node+1]))
                else :
                    gr.set_edge_weight((node, node+1), weights[node, node+1])
                    gr.set_edge_label((node, node+1), np.str(weights[node, node+1]))
            
            ## order transitions based on weights and keep n best s.t. n is higher the further away from center
            bestTransitions = np.argsort(weights[node, :])
            bestTransitions = bestTransitions[np.where(np.abs(bestTransitions-node) > minLoopLength)]
            bestTransitions = bestTransitions[0:5*(1-ranges[node])]
            print node, bestTransitions
            
            for bt in bestTransitions :
                ## check timewise neighbors of node for transitions to bt
                neighSize = 5
                neighs = arange(node-neighSize, node+neighSize+1, dtype=int)
                btWeight = weights[node, bt]
                print "weight", btWeight

                addEdge = True
                for neigh in np.delete(neighs, np.where(neighs == node)) :
                    if (neigh, bt) in gr.edges() : 
                        if gr.edge_weight((neigh, bt)) >= btWeight :
                            print "remove edge", neigh, bt, gr.edge_weight((neigh, bt))
                            gr.del_edge((neigh, bt))
                        else :
                            print "found better connected neighbour", neigh, bt, gr.edge_weight((neigh, bt))
                            addEdge = False
                
                if addEdge :
                    print "new edge", node, bt
                    if bt not in gr.nodes() :
                        gr.add_node(bt)
                    
                    if (node, bt) not in gr.edges() :
                        gr.add_edge((node, bt), wt=btWeight, label=np.str(btWeight))
                    else :
                        gr.set_edge_weight((node, bt), btWeight)
                        gr.set_edge_label((node, bt), np.str(btWeight))
#             for n in gr.neighbors(node) :
#                 print n, gr.neighbors(n),
#             print
#             print
#             for t in bestTransitions :
#                 if t not in gr.nodes() :
#                     gr.add_node(t)
                    
#                 if (node, t) not in gr.edges() :
#                     gr.add_edge((node, t), wt=weights[node, t], label=np.str(weights[node, t]))
#                 else :
#                     gr.set_edge_weight((node, t), weights[node, t])
#                     gr.set_edge_label((node, t), np.str(weights[node, t]))
#             print node, ranges[node], bestTransitions[0:5*(1-ranges[node])]


            
    drawGraph(gr, "frameGraph.png", drawOnPng)
    return gr

# <codecell>

print initialNodes
# print getShortestPath(frameGraph, initialNodes[1]+idxCorrection, initialNodes[0]+idxCorrection)
print probMat[4,:]
print 1/probMat[4,:]
print rangeCurves[0:1]
print rangeCurves[7]

# <codecell>

weightMat = distMat[1:distMat.shape[1], 0:-1]
# weightMat = distMat[0:-1, 1:distMat.shape[1]]
figure(); imshow(weightMat, interpolation='nearest')

probMat, cumProb = vtu.getProbabilities(weightMat, 1.0, True)
idxCorrection =2 #+= 1

initialNodes = np.array([9, 21])-idxCorrection
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(probMat, interpolation='nearest')
# ax.set_autoscale_on(False)
# ax.scatter(initialNodes, initialNodes, c="m", marker="s")

## compute rangeCurves using initialNodes before idxCorrection
rangeCurves = np.zeros(weightMat.shape[-1])
for node in initialNodes :
    ## the variances should be set according to situation range
#     rangeCurves += gauss(arange(0.0, len(rangeCurves)), float(node), 5.0)
    rangeCurves += smoothStep(arange(0.0, len(rangeCurves)), float(node), 13, 0.4)
rangeCurves /= np.max(rangeCurves)
figure(); plot(arange(0, len(rangeCurves)), rangeCurves)
print rangeCurves
# frameGraph = buildGraph(initialNodes, weightMat, rangeCurves, 1, 3, (1, 1), True, None)
frameGraph = buildGraph(initialNodes, 1.0/probMat, rangeCurves, 1, 3, (10, 30), True, None)

print len(frameGraph.nodes()), frameGraph.nodes()
print len(frameGraph.edges()), frameGraph.edges()

# <codecell>

queue = Queue()
queue.put(1)
queue.put(2)
queue.put(3)
queue.put(4)
queue.put(5)
queue.put(6)

print queue.qsize()
print queue.get()
print queue.qsize()
print queue.get()
print queue.qsize()

# <codecell>

print frameGraph.neighbors(20)
# figure(); imshow(movie[:, :, :, 16+idxCorrection], interpolation='nearest')
figure(); imshow(weightMat, interpolation='nearest')
print weightMat.shape

# <codecell>

## build the range curve and turn to weights for distMat
cols = np.repeat(np.reshape(np.arange(0.0, distMat.shape[-1]), (1, distMat.shape[-1])), distMat.shape[-1], axis=0)
rows = np.repeat(np.reshape(np.arange(0.0, distMat.shape[-1]), (distMat.shape[-1], 1)), distMat.shape[-1], axis=1)

rangeWeights = np.zeros(distMat.shape)
rangeCurves = np.zeros(distMat.shape[-1])
for node in initialNodes+idxCorrection :
    ## the variances should be set according to situation range
    rangeWeights += multiGauss(cols, rows, 0, [float(node), float(node)], [5.0, 5.0])
    rangeCurves += gauss(arange(0.0, len(rangeCurves)), float(node), 5.0)

rangeWeights /= np.max(rangeWeights)
rangeCurves /= np.max(rangeCurves)
    
gwv.showCustomGraph(rangeWeights)
gwv.showCustomGraph(distMat*(1-rangeWeights))
gwv.showCustomGraph(distMat*rangeWeights)
figure(); plot(arange(0, len(rangeCurves)), rangeCurves)

# <codecell>

gaussFunc = gauss(arange(0.0, len(rangeCurves)), float(initialNodes[1]), 5.0)
figure(); plot(arange(0.0, len(rangeCurves)), gaussFunc)
step = np.zeros(len(rangeCurves))
step[arange(float(initialNodes[1])-5, float(initialNodes[1])+5, dtype=int)] = 1.0
figure(); plot(arange(0.0, len(rangeCurves)), step)
print  np.convolve(gaussFunc, step, mode='same').shape
figure(); plot(arange(0.0, len(rangeCurves)), np.convolve(gaussFunc, step, mode='same'))

# <codecell>

print np.average(weightMat), np.min(weightMat), np.max(weightMat), np.median(weightMat)
print frameGraph.nodes()
print weightMat

# <codecell>

frameGraph = digraph()
frameGraph.add_nodes(arange(1, numFrames))
for p in np.argwhere(distMat[1:distMat.shape[1], 0:-1] <= 3.0) :
    print p+1
    if tuple(p+1) not in frameGraph.edges() :
        frameGraph.add_edge(p+1)

drawGraph(frameGraph, "temp.png", True)

# <codecell>

print distMat[20, 8]

# <codecell>

idxCorrection = 0
sampleData = "ribbon2_matted/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame*.png")
maxFrames = len(frames)
frames = np.sort(frames)[0:maxFrames]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames
## load full precomputed distMat for dataset
distanceMatrix = np.load(dataFolder + sampleData + "distMat.npy")
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

frameSize = cv2.imread(frames[0]).shape
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=np.uint8)
for i in range(0, numFrames) :
    movie[:, :, :, i] = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB), dtype=np.uint8)
    
    sys.stdout.write('\r' + "Loaded frame " + np.string_(i) + " of " + np.string_(numFrames))
    sys.stdout.flush()

# <codecell>

distMat = vtu.filterDistanceMatrix(distanceMatrix, 4, True)
idxCorrection += 4
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

distMatFut = estimateFutureCost(0.999, 2.0, distM at, rangeWeights)#np.ones(distMat.shape))
figure(); imshow(distMatFut, interpolation='nearest')

# <codecell>

figure(); imshow(distMat[1:len(distMat), 0:-1], interpolation='nearest')
figure(); imshow(distMatFut, interpolation='nearest')
probMat, cumProb = vtu.getProbabilities(distMatFut, 0.005, True)#distMat[1:len(distMat), 0:-1], 0.005, True)
figure(); imshow(probMat, interpolation='nearest')
figure(); imshow(cumProb, interpolation='nearest')
finalFrames = vtu.getFinalFrames(cumProb, 300, 2, 1, False, False)
print finalFrames

# <codecell>

threshDist = np.copy(distMatFut)
print np.max(threshDist), np.min(threshDist), np.median(threshDist)
threshDist[np.where(threshDist >= np.median(threshDist))] = np.max(threshDist)
figure(); imshow(distMatFut, interpolation='nearest')
figure(); imshow(threshDist, interpolation='nearest')

# <codecell>

## get probablities and plot together with inital nodes
weightMat = distMat[1:distMat.shape[1], 0:-1] # distMatFut # threshDist
probMat, cumProb = vtu.getProbabilities(weightMat, 1.0, True)
idxCorrection =5 #+= 1

## Initiliaze graph with frames for each condition 
            ## (assume the frame is somewhere in the middle of the frames for that condition)
initialNodes = np.array([122-idxCorrection, 530-idxCorrection, 867-idxCorrection, 1100-idxCorrection]) # back, right, front, back

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(probMat, interpolation='nearest')
ax.set_autoscale_on(False)
ax.scatter(initialNodes, initialNodes, c="m", marker="s")

# <codecell>

figure(); imshow(weightMat, interpolation='nearest')

# <codecell>

print np.union1d(np.ndarray.flatten(bestForwardTransitions), np.ndarray.flatten(bestBackwardTransitions))

# <codecell>

## build frameGraph by connecting all nodes as taken from video and adding the best primitive loops
bestBackwardTransitions = findBestLoops(5, 10, weightMat, 'backward')
bestForwardTransitions = findBestLoops(5, 10, weightMat, 'forward')
print bestBackwardTransitions
print bestForwardTransitions
# print "stats"
# print np.median(np.diff(bestTransitions)), np.mean(np.diff(bestTransitions))
# print np.diff(bestTransitions)

bestTransGraph = digraph()
bestTransGraph.add_nodes(np.union1d(np.ndarray.flatten(bestForwardTransitions), np.ndarray.flatten(bestBackwardTransitions)))
# for i, j in zip(frameGraph.nodes()[0:-1], frameGraph.nodes()[1:]) :
#     if (i, j) not in frameGraph.edges() :
#         frameGraph.add_edge((i, j), wt=weightMat[i, j], label=np.str(weightMat[i, j]))
#     else :
#         frameGraph.set_edge_weight((i, j), weightMat[i, j])
#         frameGraph.set_edge_label((i, j), np.str(weightMat[i, j]))
        
for edge in bestBackwardTransitions :        
    if (edge[1], edge[0]) not in bestTransGraph.edges() :
        bestTransGraph.add_edge((edge[1], edge[0]), wt=weightMat[edge[1], edge[0]], label=np.str(weightMat[edge[1], edge[0]]))
    else :
        bestTransGraph.set_edge_weight((edge[1], edge[0]), weightMat[edge[1], edge[0]])
        bestTransGraph.set_edge_label((edge[1], edge[0]), np.str(weightMat[edge[1], edge[0]]))
        
for edge in bestForwardTransitions :        
    if (edge[1], edge[0]) not in bestTransGraph.edges() :
        bestTransGraph.add_edge((edge[1], edge[0]), wt=weightMat[edge[1], edge[0]], label=np.str(weightMat[edge[1], edge[0]]))
    else :
        bestTransGraph.set_edge_weight((edge[1], edge[0]), weightMat[edge[1], edge[0]])
        bestTransGraph.set_edge_label((edge[1], edge[0]), np.str(weightMat[edge[1], edge[0]]))

# <codecell>

## init graph
frameGraph = buildGraph(initialNodes, weightMat, 1, 5, False, bestTransGraph)

# <codecell>

## remove all nodes with 1 or no neighbours
for node in frameGraph.nodes() :
    if frameGraph.edges() < 1 :
#         bestTransGraph.del_node(node)
        print node

# <codecell>

print len(frameGraph.nodes())
print len(frameGraph.edges())
print shortPathLengths[0, :]
# gwv.showCustomGraph(shortPathLengths)

# print paths
# print frameGraph.neighbors(1264)

# <codecell>

## compute length of shortest path for each pair of nodes (uses sum of edges weight as distance measure)
shortPathLengths = np.ones(weightMat.shape, dtype=uint)*sys.float_info.max

for i in frameGraph.nodes() :
    for j in frameGraph.nodes() :
        if i != j :
            path, distance = getShortestPath(frameGraph, i, j)
            if distance > 0 : 
                shortPathLengths[i, j] = distance
    sys.stdout.write('\r' + "Computed shortest paths for frame " + np.string_(i))
    sys.stdout.flush()

print
sigma = np.mean(shortPathLengths[np.where(shortPathLengths != sys.float_info.max)])
sigma = 10.0
print 'sigma', sigma
pathProbMat = np.exp((-shortPathLengths)/sigma)
## normalize columnwise instead of rowwise as done with probability based on distMat
# normTerm = np.sum(pathProbMat, axis=1)
# normTerm = cv2.repeat(normTerm, 1, shortPathLengths.shape[1])
normTerm = np.sum(pathProbMat, axis=0)
normTerm = np.repeat(np.reshape(normTerm, (len(normTerm), 1)).T, len(normTerm), axis=0)
pathProbMat = pathProbMat / normTerm
pathProbMat[np.isnan(pathProbMat)] = 0.0
print np.sum(pathProbMat, axis=0)

# <codecell>

## compute length of shortest path for each pair of nodes (uses number of edges as distance measure)
shortPathLengths = np.ones(weightMat.shape, dtype=uint)*sys.float_info.max

for i in frameGraph.nodes() :
    paths = shortest_path(frameGraph, i)[0]
#     print paths
    for j in paths :
        if j != i :
            curr = j
            length = 0
            while curr != i :
                length += 1
                curr = paths[curr]
            shortPathLengths[i, j] = length
#             print i, j, length
    sys.stdout.write('\r' + "Computed shortest paths for frame " + np.string_(i))
    sys.stdout.flush()

print
sigma = np.mean(shortPathLengths[np.where(shortPathLengths != sys.float_info.max)])
print 'sigma', sigma
pathProbMat = np.exp((-shortPathLengths)/sigma)
normTerm = np.sum(pathProbMat, axis=1)
normTerm = cv2.repeat(normTerm, 1, shortPathLengths.shape[1])
pathProbMat = pathProbMat / normTerm
pathProbMat[np.isnan(pathProbMat)] = 0.0

# <codecell>

figure(); imshow(pathProbMat, interpolation='nearest')
figure(); imshow(shortPathLengths, interpolation='nearest')

# <codecell>

np.sum(np.array([[0, 1],[2, 3]]), axis=1)

# <codecell>

close("fig1")
## traverse graph starting from initialNode and randomize jump
print initialNodes
currentNode = initialNodes[1]
finalFrames = []
finalFrames.append(currentNode)
print currentNode
sequenceLength = 1.0
maxSeqLength = 10.0
destFrame = initialNodes[1]
for i in xrange(0, 1000) :
    print
    print "frame", i, "from", currentNode
    neighs = np.array(frameGraph.node_neighbors[currentNode], dtype=int)
#     print neighs,
#     print neighs
    probs = []
    for n in frameGraph.node_neighbors[currentNode] :
        if len(frameGraph.node_neighbors[n]) > 0:
#             probs.append(frameGraph.edge_weight((currentNode, n)))
            probs.append(probMat[currentNode, n])
        else :
            neighs = np.delete(neighs, np.argwhere(neighs == n))
    probs = np.array(probs)
#     print neighs, probs
    weights = np.ones_like(probs)
#     ## give higher probability to next timewise frame
#     if sequenceLength < randomicity :
#         weights[np.where(neighs != currentNode + 1)] = 1.0/(randomicity - sequenceLength)
#     else :
#         weights[np.where(neighs == currentNode + 1)] = 0.0

#     print weights, sequenceLength, randomicity, 
    
    ## give higher probability to frame that is closest to destination representative frame
#     print neighs
#     print shortPathLengths[destFrame, neighs]
#     print weights, weights+(1.0/shortPathLengths[destFrame, neighs])


#     weights = 1.0/np.exp(shortPathLengths[destFrame, neighs]/np.abs(currentNode-destFrame))
    
#     ## give higher probability to next timewise frame
#     if sequenceLength <= randomicity :
#         weights[np.where(neighs == currentNode + 1)] *= randomicity - sequenceLength
# #     else :
# #         weights[np.where(neighs == currentNode + 1)] *= 0.0
    
#     weights /= np.sum(weights)
# #     print weights, log(weights)
    
# #     print probs,
#     probs *= weights

    ## add the probability based on distance to destination
#     print probs, neighs, destFrame
    
    probs /= np.sum(probs)
    probs += pathProbMat[neighs, destFrame]/np.sum(pathProbMat[neighs, destFrame])
    
    ## increase probability of jumping based on how long the consequent sequence has been
    p = np.exp(-np.power(sequenceLength-maxSeqLength, 2)/(maxSeqLength*4.0))
    probs[np.where(neighs == currentNode + 1)] *= 1-p
    probs[np.where(neighs != currentNode + 1)] *= p
    
#     print weights, neighs
    probs /= np.sum(probs)
    ## pick a random node to go to next
    tmp = np.random.rand()
    randNode = np.round(np.sum(np.cumsum(probs) < tmp))
    newNode = neighs[randNode]#np.argmax(probs)]]
    
    if newNode == currentNode+1:
        sequenceLength += 1.0
    else :
        print "jump",
        sequenceLength = 1.0
    
    currentNode = newNode
    finalFrames.append(currentNode)
    print currentNode, sequenceLength, probs, pathProbMat[neighs, destFrame], neighs, tmp
    
    
figure("fig1"); plot(finalFrames, 'b', np.repeat(destFrame, len(finalFrames)), "r")

# <codecell>

# gwv.showCustomGraph(shortPathLengths)
# gwv.showCustomGraph(pathProbMat)
print getShortestPath(frameGraph, 13, initialNodes[0])
# print cumsum([ 0.80486629,  0.1009968,   0.09413692])
print pathProbMat[13, initialNodes[0]], shortPathLengths[13, initialNodes[0]]
print getShortestPath(frameGraph, 1, initialNodes[0])
print pathProbMat[1, initialNodes[0]], shortPathLengths[1, initialNodes[0]]
# print getShortestPath(frameGraph, 15, 0)

# <codecell>

tmpProbs = probMat[0, [1, 13, 12, 9]]; print tmpProbs
tmpPathProbs = pathProbMat[[1, 13, 12, 9], initialNodes[0]]; print tmpPathProbs
print (tmpProbs+tmpPathProbs)/np.sum(tmpProbs+tmpPathProbs)
print (tmpProbs/np.sum(tmpProbs)+tmpPathProbs/np.sum(tmpPathProbs))/np.sum(tmpProbs/np.sum(tmpProbs)+tmpPathProbs/np.sum(tmpPathProbs))
print 0.31028041 - 0.30664143, 0.41820428 - 0.31842044

# <codecell>

# gwv.showCustomGraph(shortPathLengths)
# gwv.showCustomGraph(pathProbMat)
print getShortestPath(frameGraph, 13, initialNodes[0])
# print cumsum([ 0.80486629,  0.1009968,   0.09413692])
print pathProbMat[13, initialNodes[0]], shortPathLengths[13, initialNodes[0]]
print getShortestPath(frameGraph, 1, initialNodes[0])
print pathProbMat[1, initialNodes[0]], shortPathLengths[1, initialNodes[0]]
print pathProbMat[:, initialNodes[0]]
# print getShortestPath(frameGraph, 15, 0)

# <codecell>

tmpSeqLen = arange(0, randomicity+1, 1)
print tmpSeqLen
tmpWeight = np.exp(-np.power(tmpSeqLen-randomicity, 2)/100.0)
print tmpWeight
figure(); plot(tmpSeqLen, tmpWeight, "b")

# <codecell>

## traverse graph starting from initialNode and take highest probability neighbour
currentNode = initialNodes[0]
print currentNode
for i in xrange(0, 300) :
    neighs = gr.node_neighbors[currentNode]
#     currentNode = neighs[np.random.randint(0, len(neighs))]
#     print currentNode
    maxWeight = -1
    curNeigh = -1
    for n in gr.node_neighbors[currentNode] :
        curWeight = gr.edge_weight((currentNode, n))
        if curWeight > maxWeight and len(gr.node_neighbors[n]) > 0:
            maxWeight = curWeight
            curNeigh = n
#         print gr.edge_weight((currentNode, n))
        
#     currentNode = neighs[np.random.randint(0, len(neighs))]
    currentNode = curNeigh
    print currentNode

# <codecell>

## try finding shortest paths
paths = shortest_path(gr, initialNodes[0])[0]

curr = initialNodes[1]
finalFrames = []
finalFrames.append(curr)
print curr
while curr != initialNodes[0] :
    curr = paths[curr]
    finalFrames.append(curr)
    print curr

# <codecell>

print initialNodes
print gr.node_neighbors[initialNodes[1]]

# <codecell>

print bestNexts
# remove loop with itself
tmpBest = np.delete(bestNexts, np.where(bestNexts == node)); print tmpBest
# remove loops smaller than minLoopLength (apart from 1-length loop which means we just go to next frame)
print tmpBest-node
print node - np.delete(bestPrevs, np.where(bestPrevs == node))

# <codecell>

print np.sort(np.array(gr.nodes())+idxCorrection)

# <codecell>

# finalFrames = np.ndarray.flatten(np.array(strongly_connected_components(subGr))+idxCorrection)
# finalFrames = np.arange(30, 274)
# finalFrames = np.array(finalFrames)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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

## visualize frames automatically
# finalFrames = arange(0, numFrames)

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
img = plt.imshow(movie[:, :, :, 0])
img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
img.set_interpolation('nearest')
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
    img.set_data(movie[:, :, :, finalFrames[0]])
#     img.set_data(movie[:, :, :, 0])
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f) + ' from ' + str(finalFrames[f]))
    img.set_data(movie[:, :, :, finalFrames[f]])
#     img.set_data(movie[:, :, :, f])
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalFrames),interval=200,blit=True)

plt.show()

# <codecell>

def update_plot(i, data, scat, img, ax):
    global movie
#     plt.clf()
#     plt.subplot(211)
#     plt.imshow(movie[:, :, :, data[i]], interpolation='nearest')
#     plt.subplot(212)
    img.set_data(movie[:, :, :, data[i]])
    ax.clear()
    ax.plot(data)
    scat = ax.scatter(i, data[i])
    return scat,

visFrames = np.array(finalFrames) + idxCorrection

x = 0
y = visFrames[0]

fig = plt.figure()
plt.subplot(211)
img = plt.imshow(movie[:, :, :, x], interpolation='nearest')
ax = plt.subplot(212)
ax.plot(finalFrames)
scat = ax.scatter(x, y)

ani = animation.FuncAnimation(fig, update_plot, frames=xrange(len(visFrames)),
                              fargs=(visFrames, scat, img, ax))
plt.show()

