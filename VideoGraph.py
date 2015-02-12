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

rangeCurves = np.zeros(1272)
initialNodes = np.array([30, 142, 516, 838, 1106])-5
intervals = np.array([45, 211, 461, 169, 339])

for node, inter in zip(initialNodes, intervals) :
    ## the variances should be set according to situation range
#     rangeCurves += gauss(arange(0.0, len(rangeCurves)), float(node), 5.0)
    rangeCurves += smoothStep(arange(0.0, len(rangeCurves)), float(node), inter, 0.4)
# rangeCurves /= np.max(rangeCurves)
figure(); plot(arange(0, len(rangeCurves)), rangeCurves)
print rangeCurves

# <codecell>

print initialNodes+1

# <codecell>

range1DClasses = np.ones((80, 1272, 3))
## show classes using same color coding as ribbon ranges from the label propagation
range1DClasses[:, :, 0] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[0]), intervals[0], 0.4)
range1DClasses[:, :, 2] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[0]), intervals[0], 0.4)

range1DClasses[:, :, 1] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[1]), intervals[1], 0.4)
range1DClasses[:, :, 2] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[1]), intervals[1], 0.4)

range1DClasses[:, :, 0] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[2]), intervals[2], 0.4)
range1DClasses[:, :, 2] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[2]), intervals[2], 0.4)

range1DClasses[:, :, 0] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[3]), intervals[3], 0.4)
range1DClasses[:, :, 1] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[3]), intervals[3], 0.4)

range1DClasses[:, :, 1] -= smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[4]), intervals[4], 0.4)

figure(); imshow(range1DClasses, interpolation='nearest')
axis('off')

# <codecell>

print np.max(range1DClasses)

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

# print getShortestPath(frameGraph, 0, 400)
print shortest_path(frameGraph, 0)[1].keys()

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
        printout = '\r' + "Iteration " + np.string_(i+1) + "(" + np.string_(len(gr.nodes())) + ", " + np.string_(len(gr.edges())) + "); "
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
#             print node, bestTransitions
            
            for bt in bestTransitions :
                ## check timewise neighbors of node for transitions to bt
                neighSize = 5
                neighs = arange(node-neighSize, node+neighSize+1, dtype=int)
                btWeight = weights[node, bt]
#                 print "weight", btWeight

                addEdge = True
                for neigh in np.delete(neighs, np.where(neighs == node)) :
                    if (neigh, bt) in gr.edges() : 
                        if gr.edge_weight((neigh, bt)) >= btWeight :
#                             print "remove edge", neigh, bt, gr.edge_weight((neigh, bt))
                            gr.del_edge((neigh, bt))
                        else :
#                             print "found better connected neighbour", neigh, bt, gr.edge_weight((neigh, bt))
                            addEdge = False
                
                if addEdge :
#                     print "new edge", node, bt
                    if bt not in gr.nodes() :
                        gr.add_node(bt)
                    
                    if (node, bt) not in gr.edges() :
                        gr.add_edge((node, bt), wt=btWeight, label=np.str(btWeight))
                    else :
                        gr.set_edge_weight((node, bt), btWeight)
                        gr.set_edge_label((node, bt), np.str(btWeight))
            
    drawGraph(gr, "frameGraph.png", drawOnPng)
    return gr

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
    
    q = Queue()
    for node in gr.nodes():
        q.put(node)
        
#     for i in xrange(0, minMaxIterations[1]) :
#         ## for each node find the best node to go to next and the best node to get there from
#         printout = '\r' + "Iteration " + np.string_(i+1) + "(" + np.string_(len(gr.nodes())) + ", " + np.string_(len(gr.edges())) + "); "
#         ## check if all initNodes have been connected by graph
#         finished = True
#         ## check initNodes are connected
#         for i1, i2 in zip(initNodes, np.roll(initNodes, 1)) :
# #             print "t", i1, i2
#             if i2 in shortest_path(gr, i1)[0] and i1 in shortest_path(gr, i2)[0] :
#                 printout += "conn " + np.string_(i1) + "-" + np.string_(i2) + " "
#             else :
#                 finished = False
#         sys.stdout.write(printout)
#         sys.stdout.flush()
        
#         if finished and i >= minMaxIterations[0]: 
#             print "finished"
#             break
        
    while not q.empty() :
        
        sys.stdout.write("\rNodes remaining " + np.string_(q.qsize()))
        sys.stdout.flush()
        
        node = q.get()
        ## add node for frame node-1 and connect to node
        if node-1 >= 0 :
            if node-1 not in gr.nodes() :
                gr.add_node(node-1)
                q.put(node-1)
                
            if (node-1, node) not in gr.edges() :
                gr.add_edge((node-1, node), wt=weights[node-1, node], label=np.str(weights[node-1, node]))
            else :
                gr.set_edge_weight((node-1, node), weights[node-1, node])
                gr.set_edge_label((node-1, node), np.str(weights[node-1, node]))
        
        ## add node for frame node+1 and connect to node
        if node+1 < weights.shape[-1] :
            if node+1 not in gr.nodes() :
                gr.add_node(node+1)
                q.put(node+1)
            
            if (node, node+1) not in gr.edges() :
                gr.add_edge((node, node+1), wt=weights[node, node+1], label=np.str(weights[node, node+1]))
            else :
                gr.set_edge_weight((node, node+1), weights[node, node+1])
                gr.set_edge_label((node, node+1), np.str(weights[node, node+1]))
        
        ## order transitions based on weights and keep n best s.t. n is higher the further away from center
        bestTransitions = np.argsort(weights[node, :])
        bestTransitions = bestTransitions[np.where(np.abs(bestTransitions-node) > minLoopLength)]
        bestTransitions = bestTransitions[0:5*(1-ranges[node])]
#             print node, bestTransitions
        
        for bt in bestTransitions :
            ## check timewise neighbors of node for transitions to bt
            neighSize = 5
            neighs = arange(node-neighSize, node+neighSize+1, dtype=int)
            btWeight = weights[node, bt]
#                 print "weight", btWeight

            addEdge = True
            for neigh in np.delete(neighs, np.where(neighs == node)) :
                if (neigh, bt) in gr.edges() : 
                    if gr.edge_weight((neigh, bt)) >= btWeight :
#                             print "remove edge", neigh, bt, gr.edge_weight((neigh, bt))
                        gr.del_edge((neigh, bt))
                    else :
#                             print "found better connected neighbour", neigh, bt, gr.edge_weight((neigh, bt))
                        addEdge = False
            
            if addEdge :
#                     print "new edge", node, bt
                if bt not in gr.nodes() :
                    gr.add_node(bt)
                    q.put(bt)
                
                if (node, bt) not in gr.edges() :
                    gr.add_edge((node, bt), wt=btWeight, label=np.str(btWeight))
                else :
                    gr.set_edge_weight((node, bt), btWeight)
                    gr.set_edge_label((node, bt), np.str(btWeight))
                    
                ## add the node the opposite way
                if (bt, node) not in gr.edges() :
                    gr.add_edge((bt, node), wt=weights[bt, node], label=np.str(weights[bt, node]))
                else :
                    gr.set_edge_weight((bt, node), weights[bt, node])
                    gr.set_edge_label((bt, node), np.str(weights[bt, node]))
            
    drawGraph(gr, "frameGraph.png", drawOnPng)
    return gr

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

weighMat = distMatFut
# weightMat = distMat[1:distMat.shape[1], 0:-1]
# weightMat = distMat[0:-1, 1:distMat.shape[1]]
figure(); imshow(weightMat, interpolation='nearest')

probMat, cumProb = vtu.getProbabilities(weightMat, 1.0, None, True)
idxCorrection =5 #+= 1

## initNodes and ranges for star dot
# initialNodes = np.array([9, 21])-idxCorrection
# intervals = np.array([11, 11])
## initNodes and ranges for ribbon2_matted
initialNodes = np.array([122, 501, 838, 1106])-idxCorrection
intervals = np.array([251, 441, 169, 339])
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(probMat, interpolation='nearest')
# ax.set_autoscale_on(False)
# ax.scatter(initialNodes, initialNodes, c="m", marker="s")

## compute rangeCurves using initialNodes before idxCorrection
rangeCurves = np.zeros(weightMat.shape[-1])
for node, inter in zip(initialNodes, intervals) :
    ## the variances should be set according to situation range
#     rangeCurves += gauss(arange(0.0, len(rangeCurves)), float(node), 5.0)
    rangeCurves += smoothStep(arange(0.0, len(rangeCurves)), float(node), inter, 0.4)
# rangeCurves /= np.max(rangeCurves)
figure(); plot(arange(0, len(rangeCurves)), rangeCurves)
print rangeCurves


# frameGraph = buildGraph(initialNodes, weightMat, rangeCurves, 1, 3, (1, 1), True, None)
frameGraph = buildGraph(initialNodes, 1.0/probMat, rangeCurves, 1, 3, (10, 100), False, None)

print
print len(frameGraph.nodes()), frameGraph.nodes()
print len(frameGraph.edges()), frameGraph.edges()

# <codecell>

for i1, i2 in zip(initialNodes, np.roll(initialNodes, 1)) :
    if i2 in shortest_path(frameGraph, i1)[0] and i1 in shortest_path(frameGraph, i2)[0] :
        print "connected", i1, i2

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
distanceMatrix = np.load(dataFolder + sampleData + "vanilla_distMat.npy")
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

kernel = np.reshape(rangeCurves, (len(rangeCurves), 1))
kernel = kernel*kernel.T
# figure(); imshow(kernel, interpolation='nearest')
print kernel.shape, distMat.shape

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
#             m = np.min(distMat*weights[1:distanceMatrixFilt.shape[1], 0:-1], axis=1)
            m = np.min(distMat*weights, axis=1)
            distMat[i, :] = (distMatFilt[i, :] ** p) + alpha*m
            
        last = np.copy(current)
        current = np.copy(distMat)
        
        sys.stdout.write('\r' + "Iteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
        sys.stdout.flush()
        
        iterations += 1
    
    print
    print 'finished in', iterations, 'iterations'
    
    return distMat

distMatFut = estimateFutureCost(0.999, 2.0, distMat, kernel)#np.ones(distMat.shape))#rangeWeights)
figure(); imshow(distMatFut, interpolation='nearest')

# <codecell>

p, c = vtu.getProbabilities(distMatFut, 0.02, None, True)
figure(); imshow(c, interpolation='nearest')

# <codecell>

## compute length of shortest path for each pair of nodes (uses sum of edges weight as distance measure)
shortPathLengths = np.ones(weightMat.shape, dtype=uint)*sys.float_info.max

for i in frameGraph.nodes() :
    shtpa = shortest_path(frameGraph, i)
    shortPathLengths[i, shtpa[1].keys()] = shtpa[1].values()
#     for j in frameGraph.nodes() :
#         if i != j :
#             path, distance = getShortestPath(frameGraph, i, j)
#             if distance > 0 : 
#                 shortPathLengths[i, j] = distance
    sys.stdout.write('\r' + "Computed shortest paths for frame " + np.string_(i))
    sys.stdout.flush()

print
sigma = np.mean(shortPathLengths[np.where(shortPathLengths != sys.float_info.max)])
sigma = np.mean(intervals)/2.0#100.0
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

figure(); imshow(pathProbMat, interpolation='nearest')

# <codecell>

close("fig1")
## traverse graph starting from initialNode and randomize jump
print initialNodes
currentNode = initialNodes[3]
finalFrames = []
finalFrames.append(currentNode)
print currentNode
sequenceLength = 1.0
maxSeqLength = 100.0
destFrame = initialNodes[3]
destRange = smoothStep(arange(0.0, len(rangeCurves)), float(initialNodes[0]), intervals[0], 0.4)
# figure(); plot(arange(0.0, len(rangeCurves)), destRange)
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

    ## add the probability based on distance to destination
#     print probs, neighs, destFrame
    
    probs /= np.sum(probs)
    
    pathProbs = pathProbMat[neighs, destFrame]
    pathProbs = pathProbs/np.sum(pathProbs)
    
    if not np.isnan(pathProbs).all() :
        probs += pathProbs
    
    probs *= (1+100*destRange[neighs])
    
    ## increase probability of jumping based on how long the consequent sequence has been
    p = np.exp(-np.power(sequenceLength-maxSeqLength, 2)/(maxSeqLength*4.0))
    probs[np.where(neighs == currentNode + 1)] *= 1-p
    probs[np.where(neighs != currentNode + 1)] *= p
    
#     probs = np.power(probs, 1+destRange[neighs])
    
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
    print currentNode, sequenceLength, probs, pathProbMat[neighs, destFrame], destRange[neighs], neighs, tmp
    
    
figure("fig1"); 
for iN in initialNodes :
    plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*iN, 'g')
plot(finalFrames, 'b', np.repeat(destFrame, len(finalFrames)), "r")

# <codecell>

figure(); plot(arange(0.0, len(rangeCurves)), destRange*100)

# <codecell>

import bisect
def shortestPathFromTo(graph, source, weights):
    # Initialization
    dist     = {source: 0}
    previous = {source: None}

    # This is a sorted queue of (dist, node) 2-tuples. The first item in the
    # queue is always either a finalized node that we can ignore or the node
    # with the smallest estimated distance from the source. Note that we will
    # not remove nodes from this list as they are finalized; we just ignore them
    # when they come up.
    q = [(0, source)]

    # The set of nodes for which we have final distances.
    finished = set()

    # Algorithm loop
    while len(q) > 0:
        du, u = q.pop(0)

        # Process reachable, remaining nodes from u
        if u not in finished:
            finished.add(u)
#             print u, "tra"
            for v in xrange(0, len(graph.nodes())):#graph[u]:
#                 print v, 
                if v not in finished:
                    alt = du + weights[u, v]#graph.edge_weight((u, v))
                    if (v not in dist) or (alt < dist[v]):
                        dist[v] = alt
                        previous[v] = u
                        bisect.insort(q, (alt, v))
#                 print

    return previous, dist

# <codecell>

## try and find random path from src to tgt
src = initialNodes[0]
tgt = initialNodes[1]
curr = src
prev = curr
randPath = [src]
cost = 0.0
while curr != tgt :
    curr = np.random.randint(0, probMat.shape[-1])
    randPath.append(curr)
    cost += (1.0/probMat)[curr, prev]
    prev = curr

print cost    
print len(randPath), randPath

# <codecell>

print frameGraph.neighbors(0)
# print frameGraph.nodes()
p, d = shortestPathFromTo(frameGraph, 0, 1.0/probMat)
print p
print d

# <codecell>

## find shortest paths between initNodes in fully connected graph
fullGraph = digraph()
fullGraph.add_nodes(arange(0, weightMat.shape[-1], dtype=int))

for i in xrange(0, len(fullGraph.nodes())) :
    for j in xrange(0, len(fullGraph.nodes())) :
        fullGraph.add_edge((i, j), wt=(1.0/probMat)[i, j])
    print i, 

print len(fullGraph.nodes())
print len(fullGraph.edges())

# <codecell>

print len(fullGraph.edges())
print fullGraph.edge_weight((0, 5)), probMat[0, 3]

# <codecell>

# figure(); imshow(shortPathLengths, interpolation='nearest')
print probMat[1101, 1102], pathProbMat[1102, initialNodes[3]]
print getShortestPath(frameGraph, initialNodes[3], initialNodes[2])

# <codecell>

# finalFrames = np.ndarray.flatten(np.array(strongly_connected_components(subGr))+idxCorrection)
# finalFrames = np.arange(30, 274)
# finalFrames = np.array(finalFrames)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# <codecell>

def update_plot(i, data, scat, img, ax):
#     plt.clf()
#     plt.subplot(211)
#     plt.imshow(movie[:, :, :, data[i]], interpolation='nearest')
#     plt.subplot(212)
    img.set_data(movie[:, :, :, data[i]])
    ax.clear()
    ax.plot(data)
    for iN in initialNodes :
        ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*iN, 'g')
    ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*destFrame, 'r')
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
for iN in initialNodes :
    ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*iN, 'g')
ax.plot(np.arange(0, len(finalFrames)), np.ones(len(finalFrames))*destFrame, 'r')
scat = ax.scatter(x, y)

ani = animation.FuncAnimation(fig, update_plot, frames=xrange(len(visFrames)),
                              fargs=(visFrames, scat, img, ax), interval=33)
plt.show()

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

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalFrames),interval=33,blit=True)

plt.show()

