# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import re
import cv2
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import MazeSequenceUtils as msu

dataLoc = "/home/ilisescu/PhD/iPy/data/longmaze/"
nameLength = len(filter(None, re.split('/',dataLoc)))
nvmFile = "sparse.nvm"

colVals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

# <codecell>

cameras, cameraNames = msu.readCameraLocsNVM(dataLoc, nvmFile)
distanceMatrix =msu.getDistanceMatrix(dataLoc, cameraNames)

# figure(0)
# plt.imshow(distanceMatrix, cmap = cm.Greys_r, interpolation='nearest')
# figure(1)
# plt.imshow(distanceMatrix, interpolation='nearest')

# <codecell>

## project 3d camera locations onto best fitting plane and fit as many lines as possible

cameraLocs2D = np.zeros([len(cameras), 2])
## project all camera locations onto best fitting plane
plane = msu.fitPlane(cameras[:, 4:7])

n = plane[0:3]
p = plane[3:6]
print "fitted plane:", plane
# p = np.array([0.0, 0.0, 0.0]) ## project all camera locations on XZ plane (in vsfm up is Y???)
# n = np.array([0.0, 1.0, 0.0])

idx = 0
for cc in cameras[:, 4:7] :
    cc2D = cc - np.dot(cc - p, n)*n # project 3D point
#     print cc2D, cc
#     cameraLocs2D[idx] = [cc2D[0]/cc2D[1], cc2D[2]/cc2D[1]]
    cameraLocs2D[idx] = cc2D[[0, 2]] # after project cc onto fitted plane, discard Y since cc is being projected on XZ plane, the Y coord will be 0 so only take X and Z coord
    idx += 1

## look for all the lines 
points = cameraLocs2D
linesInliers = list()
## each line in line is a 6D array where first 3 elements give the line direction and last 3 elements give a point on line
lines = list()
while len(points) > 1 : # can't find lines with less than 2 points
    fittedLine, inliers, bestCameras = msu.ransac(points, 1.0, 500)
    if len(inliers) > 1 :
        line = msu.fitLine2D(cameraLocs2D[inliers])#cv2.fitLine(np.array(cameraLocs2D[inliers], dtype=float32), cv.CV_DIST_L2, 0, 0.01, 0.01)
        lines.append(line)
        linesInliers.append(inliers)
        points = cameraLocs2D[np.delete(np.arange(0, len(cameraLocs2D), 1), np.hstack(linesInliers))]
        
        
# order line segments based on time (i.e. first segment is the one assigned to the first sequence of frames)
sums = np.zeros(len(lines))
for i in xrange(0, len(lines)) :
    sums[i] = np.sum(linesInliers[i])
order = np.argsort(sums)
lines = np.array(lines)[order]
linesInliers = np.array(linesInliers)[order]

# <codecell>

print lines

# <codecell>

## find which camera belongs to which segment based on distance and slope
# find slopes of found lines
slopes = np.zeros(len(lines))
for i in xrange(0,len(lines)) :
    # find 2 points on line
    p1 = lines[i][2:4]+lines[i][0:2]
    p2 = lines[i][2:4]-lines[i][0:2]
    slopes[i] = (p1[1] - p2[1])/(p1[0] - p2[0])
    
print 'slopes:', slopes


## compute average distance of camera locations from their assinged line
distances = list()
for i in xrange(0, len(lines)) :
    for loc in linesInliers[i] :
        a = np.ndarray.flatten(lines[i][2:4])
        n = np.ndarray.flatten(lines[i][0:2])
        p = cameraLocs2D[loc]
        distances.append(np.linalg.norm(a - p - np.dot(np.dot(a-p, n), n))) # distance between point p and line x = a +tn


## find cameras that after fitting lines using L2 are outside ransac threshold
distThresh = 3*np.average(distances)
# contains all the cameras that are outliers (according to ransac distance threshold) after fitting of lines using L2 min
outliers = list()

# for idx in xrange(0, len(cameraLocs2D)) :
#     isOutlier = True
#     for line in lines :
#         a = np.ndarray.flatten(line[2:4])
#         n = np.ndarray.flatten(line[0:2])
#         p = cameraLocs2D[idx]
#         distance = np.linalg.norm(a - p - np.dot(np.dot(a-p, n), n)) # distance between point p and line x = a +tn
#         if distance < distThresh :
#             isOutlier = False
#         if idx == 215 :
#             print "gne", distance, distThresh
#     if isOutlier :
#         outliers.append(idx)
print distThresh
for i in xrange(0, len(lines)) :
    for cam in linesInliers[i] :
#         print cam,
        a = lines[i][2:4]
        n = lines[i][0:2]
        p = cameraLocs2D[cam]
        distance = np.linalg.norm(a - p - np.dot(np.dot(a-p, n), n)) # distance between point p and line x = a +tn
#         print distance
        if distance > distThresh :
            outliers.append(cam)
        
print 'outliers:', outliers

# <codecell>

## for each outlier check which line they are closer to and add them to their list of inliers
# cameras belonging to each fitted line
camerasForLine = np.copy(linesInliers)

## this will probably fail if the camera locations are not following maze path but follow irregular path
for outlier in outliers :
    minDist = np.finfo(float).max
    bestFit = -1
    for i in xrange(0, len(lines)) :
        # make sure this line doesn't contain outlier from ransac fitting
        if outlier in linesInliers[i] :
            camerasForLine[i] = np.delete(camerasForLine[i], np.where(camerasForLine[i] == outlier))
        a = np.ndarray.flatten(lines[i][2:4])
        n = np.ndarray.flatten(lines[i][0:2])
        p = cameraLocs2D[outlier]
        distance = np.linalg.norm(a - p - np.dot(np.dot(a-p, n), n))
        if distance < minDist :
            minDist = distance
            bestFit = i
    camerasForLine[bestFit] = np.hstack((camerasForLine[bestFit], [outlier]))

## sort camerasForLine
for l in xrange(0, len(camerasForLine)) :
    camerasForLine[l] = np.sort(camerasForLine[l])
# print camerasForLine

# <codecell>

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
cols = ['b', 'g']
## find distance from each camera location to the turn by first projecting them onto the line fitted to them
## assumes that camera locations are sorted
projectedCameraLocs2D = np.zeros_like(cameraLocs2D) ## camera locations projected onto their respective fitted lines
for l in xrange(0, len(camerasForLine)) :
    lastCameraLoc = cameraLocs2D[camerasForLine[l][-1]]
    lastCameraLoc = lastCameraLoc#/np.linalg.norm(lastCameraLoc)
    ax.scatter([lastCameraLoc[0]], [lastCameraLoc[1]], c=cols[l], marker='o')
    u = lines[l][0:2]
    for c in xrange(0, len(camerasForLine[l])) :
        loc = cameraLocs2D[camerasForLine[l][c]]
        ## project point onto vector in the line direction u
        pLoc = (np.dot(loc, u)/np.dot(u, u))*u
        projectedCameraLocs2D[camerasForLine[l][c]] = pLoc

cameraDistsToTurn = np.zeros_like(camerasForLine)
for l in xrange(0, len(camerasForLine)) :
    cameraDistsToTurn[l] = np.zeros(camerasForLine[l].shape, dtype=float)
    for c in xrange(0, len(camerasForLine[l])) :
        cameraDistsToTurn[l][c] = np.linalg.norm(projectedCameraLocs2D[camerasForLine[l][c]]-projectedCameraLocs2D[camerasForLine[l][-1]])

## plot lines and projected camera locs
for l in xrange(0, len(lines)) :
    a = lines[l][2:4]
    u = lines[l][0:2]
    for loc, pLoc in zip(cameraLocs2D[camerasForLine[l]], projectedCameraLocs2D[camerasForLine[l]]):
        
        lineNormal = (pLoc-loc)/np.linalg.norm(pLoc-loc)
        
        distanceToLine = np.linalg.norm(a - pLoc - np.dot(np.dot(a-pLoc, u), u))
        tmp = pLoc - distanceToLine*lineNormal
        ax.scatter([tmp[0]], [tmp[1]], c=cols[l], marker='x')
    p1 = a+200*u
    p2 = a-200*u
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=cols[l])

ax.scatter([0], [0], c='r', marker='x')
    
ax.set_xlabel('X')
ax.set_ylabel('Y')

## set axis range
ax.set_autoscale_on(False)
# ax.set_xlim([-35.0,-25.0])
# ax.set_ylim([-30.0,-20.0])
ax.set_xlim([-40.0,20.0])
ax.set_ylim([-40.0,20.0])

plt.show()

# <codecell>

## HACK : initialize some mock input 2d maze map (eventually it would come from drawn input or whatevs)

mazeMap = np.array([[0.0, 0.0], [0.0, 20.0], [7.0, 20.0], [7.0, 5.0], [2.0, 5.0]])

seqSegments = np.zeros([len(camerasForLine), 2]) ## contains length for each video sequence segment and associated length in frames
mapSegments = np.zeros([len(mazeMap)-1, 2]) ## contains length for each map segment and associated length in frames

for i in xrange(0, len(seqSegments)) :
    seqSegments[i, 0] = cameraDistsToTurn[i][0] ## distance of first frame to turn is length of segment
    seqSegments[i, 1] = len(camerasForLine[i])
maxSeqSegmentLen = np.max(seqSegments[:, 0]);

for i in xrange(1, len(mazeMap)) :
    mapSegments[i-1, 0] = np.linalg.norm(mazeMap[i-1]-mazeMap[i])
maxMapSegmentLen = np.max(mapSegments[:, 0]);

print maxSeqSegmentLen, maxMapSegmentLen

## find max length of video sequence segments in number of frames
maxFramesPerSegment = np.int(np.ndarray.flatten(seqSegments[np.where(seqSegments[:, 0]==maxSeqSegmentLen), 1]))
print maxFramesPerSegment

## find length in frames for each map segment
for i in xrange(0, len(mapSegments)) :
    mapSegments[i, 1] = np.round(mapSegments[i, 0]/maxMapSegmentLen*maxFramesPerSegment)

print seqSegments
print mapSegments

# <codecell>

## Turn distances to probabilities
def dist2prob(dM, normalize) :
    sigma = 0.2*np.mean(dM[np.nonzero(dM)])
    print 'sigma', sigma
    pM = np.exp((-dM)/sigma)
## proba at i,j depends on distance at i+1,j
    pM = np.vstack([pM[1:pM.shape[0], :], np.zeros([1, pM.shape[1]])])

#     pM = np.exp((1-np.roll(distMat, 1, axis=1))/sigma)
#     pM = pM[range(2, )]
## normalize probabilities row-wise
    if normalize :
        normTerm = np.sum(pM, axis=1)
        normTerm = cv2.repeat(normTerm, 1, dM.shape[1])
        pM = pM / normTerm
    return pM

## returns a diagonal kernel based on given binomial coefficients
def diagkernel(c) :
    k = np.eye(len(c))
    k = k * c/np.sum(c)
    return k

## Preserve dynamics: convolve wih binomial kernel
kernel = diagkernel(np.array([1, 4, 6, 4, 1], dtype=np.float))
print kernel
filteredDistMatrix = cv2.filter2D(distanceMatrix, -1, kernel)


## find the 
finalFrames = np.zeros(np.sum(mapSegments[:, 1]))
print finalFrames.shape
probMatrix = dist2prob(filteredDistMatrix, True)

scale = maxSeqSegmentLen/maxMapSegmentLen
frameIdx = 0
for i in xrange(0, len(mapSegments)) :
    framesNeeded = np.int(mapSegments[i, 1])
    for f in xrange(0, framesNeeded) :
        distToTurn = mapSegments[i, 0] - f*mapSegments[i, 0]/(framesNeeded-1)
        minDist = np.max([maxMapSegmentLen, maxSeqSegmentLen])+1
        bestFrame= -1
        bestDistFrames = np.zeros([len(seqSegments), 2]) # for sequence j; [j, 0] = minDist; [j, 1] = bestFrame
        for j in xrange(0, len(seqSegments)) :
            scaledDist = distToTurn*scale
            hackMod = 0
            if j == 1 : ## HACK: second segment doesn't actually go to the turn so add some extra distance
                hackMod += 10.0
            dist = np.min(np.abs(cameraDistsToTurn[j]-scaledDist+hackMod)) 
            bestDistFrames[j, 0] = dist
            bestDistFrames[j, 1] = camerasForLine[j][np.where(np.abs(cameraDistsToTurn[j]-scaledDist+hackMod) == dist)]
        
        if frameIdx > 0 : ## check distanceMatrix for dist between possible frames
#             print distanceMatrix[finalFrames[frameIdx-1], np.array(bestDistFrames[:, 1], dtype=int)]
            print "1", finalFrames[frameIdx-1], np.array(bestDistFrames[:, 1], dtype=int), probMatrix[finalFrames[frameIdx-1], np.array(bestDistFrames[:, 1], dtype=int)]
            print "2", bestDistFrames
            bestDistFrames[:, 0] = bestDistFrames[:, 0]/probMatrix[finalFrames[frameIdx-1], np.array(bestDistFrames[:, 1], dtype=int)]
        bestFrame = bestDistFrames[np.where(bestDistFrames[:, 0] == np.min(bestDistFrames[:, 0])), 1]
        print "la", bestDistFrames
        finalFrames[frameIdx] = bestFrame
        if frameIdx != np.sum(mapSegments[0:i, 1])+f :
            print frameIdx, np.sum(mapSegments[0:i, 1])+f
            raise ValueError("Inconsistent final frame index")
        frameIdx += 1
#         print bestFrame
#     print np.sum(mapSegments[0:i, 1])

# <codecell>

frames = finalFrames
for f in frames :
    print f,
# print straights[0]

# <codecell>

frames = finalFrames
for f in frames :
    print f,
# print straights[0]

# <codecell>

print finalFrames

# <codecell>

## HACK : now take 10% of frames from the end of each straight (inliers of fitted lines) to add to turns (outliers)

straights = list()
turns = list()

## each line corresponds to 1 straight
for i in xrange(1, len(lines)) :
    # idx of first outlier camera in inlier array for first straight
    idx = np.argwhere(linesInliers[i-1] == outliers[0])
    if len(idx) == 0 :
        idx = len(linesInliers[i-1])
    # length of first straight
    l = np.int(np.round(len(linesInliers[i-1])-idx - 0.1*len(linesInliers[i-1])))
    # add first straight
    straights.append(linesInliers[i-1][0:l])
    
    # build turn
    turn = np.hstack([linesInliers[i-1][l:], outliers])
    
    # idx of last outlier camera in inlier array for second straight
    idx = np.argwhere(linesInliers[1] == outliers[len(outliers)-1])
    if len(idx) == 0 :
        idx = len(linesInliers[i])
    # length of second straight
    l = np.int(np.round(len(linesInliers[i])-idx - 0.1*len(linesInliers[i])))
    
    # add turn ## HACK : using unique because I don't know ot which linesInliers array the 
               ## outliers belong to so they get added twice 
               ## (once from outliers and once from whichever linesInliers they belong to)
    turn = np.unique(np.hstack([turn, linesInliers[i][0:len(linesInliers[i])-l]]))
    turns.append(turn)
    
    # add second straight
    straights.append(linesInliers[i][len(linesInliers[i])-l:len(linesInliers[i])])

print straights
print turns

# <codecell>

## HACK : initialize some mock input 2d maze map (eventually it would come from drawn input or whatevs)

mazeMap = np.array([[0.0, 0.0], [0.0, 20.0], [7.0, 20.0], [7.0, 5.0], [2.0, 5.0]])

## now define 2 graphs where nodes are turns and edges are straights
# one for the input sequences 
# one for the input maze

## init sequence graph edges with straights' length in frames
seqEdges = np.zeros(len(straights))
for i in xrange(0, len(straights)) :
    seqEdges[i] = len(straights[i])
maxSequenceLen = np.max(seqEdges);
    
## init sequence graph nodes with [i, 0] = seq length, [i, 1] = left/right turn, [i, 2:len] = edges connected to node
seqNodes = np.zeros([len(turns), 4])
for i in xrange(0, len(turns)) :
    seqNodes[i, 0] = len(turns[i])
    ## HACK : need auto way of finding which way the turn turns
    seqNodes[i, 1] = 1
    ## HACK : need auto way of finding which straights are connected to turn
    seqNodes[i, 2] = 0
    seqNodes[i, 3] = 1

## init map graph edges with [i, 0] = segment length, [i, 1:len] = extremes from mazeMap
mapEdges = np.zeros([len(mazeMap)-1, 3])
## init map graph nodes with [i, 0] = left/right turn, [i, 1:len] = edges connected to node
mapNodes = np.zeros([len(mapEdges)-1, 3])

maxSegmentLen = 0
currentSegment = 0
for i in xrange(1, len(mazeMap)) :
    ## find the longest length of any given segment
    current = np.linalg.norm(mazeMap[i]-mazeMap[i-1])
    if current > maxSegmentLen :
        maxSegmentLen = current
    
    ## set the extremes of edge
    mapEdges[i-1, 1] = i-1
    mapEdges[i-1, 2] = i
    
    ## only add node if there's already an edge
    if i > 1 :
        mapNodes[i-2, 0] = 1
        mapNodes[i-2, 1] = currentSegment-1
        mapNodes[i-2, 2] = currentSegment
        
    currentSegment += 1

## set edge lenght for map graph
for i in xrange(0, len(mapEdges)) :
    mapEdges[i, 0] = np.round(np.linalg.norm(mazeMap[mapEdges[i, 1]]-mazeMap[mapEdges[i, 2]])*maxSequenceLen/maxSegmentLen)

print mapNodes    
print mapEdges
print maxSegmentLen, maxSequenceLen

# <codecell>

## build movie based on map graph

# go through all the nodes and get ordered list of straights and their length
# each node is as long as the sequence node they are replaced by (for now only 1 node so no need to worry which is best)
bestSeqNode = 0
nodeLen = seqNodes[bestSeqNode, 0] ## lenght of turn in frames
frames = np.zeros(np.sum(mapEdges[:, 0])+nodeLen*len(mapNodes))
print frames.shape
## add the frames representing the turns to the frames array
currentFrame = 0 
straightStartFrame = np.zeros(len(mapEdges))
count = 1
for n in mapNodes :
    prevLen = mapEdges[n[1], 0]
    nextLen = mapEdges[n[2], 0]
    frames[currentFrame+prevLen:currentFrame+prevLen+nodeLen] = turns[bestSeqNode]
    currentFrame = currentFrame+prevLen+nodeLen
    straightStartFrame[count] = currentFrame
    count +=1

print straightStartFrame

n = 0
for e in xrange(0, len(mapEdges)) :
    # find which straight sequence connects best with turn 
    minDist = np.max(distanceMatrix)
    startSequence = -1;
    for s in xrange(0, len(straights)) :
        ## need to check bestSeqNode which still needs to be found automatically somehow
        dist = distanceMatrix[straights[s][-1], turns[bestSeqNode][0]]
        if dist <= minDist :
            minDist = dist
            startSequence = s
    print startSequence
    
    ## deal with joint between end of straight and beginning of turn
    
    ## check for all sequence edges that are long enough for current map edge
    viableSequences = np.hstack(np.where(seqEdges >= mapEdges[e, 0])) # i.e long enough sequences
    startSequenceLen = len(straights[startSequence])
    if startSequence in viableSequences :
        print "using best sequence, start at frame #", straightStartFrame[e]
        startTurnFrame = straightStartFrame[e] + mapEdges[e, 0]
        frames[straightStartFrame[e]:startTurnFrame] = straights[startSequence][startSequenceLen-mapEdges[e, 0]:startSequenceLen]
    else :
        print "find best fitting, start at frame #", straightStartFrame[e]
        bestFrame = -1
        bestSequence = -1
        minDist = np.max(distanceMatrix)
        ## check which viable sequence fits best the startSequence and where
        for s in viableSequences :
            print "lala", mapEdges[e, 0], startSequenceLen, seqEdges[s]
            framesNeeded = mapEdges[e, 0] ## frames needed for current map edge e
            for toUse in xrange(1, startSequenceLen) : ## frames from start sequence to be used
                additionalNeeded = framesNeeded-toUse ## additional frames needed to be taken from viable sequence
                remaining = seqEdges[s] - additionalNeeded ## frames from viable sequence remaining that can be chosen from to stitch to start sequence
                dist = np.min(distanceMatrix[straights[startSequence][-toUse-1], straights[s][len(straights[s])-remaining-1:len(straights[s])]])
                print toUse, additionalNeeded, remaining, dist, straights[startSequence][-toUse-1]#, straights[s][remaining-1:len(straights[s])]
                if dist <= minDist :
                    minDist = dist
                    bestFrame = straights[startSequence][-toUse-1]
                    bestSequence = s
        bestMatch = np.int(np.hstack(np.where(distanceMatrix[bestFrame, :] == minDist)))
        print "best matching frames (", cameraNames[bestFrame], ",", cameraNames[bestMatch], ")"
        print np.where(straights[bestSequence]==bestMatch)
        print "best frames to stitch together (", cameraNames[straights[startSequence][np.int(np.hstack(np.where(straights[startSequence]==bestFrame)))+1]],
        print ",", cameraNames[straights[bestSequence][np.where(straights[bestSequence]==bestMatch)]], ")"
        
        a = np.int(np.hstack(np.where(straights[startSequence] == bestFrame)))
        b = np.int(np.hstack(np.where(straights[bestSequence] == bestMatch)))
        seqA = straights[startSequence][a:startSequenceLen]
        seqB = straights[bestSequence][0:b]
        
        startTurnFrame = straightStartFrame[e] + mapEdges[e, 0]
        print "gne", startTurnFrame
        frames[straightStartFrame[e]:startTurnFrame] = np.hstack((seqB[len(seqA)+len(seqB)-mapEdges[e, 0]:len(seqB)], seqA))
        
    
    ## deal with joint between end of turn and beginning of straight
    if e > 0 :     
        
        n += 1
    else :
        print "rendering 1st straight"
        
        

# <codecell>

## load downsampled frames for visualizing
def downsample(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

frameSize = np.array(cv2.imread(dataLoc+cameraNames[0]).shape[0:2])/2
movie = np.zeros(np.hstack([frameSize[0], frameSize[1], len(cameraNames)]))
im = np.array(cv2.imread(dataLoc+cameraNames[0]))/255.0
im = np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])

for i in range(0, len(cameraNames)) :
    im = np.array(cv2.imread(dataLoc+cameraNames[i]))/255.0
    im = np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126]) # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
    movie[:, :, i] = downsample(im, frameSize)
    sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(cameraNames)))
    sys.stdout.flush()

# <codecell>

## plot 2d camera locations
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')

## set axis range
ax.set_autoscale_on(False)
ax.set_xlim([-40.0,20.0])
ax.set_ylim([-40.0,20.0])


# plot fitted lines
for line in lines : 
    color = "#" + random.choice(colVals) + random.choice(colVals) + random.choice(colVals) + random.choice(colVals) + random.choice(colVals) + random.choice(colVals)
    p1 = line[2:4]+200*line[0:2]
    p2 = line[2:4]-200*line[0:2]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color)


## plot outliers
# ax.plot(cameraLocs2D[np.hstack(outliers), 0], cameraLocs2D[np.hstack(outliers), 1], marker='o', color='r', ls='-')

# ## plot straights
# for straight in straights :
#     ax.plot(cameraLocs2D[np.hstack(straight), 0], cameraLocs2D[np.hstack(straight), 1], marker='o', color='r', ls=' ')
# ## plot turns
# for turn in turns :
#     ax.plot(cameraLocs2D[np.hstack(turn), 0], cameraLocs2D[np.hstack(turn), 1], marker='o', color='b', ls=' ')

# plane = pm.fitPlane(cameras[:, 4:7])

# n = plane[0:3]
# p = plane[3:6]

for i in xrange(0, len(cameraLocs2D), 10) :
    rotation = msu.qToR(cameras[i, 0:4]).T # take transpose of result of qToR
    ## HACK for the computation of trasf
    transf = np.hstack([np.array(np.vstack([rotation, np.zeros(3)])), np.array(([cameraLocs2D[i, 0]], [0], [cameraLocs2D[i, 1]], [1]))]) # 4x4 transformation matrix
    tmp = np.dot(transf, ([0.0], [0.0],[0.5],[1.0]))
    tmp = tmp[0:3, 0]
    tmp = tmp - np.dot(tmp - plane[3:6], plane[0:3])*plane[0:3] # project 3D point
    tmp = tmp[[0, 2]]
    ax.plot([cameraLocs2D[i, 0], tmp[0]], [cameraLocs2D[i, 1], tmp[1]], c='r')
    ax.scatter(cameraLocs2D[i, 0], cameraLocs2D[i, 1], marker='o', color='g')


plt.show()

# <codecell>

frames = finalFrames
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
img = plt.imshow(movie[:, :, frames[0]])
img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
    img.set_data(movie[:, :, frames[0]])
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f))
    img.set_data(movie[:, :, frames[f]])
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(frames),interval=30,blit=True)

plt.show()

# <headingcell level=1>

# From here on there's code for visualizing the data in 3D

# <codecell>

## look for all the 3D lines 
points3D = cameras[:, 4:7]
linesInliers3D = list()
## each line in line is a 6D array where first 3 elements give the line direction and last 3 elements give a point on line
lines3D = list()
while len(points3D) > 1 : # can't find lines with less than 2 points
    fittedLine, inliers, bestCameras = msu.ransac(points3D, 1.0, 500)
    if len(inliers) > 1 :
        line = msu.fitLine3D(cameras[inliers, 4:7])#cv2.fitLine(np.array(cameras[inliers, 4:7], dtype=float32), cv.CV_DIST_L2, 0, 0.01, 0.01)
        lines3D.append(line)
        linesInliers3D.append(inliers)
        points3D = cameras[np.delete(np.arange(0, len(cameras), 1), np.hstack(linesInliers3D)), 4:7]

# <codecell>

## plot found lines and their related camera locations
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

## plot axis centerd at [0, 0, 0]

ax.plot3D([0, 10], [0, 0], [0, 0], c='r') # x
ax.plot3D([0, 0], [0, 10], [0, 0], c='g') # y
ax.plot3D([0, 0], [0, 0], [0, 10], c='b') # z

plane = msu.fitPlane(cameras[:, 4:7])
tmp = plane[2]
plane[2] = plane[1]
plane[1] = tmp
print plane

x_surf=np.arange(-50.0, 60.0, 100.0)                # generate a mesh
y_surf=np.arange(-50.0, 60.0, 100.0)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = np.zeros_like(x_surf) #np.sqrt(x_surf+y_surf)             # ex. function, which depends on x and y
print x_surf
print y_surf
print z_surf
surf = np.zeros([3, x_surf.shape[0], x_surf.shape[1]])
surf[0] = x_surf
surf[1] = y_surf
surf[2] = (-x_surf*plane[0] - y_surf*plane[1] + plane[3])/plane[2] #z_surf

print (-x_surf*plane[0] - y_surf*plane[1] - plane[3])/plane[2]

n = plane[0:3]
p = plane[3:6]
# p = np.array([0.0, 0.0, 0.0])
# n = np.array([0.0, 0.0, 1.0])
for i in xrange(0,len(x_surf)) :
    for j in xrange(0,len(x_surf)) :
        s = surf[:, i, j]
#         print s,
#         s = s - np.dot(s - p, n)*n
#         surf[:, i, j] = s


ax.plot_surface(surf[0], surf[1], surf[2], cmap=cm.Blues_r, alpha=0.2);    # plot a 3d surface plot

for line, inliers in zip(lines, linesInliers) : 
    color = "#" + random.choice(colVals) + random.choice(colVals) + random.choice(colVals) + random.choice(colVals) + random.choice(colVals) + random.choice(colVals)
    p1 = line[3:6]+200*line[0:3]
    p2 = line[3:6]-200*line[0:3]
    ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, zdir='y')
    # cq is camera quaternion, cc is camera center
    count = 0
    for cq, cc in zip(cameras[inliers, 0:4], cameras[inliers, 4:7]):
        if np.mod(count, 10) == 0 :
            print cq
            rotation = msu.qToR(cq).T # take transpose of result of qToR
            transf = np.hstack([np.array(np.vstack([rotation, np.zeros(3)])), np.array(([cc[0]], [cc[1]], [cc[2]], [1]))]) # 4x4 transformation matrix
            tmp = np.dot(transf, ([0.0], [0.0],[0.5],[1.0]))
            print "gne", tmp
            ax.scatter3D(cc[0], cc[1], cc[2], c=color, marker='o', zdir='y')
            ax.plot3D([cc[0], tmp[0]], [cc[1], tmp[1]], [cc[2], tmp[2]], c='r', zdir='y')
        count += 1

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

## set axis range
ax.set_autoscale_on(False)
ax.set_xlim([-20.0,20.0])
ax.set_ylim([-20.0,20.0])
ax.set_zlim([-20.0,20.0])

plt.show()

# <codecell>

fittedLine, inliers, bestCameras = ransac(cameras[:, 4:7], 1.0, 1000)
print "best cameras:", bestCameras, "#inliers:", len(inliers)


## plot points and fitted line
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')
# cq is camera quaternion, cc is camera center
count = 0
for cq, cc in zip(cameras[:, 0:4], cameras[:, 4:7]):
    if np.mod(count, 10) == 0 :
        rotation = qToR(cq).T # take transpose of result of qToR
        transf = np.hstack([np.array(np.vstack([rotation, np.zeros(3)])), np.array(([cc[0]], [cc[1]], [cc[2]], [1]))]) # 4x4 transformation matrix
        tmp = np.dot(transf, ([0.0], [0.0],[0.5],[1.0]))
        if count in inliers :
            ax.scatter3D(cc[0], cc[1], cc[2], c='b', marker='x')
        else :
            ax.scatter3D(cc[0], cc[1], cc[2], c='g', marker='x')
        ax.plot3D([cc[0], tmp[0]], [cc[1], tmp[1]], [cc[2], tmp[2]], c='r')
    count += 1

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

## set axis range
ax.set_autoscale_on(False)
ax.set_xlim([-40.0,0.0])
ax.set_ylim([-40.0,40.0])
ax.set_zlim([-40.0,0.0])
        
# plot fitted line
p1 = fittedLine[0, :]+70*fittedLine[1,:]
p2 = fittedLine[0, :]-70*fittedLine[1,:]

ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='b')

plt.show()

# <codecell>

values = np.array(cameraLocs2D[segments[0], :])

## fit line
A = np.hstack((values, np.ones([len(values), 1])))
print A.shape

U, S, V = np.linalg.svd(A)

# print U.shape, S.shape, V.shape

# print S
print V

params = V.T[:, len(V)-1]
print params

a = np.array([0, -params[2]/params[1]])
b = np.array([-params[2]/params[0], 0])


######
# A = np.hstack((np.reshape(values[:, 0],[len(values), 1]), np.ones([len(values), 1])))
# print A.shape

# params = np.dot(np.linalg.pinv(A), np.reshape(values[:, 1],[len(values), 1]))
# print params

# a = np.array([0, params[1]])
# b = np.array([-params[1]/params[0], 0])

######



v = a - b
n = v / np.linalg.norm(v)
print n

p1 = a+70*n
p2 = a-70*n

print p1, p2

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o', color='g', ls='-')

ax.plot(values[:, 0], values[:, 1], marker='o', color='r', ls='-')

ax.set_xlabel('X')
ax.set_ylabel('Y')

## set axis range
ax.set_autoscale_on(False)
ax.set_xlim([-80.0,50.0])
ax.set_ylim([-80.0,50.0])


n1 = np.array([params[0]/params[2],params[1]/params[2]])

p3 = a+70*n1
p4 = a-70*n1 

print 'la', p3, p4

ax.plot([p3[0], p4[0]], [p3[1], p4[1]], marker='o', color='b', ls='-')


plt.show()

# <codecell>

values = np.array(cameraLocs2D[linesInliers[1], :])
line = fitLine2D(values)

print line

a = line[2:4]
n = line[0:2]

p1 = a+70*n
p2 = a-70*n

print 'la', p1, p2

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='', color='g', ls='-')

ax.plot(values[:, 0], values[:, 1], marker='o', color='r', ls='-')

ax.set_xlabel('X')
ax.set_ylabel('Y')

## set axis range
ax.set_autoscale_on(False)
ax.set_xlim([-80.0,50.0])
ax.set_ylim([-80.0,50.0])

plt.show()

# <codecell>

values = np.array(cameras[linesInliers[1], 4:7])
line = fitLine3D(values)

a = line[3:6]
n = line[0:3]
print a, n

p1 = a+70*n
p2 = a-70*n

## plot found lines and their related camera locations
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], marker='', color='g', ls='-')

ax.plot3D(values[:, 0], values[:, 1], values[:, 2], marker='o', color='r', ls='-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

## set axis range
ax.set_autoscale_on(False)
ax.set_xlim([-40.0,0.0])
ax.set_ylim([-40.0,40.0])
ax.set_zlim([-40.0,0.0])

plt.show()

