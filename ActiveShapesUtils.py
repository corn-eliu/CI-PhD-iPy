# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
# %pylab
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# <codecell>

def discretizeLine(segmentExtremes) :
    ## discretize line segment
    lN = np.int(np.linalg.norm(segmentExtremes[1, :]-segmentExtremes[0, :]))
    lDir = (segmentExtremes[1, :]-segmentExtremes[0, :])/np.float(lN)
    discretizedSegment = np.array(np.round(np.repeat(np.reshape(np.array(segmentExtremes[1, :], dtype=int), [len(segmentExtremes[1, :]), 1]), lN+1, axis=1)-np.arange(0, lN+1)*np.reshape(lDir,[len(lDir), 1])), dtype=np.int).T
    return discretizedSegment

# <codecell>

## transforms linear index from 0 to squareSize**2-1 into 2D coords in a squareSize*squareSize square counting L-R, T-B
def linearTo2DCoord(idx, squareSize) :
    if idx < 0 or idx >= squareSize**2 :
        raise BaseException, np.string_("Bad idx("+ np.string_(idx) +") for given squareSize("+ np.string_(squareSize) +")")
    
    return np.array([np.floor(idx/squareSize), np.mod(idx, squareSize)])

## gives delta values for coords in a square of odd size so that 0 is in the middle, positive units R-B & negative L-T
def deltaCoords(idx, squareSize) :
    if (idx < 0).any() or (idx >= squareSize).any() :
        raise BaseException, np.string_("Bad idx("+ np.string_(idx) +") for given squareSize("+ np.string_(squareSize) +")")
        
    if np.mod(squareSize, 2) != 1 :
        raise BaseException, np.string_("squareSize("+ np.string_(squareSize) +") not odd")
    
    squareRange = np.arange(-np.floor(squareSize/2), np.ceil(squareSize/2)+1)
    return np.array([squareRange[idx[0]], squareRange[idx[1]]])

# <codecell>

## build dynamic programming data structures
def buildMRF(snakeIndices, distMat, nbrhoodSize, meanSpacing, alpha) :
    
    # each column is 1 point and each row is cost for respective k (K=nbrhoodSize**2)
    # unary cost is given by -(log(likelihood))
    uCosts = np.zeros([nbrhoodSize**2, len(snakeIndices)])
    
    # meanSpacing = (np.linalg.norm(bbox[0, :]-bbox[1, :])+np.linalg.norm(bbox[1, :]-bbox[2, :]))*2.0/numOfPoints
    
    for n in xrange(0, len(snakeIndices)) :
        nPrev = n-1 if n > 0 else len(snakeIndices)-1
        nNext = n+1 if n < len(snakeIndices)-1 else 0
        likelihoods = np.zeros(nbrhoodSize**2)
        priors = np.zeros(nbrhoodSize**2)
        for k in xrange(0, nbrhoodSize**2) :
            kIdx = deltaCoords(linearTo2DCoord(k, nbrhoodSize), nbrhoodSize)+snakeIndices[n, :]
            ## likelihood = distance to edge
            # need to check if I need a minus in front here
            ## don't think I need **2 because distance_transform_edt does it already
            if (kIdx >= 0).all() and (kIdx < distMat.shape).all() :
                likelihood = np.exp(-(distMat[kIdx[0], kIdx[1]]))#**2)
            else :
                likelihood = 0.0
            likelihoods[k] = likelihood
#         likelihoods = likelihoods/np.sum(likelihoods)
            
        uCosts[:, n] = -(np.log(likelihoods))
    
    ## NOTE:: can merge the two cycles and probably even vectorize it at some point for speed
        
    # first dimension is k_n, second represents arrow from w_n-1 to w_n
    # last dimension dimension is k_n-1
    # pairwise cost is given by -(log(prior))
    pCosts = np.zeros([nbrhoodSize**2, len(snakeIndices)-1, nbrhoodSize**2])
    for nPrev in xrange(0, len(snakeIndices)-1) :
        n = nPrev+1
        nNext = n+1 if n < len(snakeIndices)-1 else 0
#         print nPrev, n, nNext
        for k in xrange(0, nbrhoodSize**2) :
            costs = np.zeros(nbrhoodSize**2)
            kIdxPrevs = np.zeros((nbrhoodSize**2, snakeIndices.shape[1]))
            kIdxNexts = np.zeros((nbrhoodSize**2, snakeIndices.shape[1])) 
            for i in xrange(0, nbrhoodSize**2) :
                kIdxPrevs[i, :] = deltaCoords(linearTo2DCoord(i, nbrhoodSize), nbrhoodSize)+snakeIndices[nPrev, :]
            
            if n == len(snakeIndices)-1 and (snakeIndices[0, :] == snakeIndices[-1, :]).all() :
                ## I'm at the end of the snake and the snake is looping
                kIdxNexts = np.repeat(np.reshape(deltaCoords(linearTo2DCoord(k, nbrhoodSize), nbrhoodSize)+snakeIndices[nNext+1, :], [1, 2]), nbrhoodSize**2, axis=0)
            else :
                kIdxNexts = np.repeat(np.reshape(deltaCoords(linearTo2DCoord(k, nbrhoodSize), nbrhoodSize)+snakeIndices[nNext, :], [1, 2]), nbrhoodSize**2, axis=0)

            kIdx = deltaCoords(linearTo2DCoord(k, nbrhoodSize), nbrhoodSize)+snakeIndices[n, :]
                
            ## don't use linalg.norm because what I actually want is for length to be positive
            spaceTerms = -(meanSpacing-np.linalg.norm(kIdx-kIdxPrevs, axis=1))**2
            curveTerms = -((kIdxPrevs-2*kIdx+kIdxNexts)*(kIdxPrevs-2*kIdx+kIdxNexts)).sum(axis=1)
            
            priors = np.exp(alpha*spaceTerms+(1-alpha)*curveTerms)
#             priors = priors/np.sum(priors)
            
            pCosts[k, nPrev, :] = -np.log(priors)
    pOldCosts = np.copy(pCosts)
    
    return uCosts, pOldCosts, pCosts

# <codecell>

def findMinCostTraversal(uCosts, pCosts, nbrhoodSize, numPoints, isLooping, saveCosts) :
    ## use the unary and pairwise costs to compute the min cost paths at each node
    # each column represents point n and each row says the index of the k-state that is chose for the min cost path
    minCostPaths = np.zeros([nbrhoodSize**2, numPoints])
    # contains the min cost to reach a certain state k (i.e. row) for point n (i.e. column)
    minCosts = np.zeros([nbrhoodSize**2, numPoints])
    # the first row of minCosts is just the unary cost
    minCosts[:, 0] = uCosts[:, 0]
    minCostPaths[:, 0] = np.arange(0, nbrhoodSize**2)
    
    for n in xrange(1, numPoints) :
        for k in xrange(0, nbrhoodSize**2) :
            costs = minCosts[:, n-1] + uCosts[k, n] + pCosts[k, n-1, :]
            minCosts[k, n] = np.min(costs)
            minCostPaths[k, n] = np.argmin(costs)
    
    if saveCosts :
        costsMat = {}
        costsMat['minCosts'] = minCosts
        costsMat['minCostPaths'] = minCostPaths
        sp.io.savemat("minCosts.mat", costsMat)
    
    ## now find the min cost path starting from the right most n with lowest cost
    minCostTraversal = np.zeros(numPoints)
    ## last node is the node where the right most node with lowest cost
    minCostTraversal[-1] = np.argmin(minCosts[:, -1]) #minCostPaths[np.argmin(minCosts[:, -1]), -1]
    if np.min(minCosts[:, -1]) == np.inf :
        minCostTraversal[-1] = np.floor((nbrhoodSize**2)/2)
    
    for i in xrange(len(minCostTraversal)-2, -1, -1) :
#         print i
        minCostTraversal[i] = minCostPaths[minCostTraversal[i+1], i+1]
#     print minCostTraversal
    
    if isLooping :
        minCostTraversal[0] = minCostTraversal[-1]
        
    print np.min(minCosts[:, -1])
#     print minCostTraversal
    
    return minCostTraversal, np.min(minCosts[:, -1])

# <codecell>

## s is a pointer to a line that represents the snake so that visualization can be updated with new snake
def optimizeSnake(initialSnakeIdxs, distMat, neighbourhoodSize, alpha, s, maxIter, saveCosts) :
    snakeIdxs = np.copy(initialSnakeIdxs)
    prev = 0.0
    curr = 10000.0
    minCost = curr
    count = 0
    while ((np.abs(prev-curr)) > 5.0 or (np.abs(minCost-curr)) > 5.0) and count < maxIter:
    # for i in xrange(0, 10) :
        count += 1
        prev = curr
        tmp = 0.0
        for p in xrange(0, len(snakeIdxs)-1) :
            pNext = p+1 if p < len(snakeIdxs)-1 else 0
            tmp += np.linalg.norm(snakeIdxs[p, :]-snakeIdxs[pNext, :])
        
        tmp = tmp/len(snakeIdxs)
        meanSpacing = tmp #+ np.random.rand(1)*0.2*tmp
    #     meanSpacing = 18.0
        print "mean", meanSpacing
        
        unaryCosts, pairwiseCosts, newPairwiseCosts = buildMRF(snakeIdxs, distMat, neighbourhoodSize, meanSpacing, alpha)
        bestTraversal, curr = findMinCostTraversal(unaryCosts, pairwiseCosts, neighbourhoodSize, len(snakeIdxs), (snakeIdxs[0, :] == snakeIdxs[-1, :]).all(), saveCosts)
    #     print bestTraversal
        
        if minCost > curr :
            minCost = curr
        
        ## translate points in old snake according to best traversal
        newIdxs = np.zeros_like(snakeIdxs)
        for p in xrange(0, len(snakeIdxs)) :
            newIdxs[p, :] = deltaCoords(linearTo2DCoord(bestTraversal[p], neighbourhoodSize), neighbourhoodSize)+snakeIdxs[p, :]
        
        ## visualize new state of snake
        visSnake = np.array([newIdxs[:, 1], newIdxs[:, 0]]).T
        s.set_data(visSnake[:, 0], visSnake[:, 1])
    #     ax.plot(visSnake[:, 0], visSnake[:, 1], c='y', marker="o")
        plt.draw()
        
        snakeIdxs = np.copy(newIdxs)
    
    return snakeIdxs

