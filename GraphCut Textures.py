# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import opengm
import numpy as np
import cv2
import time
# import graph_tool as gt
from PIL import Image
import scipy.io as sio
import GraphWithValues as gwv
import sys

# <codecell>

S_IDX = 0
T_IDX = 1
ST_COST = 2
S_LABEL = 3
T_LABEL = 4
S_A_COLOR = np.arange(5, 8, dtype=int)
S_B_COLOR = np.arange(8, 11, dtype=int)
T_A_COLOR = np.arange(11, 14, dtype=int)
T_B_COLOR = np.arange(14, 17, dtype=int)
print S_IDX, T_IDX, ST_COST, S_LABEL, T_LABEL, S_A_COLOR, S_B_COLOR, T_A_COLOR, T_B_COLOR

# <codecell>

# inputImg = cv2.cvtColor(cv2.imread("graphcutTexturesInput.png"), cv2.COLOR_BGR2RGB)
inputImg = cv2.cvtColor(cv2.imread("cashews.png"), cv2.COLOR_BGR2RGB)
figure(); imshow(inputImg)

# <codecell>

patchSize = inputImg.shape[0:2]#np.array([64, 64])
overlapSize = 64
outputImgSize = np.array([400, 400])

outputImg = np.zeros(np.hstack((outputImgSize, [3])), dtype=uint8)

## precompute pixel pairs for all edges in the patch
gridEdges1D = np.array(opengm.secondOrderGridVis(patchSize[1],patchSize[0],True))
gridEdges2D = np.zeros((len(gridEdges1D), 4))

gridEdges2D[:, 0] = np.mod(gridEdges1D[:, 0], patchSize[0])
gridEdges2D[:, 1] = np.array(gridEdges1D[:, 0]/patchSize[0], dtype=int)
gridEdges2D[:, 2] = np.mod(gridEdges1D[:, 1], patchSize[0])
gridEdges2D[:, 3] = np.array(gridEdges1D[:, 1]/patchSize[0], dtype=int)

# <codecell>

def getRandomPatch(img, pSize) :
    """Gets a random patch of size pSize from img
    
        \t  img   : input image
        \t  pSize : desired patch size
           
        return: patch from img and its top-left position pos"""
    
    if img.shape[0] < pSize[0] or img.shape[1] < pSize[1] :
        raise Exception("patch size " + np.string_(pSize) + " is too big for input image " + np.string_(img.shape))
    
    pos = np.array([np.random.choice(np.arange(img.shape[0]-pSize[0], dtype=int)), 
                    np.random.choice(np.arange(img.shape[1]-pSize[1], dtype=int))])
    
    return img[pos[0]:pos[0]+pSize[0], pos[1]:pos[1]+pSize[1], :], pos

def getRandomPatchPos(imgSize, pSize) :
    """Gets a random position to place patch of size pSize in the image of size imgSize
    
        \t  imgSize : size of image to place patch in
        \t  pSize   : desired patch size
           
        return: random top-left position pos"""
    
    if imgSize[0] < pSize[0] or imgSize[1] < pSize[1] :
        raise Exception("patch size " + np.string_(pSize) + " is too big for input image " + np.string_(img.shape))
    
    pos = np.array([np.random.choice(np.arange(imgSize[0]-pSize[0], dtype=int)), 
                    np.random.choice(np.arange(imgSize[1]-pSize[1], dtype=int))])
    
    return pos


# patchA, patchPos = getRandomPatch(inputImg, patchSize)
# print patchPos
# figure(); imshow(patchA, interpolation='nearest')

# <codecell>

def visualizeOldAndNewCuts(oldCuts, newCuts, patchSize) :
    """Visualizes an image where pixel pairs are red if the edge between them is old and blue if it is new
    
        \t  oldCuts: pixel pairs for old cuts of size [N, 2, 2]
        \t  newCuts : pixel pairs for old cuts of size [N, 2, 2]
        \t  patchSize : size of overlapping patch (rows, cols)"""
    
    cutEdgesImg = np.zeros((patchSize[0], patchSize[1], 3), dtype=uint8)
    
    cutEdgesImg[oldCuts[:, 0, 0], oldCuts[:, 1, 0], 0] = 255
    cutEdgesImg[oldCuts[:, 0, 1], oldCuts[:, 1, 1], 0] = 255
    
    cutEdgesImg[newCuts[:, 0, 0], newCuts[:, 1, 0], 2] = 255
    cutEdgesImg[newCuts[:, 0, 1], newCuts[:, 1, 1], 2] = 255
    
    figure(); imshow(cutEdgesImg, interpolation='nearest')

# oldPixelPairs = np.zeros((len(cutEdgesAB), 2, 2), dtype=int)
# oldPixelPairs[:, :, 0] = np.array([np.mod(cutEdgesAB[:, 0], patSize), np.array(cutEdgesAB[:, 0]/patSize, dtype=int)]).T
# oldPixelPairs[:, :, 1] = np.array([np.mod(cutEdgesAB[:, 1], patSize), np.array(cutEdgesAB[:, 1]/patSize, dtype=int)]).T

# newPixelPairs = np.zeros((len(cutEdges), 2, 2), dtype=int)
# newPixelPairs[:, :, 0] = np.array([np.mod(cutEdges[:, 0], patSize), np.array(cutEdges[:, 0]/patSize, dtype=int)]).T
# newPixelPairs[:, :, 1] = np.array([np.mod(cutEdges[:, 1], patSize), np.array(cutEdges[:, 1]/patSize, dtype=int)]).T

# visualizeOldAndNewCuts(oldPixelPairs, newPixelPairs, (patSize, patSize))

# <codecell>

def visualizeKeptAndOverwrittenCuts(gm, patchSize, labels, firstSeamNodeIdx, pixelPairs) :
    """Visualizes an image where pixel pairs are green if the edge between them is kept and red if it is overwritten
    
        \t  gm               : opengm graphical model initialized for graphcut textures
        \t  patchSize        : size of overlapping patch (rows, cols)
        \t  labels           : labels for all variables in gm after graphcut
        \t  firstSeamNodeIdx : index of first variable to be a seam node
        \t  pixelPairs       : list of pixel pairs of old cuts in the placed patch area of size [N, 2, 2]"""

    keptEdgesImg = np.zeros((patchSize[0], patchSize[1], 3), dtype=uint8)
    counterKept = 0
    counterOverwritten = 0
    
    for factor in gm.factors(order=2) :
        ## if factor is an edge between a seam node and patch B or between a pixel node and a seam node
        if factor.variableIndices[0] >= firstSeamNodeIdx or factor.variableIndices[1] >= firstSeamNodeIdx :
            ## if the nodes of connected by current factor edge have different values, the edge has been cut
            if labels[factor.variableIndices[0]] - labels[factor.variableIndices[1]] != 0 :
                print factor.variableIndices, labels[factor.variableIndices[0]], labels[factor.variableIndices[1]],
                ## if edge is between a seam node and patch B and it has been CUT then keep the old cut
                if factor.variableIndices[0] >= firstSeamNodeIdx :
                    print "keep"
                    counterKept += 1
                    keptEdgesImg[pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 0, 0], 
                                 pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 1, 0], 1] = 255
                    keptEdgesImg[pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 0, 1], 
                                 pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 1, 1], 1] = 255
                else :
                    print
            ## if edge is between a seam node and patch B and is NOT CUT then this edge has been overwritten
            elif factor.variableIndices[0] >= firstSeamNodeIdx :
                print factor.variableIndices, labels[factor.variableIndices[0]], labels[factor.variableIndices[1]], "overwritten"
                counterOverwritten += 1
                keptEdgesImg[pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 0, 0], 
                             pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 1, 0], 0] = 255
                keptEdgesImg[pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 0, 1], 
                             pixelPairs[factor.variableIndices[0]-firstSeamNodeIdx, 1, 1], 0] = 255
    
    
    print "total kept:", counterKept
    print "total overwritten:", counterOverwritten
    figure(); imshow(keptEdgesImg, interpolation='nearest')
    
# visualizeKeptAndOverwrittenCuts(graphModel, (patSize, patSize), allLabels, patSize**2, cutPixelPairs)

# <codecell>

# tmp = set(tuple(i) for i in np.hstack((totalCutPixelPairs[:, :, 0], totalCutPixelPairs[:, :, 1])))
# tmpDict = dict((k,i) for i,k in zip(xrange(len(totalCutPixelPairs)), tmp))
# tmp2 = set(tuple(i) for i in np.argwhere(overlappingPixels))
def getPatchRegionEdges(stdGrid, rPos, cPos, patchSize, validImgSize) :
    """Computes list of possible edges in the region where a patch has been placed 
    
           stdGrid: pixel pairs for edges in the unplaced patch
           rPos: row index of placed patch
           cPos: column index of placed patch
           validImgSize: size of image used to check wether the placed patch has valid pixel coords in output img
           
        return: patchRegionEdges"""

#     patchRegionEdges = np.copy(stdGrid)
#     patchRegionEdges[:, 0:4:2] += rPos
#     patchRegionEdges[:, 1:4:2] += cPos
#     ## make sure pixels have valid coords
#     ## check they're positive
#     positivePixels = np.all(patchRegionEdges >= 0, axis=-1)
#     ## check row coord is smaller than number of rows
#     validRowPixels = np.all(patchRegionEdges[:, 0:4:2] < validImgSize[0], axis=-1)
#     ## check column coord is smaller than number of columns
#     validColPixels = np.all(patchRegionEdges[:, 1:4:2] < validImgSize[1], axis=-1)
#     patchRegionEdges = patchRegionEdges[np.logical_and(positivePixels, validRowPixels, validColPixels), :]

    patchRegionEdges = np.copy(stdGrid)
    patchRegionEdges = (np.mod(patchRegionEdges, patchSize[0]) + 
                       (np.array(patchRegionEdges/patchSize[0], dtype=int))*validImgSize[0])
    ## transform [row, col], node1 = [:, {0, 2}], node2 = [:, {1, 3}]
    patchRegionEdges2D = np.array(np.hstack((np.mod(patchRegionEdges, validImgSize[0]), np.array(patchRegionEdges/validImgSize[0], dtype=int))), dtype=int)
    patchRegionEdges2D[:, 0:2] += rPos
    patchRegionEdges2D[:, 2:4] += cPos
    ## make sure pixels have valid coords
    ## check they're positive
    positivePixels = np.all(patchRegionEdges2D >= 0, axis=-1)
    # ## check that the placed patch does not go outside of the image
    validPositivePixels = np.all(np.hstack((patchRegionEdges2D[:, 0:2] < validImgSize[0], patchRegionEdges2D[:, 2:4] < validImgSize[1])), axis=-1)
    patchRegionEdges2D = patchRegionEdges2D[np.logical_and(positivePixels, validPositivePixels), :]
    
    patchRegionEdges = patchRegionEdges2D[:, 0:2] + patchRegionEdges2D[:, 2:4]*validImgSize[0]

#     ## make sure pixels have valid coords
#     ## check they're positive
#     positivePixels = np.all(patchRegionEdges >= 0, axis=-1)
#     ## check that the placed patch does not go outside of the image
#     validPositivePixels = np.all(patchRegionEdges < np.prod(validImgSize), axis=-1)
#     patchRegionEdges = patchRegionEdges[np.logical_and(positivePixels, validPositivePixels), :]
    
    return patchRegionEdges

def getExistingEdgesInPatchRegion(existingEdges, x, y, edgeMap, patchSize, outImageSize) :
    """Finds the cut edges in the given region
    
           existingEdges: list of cut edges in the output image
           x: column coordinate of the placed patch
           y: row coordinate of the placed patch
           edgeMap: list of edges in the standard patch (in the same coordinate system as existingEdges)
           outImageSize: size of the output image to check which pixels in the placed patch are within boundaries
           
        return: commonEdges     = pixel pairs of edges within the placed patch region
                commonEdgesIdxs = indices within the existingEdges list of the edges in the placed patch region"""
    
#     ## make a dictionary using the concatenated pixel pairs as keys and indices as values (so that I don't lose original
#     ## ordering when the keys get re ordered)
#     edgeDict = dict((tuple(k),i) for i,k in zip(xrange(len(existingEdges)), np.hstack((existingEdges[:, :, 0], existingEdges[:, :, 1]))))
#     ## define a set containing the pixel pairs of all cut edges so far
#     cutEdgesSet = set(i for i in edgeDict.keys())
#     ## define a set containing the pixel pairs of all edges in the patch region
#     patchEdges2D = getPatchRegionEdges(edgeMap, y, x, outImageSize)
#     patchEdgesSet = set(tuple(i) for i in patchEdges2D)
#     ## find common edges to the two sets
#     commonEdges = cutEdgesSet.intersection(patchEdgesSet)
#     commonEdgesIdxs = np.array([edgeDict[x] for x in commonEdges])
    
    ## make a dictionary using the concatenated pixel pairs as keys and indices as values (so that I don't lose original
    ## ordering when the keys get re ordered)
    edgeDict = dict((tuple(k),i) for i,k in zip(xrange(len(existingEdges)), existingEdges[:, 0:2]))
    ## define a set containing the pixel pairs of all cut edges so far
    cutEdgesSet = set(i for i in edgeDict.keys())
    ## define a set containing the pixel pairs of all edges in the patch region
    patchEdges1D = getPatchRegionEdges(edgeMap, y, x, patchSize, outImageSize)
    patchEdgesSet = set(tuple(i) for i in patchEdges1D)
    ## find common edges to the two sets
    commonEdges = cutEdgesSet.intersection(patchEdgesSet)
    commonEdgesIdxs = np.array([edgeDict[x] for x in commonEdges], dtype=int)
    
    return np.array(list(commonEdges), dtype=int), commonEdgesIdxs

# <codecell>

def getGraphcutOnOverlap(patchA, patchB, patchAPixels, patchBPixels, oldCutEdges) :
    """Computes pixel labels using graphcut given two overlapping patches
    
        \t  patchA       : patch A
        \t  patchB       : patch B
        \t  patchAPixels : pixels that are definitely to be taken from patch A
        \t  patchBPixels : pixels that are definitely to be taken from patch B
        \t  oldCutEdges  : list of old cut edges that need a seam node [node1Idx, node2Idx, cost, 
                                                                  patchA[node1Idx, :], patchB[node1Idx, :], 
                                                                  patchA[node2Idx, :], patchB[node2Idx, :]]
           
        return: reshapedLabels = labels for each pixel 
                cutFactors     = list of cut edges [node1Idx, node2Idx, cost, patchA[node1Idx, :], 
                                                    patchB[node1Idx, :], patchA[node2Idx, :], patchB[node2Idx, :]]"""
    
    if np.all(patchA.shape != patchB.shape) :
        raise Exception("The two specified patches have different shape so graph cannot be built")
        
    if patchA.dtype != np.float64 or patchB.dtype != np.float64 :
        raise Exception("The two specified patches are not of type float64! Check there is no overflow when computing costs")
    
    h, width = patchA.shape[0:2]
    maxCost = 10000000.0#np.sys.float_info.max
    
    ## build graph
    numLabels = 2
    ## one node for each pixel in grid, one for each seam node and one for each patch
    numSeamNodes = len(oldCutEdges)
    firstSeamIdx = h*width
    numNodes = h*width+numSeamNodes+2
    gm = opengm.gm(numpy.ones(numNodes,dtype=opengm.label_type)*numLabels)
#     print "seam nodes", numSeamNodes
    ## Last 2 nodes are patch A and B respectively
    idxPatchANode = numNodes - 2
    idxPatchBNode = numNodes - 1
    
        
    ## get unary functions
    unaries = np.zeros((numNodes,numLabels))
    
    ## fix label for nodes representing patch A and B to have label 0 and 1 respectively
#     print "patches idxs", idxPatchANode, idxPatchBNode
    unaries[idxPatchANode, :] = [0.0, maxCost]
    unaries[idxPatchBNode, :] = [maxCost, 0.0]
    
    # add functions
    fids = gm.addFunctions(unaries)
    # add first order factors
    gm.addFactors(fids, arange(0, numNodes, 1))
    
    
    ## get factor indices for the overlap grid of pixels
    pairIndices = np.array(opengm.secondOrderGridVis(width,h,True))
    ## get pairwise functions for those nodes
    pairwise = np.zeros(len(pairIndices))
#     tmpOverlap = np.zeros((patchSize, overlapSize, 3))
    for pair, i in zip(pairIndices, arange(len(pairIndices))) :
        sPix = np.array([int(np.mod(pair[0],h)), int(pair[0]/h)])
        tPix = np.array([int(np.mod(pair[1],h)), int(pair[1]/h)])
        
        pairDiff = np.all(oldCutEdges[:, [S_IDX, T_IDX]] - np.sort(pair) == 0, axis = -1)
        ## if current edge pixel pair has not been cut proceed as usual
        if not np.any(pairDiff) :
            pairwise[i] = norm(patchA[sPix[0], sPix[1], :] - patchB[sPix[0], sPix[1], :])
            pairwise[i] += norm(patchA[tPix[0], tPix[1], :] - patchB[tPix[0], tPix[1], :])
            
            fid = gm.addFunction(np.array([[0.0, pairwise[i]],[pairwise[i], 0.0]]))
            gm.addFactor(fid, pair)
#         else :
#             print "pair", pair, "needs seam node"
            
    ## deal with cut edges
    for nodeFid, oldCutEdge in zip(np.arange(numNodes-numSeamNodes-2, numNodes-2), oldCutEdges) :
        sPix = np.array([int(np.mod(oldCutEdge[S_IDX],h)), int(oldCutEdge[S_IDX]/h)])
        tPix = np.array([int(np.mod(oldCutEdge[T_IDX],h)), int(oldCutEdge[T_IDX]/h)])            
        
        ## add factor between seam node and B patch
        fid = gm.addFunction(np.array([[0.0, oldCutEdge[ST_COST]],[oldCutEdge[ST_COST], 0.0]]))
        gm.addFactor(fid, np.array([nodeFid, idxPatchBNode]))
        
        ## add factor between pixel s and seam node
        ## for each seam node I need to store color of pixel s and t!!!!!!!!!!!!!!!!
        la = True
        ## check if s came from old patch A
        if oldCutEdge[S_LABEL] == 0 :
            pairwiseCost = norm(oldCutEdge[S_A_COLOR] - patchB[sPix[0], sPix[1], :])
            pairwiseCost += norm(oldCutEdge[T_A_COLOR] - patchB[tPix[0], tPix[1], :])
        ## otherwise it came from old patch B
        else :
            la = False
            pairwiseCost = norm(oldCutEdge[S_B_COLOR] - patchB[sPix[0], sPix[1], :])
            pairwiseCost += norm(oldCutEdge[T_B_COLOR] - patchB[tPix[0], tPix[1], :])
        
#         if nodeFid == 53121 :
#             print "seam node:", nodeFid, oldCutEdge[[S_IDX, T_IDX]], oldCutEdge[ST_COST], 
#             print (norm(oldCutEdge[S_A_COLOR]-oldCutEdge[S_B_COLOR]) 
#                                + norm(oldCutEdge[T_A_COLOR]-oldCutEdge[T_B_COLOR]))
#             print "pA_s", patchA[sPix[0], sPix[1], :], "pA_t", patchA[tPix[0], tPix[1], :]
#             print "A_s", oldCutEdge[S_A_COLOR], "A_t", oldCutEdge[T_A_COLOR]
#             print "B_s", oldCutEdge[S_B_COLOR], "B_t", oldCutEdge[T_B_COLOR]
#             print "C_s", patchB[sPix[0], sPix[1], :], "C_t", patchB[tPix[0], tPix[1], :]
#             print pairwiseCost, la
        
        fid = gm.addFunction(np.array([[0.0, pairwiseCost],[pairwiseCost, 0.0]]))
        gm.addFactor(fid, np.array([oldCutEdge[S_IDX], nodeFid]))
        
        ## add factor between seam node and pixel t
        ## for each seam node I need to store color of pixel s and t!!!!!!!!!!!!!!!!
        ## check if t came from old patch B
        if oldCutEdge[T_LABEL] == 1 :
            pairwiseCost = norm(patchB[sPix[0], sPix[1], :] - oldCutEdge[S_B_COLOR])
            pairwiseCost += norm(patchB[tPix[0], tPix[1], :] - oldCutEdge[T_B_COLOR])
        ## otherwise it came from old patch A
        else :
            pairwiseCost = norm(patchB[sPix[0], sPix[1], :] - oldCutEdge[S_A_COLOR])
            pairwiseCost += norm(patchB[tPix[0], tPix[1], :] - oldCutEdge[T_A_COLOR])
        
        fid = gm.addFunction(np.array([[0.0, pairwiseCost],[pairwiseCost, 0.0]]))
        gm.addFactor(fid, np.array([oldCutEdge[T_IDX], nodeFid]))
        
#         print nodeFid, oldCutEdge[S_IDX], oldCutEdge[T_IDX]
    
    # add function used for connecting the patch variables
    fid = gm.addFunction(np.array([[0.0, maxCost],[maxCost, 0.0]]))
    
    # connect patch A to definite patch A pixels
    if len(patchAPixels) :
    #     patchAFactors = np.hstack((np.arange(h, dtype=uint).reshape((h, 1)), np.ones((h, 1), dtype=uint)*idxPatchANode))
        patchAFactors = np.hstack((patchAPixels.reshape((len(patchAPixels), 1)), np.ones((len(patchAPixels), 1), dtype=uint)*idxPatchANode))
        gm.addFactors(fid, patchAFactors)
    
    # connect patch B to definite patch B pixels
    if len(patchBPixels) > 0 :
    #     patchBFactors = np.hstack((np.arange(numNodes-2-h, numNodes-2, dtype=uint).reshape((len(patchAPixels), 1)), np.ones((h, 1), dtype=uint)*idxPatchBNode))
        patchBFactors = np.hstack((patchBPixels.reshape((len(patchBPixels), 1)), np.ones((len(patchBPixels), 1), dtype=uint)*idxPatchBNode))
        gm.addFactors(fid, patchBFactors)
    
    
    graphCut = opengm.inference.GraphCut(gm=gm)
    graphCut.infer()
    
    labels = np.array(graphCut.arg(), dtype=int)
    ## make a copy of labels and update it when necessary if seam edges have been cut
    finalLabels = np.copy(labels)
    
    ## find list of cut factors from min cut
    cutFactorIdxs = np.empty((0, 2))
    cutFactorCosts = np.empty((0, 1))
    ## contains labels for pixels s and t (label 0 means patch A and label 1 means patch B)
    cutFactorLabels = np.empty((0, 2))
    ## contains colors of pixels s and t from patch A and B
    cutFactorColors = np.empty((0, 12))
    ## contains delete mask to indicate which of the oldCutEdges are overwritten and need to be deleted
    keepOldCuts = np.zeros(len(oldCutEdges), dtype=np.bool)
    ## contains indices of seam nodes that got substituted because a cut between seam and pixel has been made
    substitudedSeams = []
    for factor in gm.factors(order=2) :
        ## sort the factor indices so that I don't have to ever check if an edge between i and j exists 
        ## between j and i when looking for duplicate edges because I always ensure i < j; maybe opengm makes 
        ## sure variableIndices are sorted but I'm not sure
        varIdxs = np.sort([factor.variableIndices[0], factor.variableIndices[1]])
#         if varIdxs[0] == 23003 and varIdxs[1] == 35604 :
#             print "lalalalalal", labels[factor.variableIndices[0]], labels[factor.variableIndices[1]]
#         if idxPatchANode not in factor.variableIndices and idxPatchBNode not in factor.variableIndices :
        ## make sure the edge is cut by checking label of fators is different
        if labels[varIdxs[0]] - labels[varIdxs[1]] != 0 :
#             print varIdxs, 
            ## make sure the cut is not between a pixel and a seam node
            if (varIdxs[0] not in np.arange(firstSeamIdx, firstSeamIdx+numSeamNodes, dtype=int) and 
                varIdxs[1] not in np.arange(firstSeamIdx, firstSeamIdx+numSeamNodes, dtype=int) ) :
#                 print "cut pixel edge", varIdxs, factor.variableIndices, labels[factor.variableIndices[0]], labels[factor.variableIndices[1]]
                sPix = np.array([int(np.mod(varIdxs[0],h)), int(varIdxs[0]/h)])
                tPix = np.array([int(np.mod(varIdxs[1],h)), int(varIdxs[1]/h)])
                
                cutFactorIdxs = np.vstack((cutFactorIdxs, varIdxs))
                cutFactorCosts = np.vstack((cutFactorCosts, factor.max()))
                cutFactorLabels = np.vstack((cutFactorLabels, np.array([labels[varIdxs[0]], labels[varIdxs[1]]])))
#                 print factor.max()
                cutFactorColors = np.vstack((cutFactorColors, np.hstack((patchA[sPix[0], sPix[1], :], 
                                                                         patchB[sPix[0], sPix[1], :], 
                                                                         patchA[tPix[0], tPix[1], :], 
                                                                         patchB[tPix[0], tPix[1], :]))))
            else :
#                 print
#                 print "cut seam node", varIdxs, factor.max(),
                ## edge between seam node and patch B has been cut so it needs to be kept
                ## factor.variableIndices[0] will necessarily be a seam node since an edge between patch B and
                ## any pixel node will never get cut as the cost is very very high (100000)
                if varIdxs[1] == idxPatchBNode :
#                     print "keep",
#                     print (norm(oldCutEdges[varIdxs[0]-firstSeamIdx, S_A_COLOR]-oldCutEdges[varIdxs[0]-firstSeamIdx, S_B_COLOR]) 
#                                + norm(oldCutEdges[varIdxs[0]-firstSeamIdx, T_A_COLOR]-oldCutEdges[varIdxs[0]-firstSeamIdx, T_B_COLOR]))
                    keepOldCuts[varIdxs[0]-firstSeamIdx] = True
                ## an edge between a seam node and a pixel node has been cut
                else :
#                     print "substitute with edge", 
                    ## make sure the old cut gets deleted
                    keepOldCuts[varIdxs[1]-firstSeamIdx] = False
                    substitudedSeams.append(varIdxs[1]-firstSeamIdx)
                    
                    ## find which edge from seam node to pixel node got cut
                    pixelNodeIdx = np.copy(varIdxs[0])
                    ## the actual variables that I want the cut to be between are pixel nodes which I get from the list
                    ## of old edges and I need the colors as when I cut pixel-seam edges I only compare seam patch with the colors
                    ## in the patch the pixel node came from
                    oldCutEdgeIdx = np.copy(varIdxs[1]-firstSeamIdx)
                    varIdxs = oldCutEdges[varIdxs[1]-firstSeamIdx, 0:2]
#                     print varIdxs, "where", 
                    ## set labels for pixel nodes to reflect which seam-pixel edge got cut
                    ## cut is between s and seam
                    if np.argwhere(varIdxs==pixelNodeIdx) == 0 :
                        ## s is from patch A and t is from patch B
                        finalLabels[varIdxs[0]] = 0
                        finalLabels[varIdxs[1]] = 1
                    ## cut is between seam and t
                    else :
                        ## s is from patch B and t is from patch A
                        finalLabels[varIdxs[0]] = 1
                        finalLabels[varIdxs[1]] = 0                    
                    
                    ## add new cut now
                    sPix = np.array([int(np.mod(varIdxs[0],h)), int(varIdxs[0]/h)])
                    tPix = np.array([int(np.mod(varIdxs[1],h)), int(varIdxs[1]/h)])
#                     print sPix, tPix, varIdxs, h
                    
                    cutFactorIdxs = np.vstack((cutFactorIdxs, varIdxs))
                    cutFactorLabels = np.vstack((cutFactorLabels, np.array([finalLabels[varIdxs[0]], finalLabels[varIdxs[1]]])))
                    cutFactorCosts = np.vstack((cutFactorCosts, factor.max()))
                    
                    if finalLabels[varIdxs[0]] == 0 :
                        ## here I have s coming from old patch A (which is what patchA contains for s but not for t) and t coming from new patch B
                        ## so I need colors from old patch A and new patch B for both s and t pixels
#                         print "s is patch A"
#                         cutFactorColors = np.vstack((cutFactorColors, np.hstack((patchA[sPix[0], sPix[1], :], 
#                                                                                  patchB[sPix[0], sPix[1], :], 
#                                                                                  patchA[tPix[0], tPix[1], :], 
#                                                                                  patchB[tPix[0], tPix[1], :]))))

                        ## checking which old patch s came from and has been used to compute this edge's cost
                        if oldCutEdges[oldCutEdgeIdx, S_LABEL] == 0 :
                            cutFactorColors = np.vstack((cutFactorColors, np.hstack((patchA[sPix[0], sPix[1], :], 
                                                                                 patchB[sPix[0], sPix[1], :], 
                                                                                 oldCutEdges[oldCutEdgeIdx, T_A_COLOR], 
                                                                                 patchB[tPix[0], tPix[1], :]))))
                        else : 
                            cutFactorColors = np.vstack((cutFactorColors, np.hstack((patchA[sPix[0], sPix[1], :], 
                                                                                 patchB[sPix[0], sPix[1], :], 
                                                                                 oldCutEdges[oldCutEdgeIdx, T_B_COLOR],
                                                                                 patchB[tPix[0], tPix[1], :]))))
            
#                         if factor.variableIndices[1] == 53121 : #14426 
#                             print "seam colors:", patchA[sPix[0], sPix[1], :], patchB[sPix[0], sPix[1], :], patchA[tPix[0], tPix[1], :], patchB[tPix[0], tPix[1], :], 
#                             tmp = norm(cutFactorColors[-1, S_A_COLOR-5] - cutFactorColors[-1, S_B_COLOR-5])
#                             tmp += norm(cutFactorColors[-1, T_A_COLOR-5] - cutFactorColors[-1, T_B_COLOR-5])
#                             print tmp
#                             print cutFactorColors[-1, S_A_COLOR-5], cutFactorColors[-1, S_B_COLOR-5], cutFactorColors[-1, T_A_COLOR-5], cutFactorColors[-1, T_B_COLOR-5],
#                             tmp = norm(oldCutEdge[S_A_COLOR] - patchB[sPix[0], sPix[1], :])
#                             tmp += norm(oldCutEdge[T_A_COLOR] - patchB[tPix[0], tPix[1], :])
#                             print tmp
                    else :
                        ## here I have s coming from new patch B and t coming from old patch B
                        ## since I always assume that the ordering of the given colors is A_s, B_s, A_t, B_t and I assume s comes from A
                        ## and t comes from B, then the new patch B is taking the place of A and the old patch B is taking the place of B
                        ## since patchA only contains the color of old patch B for pixel t, I need to take the color of old patch B for pixel
                        ## s from the list of oldCutEdges
#                         print "s is patch B"
#                         cutFactorColors = np.vstack((cutFactorColors, np.hstack((patchB[sPix[0], sPix[1], :], 
#                                                                                  patchA[sPix[0], sPix[1], :], 
#                                                                                  patchB[tPix[0], tPix[1], :], 
#                                                                                  patchA[tPix[0], tPix[1], :]))))
#                         cutFactorColors = np.vstack((cutFactorColors, np.hstack((patchB[sPix[0], sPix[1], :], 
#                                                                                  oldCutEdges[oldCutEdgeIdx, S_B_COLOR], 
#                                                                                  patchB[tPix[0], tPix[1], :], 
#                                                                                  patchA[tPix[0], tPix[1], :]))))

                        ## checking which old patch t came from and has been used to compute this edge's cost
                        if oldCutEdges[oldCutEdgeIdx, T_LABEL] == 1 :
                            cutFactorColors = np.vstack((cutFactorColors, np.hstack((oldCutEdges[oldCutEdgeIdx, S_B_COLOR], 
                                                                                 patchB[sPix[0], sPix[1], :], 
                                                                                 patchA[tPix[0], tPix[1], :],  
                                                                                 patchB[tPix[0], tPix[1], :]))))
                        else :
                            cutFactorColors = np.vstack((cutFactorColors, np.hstack((oldCutEdges[oldCutEdgeIdx, S_A_COLOR], 
                                                                                 patchB[sPix[0], sPix[1], :], 
                                                                                 patchA[tPix[0], tPix[1], :],  
                                                                                 patchB[tPix[0], tPix[1], :]))))
                        
                    tmp = norm(cutFactorColors[-1, S_A_COLOR-5] - cutFactorColors[-1, S_B_COLOR-5])
                    tmp += norm(cutFactorColors[-1, T_A_COLOR-5] - cutFactorColors[-1, T_B_COLOR-5])
                    if tmp != factor.max() :
                        print "not matching!!!!", finalLabels[varIdxs[0]] == 0
                        
    if np.any(keepOldCuts[substitudedSeams]) :
        raise Exception("My assumption that seam-patch cut comes before seam-pixel cut and that seam-patch won't get cut if seam-pixel is cut, is not true!!")
    
#     print cutFactorIdxs
    reshapedLabels = np.reshape(finalLabels[0:-2-numSeamNodes], patchA.shape[0:2], 'F')
    cutFactors = np.hstack((cutFactorIdxs, cutFactorCosts, cutFactorLabels, cutFactorColors))
    print gm
#     figure(); opengm.visualizeGm(gm, plotUnaries=False, layout='sfdp', iterations=1000)
    
    return reshapedLabels, cutFactors, keepOldCuts, gm, finalLabels, labels

# pixelLabels, cutEdges = getGraphcutOnOverlap(patchLeft[:, patchSize[0]-overlapSize:, :], patchRight[:, 0:overlapSize, :])
# figure(); imshow(pixelLabels, interpolation='nearest')
# print len(cutEdges)

# <codecell>

def placePatch(rIdx, cIdx, patch, outputImg, occupancyMap) :
    
    print rIdx, cIdx
    
    r = np.max((0, rIdx))
    c = np.max((0, cIdx))
    height = patch.shape[0]
    width = patch.shape[1]
    if rIdx < 0 :
        height += rIdx
    elif rIdx > outputImg.shape[0]-patch.shape[0] :
        height = outputImg.shape[0]-rIdx
        
    if cIdx < 0 :
        width += cIdx
    elif cIdx > outputImg.shape[1]-patch.shape[1] :
        width = outputImg.shape[1]-cIdx
    
#     print r, c, height, width
    
    ### NOTE: modifying outputImg from within method so only works if outputImg is modifiable
    outputImg[r:r+height, c:c+width, :] = patch[patch.shape[0]-height:, patch.shape[1]-width:, :]
    
    newPatchOccupancy = np.zeros((outputImg.shape[0], outputImg.shape[1]), dtype=bool)
    newPatchOccupancy[r:r+height, c:c+width] += 1
    
    overlappingPixels = np.logical_and(newPatchOccupancy, occupancyMap)
    
    occupancyMap[r:r+height, c:c+width] += 1
    
    return overlappingPixels, r, height, c, width


def cutOverlap(row, h, col, w, imgPatA, imgPatB, overlapPixs, outImgSize, oldCutEdgesInOverlap, showFigs) :
    patAPixs = np.arange(0, h, dtype=uint)
    patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, h*(w-1)+1, h, dtype=uint))))
    
    patBPixs = np.ndarray.flatten(np.argwhere(np.ndarray.flatten(-overlapPixs[row:row+h, col:col+w], order='F')))
    patBPixs = np.unique(np.concatenate((patBPixs, np.arange(h*(w-1), h*w, dtype=uint))))
    
    patAPixs = np.setdiff1d(patAPixs, patBPixs)
    patA = imgPatA[row:row+h, col:col+w, :]/255.0
    patB = imgPatB[imgPatB.shape[0]-h:, imgPatB.shape[1]-w:, :]/255.0
    labels, cutEdges, oldCutsToKeep, graphModel, allLabels, originalLabels = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, oldCutEdgesInOverlap)
    
    if showFigs :
        figure(); imshow(labels, interpolation='nearest')
    
    outputPatch = np.zeros(patA.shape, dtype=uint8)
    for i in xrange(labels.shape[0]) :
        for j in xrange(labels.shape[1]) :
            if labels[i, j] == 0 :
                outputPatch[i, j, :] = patA[i, j, :]*255
            else :
                outputPatch[i, j, :] = patB[i, j, :]*255
    
    if showFigs :
        figure(); imshow(outputPatch, interpolation='nearest')
        
    cutPixelPairs = np.zeros((len(cutEdges), 2, 3))
    cutPixelPairs[:, :, 0] = np.array([np.mod(cutEdges[:, 0], h), np.array(cutEdges[:, 0]/h, dtype=int)]).T
    cutPixelPairs[:, :, 1] = np.array([np.mod(cutEdges[:, 1], h), np.array(cutEdges[:, 1]/h, dtype=int)]).T
    cutPixelPairs[:, :, 2] = np.repeat(cutEdges[:, 2].reshape((len(cutEdges), 1)), 2, axis=-1)
    cutPixelPairs[:, 0, 0:2] += row
    cutPixelPairs[:, 1, 0:2] += col
    
    print row, col, row + col*outImgSize[0]
#     print cutEdges
#     cutEdges[:, 0:2] += row + col*outImgSize[0]
    ## convert pixel coords to global output image coords
    cutEdgesTmp = np.copy(cutEdges)
    cutEdges[:, 0:2] = np.mod(cutEdges[:, 0:2], h) + row + (np.array(cutEdges[:, 0:2]/h, dtype=int)+col)*outImgSize[0]
    
    return outputPatch, cutPixelPairs, cutEdges, oldCutsToKeep, graphModel, allLabels, originalLabels, cutEdgesTmp, labels

# <codecell>

## fill outputImage with patches
outputImg = np.zeros(np.hstack((outputImgSize, [3])), dtype=uint8)

# keep track of covered pixels (at the beginning none of them are covered)
coveredPixels = np.zeros(outputImgSize, dtype=bool)
# counter = 0

## manually initialize outputImg to contain patch (i.e. whole inputImg) in the top left corner
# outputImg[0:patchSize[0], 0:patchSize[1], :] = inputImg
# coveredPixels[0:patchSize[0], 0:patchSize[1]] += 1

# while not np.all(coveredPixels) and counter < 100 :
#     ## make sure only positions that allow overlapping are picked
#     distTransform = cv2.distanceTransform(np.array(coveredPixels, dtype=uint8), cv2.cv.CV_DIST_L2, 5)
#     distTransform = cv2.threshold(distTransform, overlapSize, 1, cv2.THRESH_BINARY)[1]
#     ## positions in validPositions allow for at least overlapSize overlapping pixels
#     validPositions = np.argwhere(distTransform == 1)
#     ## get rid of positions that would not accomodate the size of the patch
#     validPositions = validPositions[np.all(validPositions < outputImgSize-patchSize, axis=-1), :]
#     ## restrict positions 
#     randPos = validPositions[np.random.choice(arange(len(validPositions))), :]
    
#     outputImg[randPos[0]:randPos[0]+patchSize[0], randPos[1]:randPos[1]+patchSize[1], :] = inputImg
#     coveredPixels[randPos[0]:randPos[0]+patchSize[0], randPos[1]:randPos[1]+patchSize[1]] += 1
    
    
#     counter += 1
# colPos = np.random.randint(-patchSize[1], 0)

# outputImg[0:rowPos + patchSize[0], 0:colPos + patchSize[1], :] = inputImg[0:rowPos + patchSize[0], 0:colPos + patchSize[1], :]

# print rowPos, colPos

# close("all")

showFigs = False
useOldPlacements = True

totalCutPixelPairs = np.empty((0, 2, 3))
totalCutEdges = np.empty((0, 17))

if useOldPlacements :
    patchPlacements = np.array([[-82,  -68],
                                [-82,  103],
                                [-82,  169],
                                [  7,  -89],
                                [  7,   88],
                                [  7,  302],
                                [126, -167],
                                [126,   14],
                                [126,  229],
                                [191,  -35],
                                [191,   62],
                                [191,  190]])
#     patchPlacements = np.array([[-89,  -67],
#                                 [-89,   16],
#                                 [-89,  140],
#                                 [-89,  205],
#                                 [-89,  284],
#                                 [ 24, -117],
#                                 [ 24,  -22],
#                                 [ 24,   94],
#                                 [ 24,  218],
#                                 [111,  -58]])
#     patchPlacements = np.array(patchPlacements)
#     patchPlacements = np.array([[-107, -122],
#                                 [-107,  -55],
#                                 [-107,   53],
#                                 [-107,  118],
#                                 [-107,  206],
#                                 [-107,  324],
#                                 [  -8, -112],
#                                 [  -8,   -4],
#                                 [  -8,  119],
#                                 [  -8,  222],
#                                 [ 104,  -30],
#                                 [ 104,   67],
#                                 [ 104,  164],
#                                 [ 104,  271],
#                                 [ 205,  -36],
#                                 [ 205,   67],
#                                 [ 205,  168],
#                                 [ 205,  263],
#                                 [ 294,  -85]])
else :
    patchPlacements = []
    
if useOldPlacements : 
    for placement in patchPlacements:
        rowPos = placement[0]
        colPos = placement[1]
        
#         if colPos == 206 or colPos == 119 :
#             showFigs = True
#         else :
#             showFigs = False
        
        prevOutputImg = np.copy(outputImg)
        overlappingPixels, r, height, c, width = placePatch(rowPos, colPos, inputImg, outputImg, coveredPixels)
        
        if showFigs :
            figure(); imshow(overlappingPixels, interpolation='nearest')
            plot([c, c+width, c+width, c, c], [r, r, r+height, r+height, r])
        
        ## if there are no overlapping pixels there is no need to use graphcut
        if np.any(overlappingPixels) :
            edges, edgeIdxs = getExistingEdgesInPatchRegion(totalCutEdges, colPos, rowPos, gridEdges1D, patchSize, outputImgSize)
            
            ## convert pixel coords from global output image coords to local patch coords
            cutEdgesLocal = np.copy(totalCutEdges[edgeIdxs, :])
            cutEdgesLocal[:, 0:2] = (np.mod(cutEdgesLocal[:, 0:2], outputImgSize[0]) - r + 
                                    (np.array(cutEdgesLocal[:, 0:2]/outputImgSize[0], dtype=int)-c)*height)
            
            cutPatch, cutPixelPairs, cutEdgesGlobal, cutsToKeep, graphModel, allLabels, originalLabels, cutEdges, finalLabels = cutOverlap(r, height, c, width, prevOutputImg, 
                                                                                inputImg, overlappingPixels, outputImgSize, 
                                                                                cutEdgesLocal, showFigs)
            
            ## delete the cut edges that are not needed anymore
            totalCutEdges = np.delete(totalCutEdges, edgeIdxs[np.negative(cutsToKeep)], axis=0)
            outputImg[r:r+height, c:c+width, :] = np.copy(cutPatch)
            totalCutPixelPairs = np.concatenate((totalCutPixelPairs, cutPixelPairs))
            totalCutEdges = np.concatenate((totalCutEdges, cutEdgesGlobal))

else :
    ## get random rows
    rowsPos = [np.random.randint(-patchSize[0]+overlapSize, 0)]
    while rowsPos[-1] < outputImgSize[0]-patchSize[0] :
        rowsPos.append(rowsPos[-1] + np.random.randint(overlapSize, patchSize[0]-overlapSize))
        
    for rowPos in rowsPos:#[0:1] :
        ## get random columns
        colPos = np.random.randint(-patchSize[1]+overlapSize, 0)
        patchPlacements.append(np.array((rowPos, colPos)))
        ## make a copy of previous state of the output image
        prevOutputImg = np.copy(outputImg)
        overlappingPixels, r, height, c, width = placePatch(rowPos, colPos, inputImg, outputImg, coveredPixels)
        
        if showFigs :
            figure(); imshow(overlappingPixels, interpolation='nearest')
            plot([c, c+width, c+width, c, c], [r, r, r+height, r+height, r])
        
        ## if there are no overlapping pixels there is no need to use graphcut
        if np.any(overlappingPixels) :
            edges, edgeIdxs = getExistingEdgesInPatchRegion(totalCutEdges, colPos, rowPos, gridEdges1D, patchSize, outputImgSize)
            
            ## convert pixel coords from global output image coords to local patch coords
            cutEdgesLocal = np.copy(totalCutEdges[edgeIdxs, :])
            cutEdgesLocal[:, 0:2] = (np.mod(cutEdgesLocal[:, 0:2], outputImgSize[0]) - r + 
                                    (np.array(cutEdgesLocal[:, 0:2]/outputImgSize[0], dtype=int)-c)*height)
            
            cutPatch, cutPixelPairs, cutEdgesGlobal, cutsToKeep, graphModel, allLabels = cutOverlap(r, height, c, width, prevOutputImg, 
                                                                                inputImg, overlappingPixels, outputImgSize, 
                                                                                cutEdgesLocal, showFigs)
            
            ## delete the cut edges that are not needed anymore
            totalCutEdges = np.delete(totalCutEdges, edgeIdxs[np.negative(cutsToKeep)], axis=0)
            outputImg[r:r+height, c:c+width, :] = np.copy(cutPatch)
            totalCutPixelPairs = np.concatenate((totalCutPixelPairs, cutPixelPairs))
            totalCutEdges = np.concatenate((totalCutEdges, cutEdgesGlobal))
        
        while colPos < outputImgSize[1]-patchSize[1] :
            colPos += np.random.randint(overlapSize, patchSize[1]-overlapSize)
            ## make a copy of previous state of the output image
            prevOutputImg = np.copy(outputImg)
            patchPlacements.append(np.array((rowPos, colPos)))
            overlappingPixels, r, height, c, width = placePatch(rowPos, colPos, inputImg, outputImg, coveredPixels)
            
            if showFigs :
                figure(); imshow(overlappingPixels, interpolation='nearest')
                plot([c, c+width, c+width, c, c], [r, r, r+height, r+height, r])
            
            ## if there are no overlapping pixels there is no need to use graphcut
            if np.any(overlappingPixels) :
                edges, edgeIdxs = getExistingEdgesInPatchRegion(totalCutEdges, colPos, rowPos, gridEdges1D, patchSize, outputImgSize)
            
                ## convert pixel coords from global output image coords to local patch coords
                cutEdgesLocal = np.copy(totalCutEdges[edgeIdxs, :])
                cutEdgesLocal[:, 0:2] = (np.mod(cutEdgesLocal[:, 0:2], outputImgSize[0]) - r + 
                                        (np.array(cutEdgesLocal[:, 0:2]/outputImgSize[0], dtype=int)-c)*height)
                
                cutPatch, cutPixelPairs, cutEdgesGlobal, cutsToKeep, graphModel, allLabels = cutOverlap(r, height, c, width, prevOutputImg, 
                                                                                    inputImg, overlappingPixels, outputImgSize, 
                                                                                    cutEdgesLocal, showFigs)
                
                ## delete the cut edges that are not needed anymore
                totalCutEdges = np.delete(totalCutEdges, edgeIdxs[np.negative(cutsToKeep)], axis=0)
                outputImg[r:r+height, c:c+width, :] = np.copy(cutPatch)
                totalCutPixelPairs = np.concatenate((totalCutPixelPairs, cutPixelPairs))
                totalCutEdges = np.concatenate((totalCutEdges, cutEdgesGlobal))
        
        
figure(); imshow(outputImg, interpolation='nearest')
figure(); imshow(coveredPixels, interpolation='nearest')

# <codecell>

tmp = 0.0
for factor in graphModel.factors(order=2) :
    if (factor.variableIndices[0] == 23003 or factor.variableIndices[0] == 23187) and factor.variableIndices[1] == 35604:
        print factor, originalLabels[factor.variableIndices[0]], originalLabels[factor.variableIndices[1]], 
        print allLabels[factor.variableIndices[0]], allLabels[factor.variableIndices[1]], factor.max()
        tmp += factor.max()
    if factor.variableIndices[1] == 16325 :
        print factor, factor.max()

# <codecell>

figure(); imshow(overlappingPixels, interpolation='nearest')
ax = gca()
ax.set_autoscale_on(False)
plot([c, c+width, c+width, c, c], [r, r, r+height, r+height, r])

# <codecell>

tmp = np.zeros(outputImgSize)
tmp[np.array(np.mod(cutEdgesGlobal[:, 0], outputImgSize[0]), dtype=int), np.array(cutEdgesGlobal[:, 0]/outputImgSize[0], dtype=int)] = 1

figure(); imshow(tmp, interpolation='nearest')

tmp2 = np.zeros(outputImgSize)
tmp2[np.array(np.mod(edges[:, 0], outputImgSize[0]), dtype=int), np.array(edges[:, 0]/outputImgSize[0], dtype=int)] = 1

figure(); imshow(tmp2, interpolation='nearest')

tmp3 = np.zeros((height, width))
tmp3[np.array(np.mod(cutEdgesLocal[:, 0], height), dtype=int), np.array(cutEdgesLocal[:, 0]/height, dtype=int)] = 1

figure(); imshow(tmp3, interpolation='nearest')

tmp4 = np.zeros((height, width))
tmp4[np.array(np.mod(cutEdges[:, 0], height), dtype=int), np.array(cutEdges[:, 0]/height, dtype=int)] = 1

figure(); imshow(tmp4, interpolation='nearest')

tmp5 = np.zeros((height, width))
tmp6 = np.array([[ 14024.,  14025.],
                 [ 14437.,  14646.],
                 [ 14426.,  14427.],
                 [ 14438.,  14647.],
                 [ 13816.,  14025.],
                 [ 14024.,  14233.]])
tmp5[np.array(np.mod(tmp6[:, 0], height), dtype=int), np.array(tmp6[:, 0]/height, dtype=int)] = 1

figure(); imshow(tmp5, interpolation='nearest')


# tmp5  = np.copy(totalCutEdges[edgeIdxs, :])
# tmp6 = np.array(np.hstack((np.mod(cutEdgesLocal[:, 0:2], outputImgSize[0]), np.array(cutEdgesLocal[:, 0:2]/outputImgSize[0], dtype=int))), dtype=int)
# tmp6[:, 0:2] -= rowPos
# tmp6[:, 2:4] -= c
# tmp5 = tmp6[:, 0:2] + tmp6[:, 2:4]*height
# tmp4 = np.zeros(outputImgSize)
# # tmp4[tmp6[:, 0], tmp6[:, 2]] = 1
# tmp4[np.array(np.mod(tmp5[:, 0], height), dtype=int), np.array(cutEdgesLocal[:, 0]/patchSize[0], dtype=int)]

# figure(); imshow(tmp4, interpolation='nearest')

# <codecell>

# alienNodes = np.array([[72, 190], 
#                        [73, 189], 
#                        [73, 190], 
#                        [74, 188], 
#                        [74, 189], 
#                        [74, 190]])
alienNodes = np.array([[73, 190], 
                       [74, 189], 
                       [74, 190]])
print finalLabels[alienNodes[:, 0], alienNodes[:, 1]]
alienNodes = alienNodes[:, 0] + alienNodes[:, 1]*height
print alienNodes
print cutEdgesLocal.shape[0], width*height+cutEdgesLocal.shape[0]
for factor in graphModel.factors(order=2) :
    if factor.variableIndices[0] in alienNodes or factor.variableIndices[1] in alienNodes:
        print factor, originalLabels[factor.variableIndices[0]], originalLabels[factor.variableIndices[1]], "---",
        print allLabels[factor.variableIndices[0]], allLabels[factor.variableIndices[1]], factor.max()

# <codecell>

figure(); imshow(finalLabels, interpolation='nearest')

# <codecell>

print finalLabels[72:76, 188:192]

# <codecell>

patAPixs = np.arange(0, height, dtype=uint)
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, height*(width-1)+1, height, dtype=uint))))

patBPixs = np.ndarray.flatten(np.argwhere(np.ndarray.flatten(-overlappingPixels[r:r+height, c:c+width], order='F')))
patBPixs = np.unique(np.concatenate((patBPixs, np.arange(height*(width-1), height*width, dtype=uint))))

patAPixs = np.setdiff1d(patAPixs, patBPixs)
patA = prevOutputImg[r:r+height, c:c+width, :]/255.0
patB = inputImg[inputImg.shape[0]-height:, inputImg.shape[1]-width:, :]/255.0

# <codecell>

figure(); imshow(patA, interpolation='nearest')
figure(); imshow(patB, interpolation='nearest')

tmp = np.zeros((height, width))
tmp[np.array(np.mod(patAPixs, height), dtype=int), np.array(patAPixs/height, dtype=int)] = 1
tmp[np.array(np.mod(patBPixs, height), dtype=int), np.array(patBPixs/height, dtype=int)] = 2
figure(); imshow(tmp, interpolation='nearest')

# <codecell>

row, h,      col, w,     imgPatA,       imgPatB,  overlapPixs,       outImgSize,    oldCutEdgesInOverlap, showFigs
r,   height, c,   width, prevOutputImg, inputImg, overlappingPixels, outputImgSize, cutEdgesLocal, showFigs

# <codecell>

# rowPos = patchPlacements[9, 0]
# colPos = patchPlacements[9, 1]

# prevOutputImg = np.copy(outputImg)
# overlappingPixels, r, height, c, width = placePatch(rowPos, colPos, inputImg, outputImg, coveredPixels)

# if showFigs :
#     figure(); imshow(overlappingPixels, interpolation='nearest')
#     plot([c, c+width, c+width, c, c], [r, r, r+height, r+height, r])

## if there are no overlapping pixels there is no need to use graphcut
if np.any(overlappingPixels) :
#     edges, edgeIdxs = getExistingEdgesInPatchRegion(totalCutEdges, colPos, rowPos, gridEdges1D, patchSize, outputImgSize)
    
#     ## convert pixel coords from global output image coords to local patch coords
#     cutEdgesLocal = np.copy(totalCutEdges[edgeIdxs, :])
#     cutEdgesLocal[:, 0:2] = (np.mod(cutEdgesLocal[:, 0:2], outputImgSize[0]) - r + 
#                             (np.array(cutEdgesLocal[:, 0:2]/outputImgSize[0], dtype=int)-c)*height)
    
    cutPatch, cutPixelPairs, cutEdgesGlobal, cutsToKeep, graphModel, allLabels, originalLabels, cutEdges, finalLabels = cutOverlap(r, height, c, width, prevOutputImg, 
                                                                        inputImg, overlappingPixels, outputImgSize, 
                                                                                cutEdgesLocal, True)

# <codecell>

print totalCutEdges[edgeIdxs[244], :]
print (norm(totalCutEdges[edgeIdxs[244], S_A_COLOR] - totalCutEdges[edgeIdxs[244], S_B_COLOR]) + 
            norm(totalCutEdges[edgeIdxs[244], T_A_COLOR] - totalCutEdges[edgeIdxs[244], T_B_COLOR]))

# <codecell>

print cutEdges[-4, 0:2]
print cutEdgesGlobal[-4, 0:2]
## 27796 27797
tmp = np.zeros(outputImgSize)
tmp[np.mod(27796, outputImgSize[0]), int(27796/outputImgSize[0])] = 1
figure(); imshow(tmp, interpolation='nearest')
print np.mod(27796, outputImgSize[0]), int(27796/outputImgSize[0]), np.mod(27796, outputImgSize[0]) + int(27796/outputImgSize[0])*outputImgSize[0]
print np.mod(27797, outputImgSize[0]), int(27797/outputImgSize[0]), np.mod(27797, outputImgSize[0]) + int(27797/outputImgSize[0])*outputImgSize[0]

# <codecell>

print cutEdges[np.ndarray.flatten(np.argwhere(cutEdgesGlobal[:, 1] == 27797)), 0:2]
print cutPatch[np.mod(12005, height), int(12005/height)]/255.0, cutPatch[np.mod(12006, height), int(12006/height)]/255.0
print allLabels[12005:12007]

# <codecell>

patAPixs = np.arange(0, height, dtype=uint)
patAPixs = np.unique(np.concatenate((patAPixs, np.arange(0, height*(width-1)+1, height, dtype=uint))))

patBPixs = np.ndarray.flatten(np.argwhere(np.ndarray.flatten(-overlappingPixels[r:r+height, c:c+width], order='F')))
patBPixs = np.unique(np.concatenate((patBPixs, np.arange(height*(width-1), height*width, dtype=uint))))

patAPixs = np.setdiff1d(patAPixs, patBPixs)
patA = prevOutputImg[r:r+height, c:c+width, :]/255.0
patB = inputImg[inputImg.shape[0]-height:, inputImg.shape[1]-width:, :]/255.0
labels, cutEdges, oldCutsToKeep, graphModel, allLabels, originalLabels = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs, cutEdgesLocal)

# <codecell>

tmp = 0.0
for factor in graphModel.factors(order=2) :
    if (factor.variableIndices[0] == 23370 or factor.variableIndices[0] == 23371) and factor.variableIndices[1] == 35588:
        print factor, originalLabels[factor.variableIndices[0]], originalLabels[factor.variableIndices[1]], 
        print allLabels[factor.variableIndices[0]], allLabels[factor.variableIndices[1]], factor.max()
        tmp += factor.max()
    if factor.variableIndices[0] == 35588 :
        print factor, factor.max()
print tmp

# <codecell>

print np.argwhere(uniqueIdxs(totalCutEdges[:, 0:2]) == False)

# <codecell>

print cutEdgesLocal.shape, width*height, width*height+cutEdgesLocal.shape[0]

# <codecell>

print cutEdgesLocal[35604-width*height, :]
print (norm(cutEdgesLocal[35604-width*height, S_A_COLOR]-cutEdgesLocal[35604-width*height, S_B_COLOR]) 
       + norm(cutEdgesLocal[35604-width*height, T_A_COLOR]-cutEdgesLocal[35604-width*height, T_B_COLOR]))

# <codecell>

print cutEdges[:, 0:2]

# <codecell>

np.argwhere(uniqueIdxs(np.mod(cutFactorIdxs, height) + r + (np.array(cutFactorIdxs/height, dtype=int)+c)*outputImgSize[0]) == False)

# <codecell>

print np.mod(cutFactorIdxs, height) + r + (np.array(cutFactorIdxs/height, dtype=int)+c)*outputImgSize[0]

# <codecell>

print patchPlacements

# <codecell>

### plot image showing old cuts on top of new cuts and an image showing which of the old cuts to keep and which to delete
# oldPixelPairs = np.zeros((len(cutEdgesLocal), 2, 2), dtype=int)
# oldPixelPairs[:, :, 0] = np.array([np.mod(cutEdgesLocal[:, 0], height), np.array(cutEdgesLocal[:, 0]/height, dtype=int)]).T
# oldPixelPairs[:, :, 1] = np.array([np.mod(cutEdgesLocal[:, 1], height), np.array(cutEdgesLocal[:, 1]/height, dtype=int)]).T

# newCutEdges = np.copy(cutEdgesGlobal)
# newCutEdges = (np.mod(newCutEdges[:, 0:2], outputImgSize[0]) - r + 
#               (np.array(newCutEdges[:, 0:2]/outputImgSize[0], dtype=int)-c)*height)

# newPixelPairs = np.zeros((len(newCutEdges), 2, 2), dtype=int)
# newPixelPairs[:, :, 0] = np.array([np.mod(newCutEdges[:, 0], height), np.array(newCutEdges[:, 0]/height, dtype=int)]).T
# newPixelPairs[:, :, 1] = np.array([np.mod(newCutEdges[:, 1], height), np.array(newCutEdges[:, 1]/height, dtype=int)]).T

# visualizeOldAndNewCuts(cutEdgesLocal, tmp5, (height, width))
# visualizeKeptAndOverwrittenCuts(graphModel, (height, width), allLabels, height*width, oldPixelPairs)

# <codecell>

## plot all the cuts and their cost
totalCutCosts = np.zeros(outputImgSize)
# totalCutCosts[np.array(totalCutPixelPairs[:, 0, 0], dtype=int), np.array(totalCutPixelPairs[:, 1, 0], dtype=int)] += totalCutPixelPairs[:, 0, 2]
# totalCutCosts[np.array(totalCutPixelPairs[:, 0, 1], dtype=int), np.array(totalCutPixelPairs[:, 1, 1], dtype=int)] += totalCutPixelPairs[:, 0, 2]
totalCutCosts[np.array(np.mod(totalCutEdges[:, S_IDX], outputImgSize[0]), dtype=int), np.array(totalCutEdges[:, S_IDX]/outputImgSize[0], dtype=int)] += totalCutEdges[:, ST_COST]
totalCutCosts[np.array(np.mod(totalCutEdges[:, T_IDX], outputImgSize[0]), dtype=int), np.array(totalCutEdges[:, T_IDX]/outputImgSize[0], dtype=int)] += totalCutEdges[:, ST_COST]
figure(); imshow(totalCutCosts, interpolation='nearest')
figure(); imshow(outputImg, interpolation='nearest')

figure(); imshow(totalCutCosts, interpolation='nearest')
for [y, x] in patchPlacements :
    plot([x, x+patchSize[1], x+patchSize[1], x, x], [y, y, y+patchSize[0], y+patchSize[0], y])
    

# <codecell>

print edgeIdxs.shape
# print cutsToKeep
print totalCutEdges.shape
print r, height, c, width
print np.mod(cutEdgesGlobal[0, 0], 400), int(cutEdgesGlobal[0, 0]/400)
print cutEdgesGlobal[:, 2]

# <codecell>

figure(); imshow(outputImg, interpolation='nearest')
figure(); imshow(coveredPixels, interpolation='nearest')

# <codecell>

def uniqueIdxs(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return ui[np.argsort(order)]

def uniquify(pixelPairs) :
    # ui = np.logical_and(uniqueIdxs(np.hstack((pixelPairs[:, :, 0], pixelPairs[:, :, 1]))), 
    #                           uniqueIdxs(np.hstack((pixelPairs[:, :, 1], pixelPairs[:, :, 0]))))
    ui = uniqueIdxs(np.hstack((pixelPairs[:, :, 0], pixelPairs[:, :, 1])))
    
    print "deleted", len(pixelPairs)-len(np.argwhere(ui)), "repeated edges"
    print pixelPairs[np.negative(ui), :, 0:2]
    
    return pixelPairs[ui, :, :]

# uniquify(totalCutPixelPairs)

# <codecell>

print totalCutEdges[np.negative(uniqueIdxs(totalCutEdges[:, 0:2])), 0:3]

# <codecell>

print totalCutEdges[np.argwhere(totalCutEdges[:, 0] == 98402), 0:3]

# <codecell>

print totalCutEdges[np.argwhere(totalCutEdges[:, 0] == 98402), 0:3]

# <codecell>

bob = np.array([[34, 67, 234.12], [23, 12, 1.223], [56, 34, 23.21], [65, 89, 13.23]])
print bob
print bob[np.argsort(bob[:, 0:2], axis=-1), :]

# <codecell>

# rowPos = patchPlacements[4][0]
# colPos = patchPlacements[4][1]

# prevOutputImg = np.copy(outputImg)
# overlappingPixels, r, height, c, width = placePatch(rowPos, colPos, inputImg, outputImg, coveredPixels)
figure(); imshow(overlappingPixels, interpolation='nearest')
tic = time.time()
# edges, edgeIdxs = getExistingEdgesInPatchRegion(totalCutPixelPairs, colPos, rowPos, gridEdges2D, outputImgSize)
edges, edgeIdxs = getExistingEdgesInPatchRegion(totalCutEdges, colPos, rowPos, gridEdges1D, patchSize, outputImgSize)
print time.time()-tic

##### SOMETHING WEIRD HAPPENING...
##### IF I SET bob[some coords] to totalCutEdges[same order indices, 2] why 
##### is (bob[some coords] - totalCutEdges[same order indices, 2]) != 0 ??????


bob = np.zeros(outputImgSize)
bob[np.array(np.mod(totalCutEdges[edgeIdxs, 0], outputImgSize[0]), dtype=int), np.array(np.array(totalCutEdges[edgeIdxs, 0]/outputImgSize[0], dtype=int), dtype=int)] = totalCutEdges[edgeIdxs, 2]
figure(); imshow(bob, interpolation='nearest')

joe = np.zeros(outputImgSize)
joe[np.array(totalCutPixelPairs[edgeIdxs2D, 0, 0], dtype=int), np.array(totalCutPixelPairs[edgeIdxs2D, 1, 0], dtype=int)] = totalCutPixelPairs[edgeIdxs2D, 0, 2]
figure(); imshow(joe, interpolation='nearest')

# <codecell>

print totalCutPixelPairs[edgeIdxs2D[342], :, 0]
print totalCutPixelPairs[edgeIdxs2D[342], 0, 2]
print totalCutEdges[edgeIdxs[0], 2]

# <codecell>

print totalCutPixelPairs[edgeIdxs2D[13], :, 0]
print bob[95, 142]
print joe[totalCutPixelPairs[edgeIdxs2D[13], 0, 0], totalCutPixelPairs[edgeIdxs2D[13], 1, 0]]
print totalCutPixelPairs[edgeIdxs2D[13], 0, 2]
print 
print totalCutEdges[edgeIdxs2D[13], 2], totalCutEdges[edgeIdxs2D[13], 0:2]

# <codecell>

joe = np.zeros(outputImgSize)
joe[np.array(totalCutPixelPairs[edgeIdxs2D, 0, 0], dtype=int), np.array(totalCutPixelPairs[edgeIdxs2D, 1, 0], dtype=int)] = totalCutPixelPairs[edgeIdxs2D, 0, 2]
print joe[np.array(totalCutPixelPairs[edgeIdxs2D, 0, 0], dtype=int), np.array(totalCutPixelPairs[edgeIdxs2D, 1, 0], dtype=int)] - totalCutPixelPairs[edgeIdxs2D, 0, 2]
print bob[np.array(np.mod(totalCutEdges[edgeIdxs, 0], outputImgSize[0]), dtype=int), np.array(np.array(totalCutEdges[edgeIdxs, 0]/outputImgSize[0], dtype=int), dtype=int)] - totalCutEdges[edgeIdxs, 2]

# <codecell>

print joe[list(np.array(totalCutPixelPairs[edgeIdxs2D, 0, 0], dtype=int)), list(np.array(totalCutPixelPairs[edgeIdxs2D, 1, 0], dtype=int))]-bob[np.array(np.mod(totalCutEdges[edgeIdxs2D, 0], outputImgSize[0]), dtype=int), np.array(np.array(totalCutEdges[edgeIdxs2D, 0]/outputImgSize[0], dtype=int), dtype=int)]

# <codecell>

print np.vstack([np.array(np.mod(totalCutEdges[edgeIdxs, 0], outputImgSize[0]), dtype=int), 
                 np.array(np.array(totalCutEdges[edgeIdxs, 0]/outputImgSize[0], dtype=int), dtype=int), 
                 totalCutEdges[edgeIdxs, 2]]).T

# <codecell>

print edges

# <codecell>

# edgeIdxs2D = np.copy(edgeIdxs)
# edges2D = np.copy(edges)
print edgeIdxs2D

# <codecell>

print r, height, c, width
print rowPos, colPos
print patchPlacements

# <codecell>

print totalCutPixelPairs
print totalCutEdges[:, 0:3]

# <codecell>

# print edges
edges1D = outputImgSize[1]*(edges[:, 1:4:2]-colPos)+(edges[:, 0:4:2]-rowPos)
print np.any(np.all(edges1D - np.array([49650, 50050]) == 0, axis = -1))
print edges1D

# <codecell>

bob = np.zeros(outputImgSize)
bob[np.array(list(common), dtype=int)[:, 0], np.array(list(common), dtype=int)[:, 1]] = 1
figure(); imshow(bob, interpolation='nearest')

# <codecell>

bckPlaces = np.copy(patchPlacements)

# <codecell>

# close("all")
print np.array(patchPlacements)

# <codecell>

# totalCutPixelPairs[:, 0, 0:2] += r
# totalCutPixelPairs[:, 1, 0:2] += c
print r, c
print totalCutPixelPairs[100:200, :, :]

# <codecell>

print width, height
cutPixelPairs = np.zeros((len(cutEdges), 2, 3))
cutPixelPairs[:, :, 0] = np.array([np.mod(cutEdges[:, 0], height), np.array(cutEdges[:, 0]/height, dtype=int)]).T
cutPixelPairs[:, :, 1] = np.array([np.mod(cutEdges[:, 1], height), np.array(cutEdges[:, 1]/height, dtype=int)]).T
cutPixelPairs[:, :, 2] = np.repeat(cutEdges[:, 2].reshape((len(cutEdges), 1)), 2, axis=-1)

print cutPixelPairs[0, 1, 0]

# <codecell>

print cutPixelPairs.shape
print np.concatenate((np.empty((0, 2, 3)), cutPixelPairs))

# <codecell>

patAPixs = arange(0, height, dtype=uint)
patBPixs = np.argwhere(np.ndarray.flatten(-overlappingPixels[r:r+height, c:c+width], order='F'))
patA = prevOutputImg[r:r+height, c:c+width, :]/255.0
patB = inputImg[patchSize[0]-height:, patchSize[1]-width:, :]/255.0
labels, cutEdges = getGraphcutOnOverlap(patA, patB, patAPixs, patBPixs)

# <codecell>

tmpOutput = np.zeros(patA.shape, dtype=uint8)
for i in xrange(labels.shape[0]) :
    for j in xrange(labels.shape[1]) :
        if labels[i, j] == 0 :
            tmpOutput[i, j, :] = patA[i, j, :]*255
        else :
            tmpOutput[i, j, :] = patB[i, j, :]*255
figure(); imshow(tmpOutput, interpolation='nearest')

# <codecell>

figure(); imshow(overlappingPixels[r:r+height, c:c+width], interpolation='nearest')

# <codecell>

figure(); imshow(labels, interpolation='nearest')

# <codecell>

figure(); imshow(patA, interpolation='nearest')
figure(); imshow(patB, interpolation='nearest')

# <codecell>

figure(); imshow(outputImg, interpolation='nearest')
figure(); imshow(prevOutputImg, interpolation='nearest')
figure(); imshow(coveredPixels , interpolation='nearest')

# <codecell>

print inputImg[patchSize[0]-height:, patchSize[1]-width:, :].shape
print height, width, patchSize, rowPos, colPos, r, c
print outputImg[r:r+height, c:c+width, :].shape

# <codecell>

print rowPos, colPos
print patchSize

# <codecell>

tmp = np.zeros(outputImgSize)
bob = validPositions[np.all(validPositions > outputImgSize-patchSize-np.array([16, 16]), axis=-1), :]
tmp[validPositions[:, 0], validPositions[:, 1]] = 1
# tmp[validPositions[np.all(validPositions + np.array([16, 16]) < outputImgSize-patchSize, axis=-1), :]] = 1
figure(); imshow(tmp)

# <codecell>

distTransform = cv2.distanceTransform(np.array(coveredPixels, dtype=uint8), cv2.cv.CV_DIST_L2, 5)
distTransform = cv2.threshold(distTransform, overlapSize, 1, cv2.THRESH_BINARY)[1]
validPositions = np.argwhere(distTransform == 1)
print validPositions.shape
tmp = validPositions[0:50, :]
print tmp
print outputImgSize-patchSize
print validPositions[np.all(validPositions < outputImgSize-patchSize, axis=-1), :][-1, :]

# <codecell>

print patchSize

# <codecell>

## try filling up outputImage with patches
# setup randomly chosen patches from inputImg in a grid like fashion in outputImg

for r in xrange(0, outputImgSize[0], patchSize[0]-overlapSize) :
    for c in xrange(0, outputImgSize[1], patchSize[1]-overlapSize) :
        patch, patchPos = getRandomPatch(inputImg, patchSize)
        
        outputImg[r:r+patchSize[0], c:c+patchSize[1], :] = patch[0:outputImgSize[0]-r, 0:outputImgSize[1]-c]
        
figure(); imshow(outputImg, interpolation='nearest')

# <codecell>

def getCutsFromLabels(labels) :
    """Figures out where the cut is based on binary labels of pixels
    
           labels: 2D grid of label values per pixel
           
        return: cutEdges list of node pairs that represent the cut"""
    
    if len(labels.shape) != 2 :
        raise Exception("Labels must be a 2d grid")
        
    rows, cols = labels.shape
        
    leftCutNodes = labels[:, 0:-1]-labels[:, 1:]
#     figure(); imshow(leftCutNodes, interpolation='nearest')
    leftCutNodes = np.argwhere(leftCutNodes != 0)
    verticalCuts = np.zeros_like(leftCutNodes)
    verticalCuts[:, 0] = leftCutNodes[:, 0] + rows*leftCutNodes[:, 1]
    verticalCuts[:, 1] = leftCutNodes[:, 0] + rows*(leftCutNodes[:, 1]+1)
    
#     print np.hstack((leftCutNodes, verticalCuts))
    
    topCutNodes = labels[0:-1, :]-labels[1:, :]
#     figure(); imshow(topCutNodes, interpolation='nearest')
    topCutNodes = np.argwhere(topCutNodes != 0)
    horizontalCuts = np.zeros_like(topCutNodes)
    horizontalCuts[:, 0] = topCutNodes[:, 0] + rows*topCutNodes[:, 1]
    horizontalCuts[:, 1] = topCutNodes[:, 0] +1 + rows*topCutNodes[:, 1]
    
#     print np.hstack((topCutNodes, horizontalCuts))
    
    return np.vstack((verticalCuts, horizontalCuts))
    
tic = time.time()
cutEdges = getCutsFromLabels(reshapedLabels)
print time.time() - tic

# <codecell>

for fId in gm.factorIds() :
    print fId

# <codecell>

figure(); imshow(overlapLeft, interpolation='nearest')

# <codecell>

reshapedLabels = np.zeros((patchSize[0], overlapSize)) 
for i in xrange(overlapSize) :
    reshapedLabels[:, i] = labels[i*patchSize[0]:(i+1)*patchSize[0]]

# <codecell>

figure(); imshow(reshapedLabels, interpolation='nearest')

# <codecell>

for i in xrange(patchSize[0]) :
    for j, x in zip(xrange(patchSize[0]-overlapSize, patchSize[0]), xrange(overlapSize)) :
        if reshapedLabels[i, x] == 0 :
            outputImg[i, j, :] = patchLeft[i, j, :]
        else :
            outputImg[i, j, :] = patchRight[i, x, :]
figure(); imshow(outputImg, interpolation='nearest')

# <codecell>

figure(); opengm.visualizeGm(gm, plotUnaries=False, layout='sfdp', iterations=1000)

# <codecell>

opengm.opengmcore.IndexVectorVector(opengm.opengmcore.IndexVector(leftPatchFactors))

# <codecell>

import graph_tool as grt
from graph_tool import draw

# <codecell>

## draw cut edges
graph = grt.generation.lattice([patchSize, overlapSize])

edgeWeights = graph.new_edge_property("double")

for edge in cutEdges :
    edgeWeights[graph.edge(edge[0], edge[1])] = 5.0
        
graph.edge_properties["edgeWeights"] = edgeWeights        
grt.draw.sfdp_layout(graph, cooling_step=0.95, epsilon=1e-2)
grt.draw.graph_draw(graph, vertex_text=graph.vertex_index, vertex_size=40,
                    edge_pen_width=grt.draw.prop_to_size(graph.edge_properties["edgeWeights"], mi=1, ma=10, power=1), 
                    vertex_font_size=12, output_size=(5000, 6000), output="graph.png")

# <codecell>

numNodes = overlapSize*patchSize
graph = grt.generation.lattice([patchSize, overlapSize])

## add extra nodes for rest of patch
patchANode = graph.add_vertex()
patchBNode = graph.add_vertex()
## add edges to correct nodes
patchAEdges = np.hstack((np.ones((patchSize, 1), dtype=uint)*int(patchANode), np.arange(patchSize, dtype=uint).reshape((patchSize, 1))))
graph.add_edge_list(patchAEdges)
patchBEdges = np.hstack((np.ones((patchSize, 1), dtype=uint)*int(patchBNode), np.arange(numNodes-2-patchSize, numNodes-2, dtype=uint).reshape((patchSize, 1))))
graph.add_edge_list(patchBEdges)

## cycle through all edges and add edges from source to target
newEdges = []
for edge in graph.edges() :
    newEdges.append([int(edge.target()), int(edge.source())])
graph.add_edge_list(newEdges)
    
## add pairwise costs
edgeWeights = graph.new_edge_property("double")

pairwise = np.zeros(graph.num_edges())

for edge, i in zip(graph.edges(), xrange(graph.num_edges())) :
#     print i, edge
    if edge.source() == patchANode or edge.source() == patchBNode or edge.target() == patchANode or edge.target() == patchBNode:
        pairwise[i] = 100000.0
        edgeWeights[edge] = pairwise[i]
    else :
        sPix = np.array([int(np.mod(int(edge.source()),patchSize)), int(int(edge.source())/patchSize)])
        tPix = np.array([int(np.mod(int(edge.target()),patchSize)), int(int(edge.target())/patchSize)])
        
        pairwise[i] = norm(patchLeft[sPix[0], sPix[1]+patchSize-overlapSize, :] - patchRight[sPix[0], sPix[1], :]) 
        pairwise[i] += norm(patchLeft[tPix[0], tPix[1]+patchSize-overlapSize] - patchRight[tPix[0], tPix[1]])
        
        edgeWeights[edge] = pairwise[i]

        
graph.edge_properties["edgeWeights"] = edgeWeights

# <codecell>

graph.set_directed(True)

# <codecell>

print graph.edge_properties["edgeWeights"].get_array().shape

# <codecell>

mc, part = grt.flow.min_cut(graph, graph.edge_properties["edgeWeights"])

# <codecell>

# srcNode = graph.add_vertex()
# newEdge = graph.add_edge(srcNode, patchANode)
# edgeWeights = graph.edge_properties["edgeWeights"]
# edgeWeights[newEdge] = 100000.0
# graph.edge_properties["edgeWeights"] = edgeWeights

# sinkNode = graph.add_vertex()
# newEdge = graph.add_edge(sinkNode, patchBNode)
# edgeWeights = graph.edge_properties["edgeWeights"]
# edgeWeights[newEdge] = 100000.0
# graph.edge_properties["edgeWeights"] = edgeWeights

# <codecell>

weights = graph.edge_properties["edgeWeights"]
# res = grt.flow.boykov_kolmogorov_max_flow(graph, srcNode, sinkNode, weights)
# part = grt.flow.min_st_cut(graph, srcNode, weights, res)
tic = time.time()
res = grt.flow.boykov_kolmogorov_max_flow(graph, patchANode, patchBNode, weights)
part = grt.flow.min_st_cut(graph, patchANode, weights, res)
print time.time()-tic

# <codecell>

mc = sum([graph.edge_properties["edgeWeights"][e] - res[e] for e in graph.edges() if part[e.source()] != part[e.target()]])
print mc

# <codecell>

print graph.edge_properties["edgeWeights"].get_array()
print pairwise

# <codecell>

sPix = np.array([int(np.mod(int(17),patchSize)), int(int(17)/patchSize)])
tPix = np.array([int(np.mod(int(18),patchSize)), int(int(18)/patchSize)])

# <codecell>

print part.get_array().shape

# <codecell>

reshapedPart = np.zeros((patchSize, overlapSize)) 
for i in xrange(overlapSize) :
    reshapedPart[:, i] = part.get_array()[i*patchSize:(i+1)*patchSize]

# <codecell>

figure(); imshow(reshapedPart, interpolation='nearest')

# <codecell>

for edge in graph.edges() :
    print edge,graph.edge_properties["edgeWeights"][edge]

# <codecell>

grt.draw.sfdp_layout(graph, cooling_step=0.95, epsilon=1e-2)
grt.draw.graph_draw(graph, vertex_text=graph.vertex_index,
                    edge_pen_width=grt.draw.prop_to_size(graph.edge_properties["edgeWeights"], mi=1, ma=10, power=1), 
                   vertex_font_size=18, vertex_fill_color=part, output_size=(2048, 2048), output="graph.png")
# grt.draw.graph_draw(graph, vertex_text=graph.vertex_index, edge_pen_width=5, mi=1, ma=10, power=1, 
#                     vertex_font_size=18, output_size=(2048, 2048), output="graph.png")

# <codecell>

# patchLeft = cv2.cvtColor(cv2.imread("leftPatch.png"), cv2.COLOR_BGR2RGB)[0:patchSize, -patchSize:]/255.0#np.copy(inputImg[0:patchSize, 0:patchSize, :])
# patchRight = cv2.cvtColor(cv2.imread("rightPatch_bis.png"), cv2.COLOR_BGR2RGB)[0:patchSize, 0:patchSize]/255.0#np.copy(patchLeft)#np.copy(inputImg[-patchSize:, 0:patchSize, :])
patchLeft = np.copy(inputImg[0:patchSize[0], 0:patchSize[0], :])/255.0
patchRight = np.copy(patchLeft)#np.copy(inputImg[-patchSize:, 0:patchSize, :])
# patchRight = np.fliplr(patchRight)

outputImg = np.zeros((patchSize[0], patchSize[0]*2-overlapSize, 3), dtype=patchLeft.dtype)
outputImg[0:patchSize[0], 0:patchSize[0], :] = np.copy(patchLeft)
outputImg[0:patchSize[0], patchSize[0]-overlapSize:, :] = np.copy(patchRight)
figure(); imshow(outputImg, interpolation='nearest')

# <codecell>

## build graph
numLabels = 2
numNodes = overlapSize*patchSize#+2
gm = opengm.gm(numpy.ones(numNodes,dtype=opengm.label_type)*numLabels)

## get unary functions
unaries = np.zeros((numNodes,numLabels))
# set unaries so that they enforce label 0 for first column of pixels and label 1 for last column
unaries[0:patchSize, :] = [0.0, 100000.0]
unaries[-patchSize:, :] = [100000.0, 0.0]
# add functions
fids = gm.addFunctions(unaries)
# add first order factors
gm.addFactors(fids, arange(0, numNodes, 1))

## fix variable labels for node repsresenting left and right patch
# gm.fixVariables([0, numNodes-1], [0, 10])

## get factor indices for the overlap grid of pixels
pairIndices = np.array(opengm.secondOrderGridVis(overlapSize,patchSize,True))#+1
## get pairwise functions for those nodes
pairwise = np.zeros(len(pairIndices))
tmpOverlap = np.zeros((patchSize, overlapSize, 3))
for pair, i in zip(pairIndices, arange(len(pairIndices))) :
    sPix = np.array([int(np.mod(pair[0],patchSize)), int(pair[0]/patchSize)])#-1
    tPix = np.array([int(np.mod(pair[1],patchSize)), int(pair[1]/patchSize)])#-1
    
    pairwise[i] = norm(patchLeft[sPix[0], sPix[1]+patchSize-overlapSize, :] - patchRight[sPix[0], sPix[1], :]) 
    pairwise[i] += norm(patchLeft[tPix[0], tPix[1]+patchSize-overlapSize] - patchRight[tPix[0], tPix[1]])
    tmpOverlap[sPix[0], sPix[1], :] = patchLeft[sPix[0], sPix[1]+patchSize-overlapSize] - patchRight[sPix[0], sPix[1]]
    tmpOverlap[tPix[0], tPix[1], :] = patchLeft[tPix[0], tPix[1]+patchSize-overlapSize] - patchRight[tPix[0], tPix[1]]
#     if np.mod(pair[0], patchSize) == 14:
#         pairwise[i] = 0.0
    fid = gm.addFunction(np.array([[0.0, pairwise[i]],[pairwise[i], 0.0]]))
    gm.addFactor(fid, pair)
#     print patchLeft[sPix[0], sPix[1]+patchSize-overlapSize], patchRight[sPix[0], sPix[1]], sPix
    
## first node is left and last is right patch respectively    
## get factor indices from left patch to first column and from last column to right patch
# connect left patch to first pixel column
# idxLeftPatchNode = numNodes - 2
# leftPatchFactors = np.hstack((np.zeros((patchSize, 1), dtype=uint), np.arange(patchSize, dtype=uint).reshape((patchSize, 1))+1))
# # connect last pixel column to right patch
# idxRightPatchNode = numNodes - 1
# rightPatchFactors = np.hstack((np.arange(numNodes-2-patchSize, numNodes-2, dtype=uint).reshape((patchSize, 1))+1, np.ones((patchSize, 1), dtype=uint)*idxRightPatchNode))
# # add these last two sets of factors to the overlap grid factors
# fid = gm.addFunction(np.array([[100000.0, 100000.0],[100000.0, 100000.0]]))
# gm.addFactors(fid, leftPatchFactors)
# gm.addFactors(fid, rightPatchFactors)

# gm.addFunctions(pairwise)

# def regularizer(labels):
#     val=abs(float(labels[0])-float(labels[1]))
# #     print labels[0], labels[1]
#     return val#*0.4  

# regs=opengm.PythonFunction(function=regularizer,shape=[numLabels,numLabels])
# fid = gm.addFunction(regs)

# gm.addFactors(fid, pairIndices)
graphCut = opengm.inference.GraphCut(gm=gm)
tic = time.time()
graphCut.infer()
print time.time() - tic
labels = graphCut.arg()
# labels = labels.reshape((patchSize, overlapSize))

print gm

