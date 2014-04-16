# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import re
import sys

# <codecell>

## transforms quaternion to a rotation matrix (taken from vSFM via Clem)
## need to take transpose of m to get correct rotation
def qToR(q):
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
   
    qq = np.sqrt(a*a+b*b+c*c+d*d)
   
    if (qq <= 0.):
        a = 1.
        b = c = d = 0.
    m = np.array([[a*a+b*b-c*c-d*d, 2*b*c-2*a*d, 2*b*d+2*a*c], [2*b*c+2*a*d, a*a-b*b+c*c-d*d, 2*c*d-2*a*b], [2*b*d-2*a*c, 2*c*d+2*a*b, a*a-b*b-c*c+d*d]])
    return m
    

# <codecell>

# threshold is the distance threshold
def ransac(points, threshold, iterations) :
    ## find best line to fit to set of points using RANSAC
    randomIdx = np.round(np.random.rand(iterations, 2)*(len(points)-1))
    
    # line that best fits the points
    bestFit = np.zeros(2*points.shape[1]) # n = bestFit[0:points.shape[1]]; a = bestFit[points.shape[1]:len(bestFit)]
#     print bestFit.shape
    # save the list of inliers for best fit (point indices)
    bestFitInliers = list()
    # save the indices of the points used for best fit
    bestFitPoints = np.zeros(2)
    
    # for bookeeping inliers while searching best fit
    inliers = list()
    count = 0
    for idxs in randomIdx :
        if idxs[0] != idxs[1] :
            inliers = list()
            a = points[idxs[0]] 
            b = points[idxs[1]]
            v = a - b
            n = v / np.linalg.norm(v)
            for idx in xrange(0, len(points)) :
                p = points[idx]
                distance = np.linalg.norm(a - p - np.dot(np.dot(a-p, n), n)) # distance between point p and line x = a +tn
                if distance <= threshold :
                    inliers.append(idx)
            
            ## check if found new best fit
            if len(inliers) > len(bestFitInliers) :
                bestFit[0:points.shape[1]] = n
                bestFit[points.shape[1]:len(bestFit)] = a
                bestFitInliers = np.copy(inliers)
                bestFitPoints = np.copy(idxs)
        else :
            print "got same indices"
        sys.stdout.write('\r' + "Ransac iteration " + np.string_(count) + " of " + np.string_(iterations))
        sys.stdout.flush()
        count += 1
    print
    
    return bestFit, bestFitInliers, bestFitPoints

# <codecell>

def fitLine2D(points) :
    if(points.shape[1]!=2):
        raise RuntimeError("Points must be 2D")

    ####### use Ax=b approach ######
#     A = np.hstack((np.reshape(points[:, 0],[len(points), 1]), np.ones([len(points), 1])))
#     x = np.dot(np.linalg.pinv(A), np.reshape(points[:, 1],[len(points), 1]))
        
#     p1 = np.array([0, x[1]])
#     p2 = np.array([-x[1]/x[0], 0])
    ################################
    ####### use svd approach ######
    A = np.hstack((points, np.ones([len(points), 1])))
    U, S, V = np.linalg.svd(A)
    x = V.T[:, len(V)-1]
    
    p1 = np.array([0, -x[2]/x[1]])
    p2 = np.array([-x[2]/x[0], 0])
    ###############################
    
    u = (p1-p2) / np.linalg.norm(p1-p2)
    
    return np.array(np.hstack((u, p1)), dtype=float)

def fitLine3D(points) :
    if(points.shape[1]!=3):
        raise RuntimeError("Points must be 3D")
        
    ####### use svd approach ######
    A = points
    r0 = np.mean(A, axis=0)
    A = A - np.repeat(np.reshape(r0, [1, len(r0)]), len(A), axis=0)
    U, S, V = np.linalg.svd(A)
    n = V.T[:, 0]
    ###############################
    
    return np.array(np.hstack((n, r0)), dtype=float)

def fitPlane(points) :
    if(points.shape[1]!=3):
        raise RuntimeError("Points must be 3D")
        
    ####### use svd approach ######
    A =  np.hstack((points, np.ones([len(points), 1])))
    U, S, V = np.linalg.svd(A)
    params = V.T[:, len(V)-1]
    
    n = np.array(params[0:-1])
    r0 = np.mean(points, axis=0) # take r0 as mean of given points
    ###############################
    
    return np.array(np.hstack((n, r0)), dtype=float)
#     return params

# <codecell>

##  compute l2 distance between given frames
def l2dist(f1, f2) : 
    img = f1 - f2
    img = img ** 2
    result = np.sqrt(np.sum(img))
    return result

# <codecell>

## read nvm file one line at a time and store lines containing camera locations
def readCameraLocsNVM(location, filename) :
    nameLength = len(filter(None, re.split('/',location)))
    ins = open(location+filename, "r")
    count = 0
    cameraList = list()
    for line in ins:
        if line.startswith(location) :
            cameraList.append(line)
            count += 1
    ins.close()
    print "found", len(cameraList), "cameras"
    
    ## process cameras to get transformation matrices
    # x = np.zeros((2,),dtype=('a10, d4, d4, d4, d4, d4, d4, d4'))
    cameras = np.empty((len(cameraList), 7), dtype=np.double) # contains quaternion rotation and camera center
    cameraNames = np.empty(len(cameraList), dtype='S20')
    idx = 0
    for camera in cameraList :
        # get frame name
        cameraNames[idx] = filter(None, re.split('/| |\t',camera))[nameLength]
        # get rotation (first 4 doubles) and camera center (last 3 doubles)
        cameras[idx] = np.array(filter(None, re.split(' |\t',camera))[2:9], dtype=np.double)
        idx += 1
    
    ## sort cameras based on frame name
    sortIdx = np.argsort(cameraNames)
    cameraNames = cameraNames[sortIdx]
    cameras = cameras[sortIdx]
    return cameras, cameraNames

# <codecell>

def getDistanceMatrix(location, frames) :
    try :
        ## load distanceMatrix from file
        distanceMatrix = np.load(location+"distanceMatrix.npy")
    except IOError :
        ## load frames
        frameSize = cv2.imread(location+frames[0]).shape
        movie = np.zeros(np.hstack([frameSize[0], frameSize[1], len(cameraNames)]))
        for i in range(0, len(frames)) :
            im = np.array(cv2.imread(location+frames[i]))/255.0
            movie[:, :, i] = np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
            sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(frames)))
            sys.stdout.flush() 
        
        ## compute distance between each pair of images
        distanceMatrix = np.zeros([len(frames), len(frames)])
        for i in range(0, len(frames)) :
            for j in range(i, len(frames)) :
                distanceMatrix[j, i] = distanceMatrix[i, j] = l2dist(movie[:, :, i], movie[:, :, j])
            sys.stdout.write('\r' + "Computing distance to frame " + np.string_(i) + " of " + np.string_(len(frames)))
            sys.stdout.flush() 
            
        ## save file
        np.save(dataLoc+"distanceMatrix", distanceMatrix)
    return distanceMatrix

