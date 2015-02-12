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
import os
from scipy import ndimage
from scipy import stats

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

dataFolder = "data/"

# <codecell>

## read frames from sequence of images
sampleData = "ribbon1_matte/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame*.png")
frames = np.sort(frames)
numFrames = len(frames)
print numFrames
if numFrames > 0 :
    frameSize = cv2.imread(frames[0]).shape
    movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]))
    for i in range(0, numFrames) :
        im = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB))#/255.0
        movie[:, :, :, i] = im#np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
        
#movie = np.dot(movie[...,:3], [0.299, 0.587, 0.144])
print 'Movie has shape', movie.shape

# <codecell>


movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]))

# <codecell>

sampleVideo = "Videos/649f810.avi"
outputData = dataFolder+sampleVideo
cap = cv2.VideoCapture(dataFolder+sampleVideo)

frames = []
tic = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        k = cv2.waitKey(30)
        
        sys.stdout.write('\r' + "Frames read: " + np.string_(len(frames)))
        sys.stdout.flush()
    else:
        break

# Release everything if job is finished
cap.release()
# movie = array(frames)

## init movie array
if len(frames) > 0 :
    movie = np.zeros(np.hstack([frames[0].shape, len(frames)]))
    for i in xrange(0, len(frames)) :
        movie[:, :, :, i] = frames[i]
        
        sys.stdout.write('\r' + "Frames saved: " + np.string_(i) + " of " + np.string_(len(frames)))
        sys.stdout.flush()

print
print time.time()-tic
# del frames

# <codecell>

def histEq(frame) :
    frame = np.array(frame, dtype=uint8)
    hstg = ndimage.measurements.histogram(frame, 0, 255, 256)
    csumhstg = np.cumsum(hstg)
    normcsum = cv2.normalize(csumhstg, None, 0, 255, cv2.NORM_MINMAX)
    eqFrame = np.zeros_like(frame)
    eqFrame = np.reshape(normcsum[frame], frame.shape)
    return eqFrame

# <codecell>

## returns a diagonal kernel based on given binomial coefficients
def diagkernel(c) :
    k = np.eye(len(c))
    k = k * c/np.sum(c)
    return k

# <codecell>

##  compute l2 distance between given frames
def l2dist(f1, f2) : 
    img = f1 - f2
    result = np.linalg.norm(img)
#     img = img ** 2
#     result = np.sqrt(np.sum(img))
    return result

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

# <codecell>

## Turn distances to probabilities
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

# <codecell>

## get a random frame based on given probabilities
def randFrame(probs) :
    indices = np.argsort(probs)[::-1] # get descending sort indices
    sortedProbs = probs[indices] # sort probs in descending order
#     print 'sortedProbs', sortedProbs
#     print np.int(probs.shape[0]/10)
    sortedProbs = sortedProbs[0:5]#np.int(probs.shape[0]/10)] # get highest 10%
#     sortedProbs = sortedProbs[0]
#     print 'sortedProbs', sortedProbs
    sortedProbs = sortedProbs/np.sum(sortedProbs) # normalize
#     print 'sortedProbs', sortedProbs
    prob = np.random.rand(1)
#     print 'prob', prob
    csum = np.cumsum(sortedProbs)
    j = 0
    while csum[j] < prob :
        j = j+1
#     print 'final j', j
    return indices[j]

# <codecell>

def doAlphaBlend(pairs, weights, movie) :
    blended = np.zeros(np.hstack((movie.shape[0:-1], len(pairs))), dtype=uint8)
    for pair, w, idx in zip(pairs, weights, xrange(0, len(pairs))) :
#         print pair, idx, w
        if pair[0] != pair[1] :
            blended[:, :, :, idx] = movie[:, :, :, pair[0]]*w+movie[:, :, :, pair[1]]*(1.0-w)
    return blended

# <codecell>

## compute distance between each pair of images
def computeDistanceMatrix(movie, outputData) :
    try :
        ## load distanceMatrix from file
        distanceMatrix = np.load(outputData+"_distanceMatrix.npy")
        print "loaded distance matrix from ", outputData, "_distanceMatrix.npy"
    except IOError :
        distanceMatrix = np.zeros([movie.shape[3], movie.shape[3]])
        distanceMatrix = distEuc(np.reshape(movie/255.0, [np.prod(movie.shape[0:-1]), movie.shape[-1]]).T)
        # for i in range(0, movie.shape[3]) :
        #     for j in range(i+1, movie.shape[3]) :
        #         distanceMatrix[j, i] = distanceMatrix[i, j] = l2dist(movie[:, :, :, i], movie[:, :, :, j])
        # #         print distanceMatrix[j, i],
        #     print (movie.shape[3]-i),
            
        ## save file
        np.save(outputData+"_distanceMatrix", distanceMatrix)
    
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(distanceMatrix, cmap = cm.Greys_r, interpolation='nearest')
    a.set_title('distance matrix grey')
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(distanceMatrix, interpolation='nearest')
    maxDist = np.max(distanceMatrix)
    plt.colorbar(ticks=[0.1*maxDist,0.3*maxDist,0.5*maxDist,0.7*maxDist,0.9*maxDist], orientation ='horizontal')
    a.set_title('distance matrix color')
    
    return distanceMatrix

# <codecell>

## Preserve dynamics: convolve wih binomial kernel
def filterDistanceMatrix(distanceMatrix, numFilterFrames, isRepetitive) :
    # numFilterFrames = 4 ## actual total size of filter is numFilterFrames*2 +1
    # isRepetitive = False ## see if this can be chosen automatically

    if isRepetitive :
        kernel = np.eye(numFilterFrames*2+1)
    else :
        coeff = sp.special.binom(numFilterFrames*2, range(0, numFilterFrames*2 +1)); print coeff
        kernel = diagkernel(coeff)
    # distanceMatrixFilt = cv2.filter2D(distanceMatrix, -1, kernel)
    distanceMatrixFilt = ndimage.filters.convolve(distanceMatrix, kernel, mode='constant')
    distanceMatrixFilt = distanceMatrixFilt[numFilterFrames:-numFilterFrames,numFilterFrames:-numFilterFrames]
    
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(distanceMatrixFilt, cmap = cm.Greys_r, interpolation='nearest')
    a.set_title('filtered matrix grey')
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(distanceMatrixFilt, interpolation='nearest')
    maxDist = np.max(distanceMatrixFilt)
    plt.colorbar(ticks=[0.1*maxDist,0.3*maxDist,0.5*maxDist,0.7*maxDist,0.9*maxDist], orientation ='horizontal')
    a.set_title('filtered matrix color')
    
    return distanceMatrixFilt

# <codecell>

## Avoid dead ends: estimate future costs
def estimateFutureCost(alpha, p, distanceMatrixFilt) :
    # alpha = 0.999
    # p = 2.0
    
    distMatFilt = distanceMatrixFilt[1:distanceMatrixFilt.shape[1], 0:-1]
    distMat = distMatFilt ** p
    
    last = np.copy(distMat)
    current = np.zeros(distMat.shape)
    
    ## while distance between last and current is larger than threshold
    iterations = 0 
    while np.linalg.norm(last - current) > 0.1 : 
        for i in range(distMat.shape[0]-1, -1, -1) :
            m = np.min(distMat, axis=1)
            distMat[i, :] = (distMatFilt[i, :] ** p) + alpha*m
            
        last = np.copy(current)
        current = np.copy(distMat)
        
        sys.stdout.write('\r' + "Iteration " + np.string_(iterations) + "; distance " + np.string_(np.linalg.norm(last - current)))
        sys.stdout.flush()
        
        iterations += 1
    
    print
    print 'finished in', iterations, 'iterations'
    #distanceMatrix = distMat
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(distMat, cmap = cm.Greys_r, interpolation='nearest')
    a.set_title('future cost matrix grey')
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(distMat, interpolation='nearest')
    maxDist = np.max(distMat)
    plt.colorbar(ticks=[0.1*maxDist,0.3*maxDist,0.5*maxDist,0.7*maxDist,0.9*maxDist], orientation ='horizontal')
    a.set_title('future cost matrix color')
    
    return distMat

# <codecell>

def getProbabilities(distMat, sigmaMult, normalizeRows) :
    ## compute probabilities from distanceMatrix and the cumulative probabilities
    probabilities = dist2prob(distMat, sigmaMult, normalizeRows)
    # since the probabilities are normalized on each row, the right most column will be all ones
    cumProb = cumsum(probabilities, axis=1)
    print probabilities.shape, cumProb.shape
    
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(probabilities, interpolation='nearest')
    a.set_title('Transition Probabilities')
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(cumProb, interpolation='nearest')
    a.set_title('Transition Cumulative Probabilities')
    
    return probabilities, cumProb

# <codecell>

## Find a sequence of frames based on final probabilities
def getFinalFrames(cumProb, totalFrames, correction) :
    # totalFrames = 1000
    finalFrames = np.zeros(totalFrames)
    
    ## prob matrix is shrunk so row indices don't match frame numbers anymore unless corrected
    # correction = numFilterFrames+1
    
    
    currentFrame = np.ceil(np.random.rand()*(cumProb.shape[0]-1))
    # currentFrame = 400
    print 'starting at frame', currentFrame+correction
    finalFrames[0] = currentFrame
    for i in range(1, totalFrames) :
    #     currentFrame = randFrame(probabilities[currentFrame, :])
        prob = np.random.rand(1)
        currentFrame = np.round(np.sum(cumProb[currentFrame, :] < prob))
        finalFrames[i] = currentFrame
        print 'frame', i, 'of', totalFrames, 'taken from frame', currentFrame+correction, prob
        
    
    finalFrames = finalFrames+correction
    
    fig = plt.figure()
    plt.plot(finalFrames)
    fig.suptitle('Final Rendered Frames')
    plt.ylabel('frame index')
    plt.xlabel('time')
    plt.ylim([0, movie.shape[3]])
    plt.yticks(arange(0, movie.shape[3], 5))
    return finalFrames

# <codecell>

## render the frames in finalFrames
def renderFinalFrames(movie, finalFrames, numInterpolationFrames) :
    finalMovie = []
    finalMovie.append(np.copy(movie[:, :, :, finalFrames[0]]))
    jump = 0
    cycleLengh = 1
    f = 1
    
    while f < len(finalFrames) :
        finalMovie.append(np.copy(movie[:, :, :, finalFrames[f]]))
        
        if numInterpolationFrames < 1 :
            continue
        
        ## if it's a jump then do some sort of frame interpolation
        if finalFrames[f] == finalFrames[f-1]+1 :
            cycleLengh +=1
        else :
            cycleLengh = 1
            jump += 1
            print "jump", jump, "from frame", finalFrames[f-1], "to frame", finalFrames[f]
            
            ## find pairs of frames to interpolate between
            toInterpolate = np.zeros([numInterpolationFrames*2, 2])
            toInterpolate[:, 0] = np.arange(finalFrames[f-1]-numInterpolationFrames+1,finalFrames[f-1]+numInterpolationFrames+1)
            toInterpolate[:, 1] = np.arange(finalFrames[f]-numInterpolationFrames,finalFrames[f]+numInterpolationFrames)
            
            ## correct for cases where we're at the beginning of the movie or at the end
            if f < numInterpolationFrames :
                toIgnore = numInterpolationFrames-f
                toInterpolate[0:toIgnore, :] = 0
            elif f > len(finalFrames)-numInterpolationFrames :
                toIgnore = numInterpolationFrames+f-len(finalFrames)
                toInterpolate[-toIgnore:numInterpolationFrames*2, :] = 0
                
            ## do alpha blending 
            blendWeights = np.arange(1.0-1.0/(numInterpolationFrames*2.0), 0.0, -1.0/((numInterpolationFrames+1)*2.0))
            blended = doAlphaBlend(toInterpolate, blendWeights, movie)
            frameIndices = xrange(f-numInterpolationFrames, f+numInterpolationFrames)
            # put blended frames into finalMovie
            for idx, b in zip(frameIndices, xrange(0, len(blended))) :
                if idx >= 0 and idx < len(finalMovie) :
                    finalMovie[idx] = np.copy(blended[:, :, :, b])
                elif idx == len(finalMovie) and idx < len(finalFrames):
                    finalMovie.append(np.copy(blended[:, :, :, b]))
                    f += 1 ## jumping frame now as otherwise I would copy it back from finalFrames instead of having the interpolated version
        
        f += 1
    
    return finalMovie

# <codecell>

distanceMatrix = computeDistanceMatrix(movie, outputData)
numFilterFrames = 4
distanceMatrixFilt = filterDistanceMatrix(distanceMatrix, numFilterFrames, True)
distMat = estimateFutureCost(0.999, 2.0, distanceMatrixFilt)
probabilities, cumProb = getProbabilities(distMat, 0.005, True)

# <codecell>

finalFrames = getFinalFrames(cumProb, 100, numFilterFrames+1)
finalMovie = renderFinalFrames(movie, finalFrames, numFilterFrames)

# <codecell>

tmp = renderFinalFrames(movie, finalFrames, numFilterFrames)

# <codecell>

print len(tmp)

# <codecell>

print finalMovie.shape

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
img = plt.imshow(np.array(finalMovie[0], dtype=uint8))
# img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
    img.set_data(np.array(finalMovie[0], dtype=uint8))
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f) + ' taken from ' + str(finalFrames[f]))
    img.set_data(np.array(finalMovie[f], dtype=uint8))
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalFrames),interval=33,blit=True)

# writer = animation.writers['ffmpeg'](fps=30)
# ani.save('demoa.mp4', writer=writer,dpi=160, bitrate=100)

plt.show()

# <codecell>

## save video
def saveMovieToAvi(finalMovie, outputData, videoName) :
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    # videoName = "_ab.avi" if doInterpolation else "_no_ab.avi"
    out = cv2.VideoWriter(outputData+videoName,fourcc, 30.0, (movie.shape[1], movie.shape[0]))
    
    for f in xrange(0, finalMovie.shape[-1]) :
        frame = cv2.cvtColor(np.array(finalMovie[:, :, :, f]), cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    # Release everything if job is finished\
    out.release()

# <codecell>

if not os.path.exists(outputData + "_loop/"):
    os.makedirs(outputData + "_loop/")
for f in xrange(0, finalMovie.shape[-1]) :
    imsave(outputData + "_loop/" + str_(f) + ".png", finalMovie[:, :, :, f])

# <codecell>

print distanceMatrix.shape, distanceMatrixFilt.shape, distMatFilt.shape, distMat.shape

# <codecell>

# equalize frames in movie for visualization purposes
eqMovie = np.zeros_like(movie)
for i in xrange(0, eqMovie.shape[2]) :
    eqMovie[:, :, i] = np.array(histEq(movie[:, :, i]), dtype=float)/255.0

