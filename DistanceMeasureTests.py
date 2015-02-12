# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab 
import numpy as np
import cv2
import glob
import time
import sys
import ssim
from PIL import Image

import VideoTexturesUtils as vtu

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

dataFolder = "/home/ilisescu/PhD/data/"

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
# sampleData = "ribbon1_matted/"
sampleData = "little_palm1/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "*.png")
maxFrames = 60#len(frames)
frames = np.sort(frames)[0:maxFrames]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames
if numFrames > 0 :
    frameSize = cv2.imread(frames[0]).shape
    movie = np.zeros(np.hstack([frameSize[0], frameSize[1], frameSize[2], numFrames]), dtype=np.uint8)
    for i in range(0, numFrames) :
        movie[:, :, :, i] = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB), dtype=np.uint8)
        
        sys.stdout.write('\r' + "Loaded frame " + np.string_(i) + " of " + np.string_(numFrames))
        sys.stdout.flush()

print        
print 'Movie has shape', movie.shape

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

def distSSIM(f1, f2) :
    ## compute_ssim returns ssim score (which is in [-1, 1]) and I need a distance measure (in [0, 1])
#     return (1 - ssim.compute_ssim(f1, f2))/2
    return (1 - getssim(f1, f2))/2

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

s = time.time()
ssimDist = np.zeros((len(frames), len(frames)))

for i in xrange(0, len(frames)) :
    for j in xrange(i+1, len(frames)) :
        ssimDist[i, j] = ssimDist[j, i] = distSSIM(frames[i], frames[j])

# distSSIM(frames[0], frames[0])
figure(); imshow(ssimDist, interpolation='nearest')
print "finished in", time.time() - s

# <codecell>

np.save("ssimDist60-little_palm", ssimDist)

# <codecell>

costMat = np.copy(vtu.filterDistanceMatrix(distanceMatrix, 4, True))
figure(); imshow(costMat, interpolation='nearest')
# only allow backwards transitions
costMat[np.triu_indices(len(costMat), k=-10+1)] = np.max(costMat)
print np.argmin(costMat), np.min(costMat), costMat.shape, costMat[38, 17]
print np.mod(np.argmin(costMat), len(costMat)), np.argmin(costMat)/len(costMat)

# <codecell>

def getssim(f1, f2) :
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = ssim.utils.get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    
    # img1 = Image.open("/home/ilisescu/PhD/data/little_palm1/frame-00001.png")
    img1 = Image.open(f1)
    img1size = img1.size
    img1_gray, img1_alpha = ssim.utils.to_grayscale(img1)
    
    # if img1_alpha is not None:
    #     img1_gray[img1_alpha == 255] = 0
        
    img1_gray_squared = img1_gray ** 2
    
    img1_gray_mu = ssim.utils.convolve_gaussian_2d(img1_gray, gaussian_kernel_1d)
    
    img1_gray_mu_squared = img1_gray_mu ** 2
    
    img1_gray_sigma_squared = ssim.utils.convolve_gaussian_2d(img1_gray_squared, gaussian_kernel_1d)
    
    img1_gray_sigma_squared -= img1_gray_mu_squared
    
    img2 = Image.open(f2)
    img2size = img2.size
    img2_gray, img2_alpha = ssim.utils.to_grayscale(img2)
    
    # if img2_alpha is not None:
    #     img2_gray[img2_alpha == 255] = 0
        
    img2_gray_squared = img2_gray ** 2
    
    img2_gray_mu = ssim.utils.convolve_gaussian_2d(img2_gray, gaussian_kernel_1d)
    
    img2_gray_mu_squared = img2_gray_mu ** 2
    
    img2_gray_sigma_squared = ssim.utils.convolve_gaussian_2d(img2_gray_squared, gaussian_kernel_1d)
    
    img2_gray_sigma_squared -= img2_gray_mu_squared
    
    l=255
    k_1=0.01
    k_2=0.03
    c_1 = (k_1 * l) ** 2
    c_2 = (k_2 * l) ** 2
            
    img_mat_12 = img1_gray * img2_gray
    img_mat_sigma_12 = ssim.utils.convolve_gaussian_2d(img_mat_12, gaussian_kernel_1d)
    img_mat_mu_12 = img1_gray_mu * img2_gray_mu
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12
    
    # Numerator of SSIM
    num_ssim = ((2 * img_mat_mu_12 + c_1) *
                (2 * img_mat_sigma_12 + c_2))
    
    # Denominator of SSIM
    den_ssim = (
        (img1_gray_mu_squared + img2_gray_mu_squared +
         c_1) *
        (img1_gray_sigma_squared +
         img2_gray_sigma_squared + c_2))
    
    ssim_map = num_ssim / den_ssim
    index = numpy.average(ssim_map)
    return index

# <codecell>

print getssim(frames[0], frames[-1])

# <codecell>

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = ssim.utils.get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

# img1 = Image.open("/home/ilisescu/PhD/data/little_palm1/frame-00001.png")
img1 = Image.open(frames[0])
img1size = img1.size
img1_gray, img1_alpha = ssim.utils.to_grayscale(img1)

# if img1_alpha is not None:
#     img1_gray[img1_alpha == 255] = 0
    
img1_gray_squared = img1_gray ** 2

img1_gray_mu = ssim.utils.convolve_gaussian_2d(img1_gray, gaussian_kernel_1d)

img1_gray_mu_squared = img1_gray_mu ** 2

img1_gray_sigma_squared = ssim.utils.convolve_gaussian_2d(img1_gray_squared, gaussian_kernel_1d)

img1_gray_sigma_squared -= img1_gray_mu_squared

# <codecell>

img2 = Image.open(frames[-1])
img2size = img2.size
img2_gray, img2_alpha = ssim.utils.to_grayscale(img2)

# if img2_alpha is not None:
#     img2_gray[img2_alpha == 255] = 0
    
img2_gray_squared = img2_gray ** 2

img2_gray_mu = ssim.utils.convolve_gaussian_2d(img2_gray, gaussian_kernel_1d)

img2_gray_mu_squared = img2_gray_mu ** 2

img2_gray_sigma_squared = ssim.utils.convolve_gaussian_2d(img2_gray_squared, gaussian_kernel_1d)

img2_gray_sigma_squared -= img2_gray_mu_squared

# <codecell>

figure(); imshow(img2_gray_sigma_squared)
# print img1_alpha
# print img1.getbands()

# <codecell>

l=255
k_1=0.01
k_2=0.03
c_1 = (k_1 * l) ** 2
c_2 = (k_2 * l) ** 2
        
img_mat_12 = img1_gray * img2_gray
img_mat_sigma_12 = ssim.utils.convolve_gaussian_2d(img_mat_12, gaussian_kernel_1d)
img_mat_mu_12 = img1_gray_mu * img2_gray_mu
img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

# Numerator of SSIM
num_ssim = ((2 * img_mat_mu_12 + c_1) *
            (2 * img_mat_sigma_12 + c_2))

# Denominator of SSIM
den_ssim = (
    (img1_gray_mu_squared + img2_gray_mu_squared +
     c_1) *
    (img1_gray_sigma_squared +
     img2_gray_sigma_squared + c_2))

ssim_map = num_ssim / den_ssim
index = numpy.average(ssim_map)
print index

# <codecell>

print frames[0], frames[65]
ssim.compute_ssim(frames[0], frames[65])

# <codecell>

ssimDist = np.load("ssimDist.npy")

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

## check how cumProbs look like
distMatFilt = vtu.filterDistanceMatrix(ssimDist, 4, True)
distMat = estimateFutureCost(0.999, 2.0, distMatFilt, np.ones(distMatFilt.shape))
probabilities, cumProb = vtu.getProbabilities(distMat, 0.005, True)
figure(); imshow(cumProb, interpolation='nearest')

# <codecell>

## test what kind of finalFrames result from computed cumProbs
startFrame = np.argmin(np.round(np.sum(cumProb < 0.5, axis=0)))
print startFrame
finalFrames = vtu.getFinalFrames(cumProb, 100, 5, startFrame, True, False)
print finalFrames
print finalFrames.shape
## check jumps 
for i in xrange(1, len(finalFrames)) :
    if finalFrames[i] != finalFrames[i-1]+1 :
        print "jump at frame", i, "from", finalFrames[i-1], "to", finalFrames[i]

# <codecell>

## test what kind of finalFrames result from computed cumProbs
startFrame = np.argmin(np.round(np.sum(cumProb < 0.5, axis=0)))
print startFrame
finalFrames = vtu.getFinalFrames(cumProb, 100, 5, startFrame, True, False)
print finalFrames
print finalFrames.shape
## check jumps 
for i in xrange(1, len(finalFrames)) :
    if finalFrames[i] != finalFrames[i-1]+1 :
        print "jump at frame", i, "from", finalFrames[i-1], "to", finalFrames[i]

# <codecell>

print ssimDist/np.max(ssimDist)

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
img = plt.imshow(movie[:, :, :, finalFrames[0]])
img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
    img.set_data(movie[:, :, :, finalFrames[0]])
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f) + ' from ' + str(finalFrames[f]))
    img.set_data(movie[:, :, :, finalFrames[f]])
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(finalFrames),interval=15,blit=True)

plt.show()

