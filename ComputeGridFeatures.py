# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
import numpy as np
import cv2
import time
import sys
import os

# <codecell>

def stencil2D(blocksPerWidth, blocksPerHeight, imageSize):
    """Given block sizes and image size, it returns indices representing each block in 3D."""
    stencils = []
    bRows = imageSize[0]/blocksPerHeight
    bCols = imageSize[1]/blocksPerWidth
    for r in xrange(0, blocksPerHeight) :
        for c in xrange(0, blocksPerWidth) :
            stencil = np.zeros(imageSize[0:-1], dtype=int)
            stencil[r*bRows:r*bRows+bRows, c*bCols:c*bCols+bCols] = np.ones((bRows, bCols))
            stencils.append(list(np.argwhere(stencil==1).T))
            
    return stencils

def stencil3D(blocksPerWidth, blocksPerHeight, imageSize) :
    """Given block sizes and image size, it returns indices representing each block in 3D."""
    
    stencils = []
    bRows = imageSize[0]/blocksPerHeight
    bCols = imageSize[1]/blocksPerWidth
    for r in xrange(0, blocksPerHeight) :
        for c in xrange(0, blocksPerWidth) :
            stencil = np.zeros(imageSize, dtype=int)
            stencil[r*bRows:r*bRows+bRows, c*bCols:c*bCols+bCols] = np.ones((bRows, bCols, imageSize[-1]))
            stencils.append(list(np.argwhere(stencil==1).T))
    
    return stencils

# <codecell>

def histFgFeatures(stencils, subDivisions, frames, mattes) :
    """Computes a feature vector as number of foreground pixels per subsection as defined in stencils.
       It assumes 2D stencils and they contain indices within the size of a frame."""
    st = time.time()
    
    numFrames = len(frames)
    features = np.zeros([numFrames, subDivisions])
    for i in xrange(0, numFrames) :
        
        t = time.time()
        
        ##load frame
        img = np.array(cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB))/255.0
        alpha = np.zeros(img.shape[0:-1])
        if os.path.isfile(mattes[i]) :
            alpha = np.array(cv2.cvtColor(cv2.imread(mattes[i]), cv2.COLOR_BGR2GRAY))/255.0
            img *= np.repeat(np.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1)), 3, axis=-1)
    
        ## use stencils to divide the image into blocks and count number of foreground pixels
        for s in xrange(0, len(stencils)) :
    #         index = s + idx*len(stencils)
            features[i, s] = len(np.argwhere(alpha[stencils[s]] != 0))
        sys.stdout.write('\r' + "Computed features for frame " + np.string_(i) + " of " + np.string_(numFrames) + " in " + np.string_(time.time() - t))
        sys.stdout.flush()
        
    print
    print "finished in", time.time() - st
    
    ## normalize
#     features /= np.repeat(np.reshape(np.linalg.norm(features, axis=-1), (numFrames, 1)), subDivisions, axis=-1)
    return features

