# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
import numpy as np

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

