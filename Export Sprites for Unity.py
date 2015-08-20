# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import scipy.sparse
import PIL.Image
import glob
import os
import sys

DICT_SPRITE_NAME = 'sprite_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SPRITE_IDX = 'sprite_idx' # stores the index in the self.trackedSprites array of the sprite used in the generated sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed

# <codecell>

dataPath = "/home/ilisescu/PhD/data/"
dataSet = "havana/"
# dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
preloadedSpritePatches = list(np.load(dataPath + dataSet + "preloadedSpritePatches.npy"))

# <codecell>

## load 
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
    print trackedSprites[-1][DICT_SPRITE_NAME], DICT_FOOTPRINTS in trackedSprites[-1].keys()

# <codecell>

imgSize = np.array(np.asarray(PIL.Image.open(dataPath+dataSet+"median.png")).shape[0:2])
# saveLoc = dataPath+dataSet+"exportedToUnity/"
saveLoc = dataPath+dataSet+"exportedToUnity_withFootprints/"

if not os.path.isdir(saveLoc):
    os.makedirs(saveLoc)
    
for spriteIdx in arange(len(trackedSprites)) :
    sortedKeys = np.sort(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())
    spriteName = trackedSprites[spriteIdx][DICT_SPRITE_NAME]
    
    if not os.path.isdir(saveLoc+spriteName+"/"):
        os.makedirs(saveLoc+spriteName+"/")
        
    spriteData = []
    
    for frameIdx in arange(len(sortedKeys)) :
        frameName = trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frameIdx]].split(os.sep)[-1]
        
        
        topLeftPos = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['top_left_pos'])
        patchSize = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['patch_size'])
        visiblePixelsGlobalIndices = preloadedSpritePatches[spriteIdx][frameIdx]['visible_indices']+topLeftPos
        
        ## when the mask touches the border of the patch there's some weird white halos going on so I enlarge the patch slightly
        ## not sure what happens when the patch goes outside of the bounds of the original image...
        topLeftPos -= 1
        patchSize += 2
        ## make sure we're within bounds
        topLeftPos[np.argwhere(topLeftPos < 0)] = 0
        patchSize[(topLeftPos+patchSize) > imgSize] += (imgSize-(topLeftPos+patchSize))[(topLeftPos+patchSize) > imgSize]
        
        if True :
            spriteData.append(np.ndarray.flatten(np.vstack((topLeftPos[::-1], trackedSprites[spriteIdx][DICT_FOOTPRINTS][sortedKeys[frameIdx]]))))
        else :
            spriteData.append(np.ndarray.flatten(np.vstack((topLeftPos[::-1], trackedSprites[spriteIdx][DICT_BBOXES][sortedKeys[frameIdx]]))))
        
#         print topLeftPos, patchSize
        patch = np.array(PIL.Image.open(dataPath+dataSet+spriteName+"-masked-blended/"+frameName))[topLeftPos[0]:topLeftPos[0]+patchSize[0], topLeftPos[1]:topLeftPos[1]+patchSize[1]]
#         figure(); imshow(patch)
#         print trackedSprites[spriteIdx][DICT_BBOXES][sortedKeys[frameIdx]]
        
        ## save patch
        PIL.Image.fromarray(np.uint8(patch)).save(saveLoc+spriteName+"/frame-{:05d}.png".format(frameIdx+1))
        
        sys.stdout.write('\r' + "Processed image " + np.string_(frameIdx) + " (" + np.string_(len(sortedKeys)) + ")")
        sys.stdout.flush()
        
    
    numpy.savetxt(saveLoc+spriteName+".csv", numpy.asarray(spriteData), delimiter=",")
    print
    print "done with sprite", trackedSprites[spriteIdx][DICT_SPRITE_NAME]

# <codecell>

print "{:05d}.png".format(2)

# <codecell>

PIL.Image.fromarray(np.uint8(patch)).save("tralala.png")

# <codecell>

spriteData = []
spriteData.append(np.ndarray.flatten(np.vstack((topLeftPos, trackedSprites[spriteIdx][DICT_BBOXES][sortedKeys[frameIdx]]))))
print spriteData

numpy.savetxt("foo.csv", numpy.asarray(spriteData), delimiter=",")

