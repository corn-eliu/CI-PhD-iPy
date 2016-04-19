# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import glob
import os
import sys

DICT_SPRITE_NAME = 'sprite_name'
DICT_BBOXES = 'bboxes'
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SPRITE_IDX = 'sprite_idx' # stores the index in the self.trackedSprites array of the sprite used in the generated sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed

# <codecell>

# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
<<<<<<< HEAD
# dataSet = "theme_park_sunny/"
dataSet = "wave1/"
# preloadedSpritePatches = list(np.load(dataPath + dataSet + "preloadedSpritePatches.npy"))
=======
dataSet = "theme_park_sunny/"
preloadedSpritePatches = list(np.load(dataPath + dataSet + "preloadedSpritePatches.npy"))
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

# <codecell>

## load 
<<<<<<< HEAD
# trackedSprites = []
# for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
#     trackedSprites.append(np.load(sprite).item())
#     print trackedSprites[-1][DICT_SPRITE_NAME]
    
## load 
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "semantic_sequence*.npy")) :
    trackedSprites.append(np.load(sprite).item())
    print trackedSprites[-1]['semantic_sequence_name']
=======
trackedSprites = []
for sprite in np.sort(glob.glob(dataPath + dataSet + "sprite*.npy")) :
    trackedSprites.append(np.load(sprite).item())
    print trackedSprites[-1][DICT_SPRITE_NAME]
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

# <codecell>

## from https://github.com/fbessho/PyPoi/blob/master/pypoi/poissonblending.py
def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(img_target.shape[0] - offset[0], img_source.shape[0]),
        min(img_target.shape[1] - offset[1], img_source.shape[1]))
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(img_target.shape[0], img_source.shape[0] + offset[0]),
        min(img_target.shape[1], img_source.shape[1] + offset[1]))
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask[img_mask == 0] = False
    img_mask[img_mask != False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y, x]:
                index = x + y * region_size[1]
                A[index, index] = 4
                if index + 1 < np.prod(region_size):
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    A[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    A[index, index - region_size[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A, b, verb=False, tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return img_target

# <codecell>

imgSize = np.array(np.asarray(PIL.Image.open(dataPath+dataSet+"median.png")).shape[0:2])
<<<<<<< HEAD
# inputFolderSuffix = "-masked"#-blended/"
inputFolderSuffix = "-maskedFlow"#-blended/"
for spriteIdx in arange(len(trackedSprites))[-1:] :
    sortedKeys = np.sort(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())
#     spriteName = trackedSprites[spriteIdx][DICT_SPRITE_NAME]
    spriteName = trackedSprites[spriteIdx]['semantic_sequence_name']
    
    if not os.path.isdir(dataPath+dataSet+spriteName+inputFolderSuffix+"-blended/"):
        os.makedirs(dataPath+dataSet+spriteName+inputFolderSuffix+"-blended/")
=======
for spriteIdx in arange(len(trackedSprites)) :
    sortedKeys = np.sort(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS].keys())
    spriteName = trackedSprites[spriteIdx][DICT_SPRITE_NAME]
    
    if not os.path.isdir(dataPath+dataSet+spriteName+"-masked-blended/"):
        os.makedirs(dataPath+dataSet+spriteName+"-masked-blended/")
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        
    for frameIdx in arange(len(sortedKeys)) :
        frameName = trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frameIdx]].split(os.sep)[-1]
        
        
<<<<<<< HEAD
        im = np.array(PIL.Image.open(dataPath+dataSet+spriteName+inputFolderSuffix+"/"+frameName))
        
        visiblePixelsGlobalIndices = np.argwhere(im[:, :, -1] != 0)
        topLeftPos = np.min(visiblePixelsGlobalIndices, axis=0)
        patchSize = np.max(visiblePixelsGlobalIndices, axis=0) - topLeftPos + 1
#         topLeftPos = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['top_left_pos'])
#         patchSize = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['patch_size'])
#         visiblePixelsGlobalIndices = preloadedSpritePatches[spriteIdx][frameIdx]['visible_indices']+topLeftPos
=======
        topLeftPos = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['top_left_pos'])
        patchSize = np.copy(preloadedSpritePatches[spriteIdx][frameIdx]['patch_size'])
        visiblePixelsGlobalIndices = preloadedSpritePatches[spriteIdx][frameIdx]['visible_indices']+topLeftPos
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        
        ## when the mask touches the border of the patch there's some weird white halos going on so I enlarge the patch slightly
        ## not sure what happens when the patch goes outside of the bounds of the original image...
        topLeftPos -= 1
        patchSize += 2
        ## make sure we're within bounds
        topLeftPos[np.argwhere(topLeftPos < 0)] = 0
        patchSize[(topLeftPos+patchSize) > imgSize] += (imgSize-(topLeftPos+patchSize))[(topLeftPos+patchSize) > imgSize]
        
        
<<<<<<< HEAD
        img_target = np.asarray(PIL.Image.open(dataPath+dataSet+"median.png"))[:, :, 0:3]
        img_target.flags.writeable = True
        
        img_mask = np.asarray(PIL.Image.open(dataPath+dataSet+spriteName+inputFolderSuffix+"/"+frameName))[topLeftPos[0]:topLeftPos[0]+patchSize[0], 
                                                                                                           topLeftPos[1]:topLeftPos[1]+patchSize[1], -1]
=======
        img_target = np.asarray(PIL.Image.open(dataPath+dataSet+"median.png"))
        img_target.flags.writeable = True
        
        img_mask = np.asarray(PIL.Image.open(dataPath+dataSet+spriteName+"-masked/"+frameName))[topLeftPos[0]:topLeftPos[0]+patchSize[0], topLeftPos[1]:topLeftPos[1]+patchSize[1], -1]
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        img_mask.flags.writeable = True
        ## make sure that borders of mask are assigned to bg
        img_mask[0, :] = 0; img_mask[-1, :] = 0; img_mask[:, 0] = 0; img_mask[:, -1] = 0
        
<<<<<<< HEAD
        img_source = np.asarray(PIL.Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frameIdx]]))[topLeftPos[0]:topLeftPos[0]+patchSize[0], 
                                                                                                                        topLeftPos[1]:topLeftPos[1]+patchSize[1], :]
        
#         sourceImg = np.asarray(PIL.Image.open(dataPath+dataSet+spriteName+inputFolderSuffix+"/"+frameName))[topLeftPos[0]:topLeftPos[0]+patchSize[0], 
#                                                                                                             topLeftPos[1]:topLeftPos[1]+patchSize[1], :-1]
#         mask = np.copy(img_mask.reshape((patchSize[0], patchSize[1], 1)))/255.0
        
#         img_source = np.array(sourceImg*mask + np.asarray(PIL.Image.open(dataPath+dataSet+"median.png"))[topLeftPos[0]:topLeftPos[0]+patchSize[0], 
#                                                                                                          topLeftPos[1]:topLeftPos[1]+patchSize[1], :]*(1.0-mask), dtype=uint8)
        
            
=======
        img_source = np.asarray(PIL.Image.open(trackedSprites[spriteIdx][DICT_FRAMES_LOCATIONS][sortedKeys[frameIdx]]))[topLeftPos[0]:topLeftPos[0]+patchSize[0], topLeftPos[1]:topLeftPos[1]+patchSize[1], :]
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        img_source.flags.writeable = True
        
        
        img_ret = blend(img_target, img_source, img_mask, offset=(topLeftPos[0], topLeftPos[1]))
        
        
        maskedFinal = np.zeros((img_target.shape[0], img_target.shape[1], 4), dtype=np.uint8)
<<<<<<< HEAD
        maskedFinal[visiblePixelsGlobalIndices[:, 0], visiblePixelsGlobalIndices[:, 1], :-1] = img_ret[visiblePixelsGlobalIndices[:, 0], 
                                                                                                       visiblePixelsGlobalIndices[:, 1], :]
        maskedFinal[visiblePixelsGlobalIndices[:, 0], visiblePixelsGlobalIndices[:, 1], -1] = 255
        
        PIL.Image.fromarray(np.uint8(maskedFinal)).save(dataPath+dataSet+spriteName+inputFolderSuffix+"-blended/"+frameName)
#         figure(); imshow(maskedFinal)
=======
        maskedFinal[visiblePixelsGlobalIndices[:, 0], visiblePixelsGlobalIndices[:, 1], :-1] = img_ret[visiblePixelsGlobalIndices[:, 0], visiblePixelsGlobalIndices[:, 1], :]
        maskedFinal[visiblePixelsGlobalIndices[:, 0], visiblePixelsGlobalIndices[:, 1], -1] = 255
        
        PIL.Image.fromarray(np.uint8(maskedFinal)).save(dataPath+dataSet+spriteName+"-masked-blended/"+frameName)
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        del img_mask
        del img_source
        del img_target
        del maskedFinal
        del img_ret
<<<<<<< HEAD
#         del sourceImg
#         del mask
=======
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        
        sys.stdout.write('\r' + "Processed image " + np.string_(frameIdx) + " (" + np.string_(len(sortedKeys)) + ")")
        sys.stdout.flush()
    print
<<<<<<< HEAD
    print "done with sprite", trackedSprites[spriteIdx]['semantic_sequence_name']

# <codecell>

# figure(); imshow(img_target)
print np.asarray(PIL.Image.open(dataPath+dataSet+"median.png")).shape

# <codecell>

del img_mask
del img_source
del img_target
del maskedFinal
del img_ret
del sourceImg
del mask
=======
    print "done with sprite", trackedSprites[spriteIdx][DICT_SPRITE_NAME]
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

