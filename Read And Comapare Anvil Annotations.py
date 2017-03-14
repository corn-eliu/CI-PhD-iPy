# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import glob

from SemanticsDefinitionTabGUI import propagateLabels

# <codecell>

basePath = "/media/ilisescu/Data1/PhD/data/"
dataset = "toy_quick"
# dataset = "candle2/stabilized"

useNarrowedAnvil = True
useToyNoBG = False

if dataset == "toy_quick" :
    ## toy_quick
    tree = ET.parse(basePath+dataset+'/annotations.anvil')
    fps = 50
    classes = np.array(["none", "C", "D", "E", "F", "G", "A", "B", "C2"])
    numFrames = np.max(np.load(basePath+dataset+"/semantic_sequence-toy_quick.npy").item()['frame_locs'].keys())+1
    ## this makes sure that I take into account little timing differences between the sequence of frames and the video for anvil (because of slitghly different fps)
    frameNumDelta = 696/695.0 ## here the toy_quick dataset actually only has 696 frames as the last few ones are somehow repeated and they were not repeated when I made th evideo for anvil
    ## in secs for each added example
    propagationTimings = np.array([ 11,  15,  22,  32,  37,  42,  46,  51,  56,  60,
                                    65,  72,  75,  80,  83,  87,  92,  95,  97, 103,
                                   118, 122, 131, 134, 138, 144, 161, 164, 168, 175,
                                   180, 184, 189, 192, 195, 199, 202, 206, 209, 212,
                                   215, 218, 222, 225])-5 # remove the secs at beginning of video before I actually start labelling
    
    ## this is without the background frames
    if useToyNoBG :
        ## every 10 seconds, how much of the final video was annotated (i.e. accuracy)
        anvilAccuracy = np.array([  0,   0,  74, 101, 101, 120, 141, 159, 159, 173,
                                  186, 186, 201, 212, 227, 241, 241, 254, 269, 281,
                                  295, 319, 319, 333, 346, 372, 372, 372, 372, 383,
                                  395, 408, 408, 420, 431, 431, 449, 449, 449, 468,
                                  481, 491, 516, 527, 537, 537, 551, 562, 572, 583,
                                  597, 608, 608, 620, 635, 647, 658, 658, 670, 695])*frameNumDelta/float(696)
        anvilTimings = (np.arange(len(anvilAccuracy), dtype=int)+1)*10-6

        anvilNarrowed = np.array([[1, -2], [1, -3], [1, -2], [1, -1], [1, -3], [0, -2], [1, -2],
                                  [0, -3], [1, -1], [0, -2], [1, -2], [0, -1], [1, -2], [0, -1],
                                  [1, -3], [1, -2], [1, -1], [0, -2], [0, -2], [1, 0], [1, -1],
                                  [1, -1], [1, 0], [1, -1], [1, -1], [0, 0], [1, 0], [1, 0],
                                  [2, -1], [1, -1], [2, 0], [1, -2], [2, -2], [1, 0], [2, 0],
                                  [2, -2], [2, 0], [2, 0], [2, -1], [1, -2], [2, 0], [2, 0],
                                  [1, 0]], int)
    else :
        ## every 10 seconds, how much of the final video was annotated (i.e. accuracy)
        anvilAccuracy = np.array([  0,   0,  74, 101, 101, 120, 141, 159, 159, 173,
                                  186, 186, 201, 212, 227, 241, 241, 254, 269, 281,
                                  295, 319, 319, 333, 346, 372, 372, 372, 372, 383,
                                  395, 408, 408, 420, 431, 431, 449, 449, 449, 468,
                                  481, 491, 516, 527, 537, 537, 551, 562, 572, 583,
                                  597, 608, 608, 620, 635, 647, 658, 658, 670, 695])*frameNumDelta/float(696)
        
        extraAccuracies = np.array([0, 51, 51, 68, 68, 88, 100, 100, 110, 110, 110, 110, 110, 115, 120])*frameNumDelta/float(696)
        anvilAccuracy = np.concatenate([anvilAccuracy*(1.0-extraAccuracies[-1]), (1.0-extraAccuracies[-1])+extraAccuracies])
        
        anvilTimings = (np.arange(len(anvilAccuracy), dtype=int)+1)*10-6

        anvilNarrowed = np.array([[0, 0], [0, 0], [1, -2], [0, 0], [1, -3], [0, 0], [1, -2],
                                  [0, 0], [1, -1], [1, -3], [0, -2], [1, -2], [0, -3], [1, -1],
                                  [0, -2], [1, -2], [0, -1], [1, -2], [0, -1], [1, -3], [1, -2],
                                  [1, -1], [0, -2], [0, -2], [1, 0], [1, -1], [1, -1], [1, 0],
                                  [1, -1], [0, 0], [1, -1], [0, 0], [0, 0], [1, 0], [1, 0],
                                  [2, -1], [1, -1], [2, 0], [1, -2], [2, -2], [1, 0], [2, 0],
                                  [2, -2], [2, 0], [2, 0], [2, -1], [1, -2], [2, 0], [2, 0],
                                  [1, 0]], int)
        
    
elif dataset == "candle2/stabilized" :
    ## candle2/stabilized
    tree = ET.parse(basePath+dataset+'/annotations.anvil')
    fps = 60
    classes = np.array(["Rest", "Left", "Right"])
#     numFrames = len(glob.glob(basePath+dataset+"/frame-0*.png"))
    numFrames = np.max(np.load(basePath+dataset+"/semantic_sequence-candle.npy").item()['frame_locs'].keys())+1
    ## this makes sure that I take into account little timing differences between the sequence of frames and the video for anvil (because of slitghly different fps)
    frameNumDelta = float(numFrames)/3978.0
    
    ## in secs for each added example
    propagationTimings = np.array([6, 16, 26, 36, 46, 63, 83, 105, 132])-3 # remove the secs at beginning of video before I actually start labelling
    ## every 10 seconds, how much of the final video was annotated (i.e. accuracy)
    anvilAccuracy = np.array([   0,   32,  154,  234,  340,  410,  410,  410,  410,  511,
                               511,  511,  541,  541,  650,  790,  790,  790,  926,  994,
                              1070, 1107, 1204, 1209, 1228, 1275, 1425, 1460, 1606, 1606,
                              1681, 1818, 3978.0/2.0])*2.0*frameNumDelta/float(numFrames)
    anvilTimings = (np.arange(len(anvilAccuracy), dtype=int)+1)*10-3
    
    anvilNarrowed = np.array([[0, 0], [7, -7], [7, 0], [10, -5], [4, -6], [3, -10], [3, -2], [5, -4],
                              [2, -1], [4, -5], [3, -1], [3, -12], [1, -2], [3, -7], [5, -5], [6, -3],
                              [1, -1], [3, -5], [4, -8], [5, -7], [2, -1], [13, -10], [34, -4], [7, -13],
                              [5, 0]], int)*2

# <codecell>

root = tree.getroot()

anvilLabels = np.zeros([numFrames, len(classes)])
print anvilLabels.shape

for child in root :
    if child.tag == "body" :
        for gchild in child :
            if gchild.attrib["type"] == "primary" :
                for ggcIdx, ggchild in enumerate(gchild) :
                    if ggchild.tag == "el" :
                        start = int(np.round(float(ggchild.attrib["start"])*fps*frameNumDelta))
                        end = int(np.round(float(ggchild.attrib["end"])*fps*frameNumDelta))
                        if ggchild.find("attribute") == None and not useToyNoBG :
                            classIdx = 0
                        else :
                            classIdx = int(np.argwhere(classes == ggchild.find("attribute").text).flatten())
                        
                        if dataset == "candle2/stabilized" and not useNarrowedAnvil :
                            end -= 1
                        
                        if useNarrowedAnvil :
                            anvilLabels[start:end+1, classIdx] = -1.0
                            start += anvilNarrowed[ggcIdx, 0]
                            end += anvilNarrowed[ggcIdx, 1]
                            anvilLabels[start:end+1, classIdx] = 1.0
                        else :
                            anvilLabels[start:end+1, classIdx] = 1.0
                        print ggcIdx, start, end, classIdx

if dataset == "toy_quick" :
    print len(np.argwhere(np.sum(anvilLabels, axis=1) == 0).flatten())
    if useToyNoBG :
        anvilLabels[np.argwhere(np.sum(anvilLabels, axis=1) == 0).flatten(), 0] = 1.0
        tmp = np.argwhere(np.sum(anvilLabels, axis=1) == -1).flatten()
        anvilLabels[tmp, 1:] = 0.0
        anvilLabels[tmp, 0] = 1.0
        validLabels = np.argwhere(anvilLabels[:, 0] != 1.0).flatten()
    else :
        validLabels = np.argwhere(np.sum(anvilLabels, axis=1) > 0.0).flatten()
else :
    validLabels = np.argwhere(np.sum(anvilLabels, axis=1) != -1).flatten()

# <codecell>

if dataset == "candle2/stabilized" :
    propagatedLabels = np.load(basePath+dataset+"/semantic_sequence-candle.npy").item()['semantics_per_frame']
elif dataset == "toy_quick" :
    propagatedLabels = np.load(basePath+dataset+"/semantic_sequence-toy_quick.npy").item()['semantics_per_frame']
print anvilLabels
print propagatedLabels

# <codecell>

cols = mpl.cm.Set1(np.arange(0.0, 1.0 + 1.0/15.0, 1.0/15.0)[:len(classes)])
cols = [tuple(i) for i in cols[:, :-1]]

fig1 = figure()
numClasses = len(classes)
clrs = np.arange(0.0, 1.0+1.0/(numClasses-1), 1.0/(numClasses-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
stackplot(np.arange(len(anvilLabels[validLabels, :])), np.row_stack(tuple([i for i in anvilLabels[validLabels, :].T])), colors=cols)
ylim([0, 1])

fig2 = figure()
stackplot(np.arange(len(propagatedLabels[validLabels, :])), np.row_stack(tuple([i for i in propagatedLabels[validLabels, :].T])), colors=cols)
ylim([0, 1])

# <codecell>

if dataset == "toy_quick" and useToyNoBG :
    print np.sum(np.sqrt(np.sum((anvilLabels[validLabels, 1:]-propagatedLabels[validLabels, 1:])**2, axis=1)))/len(validLabels)
else :
    print np.sum(np.sqrt(np.sum((anvilLabels[validLabels, :]-propagatedLabels[validLabels, :])**2, axis=1)))/len(validLabels)
print numFrames
print propagatedLabels.shape, anvilLabels.shape
if dataset == "candle2/stabilized" :
    print np.load(basePath+dataset+"/semantic_sequence-candle.npy").item().keys()
elif dataset == "toy_quick" :
    print np.load(basePath+dataset+"/semantic_sequence-toy_quick.npy").item().keys()

# <codecell>

if dataset == "candle2/stabilized" :
    semanticSequence = np.load(basePath+dataset+"/semantic_sequence-candle.npy").item()
elif dataset == "toy_quick" :
    semanticSequence = np.load(basePath+dataset+"/semantic_sequence-toy_quick.npy").item()

labelledFrames = semanticSequence['labelled_frames']
numExtraFrames = semanticSequence['num_extra_frames']
print labelledFrames
print numExtraFrames
labelledFramesArray = np.array([item for sublist in labelledFrames for item in sublist])
numExtraFramesArray = np.array([item for sublist in numExtraFrames for item in sublist])

labelledFramesClassIdxArray = np.empty(0, int)
for classIdx in [np.repeat(i, len(labelledFrames[i])) for i in xrange(len(labelledFrames))] :
    labelledFramesClassIdxArray = np.concatenate([labelledFramesClassIdxArray, classIdx])
addedOrder = np.argsort(labelledFramesArray) ## change this if order is not time order

labelledFramesProgressive = []
numExtraFramesProgressive = []
for i in xrange(len(labelledFrames)) :
    labelledFramesProgressive.append([])
    numExtraFramesProgressive.append([])
    
propagationAccuracy = np.empty(0)

distMat = np.load(semanticSequence['sequence_precomputed_distance_matrix_location'])
for idx in addedOrder[0:] :
    labelledFramesProgressive[labelledFramesClassIdxArray[idx]].append(labelledFramesArray[idx])
    numExtraFramesProgressive[labelledFramesClassIdxArray[idx]].append(numExtraFramesArray[idx])
    print labelledFramesProgressive, numExtraFramesProgressive
    propagatedLabels = propagateLabels(distMat, labelledFramesProgressive, numExtraFramesProgressive, True, 6.0/100.0)#self.semanticsSigmaSpinBox.value()/100.0)
    if True and useNarrowedAnvil :
        propagatedLabelsThreshold = 0.0
        framesToThreshold = np.max(propagatedLabels, axis=1) >= propagatedLabelsThreshold
        framesToThresholdClasses = np.argmax(propagatedLabels, axis=1).flatten()
        propagatedLabels = np.zeros_like(propagatedLabels)
        for i, classIdx in enumerate(framesToThresholdClasses) :
            if framesToThreshold[i] :
                propagatedLabels[i, classIdx] = 1
            
    if dataset == "toy_quick" and useToyNoBG :
        propagationAccuracy = np.concatenate([propagationAccuracy, [1.0-np.sum(np.sqrt(np.sum((anvilLabels[validLabels, 1:]-propagatedLabels[validLabels, 1:])**2, axis=1)))/len(validLabels)]])
    else :
        propagationAccuracy = np.concatenate([propagationAccuracy, [1.0-np.sum(np.sqrt(np.sum((anvilLabels[validLabels, :]-propagatedLabels[validLabels, :])**2, axis=1)))/len(validLabels)]])
    print "ACCURACY:", propagationAccuracy[-1]

# <codecell>

figure("accuracy")
if dataset == "toy_quick" :
    plot(propagationTimings, propagationAccuracy, linestyle='-', color="#eb0000", linewidth=4)
    plot(anvilTimings, anvilAccuracy, linestyle='--', color="#eb0000", linewidth=3)
elif dataset == "candle2/stabilized" :
    plot(propagationTimings, propagationAccuracy, linestyle='-', color="#0000eb", linewidth=4)
    plot(anvilTimings, anvilAccuracy, linestyle='--', color="#0000eb", linewidth=3)
ylim([0, 1])

# <codecell>

xlabel("Time[secs]", fontsize=25)
ylabel("Precision", fontsize=25)

# <codecell>

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

# <codecell>

figure("bob", figsize=(15, 6), dpi=100)
if dataset == "toy_quick" :
    plot(propagationTimings, propagationAccuracy, linestyle='-', color="#eb0000", linewidth=4, label="US Toy")
    plot(anvilTimings, anvilAccuracy, linestyle='--', color="#eb0000", linewidth=3, label="ANVIL Toy")
elif dataset == "candle2/stabilized" :
    plot(propagationTimings, propagationAccuracy, linestyle='-', color="#0000eb", linewidth=4, label="US Candle")
    plot(anvilTimings, anvilAccuracy, linestyle='--', color="#0000eb", linewidth=3, label="ANVIL Candle")
ylim([0, 1])
xlabel("Time[secs]", fontsize=25)
ylabel("Precision", fontsize=25)
tight_layout()
legend(loc=4)

# <codecell>


