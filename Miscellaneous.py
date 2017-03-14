# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

from PIL import Image

import numpy as np

import scipy as sp
import scipy.io as sio
import cv2
import cv
import glob
import time
import gc
import re

import sys
import os

import GraphWithValues as gwv

# dataFolder = "/home/ilisescu/PhD/data/"
dataFolder = "/media/ilisescu/Data1/PhD/data/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
dataSet = "wave1/"

dataSet = "wave2/"
dataSet = "wave3/"
dataSet = "windows/"
dataSet = "digger/"

DICT_SEQUENCE_NAME = 'semantic_sequence_name'
DICT_BBOXES = 'bboxes'
DICT_FOOTPRINTS = 'footprints' ## same as bboxes but it indicates the footprint of the sprite on the ground plane
DICT_BBOX_ROTATIONS = 'bbox_rotations'
DICT_BBOX_CENTERS = 'bbox_centers'
DICT_FRAMES_LOCATIONS = 'frame_locs'
DICT_MASK_LOCATION = 'frame_masks_location'
DICT_SEQUENCE_FRAMES = 'sequence_frames'
DICT_SEQUENCE_IDX = 'semantic_sequence_idx' # index of the instantiated sem sequence in the list of all used sem sequences for a synthesised sequence
DICT_DESIRED_SEMANTICS = 'desired_semantics' # stores what the desired semantics are for a certain sprite 
#(I could index them by the frame when the toggle happened instead of using the below but maybe ordering is important and I would lose that using a dict)
DICT_FRAME_SEMANTIC_TOGGLE = 'frame_semantic_toggle'# stores the frame index in the generated sequence when the desired semantics have changed
DICT_ICON_TOP_LEFT = "icon_top_left"
DICT_ICON_FRAME_KEY = "icon_frame_key"
DICT_ICON_SIZE = "icon_size"
DICT_REPRESENTATIVE_COLOR = 'representative_color'
DICT_OFFSET = "instance_offset"
DICT_SCALE = "instance_scale"
DICT_FRAME_SEMANTICS = "semantics_per_frame"
DICT_USED_SEQUENCES = "used_semantic_sequences"
DICT_SEQUENCE_INSTANCES = "sequence_instances"
DICT_SEQUENCE_BG = "sequence_background_image"
DICT_SEQUENCE_LOCATION = "sequence_location"
DICT_PATCHES_LOCATION = "sequence_preloaded_patches_location"
DICT_TRANSITION_COSTS_LOCATION = "sequence_precomputed_transition_costs_location"

# <headingcell level=2>

# RENDER BBOXES

# <codecell>

import pyglet
sys.path.append('/media/ilisescu/Data1/PhD/data/drumming2/meow.wav')

# <codecell>

pyglet.resource.path = ['/media/ilisescu/Data1/PhD/data/drumming2']
pyglet.resource.reindex()
music = pyglet.resource.media('meow.wav')
music.play()
# pyglet.app.run()

# <codecell>

paths = np.sort(glob.glob("E:/PhD/data/digger/semantic_sequence-*.npy"))
semanticSequences = []
for sequence in paths :
    semanticSequences.append(np.load(sequence).item())
    print semanticSequences[-1][DICT_SEQUENCE_NAME]



for frameKey in np.sort(semanticSequences[0][DICT_BBOXES].keys())[0:] :
    
    overlayImg = QtGui.QImage(QtCore.QSize(1280, 720), QtGui.QImage.Format_ARGB32)
    overlayImg.fill(QtGui.QColor.fromRgb(255, 255, 255, 0))
    
    painter = QtGui.QPainter(overlayImg)
    painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
    
    for seqIdx in xrange(len(semanticSequences)) :
        offset = np.array([0.0, 0.0])
        scale = np.array([1.0, 1.0])
        sequence = semanticSequences[seqIdx]
        if frameKey in sequence[DICT_BBOXES].keys() :
#         frameKey = np.sort(sequence[DICT_BBOXES].keys())[0]
    #     frameKey = np.sort(sequence[DICT_BBOXES].keys())[450]
    #     if seqIdx == 7 :
    #         frameKey = np.sort(sequence[DICT_BBOXES].keys())[-70]
    #         print sequence[DICT_BBOXES][frameKey], scale, offset    

            scaleTransf = np.array([[scale[0], 0.0], [0.0, scale[1]]])
            offsetTransf = np.array([[offset[0]], [offset[1]]])

            if offset[0] == 0 and offset[1] == 0 :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 3, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            else :
                painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(0, 0, 255, 255), 1, 
                                          QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

            bbox = sequence[DICT_BBOXES][frameKey]
            transformedBBox = (np.dot(scaleTransf, bbox.T) + offsetTransf)

            x, y = transformedBBox[:, 0]
            width, height = transformedBBox[:, 2] - transformedBBox[:, 0]
        #     painter.drawRoundedRect(x, y, width, height, 3, 3)

            for p1, p2 in zip(np.mod(arange(4), 4), np.mod(arange(1, 5), 4)) :
                painter.drawLine(QtCore.QPointF(transformedBBox[0, p1], transformedBBox[1, p1]), QtCore.QPointF(transformedBBox[0, p2], transformedBBox[1, p2]))


            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(255, 255, 0, 255), 1, 
                                      QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        #     painter.drawText(transformedBBox[0, 0]+5, transformedBBox[1, 0], 
        #                      transformedBBox[0, 2]-transformedBBox[0, 0]-5, 20, QtCore.Qt.AlignLeft, np.string_(seqIdx))

    print overlayImg.save("E:/PhD/data/digger/bboxes-{0:05}.png".format(frameKey+1)),
    painter.end()
    del painter

# <codecell>

tmp = Image.fromarray(np.zeros((720, 1280, 4), np.uint8))
for sIdx in xrange(len(semanticSequences)) :
    for frameKey in np.sort(semanticSequences[0][DICT_BBOXES].keys())[0:] :
        if not os.path.isfile("E:/PhD/data/digger/"+semanticSequences[sIdx][DICT_SEQUENCE_NAME] + "-maskedFlow/scribble-frame-{0:05}.png".format(frameKey+1)) :
            tmp.save("E:/PhD/data/digger/"+semanticSequences[sIdx][DICT_SEQUENCE_NAME] + "-maskedFlow/scribble-frame-{0:05}.png".format(frameKey+1))

# <codecell>

overlayImg.save("C:/Users/ilisescu/Desktop/bboxes.png")

# <headingcell level=2>

# MAKE IMAGE FROM DESIRED SEMANTICS

# <codecell>

synthSeq = np.load(dataPath+"synthesisedSequences/waveFullBusier/synthesised_sequence.npy").item()
bgImage = np.array(Image.open(dataPath+"synthesisedSequences/waveFullBusier/median.png"))[:, :, 0:3]
usedSequences = synthSeq[DICT_USED_SEQUENCES]
semanticSequences = []
bboxes = [] ## first row = top left (x, y), second row = (width, height)
for sequence in usedSequences :
    semanticSequences.append(np.load(sequence).item())
    print semanticSequences[-1][DICT_SEQUENCE_NAME]
    bboxes.append(np.concatenate((np.min(np.min(np.array(semanticSequences[-1][DICT_BBOXES].values()), axis=1), axis=0),
                                  np.max(np.max(np.array(semanticSequences[-1][DICT_BBOXES].values()), axis=1), axis=0))).reshape((2, 2)))
    bboxes[-1][1, :] = bboxes[-1][1, :]-bboxes[-1][0, :]
#     print bboxes[-1]


bboxToUse = np.zeros(len(synthSeq[DICT_SEQUENCE_INSTANCES]))
maxFrames =  327
for frame in arange(maxFrames)[0:] :
#     for i, instance in enumerate(synthSeq[DICT_SEQUENCE_INSTANCES]) :
    img = np.zeros((bgImage.shape[0], bgImage.shape[1], 4), np.uint8)
    for i, sIdx in enumerate([0, 1, 2, 3, 14, 13, 4, 5, 6, 7, 8, 9, 10, 11, 12]) :
        instance = synthSeq[DICT_SEQUENCE_INSTANCES][sIdx]
#         print len(instance[DICT_DESIRED_SEMANTICS])
#         print len(instance[DICT_SEQUENCE_FRAMES])

        desiredSemantics = instance[DICT_DESIRED_SEMANTICS][frame, :]
#         print desiredSemantics
#         print instance[DICT_OFFSET]
        offset = instance[DICT_OFFSET]
        scale = instance[DICT_SCALE]
        seqIdx = instance[DICT_SEQUENCE_IDX]
        bbox = bboxes[seqIdx]
        x, y = bbox[0, :].astype(int)
        w, h = bbox[1, :].astype(int)
        
#         if desiredSemantics[0] > 0.5 :            
#             imgToUse = cv2.resize(sitImg, (w, h))
#         else :
#             imgToUse = cv2.resize(upImg, (w, h))
        imgToUse = (cv2.resize(sitImg, (w, h))*desiredSemantics[0] + cv2.resize(upImg, (w, h))*(1.0-desiredSemantics[0])).astype(np.uint8)
        
        img[y:y+h, x:x+w, :] = (imgToUse[:, :, :]*(imgToUse[:, :, 3].reshape((h, w, 1))/255.0) +
                                  img[y:y+h, x:x+w, :]*(1.0-(imgToUse[:, :, 3].reshape((h, w, 1))/255.0))).astype(np.uint8)
    
    img = (img[:, :, 0:3]*(img[:, :, 3].reshape((img.shape[0], img.shape[1], 1))/255.0)+
           bgImage*(1.0-(img[:, :, 3].reshape((img.shape[0], img.shape[1], 1))/255.0))).astype(np.uint8)
    cv2.imwrite(dataPath+"synthesisedSequences/waveFullBusier/input-{0:05}.png".format(frame+1), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
#         fig1 = figure()
#         clrs = np.arange(0.0, 1.0+1.0/(len(desiredSemantics.T)-1), 1.0/(len(desiredSemantics.T)-1)).astype(np.string_) #['r', 'g', 'b', 'm', 'c', 'y', 'k', 'w']
#         stackplot(np.arange(len(desiredSemantics)), np.row_stack(tuple([i for i in desiredSemantics.T])), colors=clrs)

# <codecell>

sitImg = np.array(Image.open("/home/ilisescu/PhD/person_sit.png"))
upImg = np.array(Image.open("/home/ilisescu/PhD/person_up.png"))
# sitImg = cv2.resize(sitImg, (100, 50))

figure(); imshow(sitImg)
figure(); imshow(upImg)

# <codecell>

figure(); imshow(img[:, :, :3])

# <headingcell level=2>

# MAKE IMAGES FROM TETRIS INPUT

# <codecell>

synthSeq = np.load(synthSeqLocation).item()
sessionName = "6x9_sess2"

outputImgPath = "/media/ilisescu/Data1/PhD/data/windows/tetris/"+sessionName
if not os.path.isdir(outputImgPath) :
    os.mkdir(outputImgPath)
    
with open("/media/ilisescu/Data1/PhD/data/windows/tetris/"+sessionName+".txt") as f:
    lines = f.readlines()
    gridSize = np.array(lines[0].split(",")[1:3], int)[::-1] ## (rows, cols)
    lines = np.concatenate((["D,121231231233,"+"".join(np.zeros(np.prod(gridSize), int).astype(np.string_))+"\n"], lines[1:]))
    
    cellSize = 100
    spacing = 10
    
    previousFrameInstructions = np.ones(gridSize)
    previousFrame = np.ones(np.concatenate(((gridSize*cellSize) + ((gridSize+1)*spacing), [3])), np.uint8)*15
    
    PIECE_I = 0
    PIECE_J = 1
    PIECE_L = 2
    PIECE_O = 3
    PIECE_S = 4
    PIECE_T = 5
    PIECE_Z = 6
    
    
    pieces = np.array([[0, 7, PIECE_Z],
                       [8, 21, PIECE_T],
                       [22, 31, PIECE_I],
                       [32, 40, PIECE_Z],
                       [41, 55, PIECE_T],
                       [56, 65, PIECE_S],
                       [66, 73, PIECE_I],
                       [74, 79, PIECE_O],
                       [80, 92, PIECE_J],
                       [93, 105, PIECE_T],
                       [106, 113, PIECE_O],
                       [114, 122, PIECE_L],
                       [123, 126, PIECE_O],
                       [127, 130, PIECE_L]])
    currentPiece = 0
    
    for frame, line in enumerate(lines[0:]) :
        
        if pieces[currentPiece, 1] < frame :
            currentPiece += 1
        
#         print np.array(list(line.split(",")[-1])[:-1], int).reshape(gridSize, order='C')
        sys.stdout.flush()
        instructions = np.array(list(line.split(",")[-1])[:-1], int)
#         print instructions
        
        outputImg = np.copy(previousFrame)
        
        for instance, instruction in enumerate(instructions[0:]) :
            i, j = [int(instance/gridSize[1]), np.mod(instance, gridSize[1])]
            
#             if instruction == 0 :
#                 color = np.array([255, 255, 255])
#             else :
#                 color = np.array([0, 84, 194])
            if instruction != previousFrameInstructions[i, j] :
                if instruction == 0 :
                    color = np.array([245, 245, 245])
                else :
                    if pieces[currentPiece, 2] == PIECE_I :
                        color = np.array([255, 0, 0], np.uint8)
                    elif pieces[currentPiece, 2] == PIECE_J :
                        color = np.array([146, 146, 146], np.uint8)
                    elif pieces[currentPiece, 2] == PIECE_L :
                        color = np.array([219, 20, 203], np.uint8)
                    elif pieces[currentPiece, 2] == PIECE_O :
                        color = np.array([10, 45, 201], np.uint8)
                    elif pieces[currentPiece, 2] == PIECE_S :
                        color = np.array([31, 181, 24], np.uint8)
                    elif pieces[currentPiece, 2] == PIECE_T :
                        color = np.array([107, 44, 12], np.uint8)
                    else :
                        color = np.array([42, 171, 250], np.uint8)

                outputImg[i*cellSize + ((i+1)*spacing):(i+1)*cellSize + ((i+1)*spacing),
                          j*cellSize + ((j+1)*spacing):(j+1)*cellSize + ((j+1)*spacing)] = color
                
#         contrastEnhancer = ImageEnhance.Brightness(Image.fromarray(outputImg))
#         colorEnhancer = ImageEnhance.Contrast(contrastEnhancer.enhance(1.2))
#         colorEnhancer.enhance(2.0).save(outputImgPath+"/frame-{0:05}.png".format(frame+1))
        cv2.imwrite(outputImgPath+"/frame-{0:05}.png".format(frame+1), cv2.cvtColor(outputImg, cv2.COLOR_RGB2BGR))
#         Image.fromarray(outputImg).save(outputImgPath+"/frame-{0:05}.png".format(frame+1))
        previousFrameInstructions = instructions.reshape(gridSize, order='C')
        previousFrame = np.copy(outputImg)

# figure(); imshow(outputImg)

# <codecell>

figure(); imshow(outputImg)

# <codecell>

pilImg = Image.fromarray(outputImg)
pilImg.save()

# <codecell>

from PIL import ImageEnhance

# <codecell>

enhancer = ImageEnhance.Contrast(Image.fromarray(outputImg))
enhancer.enhance(1.8).show()

# <headingcell level=2>

# RESIZE IMAGES AND CHANGE THE BACKGROUND

# <codecell>

# basePath = "/media/ilisescu/Data1/PhD/demos/sprite_originalspeed/"
# basePath = "/home/ilisescu/PhD/data/synthesisedSequences/waveFull/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/waveFullBusier/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/theme_park_mixedCompatibility/no_people/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/theme_park_mixedCompatibility/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/tetris/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/digger/with_sem_compat/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/multipleCandles/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/digger/"
# basePath = "/home/ilisescu/PhD/data/synthesisedSequences/street/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/plane_departures/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/flowers/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/street_complex/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave_by_numbers_fattestbar/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave_by_numbers_top_bottom/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave_by_numbers_interlaced/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/drumming_new/"
basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/drumming_laggy/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave_by_numbers_double_trouble/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/super_mario_full/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/super_mario_planes_latest/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/USER STUDIES SEQUENCES/aron/wave_user_study_task/"
bgImage = np.array(Image.open(basePath+"bgImage.png"))
frameLocs = np.sort(glob.glob(basePath + "frame-*.png"))

if not os.path.isdir(basePath + "on_bg/") :
    os.mkdir(basePath + "on_bg/")
    
for loc in frameLocs[0:] :
    currentFrame = np.array(Image.open(loc))
    
#     if currentFrame.shape[0] >= bgImage.shape[0] :
#         currentFrame = currentFrame[currentFrame.shape[0]-bgImage.shape[0]:, :, :]
#     else :
#         currentFrame = np.concatenate((np.zeros((bgImage.shape[0]-currentFrame.shape[0],
#                                                  currentFrame.shape[1],
#                                                  currentFrame.shape[2]), np.uint8),
#                                        currentFrame), axis=0)
        
#     if currentFrame.shape[1] >= bgImage.shape[1] :
#         delta = (currentFrame.shape[1]-bgImage.shape[1])/2
#         currentFrame = currentFrame[:, delta:-delta, :]
#     else :
#         delta = (bgImage.shape[1]-currentFrame.shape[1])/2
#         currentFrame = np.concatenate((np.zeros((currentFrame.shape[0], delta, currentFrame.shape[2]), np.uint8),
#                                        currentFrame,
#                                        np.zeros((currentFrame.shape[0], delta, currentFrame.shape[2]), np.uint8)), axis=1)
    
    alphas = currentFrame[:, :, -1].reshape((currentFrame.shape[0], currentFrame.shape[1], 1))
    finalFrame = (currentFrame[:, :, 0:3]*(alphas/255.0) + bgImage[:, :, 0:3]*((255-alphas)/255.0)).astype(np.uint8)

    
    Image.fromarray(finalFrame).save(basePath + "on_bg/"+loc.split("/")[-1])

# <codecell>

# basePath = "/media/ilisescu/Data1/PhD/demos/sprite_originalspeed/"
# basePath = "/home/ilisescu/PhD/data/synthesisedSequences/waveFull/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/waveFullBusier/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/theme_park_mixedCompatibility/no_people/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/theme_park_mixedCompatibility/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/tetris/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/digger/with_sem_compat/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/multipleCandles/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/digger/"
# basePath = "/home/ilisescu/PhD/data/synthesisedSequences/street/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/plane_departures/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/flowers/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/street_complex/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave_by_numbers_fattestbar/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/super_mario_full/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/super_mario_planes_latest/"
basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/USER STUDIES SEQUENCES/moos/wave_user_study_task/"
bgImage = np.array(Image.open(basePath+"bgImage.png"))
frameLocs = np.sort(glob.glob(basePath + "frame-*.png"))

if not os.path.isdir(basePath + "on_bg/") :
    os.mkdir(basePath + "on_bg/")
    
for loc in frameLocs[0:] :
    currentFrame = np.array(Image.open(loc))
    
#     if currentFrame.shape[0] >= bgImage.shape[0] :
#         currentFrame = currentFrame[currentFrame.shape[0]-bgImage.shape[0]:, :, :]
#     else :
#         currentFrame = np.concatenate((np.zeros((bgImage.shape[0]-currentFrame.shape[0],
#                                                  currentFrame.shape[1],
#                                                  currentFrame.shape[2]), np.uint8),
#                                        currentFrame), axis=0)
        
#     if currentFrame.shape[1] >= bgImage.shape[1] :
#         delta = (currentFrame.shape[1]-bgImage.shape[1])/2
#         currentFrame = currentFrame[:, delta:-delta, :]
#     else :
#         delta = (bgImage.shape[1]-currentFrame.shape[1])/2
#         currentFrame = np.concatenate((np.zeros((currentFrame.shape[0], delta, currentFrame.shape[2]), np.uint8),
#                                        currentFrame,
#                                        np.zeros((currentFrame.shape[0], delta, currentFrame.shape[2]), np.uint8)), axis=1)
    
    alphas = currentFrame[:, :, -1].reshape((currentFrame.shape[0], currentFrame.shape[1], 1))
    finalFrame = (currentFrame[:, :, 0:3]*(alphas/255.0) + bgImage[:, :, 0:3]*((255-alphas)/255.0)).astype(np.uint8)

    
    Image.fromarray(finalFrame).save(basePath + "on_bg/"+loc.split("/")[-1])

# <codecell>

# basePath = "/media/ilisescu/Data1/PhD/demos/sprite_originalspeed/"
# basePath = "/home/ilisescu/PhD/data/synthesisedSequences/waveFull/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/waveFullBusier/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/theme_park_mixedCompatibility/no_people/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/theme_park_mixedCompatibility/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/tetris/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/digger/with_sem_compat/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/multipleCandles/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/digger/"
# basePath = "/home/ilisescu/PhD/data/synthesisedSequences/street/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/plane_departures/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/flowers/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/street_complex/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/wave_by_numbers_fattestbar/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/super_mario_full/"
# basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/super_mario_planes_latest/"
basePath = "/media/ilisescu/Data1/PhD/data/synthesisedSequences/USER STUDIES SEQUENCES/clement/wave_user_study_task/"
bgImage = np.array(Image.open(basePath+"bgImage.png"))
frameLocs = np.sort(glob.glob(basePath + "frame-*.png"))

if not os.path.isdir(basePath + "on_bg/") :
    os.mkdir(basePath + "on_bg/")
    
for loc in frameLocs[0:] :
    currentFrame = np.array(Image.open(loc))
    
#     if currentFrame.shape[0] >= bgImage.shape[0] :
#         currentFrame = currentFrame[currentFrame.shape[0]-bgImage.shape[0]:, :, :]
#     else :
#         currentFrame = np.concatenate((np.zeros((bgImage.shape[0]-currentFrame.shape[0],
#                                                  currentFrame.shape[1],
#                                                  currentFrame.shape[2]), np.uint8),
#                                        currentFrame), axis=0)
        
#     if currentFrame.shape[1] >= bgImage.shape[1] :
#         delta = (currentFrame.shape[1]-bgImage.shape[1])/2
#         currentFrame = currentFrame[:, delta:-delta, :]
#     else :
#         delta = (bgImage.shape[1]-currentFrame.shape[1])/2
#         currentFrame = np.concatenate((np.zeros((currentFrame.shape[0], delta, currentFrame.shape[2]), np.uint8),
#                                        currentFrame,
#                                        np.zeros((currentFrame.shape[0], delta, currentFrame.shape[2]), np.uint8)), axis=1)
    
    alphas = currentFrame[:, :, -1].reshape((currentFrame.shape[0], currentFrame.shape[1], 1))
    finalFrame = (currentFrame[:, :, 0:3]*(alphas/255.0) + bgImage[:, :, 0:3]*((255-alphas)/255.0)).astype(np.uint8)

    
    Image.fromarray(finalFrame).save(basePath + "on_bg/"+loc.split("/")[-1])

# <codecell>

print loc.split("/")[-1]
print basePath + "on_bg/"+loc.split("/")[-1]
# if not os.path.isdir(basePath + "on_bg/") :
#     os.mkdir(basePath + "on_bg/")

# <codecell>

## resize frames and fill emptyness with zeros
desiredSize = [720, 1280, 3]
for frame in np.sort(glob.glob("/media/ilisescu/Data1/PhD/demos/ground_plane/frame-*.png")) :
    frameName = frame.split('/')[-1] 
    im = cv2.resize(np.array(Image.open(frame))[70:70+941, 61:61+1798, :], (1280, 670), interpolation=cv2.INTER_CUBIC)
    imSize = im.shape
    topLeft = ((desiredSize[0]-imSize[0])/2, (desiredSize[1]-imSize[1])/2)
    resizedImage = np.zeros(desiredSize, dtype=uint8)
    resizedImage[topLeft[0]:topLeft[0]+imSize[0], topLeft[1]:topLeft[1]+imSize[1]] = im
    Image.fromarray(resizedImage).save("/media/ilisescu/Data1/PhD/demos/ground_plane/"+frameName)

# <codecell>

## change background of resampled sprite
basePath = "/media/ilisescu/Data1/PhD/demos/sprite_resampled/"
coords = np.loadtxt(glob.glob(basePath+"/*.csv")[0], delimiter=",")[:, 0:2].astype(np.int)
bgImage = np.array(Image.open(basePath+"median.png"))
frameLocs = np.sort(glob.glob(basePath + "frame-*.png"))
for num, loc, i in zip(arange(len(frameLocs)/5), frameLocs[::5], arange(0, len(frameLocs), 5)) :
    currentFrame = np.array(Image.open(loc))
    alphas = currentFrame[:, :, -1].reshape((currentFrame.shape[0], currentFrame.shape[1], 1))
    
    finalFrame = np.copy(bgImage)
    
    finalFrame[coords[i, 1]:coords[i, 1]+currentFrame.shape[0], 
               coords[i, 0]:coords[i, 0]+currentFrame.shape[1], :] = (currentFrame[:, :, 0:3]*(alphas/255.0) + 
                                                                      finalFrame[coords[i, 1]:coords[i, 1]+currentFrame.shape[0], 
                                                                                 coords[i, 0]:coords[i, 0]+currentFrame.shape[1], :]*(1.0-alphas/255.0))
    
#     alphas = currentFrame[:, :, -1].reshape((currentFrame.shape[0], currentFrame.shape[1], 1))
#     finalFrame = (currentFrame[:, :, 0:3]*(alphas/255.0) + bgImage[:, :, 0:3]*((255-alphas)/255.0)).astype(np.uint8)

    
    Image.fromarray(finalFrame).save(basePath+"onBG/frame-{0:05d}.png".format(num+1))

# <headingcell level=2>

# TRY EPIC FLOW

# <codecell>

## trying EpicFlow
flow = sio.loadmat("/home/ilisescu/PhD/EpicFlowTry/flow.mat")['flow']
## flow goes from im1 to im2 so I should be able to make im2 by moving pixels from im1
im1 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-00522.png"))
im2 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-01022.png"))
im1 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-01022.png"))
im2 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-01023.png"))
# im1 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-01022.png"))
# im2 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-00522.png"))
# im1 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-01025.png"))
# im2 = np.array(Image.open("/home/ilisescu/PhD/EpicFlowTry/frame-01036.png"))

# im1 = np.array(Image.open("/home/ilisescu/PhD/data/havana/frame-01025.png"))
# im2 = np.array(Image.open("/home/ilisescu/PhD/data/havana/frame-01036.png"))
flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), 
                                    cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY), 
                                    0.5, 3, 15, 3, 5, 1.1, 0)

print flow.shape

rowIdxs = np.arange(im1.shape[0]).reshape((im1.shape[0], 1)).repeat(im1.shape[1], axis=-1)
colIdxs = np.arange(im1.shape[1]).reshape((1, im1.shape[1])).repeat(im1.shape[0], axis=0)

recon = np.zeros_like(im1)
recon[np.clip(np.array(rowIdxs-flow[:, :, 1], dtype=int), 0, im1.shape[0]-1).flatten(),
             np.clip(np.array(colIdxs-flow[:, :, 0], dtype=int), 0, im1.shape[1]-1).flatten(), :] = im1[rowIdxs.flatten(),
                                                                                                        colIdxs.flatten(), :]

# recon[rowIdxs.flatten(), 
#       colIdxs.flatten(), :] = im2[np.clip(np.array(rowIdxs-flow[:, :, 1], dtype=int), 0, im1.shape[0]-1).flatten(),
#                                   np.clip(np.array(colIdxs-flow[:, :, 0], dtype=int), 0, im1.shape[1]-1).flatten(), :]

remapped = cv2.remap(im2, np.array(flow[:, :, 0]+colIdxs, dtype=np.float32),
                     np.array(flow[:, :, 1]+rowIdxs, dtype=np.float32), cv2.INTER_LINEAR)

figure(); imshow(im1)
figure(); imshow(im2)
figure(); imshow(remapped)

# <codecell>

print np.average(np.linalg.norm(flow.reshape((np.prod(flow.shape[0:2]), 2)), axis=1))

# <codecell>

print np.average(np.linalg.norm(flow.reshape((np.prod(flow.shape[0:2]), 2)), axis=1))

# <codecell>

sio.savemat("/home/ilisescu/PhD/EpicFlowTry/flowTest.mat", {'flowTest':flow})

# <codecell>

gwv.showCustomGraph(flow[:, :, 1])
# print (flow[:, :, 0]+colIdxs)

# <codecell>

figure(); imshow(np.load(dataFolder+"Videos/6489810.avi_distanceMatrix.npy"), interpolation='nearest')

# <headingcell level=2>

# COMPUTE IMAGE MEDIAN

# <codecell>

frameLocs = np.sort(glob.glob(dataFolder + dataSet + "/frame-*.png"))
frameSize = np.array(Image.open(frameLocs[0])).shape[0:2]
numOfFrames = len(frameLocs)
print numOfFrames, frameSize
medianImage = np.zeros((frameSize[0], frameSize[1], 3), dtype=np.uint8)

# <codecell>

allFrames = np.zeros((frameSize[0], frameSize[1], numOfFrames), dtype=np.uint8)
channel = 2
for i in xrange(len(frameLocs)) :
#     allFrames[:, :, i] = cv2.cvtColor(np.array(Image.open(frameLocs[i])), cv2.COLOR_RGB2LAB)[:, :, channel]
    allFrames[:, :, i] = np.array(Image.open(frameLocs[i]))[:, :, channel]
    if np.mod(i, 100) == 0 :
        sys.stdout.write('\r' + "Loaded image " + np.string_(i) + " (" + np.string_(len(frameLocs)) + ")")
        sys.stdout.flush()

# <codecell>

medianImage[:, :, channel] = np.median(allFrames, axis=-1)
# medianImage[:, :, channel] = np.mean(allFrames, axis=-1)

# <codecell>

# figure(); imshow(cv2.cvtColor(medianImage, cv2.COLOR_LAB2RGB))
figure(); imshow(medianImage)

# <codecell>

dataFolder + dataSet + "median.png"

# <codecell>

Image.fromarray(np.array(medianImage, dtype=np.uint8)).save(dataFolder + dataSet + "median.png")

# <codecell>

## combine medians of different datasets (for now specifically for the wave sequence)
allFrames = np.zeros((frameSize[0], frameSize[1], 3, 3), dtype=np.uint8)
allFrames[:, :, :, 0] = np.array(Image.open(dataFolder+"wave1/medianOne.png"))
allFrames[:, :, :, 1] = np.array(Image.open(dataFolder+"wave2/medianOne.png"))
allFrames[:, :, :, 2] = np.array(Image.open(dataFolder+"wave3/medianOne.png"))
allFrames = np.concatenate((allFrames, medianImage.reshape((frameSize[0], frameSize[1], 3, 1))), axis=-1)

medianImage = np.zeros((frameSize[0], frameSize[1], 3), dtype=np.uint8)
medianImage = np.median(allFrames, axis=-1).astype(uint8)
figure(); imshow(medianImage)

# <codecell>

gwv.showCustomGraph(np.sum((allFrames[:, :, :, 0]/255.0-allFrames[:, :, :, 1]/255.0)**2, axis=-1)**.5)

# <codecell>

Image.fromarray(np.array(medianImage, dtype=np.uint8)).save(dataFolder + "wave3/" + "median.png")

# <headingcell level=2>

# RENDER SPRITE ON BACKGROUND

# <codecell>

basePath = "/media/ilisescu/Data1/PhD/data/havana/"
bgImage = np.array(Image.open(basePath+"median.png"))
for i in np.arange(800, 800+476) :
    currentFrame = np.array(Image.open(basePath+"bus1/bus1-frame-{0:05d}.png".format(i)))
    spriteLoc = np.argwhere(currentFrame[:, :, -1] != 0)
    alphas = currentFrame[spriteLoc[:, 0], spriteLoc[:, 1], -1].reshape((len(spriteLoc), 1)).repeat(3, axis=-1)
    
    finalFrame = np.copy(bgImage)
    finalFrame[spriteLoc[:, 0], spriteLoc[:, 1], :] = (currentFrame[spriteLoc[:, 0], spriteLoc[:, 1], 0:-1]*(alphas/255.0) + 
                                                       bgImage[spriteLoc[:, 0], spriteLoc[:, 1], :]*((255-alphas)/255.0))

    
    Image.fromarray((finalFrame).astype(numpy.uint8)).save(basePath+"bus1OnMedian/bus1-frame-{0:05d}.png".format(i))

# <codecell>

img = np.array(Image.open(dataFolder+"testImage.png"))
figure(); imshow(img)

# <codecell>

square = np.vstack((np.array([[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]]).T, ones(5)))
print square
intrinsics = np.array([[640.0, 0.0, 320.0, 0.0], [0.0, 360.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 4.0], [0.0, 0.0, 0.0, 1.0]])


theta = 10.0*(np.pi/180.0)
R = np.array([[np.cos(theta), 0.0, np.sin(theta), 0.], [0.0, 1.0, 0.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta), 0.0], [0.0, 0.0, 0.0, 1.0]])

squareImg = np.dot(np.dot(intrinsics, extrinsics), np.dot(R, square))
squareImg = squareImg/squareImg[-1, :]
print squareImg[0:2, :]

figure(); 
xlim(0, 640); ylim(0, 360)
plot(squareImg[0, :], squareImg[1, :])

# <codecell>

origSquare = np.copy(squareImg)

# <codecell>

H = cv2.findHomography(squareImg[0:2, 0:-1].T, origSquare[0:2, 0:-1].T)[0]
xs = np.ndarray.flatten(np.arange(img.shape[1], dtype=float).reshape((img.shape[1], 1)).repeat(img.shape[0], axis=-1))
ys = np.ndarray.flatten(np.arange(img.shape[0], dtype=float).reshape((1, img.shape[0])).repeat(img.shape[1], axis=0))
data = np.array(np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys))), np.ones(len(ys)))), dtype=int)
imgWarped = np.zeros(img.shape, dtype=np.uint8)

warpedCoords = np.dot(H, data)
warpedCoords /= warpedCoords[-1, :]

for warpedCoord, coord in zip(np.array(warpedCoords.T, dtype=int), data.T) :
    if warpedCoord[0] < img.shape[1] and warpedCoord[0] > 0 and warpedCoord[1] < img.shape[0] and warpedCoord[1] > 0 :
        imgWarped[coord[1], coord[0], :] = img[warpedCoord[1], warpedCoord[0], :]
        
figure(); imshow(imgWarped)
figure(); imshow(img)

# <codecell>

square = np.vstack((np.array([[-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]]).T, ones(5)))
print square
intrinsics = np.eye(3, 4)#np.array([[640.0, 0.0, 320.0, 0.0], [0.0, 360.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]])
print intrinsics
print extrinsics
print np.dot(intrinsics, extrinsics)

# square = np.dot(np.dot(intrinsics, extrinsics), square)
# square = square/square[-1, :]
print square

theta = 10.0*(np.pi/180.0)
R = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
t = np.array([0.0, 0.0, 0.0])
n = np.array([0.0, 0.0, 1.0])
d = 2.0

# H = R-np.dot(t, n.T)/d
# H = np.dot(np.dot(R, np.eye(3)-np.dot(extrinsics[0:3, -1]-t, n.T)/d), extrinsics[0:3, 0:3].T)
print H
# print np.dot(t, n.T)/d

H = np.hstack((np.vstack((R, np.zeros((1, 3)))), np.array([[0.0], [0], [0], [1]])))
extrinsicsCam2 = np.dot(H, extrinsics)
print H
print extrinsicsCam2

# square = np.dot(np.dot(np.dot(intrinsics[:, 0:-1], H), np.linalg.inv(intrinsics[:, 0:-1])), square)
# square = np.dot(np.linalg.inv(intrinsics[:, 0:-1]), square)
# square =  np.dot(np.dot(intrinsics, np.dot(extrinsics, H)), square)

# cam2 = np.dot(np.dot(np.dot(intrinsics, extrinsicsCam2), 
#                      np.hstack((np.linalg.inv(intrinsics[:, 0:-1]), np.zeros((3, 1))))), 
#               np.linalg.inv(extrinsicsCam2))
print cam2.shape
# square = np.dot(np.dot(intrinsics[:, 0:3], np.dot(np.linalg.inv(extrinsics), extrinsicsCam2)[0:3, :]), square)
# square = np.dot(np.dot(intrinsics, extrinsicsCam2), square)

square = square/square[-1, :]
print square[0:2, :]

figure(); 
xlim(0, 640); ylim(0, 360)
plot(square[0, :], square[1, :], )

# <codecell>

xs = np.ndarray.flatten(np.arange(img.shape[1], dtype=float).reshape((img.shape[1], 1)).repeat(img.shape[0], axis=-1))
ys = np.ndarray.flatten(np.arange(img.shape[0], dtype=float).reshape((1, img.shape[0])).repeat(img.shape[1], axis=0))
data = np.array(np.vstack((xs.reshape((1, len(xs))), ys.reshape((1, len(ys))), np.ones(len(ys)))), dtype=int)
imgWarped = np.zeros(img.shape, dtype=np.uint8)


theta = 45.0*(np.pi/180.0)
R = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
originT = (np.array([[img.shape[1]], [img.shape[0]], [0]])/2).repeat(data.shape[-1], axis=-1)

warpedCoords = np.dot(R, data-originT)+originT
warpedCoords /= warpedCoords[-1, :]

for warpedCoord, coord in zip(np.array(warpedCoords.T, dtype=int), data.T) :
    if warpedCoord[0] < img.shape[1] and warpedCoord[0] > 0 and warpedCoord[1] < img.shape[0] and warpedCoord[1] > 0 :
        imgWarped[coord[1], coord[0], :] = img[warpedCoord[1], warpedCoord[0], :]
        
figure(); imshow(imgWarped)
figure(); imshow(img)

# <codecell>

print warpedCoords
print data

# <codecell>

print np.max(warpedCoords[0, :]), np.min(warpedCoords[0, :]), np.max(warpedCoords[1, :]), np.min(warpedCoords[1, :])
print R
print np.linalg.inv(R)

# <codecell>

print np.dot(intrinsics, extrinsicsCam2)
print np.hstack((np.linalg.inv(intrinsics[:, 0:-1]), np.zeros((3, 1))))

# <codecell>

print np.dot(extrinsicsCam2, np.linalg.inv(extrinsics))
print extrinsics
print extrinsicsCam2

# <codecell>

print np.dot(extrinsics[0:3, -1]-t, n.T)/d

# <codecell>

print np.dot(extrinsics, H).shape
print extrinsics.shape

# <codecell>

print np.dot(t, n.T)/d

# <codecell>

import sift

# <codecell>

sift.process_image(dataFolder + '/mopeds/frame-00001.png', 'tmp.key')
l1,d1 = sift.read_features_from_file('tmp.key')
im = array(Image.open(dataFolder + '/mopeds/frame-00001.png'))
sift.plot_features(im,l1)

# <codecell>

f = loadtxt('tmp2.key')
l = f[:,:4]
d = f[:,4:]

# <codecell>

def draw_circle(c,r):
    t = arange(0,1.01,.01)*2*pi
    x = r*cos(t) + c[0]
    y = r*sin(t) + c[1]
    plot(x,y,'b',linewidth=2)

figure(); imshow(im)
if False:
    [draw_circle([p[0],p[1]],p[2]) for p in l]
else:
    plot(l[:,0],l[:,1],'ob')
axis('off')

# <codecell>

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray,2,3,0.04)
# dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
corners  = np.zeros(gray.shape)
corners[dst>0.01*dst.max()]=255

# <codecell>

figure(); imshow(im)
scatter(np.argwhere(corners == 255)[:, 1], np.argwhere(corners == 255)[:, 0])

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
sampleData = "ribbon2/"
# sampleData = "ribbon1_matted/"
# sampleData = "little_palm1_cropped/"
# sampleData = "ballAnimation/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame-*.png")
mattes = glob.glob(dataFolder + sampleData + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes)

lowThresh = 96
highThresh = lowThresh*2

# <codecell>

img = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
matte = cv2.cvtColor(cv2.imread(mattes[0]), cv2.COLOR_BGR2GRAY)

imgEdges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), lowThresh, highThresh)
matteEdges = cv2.Canny(matte, lowThresh, highThresh)
matteEdges = cv2.dilate(matteEdges, np.ones((6,6),dtype=np.uint8))
matteEdges = cv2.erode(matteEdges, np.ones((6,6),dtype=np.uint8))

# figure(); imshow(imgEdges*(matte/255.0), interpolation='nearest')
figure(); imshow(matteEdges, interpolation='nearest')

# <codecell>

## find points on matte edges
edgePoints = np.argwhere(matteEdges == np.max(matteEdges))
## closes point to top-left (i.e.) [0, 0]
startPoint = edgePoints[np.argmin(np.sum(edgePoints, axis=1)), :]
print startPoint
scatter(startPoint[1], startPoint[0])

# <codecell>

edgePoints

