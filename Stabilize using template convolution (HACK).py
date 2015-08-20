# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import sys

import cv2
import time
import os
import scipy.io as sio
import glob


from PIL import Image

# <codecell>

# dataPath = "/home/ilisescu/PhD/data/"
# dataSet = "havana/"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "clouds_subsample10/"#"clouds/"
# dataSet = "theme_park_cloudy/"
# dataSet = "theme_park_sunny/"
# dataSet = "candle2/"
# dataSet = "candle3/"
dataSet = "candle3/stabilized/segmentedAndCropped/potential video texture/"

# <codecell>

## load dataSet relevant data
frameLocs = np.sort(glob.glob(dataPath + dataSet + "/frame-*.png"))
numOfFrames = len(frameLocs)
numOfTrackedSprites = 0
print numOfFrames

# <codecell>


refImage = cv2.Canny(cv2.cvtColor(cv2.imread(frameLocs[0]), cv2.COLOR_BGR2RGB), 50, 200, 3)
figure(); imshow(refImage)
startRect = np.array([[502, 380], [585, 519]]) ## candle3 potential video texture
scatter(startRect[:, 0], startRect[:, 1])

# <codecell>

## get some reference frames and align them together
numReferenceFrames = 4
# referenceFrameIdxs = arange(0, len(frameLocs), len(frameLocs)/numReferenceFrames)[0:numReferenceFrames]
referenceFrameIdxs = [0, 432, 864, 1728]

refImage = cv2.Canny(cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[0]]), cv2.COLOR_BGR2RGB), 50, 200, 3)
figure(); imshow(cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[0]]), cv2.COLOR_BGR2RGB))

template = refImage[startRect[0, 1]:startRect[1, 1]+1, startRect[0, 0]:startRect[1, 0]+1]

allTemplates = np.zeros((numReferenceFrames, template.shape[0], template.shape[1]))
allTemplates[0, :, :] = np.copy(template)

for i, j in zip(referenceFrameIdxs[1:], arange(1, numReferenceFrames)) :
    nextImage = cv2.cvtColor(cv2.imread(frameLocs[i]), cv2.COLOR_BGR2RGB)
    
    if usePatch :
        convolved = cv2.filter2D(cv2.Canny(nextImage, 50, 200, 3)[matchPatchTopLeft[1]:matchPatchTopLeft[1]+matchPatchSize[1], 
                                                                  matchPatchTopLeft[0]:matchPatchTopLeft[0]+matchPatchSize[0]], cv2.CV_32F, template)
    else :
        convolved = cv2.filter2D(cv2.Canny(nextImage, 50, 200, 3), cv2.CV_32F, template)
        
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(convolved)
    
    if usePatch :
        maxLoc += matchPatchTopLeft
        
    loc = maxLoc-np.array([template.shape[1], template.shape[0]])/2
    newRect = np.array([loc, loc + np.array([template.shape[1], template.shape[0]])])
    newRect[:, 0] = np.clip(newRect[:, 0], 0, nextImage.shape[1]-1)
    newRect[:, 1] = np.clip(newRect[:, 1], 0, nextImage.shape[0]-1)
    
    deltaPixels = startRect[0, :]-newRect[0, :]
    
    allTemplates[j, :, :] = cv2.warpAffine(cv2.Canny(nextImage, 50, 200, 3), np.array([[1, 0, deltaPixels[0]], 
                                                     [0, 1, deltaPixels[1]]], dtype=float), 
                                (nextImage.shape[1], nextImage.shape[0]))[startRect[0, 1]:startRect[1, 1]+1, startRect[0, 0]:startRect[1, 0]+1]
    
    stabilized = cv2.warpAffine(nextImage, np.array([[1, 0, startRect[0, 0]-newRect[0, 0]], 
                                                     [0, 1, startRect[0, 1]-newRect[0, 1]]], dtype=float), 
                                (nextImage.shape[1], nextImage.shape[0]))
    figure(); imshow(stabilized)

# <codecell>

## manually align frame by clicking on corresponding point in images
numReferenceFrames = 10
referenceFrameIdxs = arange(0, len(frameLocs), len(frameLocs)/numReferenceFrames)[0:numReferenceFrames]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[0]]), cv2.COLOR_BGR2RGB))

count = 0

def onclick(event):
    print "[", event.xdata, ",", event.ydata, "]"
    global count, ax
    count += 1
    if count < numReferenceFrames :
        ax.imshow(cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[count]]), cv2.COLOR_BGR2RGB))
        show()
    else :
        print "done"
    sys.stdout.flush()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

# for i in referenceFrameIdxs :
#     ax.imshow(cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[0]]), cv2.COLOR_BGR2RGB))

# <codecell>

## candle2
# deltaPixels = np.array([[ 667.887096774 , 688.102150538 ],
#                         [ 678.209677419 , 705.306451613 ],
#                         [ 678.209677419 , 698.424731183 ],
#                         [ 672.188172043 , 684.661290323 ],
#                         [ 674.768817204 , 706.166666667 ],
#                         [ 676.489247312 , 694.123655914 ],
#                         [ 677.349462366 , 698.424731183 ],
#                         [ 677.349462366 , 703.865591398 ],
#                         [ 673.908602151 , 698.424731183 ],
#                         [ 682.930107527 , 686.661290323 ]], dtype= int)
## candle3
deltaPixels = np.array([[ 567.867346939 , 655.214285714 ],
                        [ 561.030612245 , 637.683673469 ],
                        [ 561.030612245 , 636.683673469 ],
                        [ 560.030612245 , 654.132653061 ],
                        [ 561.112244898 , 658.887755102 ],
                        [ 559.602040816 , 657.051020408 ],
                        [ 561.43877551 , 648.785714286 ],
                        [ 557.765306122 , 656.132653061 ],
                        [ 567.867346939 , 650.867346939 ],
                        [ 562.357142857 , 635.010204082 ]], dtype= int)

startRect = np.array([[394, 551], [991, 719]]) ## candle3
# startRect = np.array([[394, 651], [991, 719]]) ## candle3
deltaPixels = deltaPixels[0, :] - deltaPixels
print deltaPixels

cannyThreshold1 = 40#50
cannyThreshold2 = 100#200
refImage = cv2.Canny(cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[0]]), cv2.COLOR_BGR2RGB), cannyThreshold1, cannyThreshold2, 3)
template = refImage[startRect[0, 1]:startRect[1, 1]+1, startRect[0, 0]:startRect[1, 0]+1]

allTemplates = np.zeros((numReferenceFrames, template.shape[0], template.shape[1]))
# allTemplates[0, :, :] = np.copy(template)

for i in xrange(len(deltaPixels)) :
    print i
    refImage = cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[i]]), cv2.COLOR_BGR2RGB)
    
    allTemplates[i, :, :] = cv2.warpAffine(cv2.Canny(refImage, cannyThreshold1, cannyThreshold2, 3), np.array([[1, 0, deltaPixels[i, 0]], 
                                                     [0, 1, deltaPixels[i, 1]]], dtype=float), 
                                (refImage.shape[1], refImage.shape[0]))[startRect[0, 1]:startRect[1, 1]+1, startRect[0, 0]:startRect[1, 0]+1]

# <codecell>

figure(); imshow(allTemplates[1, :, :])

# <codecell>

# for template in allTemplates :
#     figure(), imshow(template, interpolation='nearest')
for i in xrange(len(deltaPixels)) :
    print i
    refImage = cv2.cvtColor(cv2.imread(frameLocs[referenceFrameIdxs[i]]), cv2.COLOR_BGR2RGB)
    
    stabilized = cv2.warpAffine(refImage, np.array([[1, 0, deltaPixels[i, 0]], 
                                                     [0, 1, deltaPixels[i, 1]]], dtype=float), 
                                (refImage.shape[1], refImage.shape[0]))
    figure(); imshow(stabilized)

# <codecell>

# startRect = np.array([[567, 662], [792, 719]]) ## candle2
# startRect = np.array([[394, 551], [991, 719]]) ## candle3
# startRect = np.array([[394, 651], [991, 719]]) ## candle3
deltaPixels = np.zeros(2, dtype=int)
# startRect = np.array([[567, 625], [752, 719]])
matchPatchTopLeft = np.array([500, 600])
matchPatchSize = np.array([350, 120])
usePatch = False
if usePatch :
    print "Convolving only in a patch"
else :
    print "Convolving against full image"

if not os.path.isdir(dataPath+dataSet+"stabilized") :
    os.mkdir(dataPath+dataSet+"stabilized")

Image.fromarray(np.array(cv2.cvtColor(cv2.imread(frameLocs[0]), cv2.COLOR_BGR2RGB), dtype=np.uint8)).save(dataPath+dataSet+"stabilized/frame-00001.png")

# refImage = cv2.Canny(cv2.cvtColor(cv2.imread(dataPath+dataSet+"frame-{0:05}.png".format(1)), cv2.COLOR_BGR2RGB), 50, 200, 3)
# template = refImage[startRect[0, 1]:startRect[1, 1]+1, startRect[0, 0]:startRect[1, 0]+1]

# refImage = cv2.cvtColor(cv2.imread(dataPath+dataSet+"stabilized/frame-{0:05}.png".format(1)), cv2.COLOR_BGR2RGB)
for i in xrange(len(frameLocs)-1) : # 630, 650) :
    ## refImage is where I get the template from
    ## I first get the normal image, get its canny edges and then move it by however much I moved it earlier
    ## this way I don't get the extra edges beteween the edge of the image and the zeros the stabilized 
    ## image gets filled with where tehre are no colors
#     refImage = cv2.cvtColor(cv2.imread(dataPath+dataSet+"stabilized/frame-{0:05}.png".format(i+1)), cv2.COLOR_BGR2RGB)
    refImage = cv2.Canny(cv2.cvtColor(cv2.imread(dataPath+dataSet+"frame-{0:05}.png".format(i+1)), cv2.COLOR_BGR2RGB), cannyThreshold1, cannyThreshold2, 3)
#     refImage = cv2.Laplacian(cv2.cvtColor(cv2.imread(dataPath+dataSet+"frame-{0:05}.png".format(i+1)), cv2.COLOR_BGR2GRAY), cv2.CV_16S, ksize=3, scale=1, delta=0)
    refImage = cv2.warpAffine(refImage, np.array([[1, 0, deltaPixels[0]], 
                                                     [0, 1, deltaPixels[1]]], dtype=float), 
                                (refImage.shape[1], refImage.shape[0]))
    templatePrev = refImage[startRect[0, 1]:startRect[1, 1]+1, startRect[0, 0]:startRect[1, 0]+1]
    
    
    nextImage = cv2.cvtColor(cv2.imread(dataPath+dataSet+"frame-{0:05}.png".format(i+2)), cv2.COLOR_BGR2RGB)

#     template = cv2.Canny(refImage, 50, 200, 3)[startRect[0, 1]:startRect[1, 1]+1, startRect[0, 0]:startRect[1, 0]+1]
#     figure(); imshow(refImage)
#     figure(); imshow(nextImage)
#     figure(); imshow(template)
    currentMax = 0
    count = 0
    usedTemplate = 0
    for template in templatePrev.reshape((1, templatePrev.shape[0], templatePrev.shape[1])) :#allTemplates :
#     for template in allTemplates :
#     for template in np.concatenate((allTemplates, templatePrev.reshape((1, templatePrev.shape[0], templatePrev.shape[1])))) :

        if usePatch :
            convolved = cv2.filter2D(cv2.Canny(nextImage, cannyThreshold1, cannyThreshold2, 3)[matchPatchTopLeft[1]:matchPatchTopLeft[1]+matchPatchSize[1], 
                                                                      matchPatchTopLeft[0]:matchPatchTopLeft[0]+matchPatchSize[0]], cv2.CV_32F, template)
    #         convolved = cv2.filter2D(cv2.Laplacian(cv2.cvtColor(nextImage, cv2.COLOR_RGB2GRAY), cv2.CV_16S, ksize=3, scale=1, delta=0)[matchPatchTopLeft[1]:matchPatchTopLeft[1]+matchPatchSize[1], 
    #                                                                   matchPatchTopLeft[0]:matchPatchTopLeft[0]+matchPatchSize[0]], cv2.CV_32F, template)
        else :
            convolved = cv2.filter2D(cv2.Canny(nextImage, cannyThreshold1, cannyThreshold2, 3), cv2.CV_32F, template)
    #         convolved = cv2.filter2D(cv2.Laplacian(cv2.cvtColor(nextImage, cv2.COLOR_RGB2GRAY), cv2.CV_16S, ksize=3, scale=1, delta=0), cv2.CV_32F, template)

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(convolved)
    
        if usePatch :
            maxLoc += matchPatchTopLeft
            
#         print maxVal
        if currentMax < maxVal :
            usedTemplate = count
            currentMax = maxVal
            loc = maxLoc-np.array([template.shape[1], template.shape[0]])/2
            
        count += 1
        
    newRect = np.array([loc, loc + np.array([template.shape[1], template.shape[0]])])
#     print template.shape
#     print minVal, maxVal, minLoc, maxLoc
#     print newRect
    newRect[:, 0] = np.clip(newRect[:, 0], 0, nextImage.shape[1]-1)
    newRect[:, 1] = np.clip(newRect[:, 1], 0, nextImage.shape[0]-1)
    
    ######### find the pixels which will be set as zero by warpAffine because the original image was moved #########
    
    deltaPixels = startRect[0, :]-newRect[0, :]
        
    ########################### warp the nextImage by moving it to match the found newRect ###########################
    
    stabilized = cv2.warpAffine(nextImage, np.array([[1, 0, startRect[0, 0]-newRect[0, 0]], 
                                                     [0, 1, startRect[0, 1]-newRect[0, 1]]], dtype=float), 
                                (nextImage.shape[1], nextImage.shape[0]))
#     print newRect
#     figure(); imshow(zeroPixels, interpolation='nearest')
#     figure(); imshow(stabilized)
    
#     startRect = np.copy(newRect)
    sys.stdout.write('\r' + "Stabilized image " + np.string_(i+1) + " of " + np.string_(len(frameLocs)-1) + " using template " + np.string_(usedTemplate))
    sys.stdout.flush()
    
    Image.fromarray(np.array(stabilized, dtype=np.uint8)).save(dataPath+dataSet+"stabilized/frame-{0:05}.png".format(i+2))
    

#     method = cv2.TM_CCORR_NORMED

#     matches = cv2.matchTemplate(nextImage, template, method)
#     figure(); imshow(matches)

#     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matches)
#     print minVal, maxVal, minLoc, maxLoc

#     if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED :
#         loc = minLoc
#     else :
#         loc = maxLoc

#     bbox = np.array([[loc[0], loc[1]], 
#                      [loc[0]+template.shape[1], loc[1]], 
#                      [loc[0]+template.shape[1], loc[1]+template.shape[0]], 
#                      [loc[0], loc[1]+template.shape[0]]])

#     figure(); imshow(nextImage)
#     plot(bbox[[0, 1, 2, 3, 0], 0], bbox[[0, 1, 2, 3, 0], 1])

# <codecell>

tmp2 = cv2.cvtColor(cv2.imread(dataPath+dataSet+"frame-{0:05}.png".format(i+1)), cv2.COLOR_BGR2RGB)
figure(); imshow(cv2.Canny(tmp2, 50, 200, 3), interpolation='nearest')
figure(); imshow(cv2.Laplacian(cv2.cvtColor(tmp2, cv2.COLOR_RGB2GRAY), cv2.CV_16S, ksize=3, scale=1, delta=0), interpolation='nearest')

# <codecell>

# figure(); imshow(tmp, interpolation='nearest')
figure(); imshow(cv2.Canny(stabilized, 50, 200, 3), interpolation='nearest')

# <codecell>

print incrementalTemplate.shape, template.shape
print newRect
print newRect[1, :]-newRect[0, :]
print newRect-newRect[0, :]
print np.array([loc, loc + np.array([template.shape[1], template.shape[0]])])
print np.array([loc, loc + np.array([template.shape[1], template.shape[0]])])-newRect

# <codecell>

tmp = np.array([[-5, 669], [795, 727]])
print tmp

tmp[:, 0] = np.clip(tmp[:, 0], 0, nextImage.shape[1]-1)
tmp[:, 1] = np.clip(tmp[:, 1], 0, nextImage.shape[0]-1)
print tmp
print np.array([[-5, 669], [795, 727]])-tmp

# <codecell>

figure(); imshow(template)
figure(); imshow(convolved)

# <codecell>

convolved = cv2.filter2D(nextImage, cv2.CV_32F, template)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(convolved)
print template.shape
print minVal, maxVal, minLoc, maxLoc
loc = maxLoc-np.array([template.shape[1], template.shape[0]])/2
newRect = np.array([loc, loc + np.array([template.shape[1], template.shape[0]])])
print newRect
newRect[:, 0] = np.clip(newRect[:, 0], 0, nextImage.shape[1]-1)
newRect[:, 1] = np.clip(newRect[:, 1], 0, nextImage.shape[0]-1)
print newRect

# <codecell>

tmp = cv2.warpAffine(nextImage, np.array([[1, 0, startRect[0, 0]-newRect[0, 0]], [0, 1, startRect[0, 1]-newRect[0, 1]]], dtype=float), (nextImage.shape[1], nextImage.shape[0]))
figure(); imshow(tmp)

