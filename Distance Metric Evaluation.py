# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import sys
import os
import glob
import cv2

import VideoTexturesUtils as vtu
import GraphWithValues as gwv


from PySide import QtCore, QtGui

app = QtGui.QApplication(sys.argv)

# <codecell>

# dataPath = "/home/ilisescu/PhD/data"
dataPath = "/media/ilisescu/Data1/PhD/data/"
# dataSet = "splashes_water/"
# dataSet = "small_waterfall/"
# dataSet = "sisi_flag/"
<<<<<<< HEAD
dataSet = "eu_flag_ph_left/"
# dataSet = "candle1/segmentedAndCropped/"
# dataSet = "candle2/subset_stabilized/segmentedAndCropped/"
# dataSet = "candle3/stabilized/segmentedAndCropped/"
=======
# dataSet = "eu_flag_ph_left/"
# dataSet = "candle1/segmentedAndCropped/"
dataSet = "candle2/subset_stabilized/segmentedAndCropped/"
dataSet = "candle3/stabilized/segmentedAndCropped/"
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
best10Jumps = []
frames = glob.glob(dataPath + dataSet + "frame-*.png")
mattes = glob.glob(dataPath + dataSet + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
print numFrames

# <codecell>

# distanceMatrix = np.load(dataPath+dataSet+"l2_distMat.npy")
isL2 = True
if isL2 :
    distanceMatrix = np.load(dataPath+dataSet+"vanilla_distMat.npy")
#     distanceMatrix = np.load(dataPath+dataSet+"psiWeighted_allOnesWeights_distMat.npy")
else :
#     distanceMatrix = np.load(dataPath+dataSet+"psiWeighted_userLabelledExamples_distMat.npy")
    distanceMatrix = np.load(dataPath+dataSet+"psiWeighted_1000randomBadExamples_distMat.npy")
# figure(); imshow(distanceMatrix)

    ## make sure the cost of staying at the same location is as big as max cost and normalize to [0, 1]
    distanceMatrix[arange(len(distanceMatrix)), arange(len(distanceMatrix))] = np.max(distanceMatrix)
    distanceMatrix /= np.max(distanceMatrix)

gwv.showCustomGraph(distanceMatrix)

# <codecell>

gwv.showCustomGraph(np.load(dataPath+dataSet+"psiWeighted_allOnesWeights_distMat.npy"))
gwv.showCustomGraph(np.load(dataPath+dataSet+"vanilla_distMat.npy"))

# <codecell>

gwv.showCustomGraph(vtu.filterDistanceMatrix(distanceMatrix, numFilterFrames, isRepetitive))

# <codecell>

## all params (same for all I guess? but maybe I need to normalize all distances to the same interval)
numFilterFrames = 4
isRepetitive = False
alpha = 0.999
p = 2.0
sigmaMult = 0.005
normalizeRows = True
correction = numFilterFrames + 1
startFrame = 0
verbose = True
totalHoursToGenerate = 1.5
totalFramesToGenerate = int(1.5*60*60*30) # mins*secs*fps
totalJumps = 200000
print totalFramesToGenerate

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
    
    return distMat

filteredDist = vtu.filterDistanceMatrix(distanceMatrix, numFilterFrames, isRepetitive)
figure(); imshow(filteredDist)

optimizedDist = estimateFutureCost(0.999, 2.0, filteredDist)
figure(); imshow(optimizedDist)

# <codecell>

filteredDist = vtu.filterDistanceMatrix(distanceMatrix, numFilterFrames, isRepetitive)
figure(); imshow(filteredDist)

optimizedDist = vtu.estimateFutureCost(0.999, 2.0, filteredDist)
figure(); imshow(optimizedDist)

# <codecell>

rangeDistances = None

def getRangeDistances(l, frameRange, featherLevel, weightShape) :

    kernelLength = np.float(frameRange[1]-frameRange[0])
#         print kernelLength
    kernelRange = np.arange(1.0, 0.0, -(1.0)/np.floor(kernelLength*featherLevel))
#         print kernelRange
    kernel = np.zeros(np.int(kernelLength+1))
    kernel[0:len(kernelRange)] = kernelRange
    kernel = np.hstack((kernel, kernel[np.arange(len(kernel)-2, -1, -1)]))

    rangeDistances = np.arange(0, weightShape[0])
    rangeDistances = np.abs(rangeDistances - frameRange[1])
    rangeDistances[frameRange[0]:frameRange[2]+1] = rangeDistances[frameRange[0]:frameRange[2]+1]*kernel
#         print rangeDistances


    rangeDistances = np.repeat(np.reshape(rangeDistances, (1, len(rangeDistances))), len(rangeDistances), axis=0)

    return rangeDistances

<<<<<<< HEAD
middleFrame = 750-correction#1150-correction
rangeSpan = 300#600
=======
middleFrame = 1150-correction
rangeSpan = 600
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36

useOptimized = False

if isL2 :
    if useOptimized :
        distMat = optimizedDist
    else :
        distMat = filteredDist[1:, :-1]
else :
    distMat = vtu.filterDistanceMatrix(distanceMatrix, numFilterFrames, isRepetitive)
    
rangeDistances = getRangeDistances(0.7, np.array([middleFrame-((rangeSpan-1)/2), middleFrame, middleFrame+((rangeSpan-1)/2)]), 0.2, distMat.shape)

probs, cumProbs = vtu.getProbabilities(distMat, sigmaMult*15, rangeDistances, normalizeRows)
    
gwv.showCustomGraph(cumProbs)

# <codecell>

<<<<<<< HEAD
gwv.showCustomGraph(probs)

# <codecell>

gwv.showCustomGraph(rangeDistances)

# <codecell>

=======
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
## find highest probability jumps
sortedProbs = np.argsort(np.ndarray.flatten(probs))[::-1]
sortedProbs = np.array([np.array(sortedProbs/probs.shape[0], dtype=int), 
                        np.array(np.mod(sortedProbs, probs.shape[0]), dtype=int)]).T

sortedProbs += correction
jumpIndices = np.argwhere(np.abs(np.diff(sortedProbs, axis=-1)) != 1)[:, 0]
print sortedProbs[jumpIndices, :][:10]

# <codecell>

## find highest probability jumps without dead end prevention
tmpProbs, tmpCumProbs = vtu.getProbabilities(filteredDist[1:, :-1], sigmaMult, None, normalizeRows)

sortedProbs = np.argsort(np.ndarray.flatten(tmpProbs))[::-1]
sortedProbs = np.array([np.array(sortedProbs/tmpProbs.shape[0], dtype=int), 
                        np.array(np.mod(sortedProbs, tmpProbs.shape[0]), dtype=int)]).T

sortedProbs += correction
jumpIndices = np.argwhere(np.abs(np.diff(sortedProbs, axis=-1)) != 1)[:, 0]
print sortedProbs[jumpIndices, :][:10]

# <codecell>

## run stochastic video textures and store the jump locations
currentFrame = startFrame
# finalFrames = []

# if verbose :
#     print 'starting at frame', currentFrame+correction
# finalFrames.append(currentFrame)
# for i in range(1, totalFramesToGenerate) :
jumpLocations = np.zeros(distanceMatrix.shape, dtype=int)
jumpCounter = 0
prevFrame = vtu.getNewFrame(currentFrame, cumProbs)
while jumpCounter < totalJumps :
#     currentFrame = randFrame(probabilities[currentFrame, :])
    currentFrame = vtu.getNewFrame(prevFrame, cumProbs, 50)
#     print currentFrame,
#     finalFrames.append(vtu.getNewFrame(currentFrame, cumProbs))
    if currentFrame != prevFrame + 1 :
#         print "jumping to frame", finalFrames[-1]+correction, "from", finalFrames[-2]+correction, "at generated frame", 
#         print i, 'of', totalFramesToGenerate #, prob
        jumpCounter += 1
        jumpLocations[prevFrame+correction, currentFrame+correction] += 1
        
    if np.mod(jumpCounter, 1000) == 0 :
        sys.stdout.write('\r' + "Jumped " + np.string_(jumpCounter) + " times")
        sys.stdout.flush()
    prevFrame = currentFrame

# <codecell>

gwv.showCustomGraph(jumpLocations)

# <codecell>

# best10Jumps =  best10Jumps[0:-1]
best10Jumps = []

# <codecell>

# best10Jumps = [np.array([[ 478,  104, 7130],      ## 1000000 jumps
#                          [ 478,  105, 7003],
#                          [ 477,  105, 6546],
#                          [ 478,   93, 6462],
#                          [ 484,   93, 6424],
#                          [ 483,   93, 6408],
#                          [ 477,   93, 6329],
#                          [ 477,  104, 6269],
#                          [ 483,  105, 6117],
#                          [ 479,   93, 6079]])]

sortedJumps = np.argsort(np.ndarray.flatten(jumpLocations))[::-1]
sortedJumps = np.array([np.array(sortedJumps/distanceMatrix.shape[0], dtype=int), 
                        np.array(np.mod(sortedJumps, distanceMatrix.shape[0]), dtype=int)]).T
topRanking = np.hstack((sortedJumps[:200, :], jumpLocations[sortedJumps[:200, 0], sortedJumps[:200, 1]].reshape((200, 1))))
print np.concatenate((topRanking, distMat[topRanking[:, 0], topRanking[:, 1]].reshape(200, 1)), axis=-1)
# best10Jumps.append(np.hstack((sortedJumps[:10, :], jumpLocations[sortedJumps[:10, 0], sortedJumps[:10, 1]].reshape((10, 1)))))
print best10Jumps

# <codecell>

## right
print frames[1268]
print frames[1072]

# <codecell>

## right
print frames[1370]
print frames[1198]

# <codecell>

## left
print frames[591]#636]
print frames[371]#356]

# <codecell>

## left
print frames[723]#636]
print frames[512]#356]

# <codecell>

## right
print frames[789]
print frames[724]

# <codecell>

## right
print frames[223]
print frames[163]

# <codecell>

## right
print frames[193]
print frames[143]

# <codecell>

## left ish
print frames[1126]
print frames[936]

# <codecell>

sortedJumps = np.argsort(np.ndarray.flatten(jumpLocations))[::-1]
sortedJumps = np.array([np.array(sortedJumps/distanceMatrix.shape[0], dtype=int), 
                        np.array(np.mod(sortedJumps, distanceMatrix.shape[0]), dtype=int)]).T
print np.hstack((sortedJumps[:10, :], jumpLocations[sortedJumps[:10, 0], sortedJumps[:10, 1]].reshape((10, 1))))

# <codecell>

sortedJumps = np.argsort(np.ndarray.flatten(jumpLocations))[::-1]
sortedJumps = np.array([np.array(sortedJumps/distanceMatrix.shape[0], dtype=int), 
                        np.array(np.mod(sortedJumps, distanceMatrix.shape[0]), dtype=int)]).T
print np.hstack((sortedJumps[:10, :], jumpLocations[sortedJumps[:10, 0], sortedJumps[:10, 1]].reshape((10, 1))))

# <codecell>

best10Jumps = [array([[ 478,  104, 7130], ## 1000000
                       [ 478,  105, 7003],
                       [ 477,  105, 6546],
                       [ 478,   93, 6462],
                       [ 484,   93, 6424],
                       [ 483,   93, 6408],
                       [ 477,   93, 6329],
                       [ 477,  104, 6269],
                       [ 483,  105, 6117],
                       [ 479,   93, 6079]]), 
               array([[478, 105, 366], ## 50000
                       [478, 104, 361],
                       [477, 105, 335],
                       [478,  93, 333],
                       [484,  93, 329],
                       [483,  93, 328],
                       [484,  94, 328],
                       [477,  93, 324],
                       [479,  93, 315],
                       [477, 104, 314]]), 
               array([[478, 104, 366],
                       [478, 105, 362],
                       [478,  93, 349],
                       [477, 105, 344],
                       [484,  93, 321],
                       [483,  93, 319],
                       [477,  93, 317],
                       [479,  93, 310],
                       [477, 104, 305],
                       [483, 105, 297]]), 
               array([[478, 104,  43], ## 5000
                       [483,  93,  42],
                       [478,  93,  37],
                       [478, 106,  36],
                       [484,  94,  35],
                       [476, 105,  34],
                       [477,  93,  33],
                       [478, 105,  32],
                       [477, 105,  31],
                       [484,  92,  31]]), 
               array([[420, 112,   9], ## 5000 using psi weighting
                       [429, 118,   8],
                       [436, 110,   8],
                       [436, 111,   8],
                       [445, 114,   8],
                       [445, 111,   8],
                       [429,  94,   8],
                       [438, 116,   7],
                       [406, 112,   7],
                       [446,  93,   6]]), 
               array([[436,  93,  57], ## 50000 using psi weighting
                       [436, 107,  56],
                       [437,  94,  55],
                       [436, 105,  55],
                       [436,  94,  54],
                       [436, 115,  53],
                       [437, 107,  53],
                       [446, 106,  52],
                       [447,  94,  51],
                       [435, 108,  50]]), 
               array([[173, 155,  27], ## 50000 using psi weighting and manually labelled bad examples
                       [155, 172,  24], ##[[435 436 436 436 436 436 437 437 446 447 476 477 477 477 478 478 478 478 479 483 483 484 484 484]
                       [166, 177,  57], ## [107  92  93 104 106 114  93 106 105  93 104  92 103 104  92 103 104 105 92  92 104  91  92  93]]    
                       [157, 147,  22],
                       [158, 169,  21],
                       [142, 156,  20],
                       [173, 163,  20],
                       [156, 143,  20],
                       [166, 148,  20],
                       [166, 155,  20]]), 
               array([[171, 153,  24], ## 50000 using psi weighting and 1000 random examples
                       [169, 156,  24],
                       [163, 173,  22],
                       [166, 155,  21],
                       [167, 152,  20],
                       [195, 183,  20],
                       [165, 175,  19],
                       [155, 173,  19],
                       [153, 167,  19],
                       [155, 166,  19]])]

# <codecell>

figure(); plot(jumpLocations[sortedJumps[:2670, 0], sortedJumps[:2670, 1]])

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
    def setPixmap(self, pixmap) :
        if pixmap.width() > self.width() :
<<<<<<< HEAD
            super(ImageLabel, self).setPixmap(pixmap)
=======
            super(ImageLabel, self).setPixmap(pixmap.scaledToWidth(self.width()))
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        else :
            super(ImageLabel, self).setPixmap(pixmap)
        
    def resizeEvent(self, event) :
        if self.pixmap() != None :
            if self.pixmap().width() > self.width() :
                self.setPixmap(self.pixmap().scaledToWidth(self.width()))
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.setWindowTitle("Validate Jumps")
        self.resize(1280, 720)
        
        self.delta = 1
        
        self.textureTimer = QtCore.QTimer(self)
        self.textureTimer.setInterval(1000/30)
        self.textureTimer.start()
        self.textureTimer.timeout.connect(self.renderOneFrame)
        self.visualizeMovie = False
        self.currentVisFrame = 0
<<<<<<< HEAD
        self.showMatted = False
=======
        self.showMatted = True
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
        self.messageShowed = False
        
        self.labelJumpStyle = "QLabel {border: 1px solid black; background: #aa0000; color: white; padding-left: 5px; padding-right: 5px;}"
        self.labelNoJumpStyle = "QLabel {border: 1px solid gray; background: #eeeeee; color: black; padding-left: 5px; padding-right: 5px;}"
        
        self.DESIRED_EXTRA_FRAMES = 45
        self.additionalFrames = np.copy(self.DESIRED_EXTRA_FRAMES)
        
        self.validatedJumps = -np.ones(distanceMatrix.shape, dtype=int)
        
        if os.path.isfile(dataPath+dataSet+"validatedJumps.npy") :
            self.validatedJumps = np.load(dataPath+dataSet+"validatedJumps.npy")
            
        self.validatedPairs = list(np.argwhere(self.validatedJumps != -1))
        
        self.setLabelledFramesListTable()
        
        self.pairsToValidate = []#list(np.copy(best10Jumps[-1]))
        
        self.getNewPair()
        
        self.setFocus()
        
    def setRenderFps(self, value) :
        self.textureTimer.setInterval(1000/value)
        
    def finishedSettingFps(self) :
        self.setFocus()
        
    def setTextureFrame(self, im, alpha):
        im = np.ascontiguousarray(im)
        
        if alpha :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
        else :
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            
        self.movieLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        
    def renderOneFrame(self):
        if self.visualizeMovie:
            try:
                self.movie
                self.currentVisFrame
            except AttributeError:
                return
            else:
#                 if self.currentVisFrame < 0 or self.currentVisFrame >= len(self.movie) :
#                     self.currentVisFrame = 0
                self.currentVisFrame += self.delta
                if self.delta > 0 and self.currentVisFrame  >= len(self.movie) :
                    self.delta = -1
                    self.currentVisFrame = len(self.movie) - 2
                elif self.delta < 0 and self.currentVisFrame < 0 :
                    self.delta = 1
                    self.currentVisFrame = 1
                    
                frameIdx = self.movie[self.currentVisFrame]
                
                if self.showMatted and frameIdx < len(mattes) and os.path.isfile(mattes[frameIdx]) :
                    alphaMatte = cv2.cvtColor(cv2.imread(mattes[frameIdx]), cv2.COLOR_BGR2GRAY)
                    alphaMatte = np.reshape(alphaMatte, np.hstack((alphaMatte.shape[0:2], 1)))
                    self.setTextureFrame(np.concatenate((cv2.imread(frames[frameIdx]), alphaMatte), axis=-1), True)
                else :
<<<<<<< HEAD
                    self.setTextureFrame(cv2.cvtColor(cv2.imread(frames[frameIdx]), cv2.COLOR_BGR2RGB), False)
=======
                    self.setTextureFrame(cv2.imread(frames[frameIdx]), False)
>>>>>>> fe1b005d2ec4d7eb0bc61da731ff4fa25b905e36
                    
                if self.currentVisFrame == self.additionalFrames+1 or self.currentVisFrame == self.additionalFrames+2 :
                    self.movieInfo.setStyleSheet(self.labelJumpStyle)
                else :
                    self.movieInfo.setStyleSheet(self.labelNoJumpStyle)
                    
                self.movieInfo.setText(np.str_(self.currentVisFrame) + " from " + np.str_(frameIdx))
                
#                 self.currentVisFrame = np.mod(self.currentVisFrame+1, len(self.movie))
        
    def setLabelledFramesListTable(self) :
        
        if len(self.validatedPairs) > 0 :
            self.labelledFramesListTable.setRowCount(len(self.validatedPairs))
            
            for pairing, i in zip(self.validatedPairs, arange(len(self.validatedPairs))) :
                self.labelledFramesListTable.setItem(i, 0, QtGui.QTableWidgetItem(np.string_(pairing[0])))
                self.labelledFramesListTable.setItem(i, 1, QtGui.QTableWidgetItem(np.string_(pairing[1])))
                self.labelledFramesListTable.setItem(i, 2, QtGui.QTableWidgetItem(np.string_(self.validatedJumps[pairing[0], pairing[1]] == 1)))
        else :
            self.labelledFramesListTable.setRowCount(0)
            
    def getNewPair(self) :
        if len(self.pairsToValidate) > 0 :
            self.labelledFramesListTable.setEnabled(False)
            self.frame1Idx = self.pairsToValidate[0][0]
            self.frame2Idx = self.pairsToValidate[0][1]
            
            del self.pairsToValidate[0]
            
            self.setMovie()
        else :
            if not self.messageShowed :
                QtGui.QMessageBox.information(self, "No more pairs to validate", "There are no more jumps to validate left.")
                self.messageShowed = True
            self.labelledFramesListTable.setEnabled(True)
        
    def setMovie(self) :
        self.visualizeMovie = False
        
        self.additionalFrames = np.min((self.frame1Idx, numFrames-self.frame2Idx, self.DESIRED_EXTRA_FRAMES))
        
        self.movie = np.arange(self.frame1Idx-self.additionalFrames, self.frame1Idx+1, dtype=int)
        self.movie = np.concatenate((self.movie, np.arange(self.frame2Idx+1, self.frame2Idx+self.additionalFrames+2, dtype=int)))
        
#         print self.movie, self.frame1Idx, self.frame2Idx
        
        self.visualizeMovie = True
        
    def existingPairSelected(self) :
        selectedRow = self.labelledFramesListTable.currentRow()
        if selectedRow >= 0 and selectedRow < len(self.validatedPairs):
            self.frame1Idx = self.validatedPairs[selectedRow][0]
            self.frame2Idx = self.validatedPairs[selectedRow][1]
            
            self.setMovie()
        
    def keyPressEvent(self, e) :
        if e.key() == QtCore.Qt.Key_Down : ## jump is invalid
            self.validateFramePair(0)
        elif e.key() == QtCore.Qt.Key_Up : ## jump is valid
            self.validateFramePair(1)
            
    def validateFramePair(self, isValid) :
        if self.labelledFramesListTable.currentRow() >= 0 : ## I'm modifying validation of existing pair
            selectedPair = self.validatedPairs[self.labelledFramesListTable.currentRow()]
#             print selectedPair
        else :
            selectedPair = np.array([self.frame1Idx, self.frame2Idx])
#             print self.frame1Idx, self.frame2Idx, isValid

        if self.validatedJumps[selectedPair[0], selectedPair[1]] == -1 :
            self.validatedPairs.append(selectedPair)

        if self.validatedJumps[selectedPair[0], selectedPair[1]] != -1 and self.validatedJumps[selectedPair[0], selectedPair[1]] != isValid :
            if self.validatedJumps[selectedPair[0], selectedPair[1]] == 0 :
                validation = "invalid"
            else :
                validation = "valid"
                
            doOverride = QtGui.QMessageBox.question(self, "Override?", 
                                                     "The current jump has already been labelled as " + validation + ".\nOverride?", 
                                                     buttons=QtGui.QMessageBox.Yes | QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
            
            if doOverride :
                self.validatedJumps[selectedPair[0], selectedPair[1]] = isValid
        else :
            self.validatedJumps[selectedPair[0], selectedPair[1]] = isValid
            
        self.labelledFramesListTable.clearSelection()
        self.labelledFramesListTable.setCurrentCell(-1, -1)
        self.setLabelledFramesListTable()
        
        self.getNewPair()
    
    def closeEvent(self, event) :
        print "closing"
        sys.stdout.flush()
        np.save(dataPath+dataSet+"validatedJumps.npy", self.validatedJumps)
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.movieLabel = ImageLabel("Movie")
        self.movieLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.movieInfo = QtGui.QLabel("Info text")
        self.movieInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.labelledFramesListTable = QtGui.QTableWidget(0, 3)
        self.labelledFramesListTable.horizontalHeader().setStretchLastSection(True)
        self.labelledFramesListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Index f1"))
        self.labelledFramesListTable.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Index f2"))
        self.labelledFramesListTable.setHorizontalHeaderItem(2, QtGui.QTableWidgetItem("isValid"))
        self.labelledFramesListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
#         self.labelledFramesListTable.setItem(0, 0, QtGui.QTableWidgetItem("No Labelled Frames"))
        self.labelledFramesListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.labelledFramesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.labelledFramesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.labelledFramesListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.labelledFramesListTable.setFocusPolicy(QtCore.Qt.NoFocus)
        self.labelledFramesListTable.setEnabled(False)
        
        self.renderFpsSpinBox = QtGui.QSpinBox()
        self.renderFpsSpinBox.setRange(1, 60)
        self.renderFpsSpinBox.setSingleStep(1)
        self.renderFpsSpinBox.setValue(30)
        
        
        ## SIGNALS ##
        
        self.labelledFramesListTable.cellPressed.connect(self.existingPairSelected)
        self.renderFpsSpinBox.valueChanged[int].connect(self.setRenderFps)
        self.renderFpsSpinBox.editingFinished.connect(self.finishedSettingFps)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.labelledFramesListTable)
        controlsLayout.addWidget(self.renderFpsSpinBox)
        movieLayout = QtGui.QVBoxLayout()
        movieLayout.addWidget(self.movieLabel)
        movieLayout.addWidget(self.movieInfo)
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(movieLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()

app.exec_()

# <codecell>

print window.movie
print window.frame1Idx
print window.frame2Idx
print np.min((window.frame1Idx, numFrames-window.frame2Idx, window.additionalFrames))

# <codecell>

gwv.showCustomGraph(window.validatedJumps)

# <codecell>

print window.validatedPairs
print window.movie

# <codecell>

np.save(dataPath+dataSet+"validatedJumps.npy", window.validatedJumps)

# <codecell>

print best10Jumps

