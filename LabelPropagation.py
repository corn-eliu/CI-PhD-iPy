# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab 

from PIL import Image
from PySide import QtCore, QtGui

import numpy as np
import scipy as sp
import scipy.io as sio
import cv2
import cv
import glob
import time
import sys
import os
import re
from scipy import ndimage
from scipy import stats

from tsne import tsne

# from _emd import emd

import GraphWithValues as gwv
import VideoTexturesUtils as vtu
import ComputeGridFeatures as cgf

# dataFolder = "/home/ilisescu/PhD/data/"
dataFolder = "/media/ilisescu/Data1/PhD/data/"

app = QtGui.QApplication(sys.argv)

# <codecell>

## read frames from sequence of images
# sampleData = "pendulum/"
sampleData = "ribbon2/"
# sampleData = "flag_blender/"
# sampleData = "ribbon1_matted/"
# sampleData = "little_palm1_cropped/"
# sampleData = "ballAnimation/"
# sampleData = "eu_flag_ph_left/"
outputData = dataFolder+sampleData

## Find pngs in sample data
frames = glob.glob(dataFolder + sampleData + "frame-*.png")
mattes = glob.glob(dataFolder + sampleData + "matte-*.png")
frames = np.sort(frames)
mattes = np.sort(mattes)#[0:len(frames)-10]
numFrames = len(frames)
frameSize = cv2.imread(frames[0]).shape
print numFrames, len(mattes)

# <codecell>

## compute features for image
blocksPerWidth = 16
blocksPerHeight = 16
subDivisions = blocksPerWidth*blocksPerHeight

## given block sizes and img sizes build indices representing each block
imageSize = np.array(cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)).shape
stencils = cgf.stencil2D(blocksPerWidth, blocksPerHeight, imageSize)

features = cgf.histFgFeatures(stencils, subDivisions, frames, mattes)
figure(); imshow(features.T, interpolation='nearest')

# <codecell>

figure(); imshow(features[462, :].reshape((blocksPerHeight, blocksPerWidth)), interpolation='nearest')

# <codecell>

figure(); 
xlim(0, blocksPerHeight*blocksPerWidth);
bar(xrange(blocksPerHeight*blocksPerWidth), features[462, :]/np.sum(features[462, :]));

# <codecell>

print features[0, :]
print np.sum(features[0, :]/np.linalg.norm(features[0, :]))
print np.sum(features[0, :]/((1280/16)*(720/16)))
np.sum(features[0, :])/np.sum(features[99, :])

# <codecell>

sio.savemat("features.mat", {"features":features})

# <codecell>

hist2demdDistMat = np.array(sio.loadmat(dataFolder + sampleData + "hist2demd_32x48_distMat.mat")['distMat'], dtype=float)
print hist2demdDistMat

# <codecell>

figure(); imshow(np.array(sio.loadmat(dataFolder + sampleData + "hist2demd_32x48_distMat.mat")['distMat'], dtype=float), interpolation='nearest')

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

optimizedDistMat = estimateFutureCost(0.999, 2.0, vtu.filterDistanceMatrix(hist2demdDistMat/np.max(hist2demdDistMat), 4, True), np.ones((1200, 1200)))

# <codecell>

np.save(dataFolder + sampleData + "proc_hist2demd_32x48_distMat.npy", optimizedDistMat)

# <codecell>

figure(); imshow(vtu.filterDistanceMatrix(hist2demdDistMat/np.max(hist2demdDistMat), 4, True), interpolation='nearest')
figure(); imshow(optimizedDistMat, interpolation='nearest')
figure(); imshow(np.load(dataFolder + sampleData + "proc_distMat.npy"), interpolation='nearest')

# <codecell>

tmp3248 = np.load(dataFolder + sampleData + "proc_hist2demd_32x48_distMat.npy")
tmp1616 = np.load(dataFolder + sampleData + "proc_hist2demd_16x16_distMat.npy")
figure(); imshow((np.abs(tmp3248/np.max(tmp3248)-tmp1616/np.max(tmp1616)))[500:600, 500:600], interpolation='nearest')
figure(); imshow((tmp3248/np.max(tmp3248))[500:600, 500:600], interpolation='nearest')
figure(); imshow((tmp1616/np.max(tmp1616))[500:600, 500:600], interpolation='nearest')
print np.max((np.abs(tmp3248/np.max(tmp3248)-tmp1616/np.max(tmp1616))))

# <codecell>

## load precomputed distance matrix and filter for label propagation
# name = "vanilla_distMat"
# name = "histcos_16x16_distMat"
# name = "hist2demd_32x48_distMat"
# name = "hist2demd_16x16_distMat"
# name = "semantics_hog_rand50_distMat"
# name = "semantics_hog_rand50_encodedfirst_distMat"
# name = "semantics_hog_rand50_encodedlast_distMat"
# name = "semantics_hog_set50_distMat"
# name = "semantics_hog_set50_encodedlast_distMat"
# name = "semantics_hog_set150_distMat"
# name = "semantics_hog_L2_set150_distMat"
# name = "appearance_hog_rand150_distMat"
# name = "appearance_hog_L2_rand150_distMat"
# name = "appearance_hog_set150_distMat"
name = "appearance_hog_L2_set150_distMat"
distanceMatrix = np.array(np.load(outputData + name + ".npy"), dtype=np.float)
distanceMatrix /= np.max(distanceMatrix)
if True :
    distMat = vtu.filterDistanceMatrix(distanceMatrix, 2, True)
else :
    distMat = np.copy(distanceMatrix)
figure(); imshow(distMat, interpolation='nearest')
## save for matlab to compute isomap
# sio.savemat(name + ".mat", {"distMat":distMat})
distances = np.array(np.copy(distMat), dtype=float)

# <codecell>

print distMat.shape
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

close('all')

# <codecell>

# distances = np.copy(distanceMatrix)
#/np.max(distMat)
# distances = np.copy(distMat[1:distMat.shape[1], 0:-1])
# distances = np.copy(distMatFut)

if False :
    ## use dotstar
    numClasses = 2
    ## init labeled points
    labelledPoints = np.array([[9], [21]])-1
    fl = np.zeros((len(labelledPoints), numClasses))
    fl = np.eye(numClasses)
else :
    ## use ribbon2
    numClasses = 4
    ## init labeled points
    labelledPoints = np.array([[122], [501], [838], [1106]]) -4
    fl = np.eye(numClasses)
    
#     labelledPoints = np.array([[22, 122, 222], [281, 501, 721], [754, 838, 922], [956, 1106, 1256]]) -4
#     fl = np.zeros((np.prod(labelledPoints.shape), numClasses))
#     fl[0:3, 0] = 1
#     fl[3:6, 1] = 1
#     fl[6:9, 2] = 1
#     fl[9:, 3] = 1

    initPoints = np.array([122, 501, 838, 1106]) -4
    extraPoints = 16
    labelledPoints = np.zeros((numClasses, extraPoints+1), dtype=np.int)
    for i in xrange(0, len(initPoints)) :
        labelledPoints[i, :] = range(initPoints[i]-extraPoints/2, initPoints[i]+extraPoints/2+1)

    fl = np.zeros((np.prod(labelledPoints.shape), numClasses))
    for i in xrange(0, numClasses) :
        fl[i*(extraPoints+1):(i+1)*(extraPoints+1), i] = 1
    
print numClasses, labelledPoints
print fl

## order w to have labeled nodes at the top-left corner
flatLabelled = np.ndarray.flatten(labelledPoints)

# <codecell>

## do label propagation as zhu 2003
orderedDist = np.copy(distances)
sortedFlatLabelled = flatLabelled[np.argsort(flatLabelled)]
sortedFl = fl[np.argsort(flatLabelled), :]
print sortedFlatLabelled
for i in xrange(0, len(sortedFlatLabelled)) :
    #shift sortedFlatLabelled[i]-th row up to i-th row and adapt remaining rows
    tmp = np.copy(orderedDist)
    orderedDist[i, :] = tmp[sortedFlatLabelled[i], :]
    orderedDist[i+1:, :] = np.vstack((tmp[i:sortedFlatLabelled[i], :], tmp[sortedFlatLabelled[i]+1:, :]))
    #shift sortedFlatLabelled[i]-th column left to i-th column and adapt remaining columns
    tmp = np.copy(orderedDist)
    orderedDist[:, i] = tmp[:, sortedFlatLabelled[i]]
    orderedDist[:, i+1:] = np.hstack((tmp[:, i:sortedFlatLabelled[i]], tmp[:, sortedFlatLabelled[i]+1:]))
#     print len(sortedFlatLabelled)+sortedFlatLabelled[i]
    
gwv.showCustomGraph(distances)
gwv.showCustomGraph(orderedDist)

## compute weights
w, cumW = vtu.getProbabilities(orderedDist, 0.06, None, False)
gwv.showCustomGraph(w)
# gwv.showCustomGraph(cumW)

l = len(sortedFlatLabelled)
n = orderedDist.shape[0]
## compute graph laplacian
L = np.diag(np.sum(w, axis=0)) - w
# gwv.showCustomGraph(L)

## propagate labels
fu = np.dot(np.dot(-np.linalg.inv(L[l:, l:]), L[l:, 0:l]), sortedFl)

## use class mass normalization to normalize label probabilities
q = np.sum(sortedFl)+1
fu_CMN = fu*(np.ones(fu.shape)*(q/np.sum(fu)))

# <codecell>

## add labeled points to propagated labels (as labelProbs) and plot
print fu.shape
# print fu_CMN

print flatLabelled
numClasses = fl.shape[-1]

if False :
################ this bit doesn't work anymore since using #################
    ## add labeled frames to fu and plot
    labelProbs = np.array(fu[0:flatLabelled[0]])
    print labelProbs.shape
    for i in xrange(1, len(flatLabelled)) :
    #     print flatLabelled[i]+i, flatLabelled[i+1]-i
    #     print fu[flatLabelled[i]+i:flatLabelled[i+1]-i, :]
    
        labelProbs = np.vstack((labelProbs, fl[i-1, :]))
        print labelProbs.shape, flatLabelled[i-1]-(i-1), flatLabelled[i]-i
        labelProbs = np.vstack((labelProbs, fu[flatLabelled[i-1]-(i-1):flatLabelled[i]-i, :]))
        print labelProbs.shape
        
    
    
    labelProbs = np.vstack((labelProbs, fl[-1, :]))
    labelProbs = np.vstack((labelProbs, fu[flatLabelled[-1]-len(flatLabelled)+1:, :]))
    # labelProbs = labelProbs[1:, :]
    print labelProbs, labelProbs.shape
else :
    labelProbs = np.copy(np.array(fu))
    print labelProbs.shape
    for frame, i in zip(sortedFlatLabelled, np.arange(len(sortedFlatLabelled))) :
        labelProbs = np.vstack((labelProbs[0:frame, :], sortedFl[i, :], labelProbs[frame:, :]))
        
    print labelProbs.shape

clrs = ['r', 'g', 'b', 'm']
fig1 = figure()
xlabel('all points')
fig2 = figure()
xlabel('only unlabeled')
fig3 = figure()
xlabel('only unlabeled + CMN')

for i in xrange(0, numClasses) :
    figure(fig1.number); plot(labelProbs[:, i], clrs[i])
    figure(fig2.number); plot(fu[:, i], clrs[i])
    figure(fig3.number); plot(fu_CMN[:, i], clrs[i])
    
    for node in labelledPoints[i] :
        figure(fig1.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])
        figure(fig2.number); plot(np.repeat(node, 2), [0, 1.1], clrs[i])
        figure(fig3.number); plot(np.repeat(node, 2), [0, np.max(fu_CMN)], clrs[i])

# <codecell>

print labelProbs.shape
print np.sum(labelProbs)

# <codecell>

## save label propagation for visualizing within videotextgui
# np.save(outputData + "labeledPoints.npy", labeledPoints)
# np.save(outputData + "labelProbs.npy", labelProbs)
np.save(outputData + "l2dist_guiex60_mult0.045_labels.npy", {"labeledPoints": np.array(labelledPoints), "labelProbs": labelProbs})
## save label propagation for visualizing isomaps in matlab
# sio.savemat("labeledPoints.mat", {"labeledPoints":labeledPoints})
# sio.savemat("l2dist_mult0.05_labelProbs.mat", {"labelProbs":labelProbs})

# <codecell>

## save label propagation for visualizing within videotextgui
# np.save(outputData + "labeledPoints.npy", labeledPoints)
# np.save(outputData + "labelProbs.npy", labelProbs)
np.save(outputData + "learned_appereance_hog_set150_mult0.06_labels.npy", {"labeledPoints": np.array(labelledPoints), "labelProbs": labelProbs})
## save label propagation for visualizing isomaps in matlab
# sio.savemat("labeledPoints.mat", {"labeledPoints":labeledPoints})
# sio.savemat("l2dist_mult0.05_labelProbs.mat", {"labelProbs":labelProbs})

# <codecell>

## save label propagation for visualizing within videotextgui
# np.save(outputData + "labeledPoints.npy", labeledPoints)
# np.save(outputData + "labelProbs.npy", labelProbs)
np.save(outputData + "l2dist_mult0.05_labels.npy", {"labeledPoints": labeledPoints, "labelProbs": labelProbs})
## save label propagation for visualizing isomaps in matlab
sio.savemat("labeledPoints.mat", {"labeledPoints":labeledPoints})
sio.savemat("l2dist_mult0.05_labelProbs.mat", {"labelProbs":labelProbs})

# <codecell>

## load isomap from .mat and save as .npy
## it automatically loads the output from PhD/MATLAB so run this after running ComputeIsomap.m
mapPoints = sio.loadmat("../MATLAB/mapPoints.mat")["mapPoints"]
tmp = sio.loadmat("../MATLAB/predictedLabels.mat")["predictedLabels"]
predictedLabels = []
for i in xrange(0,len(np.ndarray.flatten(tmp)[0])) :
    predictedLabels.append(np.ndarray.flatten(np.ndarray.flatten(tmp)[0][i])-1)

np.save(outputData + "l2dist_mult0.05_isomap.npy", {"mapPoints": mapPoints, "predictedLabels": predictedLabels})

# <codecell>

## check that orderedDist is still symmetric
print np.sum(orderedDist[4, :] - orderedDist[:, 4])
print np.sum(orderedDist[10, :] - orderedDist[:, 10])
print np.sum(orderedDist[250, :] - orderedDist[:, 250])
## check that orderedDist has been ordered the right way
print orderedDist[0, 0:50]
print distances[117, list(flatLabelled)]
print distances[117, 0:50]
print 
print orderedDist[3, 0:50]
print distances[496, list(flatLabelled)]
print distances[496, 0:50]
print 
print orderedDist[10, 0:50]
print distances[1102, list(flatLabelled)]
print distances[1102, 0:50]

# <codecell>

## compute tsne representation for given distMat

Y = tsne(np.zeros((distMat.shape[0], 10)), distMat)

# <codecell>

## show result of tsne
print labelProbs.shape
labels = np.argmax(labelProbs, axis=-1)
reds = np.argwhere(labels == 0)
greens = np.argwhere(labels == 1)
blues = np.argwhere(labels == 2)
magentas = np.argwhere(labels == 3)
# labels = np.loadtxt("mnist2500_labels.txt");
print labels, labels.shape
print reds.shape, greens.shape, blues.shape, magentas.shape

## normalize and fit to interval [0, 1]
Y = (Y-np.min(Y))/(np.max(Y)-np.min(Y))

figure();
scatter(Y[reds,0], Y[reds,1], 20, '#ff0000');
scatter(Y[greens,0], Y[greens,1], 20, '#00ff00');
scatter(Y[blues,0], Y[blues,1], 20, '#0000ff');
scatter(Y[magentas,0], Y[magentas,1], 20, '#ff00ff');

## save results to use with videotextgui
np.save(outputData + "l2dist_mult0.05_tsnemap.npy", {"mapPoints": Y.T, "predictedLabels": [np.ndarray.flatten(reds), np.ndarray.flatten(greens), np.ndarray.flatten(blues), np.ndarray.flatten(magentas)]})

# <codecell>

## show stencil
im = np.zeros(img.shape)
im[stencils[5][0], stencils[5][1]] = 1
figure(); imshow(im)

# <codecell>

## show features
gwv.showCustomGraph(features[717, :].reshape((blocksPerDim, blocksPerDim)))
gwv.showCustomGraph(features[1165, :].reshape((blocksPerDim, blocksPerDim)))
gwv.showCustomGraph(features[1166, :].reshape((blocksPerDim, blocksPerDim)))
print np.dot(features[717, :], features[1166, :])
print np.dot(features[1165, :], features[1166, :])

# <codecell>

## compute the distance matrix where distance is dot product between feature vectors
distanceMatrix = np.ones((numFrames, numFrames))

for r in xrange(0, numFrames) :
    for c in xrange(r+1, numFrames) :
        distanceMatrix[r, c] = distanceMatrix[c, r] = np.dot(features[r, :], features[c, :])
    print r, 

distanceMatrix = 1 - distanceMatrix
figure(); imshow(distanceMatrix, interpolation='nearest')

# <codecell>

## compute 1D emd
def distance(f1, f2):
#     return np.sqrt( (f1.x - f2.x)**2  + (f1.y - f2.y)**2 + (f1.z - f2.z)**2 )
    return np.sqrt((f1 - f2)**2)#np.sqrt( (f1[0] - f2[0])**2  + (f1[1] - f2[1])**2 + (f1[2] - f2[2])**2 )

print emd((list(features[1165, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance)
print emd((list(features[717, :]), list(arange(0.0, subDivisions))), (list(features[1166, :]), list(arange(0.0, subDivisions))), distance)

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text, parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
    def setPixmap(self, pixmap) :
        if pixmap.width() > self.width() :
            super(ImageLabel, self).setPixmap(pixmap.scaledToWidth(self.width()))
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
        
        self.setWindowTitle("Semantic Labelling")
        self.resize(1280, 720)
        
        self.labelledFrames = []
#         self.labelledPairs.append([1234, 234, 0.5])
#         self.setLabelledFramesListTable()

        self.RANDOM_CHOICE = 0
        self.SET_CHOICE = 1
        self.choiceMode = self.RANDOM_CHOICE
        
        self.choiceSet = arange(100)
        
        self.getNewFrame()
        
        self.setFocus()
        
    def setLabelledFramesListTable(self) :
        
        if len(self.labelledFrames) > 0 :
            self.labelledFramesListTable.setRowCount(len(self.labelledFrames))
            
            for labelledFrame, i in zip(self.labelledFrames, arange(len(self.labelledFrames))) :
                self.labelledFramesListTable.setItem(i, 0, QtGui.QTableWidgetItem(np.string_(labelledFrame[0])))
                self.labelledFramesListTable.setItem(i, 1, QtGui.QTableWidgetItem(np.string_(labelledFrame[1])))
        else :
            self.labelledFramesListTable.setRowCount(0)
    
    def getNewFrame(self) :
        if self.choiceMode == self.RANDOM_CHOICE :
            self.getNewRandomFrame()
        elif self.choiceMode == self.SET_CHOICE :
            self.getNewFrameFromSet()
    
    def getNewFrameFromSet(self) :
        self.frameIdx = np.random.choice(self.choiceSet)
        
        while len(self.labelledFrames) > 0 and self.frameIdx in np.array(self.labelledFrames)[:, 0] :
            print "stuck"
            self.frameIdx = np.random.choice(self.choiceSet)
        
        self.setFrameImage()
            
    def getNewRandomFrame(self) :
        self.frameIdx = np.random.randint(0, len(frames))
        
        while len(self.labelledFrames) > 0 and self.frameIdx in np.array(self.labelledFrames)[:, 0] :
            print "stuck"
            self.frameIdx = np.random.randint(0, len(frames))
        
        self.setFrameImage()
        
    def setFrameImage(self) :
        ## HACK ##
        im = np.ascontiguousarray(Image.open(frames[self.frameIdx]))
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
        self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
        self.frameInfo.setText(frames[self.frameIdx])
        
    def labelledFrameSelected(self) :
        selectedRow = self.labelledFramesListTable.currentRow()
        if selectedRow >= 0 and selectedRow < len(self.labelledFrames):
            ## HACK ##
            im = np.ascontiguousarray(Image.open(frames[self.labelledFrames[selectedRow][0]]))
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
            self.frameLabel.setPixmap(QtGui.QPixmap.fromImage(qim))
            self.frameInfo.setText(frames[self.labelledFrames[selectedRow][0]])
        
    def keyPressEvent(self, e) :
        if e.key() == e.key() >= QtCore.Qt.Key_0 and e.key() <= QtCore.Qt.Key_9 :
            self.setFrameLabel(np.mod(e.key()-int(QtCore.Qt.Key_0), int(QtCore.Qt.Key_9)))
        elif e.key() == QtCore.Qt.Key_Space : ## Get new frame
            self.getNewFrame()
            
    def setFrameLabel(self, label) :
        if self.labelledFramesListTable.currentRow() >= 0 : ## I'm modifying label of existing frame
            self.labelledFrames[self.labelledFramesListTable.currentRow()][1] = label
        else :
            self.labelledFrames.append([self.frameIdx, label])
            
        self.labelledFramesListTable.clearSelection()
        self.labelledFramesListTable.setCurrentCell(-1, -1)
        self.setLabelledFramesListTable()
        
        self.getNewFrame()
        
    def changeChoiceMode(self, index) :
        if index == self.RANDOM_CHOICE :
            self.choiceMode = self.RANDOM_CHOICE
            self.choiceSetInterval.setEnabled(False)
            self.choiceSetInterval.setVisible(False)
        elif index == self.SET_CHOICE :
            self.choiceMode = self.SET_CHOICE
            self.choiceSetInterval.setEnabled(True)
            self.choiceSetInterval.setVisible(True)
        
        self.getNewFrame()
        
    def changeChoiceSet(self) :
        choiceSetText = self.choiceSetInterval.text()
        interval = np.array(re.split("-|:", choiceSetText), dtype=int)
        if len(interval) == 2 :
            self.choiceSet = arange(interval[0], interval[1]+1)
            
        self.setFocus()
        self.getNewFrame()
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel("Frame")
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.labelledFramesListTable = QtGui.QTableWidget(0, 2)
        self.labelledFramesListTable.horizontalHeader().setStretchLastSection(True)
        self.labelledFramesListTable.setHorizontalHeaderItem(0, QtGui.QTableWidgetItem("Frame Index"))
        self.labelledFramesListTable.setHorizontalHeaderItem(1, QtGui.QTableWidgetItem("Label"))
        self.labelledFramesListTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
#         self.labelledFramesListTable.setItem(0, 0, QtGui.QTableWidgetItem("No Labelled Frames"))
        self.labelledFramesListTable.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.labelledFramesListTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.labelledFramesListTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.labelledFramesListTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.labelledFramesListTable.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.choiceModeComboBox = QtGui.QComboBox()
        self.choiceModeComboBox.addItem("Random")
        self.choiceModeComboBox.addItem("Set")
        self.choiceModeComboBox.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.choiceSetInterval = QtGui.QLineEdit()
        self.choiceSetInterval.setEnabled(False)
        self.choiceSetInterval.setVisible(False)
        
        
        ## SIGNALS ##
        
        self.labelledFramesListTable.cellPressed.connect(self.labelledFrameSelected)
        self.choiceModeComboBox.currentIndexChanged[int].connect(self.changeChoiceMode)
        self.choiceSetInterval.returnPressed.connect(self.changeChoiceSet)
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        controlsLayout = QtGui.QVBoxLayout()
        controlsLayout.addWidget(self.labelledFramesListTable)
        controlsLayout.addWidget(self.choiceModeComboBox)
        controlsLayout.addWidget(self.choiceSetInterval)
        frameLayout = QtGui.QVBoxLayout()
        frameLayout.addWidget(self.frameLabel)
        frameLayout.addWidget(self.frameInfo)
        mainLayout.addLayout(controlsLayout)
        mainLayout.addLayout(frameLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

# labelledFramesRand = np.copy(labelledFrames)
labelledFramesSet = np.copy(labelledFrames)

# <codecell>

labelledFrames = np.copy(labelledFramesRand)

# <codecell>

# labelledFrames = np.array(window.labelledFrames)
print labelledFrames
# labelledFrames = labelledFrames[]
# print len(np.argwhere(labelledFrames[:, 1]==0))

# <codecell>

labelledFrames = np.array([[ 119,    0],
 [ 270,    0],
 [ 178,    0],
 [  82,   0],
 [  46,    0],
 [ 243,    0],
 [ 256,    0],
 [  36,    1],
 [ 242,    0],
 [  30,    1],
 [  21,    1],
 [ 257,    0],
 [ 104,    0],
 [ 240,    0],
 [ 138,    0],
 [ 265,    0],
 [ 254,    0],
 [ 106,    0],
 [ 208,    0],
 [ 125,    0],
 [  69,    0],
 [ 102,    0],
 [ 249,    0],
 [ 720,    1],
 [ 494,    1],
 [ 430,    1],
 [ 335,    1],
 [ 692,    1],
 [ 440,    1],
 [ 357,    1],
 [ 616,    1],
 [ 587,    1],
 [ 287,    1],
 [ 688,    1],
 [ 437,    1],
 [ 753,    1],
 [ 358,    1],
 [ 482,    1],
 [ 671,    1],
 [ 644,    1],
 [ 887,    2],
 [ 879,    2],
 [ 930,    2],
 [ 837,    2],
 [ 871,    2],
 [ 856,    2],
 [ 866,    2],
 [ 902,    2],
 [ 890,    2],
 [ 772,    2],
 [ 888,    2],
 [ 759,    2],
 [ 891,    2],
 [ 765,    2],
 [ 835,    2],
 [ 933,    2],
 [ 872,    2],
 [ 807,    2],
 [ 915,    2],
 [ 905,    2],
 [1268,    3],
 [ 954,    3],
 [1212,    3],
 [1206,    3],
 [1045,    3],
 [1128,    3],
 [ 987,    3],
 [ 970,    3],
 [ 965,    3],
 [1124,    3],
 [1092,    3],
 [1049,    3],
 [ 960,    3],
 [ 946,    3],
 [1048,    3],
 [1227,    3],
 [1253,    3],
 [1233,    3],
 [1019,    3],
 [ 953,    3]])

# <codecell>

np.save(dataFolder + sampleData + "semantic_labels_gui_set60.npy", labelledFrames)

# <codecell>

labelledFrames = np.array(window.labelledFrames)
# np.save(dataFolder + sampleData + "semantic_labels_prop_rand50.npy", labelledFrames)

# <codecell>

## get flatLabelled and fl
# labelledFrames = np.array(window.labelledFrames)
# labelledFrames = np.load(dataFolder + sampleData + "semantic_labels_gui_set60.npy")
usedLabels = np.unique(np.ndarray.flatten(labelledFrames[:, 1]))
fl = np.zeros((len(labelledFrames), len(usedLabels)))
prevIdx = 0
flatLabelled = np.empty(0, dtype=int)
labelledPoints = []
print "used labels", usedLabels
for i in usedLabels :
    iLabels = labelledFrames[np.ndarray.flatten(np.argwhere(labelledFrames[:, 1] == i)), :][:, 0]
    labelledPoints.append(iLabels)
    flatLabelled = np.concatenate((flatLabelled, iLabels))
    fl[prevIdx:prevIdx+len(iLabels), i] = 1
    prevIdx += len(iLabels)
    print i, iLabels
print flatLabelled
print fl

