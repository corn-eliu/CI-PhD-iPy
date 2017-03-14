# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

from PIL import Image
from PySide import QtCore, QtGui

import numpy as np

import glob
import cv2
import os

app = QtGui.QApplication(sys.argv)

# <codecell>

# dataPath = "/home/ilisescu/PhD/data/"
dataPath = "/media/ilisescu/Data1/PhD/data/"

# dataSet = "candle1/"
dataSet = "candle2/subset_stabilized/"
dataSet = "candle3/stabilized/"

framePaths = np.sort(glob.glob(dataPath + dataSet + "frame*.png"))
numFrames = len(framePaths)
print numFrames
imageSize = np.array(Image.open(framePaths[0])).shape[0:2]

# <codecell>

doHighPass = False
highThresh = 10

doBlur = False
kSize = 11
sigma = 2.0

doLevels = True
logisticK = 0.7
midPoint = 15
x = arange(0, 256)
figure(); plot(x, 1/(1+exp(-logisticK*(x-midPoint))))

cropSize = (650, 1000)

# <codecell>

for frame in framePaths[105:106] :
    img = cv2.imread(frame)
    intensity = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    alphaMatte = np.copy(intensity).reshape((intensity.shape[0], intensity.shape[1], 1))
        
    if doHighPass :
        alphaMatte *= alphaMatte >= highThresh
    
    if doBlur :
        alphaMatte = cv2.GaussianBlur(alphaMatte, (kSize, kSize), sigma)
        
    if doLevels :
        ## uses logistic function to adjust levels
        alphaMatte = alphaMatte.astype(float)/(1+exp(-logisticK*(np.array(alphaMatte, dtype=float)-midPoint)))
        alphaMatte = np.array(alphaMatte, dtype=int)
    
    figure(); imshow(alphaMatte[:, :, 0])

# <codecell>

## click on figure to get anchor point for video texture

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(alphaMatte[:, :, 0])

def onclick(event):
    if event.inaxes :
        x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
        anchoredFrame = np.zeros((cropSize[0], cropSize[1], 3), dtype=uint8)
        anchorPointHeight = 20
        anchoredFrame[0:cropSize[0]-anchorPointHeight, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[int(y)+anchorPointHeight-cropSize[0]:int(y), 
                                                                                                    int(x)-cropSize[1]/2:int(x)+cropSize[1]/2, :]
        figure("Vis anchor point"); imshow(anchoredFrame)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

# <codecell>

x = 697.62
y = 672.035
anchoredFrame = np.zeros((500, 500))
anchorPointHeight = 20
anchoredFrame[0:500-anchorPointHeight, :] = alphaMatte[int(y)+anchorPointHeight-500:int(y), int(x)-250:int(x)+250, 0]
figure(); imshow(anchoredFrame)

# <codecell>

Image.fromarray(np.array(np.concatenate((np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), alphaMatte), axis=-1), dtype=np.uint8)).save("testImg.png")

# <codecell>

class ImageLabel(QtGui.QLabel) :
    
    def __init__(self, text="", parent=None):
        super(ImageLabel, self).__init__(text, parent)
        
        self.image = None
        self.color = QtGui.QColor(QtCore.Qt.black)
        
    def setImage(self, image) : 
        self.image = image.copy()
        self.setMinimumSize(self.image.size())
        self.update()
        
    def setBackgroundColor(self, color) :
        self.color = color
        
    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QtGui.QPainter(self)
        
        if self.image != None :
            upperLeft = ((self.width()-self.image.width())/2, (self.height()-self.image.height())/2)
            ## draw background
            painter.setBrush(QtGui.QBrush(self.color))
            painter.drawRect(QtCore.QRect(upperLeft[0], upperLeft[1], self.image.width(), self.image.height()))
            
            ## draw image
            painter.drawImage(QtCore.QPoint(upperLeft[0], upperLeft[1]), self.image)
                
class Window(QtGui.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.setWindowTitle("Segmentation and Cropping")
        self.resize(1700, 900)
        
        self.frameIdx = 0
        self.setFrameImage()
        
        self.settingAnchorPoint = False
        self.prevMousePosition = QtCore.QPoint(0, 0)
        
        
        self.setFocus() 
        
    def setFrameImage(self) :
        ## return rgba but for whatever reason it needs to be bgra for qt to display it properly
        im = self.getMattedImage()
        if self.doCropBox.isChecked() :
            im = self.getCroppedImage(im)
        im = np.ascontiguousarray(im[:, :, [2, 1, 0, 3]])
        
        ## HACK ##
        qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
        self.frameLabel.setImage(qim)
        self.frameInfo.setText(framePaths[self.frameIdx])
        
    def getMattedImage(self) :
        ## returns rgba
        img = cv2.imread(framePaths[self.frameIdx])
        intensity = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        alphaMatte = np.copy(intensity).reshape((intensity.shape[0], intensity.shape[1], 1))

        if self.doHighPassFilterBox.isChecked() :
            alphaMatte *= alphaMatte >= self.highPassThreshSpinBox.value()

        if self.doBlurBox.isChecked() :
            alphaMatte = cv2.GaussianBlur(alphaMatte, (self.blurSizeSpinBox.value(), 
                                                       self.blurSizeSpinBox.value()), 
                                          self.blurSigmaSpinBox.value()).reshape((intensity.shape[0], intensity.shape[1], 1))

        if self.doLevelsBox.isChecked() :
            ## uses logistic function to adjust levels
            alphaMatte = alphaMatte.astype(float)/(1+exp(-self.levelsSteepnessSpinBox.value()*(np.array(alphaMatte, dtype=float)-self.levelsMidpointSpinBox.value())))
        
        alphaMatte = np.array(alphaMatte, dtype=uint8)
        
        return np.concatenate((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alphaMatte), axis=-1)
    
    def getCroppedImage(self, img) :
        width = self.cropWidthSpinBox.value()
        height = self.cropHeightSpinBox.value()
        imageH, imageW = imageSize
        centerX = self.cropCenterXSpinBox.value()
        centerY = self.cropCenterYSpinBox.value()
        anchorX = self.cropAnchorXSpinBox.value()
        anchorY = self.cropAnchorYSpinBox.value()
        
        cropped = np.zeros((height, width, img.shape[-1]), dtype=uint8)

#         print np.max((0, anchorY-centerY)), np.min((imageH, anchorY)) ## row range from image
#         print np.max((0, anchorX-centerX)), np.min((imageW, anchorX+(width-centerX))) ## col range from image
#         print 
#         patchH = np.min((imageH, anchorY)) - np.max((0, anchorY-centerY))
#         patchW = np.min((imageW, anchorX+(width-centerX))) - np.max((0, anchorX-centerX))
#         print patchH, patchW
#         print
#         print centerY-patchH, centerY ## row range to cropped
#         print (width-patchW)/2, (width-patchW)/2+patchW  ## col range to cropped
#         print 
        
        imgPatch = img[np.max((0, anchorY-centerY)):np.min((imageH, anchorY)), np.max((0, anchorX-centerX)):np.min((imageW, anchorX+(width-centerX))), :]
        
        cropped[centerY-imgPatch.shape[0]:centerY, (width-imgPatch.shape[1])/2:(width-imgPatch.shape[1])/2+imgPatch.shape[1], :] = imgPatch
        
        return cropped
        
    def changeFrame(self, idx) :
        self.frameIdx = idx
        self.setFrameImage()
        
    def setBackgroundColor(self) :
        newBgColor = QtGui.QColorDialog.getColor(QtCore.Qt.black, self, "Choose Background Color")
        self.frameLabel.setBackgroundColor(newBgColor)
         
    def mousePressEvent(self, event):
        if self.doCropBox.isChecked() :
            self.settingAnchorPoint = True
            self.prevMousePosition = event.pos()
        else :
            sizeDiff = (self.frameLabel.size() - self.frameLabel.image.size())/2
            mousePos = event.pos() - self.frameLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
            if (mousePos.x() >= 0 and mousePos.y() >= 0 and 
                    mousePos.x() < self.frameLabel.image.width() and 
                    mousePos.y() < self.frameLabel.image.height()) :
                
                self.cropAnchorXSpinBox.setValue(mousePos.x())
                self.cropAnchorYSpinBox.setValue(mousePos.y())
                
        
    def mouseReleaseEvent(self, event) :
        self.settingAnchorPoint = False
        
    def mouseMoveEvent(self, event) :
        if self.settingAnchorPoint and self.frameLabel.image != None :
            sizeDiff = (self.frameLabel.size() - self.frameLabel.image.size())/2
            mousePos = event.pos() - self.frameLabel.pos() - QtCore.QPoint(sizeDiff.width(), sizeDiff.height())
            if (mousePos.x() >= 0 and mousePos.y() >= 0 and 
                    mousePos.x() < self.frameLabel.image.width() and 
                    mousePos.y() < self.frameLabel.image.height()) :
                
                deltaMove = self.prevMousePosition - event.pos()
                self.cropAnchorXSpinBox.setValue(self.cropAnchorXSpinBox.value()+deltaMove.x())
                self.cropAnchorYSpinBox.setValue(self.cropAnchorYSpinBox.value()+deltaMove.y())
#                 print "moving inside frame", deltaMove
            self.prevMousePosition = event.pos()
                
    def cropSizeChanged(self) :
        self.cropCenterXSpinBox.setRange(1, self.cropWidthSpinBox.value())
        self.cropCenterXSpinBox.setValue(self.cropWidthSpinBox.value()/2)
        self.cropCenterYSpinBox.setRange(1, self.cropHeightSpinBox.value())
        self.cropCenterYSpinBox.setValue(self.cropHeightSpinBox.value()-20)
        
    def saveFrames(self) :
        saveLoc = QtGui.QFileDialog.getExistingDirectory(self, "Select Save Location", dataPath+dataSet)
        if os.path.isdir(saveLoc) :
            for i in xrange(len(framePaths)) :
                self.frameIdx = i
                im = self.getMattedImage()
                if self.doCropBox.isChecked() :
                    im = self.getCroppedImage(im)
                
                Image.fromarray(np.array(im, dtype=np.uint8)).save(saveLoc+"/frame-{0:05}.png".format(i+1))
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.frameLabel = ImageLabel()
        self.frameLabel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameInfo = QtGui.QLabel("Info text")
        self.frameInfo.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignHCenter)
        
        self.frameSpinBox = QtGui.QSpinBox()
        self.frameSpinBox.setRange(0, numFrames-1)
        self.frameSpinBox.setSingleStep(1)
        
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(numFrames-1)
        
        filteringControlsGroup = QtGui.QGroupBox("Filtering Controls")
        filteringControlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
                                             "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        filteringControlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        blurControlsGroup = QtGui.QGroupBox("Blurring Controls")
        blurControlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
                                        "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        blurControlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        levelsControlsGroup = QtGui.QGroupBox("Levels Controls")
        levelsControlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
                                          "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        levelsControlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        cropControlsGroup = QtGui.QGroupBox("Cropping Controls")
        cropControlsGroup.setStyleSheet("QGroupBox { margin: 5px; border: 2px groove gray; border-radius: 3px; } "+
                                        "QGroupBox::title {left: 15px; top: -7px; font: bold;}")
        cropControlsGroup.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.doHighPassFilterBox = QtGui.QCheckBox()
        
        self.highPassThreshSpinBox = QtGui.QSpinBox()
        self.highPassThreshSpinBox.setRange(1, 255)
        self.highPassThreshSpinBox.setValue(10)
        
        self.doBlurBox = QtGui.QCheckBox()
        
        self.blurSizeSpinBox = QtGui.QSpinBox()
        self.blurSizeSpinBox.setRange(1, 33)
        self.blurSizeSpinBox.setSingleStep(2)
        self.blurSizeSpinBox.setValue(11)
        
        self.blurSigmaSpinBox = QtGui.QDoubleSpinBox()
        self.blurSigmaSpinBox.setRange(0.0, 30.0)
        self.blurSigmaSpinBox.setSingleStep(0.1)
        self.blurSigmaSpinBox.setValue(2.0)
        
        self.doLevelsBox = QtGui.QCheckBox()
        
        self.levelsSteepnessSpinBox = QtGui.QDoubleSpinBox()
        self.levelsSteepnessSpinBox.setRange(0.0, 1.0)
        self.levelsSteepnessSpinBox.setSingleStep(0.01)
        self.levelsSteepnessSpinBox.setValue(0.7)
        
        self.levelsMidpointSpinBox = QtGui.QSpinBox()
        self.levelsMidpointSpinBox.setRange(1, 254)
        self.levelsMidpointSpinBox.setValue(15)
        
        self.doCropBox = QtGui.QCheckBox()
        
        self.cropWidthSpinBox = QtGui.QSpinBox()
        self.cropWidthSpinBox.setRange(1, imageSize[1])
        self.cropWidthSpinBox.setSingleStep(2)
        self.cropWidthSpinBox.setValue(imageSize[1])
        
        self.cropHeightSpinBox = QtGui.QSpinBox()
        self.cropHeightSpinBox.setRange(1, imageSize[0])
        self.cropHeightSpinBox.setSingleStep(2)
        self.cropHeightSpinBox.setValue(imageSize[0])
        
        self.cropCenterXSpinBox = QtGui.QSpinBox()
        self.cropCenterXSpinBox.setRange(1, imageSize[1])
        self.cropCenterXSpinBox.setValue(imageSize[1]/2)
        
        self.cropCenterYSpinBox = QtGui.QSpinBox()
        self.cropCenterYSpinBox.setRange(1, imageSize[0])
        self.cropCenterYSpinBox.setValue(imageSize[0]-20)
        
        self.cropAnchorXSpinBox = QtGui.QSpinBox()
        self.cropAnchorXSpinBox.setRange(1, imageSize[1])
        self.cropAnchorXSpinBox.setValue(imageSize[1]/2)
        
        self.cropAnchorYSpinBox = QtGui.QSpinBox()
        self.cropAnchorYSpinBox.setRange(1, imageSize[0])
        self.cropAnchorYSpinBox.setValue(imageSize[0]-20)
        
        self.setBackgroundColorButton = QtGui.QPushButton("&Background Color")
        self.saveFramesButton = QtGui.QPushButton("&Save Frames")
        
        ## SIGNALS ##
        
        self.frameSpinBox.valueChanged[int].connect(self.frameSlider.setValue)
        self.frameSlider.valueChanged[int].connect(self.frameSpinBox.setValue)
        self.frameSpinBox.valueChanged[int].connect(self.changeFrame)
        
        self.doHighPassFilterBox.stateChanged.connect(self.setFrameImage)
        self.highPassThreshSpinBox.editingFinished.connect(self.setFrameImage)
        self.doBlurBox.stateChanged.connect(self.setFrameImage)
        self.blurSizeSpinBox.editingFinished.connect(self.setFrameImage)
        self.blurSigmaSpinBox.editingFinished.connect(self.setFrameImage)
        self.doLevelsBox.stateChanged.connect(self.setFrameImage)
        self.levelsSteepnessSpinBox.editingFinished.connect(self.setFrameImage)
        self.levelsMidpointSpinBox.editingFinished.connect(self.setFrameImage)
        self.doCropBox.stateChanged.connect(self.setFrameImage)
        
        self.cropWidthSpinBox.editingFinished.connect(self.cropSizeChanged)
        self.cropHeightSpinBox.editingFinished.connect(self.cropSizeChanged)
        self.cropWidthSpinBox.editingFinished.connect(self.setFrameImage)
        self.cropHeightSpinBox.editingFinished.connect(self.setFrameImage)
        self.cropCenterXSpinBox.editingFinished.connect(self.setFrameImage)
        self.cropCenterYSpinBox.editingFinished.connect(self.setFrameImage)
        self.cropAnchorXSpinBox.valueChanged.connect(self.setFrameImage)
        self.cropAnchorYSpinBox.valueChanged.connect(self.setFrameImage)
        
        self.setBackgroundColorButton.clicked.connect(self.setBackgroundColor)
        self.saveFramesButton.clicked.connect(self.saveFrames)
        
        
        ## LAYOUTS ##
        
        mainLayout = QtGui.QHBoxLayout()
        filteringControlsLayout = QtGui.QGridLayout()
        filteringControlsLayout.addWidget(QtGui.QLabel("Use High-Pass Filter"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        filteringControlsLayout.addWidget(self.doHighPassFilterBox, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        filteringControlsLayout.addWidget(QtGui.QLabel("High-Pass Threshold"), 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        filteringControlsLayout.addWidget(self.highPassThreshSpinBox, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        filteringControlsGroup.setLayout(filteringControlsLayout)
        
        blurControlsLayout = QtGui.QGridLayout()
        blurControlsLayout.addWidget(QtGui.QLabel("Blur Frame Intensity"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        blurControlsLayout.addWidget(self.doBlurBox, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        blurControlsLayout.addWidget(QtGui.QLabel("Blur Size"), 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        blurControlsLayout.addWidget(self.blurSizeSpinBox, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        blurControlsLayout.addWidget(QtGui.QLabel("Blur Sigma"), 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        blurControlsLayout.addWidget(self.blurSigmaSpinBox, 2, 1, 1, 1, QtCore.Qt.AlignLeft)
        blurControlsGroup.setLayout(blurControlsLayout)
        
        levelsControlsLayout = QtGui.QGridLayout()
        levelsControlsLayout.addWidget(QtGui.QLabel("Adjust Levels"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        levelsControlsLayout.addWidget(self.doLevelsBox, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        levelsControlsLayout.addWidget(QtGui.QLabel("Levels Steepness(k)"), 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        levelsControlsLayout.addWidget(self.levelsSteepnessSpinBox, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        levelsControlsLayout.addWidget(QtGui.QLabel("Levels Midpoint(x0)"), 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        levelsControlsLayout.addWidget(self.levelsMidpointSpinBox, 2, 1, 1, 1, QtCore.Qt.AlignLeft)
        levelsControlsGroup.setLayout(levelsControlsLayout)
        
        cropControlsLayout = QtGui.QGridLayout()
        cropControlsLayout.addWidget(QtGui.QLabel("Crop Frame"), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(self.doCropBox, 0, 1, 1, 2, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(QtGui.QLabel("Crop width"), 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(self.cropWidthSpinBox, 1, 1, 1, 2, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(QtGui.QLabel("Crop Height"), 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(self.cropHeightSpinBox, 2, 1, 1, 2, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(QtGui.QLabel("Crop Center (x, y)"), 3, 0, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(self.cropCenterXSpinBox, 3, 1, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(self.cropCenterYSpinBox, 3, 2, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(QtGui.QLabel("Anchor Point (x, y)"), 4, 0, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(self.cropAnchorXSpinBox, 4, 1, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsLayout.addWidget(self.cropAnchorYSpinBox, 4, 2, 1, 1, QtCore.Qt.AlignLeft)
        cropControlsGroup.setLayout(cropControlsLayout)

        controlsVLayout = QtGui.QVBoxLayout()
        controlsVLayout.addWidget(filteringControlsGroup)
        controlsVLayout.addWidget(blurControlsGroup)
        controlsVLayout.addWidget(levelsControlsGroup)
        controlsVLayout.addWidget(cropControlsGroup)
        controlsVLayout.addWidget(self.setBackgroundColorButton)
        controlsVLayout.addWidget(self.saveFramesButton)
        controlsVLayout.addStretch()
        
        sliderLayout = QtGui.QHBoxLayout()
        sliderLayout.addWidget(self.frameSlider)
        sliderLayout.addWidget(self.frameSpinBox)
        
        frameLayout = QtGui.QVBoxLayout()
        frameLayout.addWidget(self.frameLabel)
        frameLayout.addWidget(self.frameInfo)
        frameLayout.addLayout(sliderLayout)
        
        mainLayout.addLayout(controlsVLayout)
        mainLayout.addLayout(frameLayout)
        self.setLayout(mainLayout)

# <codecell>

window = Window()
window.show()
app.exec_()

# <codecell>

summedUp = np.zeros(cv2.imread(framePaths[0]).shape, dtype=int)

for frame in framePaths :
    summedUp += cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
    
figure(); imshow(np.sum(summedUp, axis=-1)/float(np.max(np.sum(summedUp, axis=-1))))

# <codecell>

figure(); imshow(np.array(cv2.cvtColor(cv2.imread(dataPath+"candle1/frame-00022.png"), cv2.COLOR_BGR2RGB), dtype=uint8))
figure(); imshow(np.array(cv2.cvtColor(cv2.imread(dataPath+"candle2/stabilized/frame-01386.png"), cv2.COLOR_BGR2RGB), dtype=uint8))
figure(); imshow(np.array(cv2.cvtColor(cv2.imread(dataPath+"candle2/stabilized/frame-01386.png"), cv2.COLOR_BGR2RGB)*0.9, dtype=uint8))

# <codecell>

im = window.getMattedImage()
if window.doCropBox.isChecked() :
    im = window.getCroppedImage(im)

# <codecell>

QtGui.QColor(QtCore.Qt.yellow).name()

# <codecell>

figure(); imshow(window.im[:, :, 1:4])
# print np.max(window.im[:, :, :])

# <codecell>

Image.fromarray(np.array(im, dtype=np.uint8)).save("testImg.png")

