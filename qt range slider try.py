
# coding: utf-8

# In[1]:

import sys
import numpy as np
import time

from PySide import QtGui, QtCore


app = QtGui.QApplication(sys.argv)


# In[2]:

class Window(QtGui.QWidget):
    def __init__(self) :
        super(Window, self).__init__()
        
        self.createGUI()
        
        self.setWindowTitle("Range slider")
        self.resize(400, 200)        
        self.setFocus()
        
    def createGUI(self) :
        
        ## WIDGETS ##
        
        self.rangeSlider = RangeSlider(QtCore.Qt.Horizontal)
        self.rangeSlider.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Maximum)
        self.rangeSlider.setMinimum(-10)
        self.rangeSlider.setMaximum(100)
        
        self.lowValSpinBox = QtGui.QSpinBox()
        self.lowValSpinBox.setRange(self.rangeSlider.minimum(), self.rangeSlider.maximum())
        self.lowValSpinBox.setValue(self.lowValSpinBox.minimum())
        self.highValSpinBox = QtGui.QSpinBox()
        self.highValSpinBox.setRange(self.rangeSlider.minimum(), self.rangeSlider.maximum())
        self.highValSpinBox.setValue(self.highValSpinBox.maximum())
        
        
        ## SIGNALS ##
        
        self.rangeSlider.lowValueChangedSignal.connect(self.lowValSpinBox.setValue)
        self.rangeSlider.highValueChangedSignal.connect(self.highValSpinBox.setValue)
        self.lowValSpinBox.valueChanged.connect(self.rangeSlider.setLowValue)
        self.highValSpinBox.valueChanged.connect(self.rangeSlider.setHighValue)
        
        ## LAYOUTS ##
        
        hLayout = QtGui.QHBoxLayout()
        hLayout.addWidget(self.lowValSpinBox)
        hLayout.addStretch()
        hLayout.addWidget(self.highValSpinBox)
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addLayout(hLayout)
        mainLayout.addWidget(self.rangeSlider)
        
        self.setLayout(mainLayout)


# In[3]:

class RangeSlider(QtGui.QSlider) :
    lowValueChangedSignal = QtCore.Signal(int)
    highValueChangedSignal = QtCore.Signal(int)
    
    def __init__(self, orientation) :
        super(RangeSlider, self).__init__(QtCore.Qt.Horizontal)#orientation)
        
        self.handleWidth = 10.0
        self.borders = 1.0
        
        self.lowValue = 0.0
        self.highValue = 0.0
        self.numberOfTicks = 0.0
        
        self.lowHandlePos = 0
        self.highHandlePos = 0
        self.availableSliderWidth = 0
        self.startSliderWidth = 0
        self.endSliderWidth = 0
        
        self.availableLowValueSliderWidth = 0
        self.availableHighValueSliderWidth = 0
        
        self.changeLowValue = False
        self.changeHighValue = False
        
        self.lowValueSliderWasLast = True
        
    def setRange(self, minVal, maxVal) :
        self.setMinimum(minVal)
        self.setMaximum(maxVal)
        
    def setMinimum(self, val) :
        if self.maximum() - val + 1 > 1e20 :
            raise Exception("The maximum range cannot be larger than 1e20")
        super(RangeSlider, self).setMinimum(val)
        self.numberOfTicks = float(self.maximum() - val)
        self.setLowValue(0.0, True)
        
    def setMaximum(self, val) :
        if val - self.minimum() + 1 > 1e20 :
            raise Exception("The maximum range cannot be larger than 1e20")
        super(RangeSlider, self).setMaximum(val)
        self.numberOfTicks = float(val - self.minimum())
        self.setHighValue(self.numberOfTicks, True)
        
    def setLowValue(self, val, isInternalCall=False) :
        if isInternalCall :
            value = val
        else :
            value = np.min([float(val - self.minimum()), self.highValue])
        if value >= 0 and value <= self.numberOfTicks :
            self.lowValue = np.copy(value)
            self.updateSlidersGeometry()
#             print "SET LOW", isInternalCall, self.lowValue, self.highValue; sys.stdout.flush()
            if isInternalCall :
                self.lowValueChangedSignal.emit(int(self.lowValue+self.minimum()))
        
    def setHighValue(self, val, isInternalCall=False) :
        if isInternalCall :
            value = val
        else :
            value = np.max([float(val - self.minimum()), self.lowValue])
        if value >= 0 and value <= self.numberOfTicks :
            self.highValue = np.copy(value)
            self.updateSlidersGeometry()
#             print "SET HIGH", isInternalCall, self.lowValue, self.highValue; sys.stdout.flush()
            if isInternalCall :
                self.highValueChangedSignal.emit(int(self.highValue+self.minimum()))
        
    def mousePressEvent(self, event) :
        if event.button() == QtCore.Qt.LeftButton :
            if (event.posF().x() >= self.lowHandlePos-self.handleWidth/2 and event.posF().x() <= self.lowHandlePos+self.handleWidth/2 and
                event.posF().y() >= self.borders and event.posF().y() <= self.height()-self.borders) :
                
                self.changeLowValue = True
                self.lowValueSliderWasLast = True
            elif (event.posF().x() >= self.highHandlePos-self.handleWidth/2 and event.posF().x() <= self.highHandlePos+self.handleWidth/2 and
                  event.posF().y() >= self.borders and event.posF().y() <= self.height()-self.borders) :
                
                self.changeHighValue = True
                self.lowValueSliderWasLast = False
            else :
                if np.abs(event.pos().x() - self.lowHandlePos) < np.abs(event.pos().x() - self.highHandlePos) :
                    self.lowValueSliderWasLast = True
                else :
                    self.lowValueSliderWasLast = False
            
                if self.lowValueSliderWasLast :
                    if event.pos().x() - self.lowHandlePos < 0 :
                        self.setLowValue(np.max([0, self.lowValue-self.pageStep()]), True)
                    else :
                        self.setLowValue(np.min([self.highValue, self.lowValue+self.pageStep()]), True)
                else :
                    if event.pos().x() - self.highHandlePos < 0 :
                        self.setHighValue(np.max([self.lowValue, self.highValue-self.pageStep()]), True)
                    else :
                        self.setHighValue(np.min([self.numberOfTicks, self.highValue+self.pageStep()]), True)
                
                
            self.prevPoint = event.posF()
                
    def mouseMoveEvent(self, event) :
        if self.changeLowValue and self.availableLowValueSliderWidth != 0 :
            self.setLowValue(np.max([0, np.min([np.round((event.posF().x()-self.startLowValueSliderWidth)/float(self.availableLowValueSliderWidth)*self.highValue), self.highValue])]), True)
        if self.changeHighValue and self.availableHighValueSliderWidth != 0 :
            self.setHighValue(np.min([self.numberOfTicks, np.max([self.lowValue + np.round((event.posF().x()-self.startHighValueSliderWidth)/float(self.availableHighValueSliderWidth)*
                                                                                   (self.numberOfTicks-self.lowValue)), self.lowValue])]), True)
            
        self.prevPoint = event.posF()
            
    def mouseReleaseEvent(self, event) :
        if event.button() == QtCore.Qt.LeftButton :
            self.changeLowValue = False
            self.changeHighValue = False
            
    def keyPressEvent(self, event) :
        if event.key() == QtCore.Qt.Key_Left :
            if self.lowValueSliderWasLast :
                self.setLowValue(np.max([0, self.lowValue-self.singleStep()]), True)
            else :
                self.setHighValue(np.max([self.lowValue, self.highValue-self.singleStep()]), True)
        elif event.key() == QtCore.Qt.Key_Right :
            if self.lowValueSliderWasLast :
                self.setLowValue(np.min([self.highValue, self.lowValue+self.singleStep()]), True)
            else :
                self.setHighValue(np.min([self.numberOfTicks, self.highValue+self.singleStep()]), True)
                
    def wheelEvent(self, event) :
        if np.abs(event.pos().x() - self.lowHandlePos) < np.abs(event.pos().x() - self.highHandlePos) :
            self.lowValueSliderWasLast = True
        else :
            self.lowValueSliderWasLast = False
            
        if self.lowValueSliderWasLast :
            if event.delta() < 0 :
                self.setLowValue(np.max([0, self.lowValue-self.singleStep()]), True)
            else :
                self.setLowValue(np.min([self.highValue, self.lowValue+self.singleStep()]), True)
        else :
            if event.delta() < 0 :
                self.setHighValue(np.max([self.lowValue, self.highValue-self.singleStep()]), True)
            else :
                self.setHighValue(np.min([self.numberOfTicks, self.highValue+self.singleStep()]), True)
            
        time.sleep(0.01)
            
    def resizeEvent(self, event) :
        self.availableSliderWidth = self.width()-self.handleWidth-self.borders*2
        self.startSliderWidth = self.handleWidth/2+self.borders
        self.endSliderWidth = self.startSliderWidth + self.availableSliderWidth
        self.updateSlidersGeometry()
        
    def updateSlidersGeometry(self) :
        if self.maximum()-self.minimum() == 0 :
            self.availableLowValueSliderWidth = 0
            self.availableLowValueSliderWidth = 0
        else :
            self.availableLowValueSliderWidth = self.highValue/self.numberOfTicks*(self.availableSliderWidth-self.handleWidth-1)
            self.availableHighValueSliderWidth = (self.numberOfTicks-self.lowValue)/self.numberOfTicks*(self.availableSliderWidth-self.handleWidth-1)
        
        self.startLowValueSliderWidth = self.startSliderWidth
        self.lowHandlePos = np.copy(self.startLowValueSliderWidth).astype(float)
        if self.highValue != 0 :
            self.lowHandlePos += self.lowValue/self.highValue*self.availableLowValueSliderWidth
            
        self.startHighValueSliderWidth = self.startLowValueSliderWidth+self.lowHandlePos-self.startSliderWidth+self.handleWidth+1
        self.highHandlePos = np.copy(self.startHighValueSliderWidth).astype(float)
        if self.numberOfTicks-self.lowValue != 0 :
            self.highHandlePos += (self.highValue-self.lowValue)/(self.numberOfTicks-self.lowValue)*self.availableHighValueSliderWidth
            
        self.update()
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        
        painter.drawLine(QtCore.QPointF(self.startSliderWidth, self.height()/2),
                         QtCore.QPointF(self.endSliderWidth, self.height()/2))
        
        painter.drawRect(np.round(self.lowHandlePos)-self.handleWidth/2, self.borders, self.handleWidth, self.height()-self.borders*2)
        painter.drawRect(np.round(self.highHandlePos)-self.handleWidth/2, self.borders, self.handleWidth, self.height()-self.borders*2)


# In[4]:

window = Window()
window.show()
app.exec_()

