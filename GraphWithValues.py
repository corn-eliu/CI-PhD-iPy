# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt

import numpy as np

# <codecell>

def showCustomGraph(img, title = None, showColorbar = False, integerAxes = True) :
    
    if title != None :
        fig = plt.figure(title)
    else :
        fig = plt.figure()
    
    ax = fig.add_subplot(111)
    cax = ax.imshow(img, interpolation='nearest')
    
    numrows, numcols = img.shape
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = img[row,col]
            if integerAxes :
                return 'x=%d, y=%d, z=%1.4f'%(x+0.5, y+0.5, z)
            else :
                return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        else:
            if integerAxes :
                return 'x=%d, y=%d'%(x+0.5, y+0.5)
            else :
                return 'x=%1.4f, y=%1.4f'%(x, y)
    
    ax.format_coord = format_coord
    
    if showColorbar :
        cax = fig.colorbar(cax)
    
    plt.show()

