# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt

import numpy as np

# <codecell>

def showCustomGraph(img, title = None, showColorbar = False, integerAxes = True, colorbarLimits = None) :
    
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
                return 'row=%d, col=%d, z=%1.4f'%(y+0.5, x+0.5, z)
            else :
                return 'row=%1.4f, col=%1.4f, z=%1.4f'%(y, x, z)
        else:
            if integerAxes :
                return 'row=%d, col=%d'%(y+0.5, x+0.5)
            else :
                return 'row=%1.4f, col=%1.4f'%(y, x)
    
    ax.format_coord = format_coord
    
    if showColorbar :
        cax = fig.colorbar(cax)
        if colorbarLimits != None :
            cax.set_clim(colorbarLimits)
    
    plt.show()

