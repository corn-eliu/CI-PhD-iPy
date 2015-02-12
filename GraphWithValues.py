# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt

import numpy as np

# <codecell>

def showCustomGraph(img) :

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, interpolation='nearest')
    
    numrows, numcols = img.shape
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = img[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)
    
    ax.format_coord = format_coord
    plt.show()

