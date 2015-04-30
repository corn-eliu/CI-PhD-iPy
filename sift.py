""" 
Python module for use with David Lowe's SIFT code available at:
http://www.cs.ubc.ca/~lowe/keypoints/
adapted from the matlab code examples.

Jan Erik Solem, 2009-01-30
"""

from PIL import Image
import os
from numpy import *
import pylab


def process_image(imagename, resultname):
    """ process an image and save the results in a .key ascii file"""

    if imagename[-3:] != 'pgm':
        #create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
    
    #check if linux or windows 
    if os.name == "posix":
        cmmd = "sift <"+imagename+">"+resultname
    else:
        cmmd = "siftWin32 <"+imagename+">"+resultname
    
    os.system(cmmd)
    print 'processed', imagename
    
def read_features_from_file(filename):
    """ read feature properties and return in matrix form"""
    
    f = open(filename, 'r')
    header = f.readline().split()
    
    num = int(header[0]) #the number of features
    featlength = int(header[1]) #the length of the descriptor
    if featlength != 128: #should be 128 in this case
        raise RuntimeError, 'Keypoint descriptor length invalid (should be 128).' 
        
    locs = zeros((num, 4))
    descriptors = zeros((num, featlength));        

    #parse the .key file
    e =f.read().split() #split the rest into individual elements
    pos = 0
    for point in range(num):
        #row, col, scale, orientation of each feature
        for i in range(4):
            locs[point,i] = float(e[pos+i])
        pos += 4
        
        #the descriptor values of each feature
        for i in range(featlength):
            descriptors[point,i] = int(e[pos+i])
        #print descriptors[point]
        pos += 128
        
        #normalize each input vector to unit length
        descriptors[point] = descriptors[point] / linalg.norm(descriptors[point])
        #print descriptors[point]
        
    f.close()
    
    return locs,descriptors
    
def match(desc1,desc2):
    """ for each descriptor in the first image, select its match to second image
        input: desc1 (matrix with descriptors for first image), 
        desc2 (same for second image)"""
    
    dist_ratio = 0.6
    desc1_size = desc1.shape
    
    matchscores = zeros((desc1_size[0],1))
    desc2t = desc2.T #precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i,:],desc2t) #vector of dot products
        dotprods = 0.9999*dotprods
        #inverse cosine and sort, return index for features in second image
        indx = argsort(arccos(dotprods))
        
        #check if nearest neighbor has angle less than dist_ratio times 2nd
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = indx[0]
        
    return matchscores 

def match_twosided(desc1,desc2):
	""" two-sided symmetric version of match(). """
	
	matches_12 = match(desc1,desc2)
	matches_21 = match(desc2,desc1)
	
	ndx_12 = matches_12.nonzero()[0]
	
	#remove matches that are not symmetric
	for n in ndx_12:
		if matches_21[int(matches_12[n])] != n:
			matches_12[n] = 0
	
	return matches_12
    
def plot_features(im,locs):
    """ show image with features. input: im (image as array), 
        locs (row, col, scale, orientation of each feature) """
    
    pylab.gray()
    pylab.imshow(im)
    pylab.plot([p[1] for p in locs], [p[0] for p in locs], 'ob')
    pylab.axis('off')
    pylab.show()
    
def appendimages(im1,im2):
    """ return a new image that appends the two images side-by-side."""
    
    #select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))), axis=0)
    #if none of these cases they are equal, no filling needed.
        
    return concatenate((im1,im2), axis=1)
    
def plot_matches(im1,im2,locs1,locs2,matchscores, numMatches=-1):
    """ show a figure with lines joining the accepted matches in im1 and im2
        input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
        matchscores (as output from 'match'), numMatches (number of matches to plot) """
    
    im3 = appendimages(im1,im2)
    if numMatches == -1 :
        numMatches = len(matchscores)

    pylab.gray()
    pylab.imshow(im3)
    
    cols1 = im1.shape[1]
    for i in range(numMatches):
        if matchscores[i] > 0:
            pylab.plot([locs1[i,1], locs2[int(matchscores[i]),1]+cols1], [locs1[i,0], locs2[int(matchscores[i]),0]], 'c')
    pylab.axis('off')
    pylab.show()

def plot_match_displacement(im1,locs1,locs2,matchscores, numMatches=-1):
    """ show a figure with lines where feats in im1 moved to in im2
        input: im1 (image as array), locs1,locs2 (location of features), 
        matchscores (as output from 'match'), numMatches (number of matches to plot) """

    if numMatches == -1 :
        numMatches = len(matchscores)

    pylab.gray()
    pylab.imshow(im1)
    
    cols1 = im1.shape[1]
    for i in range(numMatches):
        if matchscores[i] > 0:
            pylab.arrow(locs1[i,1], locs1[i,0], locs2[int(matchscores[i]),1]-locs1[i,1], locs2[int(matchscores[i]),0]-locs1[i,0], head_width=1.0, head_length=2.0, linewidth=0.5, fc='r', ec='r')
    pylab.axis('off')
    pylab.show()
