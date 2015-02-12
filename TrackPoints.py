# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import scipy as sp
from scipy import io
import re
import cv2
import sys
import glob
import Image 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import MazeSequenceUtils as msu

dataLoc = "/home/ilisescu/PhD/iPy/data/flower/"
nameLength = len(filter(None, re.split('/',dataLoc)))
nvmFile = "sparse.nvm"

colVals = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

# <codecell>

def flowToRgb(flow) :
    hsvFlow = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=uint8)
    hsvFlow[...,1] = 255
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsvFlow[...,0] = ang*180/np.pi/2
    hsvFlow[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#     hsvFlow[...,2] = 255
    rgbFlow = cv2.cvtColor(hsvFlow,cv2.COLOR_HSV2RGB)
    return rgbFlow

# <codecell>

def filterFlow(flow, iterations) :
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsvFlow = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=double)
    hsvFlow[...,1] = 255
    hsvFlow[...,0] = ang*180/np.pi/2
    hsvFlow[...,2] = cv2.normalize(mag,None,0.0,255.0,cv2.NORM_MINMAX)
    
    filteredHsvFlow = np.array(np.copy(hsvFlow), dtype=uint8)
    for i in xrange(0, iterations) :
        filteredHsvFlow = cv2.pyrMeanShiftFiltering(filteredHsvFlow, 5, 10)
        sys.stdout.write('\r' + "Filtering flow: iteration " + np.string_(i) + " of " + np.string_(iterations))
        sys.stdout.flush()
    
    print " -->", "done"
    
    filteredMag = cv2.normalize(np.array(filteredHsvFlow[..., 2], dtype=float),None,np.min(mag),np.max(mag),cv2.NORM_MINMAX)
    filteredAng = np.array(filteredHsvFlow[..., 0], dtype=float)*np.pi*2/180
    x, y = cv2.polarToCart(filteredMag, filteredAng)
    filteredFlow = np.zeros_like(flow)
    filteredFlow[:, :, 0] = x
    filteredFlow[:, :, 1] = y
    return filteredFlow

# <codecell>

def rgbToFlow(rgb) :
    hsvFlow = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    print 'toFlow', hsvFlow[250, 250, 0], hsvFlow[250, 250, 2]
#     hsvFlow = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=uint8)
#     hsvFlow[...,1] = 255
    
#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#     hsvFlow[...,0] = ang*180/np.pi/2
#     hsvFlow[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
# #     hsvFlow[...,2] = 255
#     rgbFlow = cv2.cvtColor(hsvFlow,cv2.COLOR_HSV2RGB)
#     return rgbFlow

# <codecell>

## load frames
frameNames = np.sort(glob.glob(dataLoc + "*.png"))
frameSize = cv2.imread(frameNames[0]).shape
movie = np.zeros(np.hstack([frameSize, len(frameNames)]), dtype=uint8)
print movie.shape
for i in xrange(0, len(frameNames)):
#     im = np.array(cv2.imread(location+frames[i]))/255.0
#     movie[:, :, i] = np.dot(im[:,:,:3], [0.0722, 0.7152, 0.2126])   # matlab RGB2GRAY multiplies [0.299, 0.587, 0.144] but cv2 is BGR
#     movie[:, :, :, i] = np.array(cv2.imread(frameNames[i]))/255.0
    movie[:, :, :, i] = cv2.imread(frameNames[i])
    sys.stdout.write('\r' + "Loading frame " + np.string_(i) + " of " + np.string_(len(frameNames)))
    sys.stdout.flush()

# <codecell>

## show first image and let user draw some points
fig = plt.figure()
ax = fig.add_subplot(111)
firstFrame = cv2.cvtColor(movie[:, :, :, 0], cv2.COLOR_BGR2RGB)
ax.imshow(firstFrame)
ax.set_autoscale_on(False)
# ax.set_xlim(firstFrame.shape[1])
# ax.set_ylim(firstFrame.shape[0])

userPoints = list()
line, = ax.plot(0, 0, color='b', lw=4.0)
buttonClicked = False

def onclick(event):
    global buttonClicked
    global userPoints
    if event.button == 1 :
        buttonClicked = True
        userPoints = list()
        userPoints.append(np.array([event.xdata, event.ydata]))

def onmove(event):
    global buttonClicked
    global userPoints
    global line
    if buttonClicked == True :
        userPoints.append(np.array([event.xdata, event.ydata]))
        line.set_ydata(np.array(userPoints)[:, 1])
        line.set_xdata(np.array(userPoints)[:, 0])
        draw()
        
def onrelease(event):
    global buttonClicked
    global userPoints
    global line
    if buttonClicked == True and event.button == 1 :
        buttonClicked = False
        userPoints.append(np.array([event.xdata, event.ydata]))
        line.set_ydata(np.array(userPoints)[:, 1])
        line.set_xdata(np.array(userPoints)[:, 0])
        draw()
    
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('motion_notify_event', onmove)
cid = fig.canvas.mpl_connect('button_release_event', onrelease)

# <codecell>

## find lineSegment that best fits user given points and that will be the skeleton --> one line for now
userPoints = np.round(np.array(userPoints))
line = msu.fitLine2D(userPoints)
## project first and last userPoint onto line
u = line[0:2]
a = line[2:4]
firstPoint = userPoints[0, :]
## project point onto vector in the line direction u
pFirstPoint = (np.dot(firstPoint, u)/np.dot(u, u))*u
lastPoint = userPoints[-1, :]
## project point onto vector in the line direction u
pLastPoint = (np.dot(lastPoint, u)/np.dot(u, u))*u

## correct for line segment being moved by a certain scalar along normal for some reason
lineNormal = (pFirstPoint-firstPoint)/np.linalg.norm(pFirstPoint-firstPoint)

distanceToLine = np.linalg.norm(a - pFirstPoint - np.dot(np.dot(a-pFirstPoint, u), u))
pFirstPoint = pFirstPoint - distanceToLine*lineNormal

distanceToLine = np.linalg.norm(a - pLastPoint - np.dot(np.dot(a-pLastPoint, u), u))
pLastPoint = pLastPoint - distanceToLine*lineNormal

## show
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(firstFrame)
ax.set_autoscale_on(False)
ax.plot(userPoints[:, 0], userPoints[:, 1], color="b") ## plot user points
ax.plot([pFirstPoint[0], pLastPoint[0]],[pFirstPoint[1], pLastPoint[1]], color="g") ## plot fitted line segment


print userPoints

# <codecell>

## find flow between each pair of consecutive frames and compute SIFT features in the resulting flow
sift = cv2.SIFT()
numFrames = 2 #movie.shape[3])
flows = np.zeros([movie.shape[0], movie.shape[1], 2, numFrames-1]);    
pyr_scale = float(0.5)
poly_sigma = float(1.2)
for f in xrange(1, numFrames):
    print "pair (", f-1, ", ", f, ")"
    
    frame1 = cv2.cvtColor(movie[:, :, :, f-1],cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(movie[:, :, :, f],cv2.COLOR_BGR2GRAY)
    
#     if f > 1 :
#         flow = cv2.calcOpticalFlowFarneback(frame1, frame2, np.array(flows[:, :, :, f-2]), pyr_scale, 3, 15, 3, 5, poly_sigma, cv2.OPTFLOW_USE_INITIAL_FLOW)
#         print "gna"
#     else :
#         flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, 3, 15, 3, 5, poly_sigma, 0)
#         print "bah"
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, 3, 15, 3, 5, poly_sigma, 0)
    
    flows[:, :, :, f-1] = flow
    
    rgbFlow = flowToRgb(flow)
    
    kp = sift.detect(rgbFlow,None)
    points = cv2.KeyPoint_convert(kp)
    
    # img=cv2.drawKeypoints(gray,kp)
    figure()
    implot = plt.imshow(rgbFlow)
    
    # show key points
    plt.scatter(points[:, 0], points[:, 1], color='r', marker='x')
    
    plt.show()
    
    # plot flow as vectors
    figure()
    implot = plt.imshow(cv2.cvtColor(movie[:, :, :, f-1],cv2.COLOR_BGR2RGB), alpha=0.5, cmap=cm.Greys_r)
#     print "la", flow.shape[0], flow.shape[1]
    for i in xrange(20, flow.shape[0], 20) :
        for j in xrange(20, flow.shape[1], 20) :
            point = np.array([j, i])
            displPoint = point+flow[i, j, :]
#             print flow[i, j, :]
            plt.arrow(j, i, flow[i, j, 0], flow[i, j, 1], head_width=2.5, head_length=5.0, fc='k', ec='k', shape='left')
#             plt.plot([point[0], displPoint[0]], [point[1], displPoint[1]], color='g', marker='.')

# <codecell>

bob = io.loadmat('opt_flow.mat')
bob = bob['flow']

# <codecell>

figure()
imshow(flowToRgb(flow))
figure()
imshow(flowToRgb(bob))

# <codecell>

filteredFlows = np.zeros_like(flows)
for fl in xrange(0, flows.shape[3]) :
    filteredFlows[:, :, :, fl] = filterFlow(flows[:, :, :, fl], 50)

# <codecell>

## move user given points based on flow
# figure()
# imshow(rgbFlow)
## in userPoints [:, 0] are x coords but to use them as indices I need [:, 0] to be rows which are the y coords
tmp = np.zeros([len(userPoints), 2])
tmp[:, 0] = userPoints[:, 1]
tmp[:, 1] = userPoints[:, 0]

displPoints = np.zeros([userPoints.shape[0], userPoints.shape[1], flows.shape[3]])
for point, i in zip(userPoints, xrange(0, len(userPoints))) :
#     print point, i
    prevPoint = point
    for fl in xrange(0, flows.shape[3]) :
#         print fl, prevPoint, filteredFlows[prevPoint[1], prevPoint[0], fl]
        displPoints[i, :, fl] = prevPoint+filteredFlows[prevPoint[1], prevPoint[0], fl]
        prevPoint = displPoints[i, :, fl]

# print np.array(userPoints, dtype=int)
# print np.array(tmp[-3, :], dtype=int)
# print flow[[636, 720]]
# print rgbFlow[np.array(tmp[-3, :], dtype=int)]
# print rgbFlow[np.array(tmp, dtype=int)]
print displPoints.shape

# <codecell>

figure()
imshow(flowToRgb(flows[:, :, :, 1]))

frame1 = cv2.cvtColor(flowToRgb(flows[:, :, :, 0]), cv2.COLOR_RGB2GRAY) #cv2.cvtColor(movie[:, :, :, 0],cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(flowToRgb(flows[:, :, :, 1]), cv2.COLOR_RGB2GRAY) #cv2.cvtColor(movie[:, :, :, 1],cv2.COLOR_BGR2GRAY)

# print frame1.shape, frame2.shape

# cv2.l

bob = np.array(flows[:, :, :, 0])
tmp = cv2.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, 3, 15, 3, 5, poly_sigma, 0)
figure()
imshow(flowToRgb(tmp))

# <codecell>

figure()
imshow(movie[:, :, :, 0])
figure()
imshow(movie[:, :, :, 1])
figure()
imshow(flowToRgb(bob))

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(firstFrame)
ax.set_autoscale_on(False)
ax.plot(userPoints[:, 0], userPoints[:, 1], color="b") ## plot user points
ax.plot(displPoints[:, 0, 0], displPoints[:, 1, 0], color="r") ## plot displaced user points
ax.plot(displPoints[:, 0, 1], displPoints[:, 1, 1], color="g") ## plot displaced user points

# <codecell>

print userPoints
print displPoints[:, :, 0]
print displPoints[:, :, 1]

# <codecell>

## normalize flow to ignore magnitude and only keep direction
print flow.shape
norms = np.sqrt(np.power(flow[:, :, 0], 2) + np.power(flow[:, :, 1], 2))
norms = np.repeat(np.reshape(norms, [flow.shape[0], flow.shape[1], 1]), 2, axis=2)
norms[np.where(norms == 0)] = 1
normFlow = flow/norms
# print np.linalg.norm(

# <codecell>

figure()
imshow(rgbFlow)
filtered = rgbFlow
for i in xrange(0, 500) :
    filtered = cv2.pyrMeanShiftFiltering(filtered, 5, 10)
figure()
imshow(filtered)

# <codecell>

figure()
imshow(getRGBFlow(flow))

# <codecell>

figure()
for r in xrange(0, flow.shape[0]) :
    tmp = np.string_(hex(np.mod(r, 255)))[2:-1]
    col = "#"
    if int(r/256) == 0 :
        col = "#"
        if len(tmp) == 1 :
            col = col + "0"
        col = col + tmp + "0000"
    elif int(r/256) == 1 :
        col = "#00"
        if len(tmp) == 1 :
            col = col + "0"
        col = col + tmp + "00"
    else :
        col = "#0000"
        if len(tmp) == 1 :
            col = col + "0"
        col = col + tmp + ""
    #col = "#000" + tmp
    plt.scatter(flow[r, :, 0], flow[r, :, 1], marker='.', edgecolors='none', color=col)
#     print np.string_(hex(np.mod(r, 255)))
    sys.stdout.write('\r' + "Plotting row " + np.string_(r))
    sys.stdout.flush()

# <codecell>

## find kmeans
flowPoints = np.reshape(flow, [flow.shape[0]*flow.shape[1], flow.shape[2]])

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(flowPoints,3,None,criteria,10,flags)

# <codecell>

## reshape the given labels to image size as now each pixel in original image will have been assigned a label based on its flow
labelsImg = np.reshape(labels, [flow.shape[0], flow.shape[1]])
figure()
imshow(labelsImg)

# <codecell>

## plot frame with clusters overlaid on top
figure()
imshow(cv2.cvtColor(movie[:, :, :, f-1],cv2.COLOR_BGR2RGB))
imshow(labelsImg, alpha=0.5)

# <codecell>

## compute canny edge for first 2 frames
frame1 = cv2.cvtColor(movie[:, :, :, 0],cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(movie[:, :, :, 1],cv2.COLOR_BGR2GRAY)

edges1 = cv2.Canny(frame1, 100, 200)
edges2 = cv2.Canny(frame2, 100, 200)

dilEdges1 = sp.ndimage.morphology.binary_dilation(edges1, iterations=5)
dilEdges2 = sp.ndimage.morphology.binary_dilation(edges2, iterations=5)

pyr_scale = float(0.5)
poly_sigma = float(1.2)
flow = cv2.calcOpticalFlowFarneback(frame1*dilEdges1, frame2*dilEdges2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

figure()
imshow(cv2.cvtColor(movie[:, :, :, 0]*np.repeat(np.reshape(dilEdges1, [dilEdges1.shape[0], dilEdges1.shape[1], 1]), 3, axis=2),cv2.COLOR_BGR2RGB))
figure()
imshow(cv2.cvtColor(movie[:, :, :, 1]*np.repeat(np.reshape(dilEdges2, [dilEdges2.shape[0], dilEdges2.shape[1], 1]), 3, axis=2),cv2.COLOR_BGR2RGB))
figure()
imshow(getRGBFlow(flow))

# <codecell>

kp = sift.detect(cv2.cvtColor(movie[:, :, :, 0],cv2.COLOR_BGR2RGB), np.array(dilEdges1, dtype=uint8))
points = cv2.KeyPoint_convert(kp)

# img=cv2.drawKeypoints(gray,kp)
figure()
implot = plt.imshow(cv2.cvtColor(movie[:, :, :, 0],cv2.COLOR_BGR2RGB))

# show key points
plt.scatter(points[:, 0], points[:, 1], color='r', marker='x')

plt.show()

# <codecell>

print np.max(np.array(dilEdges1, dtype=uint))

# <codecell>

tmp = flow[300:305, 300:305, :]
print tmp
print tmp[0, :]
print np.reshape(tmp, [tmp.shape[0]*tmp.shape[1], tmp.shape[2]])[:, 0]

# <codecell>

print np.max(flow[400, :, 0])
print np.max(flow[400, :, 1])

# <codecell>

## find SIFT and plot onto image
# gray= np.dot(movie[:,:,:3, 0], [0.0722, 0.7152, 0.2126])
img = cv2.cvtColor(movie[:, :, :, 0], cv2.COLOR_BGR2RGB)
print img.shape

sift = cv2.SIFT()
kp = sift.detect(img,None)

points = cv2.KeyPoint_convert(kp)
# print points.shape

# img=cv2.drawKeypoints(gray,kp)
figure()
implot = plt.imshow(img)

# put a blue dot at (10, 20)
plt.scatter(points[:, 0], points[:, 1], color='r', marker='x')

# put a red dot, size 40, at 2 locations:
# plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)

plt.show()

# <codecell>

print np.max(movie[:, :, :, 0])
print np.max(tmp)

# <codecell>

## play frames automagically
frames = arange(0, len(frameNames))

def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        # ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw

fig=plt.figure()
img = plt.imshow(movie[:, :, :, frames[0]])
img.set_cmap(cm.Greys_r)
img.axes.set_axis_off()
ax = plt.axes()
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')

def init():
    ttl.set_text('')
    img.set_data(cv2.cvtColor(movie[:, :, :, frames[0]], cv2.COLOR_BGR2RGB))
    return img, ttl

def func(f):
    ttl.set_text('Frame ' + str(f))
    img.set_data(cv2.cvtColor(movie[:, :, :, frames[f]], cv2.COLOR_BGR2RGB))
    return img, ttl

ani = animation.FuncAnimation(fig,func,init_func=init,frames=len(frames),interval=30,blit=True)

plt.show()

