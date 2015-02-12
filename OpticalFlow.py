# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## Imports and defines
%pylab
import numpy as np
import cv2

np.set_printoptions(threshold=numpy.nan)

# <codecell>

cap = cv2.VideoCapture("data/flower/frame-%05d.png")
# cap = cv2.VideoCapture("data/crazyhorse/P%04d.JPG")
ret, frame1 = cap.read()
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


prvsFrame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#cv2.imshow('frame1', prvsFrame)
#cv2.waitKey(30)

ret, frame2 = cap.read()
nextFrame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#cv2.imshow('frame2', nextFrame)
#cv2.waitKey(30)

pyr_scale = float(0.5)
poly_sigma = float(1.2)
flow = cv2.calcOpticalFlowFarneback(prvsFrame, nextFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

#cv2.imshow('frame2',rgb)
#cv2.waitKey(30)

cap.release()
#cv2.destroyAllWindows()
#cv2.waitKey(30)

# <codecell>


# <codecell>

simpleflow = cv2.calcOpticalFlowSF(frame1, frame2, 1, 1, 1)#, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10)

simplehsv = np.zeros_like(frame1)
simplehsv[...,1] = 255
simplemag, simpleang = cv2.cartToPolar(simpleflow[...,0], simpleflow[...,1])
simplehsv[...,0] = simpleang*180/np.pi/2
simplehsv[...,2] = cv2.normalize(simplemag,None,0,255,cv2.NORM_MINMAX)
simplebgr = cv2.cvtColor(simplehsv,cv2.COLOR_HSV2BGR)
figure(); plt.imshow(cv2.cvtColor(simplebgr,cv2.COLOR_BGR2RGB), interpolation='nearest')

# <codecell>

np.save("testSimpleFlow", cv2.cvtColor(simplebgr,cv2.COLOR_BGR2RGB))

# <codecell>

img = np.zeros_like(frame1)
counter = 0
for i in range(0, simpleflow.shape[0]) :
    for j in range(0, simpleflow.shape[1]) :
        coords = [i, j]
        coords = np.array(np.round(simpleflow[i, j, :] + coords), dtype=int)
        
        #if np.logical_and(np.all(coords > np.array([0, 0])), np.all(np.hstack([coords, 0]) < np.array(frame1.shape))):
            #img[i, j, :] = frame2[coords[0], coords[1], :]
        if np.logical_and(i + simpleflow[i, j, 1] < frame2.shape[0], j + simpleflow[i, j, 0] < frame2.shape[1]) :
            img[i, j, :] = frame2[i + simpleflow[i, j, 1], j + simpleflow[i, j, 0], :]
            counter += 1
    print i,

print    
print 'pixel fill rate', 100*counter/(frame1.shape[0]*frame1.shape[1]), '%'

# <codecell>

figure()
implot = plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), alpha=0.5)
#     print "la", flow.shape[0], flow.shape[1]
for i in xrange(20, simpleflow.shape[0], 20) :
    for j in xrange(20, simpleflow.shape[1], 20) :
        point = np.array([j, i])
        displPoint = point+simpleflow[i, j, :]
#             print flow[i, j, :]
        plt.arrow(j, i, simpleflow[i, j, 0], simpleflow[i, j, 1], head_width=2.5, head_length=5.0, fc='k', ec='k', shape='left')
#             plt.plot([point[0], displPoint[0]], [point[1], displPoint[1]], color='g', marker='.')

# <codecell>

plt.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), interpolation='nearest')

# <codecell>

figure(0)
plt.imshow(cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB))
figure(1)
plt.imshow(cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB))
figure(2)
plt.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), interpolation='nearest')

# <codecell>

# img is supposed to be the same as frame1 but only taking pixel values from frame2 and moving them according to flow (flow is bad so img will be bad)
img = np.zeros_like(frame1)
counter = 0
for i in range(0, flow.shape[0]) :
    for j in range(0, flow.shape[1]) :
        coords = [i, j]
        coords = np.array(np.round(flow[i, j, :] + coords), dtype=int)
        
        #if np.logical_and(np.all(coords > np.array([0, 0])), np.all(np.hstack([coords, 0]) < np.array(frame1.shape))):
            #img[i, j, :] = frame2[coords[0], coords[1], :]
        if np.logical_and(i + flow[i, j, 1] < frame2.shape[0], j + flow[i, j, 0] < frame2.shape[1]) :
            img[i, j, :] = frame2[i + flow[i, j, 1], j + flow[i, j, 0], :]
            counter += 1
    print i,

print    
print 'pixel fill rate', 100*counter/(frame1.shape[0]*frame1.shape[1]), '%'

# <codecell>

figure(); imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest')

# <codecell>

## try and morph without using for loop
img2 = np.zeros_like(frame1)
roundFlow =  np.array(np.round(flow), dtype=int)
#print img2.shape
#img2 = frame2[roundFlow[:, :, 1], roundFlow[:, :, 0], :]
#print img2.shape
#plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
#s = [5, 8]
#print flow.shape[0:-1]
rows = np.reshape(np.repeat(np.array(range(0,flow.shape[0])), flow.shape[1]), flow.shape[0:-1]) + roundFlow[:, :, 1]
cols = np.repeat(np.reshape(np.array(range(0,flow.shape[1])), [1,flow.shape[1]]), flow.shape[0], axis=0) + roundFlow[:, :, 0]

#rows = np.reshape(np.repeat(np.array(range(0,s[0])), s[1]), s)
#cols = np.repeat(np.reshape(np.array(range(0,s[1])), [1,s[1]]), s[0], axis=0)
print rows.shape
print cols.shape

rows[np.where(rows > (roundFlow.shape[0]-1))] = 0
cols[np.where(cols > (roundFlow.shape[1]-1))] = 0

img2 = frame2[rows, cols, :]
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
cv2.imwrite('1d.png', img2)

# <codecell>

#img = np.zeros_like(frame1)
#img[...,1] = 255
#img = zeros(frame1.shape, dtype=int)
#plt.imshow(img)
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print img[300, :, :]
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
cv2.imwrite('1c.png', img)

# <codecell>

cv2.imwrite('flow2.png', bgr)

