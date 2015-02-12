# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
import cv2
from sklearn import ensemble
# from skimage.io import imread, imsave
# from skimage.color import rgb2gray
from glob import glob
# import os

# <codecell>

def trainClassifier(data_frames, trimaps) :
    # train on first frame
    # augment with x-y positional data
    clf = ensemble.ExtraTreesClassifier()
    
    print len(data_frames), len(trimaps)
    
    idxs = np.indices(data_frames[0].shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
    data = np.concatenate((data_frames[0], idxs), axis=-1)

    # extract training data
    background = data[trimaps[0] == 0]
    foreground = data[trimaps[0] == 2]
    
    for data_frame, trimap in zip(data_frames[1:], trimaps[1:]) :
        print background.shape, foreground.shape
        
        idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
        data = np.concatenate((data_frame, idxs), axis=-1)
    
        # extract training data
        background = np.vstack((background, data[trimap == 0]))
        foreground = np.vstack((foreground, data[trimap == 2]))
        
    print background.shape, foreground.shape
    
    X = np.vstack((background, foreground))
    y = np.repeat([0.0, 1.0], [background.shape[0], foreground.shape[0]])
    print clf.fit(X, y)
    return clf

# <codecell>

def trainClassifier(data_frames, trimaps) :
    # train on first frame
    # augment with x-y positional data
    clf = ensemble.ExtraTreesClassifier()
    
#         print len(data_frames), len(trimaps)
    
    idxs = np.indices(data_frames[0].shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
#         print idxs.shape, data_frames[0].shape
    data = np.concatenate((data_frames[0], idxs), axis=-1)

    # extract training data
    background = data[trimaps[0] == 0]
    foreground = data[trimaps[0] == 2]
    
    for data_frame, trimap in zip(data_frames[1:], trimaps[1:]) :
#             print background.shape, foreground.shape
        
        idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
#             print idxs.shape, data_frame.shape
        data = np.concatenate((data_frame, idxs), axis=-1)
    
        # extract training data
        background = np.vstack((background, data[trimap == 0]))
        foreground = np.vstack((foreground, data[trimap == 2]))
        
#         print background.shape, foreground.shape
    
    X = np.vstack((background, foreground))
    y = np.repeat([0.0, 1.0], [background.shape[0], foreground.shape[0]])
    print X.T.shape
    print y.shape
    print X
    print y
    clf.fit(X, y)
    return clf

frameNames = ["frame-00001.png", "frame-00074.png", "frame-00213.png"]
trimapNames = ["trimap-frame-00001.png", "trimap-frame-00074.png", "trimap-frame-00213.png"]
data = []
tmaps = []
for idx in xrange(0, 3) :
    data.append(cv2.cvtColor(cv2.imread("../data/ribbon2/" + frameNames[idx]), cv2.COLOR_BGR2RGB))
    tmaps.append(cv2.cvtColor(cv2.imread("../data/ribbon2/" + trimapNames[idx]), cv2.COLOR_BGR2GRAY))

classifier = trainClassifier(data, tmaps)

# <codecell>

def trainClassifier(data_frames, trimaps) :
    # train on first frame
    # augment with x-y positional data
    rtree_params = dict(max_depth=11, min_sample_count=5, use_surrogates=False, max_categories=15, calc_var_importance=False, nactive_vars=0, term_crit=(cv2.TERM_CRITERIA_MAX_ITER,1000,1), termcrit_type=cv2.TERM_CRITERIA_MAX_ITER)
    clf = cv2.SVM()
    
#         print len(data_frames), len(trimaps)
    
    idxs = np.indices(data_frames[0].shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
#         print idxs.shape, data_frames[0].shape
    data = np.concatenate((data_frames[0], idxs), axis=-1)

    # extract training data
    background = data[trimaps[0] == 0]
    foreground = data[trimaps[0] == 2]
    
    for data_frame, trimap in zip(data_frames[1:], trimaps[1:]) :
#             print background.shape, foreground.shape
        
        idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
#             print idxs.shape, data_frame.shape
        data = np.concatenate((data_frame, idxs), axis=-1)
    
        # extract training data
        background = np.vstack((background, data[trimap == 0]))
        foreground = np.vstack((foreground, data[trimap == 2]))
        
#         print background.shape, foreground.shape
    
    X = np.array(np.vstack((background, foreground)), dtype=np.float32)
    y = np.array(np.repeat([0, 1], [background.shape[0], foreground.shape[0]]), dtype=np.float32)
    print np.max(X), np.max(y)
#     clf.train(X, cv2.CV_ROW_SAMPLE, y, params=rtree_params)
    clf.train(X, y)
    return clf

frameNames = ["frame-00001.png", "frame-00074.png", "frame-00213.png"]
trimapNames = ["trimap-frame-00001.png", "trimap-frame-00074.png", "trimap-frame-00213.png"]
data = []
tmaps = []
for idx in xrange(0, 3) :
    data.append(cv2.cvtColor(cv2.imread("../data/ribbon2/" + frameNames[idx]), cv2.COLOR_BGR2RGB))
    tmaps.append(cv2.cvtColor(cv2.imread("../data/ribbon2/" + trimapNames[idx]), cv2.COLOR_BGR2GRAY))

classifier = trainClassifier(data, tmaps)
print classifier

# <codecell>

image = cv2.cvtColor(cv2.imread("../data/ribbon2/frame-00074.png"), cv2.COLOR_BGR2RGB)
figure(); imshow(image)
indices = np.indices(image.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.array(np.concatenate((image, indices), axis=-1), dtype=np.float32).reshape((-1, 5))
# print data.shape, data.reshape((-1, 5)).shape
# probabilities = classifier.predict(data.reshape((-1, 5)))
probs = np.float32( [classifier.predict(d) for d in data] )
print probs.shape

# alphaMatte = np.copy(probabilities[:, 1])
# alphaMatte[probabilities[:, 1]>0.15] = 1
# alphaMatte[probabilities[:, 0]>0.85] = 0

alphaMatte = np.copy(probabilities[:, 1])
alphaMatte[probabilities[:, 1]>0.5] = 1
alphaMatte[probabilities[:, 0]>0.5] = 0
filtAlphaMatte = cv2.GaussianBlur(np.array(alphaMatte*255, dtype=float32), (5, 5), 2.5)

print filtAlphaMatte.shape, np.max(filtAlphaMatte)

figure(); imshow(probabilities[:, 1].reshape(image.shape[:2]), interpolation='nearest')
figure(); imshow(filtAlphaMatte.reshape(image.shape[:2]), interpolation='nearest')

# <codecell>

figure(); imshow(probs.reshape(image.shape[:2]), interpolation='nearest')

# <codecell>

classifier = trainClassifier([cv2.cvtColor(cv2.imread("../data/ribbon2/frame-00001.png"), cv2.COLOR_BGR2RGB), 
                              cv2.cvtColor(cv2.imread("../data/ribbon2/frame-00074.png"), cv2.COLOR_BGR2RGB), 
                              cv2.cvtColor(cv2.imread("../data/ribbon2/frame-000213.png"), cv2.COLOR_BGR2RGB)], 
                             [cv2.imread("../data/ribbon2/trimap-frame-00001.png"), 
                              cv2.imread("../data/ribbon2/trimap-frame-00074.png"), 
                              cv2.imread("../data/ribbon2/trimap-frame-00213.png")])

# <codecell>

image = cv2.cvtColor(cv2.imread("../data/pendulum/000020.png"), cv2.COLOR_BGR2RGB)
figure(); imshow(image)
indices = np.indices(image.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.concatenate((image, indices), axis=-1)
probabilities = classifier.predict_proba(data.reshape((-1, 5)))

# alphaMatte = np.copy(probabilities[:, 1])
# alphaMatte[probabilities[:, 1]>0.15] = 1
# alphaMatte[probabilities[:, 0]>0.85] = 0

alphaMatte = np.copy(probabilities[:, 1])
alphaMatte[probabilities[:, 1]>0.5] = 1
alphaMatte[probabilities[:, 0]>0.5] = 0
filtAlphaMatte = cv2.GaussianBlur(np.array(alphaMatte*255, dtype=float32), (5, 5), 2.5)

print filtAlphaMatte.shape, np.max(filtAlphaMatte)

figure(); imshow(probabilities[:, 1].reshape(image.shape[:2]), interpolation='nearest')
figure(); imshow(filtAlphaMatte.reshape(image.shape[:2]), interpolation='nearest')

# <codecell>

import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from glob import glob
import os

files = sorted(glob("input_frame-*.png"))

for filename in files:
    image = imread(filename)
    # assume background is black
    background = np.zeros(image.shape)
    # assume alpha == image brightness
    alpha = rgb2gray(image)
    # matting assumes that
    # image = alpha * foreground + (1 - alpha) * background
    foreground = (image - (1.0 - alpha[..., None]) * background)
    # for some reason if you divide by alpha it doesn't work
    # even though according to the equations it is right
    # look up premultiplied alpha which never fully makes sense to me!
    result = np.empty(image.shape[:2] + (4,), dtype=np.uint8)
    result[..., 3] = alpha * 255.0
    result[..., :3] = foreground
    output_filename = "{}_foreground.png".format(os.path.basename(filename))
    imsave(output_filename, result)

# <codecell>

img = cv2.cvtColor(cv2.imread("../data/ribbon2/frame-00285.png"), cv2.COLOR_BGR2RGB)
figure(); imshow(img, interpolation='nearest')

## first scribble over foreground and background
scribble = cv2.cvtColor(cv2.imread("../data/ribbon2/frame-00285-scribble.png"), cv2.COLOR_BGR2GRAY)
figure(); imshow(scribble, interpolation='nearest')

## second propragate scribbles to rest of image using watershed
expandedScribble = np.zeros(scribble.shape, dtype=int32)
expandedScribble[scribble == 0] = 1
expandedScribble[scribble == 255] = 2
cv2.watershed(img, expandedScribble)
mask = np.zeros(expandedScribble.shape)
mask[expandedScribble == 2] = 1
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1,1),np.uint8))
figure(); imshow(mask, interpolation='nearest')

edges = cv2.Canny(np.array(mask, dtype=np.uint8), 1, 2)
edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=3)
figure(); imshow(edges, interpolation='nearest')

trimap = np.zeros(mask.shape)
trimap[mask == 1] = 2
trimap[edges == np.max(edges)] = 1
figure(); imshow(trimap, interpolation='nearest')

colorTrimap = np.zeros(np.hstack((trimap.shape, 3)), dtype=uint8)
bgIdx = np.argwhere(trimap == 0)
colorTrimap[bgIdx[:, 0], bgIdx[:, 1], 2] = 255
mgIdx = np.argwhere(trimap == 1)
colorTrimap[mgIdx[:, 0], mgIdx[:, 1], 1] = 255
fgIdx = np.argwhere(trimap == 2)
colorTrimap[fgIdx[:, 0], fgIdx[:, 1], 0] = 255

figure(); imshow(colorTrimap)

## set the borders of trimap to bg as watershed sets the border of the image as a border
# trimap[0, :] = trimap[-1, :] = 0
# trimap[:, 0] = trimap[:, -1] = 0
## trimap has 0 for bg, 2 for fg and 1 for middle ground
# trimap[trimap == 1] = 0
# trimap[trimap == -1] = 1

## make middle ground bigger and allow user to decide how much it describes boundary regions between fg and bg
## use a slider to decide num of iterations and overimpose trimap to img
# middleground = np.zeros(trimap.shape)
# middleground[trimap == 1] = 1
# middleground = cv2.dilate(middleground, np.ones((5,5),np.uint8), iterations=1)
# trimap[middleground == 1] = 1

## use the found trimap to train classifier

## use trained classifier to find a matte, binarize it by thresholding, use watershed to clean it up, blur and matte

# <codecell>

classifier = trainClassifier([img], [trimap])

# <codecell>

image = cv2.cvtColor(cv2.imread("../data/ribbon2/frame-00174.png"), cv2.COLOR_BGR2RGB)
figure(); imshow(image)
indices = np.indices(image.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.concatenate((image, indices), axis=-1)
probabilities = classifier.predict_proba(data.reshape((-1, 5)))
alphaMatte = np.copy(probabilities[:, 1])
alphaMatte[probabilities[:, 1]>0.15] = 1
alphaMatte[probabilities[:, 0]>0.85] = 0
figure(); imshow(probabilities[:, 1].reshape(image.shape[:2]), interpolation='nearest')
figure(); imshow(alphaMatte.reshape(image.shape[:2]), interpolation='nearest')

# <codecell>

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS
matte = np.array(alphaMatte, dtype=float32)
# Apply KMeans
compactness,labels,centers = cv2.kmeans(matte, 2 , criteria, 10, flags)
print matte.shape, compactness, labels.shape, centers.shape
figure(); imshow(labels.reshape(image.shape[:2]))

# <codecell>

X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))

# convert to np.float32
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
print Z.shape, ret, label.shape, center.shape

# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]

# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()

# <codecell>

##tryout the classifier
image = imread("../data/ribbon2/frame-00174.png")
newBg = imread("../data/ribbon1_newbg/bg.png")
indices = np.indices(image.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.concatenate((image, indices), axis=-1)
probabilities = classifier.predict_proba(data.reshape((-1, 5)))
print probabilities.shape
alphaMatte = probabilities[:, 1]
alphaMatte[probabilities[:, 1]>0.5] = 1
alphaMatte[probabilities[:, 0]>0.5] = 0
# alphaMatte[alphaMatte>0] = 1
# alphaMatte[probabilities[:, 1] >= 0.7] = 1
# alphaMatte[probabilities[:, 2] >= 0.7] = 0


# alphaMatte = (1-alphaMatte[:, 0])+(1-alphaMatte[:, 1])+alphaMatte[:, 2]
# alphaMatte = (1-cv2.bilateralFilter(np.array(alphaMatte[:, 0]*255, dtype=float32), 15, 128, 128)/255)+(1-cv2.bilateralFilter(np.array(alphaMatte[:, 1]*255, dtype=float32), 15, 128, 128)/255)+cv2.bilateralFilter(np.array(alphaMatte[:, 2]*255, dtype=float32), 15, 128, 128)/255
# alphaMatte /= max(alphaMatte)
# alphaMatte = alphaMatte[:, 2]
# print alphaMatte[272000, 0], alphaMatte[272000, 1], alphaMatte[272000, 2]
# print max(alphaMatte)
alphaMatte = alphaMatte.reshape(image.shape[:2])
figure(); imshow(probabilities[:, 0].reshape(image.shape[:2]), interpolation='nearest')
figure(); imshow(alphaMatte, interpolation='nearest')
# filtAlphaMatte = cv2.bilateralFilter(np.array(alphaMatte*255, dtype=float32), 15, 128, 128)/255
# filtAlphaMatte = cv2.blur(np.array(alphaMatte*255, dtype=float32), (7, 7))/255
# filtAlphaMatte = cv2.GaussianBlur(np.array(alphaMatte*255, dtype=float32), (5, 5), 2.5)/255
filtAlphaMatte = cv2.medianBlur(np.array(alphaMatte*255, dtype=float32), 5)/255
filtAlphaMatte = cv2.GaussianBlur(filtAlphaMatte, (7, 7), 10.0)
figure(); imshow(filtAlphaMatte, interpolation='nearest')

filtAlphaMatte = np.repeat(np.reshape(filtAlphaMatte, np.hstack((image.shape[0:2], 1))), image.shape[-1], axis=-1)
newImg = (image/255.0)*filtAlphaMatte+(newBg/255.0)*(1-filtAlphaMatte)
# newImg = np.round(newBg*ones_like(matte))
# newImg[:, :, -1] = (filtResult/255)
# newImg[mask==1] = image[mask==1]
figure(); imshow(newImg, interpolation='nearest')
figure(); imshow(image, interpolation='nearest')

