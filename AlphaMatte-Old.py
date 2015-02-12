# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
from sklearn import tree
from sklearn import ensemble
from sklearn import semi_supervised
from skimage.io import imread
# from skimage.io import imsave
from glob import glob
from PIL import Image
import cv2
import re

# <codecell>

def trainClassifier3(data_frames, trimaps) :
    # train on first frame
    # augment with x-y positional data
    clf = ensemble.ExtraTreesClassifier()
    
    print len(data_frames), len(trimaps)
    
    idxs = np.indices(data_frames[0].shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
    data = np.concatenate((data_frames[0], idxs), axis=-1)

    # extract training data
    background = data[trimaps[0] == 0]
    foreground = data[trimaps[0] == 255]
    midground = data[trimaps[0] == 64]
    
    for data_frame, trimap in zip(data_frames[1:], trimaps[1:]) :
        print background.shape, foreground.shape, midground.shape
        
        idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
        data = np.concatenate((data_frame, idxs), axis=-1)
    
        # extract training data
        background = np.vstack((background, data[trimap == 0]))
        foreground = np.vstack((foreground, data[trimap == 255]))
        midground = np.vstack((midground, data[trimap == 64]))
        
    print background.shape, foreground.shape, midground.shape
    
    X = np.vstack((background, midground, foreground))
    y = np.repeat([0.0, 1.0, 2.0], [background.shape[0], midground.shape[0], foreground.shape[0]])
    print clf.fit(X, y)
    return clf

# <codecell>

def trainClassifier2(data_frames, trimaps) :
    # train on first frame
    # augment with x-y positional data
    clf = ensemble.ExtraTreesClassifier()
    
    print len(data_frames), len(trimaps)
    
    idxs = np.indices(data_frames[0].shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
    data = np.concatenate((data_frames[0], idxs), axis=-1)

    # extract training data
    background = data[trimaps[0] == 0]
    foreground = data[trimaps[0] == 255]
    
    for data_frame, trimap in zip(data_frames[1:], trimaps[1:]) :
        print background.shape, foreground.shape
        
        idxs = np.indices(data_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
        data = np.concatenate((data_frame, idxs), axis=-1)
    
        # extract training data
        background = np.vstack((background, data[trimap == 0]))
        foreground = np.vstack((foreground, data[trimap == 255]))
        
    print background.shape, foreground.shape
    
    X = np.vstack((background, foreground))
    y = np.repeat([0.0, 1.0], [background.shape[0], foreground.shape[0]])
    print clf.fit(X, y)
    return clf

# <codecell>

def matteImages(clf, filenames, newBg) :
    image = imread(filenames[0])
    idxs = np.indices(image.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
    for filename in filenames:
        image = imread(filename)
        data = np.concatenate((image, idxs), axis=-1)
        probabilities = classifier.predict_proba(data.reshape((-1, 5)))
        ## old way
#         alphaMatte = alphaMatte[:, 2].reshape(image.shape[:2])
#         filtAlphaMatte = cv2.bilateralFilter(np.array(alphaMatte*255, dtype=float32), 15, 128, 128)/255
        ##
        ## new way
        alphaMatte = probabilities[:, 2]
        alphaMatte[probabilities[:, 0]>0.5] = 0
        alphaMatte[probabilities[:, 1]>0.5] = 0
        filtAlphaMatte = cv2.GaussianBlur(np.array(alphaMatte*255, dtype=float32), (5, 5), 2.5)/255
        ##
        
        filtAlphaMatte = np.repeat(np.reshape(filtAlphaMatte, np.hstack((image.shape[0:2], 1))), image.shape[-1], axis=-1)
        
        if newBg != None :
            newImg = (newBg/255.0)*(1-filtAlphaMatte)+(image/255.0)*filtAlphaMatte
        else :
            newImg = (image/255.0)*filtAlphaMatte
            
#         figure(); imshow(newImg, interpolation='nearest')

        sys.stdout.write('\r' + "Processing " + np.string_(filter(None, re.split('/',filename))[-1]))
        sys.stdout.flush()
        imsave(np.string_('/'.join(filter(None, re.split('/',filename))[0:-1]) + "/matte-" + filter(None, re.split('/',filename))[-1]), filtAlphaMatte)
        imsave(filename, newImg)

# <codecell>

testImg = imread("../data/ribbon2/frame-00838.png")
testMap = imread("../data/ribbon2/frame-00838-trimap.png")
print testImg.shape, testMap.shape

# <codecell>

print '/'.join(filter(None, re.split('[/]',"/data/ribbon2/frame-00838.png"))[0:-1])

# <codecell>

# classifier = trainClassifier(imread("../data/ribbon1/frame-00001.png"), imread("../data/ribbon1/trimap2.png"))
classifier = trainClassifier2([imread("../data/ribbon2/frame-00001.png"), imread("../data/ribbon2/frame-00285.png"), imread("../data/ribbon2/frame-00838.png")], 
                            [imread("../data/ribbon2/frame-00001-trimap.png"), imread("../data/ribbon2/frame-00285-trimap.png"), imread("../data/ribbon2/frame-00838-trimap.png")])

# <codecell>

files = sorted(glob("../data/ribbon2_matted/frame*.png"))
matteImages(classifier, files[2:], None)#imread("../data/ribbon1_newbg/bg.png"))

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

# <codecell>

inputImage = imread("../data/ribbon2/frame-00001.png")
inputTrimap = imread("../data/ribbon2/frame-00001-trimap.png")
## adapt for watershed
markers = np.zeros(inputTrimap.shape)
markers[inputTrimap == 0] = 1
markers[inputTrimap == 255] = 2
markers = np.array(markers, dtype=int32)
print markers.shape
figure(); imshow(inputImage, interpolation='nearest')
figure(); imshow(inputTrimap, interpolation='nearest')
figure(); imshow(markers, interpolation='nearest')
print inputImage.shape
cv2.watershed(inputImage, markers)
print markers
resultImage = np.copy(inputImage)
resultImage[markers != 2] = [0, 0, 0]
figure(); imshow(resultImage, interpolation='nearest')

# <codecell>

%pylab
import cv2
import numpy as np
middleground = np.zeros((200, 200))
middleground[100, 100] = 1
kernel = np.ones((5,5),np.uint8)
middleground = cv2.dilate(middleground, kernel, iterations=3)
figure(); imshow(middleground, interpolation='nearest')

# <codecell>

import GraphWithValues as gwv
gwv.showCustomGraph(middleground)

# <codecell>

##tryout the classifier
image = imread("../data/ribbon2/frame-00174.png")
newBg = imread("../data/ribbon1_newbg/bg.png")
indices = np.indices(image.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.concatenate((image, indices), axis=-1)
probabilities = classifier.predict_proba(data.reshape((-1, 5)))
print probabilities.shape
alphaMatte = probabilities[:, 2]
alphaMatte[probabilities[:, 0]>0.5] = 0
alphaMatte[probabilities[:, 1]>0.5] = 0
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
figure(); imshow(probabilities[:, 1].reshape(image.shape[:2]), interpolation='nearest')
figure(); imshow(alphaMatte, interpolation='nearest')
# filtAlphaMatte = cv2.bilateralFilter(np.array(alphaMatte*255, dtype=float32), 15, 128, 128)/255
# filtAlphaMatte = cv2.blur(np.array(alphaMatte*255, dtype=float32), (7, 7))/255
filtAlphaMatte = cv2.GaussianBlur(np.array(alphaMatte*255, dtype=float32), (5, 5), 2.5)/255
figure(); imshow(filtAlphaMatte, interpolation='nearest')

filtAlphaMatte = np.repeat(np.reshape(filtAlphaMatte, np.hstack((image.shape[0:2], 1))), image.shape[-1], axis=-1)
newImg = (image/255.0)*filtAlphaMatte+(newBg/255.0)*(1-filtAlphaMatte)
# newImg = np.round(newBg*ones_like(matte))
# newImg[:, :, -1] = (filtResult/255)
# newImg[mask==1] = image[mask==1]
figure(); imshow(newImg, interpolation='nearest')
figure(); imshow(image, interpolation='nearest')

# <codecell>

files = sorted(glob("../data/ribbon1/frame*.png"))
trimap = imread("../data/ribbon1/trimap2.png")
newBg = imread("../data/ribbon1_newbg/bg.png")

# <codecell>

files = sorted(glob("../data/ribbon1/frame*.png"))
trimap = Image.open("../data/ribbon1/trimap2.png")
newBg = imread("../data/ribbon1_newbg/bg.png")

# <codecell>

for filename in files:
    image = imread(filename)
    data = np.concatenate((image, indices), axis=-1)
    result = classifier.predict(data.reshape((-1, 5)))
    mask = np.repeat(result.reshape(np.hstack((image.shape[:2], 1))), image.shape[-1], axis=-1)
    newImg = np.zeros_like(image)
    newImg[mask==0] = newBg[mask==0] 
    newImg[mask==1] = image[mask==1]
    imsave(filename, newImg)

# <codecell>

# train on first frame
# augment with x-y positional data
classifier = ensemble.ExtraTreesClassifier()
first_frame = imread(files[0])

indices = np.indices(first_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.concatenate((first_frame, indices), axis=-1)
print data.shape

# extract training data
background = data[trimap == 0]
print background.shape
foreground = data[trimap == 255]
print foreground.shape

X = np.vstack((background, foreground))
print X.shape
y = np.repeat([0.0, 1.0], [background.shape[0], foreground.shape[0]])
print y.shape
classifier.fit(X, y)

# <codecell>

image = imread(files[897])
data = np.concatenate((image, indices), axis=-1)
result = classifier.predict_proba(data.reshape((-1, 5)))
result = result[:, 1].reshape(image.shape[:2])
print result.shape
figure(); imshow(result, interpolation='nearest')

# <codecell>

filtResult = cv2.bilateralFilter(np.array(result*255, dtype=float32), 15, 128, 128)
figure(); imshow(filtResult, interpolation='nearest')

# <codecell>

print filtResult[250:300, 1150:1200]/255

# <codecell>

# newImg = np.zeros_like(image)
matte = np.repeat(np.reshape(filtResult/255, np.hstack((newImg.shape[0:2], 1))), newImg.shape[-1], axis=-1)
newImg = (newBg/255.0)*(1-matte)+(image/255.0)*matte
# newImg = np.round(newBg*ones_like(matte))
# newImg[:, :, -1] = (filtResult/255)
# newImg[mask==1] = image[mask==1]
figure(); imshow(newImg, interpolation='nearest')
figure(); imshow(image, interpolation='nearest')

# <codecell>

testImg = image[250:300, 1150:1200, :]/255.0
testMatte = matte[250:300, 1150:1200, :]
print testImg.shape, testMatte.shape
figure(); imshow(testImg, interpolation='nearest')
figure(); imshow(testMatte, interpolation='nearest')
figure(); imshow(testImg*testMatte, interpolation='nearest')
print (testImg*testMatte).shape
print testImg[0, 0, :]
print testMatte[0, 0, :]
print testImg[0, 0, :]*testMatte[0, 0, :]/255

# <codecell>

# train on first frame
# augment with x-y positional data
# trimap = np.array(trimap.resize(np.array(trimap.size)/2))
figure(); imshow(trimap, interpolation='nearest')
first_frame = Image.open(files[0])
first_frame = np.array(first_frame.resize(np.array(first_frame.size)/2))
classifier = semi_supervised.LabelPropagation()
# first_frame = imread(files[0])

indices = np.indices(first_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.concatenate((first_frame, indices), axis=-1)
print data.shape

# extract training data
background = data[trimap == 0]
print background.shape
foreground = data[trimap == 255]
print foreground.shape
unlabeled = data[np.logical_and(trimap != 255, trimap != 0)]
print unlabeled.shape

X = np.vstack((background, foreground, unlabeled))
print X.shape
y = np.repeat([0, 1, -1], [background.shape[0], foreground.shape[0], unlabeled.shape[0]])
print y.shape
classifier.fit(X, y)

# <codecell>

# train on first frame
# augment with x-y positional data
classifier = tree.DecisionTreeClassifier(random_state=0)
first_frame = imread(files[0])

indices = np.indices(first_frame.shape[:2]).swapaxes(-1, 0).swapaxes(0, 1)
data = np.concatenate((first_frame, indices), axis=-1)
print data.shape

# extract training data
background = data[trimap == 0]
print background.shape
foreground = data[trimap == 255]
print foreground.shape

X = np.vstack((background, foreground))
print X.shape
y = np.repeat([0.0, 1.0], [background.shape[0], foreground.shape[0]])
print y.shape
classifier.fit(X, y)

# <codecell>

image = imread(files[0])
data = np.concatenate((image, indices), axis=-1)
result = classifier.predict_log_proba(data.reshape((-1, 5)))
figure(); imshow(result[:, 0].reshape(image.shape[:2]), interpolation='nearest')

# <codecell>

for filename in files:
    image = imread(filename)
    data = np.concatenate((image, indices), axis=-1)
    result = classifier.predict(data.reshape((-1, 5)))
    mask = np.repeat(result.reshape(np.hstack((image.shape[:2], 1))), image.shape[-1], axis=-1)
    newImg = np.zeros_like(image)
    newImg[mask==0] = newBg[mask==0] 
    newImg[mask==1] = image[mask==1]
    imsave(filename, newImg)

