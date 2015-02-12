# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np
from glob import glob
from PIL import Image
import cv2
import re

# <codecell>

def changeBGofAlphaImages(filenames, newBg) :
    for filename in filenames:
        image = np.array(Image.open(filename))
        
        alphaMatte = image[:, :, -1]
        image = image[:, :, 0:3]
        
#         figure(); imshow(alphaMatte)
        
        
        alphaMatte = np.repeat(np.reshape(alphaMatte, np.hstack((image.shape[0:2], 1))), image.shape[-1], axis=-1)
        
        newImg = (newBg/255.0)*(1-alphaMatte/255.0)+(image/255.0)*(alphaMatte/255.0)
#         if newBg != None :
#             newImg = (newBg/255.0)*(1-filtAlphaMatte)+(image/255.0)*filtAlphaMatte
#         else :
#             newImg = (image/255.0)*filtAlphaMatte
            
#         figure(); imshow(newImg, interpolation='nearest')

        sys.stdout.write('\r' + "Processing " + np.string_(filter(None, re.split('/',filename))[-1]))
        sys.stdout.flush()
#         imsave(np.string_('/'.join(filter(None, re.split('/',filename))[0:-1]) + "/matte-" + filter(None, re.split('/',filename))[-1]), filtAlphaMatte)
        imsave(filename, newImg)

# <codecell>

def setAlphaToImages(imgs, mts) :
    for img, mt in zip(imgs, mts):
        image = cv2.imread(img, cv2.COLOR_BGR2RGB)
        matte = Image.open(mt)
        matte = np.reshape(np.array(matte, dtype=uint8)[:, :, 0], np.hstack((image.shape[0:2], 1)))
#         print image.shape, matte.shape
        
#         figure(); imshow(image)
#         figure(); imshow(matte)
        
#         newImg = np.zeros(np.hstack((image.shape[0:-1], 4)))
#         newImg[:, :, 0:-1] = image
#         newImg[:, :, -1] = matte
#         figure(); imshow(newImg)        
        
        Image.fromstring("RGBA", (image.shape[1], image.shape[0]), np.concatenate((image, matte), axis=-1).tostring()).save(img)
        
#         cv2.imwrite(img, cv2.cvtColor(newImg, cv2.COLOR_RGBA2BGRA))
#         imsave(img, newImg)
        
#         alphaMatte = image[:, :, -1]
#         image = image[:, :, 0:3]
        
# #         figure(); imshow(alphaMatte)
        
        
#         alphaMatte = np.repeat(np.reshape(alphaMatte, np.hstack((image.shape[0:2], 1))), image.shape[-1], axis=-1)
        
#         newImg = (newBg/255.0)*(1-alphaMatte/255.0)+(image/255.0)*(alphaMatte/255.0)
# #         if newBg != None :
# #             newImg = (newBg/255.0)*(1-filtAlphaMatte)+(image/255.0)*filtAlphaMatte
# #         else :
# #             newImg = (image/255.0)*filtAlphaMatte
            
# #         figure(); imshow(newImg, interpolation='nearest')

#         sys.stdout.write('\r' + "Processing " + np.string_(filter(None, re.split('/',filename))[-1]))
#         sys.stdout.flush()
# #         imsave(np.string_('/'.join(filter(None, re.split('/',filename))[0:-1]) + "/matte-" + filter(None, re.split('/',filename))[-1]), filtAlphaMatte)
#         imsave(filename, newImg)

# <codecell>

images = sorted(glob("../data/ribbon2_transparent/frame*.png"))
mattes = sorted(glob("../data/ribbon2_transparent/matte*.png"))
setAlphaToImages(images[41:], mattes[41:])

# <codecell>

print images[41:]

# <codecell>

files = sorted(glob("../eu_flag_newbg/vt*.png"))
changeBGofAlphaImages(files, np.array(Image.open("../eu_flag_newbg/bg.png"))[:, :, 0:3])

