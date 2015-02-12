# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

import pycuda.autoinit
import pycuda.driver as drv

import numpy as np
import Image

cudaFolder = "/home/ilisescu/PhD/cuda/"

# <codecell>

img1 = Image.open("/home/ilisescu/PhD/data/ribbon2/frame-00285.png")
(w,h) = img1.size
img1 = np.array(img1.getdata()).reshape(3*w*h,1).astype(numpy.float32) / 255.0
# img1 = img1[0:100]

img2 = Image.open("/home/ilisescu/PhD/data/ribbon2/frame-01027.png")
(w,h) = img2.size
img2 = np.array(img2.getdata()).reshape(3*w*h,1).astype(numpy.float32) / 255.0
# img2 = img2[0:100]

res = numpy.zeros(img2.shape).astype(numpy.float32)

N = np.int32(len(img1))

print img1.shape, img2.shape, res.shape, N

# <codecell>

module = drv.module_from_file(cudaFolder + "l2dist/l2dist.cubin")
l2dist = module.get_function("l2dist")

# <codecell>

print np.int32(res.shape[0])

# <codecell>

%timeit l2dist(drv.In(img1), drv.In(img2), N, drv.Out(res), block=(256, 1, 1), grid=(64*8, 1))#grid=(res.shape[0]/256, 1))

# <codecell>

print img1[2023800], img2[2023800], res[2023800], img1[2023800] - img2[2023800]

# <codecell>

print np.sum(res)

# <codecell>

%timeit diffs = np.sqrt(np.power(img1-img2, 2))
%timeit np.sum(diffs)

