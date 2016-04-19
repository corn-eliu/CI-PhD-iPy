# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

import numpy as np
import Image
import time

cudaFolder = "/home/ilisescu/PhD/cuda/"

# <codecell>

img1 = Image.open("/media/ilisescu/Data1/PhD/data/ribbon2/frame-00285.png")
(w,h) = img1.size
img1 = np.array(img1.getdata()).reshape(w*h,3).astype(numpy.float32) / 255.0
# img1 = np.concatenate((img1[0:144, :], np.zeros((144, 1), numpy.float32)), axis=-1)
img1 = np.concatenate((img1, np.zeros((img1.shape[0], 1), numpy.float32)), axis=-1)

img2 = Image.open("/media/ilisescu/Data1/PhD/data/ribbon2/frame-01027.png")
(w,h) = img2.size
img2 = np.array(img2.getdata()).reshape(w*h,3).astype(numpy.float32) / 255.0
# img2 = np.concatenate((img2[0:144, :], np.zeros((144, 1), numpy.float32)), axis=-1)
img2 = np.concatenate((img2, np.zeros((img2.shape[0], 1), numpy.float32)), axis=-1)


# displacements = np.array([1, 2])
displacements = np.array([1, 2, 4, 8, 16])
displacements = np.concatenate((np.array([np.zeros(len(displacements)), displacements], int), ## EAST
                                np.array([displacements, displacements], int), ## SOUT-EAST
                                np.array([displacements, np.zeros(len(displacements))], int), ## SOUTH
                                np.array([displacements, -displacements], int), ## SOUTH-WEST
                                np.array([np.zeros(len(displacements)), -displacements], int), ## WEST
                                np.array([-displacements, -displacements], int), ## NORTH-WEST
                                np.array([-displacements, np.zeros(len(displacements))], int), ## NORTH
                                np.array([-displacements, displacements], int), ## NORTH-EAST
                                ), axis=-1).T.astype(np.int32)
## flipping the columns o I can get x, y coords
displacements = displacements[:, ::-1]


N = np.int32(len(img1))
# spacing = np.int32(3)
spacing = np.int32(32)
offset = np.array(np.max(displacements, axis=0)[::-1], np.int32)
# patchSize = np.array([4, 4], np.int32)
patchSize = np.array([64, 64], np.int32)
# imageSize = np.array([12, 12], np.int32)
imageSize = np.array([h, w], np.int32)


gridSize = np.array([np.arange(offset[0], imageSize[0]-patchSize[0]-offset[0], spacing).shape[0],
                     np.arange(offset[1], imageSize[1]-patchSize[1]-offset[1], spacing).shape[0]])
# gridSize = np.array([2, 1])

res = numpy.zeros((np.prod(gridSize), displacements.shape[0])).astype(np.float32)

print img1.shape, img2.shape, res.shape, N, gridSize, offset

# <codecell>

print img1.nbytes

# <codecell>

module = drv.module_from_file(cudaFolder + "computeFeat/computeFeat.cubin")
computeFeat = module.get_function("computeFeat")

# <codecell>

t = time.time()

# ## allocate graphics memory
# img1_gpu = drv.mem_alloc(img1.nbytes)
# img2_gpu = drv.mem_alloc(img2.nbytes)
# displacements_gpu = drv.mem_alloc(displacements.nbytes)
# res_gpu = drv.mem_alloc(res.nbytes)

# ## fill buffers
# drv.memcpy_htod(img1_gpu, img1.flatten())
# drv.memcpy_htod(img2_gpu, img2.flatten())
# drv.memcpy_htod(displacements_gpu, displacements.flatten())
# drv.memcpy_htod(res_gpu, res.flatten())

print drv.mem_get_info()
t = time.time()

img1_gpu = gpuarray.to_gpu(img1)
img2_gpu = gpuarray.to_gpu(img2)
displacements_gpu = gpuarray.to_gpu(displacements.flatten())
res_gpu = gpuarray.to_gpu(numpy.zeros((np.prod(gridSize), displacements.shape[0])).astype(np.float32))


patchSize_gpu = gpuarray.to_gpu(patchSize)
offset_gpu = gpuarray.to_gpu(offset)
imageSize_gpu = gpuarray.to_gpu(imageSize)

## run kernel
# computeFeat(img1_gpu, img2_gpu, displacements_gpu, patchSize_gpu, offset_gpu, spacing, imageSize_gpu, res_gpu, block=(displacements.shape[0], 1, 1), grid=(gridSize[1], gridSize[0]))

## read result
# res = np.empty_like(res)
# drv.memcpy_dtoh(res, res_gpu)
# res = res_gpu.get()

# print drv.mem_get_info()

print time.time()-t

# <codecell>

# del img1_gpu, img2_gpu, displacements_gpu, res_gpu, patchSize_gpu, offset_gpu, imageSize_gpu
print 3508944896 - 3477225472
print displacements.dtype

# <codecell>

print img1.nbytes + img2.nbytes + displacements.nbytes + res.nbytes + patchSize.nbytes + offset.nbytes + imageSize.nbytes

# <codecell>

print res.flatten() - result

# <codecell>

(1280*720*3*4*2)/1000000

# <codecell>

print result

# <codecell>

# rows = np.copy(res)
print rows

# <codecell>

# img1idx = np.copy(res.astype(int))
print img1idx

# <codecell>

# img2idx = np.copy(res.astype(int))
print img2idx

# <codecell>

patchYs = np.arange(offset[0], imageSize[0]-patchSize[0]-offset[0], spacing)
patchYs = patchYs.reshape((1, len(patchYs)))
patchXs = np.arange(offset[1], imageSize[1]-patchSize[1]-offset[1], spacing)
patchXs = patchXs.reshape((1, len(patchXs)))
patchLocations = np.array(np.concatenate((patchYs.repeat(len(patchXs.T)),
                                          patchXs.repeat(len(patchYs.T), axis=0).flatten())).reshape((2, len(patchXs.T)*len(patchYs.T))), int)

frame1Idxs = np.empty((0, 2), dtype=int)
frame2Idxs = np.empty((0, 2), dtype=int)
for i, locSlice in enumerate(np.arange(patchXs.shape[-1], patchLocations.shape[-1]+1, patchXs.shape[-1])) :
    frame1Idxs = np.concatenate((frame1Idxs, np.repeat(patchLocations[:, locSlice-patchXs.shape[-1]:locSlice], len(displacements[:, ::-1]), axis=1).T), axis=0)
    frame2Idxs = np.concatenate((frame2Idxs, np.array([disp + loc for loc in patchLocations[:, locSlice-patchXs.shape[-1]:locSlice].T for disp in displacements[:, ::-1]])), axis=0)

    
frame1 = np.array(Image.open("/media/ilisescu/Data1/PhD/data/ribbon2/frame-00285.png"))/255.0
frame2 = np.array(Image.open("/media/ilisescu/Data1/PhD/data/ribbon2/frame-01027.png"))/255.0

result = np.array([np.sqrt(np.sum((frame2[l2[0]:l2[0]+patchSize[0], l2[1]:l2[1]+patchSize[1]]-frame1[l1[0]:l1[0]+patchSize[0], l1[1]:l1[1]+patchSize[1]])**2)) for l2, l1 in zip(frame2Idxs, frame1Idxs)])

print result.shape

# <codecell>

print np.max(np.abs(result-res.flatten()))
print np.min(np.abs(result-res.flatten()))
print np.max(np.abs(result-res.flatten())/np.abs(result))

# <codecell>

tmp1Idxs = np.array([np.array(np.meshgrid(arange(l1[0], l1[0]+patchSize[0]), 
                                          arange(l1[1], l1[1]+patchSize[1]))).reshape((2, np.prod(patchSize))) for l1 in frame1Idxs]).swapaxes(0, 1).reshape((2, len(frame1Idxs)*np.prod(patchSize)))

tmp2Idxs = np.array([np.array(np.meshgrid(arange(l2[0], l2[0]+patchSize[0]), 
                                          arange(l2[1], l2[1]+patchSize[1]))).reshape((2, np.prod(patchSize))) for l2 in frame2Idxs]).swapaxes(0, 1).reshape((2, len(frame2Idxs)*np.prod(patchSize)))
print tmp1Idxs.shape

# <codecell>

result2 = np.sqrt(np.sum(np.sum((frame2[tmp2Idxs[0, :], tmp2Idxs[1, :], :] - frame1[tmp1Idxs[0, :], tmp1Idxs[1, :], :])**2, axis=-1).reshape((len(frame1Idxs), np.prod(patchSize))), axis=-1))

# <codecell>

print result
print res.flatten()

# <codecell>

print result - res.flatten()
print res.flatten()
print np.min(np.abs(res.flatten()-result))
print np.max(np.abs(res.flatten()-result))

# <codecell>

print frame1Idxs[:40, :]
print frame2Idxs[:40, :]
print displacements[29, :]

# <codecell>

# %timeit computeFeat(img1_gpu, img2_gpu, displacements_gpu, patchSize, offset, spacing, imgSize, res_gpu, block=(displacements.shape[0], 1, 1), grid=(2, 1))
print res.shape

# <codecell>

computeFeat(drv.In(img1), drv.In(img2), drv.In(displacements.flatten()), offset, spacing, drv.Out(res), block=(20, 1, 1), grid=(6, 4))

# <codecell>

%timeit (img1-img2)**2

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

