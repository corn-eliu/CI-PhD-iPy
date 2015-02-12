# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

% pylab
import numpy as np
import cv2

# <codecell>

def transition(value, maximum, start_point, end_point):
    return start_point + (end_point - start_point)*value/maximum

def transition3(value, maximum, (s1, s2, s3), (e1, e2, e3)):
    r1= transition(value, maximum, s1, e1)
    r2= transition(value, maximum, s2, e2)
    r3= transition(value, maximum, s3, e3)
    return (r1, r2, r3)


start_triplet = np.reshape((0, 255, 0), (1, 1, 3))
start_triplet= matplotlib.colors.rgb_to_hsv(start_triplet) #comment: green converted to HSV
start_triplet = np.ndarray.flatten(start_triplet)
end_triplet = np.reshape((255, 0, 0), (1, 1, 3))
end_triplet= matplotlib.colors.rgb_to_hsv(end_triplet) #comment: accordingly for red
end_triplet = np.ndarray.flatten(end_triplet)

print start_triplet, end_triplet
    
# print transition(arange(0, 50), 1, 0, 49)
grad = np.array(transition3(arange(0, 50), 49, start_triplet, end_triplet), dtype=np.float32)
print grad
grad = np.reshape(grad.T, (1, len(grad.T), 3))
print grad
grad = np.repeat(grad, 5, axis=0)
print grad.shape
figure(); imshow(cv2.cvtColor(grad, cv2.COLOR_HSV2RGB))

# <codecell>

gradColors = np.zeros((1, 1280, 3), dtype=np.float32)
gradColors[:, :, -1] = np.ones(gradColors.shape[0:2])
gradColors[:, :, -2] = np.ones(gradColors.shape[0:2])
gradColors[:, :, 0] = np.arange(0.0, 360.0, 360.0/gradColors.shape[1])
gradColors = np.repeat(gradColors, 50, axis=0)
print gradColors.shape
figure(); imshow(cv2.cvtColor(gradColors, cv2.COLOR_HSV2RGB))

