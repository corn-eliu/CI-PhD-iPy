# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab

# <codecell>

tmp = zeros(1280)
for i in range(1280) :
    tmp[i] += 1
    for j in range(i, 1280) :
        tmp[j] += 1

# <codecell>

print np.average(tmp)

