# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import tsne as tsne

# <codecell>

X = np.loadtxt("mnist2500_X.txt");
labels = np.loadtxt("mnist2500_labels.txt");

# <codecell>

print X.shape
print labels.shape
print np.min(X), np.max(X)

# <codecell>

tsne.tsne(

