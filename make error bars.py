# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab
import numpy as np

# <codecell>

usInteraction = np.array([1721, 1392, 1474])-np.array([660, 585, 780])
# usComputationSpeedUp = 3.7/40.0 ## speed up from 40 secs per optimization (for 300 output frames) to 3.7 second using compression of 2
# usComputationSpeedUp = 1.0/40.0 ## speed up from 40 secs per optimization (for 300 output frames) to 1 second using compression of 3
usComputationSpeedUp = 0.5/40.0 ## speed up from 40 secs per optimization (for 300 output frames) to 0.5 second using compression of 4
usComputation = np.array([660, 585, 780])*usComputationSpeedUp
print usComputation
usTime = usComputation+usInteraction
aeTime = np.array([2050, 1620, 1380])
expertTime = np.array([810, 4804, 3261])
error_kw={'ecolor':'midnightblue', 'capthick':'3', 'elinewidth':'3'}
color=('greenyellow', 'greenyellow', 'orangered', 'orangered')
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

# <codecell>

figure(figsize=(25, 3), dpi=200)
means = np.array([np.mean(usTime), np.mean(aeTime), np.mean(expertTime)])
pos = np.arange(len(means))
stds = np.array([np.std(usTime), np.std(aeTime), np.std(expertTime)])
barh(pos, means, height=0.7, xerr=stds, error_kw=error_kw, align='center', color=color[1:])
yticks(pos, ('Ours', 'AfterEffects', 'Experts'))
xlabel('Time [Secs]')
tight_layout()

# <codecell>

figure(figsize=(25, 4), dpi=200)
means = np.array([np.mean(usTime), np.mean(usTime-usComputation), np.mean(aeTime), np.mean(expertTime)])
pos = np.arange(len(means))
stds = np.array([np.std(usTime), np.std(usTime-usComputation), np.std(aeTime), np.std(expertTime)])
barh(pos, means, height=0.7, xerr=stds, error_kw=error_kw, align='center', color=color)
yticks(pos, ('Ours Total', 'Ours Interaction', 'AfterEffects', 'Experts'))
xlabel('Time [Secs]')
tight_layout()

