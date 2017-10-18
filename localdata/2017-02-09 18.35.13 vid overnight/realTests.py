import sys
sys.path.insert(0, '../j_postacquisition/')
import image_loading as jld
sys.path.insert(0, '.')#because this file is run from this folder but not actually in this folder
import accountForDrift as afd
import nCascadingNW as cnw
import simpleCC as scc

import numpy as np
import scipy.ndimage as sp

# load all sequences and convert to ordered lists
sequenceName = './localdata/2017-02-09 18.35.13 vid overnight/seq{0}/'
blurSigma = 3

#seq0
stackNumber = 0
sequenceObj = jld.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq0 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq0[i] = np.asarray(sp.gaussian_filter(sequenceObj[i].image,sigma=blurSigma),dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq0 = seq0[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq0)))

#seq1
stackNumber = 1
sequenceObj = jld.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq1 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq1[i] = np.asarray(sp.gaussian_filter(sequenceObj[i].image,sigma=blurSigma),dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq1 = seq1[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq1)))

#seq2
stackNumber = 2
sequenceObj = jld.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq2 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq2[i] = np.asarray(sp.gaussian_filter(sequenceObj[i].image,sigma=blurSigma),dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq2 = seq2[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq2)))

#seq3
stackNumber = 3
sequenceObj = jld.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq3 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq3[i] = np.asarray(sp.gaussian_filter(sequenceObj[i].image,sigma=blurSigma),dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq3 = seq3[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq3)))

# settings for drift
settings0 = {}
settings0.update({'drift':[0,0]})#starting drift corrections[-6,-1]
settings1 = {}
settings1.update({'drift':[1,-1]})#starting drift corrections[-5,-2]
settings2 = {}
settings2.update({'drift':[0,-1]})#starting drift corrections[-6,-2]
settings3 = {}
settings3.update({'drift':[12,7]})#starting drift corrections[6,6]

# seq0 with seq0
print('Comparing sequence #0 with itself...')
s0,s0b = afd.matchFrames(seq0,seq0,settings0['drift'])

print('n-Cascading Needleman-Wunsch Algorithm:')
alignment1, alignment2, rollFactor = cnw.nCascadingNWA(s0,s0b)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0:\t{0};'.format(alignment2))

print('Classic Cross-Correlation:')
alignment1, alignment2, rollFactor = scc.crossCorrelationRolling(s0,s0b)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0:\t{0};'.format(alignment2))

# seq0 with seq0 missing one frame
m = 10
print('Comparing sequence #0 with itself missing frame {0}...'.format(m))
seq0m = np.concatenate((seq0[:m],seq0[m+1:]),axis=0)
s0,s0m = afd.matchFrames(seq0,seq0m,settings0['drift'])

print('n-Cascading Needleman-Wunsch Algorithm:')
alignment1, alignment2, rollFactor = cnw.nCascadingNWA(s0,s0m,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0m:\t{0};'.format(alignment2))

print('Classic Cross-Correlation:')
alignment1, alignment2, rollFactor = scc.crossCorrelationRolling(s0,s0m)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0m:\t{0};'.format(alignment2))

# seq0 with seq0 duplicating one frame
m = 10
print('Comparing sequence #0 with itself with frame {0} duplicated...'.format(m))
seq0d = np.concatenate((seq0[:m+1],seq0[m:]),axis=0)
s0,s0d = afd.matchFrames(seq0,seq0d,settings0['drift'])

print('n-Cascading Needleman-Wunsch Algorithm:')
alignment1, alignment2, rollFactor = cnw.nCascadingNWA(s0,s0d,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0d:\t{0};'.format(alignment2))

print('Classic Cross-Correlation:')
alignment1, alignment2, rollFactor = scc.crossCorrelationRolling(s0,s0d)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0d:\t{0};'.format(alignment2))

# seq0 with seq1
print('Comparing sequence #0 with sequence #1...')
s0,s1 = afd.matchFrames(seq0,seq1,settings1['drift'])

print('n-Cascading Needleman-Wunsch Algorithm:')
alignment1, alignment2, rollFactor = cnw.nCascadingNWA(s0,s1,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #1:\t{0};'.format(alignment2))

print('Classic Cross-Correlation:')
alignment1, alignment2, rollFactor = scc.crossCorrelationRolling(s0,s1)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0d:\t{0};'.format(alignment2))

# seq0 with seq2
print('Comparing sequence #0 with sequence #2...')
s0,s2 = afd.matchFrames(seq0,seq2,settings2['drift'])

print('n-Cascading Needleman-Wunsch Algorithm:')
alignment1, alignment2, rollFactor = cnw.nCascadingNWA(s0,s2,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #2:\t{0};'.format(alignment2))

print('Classic Cross-Correlation:')
alignment1, alignment2, rollFactor = scc.crossCorrelationRolling(s0,s2)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0d:\t{0};'.format(alignment2))

# seq0 with seq3
print('Comparing sequence #0 with sequence #3...')
s0,s3 = afd.matchFrames(seq0,seq3,settings3['drift'])

print('n-Cascading Needleman-Wunsch Algorithm:')
alignment1, alignment2, rollFactor = cnw.nCascadingNWA(s0,s3,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #3:\t{0};'.format(alignment2))

print('Classic Cross-Correlation:')
alignment1, alignment2, rollFactor = scc.crossCorrelationRolling(s0,s3)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0d:\t{0};'.format(alignment2))
