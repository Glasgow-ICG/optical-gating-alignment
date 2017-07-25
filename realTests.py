from nCascadingNW import *
import sys
sys.path.insert(0, '../j_postacquisition/')
import image_loading
import numpy as np

# load all sequences and convert to ordered lists
sequenceName = './localdata/seq{0}/'

#seq0
stackNumber = 0
sequenceObj = image_loading.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq0 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq0[i] = np.asarray(sequenceObj[i].image,dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq0 = seq0[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq0)))

#seq1
stackNumber = 1
sequenceObj = image_loading.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq1 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq1[i] = np.asarray(sequenceObj[i].image,dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq1 = seq1[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq1)))

#seq2
stackNumber = 2
sequenceObj = image_loading.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq2 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq2[i] = np.asarray(sequenceObj[i].image,dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq2 = seq2[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq2)))

#seq3
stackNumber = 3
sequenceObj = image_loading.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
seq3 = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    seq3[i] = np.asarray(sequenceObj[i].image,dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
seq3 = seq3[idx,:,:]
print('Loaded sequence #{0} consisting of {1} frames;'.format(stackNumber,len(seq3)))

# seq0 with seq0
print('Comparing sequence #0 with itself...')
alignment1, alignment2, rollFactor = nCascadingNWA(seq0,seq0,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0:\t{0};'.format(alignment2))

# seq0 with seq0 missing one frame
m = 10
print('Comparing sequence #0 with itself missing frame {0}...'.format(m))
seq0m = np.concatenate((seq0[:m],seq0[m+1:]),axis=0)
alignment1, alignment2, rollFactor = nCascadingNWA(seq0,seq0m,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0m:\t{0};'.format(alignment2))

# seq0 with seq0 duplicating one frame
m = 10
print('Comparing sequence #0 with itself with frame {0} duplicated...'.format(m))
seq0d = np.concatenate((seq0[:m+1],seq0[m:]),axis=0)
alignment1, alignment2, rollFactor = nCascadingNWA(seq0,seq0d,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #0d:\t{0};'.format(alignment2))

# seq0 with seq1
print('Comparing sequence #0 with sequence #1...')
alignment1, alignment2, rollFactor = nCascadingNWA(seq0,seq1,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #1:\t{0};'.format(alignment2))

# seq0 with seq2
print('Comparing sequence #0 with sequence #2...')
alignment1, alignment2, rollFactor = nCascadingNWA(seq0,seq2,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #2:\t{0};'.format(alignment2))

# seq0 with seq3
print('Comparing sequence #1 with sequence #3...')
alignment1, alignment2, rollFactor = nCascadingNWA(seq0,seq3,log=False)
print('Wrapped by {0} steps;'.format(rollFactor))
print('Aligned sequence #0:\t{0};'.format(alignment1))
print('Aligned sequence #3:\t{0};'.format(alignment2))
