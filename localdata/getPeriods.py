# numpy, scipy
import numpy as np
import scipy as sp
from scipy.stats import norm

# Other
from pprint import pprint
import time
import math
from copy import copy,deepcopy
import os

# Local
import sys
sys.path.insert(0, '../j_postacquisition/')
import image_loading
import maintain_ref_frame_alignment as mrfa
sys.path.insert(0, '../py_sad_correlation/')
import j_py_sad_correlation as jps
sys.path.insert(0, '../cjn-python-emulator/')
import realTimeSync as rts
import longTermUpdating as ltu

settings = {}
settings.update({'drift':[-5,-2]})#starting drift corrections
settings.update({'estFramerate':79.449466058641775})#starting est frame rate
settings.update({'numExtraRefFrames':2})#padding number
settings.update({'minFramesForFit':3})#frames to fit for prediction (min)
settings.update({'maxFramesForFit':32})#frames to fit for prediction (max)
settings.update({'maxReceivedFramesToStore':260})#maximum number of frames to stores, used to prevent memory filling
settings.update({'barrierFrame':4.9949327669365964})#barrier frame in frames
settings.update({'extrapolationFactor':1.5})
settings.update({'predictionLatency':0.01})#prediction latency in seconds
settings.update({'referenceFrameCount':47})#number of reference frames including padding
settings.update({'referencePeriod':42.410156325856882})#reference period in frames
settings.update({'targetSyncPhase':(13.223884488771553-settings['numExtraRefFrames'])/(2*math.pi)})#target phase in rads
settings.update({'lastSent':0.0})

# Load Reference Frames
referenceNameFormat = '../notebooks/localdata/2017-02-09 18.35.13 vid overnight/{0}'
referenceImages = ltu.loadReference(ltu.getReference(referenceNameFormat,3))

# Load Stack 004
sequenceName = '../notebooks/localdata/2017-02-09 18.35.13 vid overnight/Stack {0:04d}/Brightfield - Prosilica/'
stackNumber = 4
sequenceObj = image_loading.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
startFrame = 121337

# order sequence
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
sequenceImages = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    sequenceImages[i] = np.asarray(sequenceObj[i].image,dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
sequenceImages = sequenceImages[idx,:,:]

# Get first full period (starting from phase 0)
p1start = 0
p1end = 0
pp_old = 0
i=0
while True:
    pp,dd,settings = rts.compareFrame(sequenceImages[i], referenceImages, settings, False)
    pp = ((pp-settings['numExtraRefFrames'])/settings['referencePeriod'])*(2*math.pi)%(2*math.pi)#convert phase to 2pi base
    if p1start==0 and pp_old>(0.9*2*math.pi) and pp<(0.1*2*math.pi):#if start of period
        p1start = i
    elif p1end==0 and p1start>0 and pp_old>(0.9*2*math.pi) and pp<(0.1*2*math.pi):#if end of period
        p1end = i-1
        print('Added period from frame {0} to frame {1};'.format(startFrame+p1start,startFrame+p1end))
        pp_old = float(pp)
        i += 1
        break
    pp_old = float(pp)
    i += 1

# Get second full period (starting from phase 0), continuing from above
p2start = p1end+1
p2end = 0
while True:
    pp,dd,settings = rts.compareFrame(sequenceImages[i], referenceImages, settings, False)
    pp = ((pp-settings['numExtraRefFrames'])/settings['referencePeriod'])*(2*math.pi)%(2*math.pi)#convert phase to 2pi base
    if p2end==0 and pp_old>(0.9*2*math.pi) and pp<(0.1*2*math.pi):#if end of period
        p2end = i-1
        print('Added period from frame {0} to frame {1};'.format(startFrame+p2start,startFrame+p2end))
        break
    pp_old = float(pp)
    i += 1

# Get first full period (starting from phase pi)
p3start = 0
p3end = 0
pp_old = 0
i=p1start
while True:
    pp,dd,settings = rts.compareFrame(sequenceImages[i], referenceImages, settings, False)
    pp = (((pp-settings['numExtraRefFrames'])/settings['referencePeriod'])*(2*math.pi)-math.pi)%(2*math.pi)#convert phase to 2pi base add offset by pi
    if p3start==0 and pp_old<((0.9*2*math.pi)) and pp<(0.1*2*math.pi):#if start of period
        p3start = i
    elif p3end==0 and p3start>0 and pp_old>(0.9*2*math.pi) and pp<(0.1*2*math.pi):#if end of period
        p3end = i-1
        print('Added period from frame {0} to frame {1};'.format(startFrame+p3start,startFrame+p3end))
        pp_old = float(pp)
        i += 1
        break
    pp_old = float(pp)
    i += 1

# Reset settingssettings = {}
settings.update({'drift':[-5,-1]})#starting drift corrections
settings.update({'estFramerate':79.925886315283961})#starting est frame rate

# Load Stack 006
sequenceName = '../notebooks/localdata/2017-02-09 18.35.13 vid overnight/Stack {0:04d}/Brightfield - Prosilica/'
stackNumber = 6
sequenceObj = image_loading.LoadAllImages(sequenceName.format(stackNumber),True,1,0,-1,None)
startFrame = 145387

# order sequence
sequenceObj = sequenceObj[0]
sequenceLength = sequenceObj.size
times = np.zeros(sequenceLength)
sz = sequenceObj[0].image.shape
sequenceImages = np.zeros([sequenceLength,sz[0],sz[1]],'uint8')
for i in range(sequenceLength):
    sequenceImages[i] = np.asarray(sequenceObj[i].image,dtype='uint8')
    times[i] = sequenceObj[i].plist['timestamp']
idx = np.argsort(times)
sequenceImages = sequenceImages[idx,:,:]

# Get last full period (starting from phase 0)
pp_old = 0
p4start = 0
p4end = 0
i=0
while i<len(sequenceImages):
    if i>=settings['numExtraRefFrames']:
        pp,dd,settings = rts.compareFrame(sequenceImages[i], referenceImages, settings, False)#ref pos including padding
        pp = ((pp-settings['numExtraRefFrames'])/settings['referencePeriod'])*(2*math.pi)%(2*math.pi)#convert phase to 2pi base
        if p4start==0 and pp_old>(0.9*2*math.pi) and pp<(0.1*2*math.pi):#if start of period
            p4start = i
        elif p4end==0 and p4start>0 and pp_old>(0.9*2*math.pi) and pp<(0.1*2*math.pi):#if end of period
            p4end = i-1
            if (len(sequenceImages)-p4end)<settings['referencePeriod']:
                print('Added period from frame {0} to frame {1};'.format(startFrame+p4start,startFrame+p4end))
                break
            else:
                #reset
                p4start = i
                p4end = 0
        pp_old = float(pp)
    i+=1
