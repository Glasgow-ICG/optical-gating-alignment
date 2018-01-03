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
import image_loading as jid
sys.path.insert(0, '../cjn-python-emulator/')
import getPeriod as gpd
import realTimeSync as rts
import longTermUpdating as ltu
import helper as hlp

# set-up
settings = hlp.initialiseSettings(framerate=80)

# load data
sequenceName = '../notebooks/localdata/arithmicish_heart_bf/Allied Vision Technologies GS650 0001f61c/'
sequenceObj = jid.LoadAllImages(sequenceName,True,1,0,-1,None)
sequence, idx = hlp.convertObj(sequenceObj)

# Get First Full Period and Reference Period from Sequence
referenceIdx, settings = gpd.doEstablishPeriodProcessingForFrameIdx(sequence,settings,False)
print('Added period of {0} from frame {1} to frame {2};'.format(settings['referencePeriod'],referenceIdx[0],referenceIdx[-1]))
period = settings['referencePeriod']

# Get Full Period offset by ~2/3pi [52 frames/38] (w/ Reference Period)
offsetStep1 = referenceIdx[-1]
sequence1 = sequence[offsetStep1:]
referenceIdx, settings = gpd.doEstablishPeriodProcessingForFrameIdx(sequence1,settings,False)
print('Added period of {0} from frame {1} to frame {2};'.format(settings['referencePeriod'],referenceIdx[0]+offsetStep1,referenceIdx[-1]+offsetStep1))

# Get Full Period offset by ~3pi [117 frames/38] (w/ Reference Period)
offsetStep2 = referenceIdx[-1]+offsetStep1
sequence2 = sequence[offsetStep2:]
referenceIdx, settings = gpd.doEstablishPeriodProcessingForFrameIdx(sequence2,settings,False)
print('Added period of {0} from frame {1} to frame {2};'.format(settings['referencePeriod'],referenceIdx[0]+offsetStep2,referenceIdx[-1]+offsetStep2))

# Get Last(ish) Period (w/ Reference Period)
offset3 = len(sequence)-math.ceil(2*period)
sequence3 = sequence[offset3:]
referenceIdx, settings = gpd.doEstablishPeriodProcessingForFrameIdx(sequence3,settings,False)
print('Added period of {0} from frame {1} to frame {2};'.format(settings['referencePeriod'],referenceIdx[0]+offset3,referenceIdx[-1]+offset3))
