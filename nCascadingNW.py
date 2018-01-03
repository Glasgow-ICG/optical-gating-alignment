import numpy as np
import math
import scipy.interpolate as spi
from pprint import pprint
#np.set_printoptions(threshold=np.nan)
# np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.0f}\t".format(x)})
import getPhase as gph
import sys
sys.path.insert(0, '../py_sad_correlation/')
import j_py_sad_correlation as jps

def constructCascade(scores,gp=0):
    # Construct grid
    nwa = np.zeros([scores.shape[0]+1,scores.shape[1]+1],'float')

    for t2 in range(scores.shape[0]+1):
        if t2==0:
            for t1 in range(1,scores.shape[1]+1):
                # ssd = np.sum((seq1[:,:,t1-1]-seq2[:,:,t2-1])**2)/seq1[:,:,t1-1].size
                mn = nwa[t2,t1-1]+gp
                nwa[t2,t1] = mn-scores[t2-1,t1-1]
        else:
            for t1 in range(scores.shape[1]+1):
                # ssd = np.sum((seq1[:,:,t1-1]-seq2[:,:,t2-1])**2)/seq1[:,:,t1-1].size
                if t1==0:
                    mn = nwa[t2-1,t1]+gp
                else:
                    mn = max([nwa[t2-1,t1-1],nwa[t2,t1-1]+gp,nwa[t2-1,t1]+gp])
                nwa[t2,t1] = mn-scores[t2-1,t1-1]
    return nwa

def interpImageSeriesZ(sequence,period,interp):
        x = np.arange(0,sequence.shape[1])
        y = np.arange(0,sequence.shape[2])
        z = np.arange(0,sequence.shape[0])

        # interp whole sequence
        zOut = np.linspace(0.0,sequence.shape[0]-1,sequence.shape[0]*interp)
        interpPoints = np.asarray(np.meshgrid(zOut,x,y,indexing='ij'))
        interpPoints = np.rollaxis(interpPoints,0,4)
        dt = sequence.dtype
        sequence = spi.interpn((z,x,y),sequence,interpPoints)
        sequence = np.asarray(sequence,dtype=dt)

        # take only up to first period
        return sequence[:int(period*interp),:,:]

# def deInterpwGaps(sequence,newLength):
#     #linear interpolation of a string/sequence to a new length
#     idx = np.linspace(0,len(sequence),newLength)
#
#     newSequence = []
#     for i in idx:
#         if i//1==i:
#             newSequence.append(sequence[int(idx)])
#
#
#
#     return sequence

def nCascadingNWA(seq1,seq2,period1,period2,target=0,gapPenalty=0,interp=1,log=False):
    ''' Assumes seq1 and seq2 are 3D numpy arrays of [t,x,y]'''
    if log:
        print('Sequence #1 has {0} frames and sequence #2 has {1} frames;'.format(len(seq1),len(seq2)))

    ls1 = float(len(seq1))
    ls2 = float(len(seq2))

    if log:
        print('Interpolating by a factor of {0} for greater precision'.format(interp))
    if interp>1:
        seq1 = interpImageSeriesZ(seq1,period1,interp)
        seq2 = interpImageSeriesZ(seq2,period2,interp)
    if log:
        print('Sequence #1 has {0} frames and sequence #2 has {1} frames;'.format(len(seq1),len(seq2)))
        print(seq1[:,0,0])
        print(seq2[:,0,0])


    # Calculate SSDs - python
    ssds = np.zeros([len(seq2),len(seq1)],'float')

    for t2 in range(len(seq2)):
        for t1 in range(len(seq1)):
            ssds[t2,t1] = np.sum(np.abs((seq1[t1].astype('float')-seq2[t2].astype('float'))))

    #print(ssds)

    # Calculate SSDs - C++
    ssds = jps.sad_grid(seq1, seq2)
    #print(ssds)

    if log:
        print ('Score matrix:')
        print(ssds)
        print('Shape: ({0},{1})'.format(ssds.shape[0],ssds.shape[1]))

    # Make Cascades
    cascades = np.zeros([len(seq2)+1,len(seq1)+1,len(seq1)],'float')
    for n in range(len(seq1)):
        cascades[:,:,n] = constructCascade(ssds,gapPenalty)
        ssds = np.roll(ssds,1,axis=1)

    # Pick Cascade and Roll Seq1
    rollFactor = np.argmax(cascades[len(seq2),len(seq1),:])
    score = np.amax(cascades[len(seq2),len(seq1),:])
    # print(score,np.iinfo(seq1.dtype).max,seq1.size)
    score = (score + (np.iinfo(seq1.dtype).max * seq1.size/10))/(np.iinfo(seq1.dtype).max * seq1.size/10)
    if score<=0:
        print('ISSUE: Negative Score')
    score = 0 if score<0 else score#
    # print(score)
    nwa = cascades[:,:,rollFactor]
    seq1 = np.roll(seq1,rollFactor,axis=2)
    if log:
        print('Cascade scores:\t',cascades[len(seq2),len(seq1),:])
        print ('Chose cascade {0} of {1}:'.format(rollFactor,len(seq1)))
        print(nwa)
        print('Shape: ({0},{1})'.format(nwa.shape[0],nwa.shape[1]))

    x = len(seq2)
    y = len(seq1)

    #  Traverse grid
    traversing = True

    # Trace without wrapping
    alignment1 = np.zeros((0,))
    alignment2 = np.zeros((0,))
    while traversing:
        xup = x-1
        yleft =  y-1
        if log:
            print('-----')
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('orig',x,y,nwa[x,y]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('diag',xup,yleft,nwa[xup,yleft]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('up  ',xup,y,nwa[xup,y]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('left',x,yleft,nwa[x,yleft]))

        if xup>=0:
            if  yleft>=0:
                options = [nwa[xup,yleft],nwa[xup,y],nwa[x,yleft]]
            else:
                if log:
                    print('Boundary Condition:\tI\'m at the left')
                options = [-np.inf,nwa[xup,y],-np.inf]
        else:
            if log:
                print('Boundary Condition:\tI\'m at the top')
            if  yleft>=0:
                options = [-np.inf,-np.inf,nwa[x,yleft]]
            else:
                if log:
                    print('Boundary Condition:\tI\'m at the top left')
                    print('Boundary Condition:\tI should not have  got here!')
                break
        direction = np.argmax(options)

        if direction==1:
            alignment1 = np.append(alignment1,-1)
            alignment2 = np.append(alignment2,xup)
            x = xup
            if log:
                print('Direction Travelled:\tI\'ve gone up')
        elif direction==0:
            alignment1 = np.append(alignment1,yleft)
            alignment2 = np.append(alignment2,xup)
            x = xup
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone diagonal')
        elif direction==2:
            alignment1 = np.append(alignment1,yleft)
            alignment2 = np.append(alignment2,-1)
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone left')
        if x==0 and y==0:
            if log:
                print('Traversing Complete')
            traversing = False

    # Reverses sequence
    alignment1 = alignment1[::-1]
    alignment2 = alignment2[::-1]

    if interp>1:
        if log:
            print('De-interpolating for result...')
        alignment1[alignment1>=0] = (alignment1[alignment1>=0]/interp)%(period1-1)
        alignment2[alignment2>=0] = (alignment2[alignment2>=0]/interp)%(period2-1)
        # rollFactor = rollFactor/interp

    if log:
        print('Aligned sequence #1 (wrapped):\t\t',alignment1)

    for i in range(len(alignment1)):
        if alignment1[i]>-1:
            alignment1[i] = (alignment1[i]-rollFactor)%(period1-1)

    # get rollFactor properly
    # print(target,alignment1)
    rollFactor = gph.getPhase(alignment1,alignment2,target,log)

    if log:
        print('Aligned sequence #1 (unwrapped):\t',alignment1)
        print('Aligned sequence #2:\t\t\t',alignment2)

    return alignment1, alignment2, rollFactor, score

def linInterp(string,position):
    #only for toy examples
    if position//1 == position:
        return string[math.floor(position)]
    else:
        interPos = position-(position//1)
        return string[math.floor(position)] + interPos*(string[math.ceil(position)]-string[math.floor(position)])

if __name__ == '__main__':
    print('Running toy example with')
    #Toy Example
    str1 = [0,64,192,255,255,192,128,128,64,0]
    str2 = [255,192,128,128,64,0,64,64,64,128,192,255]
    print('Sequence #1: {0}'.format(str1))
    print('Sequence #2: {0}'.format(str2))
    seq1  = np.asarray(str1,'uint8').reshape([len(str1),1,1])
    seq2  = np.asarray(str2,'uint8').reshape([len(str2),1,1])
    seq1 = np.repeat(np.repeat(seq1,10,1),5,2)
    seq2 = np.repeat(np.repeat(seq2,10,1),5,2)

    alignment1, alignment2, rF = nCascadingNWA(seq1,seq2,9.5,11.25,interp=2,log=True)
    print(rF)

    # Outputs for toy examples - Need to make deal with floats
    strout1 = []
    strout2 = []
    for i in alignment1:
        # print(i)
        if i<0:
            strout1.append(-1)
        else:
            strout1.append(linInterp(str1,i))
    for i in alignment2:
        # print(i)
        if i<0:
            strout2.append(-1)
        else:
            strout2.append(linInterp(str2,i))
    print('\n'.join('{0}\t{1}'.format(a, b) for a, b in zip(strout1, strout2)))


    # alignment1, alignment2, rF = nCascadingNWA(seq1,seq2,9.5,11.25,interp=4,log=True)
    # print(rF)
