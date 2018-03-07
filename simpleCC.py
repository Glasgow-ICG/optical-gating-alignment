import numpy as np
from scipy.interpolate import interpn
import math

import sys
sys.path.insert(0, '../j_postacquisition/')
import periods as jpp
sys.path.insert(0, '../cjn-python-emulator/')
import realTimeSync as rts

def crossCorrelationScores(seq1, seq2):
    ## Assumes seq1 and seq2 are 3D numpy arrays of [t,x,y]

    # Calculate cross-correlation from JT codes
    temp = np.conj(np.fft.fft(seq1, axis=0)) * np.fft.fft(seq2, axis=0)
    temp2 = np.fft.ifft(temp, axis=0)

    scores = np.sum(seq1*seq1) + np.sum(seq2*seq2) - 2 * np.sum(np.real(temp2), axis=1)

    return scores

def minimumScores(scores,useTri=True):
    # Given a set of difference scores, determine where the minimum lies from JT codes
    if useTri:
        # V-fitting for sub-integer interpolation
        # Note that 'scores' is a ring vector, with scores[0] being adjacent to scores[-1]
        y1 = scores[np.argmin(scores)-1]    # Works even when minimum is at 0
        y2 = scores[np.argmin(scores)]
        y3 = scores[(np.argmin(scores)+1) % len(scores)]
        minPos, minVal = rts.threePointTriangularMinimum(y1,y2,y3)
    else:
        # just use minimum
        minPos = np.argmin(scores)

    minPos = (minPos + np.argmin(scores)) % len(scores)

    return minPos, minVal

def matchSequenceSlicing(seq1,seq2):
    # TODO: make period specific
    newLength = max(len(seq1),len(seq2))

    x = np.arange(0.0,seq1.shape[1])
    y = np.arange(0.0,seq1.shape[2])
    z1 = np.arange(0.0,seq1.shape[0])
    z2 = np.arange(0.0,seq2.shape[0])

    zOut = np.linspace(0.0,seq1.shape[0]-1,newLength)
    interpPoints = np.asarray(np.meshgrid(zOut,x,y,indexing='ij'))
    interpPoints = np.rollaxis(interpPoints,0,4)
    seq1 = interpn((z1,x,y),seq1,interpPoints)

    zOut = np.linspace(0,seq2.shape[0]-1,newLength)
    interpPoints = np.asarray(np.meshgrid(zOut,x,y,indexing='ij'))
    interpPoints = np.rollaxis(interpPoints,0,4)
    seq2 = interpn((z2,x,y),seq2,interpPoints)

    return seq1, seq2

def resampleImageSection(seq1, thisPeriod, newLength):
    #trimmed down version of JTs periods.ResampleImageSection

    result = np.zeros([newLength,seq1.shape[1],seq1.shape[2]],'float')
    for i in range(newLength):
        desiredPos = (i / float(newLength)) * thisPeriod
        beforePos = int(desiredPos)
        afterPos = beforePos + 1
        remainder = (desiredPos - beforePos) / float(afterPos - beforePos)

        before = seq1[beforePos]
        after = seq1[int(afterPos%math.ceil(thisPeriod))]#this is different to JT
        result[i] = before * (1 - remainder) + after * remainder
    return result

def crossCorrelationRolling(seq1, seq2, period1, period2, useTri=False, log=False, numSamplesPerPeriod=80, target=0):

    if log:
        # Outputs for toy examples
        strout1 = []
        strout2 = []
        for i in seq1:
            strout1.append(i[0,0])
        for i in seq2:
            strout2.append(i[0,0])
        print('Original sequence #1:\t{0}'.format(strout1))
        print('Original sequence #2:\t{0}'.format(strout2))

    origLen1 = len(seq1)
    origLen2 = len(seq2)

    #seq1, seq2 = matchSequenceSlicing(seq1,seq2)
    if period1 != numSamplesPerPeriod:
        seq1 = resampleImageSection(seq1, period1, numSamplesPerPeriod)
    if period2 != numSamplesPerPeriod:
        seq2 = resampleImageSection(seq2, period2, numSamplesPerPeriod)

    if log:
        # Outputs for toy examples
        strout1 = []
        strout2 = []
        for i in seq1:
            strout1.append(i[0,0])
        for i in seq2:
            strout2.append(i[0,0])
        print('Resliced sequence #1:\t{0}'.format(strout1))
        print('Resliced sequence #2:\t{0}'.format(strout2))


    seq1 = makeArrayFromSequence(seq1[:numSamplesPerPeriod])
    seq2 = makeArrayFromSequence(seq2[:numSamplesPerPeriod])
    scores = crossCorrelationScores(seq1, seq2)

    rollFactor, minVal = minimumScores(scores,useTri=True)
    rollFactor = (rollFactor/len(seq1))*origLen2

    alignment1 = (np.arange(0,origLen1)-rollFactor)%origLen1
    alignment2 = np.arange(0,origLen2)

    # Here we roll seq1 for consistency with nCascadingNW.py
    if log:
        print('Alignment 1:\t{0}'.format(alignment1))
        print('Alignment 2:\t{0}'.format(alignment2))
        print('Rolled by {0}'.format(rollFactor))
    # convert to list for consistency with nCascadingNW.py
    return list(alignment1), list(alignment2), (target+rollFactor)%origLen2, minVal


def makeArrayFromSequence(seq):
    # Stolen from JT, should be rewritten (I think it can be done in a vectorised way)
    # Utility function used as part of the FFT processing
    result = np.zeros((len(seq), seq.shape[1]*seq.shape[2]))
    for i in range(len(seq)):
        result[i] = np.array(seq[i], copy=False).flatten()
    return result


if __name__ == '__main__':
    print('Running toy example with')
    #Toy Example
    str1 = [0,1,2,3,4,3,2,2,0]
    str2 = [4,3,2,1,0,0,1,1,1,2,3,4]
    print('Sequence #1: {0}'.format(str1))
    print('Sequence #2: {0}'.format(str2))
    seq1  = np.asarray(str1,'uint8').reshape([len(str1),1,1])
    seq2  = np.asarray(str2,'uint8').reshape([len(str2),1,1])
    seq1 = np.repeat(np.repeat(seq1,10,1),5,2)
    seq2 = np.repeat(np.repeat(seq2,10,1),5,2)

    alignment1, alignment2, rF = crossCorrelationRolling(seq1, seq2, 8, 11, useTri=True)

    # Outputs for toy examples
    strout1 = []
    strout2 = []
    for i in alignment1:
        if i<0:
            strout1.append(-1)
        else:
            strout1.append(str1[i])
    for i in alignment2:
        if i<0:
            strout2.append(-1)
        else:
            strout2.append(str2[i])
    print('Aligned Sequence #1: {0}'.format(strout1))
    print('Aligned Sequence #2: {0}'.format(strout2))
