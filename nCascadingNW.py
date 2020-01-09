'''Modules for phase matching two sequences based on sequence alignment.
Uses a cascading form of the Needleman Wunsch algorithm.
This module includes all necessary functions.'''

# Profiling
import cProfile
import pstats
import io

# Python Imports
import sys
import time

# Module Imports
import numpy as np

# Custom Module Imports
import j_py_sad_correlation as jps

# Local Imports
import getPhase as gtp


def linInterp(string, floatPosition):
    '''A linear interpolation function for a 'string' of ND items
    Note: this is, currently, only for uint8 images
    '''
    # Bottom position
    # equivalent to np.floor(floatPosition).astype('int)
    botPos = int(floatPosition)

    if floatPosition//1 == floatPosition:
        # equivalent to string[np.floor(floatPosition).astype('int)]
        interVal = 1.0*string[botPos]  # 1.0* needed to force numba type
    else:
        # Interpolation Ratio
        interPos = floatPosition-(floatPosition//1)

        # Top position
        topPos = int(np.ceil(floatPosition))

        # Values
        botVal = string[botPos]
        topVal = string[topPos]

        interVal = botVal + interPos*(topVal - botVal)
    
    interValInt = interVal.astype(np.uint8)

    return interValInt

def interpolateImageSeries(sequence, period, interpolationFactor=1):
    '''Interpolate a series of images along a 'time' axis.
    Note: this is, currently, only for uint8 images

    Inputs:
    * series: a PxMxN numpy array contain P images of size MxN
      * P is a time-like axis, e.g. time or phase.
    * period: float period length in units of frames
    * interpolationFactor: integer interpolation factor, e.g. 2 doubles the series length

    Outputs:
    * interpolatedSeries: a P'xMxN numpy array
      * Contains np.ceil(interpolationFactor*period) frames, i.e. P'<=interpolationFactor*P
    '''
    
    # Original coordinates
    (p, m, n) = sequence.shape

    # Interpolated space coordinates
    idtOut = np.arange(0, period, 1/interpolationFactor)  # supersample
    pOut = len(idtOut)

    # Sample at interpolated coordinates
    interpolatedSeries = np.zeros((pOut,m,n),dtype=np.uint8)
    for i in np.arange(idtOut.shape[0]):
        if idtOut[i]+1>len(sequence):  # boundary condition
            interpolatedSeries[i,...] = sequence[-1]  # TODO - this is very simplistic
        else:
            interpolatedSeries[i,...] = linInterp(sequence,idtOut[i])

    return interpolatedSeries


def fillTracebackMatrix(scoreMatrix, gapPenalty=0, log=False):
    tracebackMatrix = np.zeros((scoreMatrix.shape[0]+1, scoreMatrix.shape[1]+1), dtype=np.float64)
    
    for t2 in np.arange(tracebackMatrix.shape[0]):  # for all rows
        
        if t2 == 0:  # if the first row
            for t1 in np.arange(1, tracebackMatrix.shape[1]):  # for all but first column
                
                matchScore = scoreMatrix[t2-1, t1-1]  # get score for this combination (i.e. high score for a match)
                
                # left == insert gap into sequenceA
                insert = tracebackMatrix[t2, t1-1] - gapPenalty - matchScore  # get score to the left plus the gapPenalty (same as (t1)*gapPenalty)

                tracebackMatrix[t2, t1] = insert
                
        else:  # if any but the first row
            
            for t1 in np.arange(tracebackMatrix.shape[1]):  # for all columns
                if t1 == 0:  # if the first column
                    
                    matchScore = scoreMatrix[t2-1, t1-1]  # get score for this combination (i.e. high score for a match)
                    
                    # above == insert gap into sequenceB (or delete frame for sequenceA)
                    delete = tracebackMatrix[t2-1, t1] - gapPenalty - matchScore  # get score to the above plus the gapPenalty (same as t2*gapPenalty)

                    tracebackMatrix[t2, t1] = delete# - matchScore
                    
                else:
                    
                    matchScore = scoreMatrix[t2-1, t1-1]  # get score for this combination (i.e. high score for a match)
                    
                    # diagonal
                    match = tracebackMatrix[t2-1, t1-1] - matchScore

                    # above
                    delete = tracebackMatrix[t2-1, t1] - gapPenalty - matchScore

                    # left
                    insert = tracebackMatrix[t2, t1-1] - gapPenalty - matchScore

                    tracebackMatrix[t2, t1] = max([match,insert,delete])  # get maximum score from left, left above and above
                    
    return tracebackMatrix


def rollScores(scoreMatrix,rollFactor=1,axis=0):
    rolledScores = np.zeros(scoreMatrix.shape,dtype=scoreMatrix.dtype)
    for i in np.arange(scoreMatrix.shape[axis]):
        if axis==0:
            rolledScores[i,:] = scoreMatrix[(i-rollFactor)%scoreMatrix.shape[0],:]
        elif axis==1:
            rolledScores[:,i] = scoreMatrix[:,(i-rollFactor)%scoreMatrix.shape[1]]
    return rolledScores


def constructCascade(scoreMatrix, gapPenalty=0, axis=0, log=False):
    '''Create a 'cascade' of score arrays for use in the Needleman-Wunsch algorith.
    
    Inputs:
    * scoreMatrix: a score MxN array between two semi-periodic sequences
      * Columns represent one sequence of length M; rows the another of length N
    * gapPenalty: the Needleman-Wunsch penalty for introducing a gap (zero means no penalty, large means big penalty, i.e. less likely).
    * axis: the axis along which to roll/cascade

    Outputs:
    * cascades: a MxNx[M/N] array of cascaded traceback matrices
      * The third dimension depends on the axis parameter
    '''

    # Create 3D array to hold all cascades
    cascades = np.zeros((scoreMatrix.shape[0]+1, scoreMatrix.shape[1]+1, scoreMatrix.shape[0]), dtype=np.float64)

    # Create a new cascaded score array for each alignment (by rolling along axis)
    for n in np.arange(scoreMatrix.shape[1-axis]):  # the 1-axis tricks means we loop over 0 if axis=1 and vice versa  # TODO this doesn't seem to work?!
        if log:
            print('Getting score matrix for roll of {0} frames...'.format(n))
        cascades[:, :, n] = fillTracebackMatrix(scoreMatrix, gapPenalty=gapPenalty, log=log)
        scoreMatrix = rollScores(scoreMatrix, 1, axis=axis)

    return cascades


def traverseNW(sequence, templateSequence, nwa, log=False):
    x = templateSequence.shape[0]
    y = sequence.shape[0]

    #  Traverse grid
    traversing = True

    # Trace without wrapping
    alignmentA = []
    alignmentB = []
    while traversing:
        options = np.zeros((3,))

        xup = x-1
        yleft = y-1
        if log:  # .format() is not compatible with numba
            print('-----')
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});'.format(
                'curr', x, y, nwa[x, y], sequence[-y,0,0], templateSequence[-x,0,0]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});'.format(
                'diag', xup, yleft, nwa[xup, yleft], sequence[-yleft,0,0], templateSequence[-xup,0,0]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});'.format(
                'up  ', xup, y, nwa[xup, y], '-1 ({0})'.format(sequence[-y,0,0]), templateSequence[-xup,0,0]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f}; ({4}->{5});'.format(
                'left', x, yleft, nwa[x, yleft], sequence[-yleft,0,0], '-1 ({0})'.format(templateSequence[-x,0,0])))

        if xup >= 0:
            if yleft >= 0:
                options[:] = [nwa[xup, yleft], nwa[xup, y], nwa[x, yleft]]
            else:
                if log:
                    print('Boundary Condition:\tI\'m at the left')
                options[:] = [-np.inf, nwa[xup, y], -np.inf]
        else:
            if log:
                print('Boundary Condition:\tI\'m at the top')
            if yleft >= 0:
                options[:] = [-np.inf, -np.inf, nwa[x, yleft]]
            else:
                if log:
                    print('Boundary Condition:\tI\'m at the top left')
                    print('Boundary Condition:\tI should not have  got here!')
                break
        direction = np.argmax(options)

        if direction == 1:
            alignmentA.append(-1)
            alignmentB.append(xup)
            x = xup
            if log:
                print('Direction Travelled:\tI\'ve gone up')
        elif direction == 0:
            alignmentA.append(yleft)
            alignmentB.append(xup)
            x = xup
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone diagonal')
        elif direction == 2:
            alignmentA.append(yleft)
            alignmentB.append(-1)
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone left')
        if x == 0 and y == 0:
            if log:
                print('Traversing Complete')
            traversing = False

    # Reverses sequence
    alignmentA = np.asarray(alignmentA[::-1],dtype=np.float)
    alignmentB = np.asarray(alignmentB[::-1],dtype=np.float)

    return (alignmentA, alignmentB)


def nCascadingNWA(sequence,
                  templateSequence,
                  period,
                  templatePeriod,
                  gapPenalty=0,
                  interpolationFactor=1,
                  knownTargetFrame=0,
                  log=True):
    ''' Assumes sequence and templateSequence are 3D numpy arrays of [t,x,y]'''
    if log:
        print('Sequence #1 has {0} frames and sequence #2 has {1} frames;'.format(len(sequence),len(templateSequence)))

    ls1 = float(len(sequence))
    ls2 = float(len(templateSequence))

    if interpolationFactor is not None:
      if log:
          print('Interpolating by a factor of {0} for greater precision'.format(interpolationFactor))
      sequence = interpolateImageSeries(sequence, period, interpolationFactor=interpolationFactor)
      templateSequence = interpolateImageSeries(templateSequence, period, interpolationFactor=interpolationFactor)
      if log:
          print('Sequence #1 has {0} frames and sequence #2 has {1} frames'.format(len(sequence),len(templateSequence)))
          print(sequence[:,0,0])
          print(templateSequence[:,0,0])
    else:
      if log:
          print('No interpolation required'.format(interpolationFactor))
          print(sequence[:,0,0])
          print(templateSequence[:,0,0])

    # Calculate Score Matrix - C++
    scoreMatrix = jps.sad_grid(sequence, templateSequence)

    if log:
        print ('Score matrix:')
        print(scoreMatrix)
        print('Shape: ({0},{1})'.format(scoreMatrix.shape[0],scoreMatrix.shape[1]))
        
    # Cascade the SAD Grid
    cascades = constructCascade(scoreMatrix, gapPenalty=gapPenalty, axis=1, log=log=='verbose')
    if log:
        print('Unrolled Traceback Matrix:')
        print(cascades[:,:,0])
        print('\tDtype: {0};\tShape: ({1},{2},{3})'.format(cascades.dtype, cascades.shape[0], cascades.shape[1], cascades.shape[2]))

    # Pick Cascade and Roll sequence
    rollFactor = np.argmax(cascades[len(templateSequence),len(sequence),:])
    score = np.amax(cascades[len(templateSequence),len(sequence),:])
    # f1 = plt.figure()
    # a1 = f1.add_subplot(111)
    # a1.plot(cascades[len(templateSequence),len(sequence),:])
    # a1.scatter(rollFactor,score,s=80,c='k')
    # print(score,np.iinfo(sequence.dtype).max,sequence.size)
    score = (score + (np.iinfo(sequence.dtype).max * sequence.size/10))/(np.iinfo(sequence.dtype).max * sequence.size/10)
    if score<=0:
        print('ISSUE: Negative Score')
    score = 0 if score<0 else score#
    # print(score)
    nwa = cascades[:,:,rollFactor]
    sequence = np.roll(sequence,rollFactor,axis=2)
    if log:
        print('Cascade scores:\t',cascades[len(templateSequence),len(sequence),:])
        print ('Chose cascade {0} of {1}:'.format(rollFactor,len(sequence)))
        print(nwa)
        print('Shape: ({0},{1})'.format(nwa.shape[0],nwa.shape[1]))

    (alignmentAWrapped, alignmentB) = traverseNW(sequence, templateSequence, nwa, log=log=='verbose')
    
    if log:
        print('Aligned sequence #1 (interpolated):\t', alignmentAWrapped)
        print('Aligned sequence #2 (interpolated):\t\t\t', alignmentB)

    if interpolationFactor is not None:
        if log:
            print('De-interpolating for result...')
        # Divide by interpolation factor and modulo period
        # ignore -1s
        alignmentAWrapped[alignmentAWrapped >= 0] = (
            alignmentAWrapped[alignmentAWrapped >= 0]/interpolationFactor) % (period)
        alignmentB[alignmentB >= 0] = (
            alignmentB[alignmentB >= 0]/interpolationFactor) % (templatePeriod)
        rollFactor = (rollFactor/interpolationFactor) % (period)  # TODO: is this the right period?

    if log:
        print('Aligned sequence #1 (wrapped):\t\t',alignmentAWrapped)

    # roll Alignment A, taking care of indels
    alignmentA = []
    indels = []
    for i in np.arange(alignmentAWrapped.shape[0]):
        if alignmentAWrapped[i] > -1:
            alignmentA.append((alignmentAWrapped[i] - rollFactor) % (period))
        else:
            idx = i-1
            before = -1
            while before<0 and idx<alignmentAWrapped.shape[0]-1:
                before = alignmentAWrapped[(idx)%len(alignmentAWrapped)]
                idx = idx+1
            indels.append(before)
    for i in indels:
        alignmentA.insert(alignmentA.index(i)+1,-1)
    alignmentA = np.array(alignmentA)

    # get rollFactor properly
    rollFactor = gtp.getPhase(alignmentA,alignmentB,knownTargetFrame,log)

    if log:
        print('Aligned sequence #1 (unwrapped):\t', alignmentA)
        print('Aligned sequence #2:\t\t\t', alignmentB)
    
    return alignmentA, alignmentB, rollFactor, score

def linInterp(string,position):
    #only for toy examples
    if position//1 == position:
        return string[int(position)]  # equivalent to np.floor(position).astype(np.int)
    else:
        interPos = position-(position//1)
        return string[int(position)] + interPos*(string[int(position+1)]-string[int(position)])  # int(position+1) is equivalent to np.ceil(position).astype(np.int)

if __name__ == '__main__':
    print('Running toy example with')
    #Toy Example
    str1 = [0,64,192,255,255,192,128,128,64,0]
    str2 = [255,192,128,128,64,0,64,64,64,128,192,255]
    print('Sequence #1: {0}'.format(str1))
    print('Sequence #2: {0}'.format(str2))
    sequence  = np.asarray(str1,'uint8').reshape([len(str1),1,1])
    templateSequence  = np.asarray(str2,'uint8').reshape([len(str2),1,1])
    sequence = np.repeat(np.repeat(sequence,10,1),5,2)
    templateSequence = np.repeat(np.repeat(templateSequence,10,1),5,2)

    alignment1, alignment2, rF, score = nCascadingNWA(sequence,templateSequence,9.5,11.25,interpolationFactor=1,log=True)
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


    # alignment1, alignment2, rF = nCascadingNWA(sequence,templateSequence,9.5,11.25,interpolationFactor=4,log=True)
    # print(rF)
