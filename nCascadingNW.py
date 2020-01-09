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
                  interpolationFactor=None,
                  knownTargetFrame=0,
                  log=True):
    '''Calculating the cascading Needleman-Wunsch alignment for two semi-periodic sequences.

    For the two sequences provided, this algorithm will assume the second is the 'template'.
    The template sequence will see only minor changes for alignment, i.e. adding gaps.
    The other sequence will see rolling and gap addition.

    Inputs:
    * sequence, templateSequence: a PxMxN numpy array representing the two periods to align
    * period, remplatePeriod: the float period for sequence/templateSequence in frame units (caller must determine this)
    * gapPenalty: the Needleman-Wunsch penalty for introducing a gap as a percentage (relating to the calculated score matrix)
    * interpolationFactor: integer linear interpolation factor, e.g. a factor of 2 will double the image resolution along P
    * knownTargetFrame: integer frame (in B) for which to return the roll factor
    '''
    if templatePeriod is None:
        templatePeriod = templateSequence.shape[0]
    if period is None:
        period = sequence.shape[0]

    if log:
        print('Sequence #1 has {0} frames and sequence #2 has {1} frames'.format(len(sequence), len(templateSequence)))

    if interpolationFactor is not None:
      if log:
          print('Interpolating by a factor of {0} for greater precision...'.format(interpolationFactor))
      sequence = interpolateImageSeries(sequence, period, interpolationFactor=interpolationFactor)
      templateSequence = interpolateImageSeries(templateSequence, period, interpolationFactor=interpolationFactor)
      if log:
          print('\tSequence #1 now has {0} frames and sequence #2 now has {1} frames:'.format(
                len(sequence), len(templateSequence)))

    # Calculate Score Matrix - C++
    scoreMatrix = jps.sad_grid(sequence, templateSequence)

    if log:
        print ('Score Matrix:')
        print(scoreMatrix)
        print('\tDtype: {0};\tShape: ({1},{2})'.format(scoreMatrix.dtype, scoreMatrix.shape[0], scoreMatrix.shape[1]))
        
    # Cascade the SAD Grid
    cascades = constructCascade(scoreMatrix, gapPenalty=gapPenalty, axis=1, log=log=='verbose')
    if log:
        print('Unrolled Traceback Matrix:')
        print(cascades[:,:,0])
        print('\tDtype: {0};\tShape: ({1},{2},{3})'.format(cascades.dtype, cascades.shape[0], cascades.shape[1], cascades.shape[2]))

    # Pick Cascade and Roll sequence
    rollFactor = np.argmax(cascades[-1, -1, :])
    score = cascades[-1, -1, rollFactor]
    score = (score + (np.iinfo(sequence.dtype).max * sequence.size/10)) / (np.iinfo(sequence.dtype).max * sequence.size/10)
    if score<=0:
        print('ISSUE: Negative Score')
    score = 0 if score<0 else score

    nwa = cascades[:, :, rollFactor]
    sequence = np.roll(sequence,rollFactor,axis=2)
    if log:
        print('Cascade scores:\t',cascades[len(templateSequence),len(sequence),:])
        print ('Chose cascade {0} of {1}:'.format(rollFactor,len(sequence)))
        print(nwa)
        print('Shape: ({0},{1})'.format(nwa.shape[0],nwa.shape[1]))

    (alignmentAWrapped, alignmentB) = traverseNW(sequence, templateSequence, nwa, log=log=='verbose')
    
    if log:
        print('Rollfactor (interpolated):\t{0}'.format(rollFactor))
        print('Aligned sequence #1 (interpolated, wrapped):\t', alignmentAWrapped)
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
            print('Rollfactor:\t{0}'.format(rollFactor))
            print('Aligned sequence #1 (wrapped):\t\t',alignmentAWrapped)
            print('Aligned sequence #2:\t\t\t', alignmentB)

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
        print('Rollfactor:\t{0}'.format(rollFactor))
        print('Aligned sequence #1 (unwrapped):\t', alignmentA)
        print('Aligned sequence #2:\t\t\t', alignmentB)
    
    return alignmentA, alignmentB, rollFactor, score


def runToySequences(roll=5, gapPenalty=0, shape=(1,1), log=False):
    '''Compiles all numba functions in module.'''

    # Toy Example
    # toySequenceA and B have very slightly different rhythms but the same period
    toySequenceA = np.asarray([100, 150, 175, 200, 225, 230, 205, 180, 155, 120],dtype='uint8')
    toySequenceA = np.roll(toySequenceA, -roll)
    periodA = toySequenceA.shape[0]
    toySequenceB = np.asarray([100, 125, 150, 175, 200, 225, 230, 205, 180, 120],dtype='uint8')
    periodB = toySequenceB.shape[0] # -0.5
    if log:
        print('Running toy example with:')
        print('\tSequence A: ', toySequenceA)
        print('\tSequence B: ', toySequenceB)

    # Make sequences 3D arrays (as expected for this algorithm)
    ndSequenceA = toySequenceA[:, np.newaxis, np.newaxis]
    ndSequenceB = toySequenceB[:, np.newaxis, np.newaxis]
    ndSequenceA = np.repeat(np.repeat(ndSequenceA,shape[0],1),shape[1],2)  # make each frame actually 2D to check everything works for image frames
    ndSequenceB = np.repeat(np.repeat(ndSequenceB,shape[0],1),shape[1],2)

    alignmentA, alignmentB, rollFactor, score = nCascadingNWA(ndSequenceA, ndSequenceB, periodA, periodB, gapPenalty=gapPenalty, log=log)
    if log:
        print('Roll factor: {0} (score: {1})'.format(rollFactor, score))
        print('Alignment Maps:')
        print('\tMap A: {0}'.format(alignmentA))
        print('\tMap B: {0}'.format(alignmentB))
        # print('\tA\t\tB')
        # print('\n'.join(['{0:08.2f}\t{1:08.2f}'.format(a,b) for (a,b) in zip(alignmentA,alignmentB)]))

    # Outputs for toy examples
    alignedSequenceA = []  # Create new lists to fill with aligned values
    alignedSequenceB = []
    for i in alignmentA:  # fill new sequence A
        if i < 0:  # indel
            alignedSequenceA.append(-1)
        else:
            alignedSequenceA.append(linInterp(ndSequenceA, i)[0,0])
    for i in alignmentB:  # fill new sequence B
        if i < 0:  # indel
            alignedSequenceB.append(-1)
        else:
            alignedSequenceB.append(linInterp(ndSequenceB, i)[0,0])

    # Print
    score = 0
    for i,j in zip(alignedSequenceA,alignedSequenceB):
        if i>-1 and j>-1:
            i = float(i)
            j = float(j)
            score = score - np.abs(i-j)
        elif i>-1:
            score = score - i
        elif j>-1:
            score = score - j
    if log:
        print('\nAligned Sequences:')
        # print('\tSequence A: ', alignedSequenceA)
        # print('\tSequence B: ', alignedSequenceB)
        print('\tA\t\tB')
        print('\n'.join(['{0:8.2f}\t{1:8.2f}'.format(a,b) for (a,b) in zip(alignedSequenceA,alignedSequenceB)]))
        print('Final Score: {0}'.format(score))

    return None


if __name__ == '__main__':
    # # Compile (if using Numba)
    # t = time.time()
    # runToySequences(seqA='simple', log=False)
    # print('Compilation run took {0:.2f} milliseconds.'.format(1000*(time.time()-t)))

    # Run - timed
    # t = time.time()
    # runToySequences(seqA='simple', log=False)
    # print('Post-compilation run took {0:.2f} milliseconds.'.format(1000*(time.time()-t)))

    # # Run - Python (if using Numba)
    # t = time.time()
    # runToySequences.py_func(seqA='simple', log=False)
    # print('Equivalent Python run took {0:.2f} milliseconds.'.format(1000*(time.time()-t)))

    # Profiler
    # pr = cProfile.Profile()
    # pr.enable()

    # Run - verbose
    runToySequences(roll=7, gapPenalty=1, shape=(1024,1024), log=True)

    # Profile
    # pr.disable()
    # s = io.StringIO()
    # sortby = pstats.SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
