'''Modules for phase-matching/aligning two quasi-periodic sequences.
Uses a cascading form of the Needleman-Wunsch algorithm.
'''

# Profiling
import cProfile
import pstats
import io

# Python Imports
import sys
import time

# Module Imports
import numpy as np
from scipy.interpolate import interpn
from numba import jit, prange

# Custom Module Imports
import j_py_sad_correlation as jps

# Local Imports
import getPhase as gtp

@jit(nopython=True, parallel=False, fastmath=True, nogil=True)
def newScores(scores, gapPenalty=0):
    cascadedScores = np.zeros((scores.shape[0]+1, scores.shape[1]+1), dtype=np.float64)
    for t2 in np.arange(cascadedScores.shape[0]):  # for all rows
        if t2 == 0:  # if the first row
            for t1 in np.arange(1, cascadedScores.shape[1]):  # for all but first column
                maxScore = cascadedScores[t2, t1-1] + gapPenalty  # get score to the left plus the gapPenalty
                original = scores[t2-1, t1-1]  # get score for this combination (i.e. high score for a match)
                cascadedScores[t2, t1] = maxScore - original
        else:  # if any but the first row
            for t1 in np.arange(cascadedScores.shape[1]):  # for all columns
                if t1 == 0:  # if the first column
                    maxScore = cascadedScores[t2-1, t1] + gapPenalty  # get score to the above plus the gapPenalty
                else:
                    maxScore = max([cascadedScores[t2-1, t1-1],
                                    cascadedScores[t2, t1-1] + gapPenalty,
                                    cascadedScores[t2-1, t1] + gapPenalty])  # get maximum score from left, left above and above
                original = scores[t2-1, t1-1]  # get score for this combination (i.e. high score for a match)
                cascadedScores[t2, t1] = maxScore - original
    return cascadedScores


@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def rollScores(scores,rollFactor=1,axis=0):
    rolledScores = np.zeros(scores.shape,dtype=scores.dtype)
    for i in prange(scores.shape[axis]):
        if axis==0:
            rolledScores[i,:] = scores[(i-rollFactor)%scores.shape[0],:]
        elif axis==1:
            rolledScores[:,i] = scores[:,(i-rollFactor)%scores.shape[1]]
    return rolledScores


@jit(nopython=True, parallel=False, fastmath=True, nogil=True)
def constructCascade(scores, gapPenalty=0, axis=0):
    '''Create a 'cascade' of score arrays for use in the Needleman-Wunsch algorith.
    
    Inputs:
    * scores: a score MxN array between two semi-periodic sequences
      * Columns represent one sequence of length M; rows the another of length N
    * gapPenalty: the Needleman-Wunsch penalty for introducing a gap (zero means no penalty, large means big penalty, i.e. less likely).
    * axis: the axis along which to roll/cascade

    Outputs:
    * cascades: a MxNx[M/N] array of cascaded scores
      * The third dimension depends on the axis parameter
    '''

    # Create 3D array to hold all cascades
    cascades = np.zeros((scores.shape[0]+1, scores.shape[1]+1, scores.shape[0]), dtype=np.float64)

    # Create a new cascaded score array for each alignment (by rolling along axis)
    for n in np.arange(scores.shape[1-axis]):  # the 1-axis tricks means we loop over 0 if axis=1 and vice versa  # TODO this doesn't seem to work?!
        # rolled = rollScores(scores, n-1, axis=axis)
        cascades[:, :, n] = newScores(scores, gapPenalty)
        scores = rollScores(scores, 1, axis=axis)

    return cascades


@jit('uint8[:,:](uint8[:,:,:],float64)', nopython=True, parallel=False, fastmath=True, nogil=True)
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


@jit('uint8[:,:,:](uint8[:,:,:],float64,uint8)', nopython=True, parallel=True, fastmath=True, nogil=True)
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
    # idx = np.arange(0, sequence.shape[1])
    # idy = np.arange(0, sequence.shape[2])
    # idt = np.arange(0, sequence.shape[0])

    # Interpolated space coordinates
    # idtOut = np.arange(0, sequence.shape[0], 1/interpolationFactor)  # supersample
    idtOut = np.arange(0, period, 1/interpolationFactor)  # supersample
    pOut = len(idtOut)
    # interpPoints = np.asarray(np.meshgrid(idtOut, idx, idy, indexing='ij'))
    # interpPoints = np.moveaxis(interpPoints, 0, -1)

    # Sample at interpolated coordinates
    interpolatedSeries = np.zeros((pOut,m,n),dtype=np.uint8)
    for i in prange(idtOut.shape[0]):
        interpolatedSeries[i,...] = linInterp(sequence,idtOut[i])
    # interpolatedSeries = interpn((idt, idx, idy), sequence, interpPoints)
    # interpolatedSeries = np.asarray(interpolatedSeries, dtype=sequence.dtype)  # safety check for dtype

    # take only up to first period
    # interpolatedSeries = interpolatedSeries[:np.ceil(period*interpolationFactor).astype('int'), :, :]  # is this needed?

    return interpolatedSeries


@jit(nopython=True, parallel=False, fastmath=True, nogil=True)
def traverseNW(sequence, templateSequence, nwa, log=False):
    x = templateSequence.shape[0]
    y = sequence.shape[0]

    #  Traverse grid
    traversing = True

    # Trace without wrapping
    alignmentA = np.zeros((0,))
    alignmentB = np.zeros((0,))
    while traversing:
        options = np.zeros((3,))

        xup = x-1
        yleft = y-1
        # if log:  # .format() is not compatible with numba
        #     print('-----')
        #     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
        #         'orig', x, y, nwa[x, y]))
        #     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
        #         'diag', xup, yleft, nwa[xup, yleft]))
        #     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
        #         'up  ', xup, y, nwa[xup, y]))
        #     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
        #         'left', x, yleft, nwa[x, yleft]))

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
            alignmentA = np.append(alignmentA, -1)
            alignmentB = np.append(alignmentB, xup)
            x = xup
            if log:
                print('Direction Travelled:\tI\'ve gone up')
        elif direction == 0:
            alignmentA = np.append(alignmentA, yleft)
            alignmentB = np.append(alignmentB, xup)
            x = xup
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone diagonal')
        elif direction == 2:
            alignmentA = np.append(alignmentA, yleft)
            alignmentB = np.append(alignmentB, -1)
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone left')
        if x == 0 and y == 0:
            if log:
                print('Traversing Complete')
            traversing = False

    # Reverses sequence
    alignmentA = alignmentA[::-1]
    alignmentB = alignmentB[::-1]

    return (alignmentA, alignmentB)

@jit(nopython=False, parallel=False, fastmath=True, nogil=True)
def nCascadingNWA(sequence,
                  templateSequence,
                  period,
                  templatePeriod,
                  gapPenalty=0,
                  interpolationFactor=None,
                  knownTargetFrame=0,
                  log=False):
    '''Calculating the cascading Needleman-Wunsch alignment for two semi-periodic sequences.

    For the two sequences provided, this algorithm will assume the second is the 'template'.
    The template sequence will see only minor changes for alignment, i.e. adding gaps.
    The other sequence will see rolling and gap addition.

    Inputs:
    * sequence, templateSequence: a PxMxN numpy array representing the two periods to align
    * period, remplatePeriod: the float period for sequence/templateSequence in frame units (caller must determine this)
    * gapPenalty: the Needleman-Wunsch penalty for introducing a gap (zero means no penalty, large means big penalty, i.e. less likely).
    * interpolationFactor: integer linear interpolation factor, e.g. a factor of 2 will double the image resolution along P
    * knownTargetFrame: integer frame (in B) for which to return the roll factor
    '''
    # if log:
    #     print('Sequence #1 has {0} frames and sequence #2 has {1} frames;'.format(len(sequence), len(templateSequence)))

    # Interpolate Sequence (for finer alignment)
    if interpolationFactor is not None:# and isinstance(interpolationFactor,int):
        # if log:
        #     print('Interpolating by a factor of {0} for greater precision'.format(interpolationFactor))
        
        # interpolate
        sequence = interpolateImageSeries(sequence, period, interpolationFactor)
        templateSequence = interpolateImageSeries(templateSequence, templatePeriod, interpolationFactor)
        
        # if log:
        #     print('\tSequence #1 now has {0} frames and sequence #2 now has {1} frames'.format(
        #         len(sequence), len(templateSequence)))

    # Calculate SAD Grid - Using C++ Module
    sadGrid = jps.sad_grid(sequence, templateSequence)

    # if log:
    #     print('Score (Sum of Absolute Differences) matrix:')
        # print(sadGrid)
        # print('\tShape: ({0},{1})'.format(sadGrid.shape[0], sadGrid.shape[1]))

    # Cascade the SAD Grid
    cascades = constructCascade(sadGrid, gapPenalty=gapPenalty, axis=1)

    # Pick Cascade and Roll sequence
    rollFactor = np.argmax(cascades[-1, -1, :])  # pick cascade with largest bottom right score
    score = cascades[-1, -1, rollFactor]
    # f1 = plt.figure()
    # a1 = f1.add_subplot(111)
    # a1.plot(cascades[len(templateSequence),len(sequence),:])
    # a1.scatter(rollFactor,score,s=80,c='k')
    # print(score,np.iinfo(sequence.dtype).max,sequence.size)
    score = (score + (np.iinfo(sequence.dtype).max * sequence.size/10)) / \
        (np.iinfo(sequence.dtype).max * sequence.size/10)
    if score <= 0:
        print('ISSUE: Negative Score')
    score = 0 if score < 0 else score
    # print(score)
    nwa = cascades[:, :, rollFactor]
    sequence = np.roll(sequence, rollFactor, axis=2)
    # if log:
    #     print('Cascade scores:\t', cascades[-1, -1, :])
    #     print('Chose cascade {0} of {1}:'.format(rollFactor, len(sequence)))
    #     # print(nwa)
    #     print('Shape: ({0},{1})'.format(nwa.shape[0], nwa.shape[1]))

    ## traverse here TODO
    (alignmentA, alignmentB) = traverseNW(sequence, templateSequence, nwa, log=='verbose')

    if log:
        print('Aligned sequence #1 (interpolated):\t', alignmentA)
        print('Aligned sequence #2 (interpolated):\t\t\t', alignmentB)

    if interpolationFactor is not None:# and isinstance(interpolationFactor,int):
        if log:
            print('De-interpolating for result...')
        # Divide by interpolation factor and modulo period
        # ignore -1s
        alignmentA[alignmentA >= 0] = (
            alignmentA[alignmentA >= 0]/interpolationFactor) % (period)
        alignmentB[alignmentB >= 0] = (
            alignmentB[alignmentB >= 0]/interpolationFactor) % (templatePeriod)

    if log:
        print('Aligned sequence #1 (wrapped):\t\t', alignmentA)

    for i in range(len(alignmentA)):
        if alignmentA[i] > -1:
            alignmentA[i] = (alignmentA[i]-rollFactor) % (period)

    # get rollFactor properly
    rollFactor = gtp.getPhase(alignmentA, alignmentB, knownTargetFrame, log)

    if log:
        print('Aligned sequence #1 (unwrapped):\t', alignmentA)
        print('Aligned sequence #2:\t\t\t', alignmentB)

    return alignmentA, alignmentB, rollFactor, score

@jit(nopython=False, parallel=False, fastmath=True, nogil=True)
def numbaCompilation(seqA='simple', log=False):
    '''Compiles all numba functions in module.'''

    # Toy Example
    # this sequence will be the template
    toySequenceB = np.asarray([0, 64, 192, 255, 255, 192, 128, 128, 64, 0],dtype='uint8')
    # this sequence will be matched to B
    if seqA=='simple':
        toySequenceA = np.roll(toySequenceB, 5)
    else:
        toySequenceA = np.asarray([255, 192, 128, 128, 64, 0, 64, 64, 64, 128, 192, 255],dtype='uint8')
    if log:
        print('Running toy example with:')
        print('\tSequence A: ', toySequenceA)
        print('\tSequence B: ', toySequenceB)

    # Make sequences 3D arrays (as expected for this algorithm)
    ndSequenceA = toySequenceA[:, np.newaxis, np.newaxis]
    ndSequenceB = toySequenceB[:, np.newaxis, np.newaxis]
    ndSequenceA = np.repeat(np.repeat(ndSequenceA,700,1),500,2)  # make each frame 2D to check everything works for image frames
    ndSequenceB = np.repeat(np.repeat(ndSequenceB,700,1),500,2)

    alignmentA, alignmentB, rollFactor, score = nCascadingNWA(
        ndSequenceA, ndSequenceB, len(toySequenceA)-0.5, len(toySequenceB)-0.66, gapPenalty=-255, interpolationFactor=1, log=log)
    if log:
        # print('Roll factor: {0} (score: {1})'.format(rollFactor, score))
        print('Alignment Maps:')
        # print('\tMap A: {0}'.format(alignmentA))
        # print('\tMap B: {0}'.format(alignmentB))

    # Outputs for toy examples
    alignedSequenceA = []  # Create new lists to fill with aligned values
    alignedSequenceB = []
    for i in alignmentA:  # fill new sequence A
        if i < 0:  # no matching element so repeat the last element in the sequence
            # if len(alignedSequenceA) > 0:
            #     alignedSequenceA.append(alignedSequenceA[-1])
            # else:  # i.e. a boundary case
            alignedSequenceA.append(-1)
        else:
            alignedSequenceA.append(linInterp(ndSequenceA, np.ceil(i))[0,0])
    # if alignedSequenceA[0] == -1:  # catch boundary case
        # alignedSequenceA[0] = alignedSequenceA[-1]
    for i in alignmentB:  # fill new sequence B
        if i < 0:  # no matching element so repeat the last element in the sequence
            # if len(alignedSequenceB) > 0:
            #     alignedSequenceB.append(alignedSequenceB[-1])
            # else:  # i.e. a boundary case
            alignedSequenceB.append(-1)
        else:
            alignedSequenceB.append(linInterp(ndSequenceB, np.ceil(i))[0,0])
    # if alignedSequenceB[0] == -1:  # catch boundary case
        # alignedSequenceB[0] = alignedSequenceB[-1]

    # Print
    if log:
        print('\nAligned Sequences:')
        print('\tSequence A: ', alignedSequenceA)
        print('\tSequence B: ', alignedSequenceB)

    return None


if __name__ == '__main__':
    # TODO - nunba doesn't actually seem to give a time benefit (at the mo - interpn still be rewritten)
    # Compile
    t = time.time()
    numbaCompilation(seqA='simple', log=False)
    print('Compilation run took {0:.2f} milliseconds.'.format(1000*(time.time()-t)))

    # Run - timed
    t = time.time()
    numbaCompilation(seqA='simple', log=False)
    print('Post-compilation run took {0:.2f} milliseconds.'.format(1000*(time.time()-t)))

    # Run - Python
    t = time.time()
    numbaCompilation.py_func(seqA='simple', log=False)
    print('Equivalent Python run took {0:.2f} milliseconds.'.format(1000*(time.time()-t)))

    # Profiler
    pr = cProfile.Profile()
    pr.enable()

    # Run - verbose
    numbaCompilation(seqA='complex', log=True)

    # Profile
    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats()
    # print(s.getvalue())