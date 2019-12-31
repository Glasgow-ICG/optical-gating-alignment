'''Modules for phase-matching/aligning two quasi-periodic sequences.
Uses a cascading form of the Needleman-Wunsch algorithm.
'''

# Python Imports
import sys

# Module Imports
import numpy as np
from scipy.interpolate import interpn

# Custom Module Imports
import j_py_sad_correlation as jps

# Local Imports
import getPhase as gtp


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
    # Function to create each cascaded score array
    def newScores(scores, gapPenalty=0):
        cascadedScores = np.zeros([scores.shape[0]+1, scores.shape[1]+1], dtype='float')
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

    # Create 3D array to hold all cascades
    cascades = np.zeros([scores.shape[0]+1, scores.shape[1]+1, scores.shape[0]], dtype='float')

    # Create a new cascaded score array for each alignment (by rolling along axis)
    for n in np.arange(scores.shape[1-axis]):  # the 1-axis tricks means we loop over 0 if axis=1 and vice versa
        cascades[:, :, n] = newScores(scores, gapPenalty)
        scores = np.roll(scores, 1, axis=axis)

    return cascades


def interpImageSeriesZ(sequence, period, interpolationFactor):
    '''
    '''
    x = np.arange(0, sequence.shape[1])
    y = np.arange(0, sequence.shape[2])
    z = np.arange(0, sequence.shape[0])

    # Interpolate whole sequence
    zOut = np.linspace(0.0,
                       sequence.shape[0]-1,
                       sequence.shape[0]*interpolationFactor)
    interpPoints = np.asarray(np.meshgrid(zOut, x, y, indexing='ij'))
    interpPoints = np.rollaxis(interpPoints, 0, 4)
    sequence = interpn((z, x, y), sequence, interpPoints)
    sequence = np.asarray(sequence, dtype=sequence.dtype)

    # take only up to first period
    return sequence[:int(period*interpolationFactor), :, :]


def nCascadingNWA(sequence,
                  templateSequence,
                  period,
                  templatePeriod,
                  gapPenalty=0,
                  interpolationFactor=1,
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
    if log:
        print('Sequence #1 has {0} frames and sequence #2 has {1} frames;'.format(
            len(sequence), len(templateSequence)))

    # Interpolate Sequence (for finer alignment)
    if interpolationFactor > 1:
        if log:
            print(
                'Interpolating by a factor of {0} for greater precision'.format(interpolationFactor))
        
        # interpolate
        sequence = interpImageSeriesZ(sequence, period, interpolationFactor)
        templateSequence = interpImageSeriesZ(templateSequence, templatePeriod, interpolationFactor)
        
        if log:
            print('\tSequence #1 now has {0} frames and sequence #2 now has {1} frames'.format(
                len(sequence), len(templateSequence)))

    # Calculate SAD Grid - Using C++ Module
    sadGrid = jps.sad_grid(sequence, templateSequence)

    if log:
        print('Score (Sum of Absolute Differences) matrix:')
        print(sadGrid)
        print('\tShape: ({0},{1})'.format(sadGrid.shape[0], sadGrid.shape[1]))

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
    if log:
        print('Cascade scores:\t', cascades[-1, -1, :])
        print('Chose cascade {0} of {1}:'.format(rollFactor, len(sequence)))
        print(nwa)
        print('Shape: ({0},{1})'.format(nwa.shape[0], nwa.shape[1]))

    x = len(templateSequence)
    y = len(sequence)

    #  Traverse grid
    traversing = True

    # Trace without wrapping
    alignmentA = np.zeros((0,))
    alignmentB = np.zeros((0,))
    while traversing:
        xup = x-1
        yleft = y-1
        if log == 'Mega':
            print('-----')
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
                'orig', x, y, nwa[x, y]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
                'diag', xup, yleft, nwa[xup, yleft]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
                'up  ', xup, y, nwa[xup, y]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format(
                'left', x, yleft, nwa[x, yleft]))

        if xup >= 0:
            if yleft >= 0:
                options = [nwa[xup, yleft], nwa[xup, y], nwa[x, yleft]]
            else:
                if log == 'Mega':
                    print('Boundary Condition:\tI\'m at the left')
                options = [-np.inf, nwa[xup, y], -np.inf]
        else:
            if log == 'Mega':
                print('Boundary Condition:\tI\'m at the top')
            if yleft >= 0:
                options = [-np.inf, -np.inf, nwa[x, yleft]]
            else:
                if log == 'Mega':
                    print('Boundary Condition:\tI\'m at the top left')
                    print('Boundary Condition:\tI should not have  got here!')
                break
        direction = np.argmax(options)

        if direction == 1:
            alignmentA = np.append(alignmentA, -1)
            alignmentB = np.append(alignmentB, xup)
            x = xup
            if log == 'Mega':
                print('Direction Travelled:\tI\'ve gone up')
        elif direction == 0:
            alignmentA = np.append(alignmentA, yleft)
            alignmentB = np.append(alignmentB, xup)
            x = xup
            y = yleft
            if log == 'Mega':
                print('Direction Travelled:\tI\'ve gone diagonal')
        elif direction == 2:
            alignmentA = np.append(alignmentA, yleft)
            alignmentB = np.append(alignmentB, -1)
            y = yleft
            if log == 'Mega':
                print('Direction Travelled:\tI\'ve gone left')
        if x == 0 and y == 0:
            if log == 'Mega':
                print('Traversing Complete')
            traversing = False

    # Reverses sequence
    alignmentA = alignmentA[::-1]
    alignmentB = alignmentB[::-1]

    if interpolationFactor > 1:
        if log:
            print('De-interpolating for result...')
        alignmentA[alignmentA >= 0] = (
            alignmentA[alignmentA >= 0]/interpolationFactor) % (period-1)
        alignmentB[alignmentB >= 0] = (
            alignmentB[alignmentB >= 0]/interpolationFactor) % (templatePeriod-1)

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


def linInterp(string, floatPosition):
    '''A linear interpolation function - for toy example, i.e. __main__
    '''
    if floatPosition//1 == floatPosition:
        # equivalent to string[np.floor(floatPosition).astype('int)]
        return string[floatPosition.astype('int')]
    else:
        # Interpolation Ratio
        interPos = floatPosition-(floatPosition//1)

        # Positions
        # equivalent to np.floor(floatPosition).astype('int)
        botPos = floatPosition.astype('int')
        topPos = np.ceil(floatPosition).astype('int')

        # Values
        botVal = string[botPos]
        topVal = string[topPos]

        return botVal + interPos*(topVal - botVal)


if __name__ == '__main__':
    # Toy Example
    print('Running toy example with:')
    toySequenceA = [255, 192, 128, 128, 64, 0, 64, 64, 64,
                    128, 192, 255]  # this sequence will be matched to B
    # this sequence will be matched to B
    toySequenceA = np.roll([0, 64, 192, 255, 255, 192, 128, 128, 64, 0], 5)
    # this sequence will be the template
    toySequenceB = [0, 64, 192, 255, 255, 192, 128, 128, 64, 0]
    print('\tSequence A: {0}'.format(toySequenceA))
    print('\tSequence B: {0}'.format(toySequenceB))

    # Make sequences 3D arrays (as expected for this algorithm)
    ndSequenceA = np.asarray(toySequenceA, 'uint8')[:, np.newaxis, np.newaxis]
    ndSequenceB = np.asarray(toySequenceB, 'uint8')[:, np.newaxis, np.newaxis]

    alignmentA, alignmentB, rollFactor, score = nCascadingNWA(
        ndSequenceA, ndSequenceB, 9.5, 11.25, interpolationFactor=1, log=True)
    print('Roll factor: {0} (score: {1})'.format(rollFactor, score))
    print('Alignment Maps:')
    print('\tMap A: {0}'.format(alignmentA))
    print('\tMap B: {0}'.format(alignmentB))

    # Outputs for toy examples
    alignedSequenceA = []  # Create new lists to fill with aligned values
    alignedSequenceB = []
    for i in alignmentA:  # fill new sequence A
        if i < 0:  # no matching element so repeat the last element in the sequence
            if len(alignedSequenceA) > 0:
                alignedSequenceA.append(alignedSequenceA[-1])
            else:  # i.e. a boundary case
                alignedSequenceA.append(-1)
        else:
            alignedSequenceA.append(linInterp(toySequenceA, i))
    if alignedSequenceA[0] == -1:  # catch boundary case
        alignedSequenceA[0] = alignedSequenceA[-1]
    for i in alignmentB:  # fill new sequence B
        if i < 0:  # no matching element so repeat the last element in the sequence
            if len(alignedSequenceB) > 0:
                alignedSequenceB.append(alignedSequenceB[-1])
            else:  # i.e. a boundary case
                alignedSequenceB.append(-1)
        else:
            alignedSequenceB.append(linInterp(toySequenceB, i))
    if alignedSequenceB[0] == -1:  # catch boundary case
        alignedSequenceB[0] = alignedSequenceB[-1]

    # Print
    print('\nAligned Sequences:\n\tA\t\tB')
    print('\n'.join('{0:8}\t{1:8}'.format(a, b)
                    for a, b in zip(alignedSequenceA, alignedSequenceB)))
