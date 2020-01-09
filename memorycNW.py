'''Modules for maintaining phase lock during time-lapse imaging.
Based on cascading Needleman-Wunsch sequence alignment of reference sets.
Each new reference frame sequence should be processed after determination.
This module can correct for known drift between sequences.'''

# Python Imports
import numpy as np
from pprint import pprint
import sys
import warnings
# Local Imports
import simpleCC as scc
import nCascadingNW as cnw
import accountForDrift as afd
sys.path.insert(0, '../j_postacquisition/')
import shifts_global_solution as sgs


def processNewReferenceSequence(rawRefFrames,
                                thisPeriod,
                                thisDrift,
                                sequenceHistory,
                                periodHistory,
                                driftHistory,
                                shifts,
                                knownPhaseIndex=0,
                                knownPhase=0,
                                maxOffsetToConsider=3,
                                interpolationFactor=None,
                                gapPenalty=0,
                                log=True):
    ''' Based on memoryCC.processNewReferenceSequence

    Inputs:
    * rawRefFrames: a PxMxN numpy array representing the new reference frames
      (or a list of numpy arrays representing the new reference frames)
    * thisPeriod: the period for rawRefFrames (caller must determine this)
    * thisDrift: the drift for rawRefFrames (caller must determine this)
      * if None no drift correction is used
    * sequenceHistory: a list of previous reference frame sets
    * periodHistory: a list of the previous periods for sequenceHistory
    * driftHistory: a list of the previous drifts for sequenceHistory
      * if no drift correction is used, this is a dummy variable
    * shifts: a list of shifts previously calculated for sequenceHistory
    * knownPhaseIndex: the index of sequenceHistory for which knownPhase applies
    * knownPhase: the phase (index) we are trying to match in knownPhaseIndex
    * numSamplesPerPeriod: the number of samples to use in resampling
    * maxOffsetToConsider: how far apart historically to make comparisons
      * should be used to prevent comparing sequences that are far apart and have little similarity

    Outputs:
    * sequenceHistory: updated list of resampled reference frames
    * periodHistory: updated list of the periods for sequenceHistory
    * driftHistory: updated list of the drifts for sequenceHistory
      * if no drift correction is used, this is a dummy variable
    * shifts: updated list of shifts calculated for sequenceHistory
    * globalShiftSolution[-1]: roll factor for latest reference frames
    * residuals: residuals on least squares solution'''

    # Deal with rawRefFrames type
    if type(rawRefFrames) is list:
        rawRefFrames = np.vstack(rawRefFrames)

    # Check that the reference frames have a consistent shape
    for f in range(1, len(rawRefFrames)):
        if rawRefFrames[0].shape != rawRefFrames[f].shape:
            # There is a shape mismatch.
            if log:
                # Return an error message and code to indicate the problem.
                print('Error: There is shape mismatch within the new reference frames. Frame 0: {0}; Frame {1}: {2}'.format(rawRefFrames[0].shape, f, rawRefFrames[f].shape))
            return (sequenceHistory,
                    periodHistory,
                    driftHistory,
                    shifts,
                    -1000.0,
                    None)
    # And that shape is compatible with the history that we already have
    if len(sequenceHistory) > 1:
        if rawRefFrames[0].shape != sequenceHistory[0][0].shape:
            # There is a shape mismatch.
            if log:
                # Return an error message and code to indicate the problem.
                print('Error: There is shape mismatch with historical reference frames. Old shape: {1}; New shape: {2}'.format(sequenceHistory[0][0].shape, rawRefFrames[0].shape))
            return (sequenceHistory,
                    periodHistory,
                    driftHistory,
                    shifts,
                    -1000.0,
                    None)

    # Add latest reference frames to our sequence set
    thisResampledSequence = scc.resampleImageSection(rawRefFrames,
                                                     thisPeriod,
                                                     80)
    thisResampledSequence = thisResampledSequence.astype('uint8')
    sequenceHistory.append(thisResampledSequence)
    periodHistory.append(80)

    if thisDrift is not None:
        if len(driftHistory) > 0:
            # Accumulate the drift throughout history
            driftHistory.append([driftHistory[-1][0]+thisDrift[0],
                                 driftHistory[-1][1]+thisDrift[1]])
        else:
            driftHistory.append(thisDrift)
    else:
        if log:
            warnings.warn('No drift correction is being applied. This will seriously impact phase locking.',stacklevel=3)

    # Update our shifts array.
    # Compare the current sequence with recent previous ones
    if (len(sequenceHistory) > 1):
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(sequenceHistory) - maxOffsetToConsider - 1)
        for i in range(firstOne, len(sequenceHistory)-1):
            if log:
                print('---', i, len(sequenceHistory)-1, '---')
            if i == knownPhaseIndex:
                if log:
                    print('Using knownPhase of {0}'.format(knownPhase))
                targ = knownPhase
            else:
                if thisDrift is None:
                    alignment1, alignment2, targ, score = cnw.nCascadingNWA(sequenceHistory[knownPhaseIndex],
                                                                            sequenceHistory[i],
                                                                            periodHistory[knownPhaseIndex],
                                                                            periodHistory[i],
                                                                            log=log,
                                                                            gapPenalty=gapPenalty,
                                                                            interpolationFactor=interpolationFactor,
                                                                            knownTargetFrame=knownPhase)
                else:
                    drift = [driftHistory[i][0]-driftHistory[knownPhaseIndex][0],
                             driftHistory[i][1]-driftHistory[knownPhaseIndex][1]]
                    seq1, seq2 = afd.matchFrames(sequenceHistory[knownPhaseIndex],
                                                 sequenceHistory[i],
                                                 drift)
                    alignment1, alignment2, targ, score = cnw.nCascadingNWA(seq1,
                                                                            seq2,
                                                                            periodHistory[knownPhaseIndex],
                                                                            periodHistory[i],
                                                                            log=log,
                                                                            gapPenalty=gapPenalty,
                                                                            interpolationFactor=interpolationFactor,
                                                                            knownTargetFrame=knownPhase)
                if log:
                    print('Using target of {0}'.format(targ))
            if thisDrift is None:
                alignment1, alignment2, rollFactor, score = cnw.nCascadingNWA(sequenceHistory[i],
                                                                              sequenceHistory[-1],
                                                                              periodHistory[i],
                                                                              periodHistory[-1],
                                                                              log=log,
                                                                              gapPenalty=gapPenalty,
                                                                              interpolationFactor=interpolationFactor,
                                                                              knownTargetFrame=targ)
            else:
                seq1, seq2 = afd.matchFrames(sequenceHistory[i],
                                             sequenceHistory[-1],
                                             thisDrift)
                alignment1, alignment2, rollFactor, score = cnw.nCascadingNWA(seq1,
                                                                              seq2,
                                                                              periodHistory[i],
                                                                              periodHistory[-1],
                                                                              log=log,
                                                                              gapPenalty=gapPenalty,
                                                                              interpolationFactor=interpolationFactor,
                                                                              knownTargetFrame=targ)
            shifts.append((i,
                           len(sequenceHistory)-1,
                           rollFactor-targ,  # TODO - possible error?
                           1))  # add score here

    if log:
        print('printing shifts')
        pprint(shifts)

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = sgs.MakeShiftsSelfConsistent(shifts,
                                                                                                                                len(sequenceHistory),
                                                                                                                                periodHistory,
                                                                                                                               log=log,
                                                                                                                                knownPhaseIndex=knownPhaseIndex,
                                                                                                                                knownPhase=knownPhase)

    if log:
        print('solution:')
        pprint(globalShiftSolution)

    residuals = np.zeros([len(globalShiftSolution), ])
    for i in range(len(globalShiftSolution)-1):
        for shift in shifts:
            if shift[1] == shifts[-1][1] and shift[0] == i:
                residuals[i] = (globalShiftSolution[-1] - globalShiftSolution[i] - shift[2])
                break
        while residuals[i] > (periodHistory[i]/2):
            residuals[i] = residuals[i]-periodHistory[i]
        while residuals[i] < -(periodHistory[i]/2):
            residuals[i] = residuals[i]+periodHistory[i]

    if log:
        print('residuals:')
        pprint(residuals)
        print('Reference Frame rolling by: {0}'.format(globalShiftSolution[-1]))

    # Note for developers:
    # there are two other return statements in this function
    return (sequenceHistory,
            periodHistory,
            driftHistory,
            shifts,
            globalShiftSolution[-1],
            residuals)
