import numpy as np
import math

import nCascadingNW as cnw
import simpleCC as scc
import accountForDrift as afd
import sys
sys.path.insert(0, '../j_postacquisition/')
import shifts as shf
import shifts_global_solution as sgs

def processNewReferenceSequence(rawRefFrames, thisPeriod, resampledSequences, periodHistory, shifts, knownPhaseIndex=0, knownPhase=0, maxOffsetToConsider=3):
    # based on memoryCC
    # here rawRefFrames is a PxMxN numpy array representing the new raw reference frames

    # Add latest reference frames to our sequence set
    resampledSequences.append(rawRefFrames)
    periodHistory.append(thisPeriod)

    # Update our shifts array, comparing the current sequence with recent previous ones
    if (len(resampledSequences) > 1):
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(resampledSequences) - maxOffsetToConsider - 1)
        for i in range(firstOne, len(resampledSequences)-1):
            alignment1, alignment2, rollFactor, score = cnw.nCascadingNWA(resampledSequences[i],rawRefFrames,periodHistory[i],thisPeriod,target=knownPhaseIndex)
            shifts.append((i, len(resampledSequences)-1, rollFactor, score))

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = sgs.MakeShiftsSelfConsistent(shifts, len(resampledSequences), periodHistory, knownPhaseIndex, knownPhase)

    result = (resampledSequences,periodHistory, shifts, globalShiftSolution[-1])

    return result

def processNewReferenceSequenceWithDrift(rawRefFrames, thisPeriod, thisDrift, resampledSequences, periodHistory, driftHistory, shifts, knownPhaseIndex=0, knownPhase=0, maxOffsetToConsider=3):
    # based on memoryCC
    # here rawRefFrames is a PxMxN numpy array representing the new raw reference frames

    # Add latest reference frames to our sequence set
    resampledSequences.append(rawRefFrames)
    periodHistory.append(thisPeriod)
    driftHistory.append(thisDrift)

    # Update our shifts array, comparing the current sequence with recent previous ones
    if (len(resampledSequences) > 1):
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(resampledSequences) - maxOffsetToConsider - 1)
        for i in range(firstOne, len(resampledSequences)-1):
            drift = driftHistory[i] - thisDrift
            seq1,seq2 = afd.matchFrames(resampledSequences[i],rawRefFrames,drift)
            alignment1, alignment2, rollFactor, score = cnw.nCascadingNWA(seq1,seq2,periodHistory[i],thisPeriod,target=knownPhase)
            shifts.append((i, len(resampledSequences)-1, rollFactor, score))

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = sgs.MakeShiftsSelfConsistent(shifts, len(resampledSequences), periodHistory, knownPhaseIndex, knownPhase)

    result = (resampledSequences, periodHistory, driftHistory, shifts, globalShiftSolution[-1])

    return result
