import numpy as np
import math
from pprint import pprint

import simpleCC as scc
import accountForDrift as afd
import getPhase as gtp
import sys
sys.path.insert(0, '../j_postacquisition/')
import shifts as shf
import shifts_global_solution as sgs

def processNewReferenceSequence(rawRefFrames, thisPeriod, resampledSequences, periodHistory, shifts, knownPhaseIndex=0, knownPhase=0, numSamplesPerPeriod=80, maxOffsetToConsider=2, log=False):
    #Stolen from JT but simplified/adapted
    # here rawRefFrames is a PxMxN numpy array representing the new raw reference frames

    # Add latest reference frames to our sequence set
    thisResampledSequence = scc.resampleImageSection(rawRefFrames, thisPeriod, numSamplesPerPeriod)
    resampledSequences.append(thisResampledSequence)
    periodHistory.append(thisPeriod)

    # Update our shifts array, comparing the current sequence with recent previous ones
    if (len(resampledSequences) > 1):
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(resampledSequences) - maxOffsetToConsider - 1)
        for i in range(firstOne, len(resampledSequences)-1):
            if log:
                print('---',i,len(resampledSequences)-1,'---')
            if i==knownPhaseIndex:
                if log:
                    print('Using knownPhase of {0}'.format(knownPhase))
                targ = knownPhase
            else:
                alignment1, alignment2, targ, score = scc.crossCorrelationRolling(resampledSequences[knownPhaseIndex],resampledSequences[i],80,80,target=knownPhase)
                if log:
                    print('Using target of {0}'.format(targ))
            alignment0, alignment1, rF, score = scc.crossCorrelationRolling(resampledSequences[i][:numSamplesPerPeriod],thisResampledSequence[:numSamplesPerPeriod],80,80,target=targ)
            shifts.append((i, len(resampledSequences)-1, rF-targ, score))

    # pprint(shifts)

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = sgs.MakeShiftsSelfConsistent(shifts, len(resampledSequences), numSamplesPerPeriod, knownPhaseIndex, knownPhase)

    # pprint(globalShiftSolution)

    residuals = np.zeros([len(globalShiftSolution),])
    for i in range(len(globalShiftSolution)-1):
        diff = 0
        for shift in shifts:
            if shift[1]==shifts[-1][1] and shift[0]==i:
                residuals[i] = (globalShiftSolution[-1] - globalShiftSolution[i] - shift[2])
                break
        while residuals[i]>(numSamplesPerPeriod/2):
            residuals[i] = residuals[i]-numSamplesPerPeriod
        while residuals[i]<-(numSamplesPerPeriod/2):
            residuals[i] = residuals[i]+numSamplesPerPeriod

    result = (resampledSequences, periodHistory, shifts, globalShiftSolution[-1], residuals)

    return result

def processNewReferenceSequenceWithDrift(rawRefFrames, thisPeriod, thisDrift, resampledSequences, periodHistory, driftHistory, shifts, knownPhaseIndex=0, knownPhase=0, numSamplesPerPeriod=80, maxOffsetToConsider=2, log=False):
    #Stolen from JT but simplified/adapted
    # here rawRefFrames is a PxMxN numpy array representing the new raw reference frames

    # Resample latest reference frames and add them to our sequence set
    thisResampledSequence = scc.resampleImageSection(rawRefFrames, thisPeriod, numSamplesPerPeriod)
    resampledSequences.append(thisResampledSequence)
    periodHistory.append(thisPeriod)
    driftHistory.append(thisDrift)

    # Update our shifts array, comparing the current sequence with recent previous ones
    if (len(resampledSequences) > 1):
        # Compare this new sequence against other recent ones
        firstOne = max(0, len(resampledSequences) - maxOffsetToConsider - 1)
        for i in range(firstOne, len(resampledSequences)-1):
            if log:
                print('---',i,len(resampledSequences)-1,'---')
            if i==knownPhaseIndex:
                if log:
                    print('Using knownPhase of {0}'.format(knownPhase))
                targ = knownPhase
            else:
                drift = [driftHistory[i][0]-driftHistory[knownPhaseIndex][0], driftHistory[i][1]-driftHistory[knownPhaseIndex][1]]
                seq1,seq2 = afd.matchFrames(resampledSequences[knownPhaseIndex],resampledSequences[i],drift)
                alignment1, alignment2, targ, score = scc.crossCorrelationRolling(seq1,seq2,80,80,target=knownPhase)
                if log:
                    print('Using target of {0}'.format(targ))
            drift = [thisDrift[0]-driftHistory[i][0], thisDrift[1]-driftHistory[i][1]]
            seq1,seq2 = afd.matchFrames(resampledSequences[i],resampledSequences[-1],drift)
            alignment1, alignment2, rollFactor, score = scc.crossCorrelationRolling(seq1,seq2,80,80,target=targ)
            shifts.append((i, len(resampledSequences)-1, rollFactor-targ, score))

    # pprint(shifts)

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = sgs.MakeShiftsSelfConsistent(shifts, len(resampledSequences), numSamplesPerPeriod, knownPhaseIndex, knownPhase)

    # pprint(globalShiftSolution)

    residuals = np.zeros([len(globalShiftSolution),])
    for i in range(len(globalShiftSolution)-1):
        diff = 0
        for shift in shifts:
            if shift[1]==shifts[-1][1] and shift[0]==i:
                residuals[i] = (globalShiftSolution[-1] - globalShiftSolution[i] - shift[2])
                break
        while residuals[i]>(numSamplesPerPeriod/2):
            residuals[i] = residuals[i]-numSamplesPerPeriod
        while residuals[i]<-(numSamplesPerPeriod/2):
            residuals[i] = residuals[i]+numSamplesPerPeriod

    result = (resampledSequences, periodHistory, driftHistory, shifts, globalShiftSolution[-1], residuals)

    return result
