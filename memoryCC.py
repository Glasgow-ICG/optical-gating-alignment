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

    rawRefFrames = np.asarray(rawRefFrames)

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
                    print(score)
                    print('Using target of {0}'.format(targ))
            alignment0, alignment1, rF, score = scc.crossCorrelationRolling(resampledSequences[i][:numSamplesPerPeriod],thisResampledSequence[:numSamplesPerPeriod],80,80,target=targ)
            shifts.append((i, len(resampledSequences)-1, rF-targ, score))

    pprint(shifts)
    print(len(resampledSequences), numSamplesPerPeriod, knownPhaseIndex, knownPhase)

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

def processNewReferenceSequenceWithDrift(rawRefFrames, thisPeriod, thisDrift, resampledSequences, periodHistory, driftHistory, shifts, knownPhaseIndex=0, knownPhase=0, numSamplesPerPeriod=80, maxOffsetToConsider=2, log=True):
    #Stolen from JT but simplified/adapted
    # here rawRefFrames is a PxMxN numpy array representing the new raw reference frames

    if type(rawRefFrames) is list:
        rawRefFrames = np.vstack(rawRefFrames)

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

    if (log):
        pprint(shifts)

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = sgs.MakeShiftsSelfConsistent(shifts, len(resampledSequences), numSamplesPerPeriod, knownPhaseIndex, knownPhase)

    if (log):
        pprint(globalShiftSolution)

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

if __name__ == '__main__':
    print('Running toy example...This is BROKEN')
    numStacks = 10
    stackLength = 10
    width = 10
    height = 10

    resampledSequences = []
    periodHistory = []
    shifts = []
    driftHistory = []

    for i in range(numStacks):
        print('Stack {0}'.format(i))
        # Make new toy sequence
        thisPeriod = stackLength-0.5
        seq1 = (np.arange(stackLength)+np.random.randint(0,stackLength+1))%thisPeriod
        print('New Sequence: {0}; Period: {1} ({2})'.format(seq1,thisPeriod,len(seq1)))
        seq2  = np.asarray(seq1,'uint8').reshape([len(seq1),1,1])
        seq2 = np.repeat(np.repeat(seq2,width,1),height,2)

        # Run MCC
        resampledSequences, periodHistory, driftHistory, shifts, rF, residuals = processNewReferenceSequenceWithDrift(seq2, thisPeriod, [0,0], resampledSequences, periodHistory, driftHistory, shifts, knownPhaseIndex=0, knownPhase=stackLength/2, numSamplesPerPeriod=80, maxOffsetToConsider=3, log=True)

        print(thisPeriod*rF/80)

        # Outputs for toy examples
        seqOut = (seq1+rF)%thisPeriod
        print('Aligned Sequence: {0}'.format(seqOut))
        print('---')
