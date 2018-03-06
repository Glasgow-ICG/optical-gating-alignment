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
    # rawRefFrames: a PxMxN numpy array representing the new raw reference frames (or a list of numpy arrays representing the new raw reference frames)
    # thisPeriod: the period for the frames in rawRefFrames (caller must determine this)
    # knownPhaseIndex: the index into resampledSequences for which we have a known phase point we are trying to match
    # knownPhase: the phase we are trying to match
    # periodHistory: a list of the periods for resampledSequences
    # shifts: a list of all the shifts we have already calculated within resampledSequences
    # numSamplesPerPeriod: number of samples to use in uniform resampling of the period of data
    # maxOffsetToConsider: how far apart in the resampledSequences list to make comparisons
    #                      (this should be used to prevent comparing between sequences that are so far apart they have very little similarity)

    # Deal with rawRefFrames type
    if type(rawRefFrames) is list:
        rawRefFrames = np.vstack(rawRefFrames)

    # Check that the reference frames we have been given are compatible in shape with the history that we already have
    for seq in resampledSequences:
        for f in seq:
            if rawRefFrames[0].shape != f.image.shape:
                # There is a mismatch. Return an error code to indicate the problem
                return (resampledSequences, periodHistory, shifts, -1000.0)

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
                alignment1, alignment2, targ, score = scc.crossCorrelationRolling(resampledSequences[knownPhaseIndex],resampledSequences[i],numSamplesPerPeriod,numSamplesPerPeriod,target=knownPhase)
                if log:
                    print(score)
                    print('Using target of {0}'.format(targ))
            alignment0, alignment1, rF, score = scc.crossCorrelationRolling(resampledSequences[i][:numSamplesPerPeriod],thisResampledSequence[:numSamplesPerPeriod],numSamplesPerPeriod,numSamplesPerPeriod,target=targ)
            shifts.append((i, len(resampledSequences)-1, (rF-targ)%numSamplesPerPeriod, score))

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
    # rawRefFrames: a PxMxN numpy array representing the new raw reference frames (or a list of numpy arrays representing the new raw reference frames)
    # thisPeriod: the period for the frames in rawRefFrames (caller must determine this)
    # thisDrift: the drift for the frames in rawRefFrames (caller must determine this) --- NB Assumes this is reset after each stack (see accumulation below)
    # knownPhaseIndex: the index into resampledSequences for which we have a known phase point we are trying to match
    # knownPhase: the phase we are trying to match
    # periodHistory: a list of the periods for resampledSequences
    # driftHistory: a list of the drifts for resampledSequences
    # shifts: a list of all the shifts we have already calculated within resampledSequences
    # numSamplesPerPeriod: number of samples to use in uniform resampling of the period of data
    # maxOffsetToConsider: how far apart in the resampledSequences list to make comparisons
    #                      (this should be used to prevent comparing between sequences that are so far apart they have very little similarity)

    # Deal with rawRefFrames type
    if type(rawRefFrames) is list:
        rawRefFrames = np.vstack(rawRefFrames)

    # Check that the reference frames we have been given are compatible in shape with the history that we already have
    if len(resampledSequences)>1:
        for seq in resampledSequences:
            for f in seq:
                if rawRefFrames[0].shape != f.shape:
                    # There is a mismatch. Return an error code to indicate the problem
                    return (resampledSequences, periodHistory, shifts, -1000.0)

    # Resample latest reference frames and add them to our sequence set
    thisResampledSequence = scc.resampleImageSection(rawRefFrames, thisPeriod, numSamplesPerPeriod)
    resampledSequences.append(thisResampledSequence)
    periodHistory.append(thisPeriod)
    if len(driftHistory)>0:
        driftHistory.append([driftHistory[-1][0]+thisDrift[0],driftHistory[-1][1]+thisDrift[1]])# NB Accumulates drift
    else:
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
                alignment1, alignment2, targ, score = scc.crossCorrelationRolling(seq1,seq2,numSamplesPerPeriod,numSamplesPerPeriod,target=knownPhase)
                if log:
                    print('Using target of {0}'.format(targ))
            # drift = [thisDrift[0]-driftHistory[i][0], thisDrift[1]-driftHistory[i][1]] # NB not needed as thisDrift is NOT accumulated
            seq1,seq2 = afd.matchFrames(resampledSequences[i],resampledSequences[-1],thisDrift)
            alignment1, alignment2, rollFactor, score = scc.crossCorrelationRolling(seq1,seq2,numSamplesPerPeriod,numSamplesPerPeriod,target=targ)
            shifts.append((i, len(resampledSequences)-1, (rollFactor-targ)%numSamplesPerPeriod, score))

    if (log):
        pprint(shifts)

    (globalShiftSolution, adjustedShifts, adjacentSolution, residuals, initialAdjacentResiduals) = sgs.MakeShiftsSelfConsistent(shifts, len(resampledSequences), numSamplesPerPeriod, knownPhaseIndex, knownPhase)

    if (log):
        print('solution:')
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

    if log:
        print('residuals:')
        pprint(residuals)

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
