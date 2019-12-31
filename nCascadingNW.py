'''Modules for phase matching two sequences based on sequence alignment.
Uses a cascading form of the Needleman Wunsch algorithm.
This module includes all necessary functions.'''

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

        x = np.arange(0, sequence.shape[1])
        y = np.arange(0, sequence.shape[2])
        z = np.arange(0, sequence.shape[0])

        # interp whole sequence
        zOut = np.linspace(0.0,
                           sequence.shape[0]-1,
                           sequence.shape[0]*interp)
        interpPoints = np.asarray(np.meshgrid(zOut, x, y, indexing='ij'))
        interpPoints = np.rollaxis(interpPoints, 0, 4)
        sequence = interpn((z, x, y), sequence, interpPoints)
        sequence = np.asarray(sequence, dtype=sequence.dtype)

        # take only up to first period
        return sequence[:int(period*interp), :, :]

def nCascadingNWA(seq1,
                  seq2,
                  period1,
                  period2,
                  gapPenalty=0,
                  interp=1,
                  target=0,
                  log=True):
    ''' Assumes seq1 and seq2 are 3D numpy arrays of [t,x,y]'''
    if log:
      print('Sequence #1 has {0} frames and sequence #2 has {1} frames;'.format(len(seq1),len(seq2)))

    if interp>1:
      if log:
          print('Interpolating by a factor of {0} for greater precision'.format(interp))
      seq1 = interpImageSeriesZ(seq1,period1,interp)
      seq2 = interpImageSeriesZ(seq2,period2,interp)
      if log:
          print('Sequence #1 has {0} frames and sequence #2 has {1} frames'.format(len(seq1),len(seq2)))
          print(seq1[:,0,0])
          print(seq2[:,0,0])
    else:
      if log:
          print('No interpolation required')
          print(seq1[:,0,0])
          print(seq2[:,0,0])

    # Calculate SSDs - C++
    ssds = jps.sad_grid(seq1, seq2)

    if log:
      print ('Score matrix:')
      print(ssds)
      print('Shape: ({0},{1})'.format(ssds.shape[0],ssds.shape[1]))

    # Cascade the SAD Grid
    cascades = constructCascade(sadGrid, gapPenalty=gapPenalty, axis=1)

    # Pick Cascade and Roll Seq1
    rollFactor = np.argmax(cascades[len(seq2),len(seq1),:])
    score = np.amax(cascades[len(seq2),len(seq1),:])
    # f1 = plt.figure()
    # a1 = f1.add_subplot(111)
    # a1.plot(cascades[len(seq2),len(seq1),:])
    # a1.scatter(rollFactor,score,s=80,c='k')
    # print(score,np.iinfo(seq1.dtype).max,seq1.size)
    score = (score + (np.iinfo(seq1.dtype).max * seq1.size/10))/(np.iinfo(seq1.dtype).max * seq1.size/10)
    if score<=0:
      print('ISSUE: Negative Score')
    score = 0 if score<0 else score#
    # print(score)
    nwa = cascades[:,:,rollFactor]
    seq1 = np.roll(seq1,rollFactor,axis=2)
    if log:
      print('Cascade scores:\t',cascades[len(seq2),len(seq1),:])
      print ('Chose cascade {0} of {1}:'.format(rollFactor,len(seq1)))
      print(nwa)
      print('Shape: ({0},{1})'.format(nwa.shape[0],nwa.shape[1]))

    x = len(seq2)
    y = len(seq1)

    #  Traverse grid
    traversing = True

    # Trace without wrapping
    alignmentA = np.zeros((0,))
    alignmentB = np.zeros((0,))
    while traversing:
      xup = x-1
      yleft =  y-1
      if log == 'Mega':
          print('-----')
          print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('orig',x,y,nwa[x,y]))
          print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('diag',xup,yleft,nwa[xup,yleft]))
          print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('up  ',xup,y,nwa[xup,y]))
          print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('left',x,yleft,nwa[x,yleft]))

      if xup>=0:
          if  yleft>=0:
              options = [nwa[xup,yleft],nwa[xup,y],nwa[x,yleft]]
          else:
              if log == 'Mega':
                  print('Boundary Condition:\tI\'m at the left')
              options = [-np.inf,nwa[xup,y],-np.inf]
      else:
          if log == 'Mega':
              print('Boundary Condition:\tI\'m at the top')
          if  yleft>=0:
              options = [-np.inf,-np.inf,nwa[x,yleft]]
          else:
              if log == 'Mega':
                  print('Boundary Condition:\tI\'m at the top left')
                  print('Boundary Condition:\tI should not have  got here!')
              break
      direction = np.argmax(options)

      if direction==1:
          alignmentA = np.append(alignmentA,-1)
          alignmentB = np.append(alignmentB,xup)
          x = xup
          if log == 'Mega':
              print('Direction Travelled:\tI\'ve gone up')
      elif direction==0:
          alignmentA = np.append(alignmentA,yleft)
          alignmentB = np.append(alignmentB,xup)
          x = xup
          y = yleft
          if log == 'Mega':
              print('Direction Travelled:\tI\'ve gone diagonal')
      elif direction==2:
          alignmentA = np.append(alignmentA,yleft)
          alignmentB = np.append(alignmentB,-1)
          y = yleft
          if log == 'Mega':
              print('Direction Travelled:\tI\'ve gone left')
      if x==0 and y==0:
          if log == 'Mega':
              print('Traversing Complete')
          traversing = False

    # Reverses sequence
    alignmentA = alignmentA[::-1]
    alignmentB = alignmentB[::-1]

    if interp>1:
      if log:
          print('De-interpolating for result...')
      alignmentA[alignmentA>=0] = (alignmentA[alignmentA>=0]/interp)%(period1-1)
      alignmentB[alignmentB>=0] = (alignmentB[alignmentB>=0]/interp)%(period2-1)
      # rollFactor = rollFactor/interp

    if log:
      print('Aligned sequence #1 (wrapped):\t\t',alignmentA)

    for i in range(len(alignmentA)):
      if alignmentA[i]>-1:
          alignmentA[i] = (alignmentA[i]-rollFactor)%(period1)

    # get rollFactor properly
    # print(target,alignmentA)
    rollFactor = gtp.getPhase(alignmentA,alignmentB,target,log)

    if log:
      print('Aligned sequence #1 (unwrapped):\t',alignmentA)
      print('Aligned sequence #2:\t\t\t',alignmentB)

    return alignmentA, alignmentB, rollFactor, score

def linInterp(string, floatPosition):
    '''A linear interpolation function - for toy example, i.e. __main__'''
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
    #Toy Example
    print('Running toy example with:')
    toySequenceA = [255,192,128,128,64,0,64,64,64,128,192,255]  # this sequence will be matched to B
    toySequenceB = [0,64,192,255,255,192,128,128,64,0]  # this sequence will be the template
    print('\tSequence #1: {0}'.format(toySequenceA))
    print('\tSequence #2: {0}'.format(toySequenceB))

    # Make sequences 3D arrays (as expected for this algorithm)
    ndSequenceA  = np.asarray(toySequenceA,'uint8')[:,np.newaxis,np.newaxis]
    ndSequenceB  = np.asarray(toySequenceB,'uint8')[:,np.newaxis,np.newaxis]

    alignmentA, alignmentB, rollFactor, score = nCascadingNWA(ndSequenceA,ndSequenceB,9.5,11.25,interp=1,log=True)
    print('Roll factor: {0} (score: {1})'.format(rollFactor,score))
    print('Alignment Maps:')
    print('\Map #1: {0}'.format(alignmentA))
    print('\Map #2: {0}'.format(alignmentB))

    # Outputs for toy examples
    alignedSequenceA = []  # Create new lists to fill with aligned values
    alignedSequenceB = []
    for i in alignmentA:  # fill new sequence A
        if i<0:  # no matching element so repeat the last element in the sequence
            if len(alignedSequenceA)>0:
                alignedSequenceA.append(alignedSequenceA[-1])
            else:  # i.e. a boundary case
                alignedSequenceA.append(-1)
        else:
            alignedSequenceA.append(linInterp(toySequenceA,i))
    if alignedSequenceA[0]==-1:  # catch boundary case
        alignedSequenceA[0] = alignedSequenceA[-1]
    for i in alignmentB:  # fill new sequence B
        if i<0:  # no matching element so repeat the last element in the sequence
            if len(alignedSequenceB)>0:
                alignedSequenceB.append(alignedSequenceB[-1])
            else:  # i.e. a boundary case
                alignedSequenceB.append(-1)
        else:
            alignedSequenceB.append(linInterp(toySequenceB,i))
    if alignedSequenceB[0]==-1:  # catch boundary case
        alignedSequenceB[0] = alignedSequenceB[-1]

    # Print
    print('\nAligned Sequences:\n\tA\t\tB')
    print('\n'.join('{0:8}\t{1:8}'.format(a, b) for a, b in zip(alignedSequenceA, alignedSequenceB)))



