import numpy as np
from pprint import pprint
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.0f}\t".format(x)})

def constructCascade(scores,gp):
    # Construct grid
    nwa = np.zeros([scores.shape[0]+1,scores.shape[1]+1],'float')

    for t2 in range(scores.shape[0]+1):
        if t2==0:
            for t1 in range(1,scores.shape[1]+1):
                # ssd = np.sum((seq1[:,:,t1-1]-seq2[:,:,t2-1])**2)/seq1[:,:,t1-1].size
                mn = nwa[t2,t1-1]+gp
                nwa[t2,t1] = mn-scores[t2-1,t1-1]
        else:
            for t1 in range(scores.shape[1]+1):
                # ssd = np.sum((seq1[:,:,t1-1]-seq2[:,:,t2-1])**2)/seq1[:,:,t1-1].size
                if t1==0:
                    mn = nwa[t2-1,t1]+gp
                else:
                    mn = max([nwa[t2-1,t1-1],nwa[t2,t1-1]+gp,nwa[t2-1,t1]+gp])
                nwa[t2,t1] = mn-scores[t2-1,t1-1]
    return nwa

def nCascadingNWA(seq1,seq2,gapPenalty=0,log=False):
    ''' Assumes seq1 and seq2 are 3D numpy arrays of [t,x,y]'''
    if log:
        print('Sequence #1 has {0} frames and sequence #2 has {1} frames;'.format(len(seq1),len(seq2)))

    # Calculate SSDs
    ssds = np.zeros([len(seq2),len(seq1)],'float')

    for t2 in range(len(seq2)):
        for t1 in range(len(seq1)):
            ssds[t2,t1] = np.sum((seq1[t1]-seq2[t2])**2)/seq1[t1].size

    if log:
        print ('Score matrix:')
        print(ssds)
        print('Shape: ({0},{1})'.format(ssds.shape[0],ssds.shape[1]))

    # Make Cascades
    cascades = np.zeros([len(seq2)+1,len(seq1)+1,len(seq1)],'float')
    for n in range(len(seq1)):
        cascades[:,:,n] = constructCascade(ssds,gapPenalty)
        ssds = np.roll(ssds,1,axis=1)

    # Pick Cascade and Roll Seq1
    rollFactor = np.argmax(cascades[len(seq2),len(seq1),:])
    nwa = cascades[:,:,rollFactor]
    seq1 = np.roll(seq1,rollFactor,axis=2)
    if log:
        print ('Chose cascade {0} of {1}:'.format(rollFactor,len(seq1)))
        print(nwa)
        print('Shape: ({0},{1})'.format(nwa.shape[0],nwa.shape[1]))

    x = len(seq2)
    y = len(seq1)

    #  Traverse grid
    traversing = True

    # Trace without wrapping
    alignment1 = []
    alignment2 = []
    while traversing:
        xup = x-1
        yleft =  y-1
        if log:
            print('-----')
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('orig',x,y,nwa[x,y]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('diag',xup,yleft,nwa[xup,yleft]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('up  ',xup,y,nwa[xup,y]))
            print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('left',x,yleft,nwa[x,yleft]))

        if xup>=0:
            if  yleft>=0:
                options = [nwa[xup,yleft],nwa[xup,y],nwa[x,yleft]]
            else:
                if log:
                    print('Boundary Condition:\tI\'m at the left')
                options = [-np.inf,nwa[xup,y],-np.inf]
        else:
            if log:
                print('Boundary Condition:\tI\'m at the top')
            if  yleft>=0:
                options = [-np.inf,-np.inf,nwa[x,yleft]]
            else:
                if log:
                    print('Boundary Condition:\tI\'m at the top left')
                    print('Boundary Condition:\tI should not have  got here!')
                break
        direction = np.argmax(options)

        if direction==1:
            alignment1.append(-1)
            alignment2.append(xup)
            x = xup
            if log:
                print('Direction Travelled:\tI\'ve gone up')
        elif direction==0:
            alignment1.append(yleft)
            alignment2.append(xup)
            x = xup
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone diagonal')
        elif direction==2:
            alignment1.append(yleft)
            alignment2.append(-1)
            y = yleft
            if log:
                print('Direction Travelled:\tI\'ve gone left')
        if x==0 and y==0:
            if log:
                print('Traversing Complete')
            traversing = False

    # Reverses sequence
    alignment1 = alignment1[::-1]
    alignment2 = alignment2[::-1]

    if log:
        print('Aligned sequence #1 (wrapped):\t\t',alignment1)

    for i in range(len(alignment1)):
        if alignment1[i]>-1:
            alignment1[i] = (alignment1[i]-rollFactor)%len(seq1)

    if log:
        print('Aligned sequence #1 (unwrapped):\t',alignment1)
        print('Aligned sequence #2:\t\t\t',alignment2)

    return alignment1, alignment2, rollFactor

if __name__ == '__main__':
    print('Running toy example with')
    #Toy Example
    str1 = [0,1,2,3,4,5,6,6,7]
    str2 = [4,5,6,7,8,0,1,1,1,2,3]
    print('Sequence #1: {0}'.format(str1))
    print('Sequence #2: {0}'.format(str2))
    seq1  = np.asarray(str1).reshape([len(str1),1,1])
    seq2  = np.asarray(str2).reshape([len(str2),1,1])
    seq1 = np.repeat(np.repeat(seq1,10,1),5,2)
    seq2 = np.repeat(np.repeat(seq2,10,1),5,2)

    alignment1, alignment2, rF = nCascadingNWA(seq1,seq2,log=True)

    # Outputs for toy examples
    strout1 = []
    strout2 = []
    for i in alignment1:
        if i<0:
            strout1.append(-1)
        else:
            strout1.append(str1[i])
    for i in alignment2:
        if i<0:
            strout2.append(-1)
        else:
            strout2.append(str2[i])
    print(strout1)
    print(strout2)
