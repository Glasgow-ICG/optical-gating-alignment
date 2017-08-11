import math
import numpy as np
import scipy.ndimage as sp

def getPhase(alignment1,alignment2,phase):
    ## Find precise index of phase in alignment1
    # case where exact phase is captured in alignment sequence
    if phase in alignment1:
        idxPos = alignment1.index(phase)
        print('Exact phase found at index {0} in alignment 1'.format(idxPos))
    # otherwise find lower and upper bounds and use linear interpolation
    else:
        a1l = -1;
        a1u = -1;
        allow = False#only turn on if I've seen a value smaller than desired
        s1len = len(alignment1)
        for idx1 in range(s1len):#for each position
            if not allow and alignment1[idx1]<=phase:
                allow=True
            if allow and alignment1[idx1]>=phase:#assign bounds, ignoring gaps in lower bound
                a1l = idx1-1
                a1u = idx1
                break
        if a1l<0 and a1u<0 and not allow:
            print('ERROR: Phase not found in alignment sequence 1')
            return None
        elif a1l<0 and a1u<0 and allow:
            print('WARNING: Wrapping around alignment sequence 1')
            a1l = idx1
            a1u = idx1+1
            while alignment1[a1u%s1len]<0:
                a1u = a1u+1
        # print(a1l,a1u,allow)
        # account for gaps in lower bound
        while alignment1[a1l]<0:
            a1l = a1l-1
        print('Interpolating with lower bound of {0} and upper bound of {1}'.format(a1l,a1u%s1len))

        interPos1 = (phase-alignment1[a1l])/(alignment1[a1u%s1len]-alignment1[a1l])
        idxPos = (interPos1*(a1u-a1l))+a1l
        # print(interPos1)
        print('Phase positioned at interpolated index {0} in alignment 1'.format(idxPos))

    ## Map precise index to alignment2, considering gaps
    # case where phase captured in alignment1 is integer and valid in alignment2
    if (idxPos//1)==idxPos and alignment2[int(idxPos)]>=0:
        phase = alignment2[int(idxPos)]
        print('Exact index used in alignment 2 to give a phase of {0}'.format(phase))
        # print(phase)
    else:
        s2len = len(alignment2)
        a2l = math.floor(idxPos)
        a2u = math.ceil(idxPos)
        # print(a2l,a2u)
        # check not same value (occurs when exactly hits an index)
        if a2l==a2u:
            a2u+=1
        while alignment2[a2l]<0:
            a2l = a2l-1
        while alignment2[a2u%s2len]<0:
            a2u = a2u+1
        print('Interpolating with lower bound of {0} and upper bound of {1}'.format(a2l,a2u%s2len))

        interPos2 = (idxPos-a2l)/(a2u-a2l)
        if alignment2[a2u%s2len]<alignment2[a2l]:
            phase = (interPos2*((alignment2[a2l]+alignment2[a2u%s2len]+1)-alignment2[a2l]))+alignment2[a2l]
        else:
            phase = (interPos2*(alignment2[a2u%s2len]-alignment2[a2l]))+alignment2[a2l]
        # print(interPos2)
        print('Interpolated index used to calculated phase of {0} in alignment 2'.format(phase))

    return phase

if __name__ == '__main__':
    alignment1 = [0,1,2,3,-1,4,5]
    alignment2 = [2,3,4,5,0,-1,1]
    phase = getPhase(alignment1,alignment2,3.25)
