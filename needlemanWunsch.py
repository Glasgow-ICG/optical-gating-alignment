import numpy as np
from pprint import pprint
np.set_printoptions(threshold=np.nan)


#Test sequences
str1 = [0,1,2,3,4,5,6,6,7]
str2 = [1,1,1,2,3,4,5,6,7,8,0]
print(len(str1),len(str2))
seq1  = np.asarray(str1).reshape([1,1,len(str1)])
seq2  = np.asarray(str2).reshape([1,1,len(str2)])
seq1 = np.repeat(np.repeat(seq1,10,1),5,0)
seq2 = np.repeat(np.repeat(seq2,10,1),5,0)

# Construct grid
nwa = np.zeros([seq2.shape[2]+1,seq1.shape[2]+1],'float')

gp = 0
for t2 in range(seq2.shape[2]+1):
    if t2==0:
        for t1 in range(1,seq1.shape[2]+1):
            ssd = np.sum((seq1[:,:,t1-1]-seq2[:,:,t2-1])**2)/seq1[:,:,t1-1].size
            mn = nwa[t2,t1-1]+gp
            nwa[t2,t1] = mn-ssd
    else:
        for t1 in range(seq1.shape[2]+1):
            ssd = np.sum((seq1[:,:,t1-1]-seq2[:,:,t2-1])**2)/seq1[:,:,t1-1].size
            if t1==0:
                mn = nwa[t2-1,t1]+gp
            else:
                mn = max([nwa[t2-1,t1-1],nwa[t2,t1-1]+gp,nwa[t2-1,t1]+gp])
            nwa[t2,t1] = mn-ssd

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.0f}\t".format(x)})
print(nwa)

# #start at highest score on bottom row or rightmost column
# x = np.where(nwa[:,-1]==nwa[:,-1].max())[0][-1]
# y = np.where(nwa[-1,:]==nwa[-1,:].max())[0][-1]
# print(nwa[x,y])
# temp = np.argmax(np.asarray(nwa[x,-1],nwa[-1,y]))
# if x==seq2.shape[2] and y==seq1.shape[2]:
#     print('here1')
#     pass
# elif temp==0:#rightmost column
#     print('here2')
#     y = seq1.shape[2]
#     xstart=x
#     ystart=-1
# elif temp==1:#bottom row
#     print('here3')
#     x = seq2.shape[2]
#     ystart=y
#     xstart=-1
# print(x,y)
# print(xstart,ystart)

# #  Traverse grid
# traversing = True

# # Trace with X and Y wrapping
# alignment1 = []
# alignment2 = []
# wrappedX = False
# wrappedY = False
# while traversing:
#     xup = x-1
#     if xup == -1:
#         xup = seq2.shape[2]
#     yleft =  y-1
#     if yleft == -1:
#         yleft = seq1.shape[2]
#     print('-----')
#     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('orig',x,y,nwa[x,y]))
#     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('diag',xup,yleft,nwa[xup,yleft]))
#     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('up  ',xup,y,nwa[xup,y]))
#     print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('left',x,yleft,nwa[x,yleft]))
#
#     if xup>=0:
#         if  yleft>=0:
#             options = [nwa[xup,yleft],nwa[xup,y],nwa[x,yleft]]
#         else:
#             print('Boundary Condition:\tI\'m at the top')
#             options = [np.inf,nwa[xup,y],np.inf]
#     else:
#         print('Boundary Condition:\tI\'m at the left')
#         if  yleft>=0:
#             options = [np.inf,np.inf,nwa[x,yleft]]
#         else:
#             print('Boundary Condition:\tI\'m at the top')
#             print('Boundary Condition:\tI should not have  got here!')
#             break
#     direction = np.argmax(options)
#
#     if direction==1:
#         if x>0:
#             alignment1.append(-1)
#             alignment2.append(xup)
#         else:
#             print('Warning:\tWrapping around X period type 2')
#             wrappedX = True
#         x = xup
#         print('Direction Travelled:\tI\'ve gone up')
#     elif direction==0:
#         if y>0:
#             alignment1.append(yleft)
#         else:
#             print('Warning:\tWrapping around Y period type 0')
#             alignment1.append(-1)
#             wrappedY = True
#         if x>0:
#             alignment2.append(xup)
#         else:
#             print('Warning:\tWrapping around X period type 0')
#             alignment2.append(-1)
#             wrappedX = True
#         x = xup
#         y = yleft
#         print('Direction Travelled:\tI\'ve gone diagonal')
#     elif direction==2:
#         if y>0:
#             alignment1.append(yleft)
#             alignment2.append(-1)
#         else:
#             print('Warning:\tWrapping around Y period type 2')
#             wrappedY = True
#         y = yleft
#         print('Direction Travelled:\tI\'ve gone left')
#     if (x==0 and wrappedX==True) or (y==ystart and wrappedY==True):
#         print('Traversing Complete')
#         traversing = False

# # Roll Grid
# #so we start at bottom left and finish at top right
# roll = np.unravel_index(nwa.argmax(),nwa.shape)
# xroll = seq2.shape[2]-roll[0]
# yroll = seq1.shape[2]-roll[1]
# xroll = seq2.shape[2]-np.where(nwa[:,-1]==nwa[:,-1].max())[0][-1]
# yroll = seq1.shape[2]-np.where(nwa[-1,:]==nwa[-1,:].max())[0][-1]
# nwa = np.roll(nwa,(xroll,yroll),axis=(0,1))
# print(xroll,yroll)
# print(nwa)

x = seq2.shape[2]
y = seq1.shape[2]

#  Traverse grid
traversing = True

# Trace without wrapping
alignment1 = []
alignment2 = []
while traversing:
    xup = x-1
    yleft =  y-1
    print('-----')
    print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('orig',x,y,nwa[x,y]))
    print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('diag',xup,yleft,nwa[xup,yleft]))
    print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('up  ',xup,y,nwa[xup,y]))
    print('{0}:\tx={1:d};\ty={2:d};\tssd={3:.0f};'.format('left',x,yleft,nwa[x,yleft]))

    if xup>=0:
        if  yleft>=0:
            options = [nwa[xup,yleft],nwa[xup,y],nwa[x,yleft]]
        else:
            print('Boundary Condition:\tI\'m at the left')
            options = [-np.inf,nwa[xup,y],-np.inf]
    else:
        print('Boundary Condition:\tI\'m at the top')
        if  yleft>=0:
            options = [-np.inf,-np.inf,nwa[x,yleft]]
        else:
            print('Boundary Condition:\tI\'m at the top left')
            print('Boundary Condition:\tI should not have  got here!')
            break
    direction = np.argmax(options)

    if direction==1:
        alignment1.append(-1)
        alignment2.append(xup)
        x = xup
        print('Direction Travelled:\tI\'ve gone up')
    elif direction==0:
        alignment1.append(yleft)
        alignment2.append(xup)
        x = xup
        y = yleft
        print('Direction Travelled:\tI\'ve gone diagonal')
    elif direction==2:
        alignment1.append(yleft)
        alignment2.append(-1)
        y = yleft
        print('Direction Travelled:\tI\'ve gone left')
    if x==0 and y==0:
        print('Traversing Complete')
        traversing = False

# # Unroll aligments
# print(alignment1)
# print(alignment2)
# alignment1 = np.roll(alignment1,-yroll+1)
# alignment2 = np.roll(alignment2,-xroll+1)

# Reverses sequence
print(alignment1)
print(alignment2)
alignment1 = alignment1[::-1]
alignment2 = alignment2[::-1]
print(alignment1)
print(alignment2)

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
