from collections import defaultdict

import torch
from torch import nn
import torch.nn.utils as utils

from models.EENN.base import BranchModel

def find_equidistant_points(N, M, start):

    #if M > N:
    #    return []  # Not enough positions in the array

    if M == 1:
        last_index = (N - 1) #// 2
        return [last_index]

    interval = (N - 1) / (M - 1)  # Calculate the interval between each equidistant point

    indexes = []
    for i in range(M):
        index = round(start + i * interval)  # Calculate the index of each equidistant point
        indexes.append(index)

    return indexes

def find_equidistant_points_v3(N, mask): #returns M intermediate equidistant points + last one
    #v3 introduces a mask selection given by the array mask
    
    indexes = []
    
    M = len(mask)
    
    if M == 1:
        indexes.append((N - 1) // 2)  # Invalid input, need at least 2 equidistant points
    else:
        interval = (N - 1) / (M + 1)  # Calculate the interval between each equidistant point

        for i in range(1, M + 1):
            index = round(i * interval)  # Calculate the index of each equidistant point
            if mask[i-1]==1:
              indexes.append(index)
    
    indexes.append(N-1) #include last idx

    return indexes

def find_equidistant_points_v2(N, M): #returns M intermediate equidistant points + last one
    
    indexes = []
    
    if M ==1:
        indexes.append((N - 1) // 2)  # Invalid input, need at least 2 equidistant points
    else:
        interval = (N - 1) / (M + 1)  # Calculate the interval between each equidistant point

        for i in range(1, M + 1):
            index = round(i * interval)  # Calculate the index of each equidistant point
            indexes.append(index)
    
    indexes.append(N-1) #include last idx

    return indexes

class MobileNetV3(BranchModel,nn.Module):

    def __init__(self, first_conv, blocks, branches, depth):# final_expand_layer, feature_mix_layer, b)

        super(MobileNetV3, self).__init__()

        #self.b = b #b (or ne = num exits) includes the final exit
        self.b = sum(branches)+1 #+1=last exit
        n_blocks = len(depth)
        idx=0 
        exit_list=[]
        for i in range(0,n_blocks-1):
            idx += depth[i]
            if (branches[i]==1):
                exit_list.append(idx)
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        exit_list.append(sum(depth)) #LAST EXIT
        self.exit_idxs=exit_list #= find_equidistant_points_v2(len(blocks),b-1)#exit_idxs 
    
    def n_branches(self):
        return self.b

    def forward(self, x):
        
        i=0
        intermediate_layers = []
        x = self.first_conv(x)
        
        for idx,block in enumerate(self.blocks):
                    x = block(x)
                    if (idx==self.exit_idxs[i]): #early exit
                                intermediate_layers.append(x)
                                i+=1
        return intermediate_layers

    


    

