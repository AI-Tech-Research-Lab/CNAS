from collections import defaultdict

import torch
from torch import nn

from models.base import BranchModel

class MobileNetV3(BranchModel,nn.Module):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer):

        super(MobileNetV3, self).__init__()

        self.b = 5 ##1+auxiliary
        self.exit_idxs = [3,5,7,9,11]

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
    
    def n_branches(self):
        return self.b

    def forward(self, x):
        
        i=0
        intermediate_layers = []
        x = self.first_conv(x)
        for idx,block in enumerate(self.blocks):
                    x = block(x)
                    if (idx==(self.exit_idxs[i])-1): #early exit
                                if(i==(self.b-1)): #final exit
                                       # Remove this part?
                                       x = self.final_expand_layer(x)
                                       x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
                                       x = self.feature_mix_layer(x)    
                                intermediate_layers.append(x)
                                i+=1

        return intermediate_layers


    

