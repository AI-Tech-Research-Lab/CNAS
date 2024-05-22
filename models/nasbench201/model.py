import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *


class NASBench201Cell(nn.Module):

    """
    Implementation from https://nni.readthedocs.io/en/v2.1/_modules/nni/nas/pytorch/nasbench201/nasbench201.html
    
    Builtin cell structure of NAS Bench 201. One cell contains four nodes. The First node serves as an input node
    accepting the output of the previous cell. And other nodes connect to all previous nodes with an edge that
    represents an operation chosen from a set to transform the tensor from the source node to the target node.
    Every node accepts all its inputs and adds them as its output.

    Parameters
    ---
    cell_id: str
        the name of this cell
    encode: list
        a list of integers representing the choice of the operation for each edge of this cell going from the first node to the last node
    C_in: int
        the number of input channels of the cell
    C_out: int
        the number of output channels of the cell
    stride: int
        stride of all convolution operations in the cell
    bn_affine: bool
        If set to ``True``, all ``torch.nn.BatchNorm2d`` in this cell will have learnable affine parameters. Default: True
    bn_momentum: float
        the value used for the running_mean and running_var computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, all ``torch.nn.BatchNorm2d`` in this cell tracks the running mean and variance. Default: True
    """

    def __init__(self, cell_id, encode, C_in, C_out, stride, bn_affine=False, bn_momentum=0.1, bn_track_running_stats=True):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = 4
        self.layers = nn.ModuleList()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.bn_affine = bn_affine
        self.bn_momentum = bn_momentum
        self.bn_track_running_stats = bn_track_running_stats

        #[(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)] archs of the DAG
        self._ops = nn.ModuleList()
        offset=0
        for i in range(1,self.NUM_NODES): #
            for j in range(i):
                op = self.get_op(encode[j+offset], j)
                self._ops.append(op)
            offset+=i

        self.cell_id = cell_id
    
    def get_op(self, op_idx, layer_idx):

        name_op = NAS_BENCH_201[op_idx]
        if layer_idx==0:
            op = OPS[name_op](self.C_in, self.C_out, self.stride, self.bn_affine, self.bn_track_running_stats)
        else:
            op = OPS[name_op](self.C_in, self.C_out, 1, self.bn_affine, self.bn_track_running_stats)
        return op
    
    def forward(self, input): 
        nodes = [input]
        offset = 0
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self._ops[j+offset](nodes[j]) for j in range(i))
            nodes.append(node_feature)
            offset += i
        return nodes[-1]

class NASBenchNet(nn.Module): #NASNetNetwork model built stacking 3 stages of five NASBench201 cells each with two residual blocks with stride 2 for downsampling in between stages
    def __init__(self, cell_encode, C, num_classes, stages, cells, steps=4):
        super(NASBenchNet, self).__init__()
        self.cell_encode = cell_encode
        self._num_classes = num_classes
        self._stages = stages #number of stages
        self.NUM_NODES = steps #number of nodes per cell
        self._cells = cells #number of cells per stage
        self.norm_cells = nn.ModuleList() #list of normal cells
        self.red_cells = nn.ModuleList() #list of reduction cells
        self.stem = nn.Sequential(
                        nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(C))
        C_curr = C
        for i in range(self._stages):
            self.norm_cells.append(self._make_stage(C_curr, self._cells))
            if i<self._stages-1:
                self.red_cells.append(ResNetBasicblock(C_curr, C_curr*2, stride=2))
                C_curr *= 2
        
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_curr), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_curr, num_classes)

    def _make_stage(self, in_channels, num_cells):
        cells = []
        for i in range(num_cells):
            cells.append(NASBench201Cell(i, self.cell_encode, in_channels, in_channels, 1))
        return nn.Sequential(*cells)

    def forward(self, input):
        x = self.stem(input)
        for i in range(self._stages):
            for j in range(self._cells):
                x = self.norm_cells[i][j](x) #cell j of stage i
            if i<self._stages-1:
                x = self.red_cells[i](x)
        x = self.lastact(x)
        x = self.global_pooling(x)
        logits = self.classifier(x.view(x.size(0),-1))
        return logits