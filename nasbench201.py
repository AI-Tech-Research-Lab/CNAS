import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import List, Text, Union, Dict, Optional
import numpy as np 
from layers import ReLUConvBN, Pooling, Zero, FactorizedReduce

def edge_to_position_mapping(edges):
    edge_mapping = {}
    for k, edge in enumerate(edges):
        edge_mapping[edge] = k
    return edge_mapping

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

    def __init__(self, cell_id, encode, C_in, C_out, stride, bn_affine=True, bn_momentum=0.1, bn_track_running_stats=True):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = 4
        self.layers = nn.ModuleList()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.bn_affine = bn_affine
        self.bn_momentum = bn_momentum
        self.bn_track_running_stats = bn_track_running_stats

        self.edge_mapping = edge_to_position_mapping([(j, i) for i in range(1,self.NUM_NODES) for j in range(0, i)])
        #[(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)] archs of the DAG

        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for j in range(0, i):
                encode_idx = self.edge_mapping[(j, i)]
                node_ops.append(self.get_op(encode[encode_idx], i))
            self.layers.append(node_ops)

        self.cell_id = cell_id
    
    def get_op(self, op_idx, layer_idx):
        OPS = [ Zero(self.C_in, self.C_out, self.stride), #none
             nn.Identity() if self.stride == 1 and self.C_in == self.C_out #skip_connect
             else FactorizedReduce(self.C_in, self.C_out, self.stride if layer_idx == 0 else 1, self.bn_affine, self.bn_momentum,
                                   self.bn_track_running_stats), #skip_connect
            ReLUConvBN(self.C_in, self.C_out, 1, self.stride if layer_idx == 0 else 1, 0, 1, self.bn_affine, self.bn_momentum,
                                    self.bn_track_running_stats), #conv_1x1
            ReLUConvBN(self.C_in, self.C_out, 3, self.stride if layer_idx == 0 else 1, 1, 1, self.bn_affine, self.bn_momentum,
                                    self.bn_track_running_stats), #conv_3x3
            Pooling(self.C_in, self.C_out, self.stride if layer_idx == 0 else 1, self.bn_affine, self.bn_momentum,
                                     self.bn_track_running_stats) #avg_pool_3x3
        ]
        return OPS[op_idx]

    def forward(self, input): # pylint: disable=W0622
        """
        Parameters
        ---
        input: torch.tensor
            the output of the previous layer
        """
        nodes = [input]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]

class StemCell(nn.Module):
    def __init__(self):
        super(StemCell, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class FinalClassificationLayer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FinalClassificationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.softmax(x, dim=1)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3),
            stride=stride, 
            padding=1, #if stride == 1 else 0,  # Dynamic padding calculation for stride 2 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride!=1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(
                        in_channels, 
                        out_channels, 
                        kernel_size=(1, 1), 
                        stride=1,
                        padding=0, 
                        bias=False
                    )
                )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        return x

class NASNet(nn.Module): #NASNet model built stacking 3 stages of five NASBench201 cells each with two residual blocks with stride 2 for downsampling in between stages
    def __init__(self, cell_encode, num_classes=1000):
        super(NASNet, self).__init__()
        self.cell_encode = cell_encode
        self.stem_cell = StemCell()
        self.stage1 = self._make_stage(16, 5)
        self.residual_block1 = ResidualBlock(16, 32, stride=2)
        self.stage2 = self._make_stage(32, 5)
        self.residual_block2 = ResidualBlock(32, 64, stride=2)
        self.stage3 = self._make_stage(64, 5)
        self.classification_layer = FinalClassificationLayer(64, num_classes)

    def _make_stage(self, in_channels, num_cells):
        cells = []
        for i in range(num_cells):
            cells.append(NASBench201Cell(i, self.cell_encode, in_channels, in_channels, 1))
        return nn.Sequential(*cells)

    def forward(self, x):
        x = self.stem_cell(x)
        x = self.stage1(x)
        x = self.residual_block1(x)
        x = self.stage2(x)
        x = self.residual_block2(x)
        x = self.stage3(x)
        x = self.classification_layer(x)
        return x
    
class NASBench201(): #NASBench201 dataset

    def __init__(self, dataset='ImageNet16-120', output_path='./output', model_path='../datasets/nasbench201_info.pt', device='cpu'):
        self.archive = torch.load(model_path, map_location=device)
        self.num_nodes = 4
        self.num_operations = 5
        self.nvar = int(self.num_nodes*(self.num_nodes-1)/2) #nvar is the len of the encoding. 6 is the number of edges in a 4-node cell
        self.n_archs = 15625 # 5**6 
        self.dataset = dataset
        #self.sec_obj = sec_obj
        self.output_path = output_path

    def str2matrix(self, arch_str: Text,
                search_space: List[Text] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']) -> np.ndarray:
        """
        This func shows how to convert the string-based architecture encoding to the encoding strategy in NAS-Bench-101.

        :param
        arch_str: the input is a string indicates the architecture topology, such as
                        |nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|
        search_space: a list of operation string, the default list is the search space for NAS-Bench-201
            the default value should be be consistent with this line https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/models/cell_operations.py#L24
        :return
        the numpy matrix (2-D np.ndarray) representing the DAG of this architecture topology
        :usage
        matrix = api.str2matrix( '|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|' )
        This matrix is 4-by-4 matrix representing a cell with 4 nodes (only the lower left triangle is useful).
            [ [0, 0, 0, 0],  # the first line represents the input (0-th) node
            [2, 0, 0, 0],  # the second line represents the 1-st node, is calculated by 2-th-op( 0-th-node )
            [0, 0, 0, 0],  # the third line represents the 2-nd node, is calculated by 0-th-op( 0-th-node ) + 0-th-op( 1-th-node )
            [0, 0, 1, 0] ] # the fourth line represents the 3-rd node, is calculated by 0-th-op( 0-th-node ) + 0-th-op( 1-th-node ) + 1-th-op( 2-th-node )
        In NAS-Bench-201 search space, 0-th-op is 'none', 1-th-op is 'skip_connect',
            2-th-op is 'nor_conv_1x1', 3-th-op is 'nor_conv_3x3', 4-th-op is 'avg_pool_3x3'.
        :(NOTE)
        If a node has two input-edges from the same node, this function does not work. One edge will be overlapped.
        """
        node_strs = arch_str.split('+')
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, node_str in enumerate(node_strs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            for xi in inputs:
                op, idx = xi.split('~')
                if op not in search_space: raise ValueError('this op ({:}) is not in {:}'.format(op, search_space))
                op_idx, node_idx = search_space.index(op), int(idx)
                matrix[i+1, node_idx] = op_idx
        return matrix
    
    def matrix2str(self, matrix: np.ndarray,
                  search_space: List[Text] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']) -> Text:
        arch_str = ""
        for i in range(1, self.num_nodes):
            node_str = "|"
            for j, op_idx in enumerate(matrix[i]):
                if i>j:
                    op = search_space[int(op_idx)]
                    node_str += "{}~{}|".format(op, j)
            arch_str += node_str + "+"
        # Remove the trailing '+' character
        arch_str = arch_str[:-1]
        return arch_str

    def matrix2vector(self, matrix):
        # Flatten lower left triangle of the matrix into a vector (diagonal not included)
        #num_nodes = matrix.shape[0]
        vector=[]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i > j:
                    vector.append(int(matrix[i, j]))
        return vector

    def sample(self, n_samples):
        archs = []
        vectors= []
        for i in range(n_samples):
            # Sample a vector of length num_nodes*(num_nodes-1)/2 with values in [0, num_operations) and add if not present
            while True:
                vector = np.random.randint(0, self.num_operations, int(self.num_nodes*(self.num_nodes-1)/2))
                if not any((vector == arr).all() for arr in vectors):
                    vectors.append(vector)
                    break

        for v in vectors:
            assert sum([(v == arr).all() for arr in vectors]) == 1
            arch = {'arch': self.matrix2str(self.vector2matrix(v))}
            archs.append(arch)

        return archs

    def vector2matrix(self,vector):
        #l = len(vector)
        #num_nodes = int((1 + math.sqrt(1+8*l))/2)
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        idx = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i > j:
                    matrix[i, j] = vector[idx]
                    idx += 1
        return matrix
    
    def encode(self, config):
        return self.matrix2vector(self.str2matrix(config['arch']))
    
    def decode(self, vector):
        return {'arch':self.matrix2str(self.vector2matrix(vector))}
    
    def evaluate(self, archs, it=0):
        gen_dir = os.path.join(self.output_path, "iter_{}".format(it))
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir, exist_ok=True)
        #stats = []
        for n_subnet, arch in enumerate(archs):
            stat = self.get_info_from_arch(arch)
            net_path = os.path.join(gen_dir, "net_{}".format(n_subnet))
            if not os.path.exists(net_path):
                os.makedirs(net_path, exist_ok=True)
            save_path = os.path.join(net_path, 'net_{}.stats'.format(n_subnet)) 
            with open(save_path, 'w') as handle:
                json.dump(stat, handle)
            save_path = os.path.join(net_path, 'net_{}.subnet'.format(n_subnet))
            #config={'arch': arch}
            #print("CONFIG: ", arch)
            with open(save_path, 'w') as handle:
                json.dump(arch, handle)
            #f_obj = stat['top1']
            #s_obj = stat[self.sec_obj]
            #stats.append((f_obj, s_obj))
        #return stats

    def get_info_from_arch(self, config):
        str = config['arch']
        matrix = self.str2matrix(str)
        #matrix = self.vector2matrix(config['arch'])
        idx=0
        for i in range(self.n_archs):
            if np.array_equal(self.str2matrix(self.archive['str'][i]), matrix):
                break
            else:
                idx+=1
        info={}
        info['test-acc']=np.round(self.archive['test-acc'][self.dataset][idx],3)
        if self.dataset=='cifar10':
            val_dataset = self.dataset + '-valid'
        else:
            val_dataset = self.dataset
        info['val-acc']=np.round(self.archive['val-acc'][val_dataset][idx],3)
        info['flops']=np.round(self.archive['flops'][self.dataset][idx],3)
        info['params']=np.round(self.archive['params'][self.dataset][idx],3)
        str=self.archive['str'][idx]
        info['top1']=np.round(100 - self.archive['val-acc'][val_dataset][idx],3) # top1 error
        return info
    

