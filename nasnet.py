import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ReLUConvBN, Zero, FactorizedReduce, Identity

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
        '''
        OPS = [ Zero(self.C_in, self.C_out, self.stride), #none
            Pooling(self.C_in, self.C_out, self.stride if layer_idx == 0 else 1, self.bn_affine, self.bn_momentum,
                                     self.bn_track_running_stats), #avg_pool_3x3
            ReLUConvBN(self.C_in, self.C_out, 3, self.stride if layer_idx == 0 else 1, 1, 1, self.bn_affine, self.bn_momentum,
                                    self.bn_track_running_stats), #conv_3x3
            ReLUConvBN(self.C_in, self.C_out, 1, self.stride if layer_idx == 0 else 1, 0, 1, self.bn_affine, self.bn_momentum,
                                    self.bn_track_running_stats), #conv_1x1
            nn.Identity() if self.stride == 1 and self.C_in == self.C_out #skip_connect
             else FactorizedReduce(self.C_in, self.C_out, self.stride if layer_idx == 0 else 1, self.bn_affine, self.bn_momentum,
                                   self.bn_track_running_stats) #skip_connect
        ]
        '''

        OPS = [
             Zero(self.stride), #none
             nn.AvgPool2d(3, stride=self.stride, padding=1, count_include_pad=False), #avg pool 3x3
             ReLUConvBN(self.C_in, self.C_out, 3, self.stride if layer_idx == 0 else 1, 1, self.bn_affine), #conv 3x3
             ReLUConvBN(self.C_in, self.C_out, 3, self.stride if layer_idx == 0 else 1, 0, self.bn_affine), #conv 1x1
             Identity() if self.stride == 1 else FactorizedReduce(self.C_in, self.C_out, affine=self.bn_affine) #skip connect
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
    def __init__(self, out_channels=16):
        super(StemCell, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x #F.relu(x)

class FinalClassificationLayer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FinalClassificationLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) #Global average pooling
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