'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from models.base import BranchModel
from models.costs import module_cost


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',
                 hidden_planes=None):
        super(BasicBlock, self).__init__()
        if hidden_planes is None:
            hidden_planes = planes
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_planes)
        self.conv2 = nn.Conv2d(hidden_planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (
                                            0, 0, 0, 0, planes // 4,
                                            planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def computational_cost(self, sample_image):
        c = 0

        sample_image = self.conv1(sample_image)
        c += module_cost(sample_image, self.conv1)

        sample_image = F.relu(self.bn1(sample_image))

        sample_image = self.conv2(sample_image)
        c += module_cost(sample_image, self.conv2)

        sample_image = self.bn2(sample_image)

        return sample_image, c

    def forward(self, x):
        c1 = self.conv1(x)
        out = F.relu(self.bn1(c1))
        c2 = self.conv2(out)
        out = self.bn2(c2)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, [out]


class ResNet(BranchModel, nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.b = 10

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.apply(_weights_init)

    def n_branches(self):
        return self.b

    def computational_cost(self, sample_image):
        def blocks_cost(x, blocks):
            costs = []

            for b in blocks:
                c = b.computational_cost(x)[1]
                costs.append(c)
                x, _ = b(x)

            return x, np.cumsum(costs)

        costs = defaultdict(int)

        if sample_image is None:
            sample_image = torch.randn((1, 3, 32, 32))

        device = next(self.parameters()).device
        sample_image = sample_image.to(device)

        c1 = self.conv1(sample_image)
        costs[0] = module_cost(c1, self.conv1)

        out = F.relu(self.bn1(c1))
        out, _, cs = self.iterate_blocks(out, self.layer1, True)

        bc = costs[len(costs) - 1]
        offset = len(costs)
        for i, c in enumerate(cs):
            costs[offset + i] = c + bc

        out, _, cs = self.iterate_blocks(out, self.layer2, True)

        bc = costs[len(costs) - 1]
        offset = len(costs)
        for i, c in enumerate(cs):
            costs[offset + i] = c + bc

        out, _, cs = self.iterate_blocks(out, self.layer3, True)

        bc = costs[len(costs) - 1]
        offset = len(costs)
        for i, c in enumerate(cs):
            costs[offset + i] = c + bc

        out = F.avg_pool2d(out, out.size()[3])

        costs[len(costs) - 2] += module_cost(out, nn.AvgPool2d(out.size()[3]))

        return costs

    def iterate_blocks(self, x, blocks, calculate_costs=False):
        intermediate_o = []
        costs = []

        for b in blocks:
            costs.append(b.computational_cost(x)[1])
            x, int = b(x)
            intermediate_o.extend(int)

        if calculate_costs:
            return x, intermediate_o, costs

        return x, intermediate_o

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.ModuleList(layers)

    def forward(self, x):
        intermediate_layers = []
        c1 = self.conv1(x)
        intermediate_layers.append(c1)
        out = F.relu(self.bn1(c1))
        out, int = self.iterate_blocks(out, self.layer1)
        intermediate_layers.extend(int)
        out, int = self.iterate_blocks(out, self.layer2)
        intermediate_layers.extend(int)
        out, int = self.iterate_blocks(out, self.layer3)
        out = F.avg_pool2d(out, out.size()[3])
        intermediate_layers.extend(int[:-1])
        intermediate_layers.append(out)

        return intermediate_layers


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])
