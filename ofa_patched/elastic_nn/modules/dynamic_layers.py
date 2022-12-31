# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict
import copy

#ExitBlock
import torch
import torch.nn.functional as F
##

from ofa.layers import MBInvertedConvLayer, ConvLayer, LinearLayer, IdentityLayer, set_layer_from_config, ResNetBottleneckBlock
from ofa.imagenet_codebase.utils import MyModule, MyNetwork, int2list, get_net_device, build_activation, val2list
from ofa.elastic_nn.modules.dynamic_op import *
from ofa.elastic_nn.utils import adjust_bn_according_to_idx, copy_bn


class DynamicMBConvLayer(MyModule):
    
    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6, stride=1, act_func='relu6', use_se=False):
        super(DynamicMBConvLayer, self).__init__()
        
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        
        self.kernel_size_list = int2list(kernel_size_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)
        
        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        
        # build modules
        max_middle_channel = round(max(self.in_channel_list) * max(self.expand_ratio_list))
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicPointConv2d(max(self.in_channel_list), max_middle_channel)),
                ('bn', DynamicBatchNorm2d(max_middle_channel)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))
        
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSeparableConv2d(max_middle_channel, self.kernel_size_list, self.stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(max_middle_channel))
        
        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicPointConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))
        
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)
    
    def forward(self, x):
        in_channel = x.size(1)
        
        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = \
                make_divisible(round(in_channel * self.active_expand_ratio), 8)

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel
        
        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x
    
    @property
    def module_str(self):
        if self.use_se:
            return 'SE(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)
        else:
            return '(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)
    
    @property
    def config(self):
        return {
            'name': DynamicMBConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'stride': self.stride,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }
    
    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)

        # build the new layer
        sub_layer = MBInvertedConvLayer(
            in_channel, self.active_out_channel, self.active_kernel_size, self.stride, self.active_expand_ratio,
            act_func=self.act_func, mid_channels=middle_channel, use_se=self.use_se,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[:middle_channel, :in_channel, :, :]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(middle_channel, self.active_kernel_size).data
        )
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(self.depth_conv.se.fc.reduce.bias.data[:se_mid])

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(self.depth_conv.se.fc.expand.bias.data[:middle_channel])

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[:self.active_out_channel, :middle_channel, :, :]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width = sorted_expand_list[expand_ratio_stage]
            target_width = round(max(self.in_channel_list) * target_width)
            importance[target_width:] = torch.arange(0, target_width - importance.size(0), -1)
        
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )
        
        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # se expand: output dim 0 reorganize
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(se_expand.weight.data, 0, sorted_idx)
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 1, sorted_idx)
            # middle weight reorganize
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)
        
        # TODO if inverted_bottleneck is None, the previous layer should be reorganized accordingly
        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx


class DynamicConvLayer(MyModule):
    
    def __init__(self, in_channel_list, out_channel_list, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, act_func='relu6'):
        super(DynamicConvLayer, self).__init__()
        
        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func
        
        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func, inplace=True)
        
        self.active_out_channel = max(self.out_channel_list)
    
    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel
        
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x
    
    @property
    def module_str(self):
        return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self.kernel_size, self.stride)
    
    @property
    def config(self):
        return {
            'name': DynamicConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
        }
    
    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)
    
    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ConvLayer(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation,
            use_bn=self.use_bn, act_func=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))
        
        if not preserve_weight:
            return sub_layer
        
        sub_layer.conv.weight.data.copy_(self.conv.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)
        
        return sub_layer
        

class DynamicLinearLayer(MyModule):

    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):
        super(DynamicLinearLayer, self).__init__()
        
        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate
        
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list), max_out_features=self.out_features, bias=self.bias
        )
    
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)
    
    @property
    def module_str(self):
        return 'DyLinear(%d)' % self.out_features

    @property
    def config(self):
        return {
            'name': DynamicLinear.__name__,
            'in_features_list': self.in_features_list,
            'out_features': self.out_features,
            'bias': self.bias
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        
        sub_layer.linear.weight.data.copy_(self.linear.linear.weight.data[:self.out_features, :in_features])
        if self.bias:
            sub_layer.linear.bias.data.copy_(self.linear.linear.bias.data[:self.out_features])
        return sub_layer

class DynamicResNetBottleneckBlock(MyModule):

    def __init__(self, in_channel_list, out_channel_list, expand_ratio_list=0.25,
                 kernel_size=3, stride=1, act_func='relu', downsample_mode='avgpool_conv'):
        super(DynamicResNetBottleneckBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        # build modules
        max_middle_channel = make_divisible(
            round(max(self.out_channel_list) * max(self.expand_ratio_list)), MyNetwork.CHANNEL_DIVISIBLE)

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max(self.in_channel_list), max_middle_channel)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True)),
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max_middle_channel, kernel_size, stride)),
            ('bn', DynamicBatchNorm2d(max_middle_channel)),
            ('act', build_activation(self.act_func, inplace=True))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(max_middle_channel, max(self.out_channel_list))),
            ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
        ]))

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer(max(self.in_channel_list), max(self.out_channel_list))
        elif self.downsample_mode == 'conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list), stride=stride)),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        elif self.downsample_mode == 'avgpool_conv':
            self.downsample = nn.Sequential(OrderedDict([
                ('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
                ('conv', DynamicConv2d(max(self.in_channel_list), max(self.out_channel_list))),
                ('bn', DynamicBatchNorm2d(max(self.out_channel_list))),
            ]))
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.conv.active_out_channel = feature_dim
        self.conv2.conv.active_out_channel = feature_dim
        self.conv3.conv.active_out_channel = self.active_out_channel
        if not isinstance(self.downsample, IdentityLayer):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def module_str(self):
        return '(%s, %s)' % (
            '%dx%d_BottleneckConv_in->%d->%d_S%d' % (
                self.kernel_size, self.kernel_size, self.active_middle_channels, self.active_out_channel, self.stride
            ),
            'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
        )

    @property
    def config(self):
        return {
            'name': DynamicResNetBottleneckBlock.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'expand_ratio_list': self.expand_ratio_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
            'downsample_mode': self.downsample_mode,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicResNetBottleneckBlock(**config)

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        # build the new layer
        sub_layer = set_layer_from_config(self.get_active_subnet_config(in_channel))
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        sub_layer.conv1.conv.weight.data.copy_(
            self.conv1.conv.get_active_filter(self.active_middle_channels, in_channel).data)
        copy_bn(sub_layer.conv1.bn, self.conv1.bn.bn)

        sub_layer.conv2.conv.weight.data.copy_(
            self.conv2.conv.get_active_filter(self.active_middle_channels, self.active_middle_channels).data)
        copy_bn(sub_layer.conv2.bn, self.conv2.bn.bn)

        sub_layer.conv3.conv.weight.data.copy_(
            self.conv3.conv.get_active_filter(self.active_out_channel, self.active_middle_channels).data)
        copy_bn(sub_layer.conv3.bn, self.conv3.bn.bn)

        if not isinstance(self.downsample, IdentityLayer):
            sub_layer.downsample.conv.weight.data.copy_(
                self.downsample.conv.get_active_filter(self.active_out_channel, in_channel).data)
            copy_bn(sub_layer.downsample.bn, self.downsample.bn.bn)

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        return {
            'name': ResNetBottleneckBlock.__name__,
            'in_channels': in_channel,
            'out_channels': self.active_out_channel,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.active_expand_ratio,
            'mid_channels': self.active_middle_channels,
            'act_func': self.act_func,
            'groups': 1,
            'downsample_mode': self.downsample_mode,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        # conv3 -> conv2
        importance = torch.sum(torch.abs(self.conv3.conv.conv.weight.data), dim=(0, 2, 3))
        if isinstance(self.conv2.bn, DynamicGroupNorm):
            channel_per_group = self.conv2.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.conv3.conv.conv.weight.data = torch.index_select(self.conv3.conv.conv.weight.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
        self.conv2.conv.conv.weight.data = torch.index_select(self.conv2.conv.conv.weight.data, 0, sorted_idx)

        # conv2 -> conv1
        importance = torch.sum(torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3))
        if isinstance(self.conv1.bn, DynamicGroupNorm):
            channel_per_group = self.conv1.bn.channel_per_group
            importance_chunks = torch.split(importance, channel_per_group)
            for chunk in importance_chunks:
                chunk.data.fill_(torch.mean(chunk))
            importance = torch.cat(importance_chunks, dim=0)
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(round(max(self.out_channel_list) * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for expand in sorted_expand_list
            ]
            right = len(importance)
            base = - len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

        self.conv2.conv.conv.weight.data = torch.index_select(self.conv2.conv.conv.weight.data, 1, sorted_idx)
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(self.conv1.conv.conv.weight.data, 0, sorted_idx)

        return None

class ExitBlock(MyModule):

    """Exit Block definition.
    This allows the model to terminate early when it is confident for classification.
    The block returns:
    (1) pred, that is the predicted outcome
    (2) conf, that is the confidence value of the predicted outcome
    The block must be defined by the OFA supernet and passed in the init list for the Mbv3 block
    """

    def __init__(self, n_classes, final_expand_width, feature_dim, last_channel, dropout_rate):
        super(ExitBlock, self).__init__()

        # final expand layer, feature mix layer & classifier
        if len(final_expand_width) == 1:
            self.final_expand_layer = ConvLayer(max(feature_dim), max(final_expand_width), kernel_size=1, act_func='h_swish')
            self.feature_mix_layer = ConvLayer(
                max(final_expand_width), max(last_channel), kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
            )
        else:
            self.final_expand_layer = DynamicConvLayer(
                in_channel_list=feature_dim, out_channel_list=final_expand_width, kernel_size=1, act_func='h_swish'
            )
            self.feature_mix_layer = DynamicConvLayer(
                in_channel_list=final_expand_width, out_channel_list=last_channel, kernel_size=1,
                use_bn=False, act_func='h_swish',
            )
        if len(set(last_channel)) == 1:
            self.classifier = LinearLayer(max(last_channel), n_classes, dropout_rate=dropout_rate)
        else:
            self.classifier = DynamicLinearLayer(
                in_features_list=last_channel, out_features=n_classes, bias=True, dropout_rate=dropout_rate)
    
    def confidence(self,x):
        prob = F.softmax(x)
        top2_prob, top2_index = torch.topk(prob,2) 
        sm = torch.diff(top2_prob,dim=1)*(-1)
        return sm 

    def forward(self, x):

        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        conf = self.confidence(x)
        return x, conf