# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict
import torch
import torch.nn as nn
from ofa.utils import MyModule, MyNetwork, build_activation, get_same_padding, SEModule, ShuffleLayer, make_divisible
import torch.nn.init as init
import torch.nn.functional as F

def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ResNetBottleneckBlock.__name__: ResNetBottleneckBlock,
        BasicBlock.__name__:BasicBlock, #NEW
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)

class OurReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):

        return 0.01 * torch.pow(x, 2) + 0.5 * x
        #return torch.pow(x, 2)


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def _make_layer(planes, num_blocks, stride, _planes, exp):
        in_planes = _planes
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride, 'A', exp))
            in_planes = planes * exp

        return nn.Sequential(*layers)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MyModule):

    def __init__(self, in_planes, planes, stride=1, option='A', exp = 1):

        super(BasicBlock, self).__init__()

        self.in_channels = in_planes
        self.out_channels = planes
        self.kernel_size = 3
        self.stride = stride
        self.option = option
        self.expand_ratio = exp

        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1)

        self.shortcut = None #nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ConvLayer(in_planes, exp * planes, kernel_size=1, stride=stride, bias=False),
                     #nn.BatchNorm2d(exp * planes)
                )

    def forward(self, x):
        #ourRelu = OurReLU()
        #out = ourRelu(self.conv1(x))
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.shortcut is not None:
          out += self.shortcut(x)
        #out = ourRelu(out)
        out = F.relu(out)
        return out

    @property
    def module_str(self):
        return '(%s, %s)' % (
			'%dx%d_BasicBlock_%d->%d->%d' % (
				self.kernel_size, self.kernel_size, self.in_channels, self.out_channels,
				self.stride
			),
			'Lambda' if self.shortcut is not None else 'None',
		)

    @property
    def config(self):
      return {
        'name': BasicBlock.__name__,
        'in_planes': self.in_channels,
        'planes': self.out_channels,
        'stride': self.stride,
        'option': self.option,
        'exp': self.expand_ratio,
      }

    @staticmethod
    def build_from_config(config):
      return BasicBlock(**config)


class My2DLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 use_bn=True, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm2d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                # dropout before weight operation
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class ConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])
        conv_str += '_O%d' % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class DepthConvLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False,
                 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
        # default normal 3x3_DepthConv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order,
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict['depth_conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
            dilation=self.dilation, groups=self.in_channels, bias=False
        )
        weight_dict['point_conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, groups=self.groups, bias=self.bias
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict['shuffle'] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            conv_str = '%dx%d_DilatedDepthConv' % (kernel_size[0], kernel_size[1])
        else:
            conv_str = '%dx%d_DepthConv' % (kernel_size[0], kernel_size[1])
        conv_str += '_O%d' % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            'name': DepthConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
            'has_shuffle': self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)


class PoolingLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 pool_type, kernel_size=2, stride=2,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == 'avg':
            weight_dict['pool'] = nn.AvgPool2d(
                self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False
            )
        elif self.pool_type == 'max':
            weight_dict['pool'] = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return '%dx%d_%sPool' % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            'name': PoolingLayer.__name__,
            'pool_type': self.pool_type,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            **super(PoolingLayer, self).config
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)


class IdentityLayer(My2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class LinearLayer(MyModule):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_features)
            else:
                modules['bn'] = nn.BatchNorm1d(out_features)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class ZeroLayer(MyModule):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)


class MBInvertedConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, act_func='relu6', use_se=False):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', build_activation(self.act_func, inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', build_activation(self.act_func, inplace=True))
        ]
        if self.use_se:
            depth_conv_modules.append(('se', SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = '%dx%d_MBConv%d_%s' % (self.kernel_size, self.kernel_size, expand_ratio, self.act_func.upper())
        if self.use_se:
            layer_str = 'SE_' + layer_str
        layer_str += '_O%d' % self.out_channels
        return layer_str

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

class ResidualBlock(MyModule):

	def __init__(self, conv, shortcut):
		super(ResidualBlock, self).__init__()

		self.conv = conv
		self.shortcut = shortcut

	def forward(self, x):
		if self.conv is None or isinstance(self.conv, ZeroLayer):
			res = x
		elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
			res = self.conv(x)
		else:
			res = self.conv(x) + self.shortcut(x)
		return res

	@property
	def module_str(self):
		return '(%s, %s)' % (
			self.conv.module_str if self.conv is not None else None,
			self.shortcut.module_str if self.shortcut is not None else None
		)

	@property
	def config(self):
		return {
			'name': ResidualBlock.__name__,
			'conv': self.conv.config if self.conv is not None else None,
			'shortcut': self.shortcut.config if self.shortcut is not None else None,
		}

	@staticmethod
	def build_from_config(config):
		conv_config = config['conv'] if 'conv' in config else config['mobile_inverted_conv']
		conv = set_layer_from_config(conv_config)
		shortcut = set_layer_from_config(config['shortcut'])
		return ResidualBlock(conv, shortcut)

	@property
	def mobile_inverted_conv(self):
		return self.conv


class ResNetBottleneckBlock(MyModule):

	def __init__(self, in_channels, out_channels,
	             kernel_size=3, stride=1, expand_ratio=0.25, mid_channels=None, act_func='relu', groups=1,
	             downsample_mode='avgpool_conv'):
		super(ResNetBottleneckBlock, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.kernel_size = kernel_size
		self.stride = stride
		self.expand_ratio = expand_ratio
		self.mid_channels = mid_channels
		self.act_func = act_func
		self.groups = groups

		self.downsample_mode = downsample_mode

		if self.mid_channels is None:
			feature_dim = round(self.out_channels * self.expand_ratio)
		else:
			feature_dim = self.mid_channels

		feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
		self.mid_channels = feature_dim

		# build modules
		self.conv1 = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
			('bn', nn.BatchNorm2d(feature_dim)),
			('act', build_activation(self.act_func, inplace=True)),
		]))

		pad = get_same_padding(self.kernel_size)
		self.conv2 = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=groups, bias=False)),
			('bn', nn.BatchNorm2d(feature_dim)),
			('act', build_activation(self.act_func, inplace=True))
		]))

		self.conv3 = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(feature_dim, self.out_channels, 1, 1, 0, bias=False)),
			('bn', nn.BatchNorm2d(self.out_channels)),
		]))

		if stride == 1 and in_channels == out_channels:
			self.downsample = IdentityLayer(in_channels, out_channels)
		elif self.downsample_mode == 'conv':
			self.downsample = nn.Sequential(OrderedDict([
				('conv', nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)),
				('bn', nn.BatchNorm2d(out_channels)),
			]))
		elif self.downsample_mode == 'avgpool_conv':
			self.downsample = nn.Sequential(OrderedDict([
				('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
				('conv', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)),
				('bn', nn.BatchNorm2d(out_channels)),
			]))
		else:
			raise NotImplementedError

		self.final_act = build_activation(self.act_func, inplace=True)

	def forward(self, x):
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
			'%dx%d_BottleneckConv_%d->%d->%d_S%d_G%d' % (
				self.kernel_size, self.kernel_size, self.in_channels, self.mid_channels, self.out_channels,
				self.stride, self.groups
			),
			'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
		)

	@property
	def config(self):
		return {
			'name': ResNetBottleneckBlock.__name__,
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'expand_ratio': self.expand_ratio,
			'mid_channels': self.mid_channels,
			'act_func': self.act_func,
			'groups': self.groups,
			'downsample_mode': self.downsample_mode,
		}

	@staticmethod
	def build_from_config(config):
		return ResNetBottleneckBlock(**config)

