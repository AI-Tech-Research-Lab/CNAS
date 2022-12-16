import torch
import torch.nn as nn

from ofa.layers import set_layer_from_config, ConvLayer, IdentityLayer, LinearLayer
from ofa.layers import BasicBlock, _weights_init, OurReLU, _make_layer
from ofa.layers import ResNetBottleneckBlock, ResidualBlock
from ofa.utils import make_divisible, MyNetwork, MyGlobalAvgPool2d

import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNets','ResNetsHE', 'ResNet50HE', 'ResNet50DHE']

BASE_DEPTH_LIST = [3,3,3] 
STAGE_WIDTH_LIST = [16,32,64]
STRIDE_LIST = [1,2,2]


class ResNetsHE(MyNetwork):

	def __init__(self, input_stem, blocks, classifier):
		super(ResNetsHE, self).__init__()

		self.input_stem = nn.ModuleList(input_stem)	
		self.blocks = nn.ModuleList(blocks)
		self.classifier = classifier
		self.apply(_weights_init)

	def forward(self, x):

		for layer in self.input_stem:
				x = layer(x)
		#ourRelu = OurReLU()
		#x = ourRelu(x)
		x = F.relu(x)
		for block in self.blocks:
				x = block(x)
		ks = x.size(dim=3)
		x = F.avg_pool2d(input = x, 
         kernel_size = (ks,ks))
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = ''
		for layer in self.input_stem:
			_str += layer.module_str + '\n'
		for block in self.blocks.children():
			_str += block.module_str + '\n'
		_str += self.classifier.module_str
		return _str

	@property
	def config(self):
		blocks = []
		for seq in self.blocks:
				for b in seq.children():
						blocks.append(b)

             
		return {
			'name': ResNetsHE.__name__,
			'bn': self.get_bn_param(),
			'input_stem': [
				layer.config for layer in self.input_stem 
			],
			'blocks': blocks,
			'classifier': self.classifier.config,
		}

	@staticmethod
	def build_from_config(config):
		classifier = set_layer_from_config(config['classifier'])

		input_stem = []
		for layer_config in config['input_stem']:
			input_stem.append(set_layer_from_config(layer_config))
		blocks = []
		for block in config['blocks']:
			blocks.append(set_layer_from_config(block.config))

		net = ResNetsHE(input_stem, blocks, classifier)
		if 'bn' in config:
			net.set_bn_param(**config['bn'])
		else:
			net.set_bn_param(momentum=0.1, eps=1e-5)

		return net

	def zero_last_gamma(self):
		for m in self.modules():
			if isinstance(m, BasicBlock) and isinstance(m.downsample, IdentityLayer):
				m.conv3.bn.weight.data.zero_()

	@property
	def grouped_block_index(self):
		info_list = []
		block_index_list = []
		for i, block in enumerate(self.blocks):
			if not isinstance(block.downsample, IdentityLayer) and len(block_index_list) > 0:
				info_list.append(block_index_list)
				block_index_list = []
			block_index_list.append(i)
		if len(block_index_list) > 0:
			info_list.append(block_index_list)
		return info_list
	
	def load_state_dict(self, state_dict, **kwargs):
		super(ResNetsHE, self).load_state_dict(state_dict)

class ResNetHE(ResNetsHE):
    def __init__(self, num_blocks = [3,3,3], exp_list = [1,1,1], num_classes=10):
        
        conv1 = ConvLayer(3, 16, kernel_size=3, stride=1)
        #bn1 = nn.BatchNorm2d(16)
        layers = []
        in_features = STAGE_WIDTH_LIST[0]
        for idx, n in enumerate(num_blocks):

             planes = STAGE_WIDTH_LIST[idx]
             if idx !=0:
               _planes = STAGE_WIDTH_LIST[idx-1]
             else:
               _planes = STAGE_WIDTH_LIST[0] 
             _planes *= exp_list[idx]

             layers.append(_make_layer(planes, num_blocks = n, stride=STRIDE_LIST[idx],
              _planes =  _planes, exp = exp_list[idx]))
             in_features = planes

        linear = LinearLayer(in_features, num_classes)

        super(ResNetHE, self).__init__(
           input_stem = nn.ModuleList([conv1]),
           blocks = nn.ModuleList(layers),
           classifier = linear)

    def config(self):
       return super(ResNetHE, self).config

    def build_from_config(self,config):
       return super(ResNetHE, self).build_from_config(config)

'''
class ResNetHE(ResNetsHE):
    def __init__(self, num_blocks = [3,3,3], exp_list = [1,1,1], num_classes=10):
        
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(16)
        layers = []
        in_features = STAGE_WIDTH_LIST[0]
        for idx, n in enumerate(num_blocks):

             planes = STAGE_WIDTH_LIST[idx]
             if idx !=0:
               _planes = STAGE_WIDTH_LIST[idx-1]
             else:
               _planes = STAGE_WIDTH_LIST[0] 
             _planes *= exp_list[idx]

             layers.append(_make_layer(planes, num_blocks = n, stride=STRIDE_LIST[idx],
              _planes =  _planes, exp = exp_list[idx]))
             in_features = planes

        linear = nn.Linear(in_features, num_classes)

        super(ResNetHE, self).__init__(
           input_stem = nn.ModuleList([conv1,bn1]),
           blocks = nn.ModuleList(layers),
           classifier = linear)

    def config(self):
       return super(ResNetHE, self).config

    def build_from_config(self,config):
       return super(ResNetHE, self).build_from_config(config)

############################
class ResNetsHE(MyNetwork):

	BASE_DEPTH_LIST = [2, 2, 4, 2]
	STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

	def __init__(self, input_stem, blocks, classifier):
		super(ResNetsHE, self).__init__()

		self.input_stem = nn.ModuleList(input_stem)
		self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
		self.blocks = nn.ModuleList(blocks)
		self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
		self.classifier = classifier

	def forward(self, x):
		for layer in self.input_stem:
			x = layer(x)
		x = self.max_pooling(x)
		for block in self.blocks:
			x = block(x)
		x = self.global_avg_pool(x)
		x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = ''
		for layer in self.input_stem:
			_str += layer.module_str + '\n'
		_str += 'max_pooling(ks=3, stride=2)\n'
		for block in self.blocks:
			_str += block.module_str + '\n'
		_str += self.global_avg_pool.__repr__() + '\n'
		_str += self.classifier.module_str
		return _str

	@property
	def config(self):
		return {
			'name': ResNetsHE.__name__,
			'bn': self.get_bn_param(),
			'input_stem': [
				layer.config for layer in self.input_stem
			],
			'blocks': [
				block.config for block in self.blocks
			],
			'classifier': self.classifier.config,
		}

	@staticmethod
	def build_from_config(config):
		classifier = set_layer_from_config(config['classifier'])

		input_stem = []
		for layer_config in config['input_stem']:
			input_stem.append(set_layer_from_config(layer_config))
		blocks = []
		for block_config in config['blocks']:
			blocks.append(set_layer_from_config(block_config))

		net = ResNetsHE(input_stem, blocks, classifier)
		if 'bn' in config:
			net.set_bn_param(**config['bn'])
		else:
			net.set_bn_param(momentum=0.1, eps=1e-5)

		return net

	def zero_last_gamma(self):
		for m in self.modules():
			if isinstance(m, ResNetBottleneckBlock) and isinstance(m.downsample, IdentityLayer):
				m.conv3.bn.weight.data.zero_()

	@property
	def grouped_block_index(self):
		info_list = []
		block_index_list = []
		for i, block in enumerate(self.blocks):
			if not isinstance(block.downsample, IdentityLayer) and len(block_index_list) > 0:
				info_list.append(block_index_list)
				block_index_list = []
			block_index_list.append(i)
		if len(block_index_list) > 0:
			info_list.append(block_index_list)
		return info_list
	
	def load_state_dict(self, state_dict, **kwargs):
		super(ResNetsHE, self).load_state_dict(state_dict)


class ResNet50HE(ResNetsHE):

	def __init__(self, n_classes=1000, width_mult=1.0, bn_param=(0.1, 1e-5), dropout_rate=0,
	             expand_ratio=None, depth_param=None):

		expand_ratio = 0.25 if expand_ratio is None else expand_ratio

		input_channel = make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
		stage_width_list = ResNetsHE.STAGE_WIDTH_LIST.copy()
		for i, width in enumerate(stage_width_list):
			stage_width_list[i] = make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

		depth_list = [3, 4, 6, 3]
		if depth_param is not None:
			for i, depth in enumerate(ResNetsHE.BASE_DEPTH_LIST):
				depth_list[i] = depth + depth_param

		stride_list = [1, 2, 2, 2]

		# build input stem
		input_stem = [ConvLayer(
			3, input_channel, kernel_size=7, stride=2, use_bn=True, act_func='square', ops_order='weight_bn_act',
		)]

		# blocks
		blocks = []
		for d, width, s in zip(depth_list, stage_width_list, stride_list):
			for i in range(d):
				stride = s if i == 0 else 1
				bottleneck_block = ResNetBottleneckBlock(
					input_channel, width, kernel_size=3, stride=stride, expand_ratio=expand_ratio,
					act_func='square', downsample_mode='conv',
				)
				blocks.append(bottleneck_block)
				input_channel = width
		# classifier
		classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)
		
		super(ResNet50HE, self).__init__(input_stem, blocks, classifier)

		# set bn param
		self.set_bn_param(*bn_param)


class ResNet50D(ResNetsHE):

	def __init__(self, n_classes=1000, width_mult=1.0, bn_param=(0.1, 1e-5), dropout_rate=0,
	             expand_ratio=None, depth_param=None):

		expand_ratio = 0.25 if expand_ratio is None else expand_ratio

		input_channel = make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
		mid_input_channel = make_divisible(input_channel // 2, MyNetwork.CHANNEL_DIVISIBLE)
		stage_width_list = ResNetsHE.STAGE_WIDTH_LIST.copy()
		for i, width in enumerate(stage_width_list):
			stage_width_list[i] = make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE)

		depth_list = [3, 4, 6, 3]
		if depth_param is not None:
			for i, depth in enumerate(ResNetsHE.BASE_DEPTH_LIST):
				depth_list[i] = depth + depth_param

		stride_list = [1, 2, 2, 2]

		# build input stem
		input_stem = [
			ConvLayer(3, mid_input_channel, 3, stride=2, use_bn=True, act_func='square'),
			ResidualBlock(
				ConvLayer(mid_input_channel, mid_input_channel, 3, stride=1, use_bn=True, act_func='square'),
				IdentityLayer(mid_input_channel, mid_input_channel)
			),
			ConvLayer(mid_input_channel, input_channel, 3, stride=1, use_bn=True, act_func='square')
		]

		# blocks
		blocks = []
		for d, width, s in zip(depth_list, stage_width_list, stride_list):
			for i in range(d):
				stride = s if i == 0 else 1
				bottleneck_block = ResNetBottleneckBlock(
					input_channel, width, kernel_size=3, stride=stride, expand_ratio=expand_ratio,
					act_func='square', downsample_mode='avgpool_conv',
				)
				blocks.append(bottleneck_block)
				input_channel = width
		# classifier
		classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)

		super(ResNet50D, self).__init__(input_stem, blocks, classifier)

		# set bn param
		self.set_bn_param(*bn_param)
'''