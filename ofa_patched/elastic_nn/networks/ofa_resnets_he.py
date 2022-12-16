import random

from ofa.elastic_nn.modules.dynamic_layers import DynamicConvLayer, DynamicLinearLayer
from ofa.elastic_nn.modules.dynamic_layers import DynamicResNetBottleneckBlock
from ofa.layers import IdentityLayer, ResidualBlock
from ofa.imagenet_codebase.networks.resnets_he import ResNetsHE,ResNetHE
from ofa.utils import make_divisible, val2list, MyNetwork

__all__ = ['OFAResNetsHE']

BASE_DEPTH_LIST = [3,3,3] 
STAGE_WIDTH_LIST = [16,32,64]
STRIDE_LIST = [1,2,2]


class OFAResNetsHE():

  #expand_ratio_list = [1]
  #depth_list = [3,4,5,6,7,8]

	def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0,
	           depth_list=2, expand_ratio_list=0.25, width_mult_list=1.0):

		self.num_blocks = 3
		self.n_classes = n_classes
		self.depth_list = val2list(depth_list)
		self.expand_ratio_list = val2list(expand_ratio_list)
		self.width_mult_list = val2list(width_mult_list)
		# sort
		self.depth_list.sort()
		self.expand_ratio_list.sort()
		self.width_mult_list.sort()
		self.stride_list = STRIDE_LIST.copy()
		self.stage_width_list = STAGE_WIDTH_LIST.copy()

	@property
	def ks_list(self):
		return [3]

	@staticmethod
	def name():
		return 'OFAResNetsHE'

	@property
	def module_str(self):
		raise ValueError('do not support this function')

	@property
	def config(self):
		raise ValueError('do not support this function')

	@staticmethod
	def build_from_config(config):
		raise ValueError('do not support this function')

	def load_state_dict(self, state_dict, **kwargs):
		raise ValueError('do not support this function')

	""" set, sample and get active sub-networks """

	def set_max_net(self):
		depth = max(self.depth_list)
		exp = e=max(self.expand_ratio_list)
		d = []
		e = []
		for stage_id in range(len(BASE_DEPTH_LIST)):
		  d.append(depth)
		  e.append(exp)

		self.set_active_subnet(d=d, e=e, w=len(self.width_mult_list) - 1)

	def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
		self.active_subnet = ResNetHE(d,e,self.n_classes)

	def sample_active_subnet(self):
		# sample expand ratio
		expand_setting = []
		for stage_id in range(len(BASE_DEPTH_LIST)):
			expand_setting.append(random.choice(self.expand_ratio_list))

		# sample depth
		depth_setting = [] #[random.choice([max(self.depth_list), min(self.depth_list)])]
		for stage_id in range(len(BASE_DEPTH_LIST)):
			depth_setting.append(random.choice(self.depth_list))
    
    # sample width_mult
		width_mult_setting = [random.choice(self.width_mult_list)]


		arch_config = {
			'd': depth_setting,
			'e': expand_setting,
			'w': width_mult_setting
		}
		self.set_active_subnet(**arch_config)
		return arch_config

	def get_active_subnet(self, preserve_weight=True):
		return self.active_subnet

	def get_active_net_config(self):
		return self.active_subnet.config()


	""" Width Related Methods """

	def re_organize_middle_weights(self, expand_ratio_stage=0):
		for block in self.blocks:
			block.re_organize_middle_weights(expand_ratio_stage)