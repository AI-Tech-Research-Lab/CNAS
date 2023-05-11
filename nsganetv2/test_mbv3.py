from evaluator import OFAEvaluator
from ofa.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
from utils import get_net_info, get_adapt_net_info
import json

'''
subnet = './../benchmarks/EDANAS/cifar10-mbv3-adaptive/final/net-trade-off_4@3_NAS/net.subnet'
supernet = './ofa_nets/ofa_mbv3_d234_e346_k357_w1.0'
pretrained = False
config = json.load(open(subnet))
n_classes = 10
ofa = OFAEvaluator(n_classes=1000,
model_path=supernet,
pretrained = pretrained)
input_shape = (3,config['r'],config['r'])
subnet, _ = ofa.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})
info = get_net_info(subnet,input_shape)
'''

subnet = './../benchmarks/EDANAS/cifar10-mbv3-adaptive/final/net-trade-off_4@3_NAS/net.subnet'
pretrained = False
config = json.load(open(subnet))
n_classes = 10
ofa = OFAMobileNetV3(n_classes=10)
input_shape = (3,config['r'],config['r'])
ofa.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'])
subnet = ofa.get_active_subnet(preserve_weight=True)
info = get_net_info(subnet,input_shape)








