from evaluator import OFAEvaluator
from utils import get_net_info, get_adapt_net_info
import json

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









