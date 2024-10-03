from NasSearchSpace.ofa.search_space import getOFASearchSpace
from NasSearchSpace.ofa.evaluator import OFAEvaluator
from utils import get_net_info

ss=getOFASearchSpace('resnet50', 128, 224, 4)
ofa = OFAEvaluator(n_classes=10,model_path='NasSearchSpace/ofa/supernets/ofa_supernet_resnet50',pretrained=True)
configs=ss.initialize()
print(configs)
min=ofa.sample(configs[0])[0]
max=ofa.sample(configs[1])[0]
#count params
res=128
input_shape = (3, res, res)
info = get_net_info(min, input_shape=input_shape, print_info=True)['params']
print(info)
res=224
input_shape = (3, res, res)
info = get_net_info(max, input_shape=input_shape, print_info=True)['params']
print(info)