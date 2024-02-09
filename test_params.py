from ofa_evaluator import OFAEvaluator
from utils import get_net_info
from search_space import OFASearchSpace

ss = OFASearchSpace('mobilenetv3',128,224,4)
net_small, net_big = ss.initialize(0)

supernet_path = 'supernets/ofa_mbv3_d234_e346_k357_w1.0'
ofa = OFAEvaluator(n_classes=10,
model_path = supernet_path,
pretrained = True)

r=128
input_shape = (3,net_big['r'],net_big['r'])

#print("INPUT SHAPE:", input_shape)
print("CIFAR10 BIGGEST")
model, _ = ofa.sample(net_big)
print("Info")
print(get_net_info(model, input_shape))

x = ss.encode(net_big)
print("ENCODING: ",x)
print("DECODING: ",ss.decode(x))

'''
print("CIFAR10 BIGGEST")
model, _ = ofa.sample(net_big)
print("Info")
print(get_net_info(model, input_shape))

supernet_path = 'supernets/ofa_mbv3_d234_e346_k357_w1.0'
ofa = OFAEvaluator(n_classes=100,
model_path = supernet_path,
pretrained = True)

print("CIFAR100 SMALLEST")
model, _ = ofa.sample(net_small)
print("Info")
print(get_net_info(model, input_shape))
print("CIFAR100 BIGGEST")
model, _ = ofa.sample(net_big)
print("Info")
print(get_net_info(model, input_shape))
'''
