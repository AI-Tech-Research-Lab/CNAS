from search_space.ofa import OFASearchSpace
from evaluator import OFAEvaluator,get_net_info
from ofa.elastic_nn.modules.dynamic_layers import ExitBlock
import torch
from ofa.model_zoo import ofa_net
import numpy as np
from utils import get_net_info

#Compute MACS of the exit gate (< MACs of the whole net)

n_classes = 10
input_shape = (3,40,40)
ofa_ee = ofa_net(n_classes,'ofa_eembv3_d234_e346_k357_w1.0', pretrained=False)
ofa = ofa_net(n_classes,'ofa_mbv3_d234_e346_k357_w1.0', pretrained=False)
d = [2,2,2,2,2]
ofa.set_active_subnet(ks=7, e=6, d=d)
m1 = ofa.get_active_subnet(preserve_weight=True)
ofa_ee.set_active_subnet(ks=7, e=6, d=d)
m2 = ofa_ee.get_active_subnet(preserve_weight=True)
m2.eval()
m2.threshold = [1,1,0,1]
input = torch.randn(1, 3, 40, 40)
x,counts = m2(input)
print(x.shape[0])
print(counts)

#These two conditions are equals to force to classify all samples with exit gate:
#m2.eval()
#m2.threshold = 0 #equals to classify all sample with exit gate
#Check that computation skips the final layers..

#info1 = get_net_info(m1,input_shape)
info2 = get_net_info(m2,input_shape)

#print(info1)
print(info2)



'''
depth = [2, 3, 4]
nb = 5
d = np.random.choice(depth, nb, replace=True).tolist()
print("CONFIG")
print(d)
ofa.set_active_subnet(ks=7, e=6, d=d)
m= ofa.get_active_subnet(preserve_weight=True)
input = torch.randn(96, 3, 40, 40)
m.threshold = 0.0005
m.train()
x = m(input)
'''

'''
final_expand_width = [960]
feature_dim = [112]
last_channel = [1280]
dropout_rate = 0.1
exit = ExitBlock(n_classes,final_expand_width,feature_dim,last_channel,dropout_rate)
input = torch.randn(20, 112, 40, 40)
x,conf=exit(input)
print(conf)
'''

'''
lr = 40
ur = 40
n_doe = 1
ss = OFASearchSpace('mobilenetv3',lr,ur)
eval = OFAEvaluator(n_classes=1000,
model_path='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0',
pretrained = True)
m1_config = ss.sample(n_samples = n_doe, d = [2,3])[0]
m2_config = ss.increase_config(m1_config)

# encode m1,m2
m1_encode = ss.encode(m1_config)
m2_encode = ss.encode(m2_config)

# decode

m1_config = ss.decode(m1_encode)
m2_config = ss.decode(m2_encode)
print(m1_config)
print(m2_config)

#m1,_ = eval.sample(m1_config)
#m2,_ = eval.sample(m2_config)


#info1 = get_net_info(m1,(40,40))
#info2 = get_net_info(m2,(40,40))



#sample subnets from OFA through config

m1,_ = eval.sample(m1_config)
m2,_ = eval.sample(m2_config)
'''


'''
ofa.set_active_subnet(ks=7, e=6, d=[3,2,4,2,3])

for stage_id, block_idx in enumerate(ofa.block_group_info):
            print("STAGE ID")
            print(stage_id)
            depth = ofa.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            ofa.blocks[-1].mobile_inverted_conv.active_out_channel
'''
#m1 = ofa_network.get_active_subnet(preserve_weight=True)
#input = torch.randn(1, 3, 40, 40)
#x = m1(input)

#ofa_network.set_active_subnet(ks=7, e=6, d=4)
#m2 = ofa_network.get_active_subnet(preserve_weight=True)









