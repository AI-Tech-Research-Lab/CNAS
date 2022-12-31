from search_space.ofa import OFASearchSpace
from evaluator import OFAEvaluator,get_net_info
from ofa.elastic_nn.modules.dynamic_layers import ExitBlock
import torch
from ofa.model_zoo import ofa_net

n_classes = 10
#ofa = ofa_net(n_classes,'ofa_mbv3_d234_e346_k357_w1.0', pretrained=False)
final_expand_width = [960]
feature_dim = [160]
last_channel = [1280]
dropout_rate = 0.1
exit = ExitBlock(n_classes,final_expand_width,feature_dim,last_channel,dropout_rate)
input = torch.randn(20, 160, 40, 40)
x,conf=exit(input)


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









