from search_space.ofa import OFASearchSpace
from evaluator import OFAEvaluator,get_net_info
from ofa.model_zoo import ofa_net

ofa_network = ofa_net(10,'ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
    
# Manually set the sub-network

ofa_network.set_active_subnet(ks=7, e=6, d=3)
m1 = ofa_network.get_active_subnet(preserve_weight=True)

ofa_network.set_active_subnet(ks=7, e=6, d=4)
m2 = ofa_network.get_active_subnet(preserve_weight=True)

'''
lr = 40
ur = 40
n_doe = 1
ss_small = OFASearchSpace('mobilenetv3',lr,ur,5)
#ss_big = OFASearchSpace('mobilenetv3',lr,ur,5)
eval = OFAEvaluator(n_classes=10)
m1_config = ss_small.initialize(n_doe)[0]
#m2_config = ss_big.initialize(n_doe)[0]
print(m1_config)
#print(m2_config)


# encode m1,m2
m1_encode = ss_small.encode(m1_config)
#m2_encode = ss_big.encode(m2_config)
#print(m1_encode)

# decode

m1_config = ss_small.decode(m1_encode)
#m2_config = ss_big.decode(m2_encode)
#print(m1_config)
#print(m2_config)



#sample subnets from OFA

m1,_ = eval.sample(m1_config)
#m2,_ = eval.sample(m2_config)
'''

info1 = get_net_info(m1,(40,40))
info2 = get_net_info(m2,(40,40))






