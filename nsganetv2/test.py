from search_space.ofa import OFASearchSpace
from evaluator import OFAEvaluator,get_net_info

lr = 40
ur = 40
n_doe = 1
ss = OFASearchSpace('mobilenetv3',lr,ur)
eval = OFAEvaluator(n_classes=10)
m1_config = ss.initialize(n_doe)[0]
print(m1_config)


# encode m1,m2
m1_encode = ss.encode(m1_config)
print(m1_encode)


# decode

m1_config = ss.decode(m1_encode)
print(m1_config)


m1,_ = eval.sample(m1_config)

'''
info1 = get_net_info(m1,(40,40))

'''

'''
#sample subnets from OFA through config

m1,_ = eval.sample(m1_config)
m2,_ = eval.sample(m2_config)
'''

'''
from ofa.model_zoo import ofa_net

ofa_network = ofa_net(10,'ofa_mbv3_d234_e346_k357_w1.0', pretrained=False)

ofa_network.set_active_subnet(ks=7, e=6, d=3)
m1 = ofa_network.get_active_subnet(preserve_weight=True)

ofa_network.set_active_subnet(ks=7, e=6, d=4)
m2 = ofa_network.get_active_subnet(preserve_weight=True)
'''








