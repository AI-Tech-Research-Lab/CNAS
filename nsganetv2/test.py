from search_space.ofa import OFASearchSpace
from evaluator import OFAEvaluator,get_net_info

lr = 40
ur = 40
n_doe = 1
ss = OFASearchSpace('mobilenetv3',lr,ur)
eval = OFAEvaluator(n_classes=1000,
model_path='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0',
pretrained = True)
m1_config = ss.initialize(n_doe)[0]

m2_config = m1_config
new_d = []
for i in range(len(m1_config['d'])):
    item = m1_config['d'][i] + 1
    new_d.append(item)
m2_config['d'] = new_d
print(m1_config)
print(m2_config)


# encode m1,m2
m1_encode = ss.encode(m1_config)
m2_encode = ss.encode(m2_config)

# decode

m1_config = ss.decode(m1_encode)
m2_config = ss.decode(m2_encode)
print(m1_config)
print(m2_config)

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








