from search_space.ofa import OFASearchSpace
from evaluator import OFAEvaluator

lr = 40
ur = 80
n_doe = 1
ss_small = OFASearchSpace('mobilenetv3',lr,ur,3)
ss_big = OFASearchSpace('mobilenetv3',lr,ur,5)
eval = OFAEvaluator(n_classes=10)
m1_config = ss_small.sample(n_doe)[0]
m2_config = ss_big.sample(n_doe)[0]
#print(m1_config)
#print(m2_config)


# encode m1,m2
m1_encode = ss_small.encode(m1_config)
m2_encode = ss_big.encode(m2_config)
#print(m1_encode)

# decode

m1_config = ss_small.decode(m1_encode)
m2_config = ss_big.decode(m2_encode)
print(m1_config)
print(m2_config)

#build from config

m1 = eval.sample(m1_config)
m2 = eval.sample(m2_config)





