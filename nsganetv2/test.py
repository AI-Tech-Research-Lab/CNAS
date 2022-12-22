from search_space.ofa import OFASearchSpace


lr = 40
ur = 80
n_doe = 1
ss_small = OFASearchSpace('mobilenetv3_small',lr,ur)
ss_big = OFASearchSpace('mobilenetv3_big',lr,ur)
m1_config = ss_small.initialize(n_doe)[0]
m2_config = ss_big.initialize(n_doe)[1]
print(m1_config)
print(m2_config)


# encode m1,m2
m1_encode = ss_small.encode(m1_config[0])
m2_encode = ss_big.encode(m2_config[0])
#print(m1_encode)

# decode

m1_config = ss_small.decode(m1_encode)
m2_config = ss_big.decode(m2_encode)
#print(m1_config)
#print(m2_config)





