from NasSearchSpace.ofa.search_space import getOFASearchSpace
from ofa_evaluator import OFAEvaluator

ss=getOFASearchSpace('resnet50', 128, 224, 4)
ofa = OFAEvaluator(n_classes=10,model_path='NasSearchSpace/ofa/supernets/ofa_supernet_resnet50',pretrained=True)
configs=ss.sample(10)
for c in configs:
    subnet=ofa.sample(c)
#print(configs)