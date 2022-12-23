from search_space.ofa import OFASearchSpace
from evaluator import OFAEvaluator,get_net_info

#CONFIG
n_classes = 10
supernet = '../data/ofa_mbv3_d234_e346_k357_w1.0'
pretrained = True
n_epochs = 5
trn_batch_size = 128
vld_batch_size = 200
img_size = 40
data = '../data/cifar10'
dataset = 'cifar10'


ss = OFASearchSpace(supernet = 'mobilenetv3', lr = img_size, ur = img_size)

config = ss.initialize(n_doe = 1)[0]
evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet, pretrained = pretrained)
subnet, _ = evaluator.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})
resolution = config['r']

OFAEvaluator.eval(
        subnet, data_path=data, dataset=dataset, n_epochs=n_epochs,
        resolution=resolution, trn_batch_size=trn_batch_size, vld_batch_size=vld_batch_size)
