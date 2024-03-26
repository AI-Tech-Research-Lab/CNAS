import os
import json
import torch
import argparse
import numpy as np
import math

from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAResNets 
from ofa.imagenet_classification.run_manager import RunManager
from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
#from ofa.utils import download_url

import warnings
warnings.simplefilter("ignore")

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1


def parse_string_list(string):
    if isinstance(string, str):
        # convert '[5 5 5 7 7 7 3 3 7 7 7 3 3]' to [5, 5, 5, 7, 7, 7, 3, 3, 7, 7, 7, 3, 3]
        return list(map(int, string[1:-1].split()))
    else:
        return string
        
def pad_none(x, depth, max_depth):
    new_x, counter = [], 0
    for d in depth:
        for _ in range(d):
            new_x.append(x[counter])
            counter += 1
        if d < max_depth:
            new_x += [None] * (max_depth - d)
    return new_x

def validate_config(config, max_depth=4):
    kernel_size, exp_ratio, depth = config['ks'], config['e'], config['d']

    if isinstance(kernel_size, str): kernel_size = parse_string_list(kernel_size)
    if isinstance(exp_ratio, str): exp_ratio = parse_string_list(exp_ratio)
    if isinstance(depth, str): depth = parse_string_list(depth)

    assert (isinstance(kernel_size, list) or isinstance(kernel_size, int))
    assert (isinstance(exp_ratio, list) or isinstance(exp_ratio, int))
    assert isinstance(depth, list)

    if len(kernel_size) < len(depth) * max_depth:
        kernel_size = pad_none(kernel_size, depth, max_depth)
    if len(exp_ratio) < len(depth) * max_depth:
        exp_ratio = pad_none(exp_ratio, depth, max_depth)

    # return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'w': config['w']}
    return {'ks': kernel_size, 'e': exp_ratio, 'd': depth}

class OFAEvaluator:
    """ based on OnceForAll supernet taken from https://github.com/mit-han-lab/once-for-all """
    def __init__(self,
                 n_classes=1000,
                 model_path='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0',
                 pretrained = False,
                 kernel_size=None, exp_ratio=None, depth=None, threshold = None):
                 
        # default configurations (MBV3)
        self.kernel_size = [3, 5, 7] if kernel_size is None else kernel_size  # depth-wise conv kernel size
        self.exp_ratio = [3, 4, 6] if exp_ratio is None else exp_ratio  # expansion rate
        self.depth = [2, 3, 4] if depth is None else depth  # number of MB block repetition 

        if 'w1.0' in model_path or 'resnet50' in model_path:
            self.width_mult = 1.0
        elif 'w1.2' in model_path:
            self.width_mult = 1.2
        else:
            raise ValueError
        
        print("MODEL PATH: ", model_path)

        if ('ofa_mbv3' in model_path):
            self.engine = OFAMobileNetV3(
                n_classes=n_classes,
                dropout_rate=0, width_mult=self.width_mult, ks_list=self.kernel_size,
                expand_ratio_list=self.exp_ratio, depth_list=self.depth)
            
        elif 'resnet50' in model_path:
            # default configurations
            #ks is 3 by default for resnet
            self.kernel_size = [3] if kernel_size is None else kernel_size  # depth-wise conv kernel size
            self.exp_ratio = [0.2,0.25,0.35] if exp_ratio is None else exp_ratio  # expansion rate
            self.depth = [0,1,2] if depth is None else depth  # number of MB block repetition

            self.engine = OFAResNets(n_classes = n_classes,
              bn_param=(0.1, 1e-5),
              dropout_rate = 0,
              depth_list = self.depth,
              expand_ratio_list = self.exp_ratio,
              width_mult_list = self.width_mult
              ) 
        else:

          raise NotImplementedError 
            
        if(pretrained):

            init = torch.load(model_path, map_location='cpu')['state_dict']

            ##FIX size mismatch error##### 
            init['classifier.linear.weight'] = init['classifier.linear.weight'][:n_classes]
            init['classifier.linear.bias'] = init['classifier.linear.bias'][:n_classes]
            ##############################

            self.engine.load_state_dict(init)  
            
    def sample(self, config=None):
        """ randomly sample a sub-network """
        if config is not None:
              self.engine.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'])
        else:
            config = self.engine.sample_active_subnet()

        subnet = self.engine.get_active_subnet(preserve_weight=True)
        return subnet, config

    @staticmethod
    def save_net_config(path, net, config_name='net.config'):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(path, config_name)
        json.dump(net.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

    @staticmethod
    def save_net(path, net, model_name):
        """ dump net weight as checkpoint """
        if isinstance(net, torch.nn.DataParallel):
            checkpoint = {'state_dict': net.module.state_dict()}
        else:
            checkpoint = {'state_dict': net.state_dict()}
        model_path = os.path.join(path, model_name)
        torch.save(checkpoint, model_path)
        print('Network model dump to %s' % model_path)

    @staticmethod   
    def eval(subnet, data_path, dataset='imagenet', n_epochs=0, resolution=(224,224), trn_batch_size=128, vld_batch_size=250,
             num_workers=4, valid_size=None, is_test=True, log_dir='.tmp/eval', measure_latency=None, no_logs=False,
             reset_running_statistics=True, pmax = 2, fmax = 100, amax = 5, wp = 1, wf = 1/40, wa = 1, penalty = 10**10):

        lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        info = get_net_info( ## compute net info (params, etc..)
              subnet, (3, resolution, resolution), measure_latency=measure_latency,
              print_info=False, clean=True, lut=lut, pmax = pmax, fmax = fmax, amax = amax, wp = wp, wf = wf, wa = wa, penalty = penalty)

        run_config = get_run_config(  ## get data provider class
            dataset=dataset, data_path=data_path, image_size=resolution, n_epochs=n_epochs,
            train_batch_size=trn_batch_size, test_batch_size=vld_batch_size,
            n_worker=num_workers, valid_size=valid_size)

        # set the image size. You can set any image size from 192 to 256 here
        run_config.data_provider.assign_active_img_size(resolution)

        if n_epochs > 0:
            # for datasets other than the one supernet was trained on (ImageNet)
            # a few epochs of training need to be applied
            ''' these lines are commented to avoid AttributeError: 'MobileNetV3' object has no attribute 'reset_classifier'
            subnet.reset_classifier(
                last_channel=subnet.classifier.in_features,
                n_classes=run_config.data_provider.n_classes, dropout_rate=cfgs.drop_rate)
            '''

        run_manager = RunManager(log_dir, subnet, run_config, init=False)
        
        if reset_running_statistics:
            # run_manager.reset_running_statistics(net=subnet, batch_size=vld_batch_size)
            run_manager.reset_running_statistics(net=subnet)

        if n_epochs > 0:
            cfgs.subnet = subnet
            subnet = run_manager.train(cfgs)

        loss, top1, top5 = run_manager.validate(net=subnet, is_test=is_test, no_logs=no_logs)

        info['loss'], info['top1'], info['top5'] = loss, top1, top5

        save_path = os.path.join(log_dir, 'net.stats') if cfgs.save is None else cfgs.save
        if cfgs.save_config:
            OFAEvaluator.save_net_config(log_dir, subnet, "net.config")
            OFAEvaluator.save_net(log_dir, subnet, "net.init")
        with open(save_path, 'w') as handle:
            json.dump(info, handle)
        
        print(info)





























































