# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch
import torch.nn as nn
import numpy as np

# from layers import *
from ofa.layers import set_layer_from_config, MBInvertedConvLayer, ConvLayer, IdentityLayer, LinearLayer
from ofa.imagenet_codebase.utils import MyNetwork, make_divisible
from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock


class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier):

        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                mb_conv = MBInvertedConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        feature_dim = feature_dim * 6
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg

from ofa.elastic_nn.modules.dynamic_layers import ExitBlock

class EEMobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier, 
    n_classes, dropout_rate, d_list, t_list):

        super(EEMobileNetV3, self).__init__()

        base_stage_width = [24, 40, 80, 112]

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.t_list = t_list
        self.exit_idxs = []
        self.exit_list = []
        n_blocks = len(base_stage_width)+1
        idx = 1
        for i in range(0,n_blocks-1,1):
            idx += d_list[i]
            if (t_list[i]!=1):
                feature_dim = [base_stage_width[i]]
                final_expand_width = [feature_dim[0] * 6] #960
                last_channel = [feature_dim[0] * 8] #1280
                self.exit_idxs.append(idx)
                self.exit_list.append(ExitBlock(n_classes,final_expand_width,feature_dim,last_channel,dropout_rate))
        self.n_exit = len(self.exit_list)
        if (self.n_exit!=0):
            self.exit_list = nn.ModuleList(self.exit_list)

    def forward(self, x):
        #NOT WORKING
        x = self.first_conv(x)

        preds = [] 
        idxs = []

        if(self.training): #training 
            i = 0
            for idx,block in enumerate(self.blocks):
                if(self.n_exit!=0):
                    if (idx==self.exit_idxs[i]): #exit block
                        pred, _ = self.exit_list[i](x)
                        preds.append(pred)
                        if(i<(self.n_exit-1)):
                            i+=1
                x = block(x)
            x = self.final_expand_layer(x)
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
            x = self.feature_mix_layer(x)
            x = torch.squeeze(x)
            x = self.classifier(x)
            preds.append(x)
            return preds
        else:
            i = 0
            counts = np.zeros(self.n_exit+1)
            for idx,block in enumerate(self.blocks):
                if(self.n_exit!=0):
                    if (idx==self.exit_idxs[i]): #exit block
                            exit_block = self.exit_list[i]
                            exit_block.to(torch.device('cuda')) #param tensors to GPU
                            pred, conf = exit_block(x)
                            conf = torch.squeeze(conf)
                            mask = conf >= self.t_list[i]
                            mask = mask.cpu() #gpu>cpu memory
                            p = np.where(np.array(mask)==False)[0] #idxs of non EE predictions
                            counts[i] = torch.sum(mask).item()
                            x = x[mask==0,:,:,:]
                            pred = pred[mask==1,:]
                            del mask 
                            del conf
                            preds.append(pred)
                            idxs.append(p)
                            # FIX bug that for one sample x.shape = (0,1,,,,) when empty
                            #if (x.shape[0]==0): # no more samples 
                            #    x = torch.squeeze(x)
                            #    break
                            if(i<(self.n_exit-1)):
                                i+=1
                x = block(x)

            counts[-1] = x.shape[0] #n samples classified normally by the last exit
            x = self.final_expand_layer(x)
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
            x = self.feature_mix_layer(x)
            x = torch.squeeze(x)
            x = self.classifier(x)

            if(self.n_exit!=0):

                preds.append(x)

                #mix predictions of all exits
                tensors = []
                for i in range(len(preds)-1,0,-1): #mix predictions of all exits
                    tensors = list(torch.unbind(preds[i-1],axis=0))
                    iter = idxs[i-1]
                    pred = preds[i]
                    for j,idx in enumerate(iter):
                        tensors.insert(idx,pred[j])
                    preds[i-1] = torch.stack(tensors,axis=0)
                    del preds[i]
                
                x = preds[0]
                del preds[0]
                del pred
            
            return x,counts
        

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = EEMobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                mb_conv = MBInvertedConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        feature_dim = feature_dim * 6
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg

class MobileNetV3Large(MobileNetV3):

    def __init__(self, n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0.2,
                 ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        input_channel = 16
        last_channel = 1280

        input_channel = make_divisible(input_channel * width_mult, 8)
        last_channel = make_divisible(last_channel * width_mult, 8) if width_mult > 1.0 else last_channel

        cfg = {
            #    k,     exp,    c,      se,         nl,         s,      e,
            '0': [
                [3,     16,     16,     False,      'relu',     1,      1],
            ],
            '1': [
                [3,     64,     24,     False,      'relu',     2,      None],  # 4
                [3,     72,     24,     False,      'relu',     1,      None],  # 3
            ],
            '2': [
                [5,     72,     40,     True,       'relu',     2,      None],  # 3
                [5,     120,    40,     True,       'relu',     1,      None],  # 3
                [5,     120,    40,     True,       'relu',     1,      None],  # 3
            ],
            '3': [
                [3,     240,    80,     False,      'h_swish',  2,      None],  # 6
                [3,     200,    80,     False,      'h_swish',  1,      None],  # 2.5
                [3,     184,    80,     False,      'h_swish',  1,      None],  # 2.3
                [3,     184,    80,     False,      'h_swish',  1,      None],  # 2.3
            ],
            '4': [
                [3,     480,    112,    True,       'h_swish',  1,      None],  # 6
                [3,     672,    112,    True,       'h_swish',  1,      None],  # 6
            ],
            '5': [
                [5,     672,    160,    True,       'h_swish',  2,      None],  # 6
                [5,     960,    160,    True,       'h_swish',  1,      None],  # 6
                [5,     960,    160,    True,       'h_swish',  1,      None],  # 6
            ]
        }

        cfg = self.adjust_cfg(cfg, ks, expand_ratio, depth_param, stage_width_list)
        # width multiplier on mobile setting, change `exp: 1` and `c: 2`
        for stage_id, block_config_list in cfg.items():
            for block_config in block_config_list:
                if block_config[1] is not None:
                    block_config[1] = make_divisible(block_config[1] * width_mult, 8)
                block_config[2] = make_divisible(block_config[2] * width_mult, 8)

        first_conv, blocks, final_expand_layer, feature_mix_layer, classifier = self.build_net_via_cfg(
            cfg, input_channel, last_channel, n_classes, dropout_rate
        )
        super(MobileNetV3Large, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
