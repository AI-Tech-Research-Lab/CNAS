import csv
import os
import copy
import json
#import yaml
import numpy as np
from collections import OrderedDict
from torchprofile import profile_macs

import torch
from torch.nn import Conv2d,ReLU,Linear,Sequential,Flatten,BatchNorm2d,AvgPool2d,MaxPool2d
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover

from ofa.utils.pytorch_utils import count_parameters
from NasSearchSpace.ofa.evaluator import OFAEvaluator  
from NasSearchSpace.nasbench201.search_space import NASBench201SearchSpace
from NasSearchSpace.nasbench201.model import NASBenchNet

DEFAULT_CFG = {
    'gpus': '0', 'config': None, 'init': None, 'trn_batch_size': 128, 'vld_batch_size': 250, 'num_workers': 4,
    'n_epochs': 0, 'save': None, 'resolution': 224, 'valid_size': 10000, 'test': True, 'latency': None,
    'verbose': False, 'classifier_only': False, 'reset_running_statistics': True,

}

target_layers = {'Conv2D':Conv2d,
                 'Flatten':Flatten,
                 'Dense':Linear,
                 'BatchNormalization':BatchNorm2d,
                 'AveragePooling2D':AvgPool2d,
                 'MaxPooling2D':MaxPool2d
                 }

activations = {}

def hook_fn(m, i, o):
    #if (o.shape != NULL):
    activations[m] = [i,o]#.shape  #m is the layer

def get_all_layers(net):
  layers = {}
  names = {}
  index = 0
  for name, layer in net.named_modules():#net._modules.items():
    #print(name)
    layers[index] = layer
    names[index] = name
    index = index + 1
    
  #If it is a sequential or a block of modules, don't register a hook on it
  # but recursively register hook on all it's module children
  length = len(layers)
  for i in range(length):
    if (i==(length-1)):
      layers[i].register_forward_hook(hook_fn)
    else:
      if ((isinstance(layers[i], nn.Sequential)) or   #sequential
          (names[i+1].startswith(names[i] + "."))):  #container of layers
        continue
      else:
        layers[i].register_forward_hook(hook_fn)

def profile_activation_size(model,input):
    activations.clear()
    get_all_layers(model) #add hooks to model layers
    out = model(input) #computes activation while passing through layers
    
    total = 0
    
    for name, layer in model.named_modules():
      for label, target in target_layers.items():
        if(isinstance(layer,target)):
          #print(name)

          activation_shape = activations[layer][1].shape
          activation_size = 1
          for i in activation_shape:
            activation_size = activation_size * i
          total = total + activation_size
    
    return total


def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau

'''
def bash_command_template_single_exit(**kwargs):

    gpus = kwargs.pop('gpus', DEFAULT_CFG['gpus'])
    cfg = OrderedDict()

    cfg['subnet'] = kwargs['subnet']
    cfg['data'] = kwargs['data']
    cfg['dataset'] = kwargs['dataset']
    cfg['n_classes'] = kwargs['n_classes']
    cfg['supernet'] = kwargs['supernet']
    cfg['pretrained'] = kwargs['pretrained']
    cfg['pmax'] = kwargs['pmax']
    cfg['fmax'] = kwargs['fmax']
    cfg['amax'] = kwargs['amax']
    cfg['wp'] = kwargs['wp']
    cfg['wf'] = kwargs['wf']
    cfg['wa'] = kwargs['wa']
    cfg['penalty'] = kwargs['penalty']
    cfg['config'] = kwargs.pop('config', DEFAULT_CFG['config'])
    cfg['init'] = kwargs.pop('init', DEFAULT_CFG['init'])
    cfg['save'] = kwargs.pop('save', DEFAULT_CFG['save'])
    cfg['trn_batch_size'] = kwargs.pop('trn_batch_size', DEFAULT_CFG['trn_batch_size'])
    cfg['vld_batch_size'] = kwargs.pop('vld_batch_size', DEFAULT_CFG['vld_batch_size'])
    cfg['num_workers'] = kwargs.pop('num_workers', DEFAULT_CFG['num_workers'])
    cfg['n_epochs'] = kwargs.pop('n_epochs', DEFAULT_CFG['n_epochs'])
    cfg['resolution'] = kwargs.pop('resolution', DEFAULT_CFG['resolution'])
    cfg['valid_size'] = kwargs.pop('valid_size', DEFAULT_CFG['valid_size'])
    cfg['test'] = kwargs.pop('test', DEFAULT_CFG['test'])
    cfg['latency'] = kwargs.pop('latency', DEFAULT_CFG['latency'])
    cfg['verbose'] = kwargs.pop('verbose', DEFAULT_CFG['verbose'])
    cfg['classifier_only'] = kwargs.pop('classifier_only', DEFAULT_CFG['classifier_only'])
    cfg['reset_running_statistics'] = kwargs.pop(
        'reset_running_statistics', DEFAULT_CFG['reset_running_statistics'])

    execution_line = "CUDA_VISIBLE_DEVICES={} python evaluator.py".format(gpus)
    for k, v in cfg.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    execution_line += ' &'
    return execution_line
'''

def bash_command_template_multi_exits(**kwargs):

    gpus = kwargs.pop('gpus', DEFAULT_CFG['gpus'])
    cfg = OrderedDict()

    cfg['dataset'] = kwargs['dataset']
    cfg['model'] = kwargs['model']
    cfg['device'] = gpus #kwargs['device']
    cfg['resolution'] = kwargs['res']
    cfg['model_path'] = kwargs['subnet']
    cfg['output_path'] = kwargs['save']
    cfg['pretrained'] = kwargs['pretrained']
    cfg['supernet_path'] = kwargs['supernet_path']
    cfg['batch_size'] = kwargs.pop('trn_batch_size', DEFAULT_CFG['trn_batch_size'])
    cfg['mmax'] = kwargs['mmax']
    cfg['top1min'] = kwargs['top1min']
    cfg['method'] = kwargs['method']
    cfg['val_split'] = kwargs['val_split']
    cfg['w_gamma'] = kwargs['w_gamma']
    cfg['w_beta'] = kwargs['w_beta']
    cfg['w_alpha'] = kwargs['w_alpha']
    cfg['support_set'] = kwargs['support_set']
    cfg['tune_epsilon'] = kwargs['tune_epsilon']
    cfg['backbone_epochs'] = kwargs['n_epochs']
    cfg['warmup_ee_epochs'] = kwargs['warmup_ee_epochs']
    cfg['ee_epochs'] = kwargs['ee_epochs']
    cfg['n_workers'] = kwargs['n_workers']

    execution_line = "python ee_train.py".format(gpus)
    for k, v in cfg.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    execution_line += ' &'
    return execution_line

def bash_command_template_single_exit(**kwargs):

    gpus = kwargs.pop('gpus', DEFAULT_CFG['gpus'])
    cfg = OrderedDict()
    
    cfg['dataset'] = kwargs['dataset']
    cfg['data'] = kwargs['data']
    cfg['model'] = kwargs['model']
    cfg['device'] = gpus #kwargs['device']
    cfg['model_path'] = kwargs['subnet']
    cfg['output_path'] = kwargs['save']
    cfg['pretrained'] = kwargs['pretrained']
    cfg['supernet_path'] = kwargs['supernet_path']
    cfg['epochs'] = kwargs.pop('n_epochs', DEFAULT_CFG['n_epochs'])
    cfg['batch_size'] = kwargs.pop('trn_batch_size', DEFAULT_CFG['trn_batch_size'])
    cfg['optim'] = kwargs['optim']
    cfg['sigma_min'] = kwargs['sigma_min']
    cfg['sigma_max'] = kwargs['sigma_max']
    cfg['sigma_step'] = kwargs['sigma_step']
    cfg['alpha'] = kwargs['alpha']
    cfg['res'] = kwargs['res']
    cfg['func_constr'] = kwargs['func_constr']
    cfg ['pmax'] = kwargs['pmax']
    cfg ['mmax'] = kwargs['mmax']
    cfg ['amax'] = kwargs['amax']
    cfg ['wp'] = kwargs['wp']
    cfg ['wm'] = kwargs['wm']
    cfg ['wa'] = kwargs['wa']
    cfg['penalty'] = kwargs['penalty']
    cfg['alpha_norm'] = kwargs['alpha_norm']
    cfg['val_split'] = kwargs['val_split']
    cfg['n_workers'] = kwargs['n_workers']
    cfg['quantization'] = kwargs['quantization']

    execution_line = "python train.py".format(gpus)
    for k, v in cfg.items():
        if v is not None:
            if isinstance(v, bool):
                if v:
                    execution_line += " --{}".format(k)
            else:
                execution_line += " --{} {}".format(k, v)
    execution_line += ' &'
    return execution_line

def get_template_by_type(gpus, subnet, save, type, **kwargs):

    if type == 'single_exit':
        return bash_command_template_single_exit(gpus=gpus, subnet=subnet, save=save, **kwargs)
    elif type == 'multi_exits':
        return bash_command_template_multi_exits(gpus=gpus, subnet=subnet, save=save, **kwargs)
    else:
        raise ValueError('Unknown template type: {}'.format(type))


def prepare_eval_folder(path, configs, gpu=2, n_gpus=8, gpu_list=None, type='single-exit', iter_num=0, slurm=False, **kwargs):
    """ create a folder for parallel evaluation of a population of architectures """

    os.makedirs(path, exist_ok=True)
    gpu_template = ','.join(['{}'] * gpu)
    if gpu_list is not None:
        n_gpus = len(gpu_list)
        gpus = [gpu_template.format(i, i + 1) for i in gpu_list]
    else:
        gpus = [gpu_template.format(i, i + 1) for i in range(0, n_gpus, gpu)]
    bash_file = ['#!/bin/bash']
    for i in range(0, len(configs), n_gpus//gpu):
        for j in range(n_gpus//gpu):
            if i + j < len(configs):
                experiment_path = os.path.join(path, 'net_{}'.format(i + j))
                os.makedirs(experiment_path, exist_ok=True)
                job = os.path.join(experiment_path, "net_{}.subnet".format(i + j))
                with open(job, 'w') as handle:
                    json.dump(configs[i + j], handle)
                bash_file.append(get_template_by_type(gpus=gpus[j], subnet=job, save=experiment_path, type=type, **kwargs))
        bash_file.append('wait')

    with open(os.path.join(path, 'run_bash.sh'), 'w') as handle:
        for line in bash_file:
            handle.write(line + os.linesep)
    
    if slurm:
        # Create the SLURM script for this iteration with EDANAS_ITERx as job name
        slurm_script_path = os.path.join(path, f'run_slurm.sh')
        create_slurm_script(slurm_script_path, iter_num, path)

def create_slurm_script(slurm_script_path, iter_num, experiment_path):
    """ Helper function to create a SLURM script with the dynamic job name EDANAS_ITERx """
    
    slurm_script_content = f"""#!/bin/bash
# SLURM script to be runned through `sbatch job.sh`
# In the following slurm options, customize (if needed) only the ones with comments
#SBATCH --job-name="EDANAS_ITER{iter_num}"   # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                    # Number of threads
#SBATCH --time=72:00:00                      # Walltime limit
#SBATCH --gpus=1                             # Num gpus
#SBATCH --partition=gpu                      # GPU partition
#SBATCH --account=pittorino
#SBATCH --mail-type=NONE
#SBATCH --mail-user=fabrizio.pittorino@unibocconi.it
#SBATCH --output=out/%x_%j.out               # Where to write standard output
#SBATCH --error=err/%x_%j.err                # Where to write standard error

# PARTITIONS
# If you have cpu job change partition to compute.
# defq, timelimit 3 days, Nodes=cnode0[1-4] (CPU)
# compute, timelimit 15 days, Nodes=cnode0[5-8] (CPU)
# gpu, timelimit 3 days, Nodes=gnode0[1-4] (GPU)
# debug, timelimit 30 minutes, Nodes=cnode01,gnode04 (short test on either CPU or GPU)
# QOS long: long jobs (7 days max) on gnode0[1-4] (GPU)

### export PATH="/home/Pittorino/miniconda3/bin:$PATH"
#export PATH="/home/Pittorino/miniconda3:$PATH"

module load cuda/12.3
#conda activate timefs

# Call the generated run_bash.sh in the experiment path
bash {os.path.join(experiment_path, 'run_bash.sh')}
"""
    # Write the SLURM script to the specified path
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script_content)


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            try:
                X[i, np.random.choice(is_false)] = True
                X[i, np.random.choice(is_true)] = False
            except ValueError:
                pass

        return X


class LatencyEstimator(object):
    """
    Modified from https://github.com/mit-han-lab/proxylessnas/blob/
    f273683a77c4df082dd11cc963b07fc3613079a0/search/utils/latency_estimator.py#L29
    """
    def __init__(self, fname):
        # fname = download_url(url, overwrite=True)

        with open(fname, 'r') as fp:
            self.lut = yaml.load(fp, yaml.SafeLoader)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, ltype: str, _input, output, expand=None,
                kernel=None, stride=None, idskip=None, se=None):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `first_conv`: The initial stem 3x3 conv with stride 2
                2. `final_expand_layer`: (Only for MobileNet-V3)
                    The upsample 1x1 conv that increases num_filters by 6 times + GAP.
                3. 'feature_mix_layer':
                    The upsample 1x1 conv that increase num_filters to num_features + torch.squeeze
                3. `classifier`: fully connected linear layer (num_features to num_classes)
                4. `MBConv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        :param se: indicate whether has squeeze-and-excitation
        """
        infos = [ltype, 'input:%s' % self.repr_shape(_input),
                 'output:%s' % self.repr_shape(output), ]
        if ltype in ('MBConv',):
            assert None not in (expand, kernel, stride, idskip, se)
            infos += ['expand:%d' % expand, 'kernel:%d' % kernel,
                      'stride:%d' % stride, 'idskip:%d' % idskip, 'se:%d' % se]
        key = '-'.join(infos)
        return self.lut[key]['mean']


def look_up_latency(net, lut, resolution=224):
    def _half(x, times=1):
        for _ in range(times):
            x = np.ceil(x / 2)
        return int(x)

    predicted_latency = 0

    # first_conv
    predicted_latency += lut.predict(
        'first_conv', [resolution, resolution, 3],
        [resolution // 2, resolution // 2, net.first_conv.out_channels])

    # final_expand_layer (only for MobileNet V3 models)
    input_resolution = _half(resolution, times=5)
    predicted_latency += lut.predict(
        'final_expand_layer',
        [input_resolution, input_resolution, net.final_expand_layer.in_channels],
        [input_resolution, input_resolution, net.final_expand_layer.out_channels]
    )

    # feature_mix_layer
    predicted_latency += lut.predict(
        'feature_mix_layer',
        [1, 1, net.feature_mix_layer.in_channels],
        [1, 1, net.feature_mix_layer.out_channels]
    )

    # classifier
    predicted_latency += lut.predict(
        'classifier',
        [net.classifier.in_features],
        [net.classifier.out_features]
    )

    # blocks
    fsize = _half(resolution)
    for block in net.blocks:
        idskip = 0 if block.config['shortcut'] is None else 1
        se = 1 if block.config['mobile_inverted_conv']['use_se'] else 0
        stride = block.config['mobile_inverted_conv']['stride']
        out_fz = _half(fsize) if stride > 1 else fsize
        block_latency = lut.predict(
            'MBConv',
            [fsize, fsize, block.config['mobile_inverted_conv']['in_channels']],
            [out_fz, out_fz, block.config['mobile_inverted_conv']['out_channels']],
            expand=block.config['mobile_inverted_conv']['expand_ratio'],
            kernel=block.config['mobile_inverted_conv']['kernel_size'],
            stride=stride, idskip=idskip, se=se
        )
        predicted_latency += block_latency
        fsize = out_fz

    return predicted_latency


def get_net_info(net, input_shape=(3, 224, 224), print_info=False):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/
    35ddcb9ca30905829480770a6a282d49685aa282/ofa/imagenet_codebase/utils/pytorch_utils.py#L139
    """

    # artificial input data
    inputs = torch.randn(1, 3, input_shape[-2], input_shape[-1])

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        net = net.to(device)
        cudnn.benchmark = True
        inputs = inputs.to(device)

    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module
    
    net.eval() # this avoids batch norm error https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274

    # parameters
    net_info['params'] = np.round(count_parameters(net)/1e6,2)

    net = copy.deepcopy(net)

    net_info['macs'] = np.round(profile_macs(net, inputs)/1e6,2)

    # activation_size
    net_info['activations'] = np.round(profile_activation_size(net, inputs)/1e6,2)

    if print_info:
        # print(net)
        print('Total training params: %.2fM' % (net_info['params']))
        print('Total MACs: %.2fM' % ( net_info['macs']))
        print('Total activations: %.2fM' % (net_info['activations']))

    return net_info

def get_network_search(model, subnet_path, n_classes=10, supernet='supernets/ofa_mbv3_d234_e346_k357_w1.0', pretrained=True, func_constr=False):

    config = json.load(open(subnet_path))

    if model!='nasbench':
        ofa = OFAEvaluator(n_classes=n_classes,
        model_path=supernet,
        pretrained = pretrained)
        r=config.get("r",None)
        subnet, _ = ofa.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})
    else:
        print("NASBench201 model selected")
        bench=NASBench201SearchSpace()
        vector=bench.matrix2vector(bench.str2matrix(config['arch']))
        r=32
        subnet=NASBenchNet(cell_encode=vector, C=16, num_classes=n_classes, stages=3, cells=5, steps=4)

    # Functional constraints
    if(func_constr):
        print("Functional constraints applied")
        subnet=substitute_activation(subnet,nn.ReLU,OurReLU())

    return subnet, r

def get_subnet_folder(exp_path, subnet):
    """ search for a subnet folder in the experiment folder filtering by subnet architecture """
    import glob
    split = exp_path.rsplit("_",1)
    maxiter = int(split[1])
    path = exp_path.rsplit("/",1)[0]
    folders=[]

    for file in glob.glob(os.path.join(path + '/iter_*', "net_*/net_*.subnet")):
        arch = json.load(open(file))  
        pre,ext= os.path.splitext(file)
        split = pre.rsplit("_",3)  
        split2 = split[1].rsplit("/",1)
        niter = int(split2[0])
        if arch == subnet and niter <= maxiter:
            folder_path = pre.rsplit("/",1)[0]
            folders.append(folder_path) # add folder to list (handle duplicates)
    
    if(len(folders)==0):
        print("Error: no subnet found in archive!")
        return None
    
    return folders[-1]

def tiny_ml(params,macs,activations,pmax,mmax,amax,wp,wm,wa,penalty):
  output = wp*(params + penalty*max(0,params-pmax)) + wm*(macs + penalty*max(0,macs-mmax)) + wa*(activations + penalty*max(0,activations-amax))
  return output

class OurReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):

        return 0.01 * torch.pow(x, 2) + 0.5 * x
        #return torch.pow(x, 2)

def substitute_activation(model, old_activation, new_activation):
    """
    Replace all old activations in a PyTorch model with a new activation function.

    Args:
        model (nn.Module): PyTorch neural network model.
        new_activation (nn.Module): Custom activation function to replace old activation.

    Returns:
        nn.Module: PyTorch model with old activations replaced by new activation.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("Input model must be a PyTorch nn.Module")

    for name, module in model.named_children():
        if isinstance(module, old_activation):
            setattr(model, name, new_activation)
        else:
            substitute_activation(module, old_activation, new_activation)

    return model
