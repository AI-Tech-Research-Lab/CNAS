import copy
import json
import random
import os

import numpy as np
from ofa_evaluator import OFAEvaluator
from ofa.utils.pytorch_utils import count_parameters

import torch
from torch.nn import Conv2d,ReLU,Linear,Sequential,Flatten,BatchNorm2d,AvgPool2d,MaxPool2d, Identity
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchprofile import profile_macs

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset

from train_utils import get_device    

from models.base import BinaryIntermediateBranch, IntermediateBranch
from models.mobilenet_v3 import EEMobileNetV3, FinalClassifier
from models.resnet import resnet20
from models.alexnet import AlexNet
from models.costs import module_cost
from evaluators import binary_eval


def get_intermediate_backbone_cost(backbone, input_size):
    # Compute the MACs of the backbone up to the b-th exit for each exit

    #backbone_macs_i = [0] * backbone.b
    b_params = []
    b_macs = []

    for b in range(backbone.b):
            params, macs = get_backbone_i_cost(backbone, input_size, b)
            b_params.append(params)
            b_macs.append(macs)
        
    return b_params, b_macs

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
    net_info['params'] = count_parameters(net)

    net = copy.deepcopy(net)

    net_info['macs'] = int(profile_macs(net, inputs))

    # activation_size
    #net_info['activations'] = int(profile_activation_size(net, inputs))

    if print_info:
        # print(net)
        print('Total training params: %.2fM' % (net_info['params'] / 1e6))
        print('Total MACs: %.2fM' % ( net_info['macs'] / 1e6))
        #print('Total activations: %.2fM' % (net_info['activations'] / 1e6))

    return net_info

def get_backbone_i_cost(backbone,input_size,num_exit):
    # Compute the MACs of the backbone up to the b-th exit for a given exit
    net = copy.deepcopy(backbone)
    idx = net.exit_idxs[num_exit] + 1
    net.blocks = net.blocks[:idx]
    #net.exit_idxs = [net.exit_idxs[num_exit]]
    info = get_net_info(net,input_size) # MACs of the backbone up to the b-th exit
    return info['params']/1e6, info['macs']/1e6

def get_classifier_i_cost(predictor, input_sample):
    #print("PREDICTOR",predictor)
    layers = [module for module in predictor.modules() if not isinstance(module, torch.nn.Sequential)]
    macs = 0
    for name, m_int in predictor.named_children():
        #print("NAME",name)
        c = module_cost(input_sample, m_int)
        #print("INPUT SAMPLE ", input_sample.shape)
        input_sample = m_int(input_sample)
        #print("OUTPUT SAMPLE ", input_sample.shape)
        macs += c
        #print("MACS",macs)
    params = 0
    for l in layers:
        params += sum(p.numel() for p in l.parameters() if p.requires_grad)
    params = params/1e6
    macs = macs/1e6
    return params, macs


def get_intermediate_classifiers_cost(model, predictors, image_size):

    model=copy.deepcopy(model)
    x = torch.randn((1,) + image_size)

    device = get_device(model)

    model.to(device)
    predictors.to(device)
    x=x.to(device)

    model.eval() # this avoids batch norm error https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    outputs = model(x)

    c_params=[]
    c_macs=[]
    #c_act=[]

    for i, o in enumerate(outputs):
        params,macs = get_classifier_i_cost(predictors[i],o)
        c_params.append(params)
        c_macs.append(macs)

    return c_params,c_macs

def get_ee_scores(backbone, testloader, classifiers, backbone_macs_i, classifiers_macs, epsilon, cumulative_threshold):

    scores,exits_counter = binary_eval(model=backbone,
                                    dataset_loader=testloader,
                                    predictors=classifiers,
                                    epsilon = epsilon,
                                    cumulative_threshold=cumulative_threshold
                                    )
           
    n_samples = 10000 #len of cifar10 eval dataset
    weights = []
    for ex in exits_counter.values():
            weights.append(ex/n_samples)
    avg_macs = 0

    # For each b-th exit the avg_macs is the percentage of samples exiting from the exit 
    # multiplied by the sum of the MACs of the backbone up to the b-th exit + MACs of the b-th exit 

    for b in range(backbone.b):
        avg_macs += weights[b] * (backbone_macs_i[b] + classifiers_macs[b])

    return scores, weights, avg_macs

def calculate_maxpool_kernel_size(num_channels):
        # Calculate the kernel size based on the number of channels
        if num_channels >= 16:
            return 4
        elif num_channels >= 8:
            return 2
        else:
            return 1
        
'''
def get_intermediate_classifiers_static(model,
                                 image_size,
                                 n_classes,
                                 binary_branch=False,
                                 fix_last_layer=False):
    predictors = nn.ModuleList()
    x = torch.randn((1,) + image_size)
    outputs = model(x)

    for i, o in enumerate(outputs):
        chs = o.shape[1]

        if i == (len(outputs) - 1):
            od = torch.flatten(o, 1).shape[-1]

            if binary_branch:
                if fix_last_layer:
                    linear_layers = nn.Sequential(*[nn.ReLU(),
                                                    nn.Linear(od, n_classes)])

                    b = BinaryIntermediateBranch(preprocessing=nn.Flatten(),
                                                 classifier=linear_layers,
                                                 return_one=True)
                else:
                    linear_layers = nn.Sequential(*[nn.ReLU(),
                                                    nn.Linear(od,
                                                              n_classes + 1)])

                    b = BinaryIntermediateBranch(preprocessing=nn.Flatten(),
                                                 classifier=linear_layers,
                                                 )
            else:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, n_classes)])

                b = IntermediateBranch(preprocessing=nn.Flatten(),
                                       classifier=linear_layers)

            predictors.append(b)
        else:

            if o.shape[-1] >= 6:
                seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(chs, 128,
                              kernel_size=3, stride=1),
                    nn.MaxPool2d(3),
                    nn.ReLU())
            else:
                seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(chs, 128,
                              kernel_size=2, stride=1),
                    # nn.MaxPool2d(2),
                    nn.ReLU())

            seq.add_module('flatten', nn.Flatten())

            output = seq(o)
            output = torch.flatten(output, 1)
            od = output.shape[-1]

            if binary_branch:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, n_classes + 1)])

                predictors.append(
                    BinaryIntermediateBranch(preprocessing=seq,
                                             classifier=linear_layers,
                                             ))

            else:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, n_classes)])

                predictors.append(IntermediateBranch(preprocessing=seq,
                                                     classifier=linear_layers))
            predictors[-1](o)

    return predictors
'''

def get_intermediate_classifiers_adaptive(model, final_classifier,
                                 image_size,
                                 n_classes,
                                 binary_branch=False):
                                 #fix_last_layer=False):
    
    predictors = nn.ModuleList()
    model = copy.deepcopy(model)
    final_classifier = copy.deepcopy(final_classifier)
    x = torch.randn((1,) + image_size)
    model.eval() # this avoids batch norm error https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    #outputs = get_blocks_mbv3(num_classes,x)
    #device of the model
    # Move the model to the CPU
    device = torch.device("cpu")
    model = model.to(device)
    fc = final_classifier.to(device)
    # Move the input tensor to the same device as the model
    x = x.to(device)
    outputs = model(x) 
    filters = [32,64,128]
    seq = nn.Sequential(fc.final_expand_layer, fc.global_avg_pool, fc.feature_mix_layer, nn.Flatten())
    cl = fc.classifier
    final_classifier = BinaryIntermediateBranch(preprocessing=seq,
                                                 classifier=cl,
                                                 return_one=True)
    predictors.append(final_classifier)
    outputs_ee = outputs[:-1]
    for i, o in enumerate(reversed(outputs_ee)):
        
        i = len(outputs) - i - 2

        chs = o.shape[1]

        conv1 = nn.Conv2d(in_channels=chs, out_channels=filters[0], kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=1, padding=1)
        relu = nn.ReLU()

        seq = nn.Sequential(
            relu,
            conv1,
            relu,
            conv2,
            relu,
            conv3,
            relu)

        seq.add_module('flatten', nn.Flatten())

        output = seq(o)
        output = torch.flatten(output, 1)
        od = output.shape[-1]

        linear_layers = nn.Sequential(*[nn.ReLU(),
                                        nn.Linear(od, n_classes + 1)])
        #num_classes + 1 for the confidence
        
        pred = BinaryIntermediateBranch(preprocessing=seq,
                                        classifier=linear_layers,
                                        )

        _, b_macs_i = get_backbone_i_cost(model, image_size, i)
        _, c_macs_i = get_classifier_i_cost(pred, o)
        _, b_macs_next = get_backbone_i_cost(model, image_size, i+1)
        #if len(predictors) == 1: #compute the macs of the final classifier         
        #    c_macs_next = int(profile_macs(predictors[-1], outputs[i+1]))/1e6 #get_net_info(predictors[-1], (input_sample[-3],input_sample[-2],input_sample[-1]))['macs']
        #else:
        _, c_macs_next = get_classifier_i_cost(predictors[-1], outputs[i+1])
        print("MACS:", c_macs_next)
        max_ks = calculate_maxpool_kernel_size(chs)
        ks = 1
        while((b_macs_i + c_macs_i) >= (b_macs_next + c_macs_next) 
                and ks <= max_ks):
            max_pool = nn.MaxPool2d(kernel_size=ks, stride=ks)
            new_seq = nn.Sequential(max_pool,*seq)
            output = new_seq(o)
            output = torch.flatten(output, 1)
            od = output.shape[-1]
            linear_layers = nn.Sequential(*[nn.ReLU(),
                                            nn.Linear(od, n_classes + 1)])
            pred = BinaryIntermediateBranch(preprocessing=new_seq,
                                        classifier=linear_layers,
                                        )
            _, c_macs_i = get_classifier_i_cost(pred, o)
            ks = ks * 2

        predictors.append(pred)
        #else add max pool

    predictors = nn.ModuleList(list(reversed(predictors)))

    return predictors

def extract_balanced_subset(train_loader, subset_percentage, n_classes):

    # Determine the percentage of the total dataset to extract as a subset
    # Calculate the size of the subset based on the percentage
    train_dataset = train_loader.dataset
    total_samples = len(train_dataset)
    subset_size = int(subset_percentage * total_samples)

    # Calculate the number of samples per class for the balanced subset
    samples_per_class = subset_size // n_classes

    # Get all indices of the dataset
    all_indices = list(range(total_samples))

    # Shuffle the indices
    random.shuffle(all_indices)

    # Initialize a list to store the indices of the balanced subset
    subset_indices = []

    # Count the number of samples for each class
    class_counts = [0] * n_classes

    # Iterate over the shuffled dataset to extract the balanced subset
    for index in all_indices:
        data, target = train_dataset[index]
        class_label = int(target)

        # Check if adding a sample for this class exceeds the limit
        if class_counts[class_label] < samples_per_class:
            subset_indices.append(index)
            class_counts[class_label] += 1

        # Check if we have collected enough samples for each class
        if all(count == samples_per_class for count in class_counts):
            break

    # Create a Subset object and loader for the balanced subset indices
    balanced_support_dataset = Subset(train_dataset, subset_indices)
    bs = 64
    balanced_support_loader = DataLoader(balanced_support_dataset, batch_size=bs, shuffle=True)
    
    
    # Remove the balanced subset samples from the original dataset
    remaining_indices = list(set(range(total_samples)) - set(subset_indices))
    remaining_dataset = Subset(train_dataset, remaining_indices)

    # Create a new train loader for the modified dataset
    modified_train_loader = DataLoader(remaining_dataset, batch_size=bs, shuffle=True)

    return balanced_support_loader, modified_train_loader

def is_dataset_balanced(train_loader,num_classes):
    class_counts = [0] * num_classes

    # Count the number of samples for each class in the dataset
    for _, targets in train_loader:
        for target in targets:
            class_counts[target] += 1

    # Check if the number of samples in each class is the same
    is_balanced = all(count == class_counts[0] for count in class_counts)

    return is_balanced
    
def calculate_centroids(dataset, model, n_classes):

    #device = next(iter(model.parameters())).device()
    device = next(iter(model.parameters())).device
    
    # fetch the first element of the first sample of the dataset
    x = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))[0] 
    x = x.to(device)

    with torch.no_grad():
    
        centroids = model(x)[0].unsqueeze(0) # add a new dimension 
        centroids = centroids.expand(n_classes, -1) # duplicate tensors for n_classes
        
        centroids = torch.zeros_like(centroids) # initialize a tensor with same shape and set elements to zero
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        counter = torch.zeros(len(centroids), 1, device=device)
        
        for d in dataloader:
            x, y = d
            x = x.to(device)
            y = y.to(device)

            embeddings = model(x)
            # adds the embeddings for the corresponding class
            centroids = torch.index_add(centroids, 0, y, embeddings) 
            # update the counter for the corresponding class
            counter = torch.index_add(counter, 0, y, torch.ones_like(y, dtype=counter.dtype)[:, None])

        centroids = centroids / counter #compute the average data point for each class

    return centroids

def calculate_centroids_confidences(dataset, model, predictors, n_classes):
    # n_classes: number of classes
    # predictors: list of branches (exits)

    n_exits = len(predictors)

    # device = next(iter(model.parameters())).device()
    device = next(iter(model.parameters())).device
    
    # fetch the first element of the first sample of the dataset
    x = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))[0] 
    x = x.to(device)

    with torch.no_grad():
    
        centroids = torch.zeros(n_classes, n_exits, device=device) # initialize centroids matrix
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        counter = torch.zeros(n_classes, device=device)  # Initialize counter as a vector
        
        for d in dataloader:
            x, y = d
            x = x.to(device)
            y = y.to(device)

            bos = model(x)
            for j, bo in enumerate(bos):
                _, c = predictors[j](bo)
                # adds the embeddings for the corresponding class
                centroids = torch.index_add(centroids, 1, y, c) 
                # update the counter for the corresponding class
                counter = torch.index_add(counter, 0, y, torch.ones_like(y, dtype=counter.dtype)[:, None])

        # compute the average data point for each class and exit
        centroids = centroids / counter[:, None]

    return centroids


def calculate_centroids_scores(dataloader, model, predictors, n_classes):
    # n_classes: number of classes
    # predictors: list of branches (exits)
    # dataloader: dataloader for a BALANCED dataset (equal num of samples for each class)

    n_exits = len(predictors)

    # device = next(iter(model.parameters())).device()
    device = next(iter(model.parameters())).device

    slice_dim = len(dataloader.dataset) // n_classes
    
    with torch.no_grad():
        centroids = torch.zeros(n_classes, n_exits, device=device) # initialize centroids matrix

        # Divide the dataloader samples for each class
        class_slices = [torch.tensor([idx for idx, (x,y) in enumerate(dataloader.dataset) if y == class_idx]) 
                        for class_idx in range(n_classes)]

        for class_idx, class_slice in enumerate(class_slices):
            class_dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=slice_dim, 
                                                           sampler=class_slice, 
                                                           num_workers=4,
                                                           pin_memory=True)

            for d in class_dataloader:
                x, y = d
                x = x.to(device)
                y = y.to(device)

                #assert all(label == class_idx for label in y), "Assertion failed: Labels do not match class_idx."

                bos = model(x)
                for j, bo in enumerate(bos):
                    l, _ = predictors[j](bo)
                    y_hat = torch.argmax(l, dim=1)
                    correct_predictions = torch.eq(y_hat, y).sum()#.item()
                    total_samples = y.size(0)
                    branch_score = correct_predictions / total_samples
                    # adds the embeddings for the corresponding class
                    centroids[class_idx][j] = branch_score# * torch.ones(n_exits, device=device)
                    # update the counter for the corresponding class
                    #counter[class_idx] += total_samples

    return centroids

def get_subnet_folder_by_backbone(exp_path, backbone, nsubnet):
        """ search for a subnet folder in the experiment folder filtering by subnet architecture """
        import glob
        split = exp_path.rsplit("_",1)
        maxiter = int(split[1])
        path = exp_path.rsplit("/",1)[0] 
        folder_path=None

        for file in glob.glob(os.path.join(path + '/iter_*', "net_*/net_*.subnet")):
            arch = json.load(open(file))  
            pre,ext= os.path.splitext(file)
            num=int(pre.rsplit("_",1)[1]) 
            split = pre.rsplit("_",3)  
            split2 = split[1].rsplit("/",1)
            niter = int(split2[0])
            arch = {'ks':arch['ks'],'e':arch['e'],'d':arch['d']}
            if arch == backbone and (niter<maxiter or (niter==maxiter and num<nsubnet)) :
                folder_path = pre.rsplit("/",1)[0]
                return folder_path

def cp_exps(src_path, dest_path, only_backbone = False):
        """ copy experiments from src_path to dest_path """
        import glob
        import os
        import shutil
        split = src_path.rsplit("_",1)
        maxiter = int(split[1]) 
        path = src_path.rsplit("/",1)[0] 
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for folder in glob.glob(os.path.join(path + '/iter_*', "net_*")):
            split = folder.rsplit("_",2)  
            split2 = split[1].rsplit("/",1)
            niter = int(split2[0])
            nsubnet = int(split[2])
            if (niter <= maxiter):
                iter_folder = os.path.join(dest_path,'iter_'+str(niter))
                if not os.path.exists(iter_folder):
                      os.makedirs(iter_folder)
                dest_folder = os.path.join(iter_folder,'net_' + str(nsubnet))
                if not os.path.exists(dest_folder):
                      os.makedirs(dest_folder)
                shutil.rmtree(dest_folder, ignore_errors=True)
                src_folder = path + '/iter_' + str(niter) + '/net_' + str(nsubnet)
                shutil.copytree(src_folder, dest_folder)
                if(only_backbone):
                    os.remove(dest_folder + '/bb.pt')
                    os.remove(dest_folder + '/classifiers.pt')
                    os.remove(dest_folder + '/net_' + str(nsubnet) + '.stats')

def ece_score(method, dataset, bins=30):

    from eval import get_predictions

    true, predictions, probs = get_predictions(method, dataset)
    probs = np.max(probs, -1)

    prob_pred = np.zeros((0,))
    prob_true = np.zeros((0,))
    ece = 0

    mce = []

    for b in range(1, int(bins) + 1):
        i = np.logical_and(probs <= b / bins, probs > (b - 1) / bins)  # indexes for p in the current bin

        s = np.sum(i)

        if s == 0:
            prob_pred = np.hstack((prob_pred, 0))
            prob_true = np.hstack((prob_true, 0))
            continue

        m = 1 / s
        acc = m * np.sum(predictions[i] == true[i])
        conf = np.mean(probs[i])

        prob_pred = np.hstack((prob_pred, conf))
        prob_true = np.hstack((prob_true, acc))
        diff = np.abs(acc - conf)

        mce.append(diff)

        ece += (s / len(true)) * diff

    return ece, prob_pred, prob_true, mce

def get_eenn(subnet, subnet_path, res, n_classes, get_binaries=False):

    #model, res = get_eenn_from_OFA(subnet_path, n_classes, supernet, pretrained)
    config = json.load(open(subnet_path))
    backbone = EEMobileNetV3(subnet.first_conv, subnet.blocks, config['b'], config['d'])
    final_classifier = FinalClassifier(subnet.final_expand_layer, subnet.feature_mix_layer, subnet.classifier)
    img_size = (3, res, res)
    classifiers = get_intermediate_classifiers_adaptive(backbone,
                                        final_classifier,
                                        img_size,
                                        n_classes,
                                        binary_branch=get_binaries)
    
    return backbone, classifiers

def save_eenn(backbone, classifiers, best_backbone, best_classifiers, best_score, epoch, optimizer, ckpt_path):
    checkpoint = {
        'backbone_state': backbone.state_dict(),
        'classifiers_state': classifiers.state_dict(),
        'best_backbone_state': best_backbone, #already state_dict
        'best_classifiers_state': best_classifiers, #same
        'optimizer_state': optimizer.state_dict(),
        'best_eval_score': best_score,
        'epoch': epoch,
    }
    torch.save(checkpoint, ckpt_path)

'''
def get_eenn_from_OFA(subnet_path, n_classes=10, supernet='supernets/ofa_mbv3_d234_e346_k357_w1.0', pretrained=True, early_exit=False):

    print("SUBNET PATH: ", subnet_path)
    config = json.load(open(subnet_path))
    ofa = OFAEvaluator(n_classes=n_classes,
    model_path=supernet,
    pretrained = pretrained)
    r=config.get("r",32)
    input_shape = (3,r,r)
    subnet, _ = ofa.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})
    subnet = EEMobileNetV3(subnet.first_conv, subnet.blocks, config['b'], config['d'], subnet.final_expand_layer, subnet.feature_mix_layer, subnet.classifier)
    return subnet, r
'''