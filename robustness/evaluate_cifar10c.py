"""Evaluate models on ImageNet-C"""
# Evaluation functions from the repo of the paper: Robustness properties of Facebook's ResNeXt WSL models

import json
import os
import time
import argparse
import sys

sys.path.append(os.getcwd())

import numpy as np
from collections import OrderedDict
from ofa_evaluator import OFAEvaluator
import torch
import torch.utils.data
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from trainers.entropic.utility.corruptions import ApplyDistortion
import torch.nn as nn
import torch.optim as optim
from train_utils import load_checkpoint, train, validate
from PIL import Image

sys.path.append(os.getcwd())


parser = argparse.ArgumentParser(description='Evaluate models on ImageNet-C')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', type=str, default='alexnet',
                    choices=['resnext101_32x8d', 'resnext101_32x8d_wsl', 'resnext101_32x16d_wsl',
                             'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl'], help='evaluated model')
parser.add_argument('--workers', default=4, type=int, help='no of data loading workers')
parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--supernet_path', default='./supernets/ofa_mbv3_d234_e346_k357_w1.0', type=str, help='path of the supernet to evaluate')
parser.add_argument('--model_path', default='', type=str, help='path of the model to evaluate')

def compute_mCE_CIFARC(data, model, device, res=32, batch_size=64, workers=4, model_name="", save_stats=False):

    # This function takes as input a model and computes the mCE on the CIFAR-10/100-C dataset

    distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                   'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                   'jpeg_compression']
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    distortion_errors = {}

    print("SEEN RESOLUTION: ", res)

    for distortion_name in distortions:

        errs = []
        scores = {}

        for severity in range(1, 6):
            print(distortion_name, str(severity))

            valdir = data + '/' + distortion_name + '/' + str(severity)
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize((res, res)),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
            

            # evaluate on validation set
            acc1 = validate(val_loader, model, device)
            errs.append(100. - acc1)
            scores[severity] = np.round(errs[-1],2) # 100 - acc1

        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 1. * np.mean(errs)))
        scores['mean'] = np.round(np.mean(errs),2)
        distortion_errors[distortion_name] = scores

        if save_stats:
            file_path = 'cifarc_errors_' + model_name + '.json'
            # Open the file in write mode
            with open(file_path, 'w') as file:
                # Serialize the dictionary and write it to the file
                json.dump(distortion_errors, file)

    #np.save('imagenetc_errors_' + model_name + '.npy', np.array(distortion_errors))

    return distortion_errors

def save_transformed_images(images, labels, dataset, distortion_name, severity, res):
    folder_path = os.path.join('../datasets/transform_' + dataset + '_res' + str(res) + '/' + distortion_name + '/' + str(severity))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, (img,label) in enumerate(zip(images,labels)):
        label_path = folder_path + '/' + str(label)
        if not os.path.exists(label_path):
          os.makedirs(label_path)
        img_name = f"image_{i}.png"
        img_path = os.path.join(label_path, img_name)
        #torchvision.utils.save_image(img, img_path)
        img = img * 255
        img=img.astype(np.uint8)
        img = Image.fromarray(img)
        #print(label)
        #img.show()
        img.save(img_path)
    
def save_distortions(dataset, res):

    distortions = ['gaussian_noise']

    for distortion_name in distortions:

        for severity in range(1, 6):

                print(distortion_name, str(severity))

                if not os.path.exists(os.path.join('../datasets/transform_' + dataset + '_res' + str(res) + '/' + distortion_name + '/' + str(severity))):

                    t=transforms.Compose([
                        transforms.Resize((res, res)),
                        ApplyDistortion(distortion_name, severity), # Apply the distortion function
                    ])
                    if dataset=='cifar10':
                        val_set = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=t)
                    elif dataset=='cifar100':
                        val_set = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=True, transform=t)
                    labels=val_set.targets
                    images = [t(Image.fromarray((img))) for img in val_set.data]
                    print("Saving images...")
                    save_transformed_images(images, labels, dataset, distortion_name, severity, res)

def compute_mCE(dataset, model, device, res=32, load_ood=False, batch_size=64, workers=4, model_name="", save_stats=False):

    # This function takes as input a model and computes the mCE on the CIFAR 10/100 dataset with corruption transformations applied
    
    distortions = [
                   'gaussian_noise', 
                     'shot_noise', 
                    'impulse_noise', 
                    'defocus_blur', 
                    'glass_blur', 
                    'motion_blur', 
                    'zoom_blur', 
                    'snow',
                   'frost', 
                   'fog', 
                   'brightness', 
                   'contrast', 
                   'elastic_transform', 
                   'pixelate', 
                   'jpeg_compression'
                   ]
    
    saved_distortions = ['gaussian_noise']
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    distortion_errors = {}

    print("SEEN RESOLUTION: ", res)

    for distortion_name in distortions:

        errs = []
        scores = {}

        for severity in range(1, 6):
            print(distortion_name, str(severity))
            print("LOAD OOD: ", load_ood)

            if load_ood and distortion_name in saved_distortions:
                
                print("Looking for dataset...", '../datasets/transform_' + dataset + '_res' + str(res)  + '/' + distortion_name + '/' + str(severity))
                if os.path.exists('../datasets/transform_' + dataset + '_res' + str(res) + '/' + distortion_name + '/' + str(severity)):

                    valdir = os.path.join('../datasets/transform_' + dataset + '_res' + str(res), distortion_name + '/' + str(severity))
                    val_set = datasets.ImageFolder(valdir, transforms.Compose([
                            #transforms.Resize((res, res)),
                            transforms.ToTensor(),
                            normalize,
                        ]))
                
                else: 

                    print("error: no saved images...")

            else:

                t=transforms.Compose([
                        transforms.Resize((res, res)),
                        ApplyDistortion(distortion_name, severity), # Apply the distortion function
                        transforms.ToTensor(),
                        normalize,
                    ])
                
                if dataset=='cifar10':
                    val_set = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=t)
                elif dataset=='cifar100':
                    val_set = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=True, transform=t)

            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
            
            print("VAL LOADER: ", len(val_loader))

            # evaluate on validation set
            acc1 = validate(val_loader, model, device)
            errs.append(100. - acc1)
            scores[severity] = np.round(errs[-1],2) # 100 - acc1

        print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 1. * np.mean(errs)))
        scores['mean'] = np.round(np.mean(errs),2)
        distortion_errors[distortion_name] = scores

        if save_stats:
            file_path = 'cifarc_errors_' + model_name + '.json'
            # Open the file in write mode
            with open(file_path, 'w') as file:
                # Serialize the dictionary and write it to the file
                json.dump(distortion_errors, file)

    #np.save('imagenetc_errors_' + model_name + '.npy', np.array(distortion_errors))

    return distortion_errors

def load_model(model_name):
    "Loads one of the pretrained models."
    if model_name in ['resnext101_32x8d_wsl', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl',
                      'resnext101_32x48d_wsl']:
        model = torch.hub.load('facebookresearch/WSL-Images', model_name)
    elif model_name == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    elif model_name == 'alexnet':
      model = torchvision.models.alexnet(pretrained=True)
    else:
        raise ValueError('Model not available.')

    model = torch.nn.DataParallel(model).cuda()
    print('Loaded model:', model_name)

    return model

if __name__ == "__main__":

    args = parser.parse_args()

    #load_dataset('../datasets/CIFAR-10-C/labels.npy')

    supernet_path = args.supernet_path
    if args.model_path is not None:
        model_path = args.model_path
    print("Model path: ", model_path)
    config = json.load(open(args.model_path))

    ofa = OFAEvaluator(n_classes=args.n_classes,
    model_path = supernet_path,
    pretrained = True)
    r=config.get("r",32)
    input_shape = (3,r,r)
    model, _ = ofa.sample(config)

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) # cifar10/100
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if os.path.exists('checkpoint.pth'):
        model, optimizer = load_checkpoint(model, optimizer)
    else: 
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    normalize,
                ])
        trainset = datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

        valset = datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform)
        test_loader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        n_epochs = 20
        train(train_loader, test_loader, n_epochs, model, criterion, optimizer)

        distortion_errors = compute_mCE(args.data, args.model_name, model)