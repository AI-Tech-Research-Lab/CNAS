import os
import sys
import json
import copy
import logging
import argparse
import numpy as np
from datetime import datetime


import torch
import torch.nn as nn
import torchvision.utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from codebase.data_providers.autoaugment import CIFAR10Policy

from evaluator import OFAEvaluator
from torchprofile import profile_macs
from codebase.networks import NSGANetV2


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, or cinic10')
parser.add_argument('--batch-size', type=int, default=96, help='batch size')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
parser.add_argument('--n_gpus', type=int, default=1, help='number of available gpus for training')
parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--save', type=str , default=None, help='folder to dump output')
parser.add_argument('--topk', type=int, default=10, help='top k checkpoints to save')
parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate a pretrained model')
# model related
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--model1-config', type=str, default=None,
                    help='location of a json file of specific model declaration')
parser.add_argument('--model2-config', type=str, default=None,
                    help='location of a json file of specific model declaration')
parser.add_argument('--initial-checkpoint1', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--initial-checkpoint2', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--drop', type=float, default=0.2,
                    help='dropout rate')
parser.add_argument('--drop-path', type=float, default=0.2, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--img-size', type=int, default=224,
                    help='input resolution (192 -> 256)')
parser.add_argument('--threshold', type=float, default=0.2,
                    help='threshold for score margin')
args = parser.parse_args()

dataset = args.dataset

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.save is not None:
    save = os.path.join(args.save,datetime.now().strftime("%Y%m%d-%H%M%S") + args.dataset + args.model + str(args.img_size) )

    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)
    print('Experiment dir : {}'.format(save))

    fh = logging.FileHandler(os.path.join(save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

device = 'cuda'

NUM_CLASSES = 100 if 'cifar100' in dataset else 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", args)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    best_acc = 0  # initiate a artificial best accuracy so far
    counter = 0
    patience = 20
    top_checkpoints = []  # initiate a list to keep track of

    # Data
    train_transform, valid_transform = _data_transforms(args)
    if dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cinic10':
        train_data = torchvision.datasets.ImageFolder(
            args.data + 'train_and_valid', transform=train_transform)
        valid_data = torchvision.datasets.ImageFolder(
            args.data + 'test', transform=valid_transform)
    else:
        raise KeyError


    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=200, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    # load model m1

    net_config = json.load(open(args.model1_config))
    m1 = NSGANetV2.build_from_config(net_config, drop_connect_rate=args.drop_path)

    NSGANetV2.reset_classifier(
        m1, last_channel=m1.classifier.in_features,
        n_classes=NUM_CLASSES, dropout_rate=args.drop)

    # load model m2
    
    net_config = json.load(open(args.model2_config))
    m2 = NSGANetV2.build_from_config(net_config, drop_connect_rate=args.drop_path)

    NSGANetV2.reset_classifier(
        m2, last_channel=m2.classifier.in_features,
        n_classes=NUM_CLASSES, dropout_rate=args.drop)
    
    # calculate #Paramaters and #FLOPS of the models
    inputs = torch.randn(1, 3, args.img_size, args.img_size)

    info_m1 = {}
    info_m1['flops'] = profile_macs(copy.deepcopy(m1), inputs) / 1e6
    info_m1['params'] = sum(p.numel() for p in m1.parameters() if p.requires_grad) / 1e6
    info_m1['net_name'] = "net_flops@{:.0f}".format(info_m1['flops'])
    logging.info('#params {:.2f}M, #flops {:.0f}M'.format(info_m1['params'], info_m1['flops']))

    info_m2 = {}
    info_m2['flops'] = profile_macs(copy.deepcopy(m2), inputs) / 1e6
    info_m2['params'] = sum(p.numel() for p in m2.parameters() if p.requires_grad) / 1e6
    info_m2['net_name'] = "net_flops@{:.0f}".format(info_m2['flops'])
    logging.info('#params {:.2f}M, #flops {:.0f}M'.format(info_m2['params'], info_m2['flops']))

    if args.n_gpus > 1:
        m1 = nn.DataParallel(m1)  # data parallel in case more than 1 gpu available
        m2 = nn.DataParallel(m2)

    n_epochs = args.epochs

    criterion = nn.CrossEntropyLoss().to(device)

    #M1 LOAD STATE

    parameters_m1 = filter(lambda p: p.requires_grad, m1.parameters())

    optimizer_m1 = optim.SGD(parameters_m1,
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    scheduler_m1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_m1, n_epochs)

    ####
    if(args.initial_checkpoint1 != ''):
    ####
      init = torch.load(args.initial_checkpoint1, map_location='cuda:0') # pretrained weigths
      m1.load_state_dict(init['state_dict'])
      optimizer_m1.load_state_dict(init['optimizer_state_dict'])
    
    #M2 LOAD STATE

    parameters_m2 = filter(lambda p: p.requires_grad, m2.parameters())

    optimizer_m2 = optim.SGD(parameters_m2,
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

    scheduler_m2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_m2, n_epochs)

    ####
    if(args.initial_checkpoint2 != ''):
    ####
      init = torch.load(args.initial_checkpoint2, map_location='cuda:0') # pretrained weigths
      m2.load_state_dict(init['state_dict'])
      optimizer_m2.load_state_dict(init['optimizer_state_dict'])
    
    m1 = m1.to(device)
    m2 = m2.to(device)


    if args.evaluate:
        adaptive_infer(valid_queue, m1, m2, criterion, args.threshold)
        #infer(valid_queue,m1,criterion)
        sys.exit(0)
    else:
        print("Not implemented!")
    
    return

# Training
def train(train_queue, net, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        # upsample by bicubic to match imagenet training size
        inputs = F.interpolate(inputs, size=args.img_size, mode='bicubic', align_corners=False)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)

    logging.info('train acc %f', 100. * correct / total)

    return train_loss/total, 100.*correct/total


def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    logging.info('valid acc %f', 100. * correct / total)

    return test_loss/total, acc

def adaptive_infer(valid_queue, m1, m2, criterion, threshold):
    m1.eval()
    m2.eval()
    test_loss = 0
    correct = 0
    total = 0
    count_m2 = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue): #bs of valid_queue set to 1
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = m1(inputs)
            
            '''
            if get_score_margin(outputs) >= threshold:
                outputs = m2(inputs)
                count_m2 += 1
            '''
            
            if (inputs.shape[0] == 1): #bs set to 1
               outputs = outputs[None,:]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    logging.info('valid acc %f', 100. * correct / total)
    logging.info('Usage percentage of model m2: %f', 100. * count_m2 / total)

    return test_loss/total, acc


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms(args):

    if 'cifar' in args.dataset:
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    elif 'cinic' in args.dataset:
        norm_mean = [0.47889522, 0.47227842, 0.43047404]
        norm_std = [0.24205776, 0.23828046, 0.25874835]
    else:
        raise KeyError

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.RandomHorizontalFlip(),
    ])

    if args.autoaugment:
        train_transform.transforms.append(CIFAR10Policy())

    train_transform.transforms.append(transforms.ToTensor())

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    train_transform.transforms.append(transforms.Normalize(norm_mean, norm_std))

    valid_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=3),  # BICUBIC interpolation
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    return train_transform, valid_transform

## ADAPTIVE MODELS ##

# Compute score margin
def get_score_margin(outputs):
    prob = F.softmax(outputs)
    top2_prob, top2_index = torch.topk(prob,2)  
    l = top2_prob.tolist()
    score_margin = l[0] - l[1]
    return score_margin

if __name__ == '__main__':
    main()