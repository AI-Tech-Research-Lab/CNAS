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


from evaluator import OFAEvaluator
from torchprofile import profile_macs
from codebase.networks import NSGANetV2


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Wood Training')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wood', help='cifar10, cifar100, or cinic10')
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
parser.add_argument('--model-config', type=str, default=None,
                    help='location of a json file of specific model declaration')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--drop', type=float, default=0.2,
                    help='dropout rate')
parser.add_argument('--drop-path', type=float, default=0.2, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--img-size', type=int, default=224,
                    help='input resolution (192 -> 256)')
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

NUM_CLASSES = 5 


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", args)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    best_acc = 0  # initiate a artificial best accuracy so far
    top_checkpoints = []  # initiate a list to keep track of

    # Data
    train_transform, valid_transform = _data_transforms(args)
    train_data = torchvision.datasets.ImageFolder(
            args.data + 'train', transform=train_transform)
    valid_data = torchvision.datasets.ImageFolder(
            args.data + 'valid', transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    net_config = json.load(open(args.model_config))
    net = NSGANetV2.build_from_config(net_config, drop_connect_rate=args.drop_path)
    
    ####
    if(args.initial_checkpoint != ''):
    ####
       init = torch.load(args.initial_checkpoint, map_location='cpu')['state_dict']
       net.load_state_dict(init)

    NSGANetV2.reset_classifier(
        net, last_channel=net.classifier.in_features,
        n_classes=NUM_CLASSES, dropout_rate=args.drop)

    # calculate #Paramaters and #FLOPS
    inputs = torch.randn(1, 3, args.img_size, args.img_size)
    flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
    params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    net_name = "net_flops@{:.0f}".format(flops)
    logging.info('#params {:.2f}M, #flops {:.0f}M'.format(params, flops))

    if args.n_gpus > 1:
        net = nn.DataParallel(net)  # data parallel in case more than 1 gpu available

    net = net.to(device)

    n_epochs = args.epochs

    parameters = filter(lambda p: p.requires_grad, net.parameters())

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(parameters,
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    if args.evaluate:
        infer(valid_queue, net, criterion)
        sys.exit(0)

    for epoch in range(n_epochs):

        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

        train(train_queue, net, criterion, optimizer)
        _, valid_acc = infer(valid_queue, net, criterion)

        # checkpoint saving
        if save:
            if len(top_checkpoints) < args.topk:
                OFAEvaluator.save_net(save, net, net_name+'.ckpt{}'.format(epoch))
                top_checkpoints.append((os.path.join(save, net_name+'.ckpt{}'.format(epoch)), valid_acc))
            else:
                idx = np.argmin([x[1] for x in top_checkpoints])
                if valid_acc > top_checkpoints[idx][1]:
                    OFAEvaluator.save_net(save, net, net_name + '.ckpt{}'.format(epoch))
                    top_checkpoints.append((os.path.join(save, net_name+'.ckpt{}'.format(epoch)), valid_acc))
                    # remove the idx
                    os.remove(top_checkpoints[idx][0])
                    top_checkpoints.pop(idx)
                    print(top_checkpoints)

            if valid_acc > best_acc:
                OFAEvaluator.save_net(save, net, net_name + '.best')
                best_acc = valid_acc

        scheduler.step()

    OFAEvaluator.save_net_config(save, net, net_name+'.config')


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

    
def normalize():
        return transforms.Normalize(
            mean= [0.5562666666666667, 0.4514, 0.38713333333333333],
            std= [0.32226666666666665, 0.23883333333333334, 0.22890000000000002])

def build_train_transform(image_size=None, print_log=True, auto_augment='rand-m9-mstd0.5'):
         if image_size is None:
            image_size = args.image_size
         return transforms.Compose([           
              #transforms.CenterCrop(image_size),
              transforms.Resize(image_size),
              transforms.ToTensor(),
              normalize(),
          ])

def build_valid_transform(image_size=None):
        if image_size is None:
            image_size = args.img_size
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize(),
        ])

def _data_transforms(args):

    train_transform = build_train_transform()
    valid_transform = build_valid_transform()

    return train_transform, valid_transform


if __name__ == '__main__':
    main()
