import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from src.warmup_scheduler import GradualWarmupScheduler

import json

from entropic.models.mlp import perceptron, committee, MLP
from entropic.models.cnn import CNN

from entropic.optimizers.sam import SAM

from evaluator import OFAEvaluator

def get_loss(args):

    if args.loss == "xent":
        return nn.CrossEntropyLoss()
    elif args.loss == "mse":
        def MSE_classification(input, target): # attention: MSE for classification, output sigmoids and 1-hot encoding
            return F.mse_loss(torch.sigmoid(input), F.one_hot(target, num_classes=args.nclasses).float()) 
        return MSE_classification
    elif args.loss == "binxent":
        return nn.BCEWithLogitsLoss() # check documentation. 0,1 or -1,+1 labels?
    elif args.loss == "binmse":
        def MSE_classification(input, target): # attention: MSE for binary classification, output sigmoid
            return F.mse_loss(torch.sigmoid(input), target) 

def get_optimizer(model, args):

    if args.opt == "sgd":
        nesterov = False if args.opt == "sgd" else True
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, nesterov=nesterov, weight_decay=args.wd)
    elif args.opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == "sam":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, momentum=args.mom)
        return optimizer
    elif args.opt == "rsgd":
        return NameError()
    elif args.opt == "esgd":
        return NameError()

def get_model(args):

    if args.model == "perceptron":
        return perceptron(args.N, args.nclasses, bias=args.bias)
    elif args.model == "committee":
        return committee(args.N, args.K, args.nclasses, bias=args.bias) # TODO
    elif args.model == "mlp":
        return MLP(args.N, args.K, args.nclasses, bias=args.bias)
    elif args.model == "cnn":
        return CNN()
    elif args.model.startswith("ofa_mbv3"):
        return OFA_MBV3(args)
    elif args.model.startswith("vgg"):
        return NameError()
    elif args.model.startswith("resnet"):
        return NameError()

def OFA_MBV3(args):
    #supernet = './../../../../EDANAS_CBN/ofa_nets/ofa_cbnmbv3'
    config = json.load(open(args.subnet_path))

    ofa = OFAEvaluator(n_classes=args.nclasses,
    model_path=args.supernet,
    pretrained = args.pretrained)
    subnet, _ = ofa.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d'], 'r': config['r']})
    return subnet

def get_lr_scheduler(optimizer, args):
    if args.droplr:
        if args.droplr == 'cosine':
            print("Cosine lr schedule")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0., last_epoch=-1)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        elif args.droplr < 1. and args.droplr != 0.:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.droplr)
        else:
            gamma_sched = 1. / args.droplr if args.droplr > 0 else 1
            #if args.drop_mstones is not None:
            #    mstones = [int(h) for h in args.drop_mstones.split('_')[1:]]
            #else:
            mstones = [args.epochs//2, args.epochs*3//4, args.epochs*15//16]
            print("MileStones %s" % mstones)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mstones, gamma=gamma_sched)
    else:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

    #if args.warmup:
    #    scheduler = GradualWarmupScheduler(optimizer, multiplier=args.k, total_epoch=5, 
    #                                    after_scheduler=scheduler)
                                        
    return scheduler