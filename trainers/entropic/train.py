import argparse
import torch
import warnings

import sys
import os
import json
import numpy as np
import math
import copy

sys.path.append(os.getcwd())
sys.path.append('/home/gambella/EDANAS_CBN/trainers/entropic')

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
#from dataset.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
#from utility.step_lr import StepLR
from torch.optim.lr_scheduler import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.perturb import get_net_info_runtime

from sam import SAM

from ofa_evaluator import OFAEvaluator 
from train_utils import get_dataset, save_checkpoint, load_checkpoint, validate
#from evaluators.evaluate_cifar10c import compute_mCE #, compute_mCE2
from utils import get_net_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

    #NEW##
    parser.add_argument('--optim', type=str, default='SAM', help='algorithm to use for training')
    parser.add_argument("--sigma_min", default=0.05, type=float, help="min noise perturbation intensity")
    parser.add_argument("--sigma_max", default=0.05, type=float, help="max noise perturbation intensity")
    parser.add_argument("--sigma_step", default=0.0, type=float, help="step noise perturbation intensity")
    parser.add_argument('--device', type=str, default='cpu', help='device to use for training / testing')
    parser.add_argument('--data', type=str, default='/mnt/datastore/ILSVRC2012', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='imagenet', help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--model', type=str, default='mobilenetv3', help='name of the model (mobilenetv3, ...)')
    parser.add_argument('--n_classes', type=int, default=1000, help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0', help='file path to supernet weights')
    parser.add_argument('--model_path', type=str, default=None, help='file path to subnet')
    parser.add_argument('--output_path', type=str, default=None, help='file path to save results')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained weights')
    parser.add_argument('--ood_eval', action='store_true', default=False, help='evaluate OOD robustness')
    parser.add_argument('--save_ckpt', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--eval_robust', action='store_true', default=False, help='evaluate robustness')   
    parser.add_argument('--load_ood', action='store_true', default=False, help='load pretrained OOD folders') 
    parser.add_argument('--ood_data', type=str, default=None, help='OOD dataset')
    parser.add_argument('--alpha', default=0.5, type=float, help="weight for top1_robust")
    parser.add_argument('--res', default=32, type=int, help="default resolution for training")  
    parser.add_argument('--pmax', default=2.0, type=float, help="constraint on params")
    parser.add_argument('--p', default=0.0, type=float, help="penalty on params")
    parser.add_argument('--alpha_norm', default=1.0, type=float, help="weight for top1_robust normalization")

    args = parser.parse_args()

    initialize(args, seed=42)
    
    device = args.device
    if torch.cuda.is_available() and device != 'cpu':
        device = 'cuda:{}'.format(device)
        print("Running on GPU")
    else:
        warnings.warn("Device not found or CUDA not available.")
    
    device = torch.device(device)

    # Get the model (subnet) from the OFA supernet

    supernet_path = args.supernet_path
    if args.model_path is not None:
        model_path = args.model_path
    print("Model path: ", model_path)
    config = json.load(open(args.model_path))

    ofa = OFAEvaluator(n_classes=args.n_classes,
    model_path = supernet_path,
    pretrained = True)
    #print("CONFIG:", config)
    r=config.get("r",args.res)
    input_shape = (3,r,r)
    print("INPUT SHAPE:", input_shape)
    model, _ = ofa.sample(config)

    #dataset = Cifar(args.batch_size, args.threads, args.data)
    print(f"DATASET: {args.dataset}")
    train_set, test_set, _, _ = get_dataset(args.dataset, augmentation=True, resolution=r)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    log = Log(log_each=10)
    #model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    print("DEVICE:", device)

    model.to(device)
    epochs = args.epochs
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=epochs, gamma=1.0)
    
    if (os.path.exists(os.path.join(args.output_path,'ckpt.pth'))):

        model, optimizer = load_checkpoint(model, optimizer, os.path.join(args.output_path,'ckpt.pth'))
        print("Loaded checkpoint")
    
    else:

        # TODO: check cross-validated early stop
        # TODO: check SGD with CNAS trainer: https://github.com/matteogambella/NAS/blob/master/nsganetv2/codebase/run_manager/__init__.py
        for epoch in range(epochs):
            model.train()
            log.train(model, optimizer, len_dataset=len(train_loader))

            for batch in train_loader:
                inputs, targets = (b.to(device) for b in batch)

                # first forward-backward step
                if args.optim == "SAM":
                    enable_running_stats(model)
                elif args.optim == "SGD":
                    optimizer.zero_grad()
                    
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()

                if args.optim == "SGD":
                    optimizer.step()
                elif args.optim == "SAM":
                    optimizer.first_step(zero_grad=True)
                    # second forward-backward step
                    disable_running_stats(model)
                    smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                    optimizer.second_step(zero_grad=True)

                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets
                    log(model, loss.cpu(), correct.cpu(), scheduler.get_lr()[0])
                    #scheduler(epoch)
                    scheduler.step()

            model.eval()
            log.eval(model, optimizer, len_dataset=len(test_loader))

            with torch.no_grad():
                acc=0
                for batch in test_loader:
                    inputs, targets = (b.to(device) for b in batch)
                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    log(model, loss.cpu(), correct.cpu())
                    acc+=correct.sum().item()
                acc/=len(test_loader.dataset)
                if acc > log.best_accuracy:
                    best_model = copy.deepcopy({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()})

        log.flush(model, optimizer)
        model.load_state_dict(best_model['state_dict']) # load best model for inference 
        optimizer.load_state_dict(best_model['optimizer']) # load optim for further training 
        
        if (args.save_ckpt):
            save_checkpoint(model, optimizer, os.path.join(args.output_path,'ckpt.pth'))

    results={}

    if args.ood_eval:
        
        #results['mCE'] = compute_mCE_CIFARC(args.ood_data, model, device, res=args.res)
        results['mCE2'] = compute_mCE(args.dataset, model, device, res=args.res, load_ood=args.load_ood)
    
    #DICTIONARY for stats
    top1 = (1 - log.best_accuracy) * 100

    #Model cost
    if args.optim == 'SAM' or args.eval_robust:
        sigma_step = args.sigma_step
        if args.sigma_max == args.sigma_min:
            sigma_step = 1
        n=round((args.sigma_max-args.sigma_min)/sigma_step)+1
        sigma_list = [round(args.sigma_min + i * args.sigma_step, 2) for i in range(n)] 

        info = get_net_info_runtime(device, model, train_loader, sigma_list, input_shape, print_info=True)
        results['robustness'] = info['robustness'][0]
        print("ROBUSTNESS:", info['robustness'][0])
        alpha = args.alpha
        alpha_norm = args.alpha_norm
        results['top1_robust'] = np.round(alpha * top1 + alpha_norm * (1-alpha) * info['robustness'][0],2)

    else:
        info = get_net_info(model, input_shape=input_shape)

    results['top1'] = np.round(top1,2)
    print("FINAL best acc", log.best_accuracy * 100, "\n")
    results['macs'] = np.round(info['macs'] / 1e6, 2)
    results['params'] = np.round(info['params'] / 1e6, 2)
    results['c_params'] = results['params'] + args.p*max(0,results['params']-args.pmax)

    n_subnet = args.output_path.rsplit("_", 1)[1] 
    
    save_path = os.path.join(args.output_path, 'net_{}.stats'.format(n_subnet)) 

    with open(save_path, 'w') as handle:
        json.dump(results, handle)
