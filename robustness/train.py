import argparse
import torch
import warnings

import sys
import os
import json
import numpy as np
import math
import copy
import logging
import datetime

sys.path.append(os.getcwd())

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler, DataLoader
from utility.perturb import get_net_info_runtime

from ofa_evaluator import OFAEvaluator 
from train_utils import get_dataset, get_optimizer, get_loss, get_lr_scheduler, get_data_loaders, save_checkpoint, load_checkpoint, validate, initialize_seed, Log, \
      smooth_crossentropy, train
from evaluate_cifar10c import compute_mCE 
from utils import get_net_info, get_net_from_OFA

#--trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000
#init_lr=0.01, lr_schedule_type='cosine' weight_decay=4e-5, label_smoothing=0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--nesterov", default=False, type=bool, help="True if you want to use Nesterov momentum.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.") #0.1
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="L2 weight decay.") #4e-5
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument('--use_val', action='store_true', default=False, help='use validation set')
    parser.add_argument('--save', action='store_true', default=True, help='save log of experiments')
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
    parser.add_argument('--eval_test', action='store_true', default=True, help='evaluate test accuracy')  
    parser.add_argument('--load_ood', action='store_true', default=False, help='load pretrained OOD folders') 
    parser.add_argument('--ood_data', type=str, default=None, help='OOD dataset')
    parser.add_argument('--alpha', default=0.5, type=float, help="weight for top1_robust")
    parser.add_argument('--res', default=32, type=int, help="default resolution for training")  
    parser.add_argument('--pmax', default=2.0, type=float, help="constraint on params")
    parser.add_argument('--p', default=0.0, type=float, help="penalty on params")
    parser.add_argument('--alpha_norm', default=1.0, type=float, help="weight for top1_robust normalization")

    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    logging.info('Experiment dir : {}'.format(args.output_path))

    fh = logging.FileHandler(os.path.join(args.output_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    device = args.device
    use_cuda=False
    if torch.cuda.is_available() and device != 'cpu':
        device = 'cuda:{}'.format(device)
        logging.info("Running on GPU")
        use_cuda=True
    else:
        logging.info("No device found")
        logging.warning("Device not found or CUDA not available.")
    
    device = torch.device(device)
    initialize_seed(42, use_cuda)

    supernet_path = args.supernet_path
    if args.model_path is not None:
        model_path = args.model_path
    logging.info("Model: %s", args.model)

    model, res = get_net_from_OFA(model_path, args.n_classes, args.supernet_path, pretrained=True)

    logging.info(f"DATASET: {args.dataset}")
    logging.info("Resolution: %s", res)
    train_loader, val_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=args.threads, 
                                            use_val=args.use_val, img_size=res, augmentation=True, eval_test=args.eval_test)
    
    if val_loader is None:
        val_loader = test_loader

    log = Log(log_each=10)

    model.to(device)
    epochs = args.epochs

    optimizer = get_optimizer(model.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay, args.rho, args.adaptive, args.nesterov)

    criterion = get_loss('ce')
    
    scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=epochs)
    
    if (os.path.exists(os.path.join(args.output_path,'ckpt.pth'))):

        model, optimizer = load_checkpoint(model, optimizer, os.path.join(args.output_path,'ckpt.pth'))
        logging.info("Loaded checkpoint")
        top1 = validate(val_loader, model, device, print_freq=100)/100

    else:
        logging.info("Start training...")
        top1, model, optimizer = train(train_loader, val_loader, epochs, model, device, optimizer, criterion, scheduler, log, ckpt_path=os.path.join(args.output_path,'ckpt.pth'))
        logging.info("Training finished")

    results={}
    logging.info(f"VAL ACCURACY: {np.round(top1*100,2)}")
    top1_err = (1 - top1) * 100
    if args.eval_test:
        top1_test = validate(test_loader, model, device, print_freq=100)
        logging.info(f"TEST ACCURACY: {top1_test}")

    if args.ood_eval:
        
        #results['mCE'] = compute_mCE_CIFARC(args.ood_data, model, device, res=args.res)
        results['mCE2'] = compute_mCE(args.dataset, model, device, res=args.res, load_ood=args.load_ood)

    input_shape = (3, res, res)
    #Model cost
    if args.optim == 'SAM' or args.eval_robust:
        sigma_step = args.sigma_step
        if args.sigma_max == args.sigma_min:
            sigma_step = 1
        n=round((args.sigma_max-args.sigma_min)/sigma_step)+1
        sigma_list = [round(args.sigma_min + i * args.sigma_step, 2) for i in range(n)] 

        info = get_net_info_runtime(device, model, val_loader, sigma_list, input_shape, print_info=True)
        results['robustness'] = info['robustness'][0]
        logging.info(f"ROBUSTNESS: {info['robustness'][0]}")
        alpha = args.alpha
        alpha_norm = args.alpha_norm
        results['top1_robust'] = np.round(alpha * top1_err + alpha_norm * (1-alpha) * info['robustness'][0],2)

    else:
        info = get_net_info(model, input_shape=input_shape)

    results['top1'] = np.round(top1_err,2)
    results['macs'] = np.round(info['macs'] / 1e6, 2)
    results['params'] = np.round(info['params'] / 1e6, 2)
    results['c_params'] = results['params'] + args.p*max(0,results['params']-args.pmax)

    n_subnet = args.output_path.rsplit("_", 1)[1] 
    
    save_path = os.path.join(args.output_path, 'net_{}.stats'.format(n_subnet)) 

    with open(save_path, 'w') as handle:
        json.dump(results, handle)
    