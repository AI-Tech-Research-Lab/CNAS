import json
import logging
import os
import warnings
from copy import deepcopy
from itertools import chain
import datetime
import copy
import argparse
import numpy as np
import torch
from torchvision.datasets import ImageFolder

import sys
sys.path.append(os.getcwd())

from evaluators import binary_eval, entropy_eval, standard_eval, \
    branches_eval, binary_statistics, binary_statistics_cumulative, ece_score
from trainer import binary_bernulli_trainer, joint_trainer, \
    standard_trainer
from train_utils import get_data_loaders, get_optimizer, get_lr_scheduler, initialize_seed, train, validate, load_checkpoint, Log
from utils_ee import get_intermediate_backbone_cost, get_intermediate_classifiers_cost, get_ee_scores, get_subnet_folder_by_backbone, get_net_info, get_eenn
from utils import get_net_from_OFA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mobilenetv3', help='name of the model (mobilenetv3, ...)')
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    #parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="Base learning rate at the start of the training.") #0.1
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0004, type=float, help="L2 weight decay.")
    parser.add_argument('--use_val', action='store_true', default=False, help='use validation set')
    parser.add_argument('--optim', type=str, default='SGD', help='algorithm to use for training')
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument('--dataset', type=str, default='imagenet', help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument("--data_aug", default=True, type=bool, help="True if you want to use data augmentation.")
    parser.add_argument('--save', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for training / testing')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0', help='file path to supernet weights')
    parser.add_argument('--model_path', type=str, default=None, help='file path to subnet')
    parser.add_argument('--output_path', type=str, default=None, help='file path to save results')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained weights')
    parser.add_argument('--mmax', type=float, default=1000, help='maximum number of MACS allowed')
    parser.add_argument('--top1min', type=float, default=0.0, help='minimum top1 accuracy allowed')
    parser.add_argument("--use_early_stopping", default=True, type=bool, help="True if you want to use early stopping.")
    parser.add_argument("--early_stopping_tolerance", default=5, type=int, help="Number of epochs to wait before early stopping.")

    #method: bernulli
    parser.add_argument("--method", type=str, default='bernulli', help="Method to use for training: bernulli or joint")
    parser.add_argument("--fix_last_layer", default=True, action='store_true', help="True if you want to fix the last layer of the backbone.")
    parser.add_argument("--gg_on", default=False, action='store_true', help="True if you want to use the global gate.")
    parser.add_argument("--load_backbone_from_archive", default=False, action='store_true', help="True if you want to use a pre-trained backbone from archive")
    parser.add_argument('--eval_test', action='store_true', default=True, help='evaluate test accuracy')
    parser.add_argument("--backbone_epochs", default=5, type=int, help="Number of epochs to train the backbone.")
    parser.add_argument("--warmup_ee_epochs", default=2, type=int, help="Number of epochs to warmup the EENN")
    parser.add_argument("--ee_epochs", default=3, type=int, help="Number of epochs to train the EENN")
    parser.add_argument("--priors", default=0.5, type=float, help="Prior probability for the Bernoulli distribution.")
    parser.add_argument("--joint_type", default='logits', type=str, help="Type of joint training: logits, predictions or losses.")
    parser.add_argument("--beta", default=1, type=float, help="Beta parameter for the Bernoulli distribution.")
    parser.add_argument("--sample", default=False, type=bool, help="True if you want to sample from the Bernoulli distribution.")
    #parser.add_argument("--recursive", default=True, type=bool, help="True if you want to use recursive training.") #not used
    parser.add_argument("--normalize_weights", default=True, type=bool, help="True if you want to normalize the weights.")
    parser.add_argument("--prior_mode", default='ones', type=str, help="Mode for the prior: ones or zeros.")
    parser.add_argument("--regularization_loss", default='bce', type=str, help="Loss for the regularization.")
    parser.add_argument("--temperature_scaling", default=True, type=bool, help="True if you want to use temperature scaling.")
    parser.add_argument("--regularization_scaling", default=False, type=bool, help="True if you want to use regularization scaling.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability.")
    parser.add_argument("--support_set", default=False, type=bool, help="True if you want to use the support set.")
    parser.add_argument("--w_alpha", default=1.0, type=float, help="Weight for the accuracy loss.")
    parser.add_argument("--w_beta", default=0.0, type=float, help="Weight for the MACs loss.")
    parser.add_argument("--w_gamma", default=0.0, type=float, help="Weight for the calibration loss.")
    parser.add_argument("--train_weights", default=True, action='store_true', help="True if you want to train the weights.")

    args = parser.parse_args()

    #log = logging.getLogger(__name__)
    #log.info(" message ")

    device = args.device
    print("torch cuda available: ", torch.cuda.is_available())
    use_cuda=False
    if torch.cuda.is_available() and device != 'cpu':
        device = 'cuda:{}'.format(device)
        print("Running on GPU")
        use_cuda=True
    else:
        print("No device found")
        warnings.warn("Device not found or CUDA not available.")
    
    device = torch.device(device)
    initialize_seed(42, use_cuda)

    supernet_path = args.supernet_path
    if args.model_path is not None:
        model_path = args.model_path
    print("Model path: ", model_path)

    if args.method == 'bernulli':
        get_binaries = True
    else:
        get_binaries = False

    early_stopping = None

    fix_last_layer = False
    if get_binaries:
        fix_last_layer = args.fix_last_layer
    
    backbone, res = get_net_from_OFA(subnet_path=args.model_path, 
                                supernet=args.supernet_path, 
                                n_classes=args.n_classes, 
                                pretrained=args.pretrained)
    
    print("Resolution: ", res)

    train_loader, val_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=args.threads, 
                                            use_val=args.use_val, img_size=res, augmentation=True, eval_test=args.eval_test)
    
    if val_loader is None:
        val_loader = test_loader

    train_log = Log(log_each=10)

    #parameters = chain(backbone.parameters(), classifiers.parameters())

    optimizer = get_optimizer(backbone.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay)
    
    scheduler = get_lr_scheduler(optimizer, 'step', epochs=args.backbone_epochs)

    if (os.path.exists(os.path.join(args.output_path,'backbone.pt'))):

        backbone, optimizer = load_checkpoint(backbone, optimizer, os.path.join(args.output_path,'backbone.pt'))
        print("Loaded checkpoint")
        top1 = validate(val_loader, backbone, device, print_freq=100)
        print("Loaded model accuracy:", np.round(top1,2))
        top1/=100

    else:
        
        top1, backbone, optimizer = train(train_loader, val_loader, args.backbone_epochs, backbone, device, optimizer, scheduler, train_log, ckpt_path=args.output_path)

    #Create the EENN on top of the trained backbone

    backbone, classifiers = get_eenn(subnet=backbone, subnet_path=args.model_path, res=res, n_classes=args.n_classes, get_binaries=get_binaries)

    # MODEL COST PROFILING

    input_size = (3, res, res)
    
    net = copy.deepcopy(backbone)
    if args.model == 'mobilenetv3':
        net.exit_idxs=[net.exit_idxs[-1]] #take only the final exit
        b_params, b_macs = get_intermediate_backbone_cost(backbone, input_size)
    else:
        dict_macs = net.computational_cost(torch.randn((1, 3, 32, 32)))
        b_macs = []
        for m in dict_macs.values():
                b_macs.append(m/1e6)
        b_params=[] #Not implemented
        
    c_params, c_macs = get_intermediate_classifiers_cost(backbone, classifiers, input_size)

    results = {}

    results['classifiers_params'] = c_params
    results['backbone_params_i'] = b_params
    results['classifiers_macs'] = c_macs
    results['backbone_macs_i'] = b_macs

    print("Backbone MACS: ", b_macs)
    print("Classifiers MACS: ", c_macs)  
    print("Backbone params: ", b_params)
    print("Classifiers params: ", c_params)
      
    # GLOBAL GATE to switch on/off the EECs (not used)
    if(args.gg_on):
        print("Training with global gate")
    else: 
        print("Training without global gate")
    
    results = {}

    '''
    if args.load_backbone_from_archive: 
  
        iter_path = path.rsplit("/",1)[0] 
        
        #CHECK BACKBONE IN ARCHIVE
        arch = json.load(open(os.path.join(path,'net_'+str(n_subnet)+'.subnet')))
        arch_b={'ks':arch['ks'],'e':arch['e'],'d':arch['d']}
        backbone_dir=get_subnet_folder_by_backbone(iter_path,arch_b,n_subnet)

        if backbone_dir is None:
            pre_trained_model_path = os.path.join(path, 'bb_s.pt')
            pre_trained_classifier_path = os.path.join(path, 'c_s.pt')
            backbone_dir = path
        else:
            print("LOADED BACKBONE FROM " + backbone_dir)
    '''

    test_scores = standard_eval(model=backbone,
                                dataset_loader=test_loader,
                                classifier=classifiers[
                                    -1])
    
    results['backbone_top1'] = test_scores * 100

    print('Pre trained model scores : {}, {}'.format(-1,test_scores))
    
    if os.path.exists(os.path.join(args.output_path, 'bb.pt')): # and load:
        
        #log.info('Model loaded')

        backbone.to(device)
        classifiers.to(device)

        backbone.load_state_dict(torch.load(
            os.path.join(path, 'bb.pt'), map_location=device))

        loaded_state_dict = torch.load(os.path.join(
            path, 'classifiers.pt'), map_location=device)

        # old code compatibility
        loaded_state_dict = {k: v for k, v in
                                loaded_state_dict.items()
                                if 'binary_classifier' not in k}

        classifiers.load_state_dict(loaded_state_dict)

        stats_ece = ece_score(model=backbone,predictors=classifiers, dataset_loader=testloader)
        ece_scores={}
        for i,k in enumerate(stats_ece):
            scores = stats_ece[i]
            ece_scores[i]=scores[0]
        results['ece_scores']=ece_scores
        #pre_trained_path = os.path.join('~/branch_models/','{}'.format(dataset_name),'{}'.format(model_name))
        # Construct the file path using the same format as when the file was saved
        save_path = os.path.join(path, 'net_{}.stats'.format(n_subnet))

        # Load the JSON data from the file
        with open(save_path, 'r') as handle:
            json_data = json.load(handle)

        # Access the "support_conf" field directly
        support_conf = json_data["support_conf"]
        sigma = json_data["global_gate"]

        # Now you can use the 'support_conf_value' variable, which contains the value of "support_conf"
        #print("Support Confidence:", support_conf)
        
    else:

        backbone.to(device)
        classifiers.to(device)

        parameters = chain(backbone.parameters(),
                            classifiers.parameters())

        optimizer = get_optimizer(parameters, args.optim, args.learning_rate, args.momentum, args.weight_decay)

        # load weights from previous optimizer

        if args.method == 'bernulli':

            epochs = args.warmup_ee_epochs + args.ee_epochs # Total number of epochs
            res = binary_bernulli_trainer(model=backbone,
                                            predictors=classifiers,
                                            optimizer=optimizer,
                                            train_loader=train_loader,
                                            epochs=epochs,
                                            prior_parameters=args.priors,
                                            ckpt_path=None,
                                            joint_type=args.joint_type,
                                            beta=args.beta,
                                            sample=args.sample,
                                            prior_mode=args.prior_mode,
                                            eval_loader=val_loader,
                                            #recursive=args.recursive,
                                            test_loader=test_loader,
                                            fix_last_layer=fix_last_layer,
                                            normalize_weights=
                                            args.normalize_weights,
                                            temperature_scaling=
                                            args.temperature_scaling,
                                            regularization_loss=
                                            args.regularization_loss,
                                            regularization_scaling=
                                            args.regularization_scaling,
                                            dropout=args.dropout,
                                            #backbone_epochs=backbone_epochs,
                                            early_stopping=early_stopping,
                                            gg_on=args.gg_on,
                                            support_set=args.support_set,
                                            mmax = args.mmax,
                                            w_alpha=args.w_alpha,
                                            w_beta=args.w_beta,
                                            w_gamma=args.w_gamma,
                                            n_epoch_gamma=args.ee_epochs,
                                            n_classes=args.n_classes
                                            )[0]

            backbone_dict, classifiers_dict, support_conf, global_gate = res
            if support_conf is not None:
                support_conf = torch.mean(support_conf, dim=0).tolist() # compute the average on the n_classes dimension
            sigma=torch.nn.Sigmoid()(global_gate).tolist()

            backbone.load_state_dict(backbone_dict)
            classifiers.load_state_dict(classifiers_dict)

        elif args.method == 'joint':
            weights = torch.tensor([1.0] * backbone.n_branches(), device=device)

            if args.train_weights:
                #weights = torch.tensor(weights, device=device, dtype=torch.float)

                if args.joint_type == 'predictions':
                    weights = weights.unsqueeze(-1)
                    weights = weights.unsqueeze(-1)

                weights = torch.nn.Parameter(weights)
                parameters = chain(backbone.parameters(),
                                    classifiers.parameters(),
                                    [weights])

                optimizer = get_optimizer(parameters=parameters,
                                            name=args.optim,
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

            res = joint_trainer(model=backbone, predictors=classifiers,
                                optimizer=args.optim,
                                weights=weights, train_loader=train_loader,
                                epochs=args.ee_epochs,
                                scheduler=None, joint_type=args.joint_type,
                                test_loader=test_loader,
                                eval_loader=val_loader,
                                early_stopping=early_stopping)[0]

            backbone_dict, classifiers_dict = res

            backbone.load_state_dict(backbone_dict)
            classifiers.load_state_dict(classifiers_dict)

        elif method_name == 'standard':

            res = standard_trainer(model=backbone,
                                    predictors=classifiers,
                                    optimizer=args.optimizer,
                                    train_loader=train_loader,
                                    epochs=args.ee_epochs,
                                    scheduler=None,
                                    test_loader=test_loader,
                                    eval_loader=val_loader,
                                    early_stopping=early_stopping)[0]

            backbone_dict, classifiers_dict = res

            backbone.load_state_dict(backbone_dict)
            classifiers.load_state_dict(classifiers_dict)

        else:
            assert False

        if args.save:
            torch.save(backbone.state_dict(), os.path.join(path,
                                                            'bb.pt'))
            torch.save(classifiers.state_dict(),
                        os.path.join(path,
                                    'classifiers.pt'))


    train_scores = standard_eval(model=backbone,
                                    dataset_loader=trainloader,
                                    classifier=classifiers[-1])

    test_scores = standard_eval(model=backbone,
                                dataset_loader=testloader,
                                classifier=classifiers[-1])

    log.info('Last layer train and test scores : {}, {}'.format(train_scores,test_scores))

    if method_name != 'standard':

        results['support_conf']=support_conf#.tolist()
        results['global_gate']=sigma#.tolist()

    if 'bernulli' in method_name:

        ## ECE SCORE ##

        stats_ece = ece_score(model=backbone,predictors=classifiers, dataset_loader=test_loader)
        ece_scores={}
        for i,k in enumerate(stats_ece):
            ece_scores[i]=k[0]
        results['ece_scores']=ece_scores

        ## TUNING THRESHOLDS ##
        
        cumulative_threshold_scores = {}

        best_scores = {}
        best_score=0.0
        best_epsilon=0.1
        best_counters=[0]*backbone.n_branches()
        best_cumulative=True

        for epsilon in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]:#, 0.9, 0.95, 0.98]:
            a, b = binary_eval(model=backbone,
                                dataset_loader=test_loader,
                                predictors=classifiers,
                                epsilon=[
                                            0.7 if epsilon <= 0.7 else epsilon] +
                                        [epsilon] *
                                        (backbone.n_branches() - 1),
                                # epsilon=[epsilon] *
                                #         (backbone.n_branches()),
                                cumulative_threshold=True,
                                sample=False)

            a, b = dict(a), dict(b)

            # log.info('Epsilon {} scores: {}, {}'.format(epsilon,
            #                                             dict(a), dict(b)))

            s = '\tCumulative binary {}. '.format(epsilon)

            for k in sorted([k for k in a.keys() if k != 'global']):
                s += 'Branch {}, score: {}, counter: {}. '.format(k,
                                                                    np.round(
                                                                        a[
                                                                            k] * 100,
                                                                        2),
                                                                    b[k])
            s += 'Global score: {}'.format(a['global'])
            #log.info(s)

            cumulative_threshold_scores[epsilon] = {'scores': a,
                                                    'counters': b}
            
            if(a['global']>best_score):
                best_score=a['global']
                best_epsilon=epsilon
                best_counters=b
                best_scores=a
                print("New best threshold: {}".format(best_epsilon))
                print("New best score: {}".format(best_score))
        

        #results['cumulative_results'] = cumulative_threshold_scores

    n_samples = len(test_set)
    weights = []
    for ex in best_counters.values():
            weights.append(ex/n_samples)
    
    
    # For each b-th exit the avg_macs is the percentage of samples exiting from the exit 
    # multiplied by the sum of the MACs of the backbone up to the b-th exit + MACs of the b-th exit 

    #print("INFO GET_AVG_MACS")
    #print(weights)
    #print(backbone_macs_i)
    #print(classifiers_macs)

    avg_macs = 0
    for b in range(backbone.b):
        avg_macs += weights[b] * (b_macs[b] + c_macs[b])

    # Repair action: adjust the thresholds to make the network fit in terms of MACs
    constraint_compl = mmax
    constraint_acc = top1min
    i=backbone.b-2#cycle from the second last elem
    repaired = False
    epsilon=[ 0.7 if best_epsilon <= 0.7 else best_epsilon] + [best_epsilon] * (backbone.n_branches() - 1)
    best_epsilon = epsilon
    if(a['global']>=constraint_acc):
        while (i>=0 and avg_macs>constraint_compl): #cycle from the second last elem
            #print("CONSTRAINT MACS VIOLATED: REPAIR ACTION ON BRANCH {}".format(i))
            epsilon[i] = epsilon[i] - 0.1 
            a, b = binary_eval(model=backbone,
                                dataset_loader=testloader,
                                predictors=classifiers,
                                epsilon=epsilon,
                                # epsilon=[epsilon] *
                                #         (backbone.n_branches()),
                                cumulative_threshold=True,
                                sample=False)
            a, b = dict(a), dict(b)
            if(a['global']<constraint_acc):
                #print("ACC VIOLATED")
                #print(a['global'])
                if i>=1:
                    i=i-1
                    continue
                else:
                    break
            best_epsilon = epsilon
            #print("Evaluating config {}".format(str(epsilon)))
            n_samples = len(test_set) #len of cifar10 eval dataset
            weights = []
            for ex in b.values():
                    weights.append(ex/n_samples)
            avg_macs = 0
            for b in range(backbone.b):
                avg_macs += weights[b] * (b_macs[b] + c_macs[b])
            best_scores=a
            
            if(avg_macs<=constraint_compl):
                repaired=True
                break
            if(epsilon[i]<=0.11):
                i=i-1   
    
    #print("Solution repaired: {}".format(repaired))
    results["exits_ratio"]=weights
    #results['backbone_macs_i'] = b_macs
    results['avg_macs'] = avg_macs
    results['epsilon'] = best_epsilon#.tolist()
    results['cumulative_threshold'] = best_cumulative

    #The branch score of the binary_eval is the percentage of samples of the dataset EXITING 
    #FROM THAT BRANCH correctly classified by the the branch
    
    results['top1'] = best_scores['global'] * 100
    results['branch_scores'] = best_scores

    #log.info('Best epsilon: {}'.format(best_epsilon))
    #log.info('Best cumulative threshold: {}'.format(best_cumulative))
    #log.info('Branches scores on exiting samples: {}'.format(best_scores))
    #log.info('Exit ratios: {}'.format(weights))
    #log.info('Average MACS: {:.2f}'.format(avg_macs))
    
    if args.save:
        save_path = os.path.join(path, 'net_{}.stats'.format(n_subnet)) # #exp__path = ..iter_x/exp_y 

        with open(save_path, 'w') as handle:
            json.dump(results, handle)
    
    # Get the current date and time
    current_time = datetime.datetime.now()

    # Print the current time
    print("Current time:", current_time)

    #log.info('#' * 100)
    
