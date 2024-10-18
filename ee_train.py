import json
import logging
import os
from itertools import chain
import copy
import argparse
import numpy as np
from EarlyExits.models.efficientnet import EEEfficientNet
import torch
import torch.nn as nn

import sys
sys.path.append(os.getcwd())

from train_utils import get_data_loaders, get_optimizer, get_loss, get_lr_scheduler, initialize_seed, train, validate, load_checkpoint, Log
from utils import get_network_search

from EarlyExits.evaluators import sm_eval, binary_eval, standard_eval, ece_score
from EarlyExits.trainer import binary_bernulli_trainer, joint_trainer
from EarlyExits.utils_ee import get_ee_efficientnet, get_intermediate_backbone_cost, get_intermediate_classifiers_cost, get_subnet_folder_by_backbone, get_eenn
import torchvision.models as models

#--trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000
#init_lr=0.01, lr_schedule_type='cosine' weight_decay=4e-5, label_smoothing=0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mobilenetv3', help='name of the model (mobilenetv3, ...)')
    parser.add_argument('--ofa', action='store_true', default=True, help='s')
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    #parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate at the start of the training.") #0.1
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--n_workers", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="L2 weight decay.")
    parser.add_argument('--val_split', default=0.0, type=float, help='use validation set')
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
    parser.add_argument("--resolution", default=32, type=int, help="Image resolution.")
    parser.add_argument("--func_constr", action='store_true', default=False, help='use functional constraints')

    #method: bernulli
    parser.add_argument("--method", type=str, default='bernulli', help="Method to use for training: bernulli or joint")
    parser.add_argument("--fix_last_layer", default=True, action='store_true', help="True if you want to fix the last layer of the backbone.")
    parser.add_argument("--gg_on", default=False, action='store_true', help="True if you want to use the global gate.")
    parser.add_argument("--load_backbone_from_archive", default=False, action='store_true', help="True if you want to use a pre-trained backbone from archive")
    parser.add_argument('--eval_test', action='store_true', default=True, help='evaluate test accuracy')
    parser.add_argument("--backbone_epochs", default=5, type=int, help="Number of epochs to train the backbone.")
    parser.add_argument("--warmup_ee_epochs", default=2, type=int, help="Number of epochs to warmup the EENN")
    parser.add_argument("--ee_epochs", default=0, type=int, help="Number of epochs to train the EENN using the support set")
    parser.add_argument("--priors", default=0.5, type=float, help="Prior probability for the Bernoulli distribution.")
    parser.add_argument("--joint_type", default='losses', type=str, help="Type of joint training: logits, predictions or losses.")
    parser.add_argument("--beta", default=1, type=float, help="Beta parameter for the Bernoulli distribution.")
    parser.add_argument("--sample", default=False, type=bool, help="True if you want to sample from the Bernoulli distribution.")
    #parser.add_argument("--recursive", default=True, type=bool, help="True if you want to use recursive training.") #not used
    parser.add_argument("--normalize_weights", default=True, type=bool, help="True if you want to normalize the weights.")
    parser.add_argument("--prior_mode", default='ones', type=str, help="Mode for the prior: ones or zeros.")
    parser.add_argument("--regularization_loss", default='bce', type=str, help="Loss for the regularization.")
    parser.add_argument("--temperature_scaling", default=True, type=bool, help="True if you want to use temperature scaling.")
    parser.add_argument("--regularization_scaling", default=False, type=bool, help="True if you want to use regularization scaling.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability.")
    parser.add_argument("--support_set", default=False, action='store_true', help="True if you want to use the support set.")
    parser.add_argument("--w_alpha", default=1.0, type=float, help="Weight for the accuracy loss.")
    parser.add_argument("--w_beta", default=1.0, type=float, help="Weight for the MACs loss.")
    parser.add_argument("--w_gamma", default=1.0, type=float, help="Weight for the calibration loss.")
    parser.add_argument("--train_weights", default=False, action='store_true', help="True if you want to train the weights.")
    parser.add_argument("--tune_epsilon", default=False, action='store_true', help="True if you want to tune the epsilon.")

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

    if args.method == 'bernulli':
        get_binaries = True
    else:
        get_binaries = False

    early_stopping = None

    fix_last_layer = False
    if get_binaries:
        fix_last_layer = args.fix_last_layer
    
    if args.dataset=='cifar100':
        n_classes=100
    elif args.dataset=='ImageNet16':
        n_classes=120
    else:
        n_classes=10

    if 'mobilenetv3' in args.model:
        n_subnet = args.output_path.rsplit("_", 1)[1]
        save_path = os.path.join(args.output_path, 'net_{}.stats'.format(n_subnet))

        supernet_path = args.supernet_path
        if args.model_path is not None:
            model_path = args.model_path
        logging.info("Model: %s", args.model)
        
        backbone, res = get_network_search(model=args.model,
                                    subnet_path=args.model_path, 
                                    supernet=args.supernet_path, 
                                    n_classes=n_classes, 
                                    pretrained=args.pretrained,
                                    func_constr=args.func_constr)
    else:
        backbone=models.efficientnet_b0(weights='DEFAULT') #EEEfficientNet()
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # Dropout for regularization
            nn.Linear(1280, n_classes, bias=True)  # Fully connected layer
        )
        save_path = os.path.join(args.output_path, 'net.stats')
        res = args.resolution

    if res is None:
        res = args.resolution

    logging.info(f"DATASET: {args.dataset}")
    logging.info("Resolution: %s", res)
    logging.info("Number of classes: %s", n_classes)
    print("EE epochs: ", args.ee_epochs)

    train_loader, val_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=args.n_workers, 
                                            val_split=args.val_split, img_size=res, augmentation=True, eval_test=args.eval_test)
    
    if val_loader is not None:
        n_samples=len(val_loader.dataset)
    else:
        val_loader = test_loader
        n_samples=len(test_loader.dataset)

    print("Train samples: ", len(train_loader.dataset))
    print("Val samples: ", len(val_loader.dataset))

    train_log = Log(log_each=10)

    #parameters = chain(backbone.parameters(), classifiers.parameters())

    optimizer = get_optimizer(backbone.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay)

    criterion = get_loss('ce')
    
    scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=args.backbone_epochs)

    if (os.path.exists(os.path.join(args.output_path,'backbone.pth'))):

        backbone, optimizer = load_checkpoint(backbone, optimizer, device, os.path.join(args.output_path,'backbone.pth'))
        logging.info("Loaded checkpoint")
        top1 = validate(val_loader, backbone, device, print_freq=100)/100

    else:

        if args.backbone_epochs > 0:
            logging.info("Start training...")
            top1, backbone, optimizer = train(train_loader, val_loader, args.backbone_epochs, backbone, device, optimizer, criterion, scheduler, train_log, ckpt_path=os.path.join(args.output_path,'backbone.pth'))
            logging.info("Training finished")
    
    if args.backbone_epochs == 0:
        top1 = validate(val_loader, backbone, device, print_freq=100)/100
    logging.info(f"VAL ACCURACY BACKBONE: {np.round(top1*100,2)}")
    if args.eval_test:
        top1_test = validate(test_loader, backbone, device, print_freq=100)
        logging.info(f"TEST ACCURACY BACKBONE: {top1_test}")
    
    results={}
    results['backbone_top1'] = np.round((1-top1)*100,2)

    #Create the EENN on top of the trained backbone

    if 'mobilenetv3' in args.model:
        backbone, classifiers, epsilon = get_eenn(subnet=backbone, subnet_path=args.model_path, res=res, n_classes=n_classes, get_binaries=get_binaries)
    else:
        backbone, classifiers, epsilon = get_ee_efficientnet(model=backbone, img_size=res, n_classes=n_classes, get_binaries=get_binaries)

    #------------------------------
    # Added to create the onnx file of subnet found
    # backbone = model
    torch_input = torch.randn(1, 3, res, res).to(device) # device = cuda:{}. It has to run on GPU & 4D
    torch.onnx.export(backbone, torch_input, 'multi_exits_cifar10.onnx', opset_version=11)  # create onnx file
    print("onnx created!!!")
    #------------------------------

    # MODEL COST PROFILING
    input_size = (3, res, res)
    
    net = copy.deepcopy(backbone)
    if args.model == 'cbnmobilenetv3' or args.model == 'eemobilenetv3' or args.model == 'efficientnet':
        #net.exit_idxs=[net.exit_idxs[-1]] #take only the final exit
        b_params, b_macs = get_intermediate_backbone_cost(backbone, input_size)
    else:
        dict_macs = net.computational_cost(torch.randn((1, 3, res, res)))
        b_macs = []
        for m in dict_macs.values():
                b_macs.append(m/1e6)
        b_params=[] #Not implemented
        
    c_params, c_macs = get_intermediate_classifiers_cost(backbone, classifiers, input_size)

    max_cost = b_macs[-1] + c_macs[-1]

    '''
    if args.mmax is not None and max_cost < args.mmax:
        logging.warning("The maximum cost is lower than the constraint")
        sys.exit()
    '''

    results['classifiers_params'] = c_params
    results['backbone_params_i'] = b_params
    results['classifiers_macs'] = c_macs
    results['backbone_macs_i'] = b_macs

    print("Backbone MACS: ", b_macs)
    print("Classifiers MACS: ", c_macs)  
    print("Backbone params: ", b_params)
    print("Classifiers params: ", c_params)

    if backbone.n_branches()==1:
        print("Single branch model")
        results['exits_ratio']=[1.0]
        results['avg_macs']=b_macs[-1]+c_macs[-1]
        results['top1']=np.round(100-top1*100,2)
        results['branch_scores']={'global':top1}
        results['params']=b_params[-1]+sum(c_params)
        results['macs']=b_macs[-1]+c_macs[-1]
        with open(save_path, 'w') as handle:
            json.dump(results, handle)
        sys.exit()


    # GLOBAL GATE to switch on/off the EECs (not used)
    '''
    if(args.gg_on):
        logging.info("Training with global gate")
    else: 
        logging.info("Training without global gate")
    '''
    
    if args.load_backbone_from_archive: 
  
        iter_path = args.output_path.rsplit("/",1)[0] 
        
        #CHECK BACKBONE IN ARCHIVE
        arch = json.load(open(os.path.join(args.output_path,'net_{}.subnet'.format(n_subnet))))
        arch_b={'ks':arch['ks'],'e':arch['e'],'d':arch['d']}
        backbone_dir=get_subnet_folder_by_backbone(iter_path,arch_b)

        if backbone_dir is None:
            pre_trained_model_path = os.path.join(args.output_path, 'bb_s.pt')
            pre_trained_classifier_path = os.path.join(args.output_path, 'c_s.pt')
            backbone_dir = args.output_path
        else:
            print("LOADED BACKBONE FROM " + backbone_dir)
    
    if os.path.exists(os.path.join(args.output_path, 'bb.pt')): # and load:
        
        logging.info('Model loaded')

        backbone.to(device)
        classifiers.to(device)

        backbone.load_state_dict(torch.load(
            os.path.join(args.output_path, 'bb.pt'), map_location=device))

        loaded_state_dict = torch.load(os.path.join(
            args.output_path, 'classifiers.pt'), map_location=device)

        # old code compatibility
        loaded_state_dict = {k: v for k, v in
                                loaded_state_dict.items()
                                if 'binary_classifier' not in k}

        classifiers.load_state_dict(loaded_state_dict)

        # Load the JSON data from the file
        with open(save_path, 'r') as handle:
            json_data = json.load(handle)

        # Access the "support_conf" field directly
        #support_conf = json_data["support_conf"]
        #sigma = json_data["global_gate"]

        # Now you can use the 'support_conf_value' variable, which contains the value of "support_conf"
        #print("Support Confidence:", support_conf)
        
    else:
        
        logging.info("Start training of the EENN...")
        backbone.to(device)
        classifiers.to(device)

        parameters = chain(backbone.parameters(),
                            classifiers.parameters())

        optimizer = get_optimizer(parameters, args.optim, args.learning_rate, args.momentum, args.weight_decay)

        epochs = args.warmup_ee_epochs + args.ee_epochs # Total number of epochs
        scheduler = get_lr_scheduler(optimizer, 'step', epochs=epochs)
        if not args.support_set: 
            n_epoch_gamma = epochs 
        else:
            logging.info("Support set enabled")
            n_epoch_gamma = args.warmup_ee_epochs

        # load weights from previous optimizer

        if args.method == 'bernulli':

            res = binary_bernulli_trainer(model=backbone,
                                            predictors=classifiers,
                                            optimizer=optimizer,
                                            scheduler= scheduler, 
                                            resolution=res,
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
                                            n_epoch_gamma=n_epoch_gamma,
                                            n_classes=n_classes,
                                            n_workers=args.n_workers
                                            )[0]

            backbone_dict, classifiers_dict, support_conf, global_gate = res
            if support_conf is not None:
                support_conf = torch.mean(support_conf, dim=0).tolist() # compute the average on the n_classes dimension
            #sigma=torch.nn.Sigmoid()(global_gate).tolist()

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
                                            lr=args.learning_rate,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)

            res = joint_trainer(model=backbone, predictors=classifiers,
                                optimizer=optimizer,
                                weights=weights, train_loader=train_loader,
                                epochs=args.ee_epochs,
                                scheduler=scheduler, joint_type=args.joint_type,
                                test_loader=test_loader,
                                eval_loader=val_loader,
                                early_stopping=early_stopping)[0]

            backbone_dict, classifiers_dict = res

            '''
        elif args.method == 'standard':

            res = standard_trainer(model=backbone,
                                    predictors=classifiers,
                                    optimizer=args.optimizer,
                                    train_loader=train_loader,
                                    epochs=args.ee_epochs,
                                    scheduler=scheduler,
                                    test_loader=test_loader,
                                    eval_loader=val_loader,
                                    early_stopping=early_stopping)[0]
            '''

            backbone_dict, classifiers_dict = res

        else:
            assert False

        backbone.load_state_dict(backbone_dict)
        classifiers.load_state_dict(classifiers_dict)

        if args.save:
            torch.save(backbone.state_dict(), os.path.join(args.output_path,
                                                            'bb.pt'))
            torch.save(classifiers.state_dict(),
                        os.path.join(args.output_path,
                                    'classifiers.pt'))


    #train_scores = standard_eval(model=backbone, dataset_loader=train_loader, classifier=classifiers[-1])

    #test_scores = standard_eval(model=backbone, dataset_loader=test_loader, classifier=classifiers[-1])

    #logging.info('Last layer train and test scores : {}, {}'.format(train_scores,test_scores))

    #if args.method != 'standard':
            
            
    if not args.tune_epsilon:
        best_epsilon = epsilon
        best_cumulative=False
        best_scores, best_counters = sm_eval(model=backbone,
                                dataset_loader=val_loader,
                                predictors=classifiers,
                                epsilon=best_epsilon,
                                cumulative_threshold=False,
                                sample=False)
        weights = []
        for ex in best_counters.values():
                weights.append(ex/n_samples)

        # For each b-th exit the avg_macs is the percentage of samples exiting from the exit 
        # multiplied by the sum of the MACs of the backbone up to the b-th exit + MACs of the b-th exit 

        avg_macs = 0
        for b in range(backbone.b):
            avg_macs += weights[b] * (b_macs[b] + c_macs[b])
    else:

        results['support_conf']=support_conf#.tolist()
        #results['global_gate']=sigma#.tolist()

        ## TUNING THRESHOLDS ##
        
        cumulative_threshold_scores = {}

        best_scores = {}
        best_score=0.0
        best_epsilon=0.1
        best_counters=[0]*backbone.n_branches()
        best_cumulative=True
        
        #1. Find epsilon with best accuracy
        for epsilon in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]:#, 0.9, 0.95, 0.98]:
            print("Evaluating epsilon: ", epsilon)
            a, b = binary_eval(model=backbone,
                                dataset_loader=val_loader,
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
                print("New best counters: {}".format(best_counters))
        

            #results['cumulative_results'] = cumulative_threshold_scores
        
        #2 Adjust epsilons to fit the constraints
        weights = []
        for ex in best_counters.values():
                weights.append(ex/n_samples)

        # For each b-th exit the avg_macs is the percentage of samples exiting from the exit 
        # multiplied by the sum of the MACs of the backbone up to the b-th exit + MACs of the b-th exit 

        avg_macs = 0
        for b in range(backbone.b):
            avg_macs += weights[b] * (b_macs[b] + c_macs[b])

        # Repair action: adjust the thresholds to make the network fit in terms of MACs
        constraint_compl = args.mmax
        constraint_acc = args.top1min
        i=backbone.b-2#cycle from the second last elem
        repaired = False
        epsilon=[ 0.7 if best_epsilon <= 0.7 else best_epsilon] + [best_epsilon] * (backbone.n_branches() - 1)
        best_epsilon = epsilon
        if(a['global']>=constraint_acc):
            while (i>=0 and avg_macs>constraint_compl): #cycle from the second last elem
                print("CONSTRAINT MACS VIOLATED: REPAIR ACTION ON BRANCH {}".format(i))
                epsilon[i] = epsilon[i] - 0.1 
                print("New epsilon: ", epsilon)
                a, b = binary_eval(model=backbone,
                                    dataset_loader=val_loader,
                                    predictors=classifiers,
                                    epsilon=epsilon,
                                    # epsilon=[epsilon] *
                                    #         (backbone.n_branches()),
                                    cumulative_threshold=True,
                                    sample=False)
                a, b = dict(a), dict(b)
                print(a['global'])
                if(a['global']<constraint_acc):
                    print("ACC VIOLATED")
                    #print(a['global'])
                    if i>=1:
                        i=i-1
                        continue
                    else:
                        break
                best_epsilon = epsilon

                weights = []
                for ex in b.values():
                        weights.append(ex/n_samples)
                avg_macs = 0
                for b in range(backbone.b):
                    avg_macs += weights[b] * (b_macs[b] + c_macs[b])
                best_scores=a
                best_counters=b
                
                if(avg_macs<=constraint_compl):
                    repaired=True
                    break
                if(epsilon[i]<=0.11):
                    i=i-1   
                    
    #COMPUTE ECE SCORES FOR CALIBRATION EVALUATION
    stats_ece = ece_score(model=backbone,predictors=classifiers, dataset_loader=val_loader)
    ece_scores={}
    for i,k in enumerate(stats_ece):
        scores = stats_ece[i]
        ece_scores[i]=scores[0]
    results['ece_scores']=ece_scores
    
    #print("Solution repaired: {}".format(repaired))
    results["exits_ratio"]=weights
    #results['backbone_macs_i'] = b_macs
    results['avg_macs'] = avg_macs
    results['epsilon'] = best_epsilon#.tolist()
    results['cumulative_threshold'] = best_cumulative

    #The branch score of the binary_eval is the percentage of samples of the dataset EXITING 
    #FROM THAT BRANCH correctly classified by the the branch
    
    results['top1'] = (1-best_scores['global']) * 100 #top1 error
    results['branch_scores'] = best_scores
    results['params']=b_params[-1]+sum(c_params)
    results['macs']=b_macs[-1]+c_macs[-1]

    #log.info('Best epsilon: {}'.format(best_epsilon))
    #log.info('Best cumulative threshold: {}'.format(best_cumulative))
    #log.info('Branches scores on exiting samples: {}'.format(best_scores))
    #log.info('Exit ratios: {}'.format(weights))
    #log.info('Average MACS: {:.2f}'.format(avg_macs))
    
    with open(save_path, 'w') as handle:
            json.dump(results, handle)

    #log.info('#' * 100)
    
