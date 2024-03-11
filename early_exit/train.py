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
    standard_trainer#, adaptive_trainer
from utils import get_dataset, get_optimizer, EarlyStopping, get_net_info, get_eenn, get_intermediate_backbone_cost, get_intermediate_classifiers_cost, \
    get_ee_scores,get_subnet_folder_by_backbone

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mobilenetv3', help='name of the model (mobilenetv3, ...)')
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="Base learning rate at the start of the training.") #0.1
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0004, type=float, help="L2 weight decay.")
    parser.add_argument("--val_split", default=0.0, type=float, help="Split of the training set used for the validation set.")
    parser.add_argument('--optim', type=str, default='SGD', help='algorithm to use for training')
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument('--dataset', type=str, default='imagenet', help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument("--data_aug", default=True, type=bool, help="True if you want to use data augmentation.")
    parser.add_argument('--save_ckpt', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for training / testing')
    parser.add_argument('--n_classes', type=int, default=1000, help='number of classes of the given dataset')
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
    parser.add_argument("--fix_last_layer", default=False, type=bool, help="True if you want to fix the last layer of the backbone.")

    '''
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
    '''

    args = parser.parse_args()

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

    fix_last_layer = False
    if get_binaries:
        fix_last_layer = args.fix_last_layer

    backbone, classifiers = get_eenn(model_name, image_size=input_size,
                                        n_classes=args.n_classes,
                                        get_binaries=get_binaries,
                                        fix_last_layer=fix_last_layer,
                                        model_path=model_path,
                                        pretrained=args.pretrained
                                        )

    print(f"DATASET: {args.dataset}")
    train_loader, val_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=args.threads, 
                                            val_fraction=args.val_fraction, img_size=res, augmentation=True, eval_test=args.eval_test)
    
    if val_loader is None:
        val_loader = test_loader

    log = Log(log_each=10)

    print("DEVICE:", device)

    model.to(device)
    epochs = args.epochs

    optimizer = get_optimizer(backbone.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay, args.rho, args.adaptive)
    
    scheduler = get_lr_scheduler(optimizer, 'step', epochs=epochs)

    # Training the backbone

    if (os.path.exists(os.path.join(args.output_path,'ckpt.pth'))):

        backbpne, optimizer = load_checkpoint(backbone, optimizer, os.path.join(args.output_path,'ckpt.pth'))
        print("Loaded checkpoint")
        top1 = validate(val_loader, backbone, device, print_freq=100)
        print("Loaded model accuracy:", np.round(top1,2))
        top1/=100

    else:
        
        top1, backbone, optimizer = train(train_loader, val_loader, epochs, backbone, device, optimizer, scheduler, log, ckpt_path=args.output_path)

    '''
        
        gg_on = method_cfg.get('global_gate', False)

        if(gg_on):
            print("Training with global gate")
        else: 
            print("Training without global gate")
        
        results = {}
        
        # MODEL COST PROFILING
        
        net = copy.deepcopy(backbone)
        if model_name == 'mobilenetv3':
            net.exit_idxs=[net.exit_idxs[-1]] #take only the final exit
            b_params, b_macs = get_intermediate_backbone_cost(backbone, input_size)
        else:
            dict_macs = net.computational_cost(torch.randn((1, 3, 32, 32)))
            b_macs = []
            for m in dict_macs.values():
                    b_macs.append(m/1e6)
            b_params=[] #Not implemented
         

        c_params, c_macs = get_intermediate_classifiers_cost(backbone, classifiers, input_size)

        results['classifiers_params'] = c_params
        results['backbone_params_i'] = b_params
        results['classifiers_macs'] = c_macs
        results['backbone_macs_i'] = b_macs

        #results['classifiers_activations']=c_act

        if method_cfg.get('pre_trained', False): #get() provides the value of the field pre_trained of method_cfg. False is the default value if
                                                 #the field does not exist 

            
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
                pre_trained_model_path = os.path.join(backbone_dir, 'bb_s.pt')
                pre_trained_classifier_path = os.path.join(backbone_dir, 'c_s.pt')
            
            log.info('Pre trained model path {}'.format(backbone_dir))

            #pre_trained_model_path = os.path.join(path, 'bb_s.pt')
            #pre_trained_classifier_path = os.path.join(path, 'c_s.pt')

            pretrained_backbone, pretrained_classifiers = get_model(
                model_name,
                image_size=input_size,
                n_classes=n_classes,
                get_binaries=False,
                model_path=model_path)
            
            
            if os.path.exists(pre_trained_model_path) and \
                    os.path.exists(pre_trained_classifier_path):
                log.info('Pre trained model loaded')
                
                pretrained_backbone.load_state_dict(
                    torch.load(pre_trained_model_path,
                               map_location=device))
                
                state_dict = torch.load(pre_trained_classifier_path,
                               map_location=device)

                # Define a prefix for the weights you want to filter (e.g., "1.classifier.1")
                #prefix = str(pretrained_backbone.n_branches()-1)

                prefix=0
                for el in state_dict.keys():
                    prefix=max(prefix,int(el.split('.')[0]))
                prefix=str(prefix)
                
                # Filter the state_dict to select only the weights with the specified prefix and remove it 
                filtered_state_dict = {key[len(prefix)+1:]: value for key, value in state_dict.items() if key.startswith(prefix)}
                
                pretrained_classifiers[-1].load_state_dict(filtered_state_dict)
                
            else:
            
                #os.makedirs(pre_trained_path, exist_ok=True)

                log.info('Training the base model')

                pretrained_backbone.to(device)
                pretrained_classifiers.to(device)

                train_set, test_set, input_size, n_classes = \
                    get_dataset(name=dataset_name,
                                model_name=None,
                                augmentation=True)

                pre_trainloader = torch.utils.data.DataLoader(train_set,
                                                                batch_size=batch_size,
                                                                shuffle=True)

                pre_testloader = torch.utils.data.DataLoader(test_set,
                                                                batch_size=batch_size,
                                                                shuffle=False)

                parameters = chain(pretrained_backbone.parameters(),
                                    pretrained_classifiers.parameters())

                optimizer = get_optimizer(parameters=parameters,
                                            name='sgd',
                                            lr=0.01,
                                            momentum=0.9,
                                            weight_decay=0)


                res = standard_trainer(model=pretrained_backbone,
                                        predictors=pretrained_classifiers,
                                        optimizer=optimizer,
                                        train_loader=pre_trainloader,
                                        epochs=epochs,
                                        scheduler=None,
                                        early_stopping=early_stopping,
                                        test_loader=pre_testloader,
                                        eval_loader=eval_loader,
                                        ckpt_path = ckpt_path
                                        )[0]

                backbone_dict, classifiers_dict = res
                # classifiers.load_state_dict(classifiers_dict)
                
                torch.save(backbone_dict,
                            pre_trained_model_path)
                torch.save(classifiers_dict,
                            pre_trained_classifier_path)
                

                pretrained_classifiers.load_state_dict(classifiers_dict)
                pretrained_backbone.load_state_dict(backbone_dict)

                log.info('Pre trained model Saved.')

            # train_scores = standard_eval(model=pretrained_backbone,
            #                              dataset_loader=trainloader,
            #                              classifier=pretrained_classifiers[
            #                                  -1])

            test_scores = standard_eval(model=pretrained_backbone,
                                        dataset_loader=testloader,
                                        classifier=pretrained_classifiers[
                                            -1])
            
            results['backbone_top1'] = test_scores * 100

            log.info('Pre trained model scores : {}, {}'.format(-1,test_scores))


            backbone.load_state_dict(pretrained_backbone.state_dict())

        if os.path.exists(os.path.join(path, 'bb.pt')) and load:
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

            # optimizer = optim.SGD(chain(backbone1.parameters(),
            #                             backbone2.parameters(),
            #                             backbone3.parameters(),
            #                             classifier.parameters()), lr=0.01,
            #                       momentum=0.9)
            log.info('Training started.')

            parameters = chain(backbone.parameters(),
                               classifiers.parameters())

            optimizer = get_optimizer(parameters=parameters,
                                      name=optimizer_name,
                                      lr=lr,
                                      momentum=momentum,
                                      weight_decay=weight_decay)

            if method_name == 'bernulli':

                priors = method_cfg.get('priors', 0.5)
                joint_type = method_cfg.get('joint_type', 'predictions')
                beta = method_cfg.get('beta', 1e-3)
                sample = method_cfg.get('sample', True)

                recursive = method_cfg.get('recursive', False)
                normalize_weights = method_cfg.get('normalize_weights', False)
                prior_mode = method_cfg.get('prior_mode', 'ones')
                regularization_loss = method_cfg.get('regularization_loss',
                                                     'bce')
                temperature_scaling = method_cfg.get('temperature_scaling',
                                                     True)
                regularization_scaling = method_cfg.get(
                    'regularization_scaling',
                    True)

                dropout = method_cfg.get('dropout', 0.0)
                backbone_epochs = method_cfg.get('backbone_epochs', 0.0)
                support_set = method_cfg.get('support_set', False)

                
                w_alpha = method_cfg.get('w_alpha',1.0)
                w_beta = method_cfg.get('w_beta',1.0)
                w_gamma = method_cfg.get('w_gamma',1.0)
                n_epoch_gamma = method_cfg.get('n_epoch_gamma',5)

                if normalize_weights:
                    assert fix_last_layer

                res = binary_bernulli_trainer(model=backbone,
                                              predictors=classifiers,
                                              optimizer=optimizer,
                                              train_loader=trainloader,
                                              epochs=epochs,
                                              prior_parameters=priors,
                                              ckpt_path=ckpt_path,
                                              joint_type=joint_type,
                                              beta=beta,
                                              sample=sample,
                                              prior_mode=prior_mode,
                                              eval_loader=eval_loader,
                                              recursive=recursive,
                                              test_loader=testloader,
                                              fix_last_layer=fix_last_layer,
                                              normalize_weights=
                                              normalize_weights,
                                              temperature_scaling=
                                              temperature_scaling,
                                              regularization_loss=
                                              regularization_loss,
                                              regularization_scaling=
                                              regularization_scaling,
                                              dropout=dropout,
                                              backbone_epochs=
                                              backbone_epochs,
                                              early_stopping=early_stopping,
                                              gg_on=gg_on,
                                              support_set=support_set,
                                              mmax = mmax,
                                              w_alpha=w_alpha,
                                              w_beta=w_beta,
                                              w_gamma=w_gamma,
                                              n_epoch_gamma=n_epoch_gamma,
                                              n_classes=n_classes
                                              )[0]

                backbone_dict, classifiers_dict, support_conf, global_gate = res
                if support_conf is not None:
                    support_conf = torch.mean(support_conf, dim=0).tolist() # compute the average on the n_classes dimension
                sigma=torch.nn.Sigmoid()(global_gate).tolist()

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            elif method_name == 'joint':
                joint_type = method_cfg.get('joint_type', 'losses')
                weights = method_cfg.get('weights', None)
                train_weights = method_cfg.get('train_weights', False)

                if train_weights:
                    weights = torch.tensor(weights, device=device,
                                           dtype=torch.float)

                    if joint_type == 'predictions':
                        weights = weights.unsqueeze(-1)
                        weights = weights.unsqueeze(-1)

                    weights = torch.nn.Parameter(weights)
                    parameters = chain(backbone.parameters(),
                                       classifiers.parameters(),
                                       [weights])

                    optimizer = get_optimizer(parameters=parameters,
                                              name=optimizer_name,
                                              lr=lr,
                                              momentum=momentum,
                                              weight_decay=weight_decay)

                res = joint_trainer(model=backbone, predictors=classifiers,
                                    optimizer=optimizer,
                                    weights=weights, train_loader=trainloader,
                                    epochs=epochs,
                                    scheduler=None, joint_type=joint_type,
                                    test_loader=testloader,
                                    eval_loader=eval_loader,
                                    early_stopping=early_stopping)[0]

                backbone_dict, classifiers_dict = res

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            elif method_name == 'adaptive':
                reg_w = method_cfg.get('reg_w', 1)

                res = adaptive_trainer(model=backbone,
                                       predictors=classifiers,
                                       optimizer=optimizer,
                                       train_loader=trainloader,
                                       epochs=epochs,
                                       scheduler=None,
                                       test_loader=testloader,
                                       eval_loader=eval_loader,
                                       early_stopping=early_stopping,
                                       reg_w=reg_w)[0]

                backbone_dict, classifiers_dict = res

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            elif method_name == 'standard':

                res = standard_trainer(model=backbone,
                                       predictors=classifiers,
                                       optimizer=optimizer,
                                       train_loader=trainloader,
                                       epochs=epochs,
                                       scheduler=None,
                                       test_loader=testloader,
                                       eval_loader=eval_loader,
                                       early_stopping=early_stopping)[0]

                backbone_dict, classifiers_dict = res

                backbone.load_state_dict(backbone_dict)
                classifiers.load_state_dict(classifiers_dict)

            else:
                assert False

            if save:
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

            stats_ece = ece_score(model=backbone,predictors=classifiers, dataset_loader=testloader)
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
                                   dataset_loader=testloader,
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
        log.info('Branches scores on exiting samples: {}'.format(best_scores))
        log.info('Exit ratios: {}'.format(weights))
        log.info('Average MACS: {:.2f}'.format(avg_macs))
        
        if save:
            save_path = os.path.join(path, 'net_{}.stats'.format(n_subnet)) # #exp__path = ..iter_x/exp_y 
    
            with open(save_path, 'w') as handle:
                json.dump(results, handle)
        
        # Get the current date and time
        current_time = datetime.datetime.now()

        # Print the current time
        print("Current time:", current_time)

        log.info('#' * 100)
       
