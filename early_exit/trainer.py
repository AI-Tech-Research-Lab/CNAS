import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

import os

from evaluators import standard_eval, branches_eval, binary_eval, \
    binary_statistics
from models.base import BranchModel
from models.mobilenet_v3 import EEMobileNetV3
from utils_ee import calculate_centroids_confidences, calculate_centroids_scores, extract_balanced_subset, get_device, \
 get_intermediate_backbone_cost, get_intermediate_classifiers_cost, save_eenn
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

#standard: train the network with only the final exit (last element of predictors)
def standard_trainer(model: BranchModel,
                     predictors: nn.Module,
                     optimizer,
                     train_loader,
                     epochs,
                     scheduler=None,
                     early_stopping=None,
                     test_loader=None, eval_loader=None, ckpt_path=None):
    
    device = get_device(model)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_model_i = 0
    best_eval_score = -1

    model.to(device)

    print("Checkpoint path: ", ckpt_path)
    save_interval = 1

    if(ckpt_path is not None and os.path.exists(ckpt_path)):
        ckpt = load_model(device, ckpt_path)
        model.load_state_dict(ckpt['backbone_state'])
        predictors.load_state_dict(ckpt['classifiers_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        best_model = ckpt['best_backbone_state']
        best_predictors = ckpt['best_classifiers_state']
        best_eval_score = ckpt['best_eval_score']
        epoch = ckpt['epoch']
        bar = tqdm(range(epoch+1,epochs), leave=True)
        print("Resume from epoch: ", epoch)
    else: 
        print("No checkpoint found")
        bar = tqdm(range(epochs), leave=True)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()

    for p in predictors[:-1]: #freeze all the classifiers except the last one
        for param in p.parameters():
            if param.requires_grad:
                param.requires_grad = False

    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)[-1]
            pred = predictors[-1].logits(pred)

            loss = nn.functional.cross_entropy(pred, y, reduction='none')
            losses.extend(loss.tolist())
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            """eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)"""
            branches_scores = branches_eval(model, predictors, eval_loader)
            # branches_scores = {k: v[1] for k, v in branches_scores.items()}
            eval_scores = branches_scores['final']
        else:
            eval_scores = None

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            r = early_stopping.step(eval_scores) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = deepcopy(model.state_dict())
                best_predictors = deepcopy(predictors.state_dict())

                best_model_i = epoch
        else:
            if (eval_scores is not None and eval_scores >= best_eval_score) \
                    or eval_scores is None:

                if eval_scores is not None:
                    best_eval_score = eval_scores

                best_model = deepcopy(model.state_dict())
                best_predictors = deepcopy(predictors.state_dict())

                best_model_i = epoch
        
        #Save the checkpoint every x epoch
        if ckpt_path is not None and (epoch+1)%save_interval == 0:
            save_model(model, predictors, best_model,
                       best_predictors, best_eval_score, epoch, optimizer, ckpt_path)
            #print("Checkpoint saved at epoch: ", epoch)
        
        '''
        train_scores = standard_eval(model=model,
                                     dataset_loader=train_loader,
                                     classifier=predictors[-1])

        test_scores = standard_eval(model=model,
                                    dataset_loader=test_loader,
                                    classifier=predictors[-1])
        
        bar.set_postfix({'Train score': train_scores, 'Test score': test_scores,'Eval score': eval_scores if eval_scores != 0 else 0, 'Mean loss': mean_loss})

        scores.append((train_scores, eval_scores, test_scores))
        '''
        bar.set_postfix({'Eval score': eval_scores if eval_scores != 0 else 0, 'Mean loss': mean_loss})
    
    # Remove checkpoint 
    if os.path.exists(ckpt_path):
      os.remove(ckpt_path)

    return (best_model, best_predictors), \
           scores, \
           mean_losses
           #scores[best_model_i] if len(scores) > 0 else 0, 


def joint_trainer(model: BranchModel,
                  predictors: nn.ModuleList,
                  optimizer,
                  train_loader,
                  epochs,
                  scheduler=None,
                  weights=None,
                  joint_type='logits',
                  early_stopping=None,
                  test_loader=None, eval_loader=None):
    device = get_device(model)

    if joint_type not in ['losses', 'logits']:
        raise ValueError

    if weights is None:
        weights = torch.tensor([1.0] * model.n_branches(), device=device)

    # weights = torch.tensor([1.0] * model.n_branches(), device=device)

    if not isinstance(weights, (torch.Tensor, torch.nn.Parameter)):
        if isinstance(weights, (int, float)):
            weights = torch.tensor([weights] * model.n_branches(),
                                   device=device, dtype=torch.float)

        else:
            weights = torch.tensor(weights, device=device, dtype=torch.float)

        if joint_type == 'logits':
            weights = weights.unsqueeze(-1)
            weights = weights.unsqueeze(-1)

    scores = []
    mean_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()
    best_model_i = 0
    best_eval_score = -1

    model.to(device)
    predictors.to(device)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()
    bar = tqdm(range(epochs), leave=True)
    for epoch in bar:

        losses = []

        for i, (x, y) in enumerate(train_loader):
            model.train()
            predictors.train()

            x, y = x.to(device), y.to(device)

            preds = model(x)
            logits = []

            for j, bo in enumerate(preds):
                l = predictors[j].logits(bo)
                logits.append(l)

            preds = torch.stack(logits, 0)

            if joint_type == 'logits':
                preds = weights * preds
                f_hat = preds.sum(0)

                loss = nn.functional.cross_entropy(f_hat, y, reduction='mean')

            else:
                loss = torch.stack(
                    [nn.functional.cross_entropy(p, y, reduction='mean')
                     for p in preds[:-1]], 0)

                loss = loss * weights[:-1]
                loss = loss.sum()

                loss += nn.functional.cross_entropy(preds[-1], y,
                                                    reduction='mean')

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            """eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)"""
            branches_scores = branches_eval(model, predictors, eval_loader)
            eval_scores = branches_scores['final']

        else:
            eval_scores = None

        mean_loss = sum(losses) / len(losses)
        mean_losses.append(mean_loss)

        if early_stopping is not None:
            print(early_stopping.current_value, eval_scores,
                  early_stopping.tolerance)

            r = early_stopping.step(eval_scores) if eval_loader is not None \
                else early_stopping.step(mean_loss)

            if r < 0:
                break
            elif r > 0:
                best_model = deepcopy(model.state_dict())
                best_predictors = deepcopy(predictors.state_dict())

                best_model_i = epoch
        else:
            print(eval_scores, best_eval_score)

            if (eval_scores is not None and eval_scores >= best_eval_score) \
                    or eval_scores is None:

                if eval_scores is not None:
                    best_eval_score = eval_scores

                best_model = deepcopy(model.state_dict())
                best_predictors = deepcopy(predictors.state_dict())

                best_model_i = epoch

        train_scores = standard_eval(model=model,
                                     dataset_loader=train_loader,
                                     classifier=predictors[-1])

        test_scores = standard_eval(model=model,
                                    dataset_loader=test_loader,
                                    classifier=predictors[-1])

        scores.append(test_scores)

        bar.set_postfix(
            {'Train score': train_scores, 'Test score': test_scores,
             'Eval score': eval_scores if eval_scores != 0 else 0,
             'Mean loss': mean_loss})
        scores.append((train_scores, eval_scores, test_scores))

    return (best_model, best_predictors), scores, scores[best_model_i] if len(
        scores) > 0 else 0, mean_losses


# sample can be set to False (check)

def binary_bernulli_trainer(model: BranchModel,
                            predictors: nn.ModuleList,
                            optimizer,
                            train_loader,
                            resolution,
                            epochs,
                            prior_parameters,
                            ckpt_path=None,
                            beta=1e-3,
                            joint_type='logits',
                            sample=True, 
                            scheduler=None,
                            early_stopping=None,
                            test_loader=None,
                            eval_loader=None,
                            #recursive=False,
                            fix_last_layer=False,
                            normalize_weights=True,
                            prior_mode='ones',
                            regularization_loss='bce',
                            temperature_scaling=True,
                            regularization_scaling=False,
                            dropout=0,
                            backbone_epochs=0,
                            gg_on=False, #global gate on
                            support_set=False, #not used actually
                            mmax=None,
                            w_alpha=1.0,
                            w_beta=1.0,
                            w_gamma=1.0,
                            n_epoch_gamma=0,
                            n_classes=10,
                            n_workers=2
                            ):
    
    def energy_loss(exits_costs, confidence_scores): 
    
    # This loss measures the ratio of MACs as the MACs of the EENN
    # measured using the confidence as exit ratio divided by the 
    # the MACs of the backbone 

        if normalize_weights:
            a, b = torch.split(confidence_scores,
                               [confidence_scores.shape[1] - 1, 1],
                               dim=1)

            c = torch.cumprod(1 - a, 1)

            cat = torch.cat((torch.ones_like(b), c), 1)
            confidence_scores = confidence_scores * cat
        
        confidence_scores = confidence_scores.squeeze()

        w = torch.tensor(exits_costs).to(device)

        # Perform weighted sum and compute mean

        if(mmax is not None and exits_costs[-1]>mmax):
            macs_list = []
            for row in confidence_scores:
                m = torch.dot(row, w)
                #squared_diff = torch.pow(torch.max(torch.tensor(0.0).to(device), weighted_sum - constraint), 2)
                macs_list.append(m)

            # Compute the mean along the first dimension
            avg_macs = torch.mean(torch.stack(macs_list))

            # Compute the loss as the squared constraint violation
            normalized_cv = max(0,avg_macs.item()-mmax)/abs(exits_costs[-1]-mmax)
            loss = torch.Tensor([normalized_cv]).to(device)
        else:
            loss= torch.Tensor([0.0]).to(device)
        
        
        return loss


    def classification_loss(logits, ground_truth, confidence_scores,
                            drop=False):
        if normalize_weights:
            a, b = torch.split(confidence_scores,
                               [confidence_scores.shape[1] - 1, 1],
                               dim=1)

            c = torch.cumprod(1 - a, 1)

            cat = torch.cat((torch.ones_like(b), c), 1)
            confidence_scores = confidence_scores * cat

        if joint_type == 'logits':
            if normalize_weights:
                p1, p2 = torch.split(logits,
                                     [logits.shape[1] - 1, 1],
                                     dim=1)

                d1, _ = torch.split(confidence_scores,
                                    [logits.shape[1] - 1, 1],
                                    dim=1)

                p1 = p1 * d1
                f_hat = p1.sum(1)
                p2 = p2.squeeze()

                loss = nn.functional.cross_entropy(f_hat, ground_truth,
                                                   reduction='mean')
                loss += nn.functional.cross_entropy(p2, ground_truth,
                                                    reduction='mean')

            else:
                logits = logits * confidence_scores
                f_hat = torch.sum(logits, 1)
                # f_hat /= distributions.sum(1)

                loss = nn.functional.cross_entropy(f_hat, ground_truth,
                                                   reduction='mean')
        else:

            loss = torch.stack(
                [nn.functional.cross_entropy(logits[:, pi], ground_truth,
                                             reduction='none')
                 for pi in range(preds.shape[1])], -1)

            confidence_scores = confidence_scores.squeeze()
            loss = loss * confidence_scores

            loss = loss.mean(0)
            loss = loss.sum()

        return loss
    
    def support_loss (support_matrix, y, distributions):
        import torch.nn.functional as F
        # Ensure the matrix and y have the same number of classes (N)
        #assert support_matrix.shape[0] == len(y)

        mse_values = torch.zeros(len(distributions), device=device)
        
        for i in range(len(distributions)):
            label = y[i]
            class_confidences = distributions[i]
            class_average_confidences = support_matrix[label]
            # Calculate the Mean Squared Error (MSE) between class_average_confidences and class_confidences
            mse = F.mse_loss(class_average_confidences, class_confidences)
            mse_values[i]=mse

        # Calculate the average MSE for the batch
        avg_mse = mse_values.sum()/len(distributions)

        return avg_mse
    
    def gate_loss(sigma):
        import torch.nn.functional as F
        #loss = torch.norm(sigma, p=1)/model.n_branches()
        n_branches = sigma.size(0)
        loss = torch.sum(torch.abs(sigma * torch.log(sigma) + (1 - sigma) * torch.log(1 - sigma)))
        norm_loss = loss/n_branches
        return norm_loss

    log = logging.getLogger(__name__)
    
    input_size = (3,resolution,resolution)

    if(isinstance(model,EEMobileNetV3)):
       _, b_macs = get_intermediate_backbone_cost(model, input_size)
    else:
       dict_macs = model.computational_cost(torch.randn((1, 3, resolution, resolution)))
       b_macs = []
       for m in dict_macs.values():
            b_macs.append(m/1e6)
    _, c_macs = get_intermediate_classifiers_cost(model, predictors, input_size)

    exit_costs = [] #gamma computational cost
    inv_costs = []
    for b, c in zip(b_macs, c_macs):
        ex = b + c
        exit_costs.append(ex)
        inv_costs.append(1/ex)
    ####

    c = 8 # Arbitrary value s.t. sigma(c) is close to 1
    global_gate = nn.Parameter(c * torch.ones(model.n_branches()),requires_grad=False) #vector of scalars

    device = get_device(model)
    model.to(device)
    predictors.to(device)

    if joint_type not in ['losses', 'logits']:
        raise ValueError

    if prior_mode not in ['entropy', 'ones', 'probability']:
        raise ValueError

    scores = []
    mean_losses = []
    mean_energy_losses = []
    mean_gate_losses = []
    mean_support_losses = []

    best_model = model.state_dict()
    best_predictors = predictors.state_dict()

    best_model_i = 0
    best_eval_score = -1

    if early_stopping is not None:
        early_stopping.reset()

    model.train()

    # epochs = epochs + backbone_epochs

    #print("Checkpoint path: ", ckpt_path)
    save_interval = 5

    if(ckpt_path is not None and os.path.exists(ckpt_path)):
        ckpt = load_model(device, ckpt_path)
        model.load_state_dict(ckpt['backbone_state'])
        predictors.load_state_dict(ckpt['classifiers_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        best_model = ckpt['best_backbone_state']
        best_predictors = ckpt['best_classifiers_state']
        best_eval_score = ckpt['best_eval_score']
        epoch = ckpt['epoch']
        bar = range(epoch+1,epochs)
        print("Resume from epoch: ", epoch)
    else: 
        #print("No checkpoint found")
        bar = tqdm(range(epochs), leave=True)

    bar = range(epochs)#tqdm(range(epochs), leave=True)

    if temperature_scaling and sample:
        # temperatures = [np.exp(-10 * i / epochs)
        #                 for i in range(epochs // 3)]
        # temperatures += [0] * (epochs - len(temperatures))

        temperatures = [20 / (2 ** i) for i in range(epochs)]
        temperatures = [5 if t < 5 else t for t in temperatures]

        log.info('Temperature scaling: {}'.format(temperatures))
    else:
        temperatures = [1] * epochs

    if regularization_scaling:
        prior_weights = [0] * (epochs // 5)
        prior_weights += list(
            np.linspace(0.1, 1, (epochs - len(prior_weights)) + 1))
        log.info('Regularization scaling: {}'.format(prior_weights))
    else:
        prior_weights = [1] * epochs
    
    #w_s = torch.tensor(100.0, requires_grad=False, dtype=torch.float32)
    #w_s = 0 #nn.Parameter(torch.tensor(0.0))
    w_gg=0

    support_matrix = None

    '''
    for p in predictors[:-1]: #switch on all the classifiers before the last one
        for param in p.parameters():
            if not param.requires_grad:
                print("The predictor is frozen")
                param.requires_grad = True
    '''

    ori_train_loader = copy.deepcopy(train_loader)
    
    for epoch in bar:

        model.train()
        predictors.train()

        losses = []
        kl_losses = []
        energy_losses = []
        gate_losses = []
        support_losses = []

        current_temperature = temperatures[epoch]
        current_prior_w = prior_weights[epoch]


        #SUPPORT SET

        if epoch == n_epoch_gamma:
            #w_gg=10
            print("SUPPORT SET ON")
            
            if (gg_on):
                print("GLOBAL GATE ON")
                ## Trainable paramater
                c = 0 # Arbitrary value s.t. sigma(c) is 0.5, the maximum point of the entropy function
                global_gate = nn.Parameter(c * torch.ones(model.n_branches()),requires_grad=True) #vector of scalars
                optimizer.add_param_group({'params': global_gate, 'lr': 0.1})
    

        if epoch >= n_epoch_gamma:
            
            support_loader, train_loader = extract_balanced_subset(ori_train_loader, subset_percentage=0.1, n_classes=n_classes, n_workers=n_workers)
            support_matrix = calculate_centroids_scores(support_loader, model, predictors, n_classes=n_classes)
            
        for bi, (x, y) in tqdm(enumerate(train_loader), leave=False,
                               total=len(train_loader)):

            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                bos = model(x)

            # if backbone_epochs > 0:
            #     if epoch > backbone_epochs:
            #         with torch.no_grad():
            #             bos = model(x)
            #     else:
            #         bos = model(x)
            #         current_prior_w = 0
            # else:
            #     bos = model(x)

            sigma = nn.Sigmoid()(global_gate)

            distributions, logits = [], []

            for j, bo in enumerate(bos):
                l, b = predictors[j](bo) #b is the confidence
                #matrix[:,j]=b.view(-1)
                if(gg_on):
                   b = b * sigma[j] # b multiplied by the sigmoid of global value
                distributions.append(b)
                logits.append(l)

            preds = torch.stack(logits, 1)
            distributions = torch.stack(distributions, 1)

            reg_term = 0

            if beta > 0 and current_prior_w > 0:

                weights = None

                with torch.no_grad():
                    if prior_mode == 'probability':
                        _y = y.unsqueeze(-1).expand(-1,
                                                    distributions.shape[1]) \
                            .unsqueeze(-1)
                        sf = torch.softmax(preds, -1)

                        prior_gt = torch.gather(sf, -1, _y)

                    elif prior_mode == 'entropy':
                        sf = torch.softmax(preds, -1)
                        h = -(sf + 1e-12).log() * sf
                        h = h / np.log(sf.shape[-1])
                        h = h.sum(-1)
                        prior_gt = 1 - h
                        prior_gt = prior_gt.unsqueeze(-1)

                    elif prior_mode == 'ones':
                        _y = y.unsqueeze(1)
                        mx = torch.argmax(preds, -1)
                        prior_gt = (mx == _y).float()
                        prior_gt = prior_gt.unsqueeze(-1)

                        # prior_gt_flat = torch.flatten(prior_gt, 0)
                        # tot = prior_gt_flat.shape[0]
                        #
                        # ones = prior_gt_flat.sum(0)
                        # zeros = tot - ones
                        #
                        # w1 = tot / (2 * ones)
                        #
                        # w2 = tot / (2 * zeros)
                        #
                        # weights = w1 * prior_gt_flat + \
                        #           w2 * (1 - prior_gt_flat)
                        #
                        # weights = torch.nan_to_num(weights, 1)
                        #
                        # weights = weights.view(mx.shape).unsqueeze(-1)

                        tot = preds.shape[0]
                        ones = prior_gt.sum(0)
                        zeros = tot - ones

                        w1 = tot / (2 * ones)
                        torch.nan_to_num_(w1, 0)
                        w1 = w1.unsqueeze(0).expand(tot, -1, -1)

                        w2 = tot / (2 * zeros)
                        torch.nan_to_num_(w2, 0)

                        w2 = w2.unsqueeze(0).expand(tot, -1, -1)

                        weights = w1 * prior_gt + w2 * (1 - prior_gt)

                # d = torch.clamp(distributions, 1e-8, 1 - 1e-8)
                # d = distributions

                # if fix_last_layer:
                #     d = d[:, :-1]
                #
                #     if weights is not None:
                #         weights = weights[:, :-1]
                #
                #     bce = nn.functional.binary_cross_entropy(
                #         d, prior_gt[:, :-1],
                #         reduction='none', weight=weights)
                # else:

                reg_term = nn.functional.binary_cross_entropy(
                    torch.clamp(distributions, 1e-4, 1 - 1e-4),
                    prior_gt,
                    reduction='none',
                    weight=weights)

                # reg_term = nn.functional.mse_loss(distributions,
                #                              prior_gt,
                #                              reduction='none')

                # reg_term = torch.abs(d - prior_gt)

                if fix_last_layer:
                    reg_term = reg_term[:, :-1]

                # bce = torch.abs(d - prior_gt)
                # bce = torch.sqrt(bce)
                # print(bce.mean(), prior_gt.mean())

                reg_term = reg_term.mean()
                # print(reg_term)

                reg_term *= beta

                kl_losses.append(reg_term.item())

            else:
                kl_losses.append(0)

            if sample and current_temperature > 0:
                distributions = RelaxedBernoulli(current_temperature,
                                                 distributions).rsample()

                if fix_last_layer:
                    distributions[:, -1] = 1

            if dropout > 0:
                with torch.no_grad():
                    assert dropout < 1

                    mask = torch.bernoulli(1 - distributions)

                    if fix_last_layer:
                        mask[:, -1] = 1

                    distributions = mask * distributions

            cl_loss = classification_loss(preds, y, distributions, drop=True)

            losses.append(cl_loss.item())
            
            #w_energy = 1 

            en_loss = energy_loss(exit_costs, distributions)
            energy_losses.append(en_loss.item())            

            loss = w_alpha*cl_loss + (reg_term * current_prior_w) + w_beta * en_loss

            if(epoch>=n_epoch_gamma):

                if(gg_on):
                #g_loss = torch.norm(sigma, p=1)/model.n_branches() #reg l1 norm of the sigma global gate
                   w_gg=10
                   g_loss = gate_loss(sigma)
                   gate_losses.append(g_loss.item())
                   loss += w_gg * g_loss
                
                sup_loss = support_loss(support_matrix, y, distributions)
                support_losses.append(sup_loss.item())
                #w_s = 1
                loss += + w_gamma * sup_loss
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = np.mean(losses)
        mean_gate_loss = np.mean(gate_losses)
        mean_energy_loss = np.mean(energy_losses)
        mean_support_loss = np.mean(support_losses)

        if scheduler is not None:
            if isinstance(scheduler, (StepLR, MultiStepLR)):
                scheduler.step()
            elif hasattr(scheduler, 'step'):
                scheduler.step()

        if eval_loader is not None:
            """eval_scores = standard_eval(model, eval_loader, topk=[1, 5],
                                        device=device)"""
            branches_scores = branches_eval(model, predictors, eval_loader)
            eval_scores = branches_scores['final']

        else:
            eval_scores = None

        if early_stopping is not None:
            r = early_stopping.step(eval_scores) if eval_loader is not None \
                else early_stopping.step(mean_loss)
            
            print("Early stopping current value, eval scores")
            print(early_stopping.current_value, eval_scores)

            if r < 0:
                break
            elif r > 0:
                best_model = deepcopy(model.state_dict())
                best_predictors = deepcopy(predictors.state_dict())

                best_model_i = epoch

        else:
            if (eval_scores is not None and eval_scores >= best_eval_score) \
                    or eval_scores is None:
                print(best_eval_score, eval_scores)

                if eval_scores is not None:
                    best_eval_score = eval_scores

                best_model = deepcopy(model.state_dict())
                best_predictors = deepcopy(predictors.state_dict())

                best_model_i = epoch

        #Save the checkpoint every x epoch
        if ckpt_path is not None and (epoch+1)%save_interval == 0:
            save_eenn(model, predictors, best_model, best_predictors, best_eval_score, epoch, optimizer, ckpt_path)
            #print("Checkpoint saved")
        
        '''
        train_scores = standard_eval(model=model,
                                     dataset_loader=train_loader,
                                     classifier=predictors[-1])

        test_scores = standard_eval(model=model,
                                    dataset_loader=test_loader,
                                    classifier=predictors[-1])

        scores.append(test_scores)

        s = branches_eval(model=model,
                          dataset_loader=test_loader,
                          predictors=predictors)
        s = dict(s)
        #print(s)

        if current_prior_w > 0:
            correct_stats, incorrect_stats = \
                binary_statistics(model=model,
                                  dataset_loader=test_loader,
                                  predictors=predictors)

            #print([(k, np.mean(v), np.std(v)) for k, v in correct_stats.items()])
            #print([(k, np.mean(v), np.std(v)) for k, v in incorrect_stats.items()])

            for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                            0.7, 0.8, 0.9, 0.95, 0.98]:
                # for epsilon in [0.9]:
                prior_gt, b = binary_eval(model=model,
                                          dataset_loader=test_loader,
                                          predictors=predictors,
                                          epsilon=[0.7 if epsilon <= 0.7
                                                   else epsilon] + ([epsilon]
                                                                    * (
                                                                            model.n_branches() - 1)),
                                          cumulative_threshold=
                                          normalize_weights)

                prior_gt, b = dict(prior_gt), dict(b)

                s = '\tEpsilon {}. '.format(epsilon)
                for k in sorted([k for k in prior_gt.keys() if k != 'global']):
                    s += 'B: {}, S: {}, C: {}. '.format(k,
                                                        np.round(prior_gt[k]
                                                                 * 100,
                                                                 2),
                                                        b[k])
                s += 'GS: {}'.format(prior_gt['global'])

                #print(s)
        '''

        mean_kl_loss = np.mean(kl_losses)
        mean_losses.append(mean_loss)
        mean_gate_losses.append(mean_gate_loss)
        mean_energy_losses.append(mean_energy_loss)
        mean_support_losses.append(mean_support_loss)

        
        if(epoch>=n_epoch_gamma):
            #bar.set_postfix(
            print(
                {
                    'N. epoch': epoch,
                    't': current_temperature, 'w': current_prior_w,
                    #'Train score': train_scores, 'Test score': test_scores,
                    'Eval score': eval_scores if eval_scores != 0 else 0,
                    #'Global gate': sigma,
                    'Mean loss': mean_loss, #'Gate loss': mean_gate_loss, 
                    'Energy loss': mean_energy_loss,
                    'Support loss': mean_support_loss, 'Mean kl loss': mean_kl_loss
                })
        else:
            print(
                {   'N. epoch': epoch,
                    't': current_temperature, 'w': current_prior_w,
                    #'Train score': train_scores, 'Test score': test_scores,
                    'Eval score': eval_scores if eval_scores != 0 else 0,
                    #'Global gate': sigma,
                    'Mean loss': mean_loss, #'Gate loss': mean_gate_loss, 
                    'Energy loss': mean_energy_loss,
                    'Mean kl loss': mean_kl_loss
                })
            
    # Remove temporary checkpoint 
    #if ckpt_path is not None and os.path.exists(ckpt_path):
      #os.remove(ckpt_path)

    return (best_model, best_predictors, support_matrix, global_gate), \
           mean_losses, mean_gate_losses, mean_energy_losses, mean_support_losses
           #scores, scores[best_model_i], 
           
               
