import copy
import os
import torch
import numpy as np

import sys
sys.path.append(os.getcwd())

from trainers.cbn.evaluators import binary_eval, get_confidences
from train_utils import get_dataset
from trainers.cbn.utils import get_intermediate_backbone_cost, get_intermediate_classifiers_cost, get_model
from explainability import plot_histograms, plot_histogramsV2

import seaborn as sns
import matplotlib.pyplot as plt

batch_size = 64
get_binaries = True
fix_last_layer = True   
dataset_name = 'cifar10'  
model_name='mobilenetv3'  
path = '../results/cifar10-cbn-mbv3-8sept-nocalibration/final/net-trade-off'
supernet='supernets/ofa_mbv3_d234_e346_k357_w1.0'
#model_path = os.path.join(path,'net_1.subnet')
model_path = os.path.join(path,'net_54.subnet')
pre_trained_model_path = os.path.join(path, 'bb.pt')
pre_trained_classifier_path = os.path.join(path, 'classifiers.pt')


train_set, test_set, input_size, n_classes = \
                    get_dataset(name=dataset_name,
                                model_name=None,
                                augmentation=True)

testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)

backbone, classifiers = get_model(model_name, image_size=input_size,
                                          n_classes=n_classes,
                                          get_binaries=get_binaries,
                                          fix_last_layer=fix_last_layer,
                                          model_path=model_path,
                                          pretrained=True,
                                          supernet=supernet
                                          )

backbone.load_state_dict(torch.load(pre_trained_model_path))
classifiers.load_state_dict(torch.load(pre_trained_classifier_path))

cumulative_threshold_scores = {}

best_scores = {}
best_score=0.0
best_epsilon=0.1
best_counters=[0]*backbone.n_branches()
best_cumulative=True

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
constraint_compl = 2.7
constraint_acc = 0.65
i=backbone.b-2#cycle from the second last elem
repaired = False
epsilon=[ 0.7 if best_epsilon <= 0.7 else best_epsilon] + [best_epsilon] * (backbone.n_branches() - 1)
best_epsilon = epsilon
if(a['global']>=constraint_acc):
    while (i>=0 and avg_macs>constraint_compl): #cycle from the second last elem
        #print("CONSTRAINT MACS VIOLATED: REPAIR ACTION ON BRANCH {}".format(i))
        epsilon[i] = epsilon[i] - 0.1 
        '''
        if(epsilon[0]<1e-2):
            #print("SOLUTION IS NOT ADMISSIBLE")
            break
        '''
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
        
        '''
        print("BRANCHES SCORES")
        print(a)
        print("WEIGHTS")
        print(weights)
        print("AVG MACS")
        print(avg_macs)
        '''
        
        if(avg_macs<=constraint_compl):
            repaired=True
            break
        if(epsilon[i]<=0.11):
            i=i-1   

print("FINAL SCORE")
print(best_scores['global'] * 100)

'''
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
'''