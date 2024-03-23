from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from torch import nn

import os, sys
sys.path.append(os.getcwd())

from early_exit.models.base import BranchModel, IntermediateBranch
from train_utils import get_device


def accuracy_score(expected: np.asarray, predicted: np.asarray, topk=None):
    if topk is None:
        topk = [1, 5]

    if isinstance(topk, int):
        topk = [topk]

    expected, predicted = np.asarray(expected), np.asarray(predicted)
    assert len(expected) == len(predicted)
    assert len(predicted.shape) == 2 and len(expected.shape) == 1
    assert predicted.shape[1] >= max(topk)

    res = {k: 0 for k in topk}

    total = len(expected)

    for t, p in zip(expected, predicted):
        for k in topk:
            if t in p[:k]:
                res[k] += 1

    res = {k: v / total for k, v in res.items()}

    return res


@torch.no_grad()
def standard_eval(model: BranchModel,
                  classifier: IntermediateBranch,
                  dataset_loader):
    device = get_device(model)

    model.eval()
    classifier.eval()

    total = 0
    correct = 0

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        final_preds = classifier.logits(model(x)[-1])
        pred = torch.argmax(final_preds, 1)

        total += y.size(0)
        correct += (pred == y).sum().item()

    score = correct / total

    return score


@torch.no_grad()
def branches_eval(model: BranchModel, predictors, dataset_loader):
    device = get_device(model)

    model.eval()
    predictors.eval()

    corrects = defaultdict(int)
    tot = 0

    for x, y in dataset_loader:
        tot += x.shape[0]

        x, y = x.to(device), y.to(device)

        preds = model(x)

        for j, bo in enumerate(preds):
            l = predictors[j].logits(bo)
            p = torch.argmax(l, 1)

            correct = (p == y).sum().item()
            corrects[j] += correct

    scores = {k: v / tot for k, v in corrects.items()}
    scores['final'] = scores.pop(len(predictors) - 1)

    return scores


@torch.no_grad()
def entropy_eval(model: BranchModel,
                 predictors: nn.ModuleList,
                 threshold: Union[List[float], float],
                 dataset_loader):
    model.eval()
    predictors.eval()
    device = get_device(model)

    if isinstance(threshold, float):
        threshold = [threshold] * model.n_branches()

    exits_counter = defaultdict(int)
    exits_corrected = defaultdict(int)

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        preds = model(x)

        distributions, logits = [], []

        for j, bo in enumerate(preds):
            l = predictors[j].logits(bo)
            logits.append(l)

        logits = torch.stack(logits, 0)

        for bi in range(x.shape[0]):
            found = False

            for i, predictor in enumerate(predictors):
                p = logits[i][bi]  # .unsqueeze(0)
                sf = nn.functional.softmax(p, -1)
                h = -(sf + 1e-12).log() * sf
                h = h / np.log(sf.shape[-1])
                h = h.sum()

                if h < threshold[i]:
                    pred = torch.argmax(p)

                    if pred == y[bi]:
                        exits_corrected[i] += 1

                    exits_counter[i] += 1
                    found = True
                    break

            if not found:
                i = len(predictors) - 1
                p = logits[i][bi]

                exits_counter[i] += 1
                pred = torch.argmax(p)

                if pred == y[bi]:
                    exits_corrected[i] += 1

    branches_scores = {}
    tot = 0
    correctly_predicted = 0

    for k in exits_counter:
        correct = exits_corrected[k]
        counter = exits_counter.get(k, 0)

        if counter == 0:
            score = 0
        else:
            score = correct / counter

        branches_scores[k] = score

        tot += counter
        correctly_predicted += correct

    branches_scores['global'] = correctly_predicted / tot

    return branches_scores, exits_counter

@torch.no_grad()
def get_confidences(model: BranchModel,
                predictors: nn.ModuleList,
                dataset_loader,
                cumulative_threshold=False,
                global_gate=None):

    model.eval()
    predictors.eval()
    device = get_device(model)

    
    if global_gate is None:
        sigma = [1] * model.n_branches()
    else:
        sigma = global_gate

    confidences = [[] for _ in range(model.n_branches())]

    #print("SIGMA: ", sigma )

    for x, y in dataset_loader:

        x, y = x.to(device), y.to(device)

        preds = model(x)

        distributions, logits = [], []
        
        for j, bo in enumerate(preds):
            l, b = predictors[j](bo)
            b = b * sigma[j]
            distributions.append(b)
            logits.append(l)
        
        distributions = torch.stack(distributions, 0)
        #print("DISTRIBUTIONS SHAPE: ", distributions.shape)
        #print(distributions[:10])

        if cumulative_threshold:
            distributions[1:] = distributions[1:] * \
                                torch.cumprod(1 - distributions[:-1], 0)
            distributions = torch.cumsum(distributions, dim=0)

        logits = torch.stack(logits, 0)

        #print(distributions[:10])

        for bi in range(x.shape[0]):
            #print("Batch input: ", bi)
            for i in range(logits.shape[0]):
                b = distributions[i][bi]
                #print("Branch: ", i)
                #print("Confidence: ", b)
                confidences[i].append(b.item())
    
    return confidences

@torch.no_grad()
def binary_eval(model: BranchModel,
                predictors: nn.ModuleList,
                dataset_loader,
                epsilon: Union[List[float], float] = None,
                cumulative_threshold=False,
                sample=False,
                global_gate=None): 
    
    model.eval()
    predictors.eval()
    device = get_device(model)

    if epsilon is None:
        epsilon = 0.5

    if isinstance(epsilon, float):
        epsilon = [epsilon] * model.n_branches()
    
    if global_gate is None:
        sigma = [1] * model.n_branches()
    else:
        sigma = global_gate#nn.Sigmoid()(global_gate)

    #print("Evaluating with thresholds:")
    #print(epsilon)

    exits_counter = {key: 0 for key in range(model.n_branches())}#defaultdict(int)
    exits_corrected = {key: 0 for key in range(model.n_branches())}#defaultdict(int)

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        preds = model(x)

        distributions, logits = [], []
        
        count=0
        for j, bo in enumerate(preds):
            l, b = predictors[j](bo)
            b = b * sigma[j]
            distributions.append(b)
            logits.append(l)
        
        distributions = torch.stack(distributions, 0)

        if cumulative_threshold:
            distributions[1:] = distributions[1:] * \
                                torch.cumprod(1 - distributions[:-1], 0)
            distributions = torch.cumsum(distributions, dim=0)

        logits = torch.stack(logits, 0)

        for bi in range(x.shape[0]):
            found = False
            for i in range(logits.shape[0]):

                b = distributions[i][bi]

                if b >= epsilon[i]:
                    p = logits[i][bi]
                    pred = torch.argmax(p)
                    if pred == y[bi]:
                        exits_corrected[i] += 1

                    exits_counter[i] += 1

                    found = True
                    break

            if not found:
                i = len(predictors) - 1
                p = logits[i][bi]

                exits_counter[i] += 1
                pred = torch.argmax(p)

                if pred == y[bi]:
                    exits_corrected[i] += 1

    branches_scores = {}
    tot = 0
    correctly_predicted = 0

    for k in range(model.n_branches()):

        correct = exits_corrected[k]
        counter = exits_counter.get(k, 0)

        if counter == 0:
            score = 0
        else:
            score = correct / counter

        branches_scores[k] = score

        tot += counter
        correctly_predicted += correct

    branches_scores['global'] = correctly_predicted / tot

    return branches_scores, exits_counter

def score_margin(logits):
    s=torch.nn.functional.softmax(logits, dim=1) #softmax
    sm = torch.max(s,dim=1)[0]-torch.topk(s, 2, dim=1)[0][:,1] #largest - second largest value in the softmax
    return sm 

@torch.no_grad()
def sm_eval(model: BranchModel,
                predictors: nn.ModuleList,
                dataset_loader,
                epsilon: Union[List[float], float] = None,
                cumulative_threshold=False,
                sample=False,
                global_gate=None): 
    
    model.eval()
    predictors.eval()
    device = get_device(model)

    if epsilon is None:
        epsilon = 0.5

    if isinstance(epsilon, float):
        epsilon = [epsilon] * model.n_branches()
    
    if global_gate is None:
        sigma = [1] * model.n_branches()
    else:
        sigma = global_gate#nn.Sigmoid()(global_gate)

    #print("Evaluating with thresholds:")
    #print(epsilon)

    exits_counter = {key: 0 for key in range(model.n_branches())}#defaultdict(int)
    exits_corrected = {key: 0 for key in range(model.n_branches())}#defaultdict(int)

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        preds = model(x)

        distributions, logits = [], []
        
        count=0
        for j, bo in enumerate(preds):
            l = predictors[j](bo)
            #b = b * sigma[j]
            #print("LOGITS: ", l[:10])
            # Compute a tensor per each logit with the difference between the highest and second highest probabilities
            b = score_margin(l)
            #print("B: ", b[:10])
            distributions.append(b)
            logits.append(l)
        
        distributions = torch.stack(distributions, 0)

        if cumulative_threshold:
            distributions[1:] = distributions[1:] * \
                                torch.cumprod(1 - distributions[:-1], 0)
            distributions = torch.cumsum(distributions, dim=0)

        logits = torch.stack(logits, 0)

        for bi in range(x.shape[0]):
            found = False
            for i in range(logits.shape[0]):

                b = distributions[i][bi]

                if b >= epsilon[i]:
                    p = logits[i][bi]
                    pred = torch.argmax(p)
                    if pred == y[bi]:
                        exits_corrected[i] += 1

                    exits_counter[i] += 1

                    found = True
                    break

            if not found:
                i = len(predictors) - 1
                p = logits[i][bi]

                exits_counter[i] += 1
                pred = torch.argmax(p)

                if pred == y[bi]:
                    exits_corrected[i] += 1

    branches_scores = {}
    tot = 0
    correctly_predicted = 0

    for k in range(model.n_branches()):

        correct = exits_corrected[k]
        counter = exits_counter.get(k, 0)

        if counter == 0:
            score = 0
        else:
            score = correct / counter

        branches_scores[k] = score

        tot += counter
        correctly_predicted += correct

    branches_scores['global'] = correctly_predicted / tot

    return branches_scores, exits_counter


@torch.no_grad()
def binary_statistics(model: BranchModel,
                      predictors: nn.ModuleList,
                      dataset_loader):
    model.eval()
    predictors.eval()
    device = get_device(model)

    correct = defaultdict(list)
    incorrect = defaultdict(list)

    for i in range(len(predictors)):
        predictor = predictors[i]

        for x, y in dataset_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)[i]

            pred, hs = predictor(pred)

            pred = torch.argmax(pred, 1)

            for p, h, y in zip(pred, hs, y):
                h = h.item()
                if y == p:
                    correct[i].append(h)
                else:
                    incorrect[i].append(h)

    return dict(correct), dict(incorrect)


@torch.no_grad()
def binary_statistics_cumulative(model: BranchModel,
                                 predictors: nn.ModuleList,
                                 dataset_loader,
                                 th=-1):
    model.eval()
    predictors.eval()
    device = get_device(model)

    correct = defaultdict(list)
    incorrect = defaultdict(list)

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)

        bos = model(x)

        distributions, logits = [], []

        for j, bo in enumerate(bos):
            l, b = predictors[j](bo)
            distributions.append(b)
            logits.append(l)

        preds = torch.stack(logits, 1)
        distributions = torch.stack(distributions, 1)

        a, b = torch.split(distributions,
                           [distributions.shape[1] - 1, 1],
                           dim=1)

        c = torch.cumprod(1 - a, 1)

        cat = torch.cat((torch.ones_like(b), c), 1)
        distributions = distributions * cat
        distributions = torch.cumsum(distributions, 1)

        pred = torch.argmax(preds, -1)

        for pr, hs, y in zip(pred, distributions, y):
            for i, (p, h) in enumerate(zip(pr, hs)):
                if h >= th:
                    h = h.item()
                    if y == p:
                        correct[i].append(h)
                        break
                    else:
                        incorrect[i].append(h)

    return dict(correct), dict(incorrect)

@torch.no_grad()
def get_predictions(model: BranchModel,
                    predictors: nn.ModuleList,
                    dataset_loader):
    
    #Return an array of [true_labels, predicted_labels, probabilities]

    model.eval()
    predictors.eval()
    device = get_device(model)

    N = model.n_branches()  
    probs = list([] for _ in range(N))
    predictions = list([] for _ in range(N))

    true = []

    for x, y in dataset_loader:

        x, y = x.to(device), y.to(device)

        bos = model(x)

        true.extend(y.tolist())

        for j, bo in enumerate(bos):
            l, _ = predictors[j](bo)
            prob = torch.nn.functional.softmax(l, -1)
            probs[j].extend(prob.tolist())
            p=torch.argmax(l, dim=1)
            predictions[j].extend(p.tolist())
    
    true = np.asarray(true)
    for j in range(N):
        predictions = np.asarray(predictions)
        probs = np.asarray(probs)
    
    return true, predictions, probs

@torch.no_grad()
def ece_score(model, predictors, dataset_loader, bins=30):

    true, predictions, probs = get_predictions(model,predictors, dataset_loader)

    stats=[]

    for j in range(model.n_branches()):

        ps = np.max(probs[j], -1)

        prob_pred = np.zeros((0,))
        prob_true = np.zeros((0,))
        ece = 0

        mce = []

        for b in range(1, int(bins) + 1):
            i = np.logical_and(ps <= b / bins, ps > (b - 1) / bins)  # indexes for p in the current bin
            s = np.sum(i)

            if s == 0:
                prob_pred = np.hstack((prob_pred, 0))
                prob_true = np.hstack((prob_true, 0))
                continue

            m = 1 / s
            acc = m * np.sum(predictions[j][i] == true[i])
            conf = np.mean(ps[i])

            prob_pred = np.hstack((prob_pred, conf))
            prob_true = np.hstack((prob_true, acc))
            diff = np.abs(acc - conf)

            mce.append(diff)

            ece += (s / len(true)) * diff

        stats.append((ece, prob_pred, prob_true, mce))

    return stats