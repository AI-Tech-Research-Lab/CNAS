import logging

import numpy as np
import torch
from torch import nn

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

import os

from utils import get_device
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

# Function to test what classes performed well
def calc_accuracy(device, model, loader):
    correct = 0
    num_total = len(loader.dataset)
    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            correct += (torch.sum(c)).item()
    return 100 * correct / num_total


#standard: train the network with only the final exit (last element of predictors)
def standard_trainer(model,
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

    bar = tqdm(range(epochs), leave=True)

    if early_stopping is not None:
        early_stopping.reset()

    model.train()

    for epoch in bar:
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

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
            
            eval_scores = calc_accuracy(model, eval_loader, device)
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

                best_model_i = epoch
        else:
            if (eval_scores is not None and eval_scores >= best_eval_score) \
                    or eval_scores is None:

                if eval_scores is not None:
                    best_eval_score = eval_scores

                best_model = deepcopy(model.state_dict())

                best_model_i = epoch

        train_scores = calc_accuracy(model=model,
                                     loader=train_loader,
                                     device=device)

        test_scores = calc_accuracy(model=model,
                                    loader=test_loader,
                                    device=device)
        
        bar.set_postfix({'Train score': train_scores, 'Test score': test_scores,'Eval score': eval_scores if eval_scores != 0 else 0, 'Mean loss': mean_loss})

        scores.append((train_scores, eval_scores, test_scores))
    
    # Remove checkpoint 
    if os.path.exists(ckpt_path):
      os.remove(ckpt_path)

    return best_model, \
           scores, \
           mean_losses