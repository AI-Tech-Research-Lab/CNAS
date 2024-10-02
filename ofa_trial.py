import json
import logging
import os
import numpy as np
import torch
from NasSearchSpace.ofa.evaluator import OFAEvaluator
from quantization.drift import train_with_drift, validate_drift
from quantization.quant_dorefa import quantize_layers
from train_utils import Log, get_data_loaders, get_loss, get_lr_scheduler, get_optimizer, initialize_seed, load_checkpoint, train, validate

dataset = 'cifar10'
res = 160
args = {'dataset': dataset, 'batch_size': 128, 'n_workers': 4, 'val_split': 0.1, 'eval_test': False, 'epochs': 5, 'seed': 12,
        'optim': 'sam', 'learning_rate': 0.1, 'momentum': 0.9, 'weight_decay': 5e-5, 'rho': 2.0, 'adaptive': True, 'nesterov': False, 'lr_min': 0, 'output_path': 'results/prova-ofa'}
from types import SimpleNamespace

# Convert dictionary to SimpleNamespace object
os.makedirs(args['output_path'], exist_ok=True)
args = SimpleNamespace(**args)
initialize_seed(args.seed,  torch.cuda.is_available())
device='cuda'
train_loader, val_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=args.n_workers, 
                                            val_split=args.val_split, img_size=res, augmentation=True, eval_test=args.eval_test)
ofa = OFAEvaluator(n_classes=10,model_path='NasSearchSpace/ofa/supernets/ofa_supernet_resnet50',pretrained=True)
#read config from json file

'''
with open(os.path.join(args.output_path,'net.subnet'), 'r') as f:
    config = json.load(f)
'''
model,config = ofa.sample()
print(config)
#model = ofa.sample(config)[0]
# save config to json file
with open(os.path.join(args.output_path,'net.subnet'), 'w') as f:
    json.dump(config, f)

#model = quantize_layers(model)
model.to(device)
epochs = args.epochs
results={}

log = Log(log_each=10)

optimizer = get_optimizer(model.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay, args.rho, args.adaptive, args.nesterov)

criterion = get_loss('ce')

scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=epochs, lr_min=args.lr_min)

if (os.path.exists(os.path.join(args.output_path,'ckpt.pth'))):
    model, optimizer = load_checkpoint(model, optimizer, device, os.path.join(args.output_path,'ckpt.pth'))
    logging.info("Loaded checkpoint")
    top1 = validate(val_loader, model, device, print_freq=100)
else:
    logging.info("Start training...")
    top1, model, optimizer = train(train_loader, val_loader, epochs, model, device, optimizer, criterion, scheduler, log, ckpt_path=os.path.join(args.output_path,'ckpt.pth'))
    logging.info("Training finished")

results['top1']=np.round(top1,2)
model = quantize_layers(model, nbit_w=4, nbit_a=8, q_alpha_w=0.5625, q_alpha_a=1)

log = Log(log_each=10)

optimizer = get_optimizer(model.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay, args.rho, args.adaptive, args.nesterov)

criterion = get_loss('ce')

scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=epochs, lr_min=args.lr_min)

if (os.path.exists(os.path.join(args.output_path,'ckptq.pth'))):
    model, optimizer = load_checkpoint(model, optimizer, device, os.path.join(args.output_path,'ckptq.pth'))
    logging.info("Loaded checkpoint")
    top1 = validate(val_loader, model, device, print_freq=100)
else:
    logging.info("Start training...")
    top1, model, optimizer = train_with_drift(train_loader, val_loader, epochs, model, device, optimizer, criterion, scheduler, log, ckpt_path=os.path.join(args.output_path,'ckptq.pth'))
    logging.info("Training finished")

results['top1q']=np.round(top1,2)
# Assuming validate_drift returns a tuple (avg_top1_drift, top1_drifts)
avg_top1_drift, top1_drifts = validate_drift(val_loader, model, device)

# Round avg_top1_drift separately
avg_top1_drift = round(avg_top1_drift, 2)

# Round each accuracy in top1_drifts separately
top1_drifts = np.round(top1_drifts, 2).tolist()

# Save the results
results['avg_top1_drift'] = avg_top1_drift
results['top1_drifts'] = top1_drifts

with open(os.path.join(args.output_path,'results.json'), 'w') as f:
    json.dump(results, f)