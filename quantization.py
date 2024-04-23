import json
import os
from ofa_evaluator import OFAEvaluator

import torch
import torch.nn as nn
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
from torch.ao.quantization import FakeQuantize, QConfig
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
import torch.ao.quantization.quantize_fx as quantize_fx
from train_utils import Log, get_data_loaders, get_dataset, get_lr_scheduler, load_checkpoint, train, validate, initialize_seed
import copy

from torch.utils.data import random_split, DataLoader

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

def get_subnet():
    config = 'net.subnet'
    n_classes = 10
    supernet = 'supernets/ofa_mbv3_d234_e346_k357_w1.0'
    ofa_name = 'ofa_mbv3'
    pretrained = True
    net_config = json.load(open(config))
    evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet, pretrained=pretrained)
    subnet, _ = evaluator.sample(net_config)
    return subnet, net_config

# Download and prepare the CIFAR-10 dataset
initialize_seed(42, use_cuda=True)
dataset = 'cifar10'
batch_size = 128
val_split = 0.1
res = 32
eval_test = True
train_loader, val_loader, test_loader = get_data_loaders(dataset=dataset, batch_size=batch_size, threads=4, 
                                            val_split=val_split, img_size=res, augmentation=True, eval_test=eval_test)
    
if val_loader is None:
    val_loader = test_loader

# a tuple of one or more example inputs are needed to trace the model
example_inputs = next(iter(train_loader))[0]

# Create an instance of the model
model_fp, net_config = get_subnet() 
params = sum(p.numel() for p in model_fp.parameters())
print("PARAMS: ",str(params))
#print(model)

depth = net_config['d']
qmask = [1,0,1,0,1]
n_blocks = len(depth)
idx=0 
qlist=[]
for i in range(n_blocks):
    idx += depth[i]
    qlist.append(idx)

print("QLIST: ", qlist)

# Training loop
epochs=2
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_fp.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
log = Log(log_each=100)
scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=epochs)

#Train the FP32 model for few epochs
if os.path.exists('ckptMBV3.pth'):
    print('checkpoint found')
    load_checkpoint(model_fp, optimizer, device, filename='ckptMBV3.pth')
else:
    print('start standard training')
    train(train_loader,val_loader,epochs,model_fp,device,optimizer,criterion,scheduler,log,'ckptMBV3.pth')
print('FP32 model')
print_model_size(model_fp)
validate(test_loader, model_fp, device, 100)


# Custom config

'''
BA = 8  # 7, 6, 5, 4, 3, 2
act = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, 
quant_min=int(-(2 ** BA) / 2), quant_max=int((2 ** BA) / 2 - 1), 
#zero_point=0, 
dtype=torch.qint8, 
qscheme=torch.per_tensor_symmetric, 
                                 reduce_range=False) 

BW = 8  # 7, 6, 5, 4, 3, 2
weights = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, 
quant_min=int(-(2 ** BW) / 2), quant_max=int((2 ** BW) / 2 - 1), 
#zero_point=0, 
dtype=torch.qint8, 
qscheme=torch.per_tensor_symmetric, 
                                 reduce_range=False)

qconfig = QConfig(weight=weights, activation=act)
'''

qconfig=QConfig(weight=torch.quantization.default_qconfig.weight, activation=torch.quantization.default_qconfig.activation)
qconfig_mapping = get_default_qconfig_mapping("x86")

# Apply custom qconfigs to the specified blocks
quant_on=qlist[0]
i=0
for idx,(name,block) in enumerate(model_fp.named_modules()):
    if (i<len(qlist) and idx==qlist[i]): 
        quant_on = qmask[i]
        print("Quantization switched to ", quant_on)
        i+=1
    if quant_on: #if layer of a block to quantize
        qconfig_mapping.add(name, qconfig)

'''
# Dynamic quantization
print("Dynamic quantization")

# Copy model to quantize
device = torch.device('cpu')
model_to_quantize = copy.deepcopy(model_fp).to(device)
model_to_quantize.eval()
qconfig = torch.ao.quantization.default_dynamic_qconfig
qconfig_mapping = QConfigMapping().set_global(qconfig)

# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, 
                  example_inputs)
# no calibration needed when we only have dynamic/weight_only quantization
# quantize
model_quantized_dynamic = quantize_fx.convert_fx(model_prepared)
#model_quantized_dynamic
print_model_size(model_quantized_dynamic)
validate(test_loader, model_quantized_dynamic, device, 100)
'''

# Post-training static quantization
print("Post-training static quantization")
device = torch.device('cpu')
model_to_quantize = copy.deepcopy(model_fp).to(device)
#qconfig_mapping = get_default_qconfig_mapping("x86")
#qconfig_mapping = QConfigMapping().set_global(qconfig)
model_to_quantize.eval()
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
# calibrate 
with torch.no_grad():
    for i in range(20):
        batch = next(iter(train_loader))[0]
        output = model_prepared(batch.to(device))
# quantize
model_quantized_static = quantize_fx.convert_fx(model_prepared)
print_model_size(model_quantized_static)
validate(test_loader, model_quantized_static,device,100)

'''

# Quantization-aware training

device = torch.device('cuda')

#qconfig_mapping = get_default_qat_qconfig_mapping("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)
model_to_quantize = copy.deepcopy(model_fp)
model_to_quantize.train()
# prepare
model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

#Train the FP32 model for few epochs
if os.path.exists('ckptQMBV3.pth'):
    print('checkpoint found')
    checkpoint = torch.load('ckptQMBV3.pth')
    model_prepared.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('start QAT training')
    device = torch.device('cuda')
    train(train_loader,val_loader,epochs,model_prepared,device,optimizer,criterion,scheduler,log,'ckptQMBV3.pth')

# quantize
device = torch.device('cpu')
model_prepared.to(device)
model_quantized_trained = quantize_fx.convert_fx(model_prepared)
print_model_size(model_quantized_trained)
validate(test_loader, model_quantized_trained,device,100)
'''