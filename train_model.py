import json
import os
import torch
from ofa_evaluator import OFAEvaluator
from train_utils import get_dataset, train_mix, train, validate
import torch.nn as nn

def get_subnet():
    config = 'net.subnet'
    n_classes = 10
    supernet = 'supernets/ofa_mbv3_d234_e346_k357_w1.0'
    ofa_name = 'ofa_qmbv3'
    pretrained = True
    net_config = json.load(open(config))
    evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet, pretrained=pretrained)
    subnet, _ = evaluator.sample(net_config)
    return subnet


from train_utils import get_dataset

# Download and prepare the CIFAR-10 dataset
train_set, test_set, _, _ = get_dataset('cifar10', augmentation=True, resolution=84)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)


# Create an instance of the model
model = get_subnet() 
params = sum(p.numel() for p in model.parameters())

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
model.to(device)


#Train the FP32 model for few epochs
if os.path.exists('ckpt.pth'):
    print('checkpoint found')
    checkpoint = torch.load('ckpt.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('start training')
    train(train_loader,val_loader,5,model,device,criterion,optimizer,100, 'ckpt.pth')

validate(val_loader, model,device,100)


'''
#Train the FP32 model for few epochs
if os.path.exists('ckpt2.pth'):
    print('checkpoint found')
    checkpoint = torch.load('ckpt2.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('start training')
    train_mix(train_loader,val_loader,5,model,device,criterion,optimizer,100, 'ckpt2.pth')

validate(val_loader, model,device,100)
'''
