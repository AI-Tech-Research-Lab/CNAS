import torch
import os
from torchsummary import summary
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.quantization import QuantStub, DeQuantStub, FakeQuantize, QConfig
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

import json
import os
from ofa_evaluator import OFAEvaluator

config = 'net.subnet'
n_classes = 10
supernet = 'supernets/ofa_mbv3_d234_e346_k357_w1.0'
ofa_name = 'ofa_qmbv3'
pretrained = True
net_config = json.load(open(config))
evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet, model_name=ofa_name, pretrained=pretrained)
subnet, _ = evaluator.sample(net_config)
#print(subnet)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Update input channels to 3
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        # Fully connected layer
        # Update the input size based on CIFAR-10 image size (32x32 pixels for CIFAR-10)
        self.fc = nn.Linear(64 * 32 * 32, 10)  # Update output classes to 10

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Input: [batch_size, 3, 32, 32] for CIFAR-10

        x = self.quant(x)
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Flatten before fully connected layer
        x = x.reshape(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        x = self.dequant(x)
        
        # Output: [batch_size, 10] for CIFAR-10 (10 classes)
        return x

'''
class QuantWrapper(nn.Module):
    def __init__(self, model):
        super(QuantWrapper, self).__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Perform quantization before the forward pass through the model
        x = self.quant(x)
        
        # Forward pass through the model
        x = self.model(x)
        
        # Perform dequantization at the end
        x = self.dequant(x)
        return x
'''

# Download and prepare the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# Set the number of worker processes (adjust the value according to your system)
num_workers = 4

# Training dataset and loader
train_dataset = datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

# Validation dataset and loader
val_dataset = datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

# Create an instance of the model
model = CNNModel()
params = sum(p.numel() for p in model.parameters())
print("PARAMS: ",str(params))

# BUILT-IN QCONFIG
#quantization_config = torch.quantization.get_default_qat_qconfig('fbgemm')

# CUSTOM QCONFIG, B BITS

B = 8  # 7, 6, 5, 4, 3, 2
act = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, 
quant_max=int(2 ** B - 1), dtype=torch.quint8, qscheme=torch.per_tensor_affine, 
reduce_range=False)

weights = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, 
quant_min=int(-(2 ** B) / 2), quant_max=int((2 ** B) / 2 - 1), dtype=torch.qint8, 
qscheme=torch.per_channel_symmetric, reduce_range=False)

model.qconfig = QConfig(activation=act, weight=weights)

# Apply quantization to the model
#model.qconfig = quantization_config
model = torch.quantization.prepare_qat(model, inplace=True)
#model = QuantWrapper(model)
#print(model)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from train_utils import train,validate
if os.path.exists('checkpoint.pth'):
    print('checkpoint found')
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('start training')
    train(train_loader,val_loader,1,model,device,criterion,optimizer,100)

print("MODEL STATS BEFORE QUANT")
params = sum(p.numel() for p in model.parameters())
print("PARAMS: ",str(params))
validate(val_loader, model,device,100)

torch.quantization.convert(model, inplace=True)

print("MODEL STATS AFTER QUANT")
params = sum(p.numel() for p in model.parameters())
print("PARAMS: ",str(params))
validate(val_loader, model,device,100)