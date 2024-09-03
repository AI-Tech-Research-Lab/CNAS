import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare, convert, FakeQuantize, QConfig
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from train_utils import Log, get_loss, get_lr_scheduler, load_checkpoint, train, validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DWVICE: ",device)
# 1. Load the pretrained ResNet model
model = models.resnet18(weights=True)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = get_loss('ce')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_log = Log(log_each=10)
n_epochs=5
scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=n_epochs)

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

# Adding Quantization Stubs
class QuantizedResNet(nn.Module):
    def __init__(self, model_fp):
        super(QuantizedResNet, self).__init__()
        self.quant = QuantStub()  # Quantize input
        self.model = model_fp      # Full precision model
        self.dequant = DeQuantStub()  # Dequantize output
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# Instantiate the quantized model
quantized_model = QuantizedResNet(model)
params = sum(p.numel() for p in quantized_model.parameters())
model.eval()
# Fuse Conv, BN, and ReLU layers
fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)
for module_name, module in model.named_children():
    if "layer" in module_name:
        for basic_block_name, basic_block in module.named_children():
            fuse_modules(basic_block, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], inplace=True)

quantized_model.train()
if os.path.exists('ckpt_fp.pth'):
    print('checkpoint found')
    quantized_model, optimizer = load_checkpoint(quantized_model, optimizer, device, 'ckpt_fp.pth')
else:
    print('start training')
    train(train_loader,val_loader,n_epochs,quantized_model,device,optimizer,criterion,scheduler,train_log,'ckpt_fp.pth')

params = sum(p.numel() for p in quantized_model.parameters())
print("PARAMS BEFORE QUANT: ",str(params))
validate(val_loader, quantized_model,device,100)

# 5. Quantization steps
# Prepare the model for quantization aware training
#quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
#prepare(quantized_model, inplace=True)

# CUSTOM QCONFIG, B BITS

B = 8  # 7, 6, 5, 4, 3, 2
act = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, 
quant_max=int(2 ** B - 1), dtype=torch.quint8, qscheme=torch.per_tensor_affine, 
reduce_range=False)

weights = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, 
quant_min=int(-(2 ** B) / 2), quant_max=int((2 ** B) / 2 - 1), dtype=torch.qint8, 
qscheme=torch.per_channel_symmetric, reduce_range=False)

#quantized_model.qconfig = QConfig(activation=act, weight=weights)
quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

# Apply quantization to the model
#model.qconfig = quantization_config
quantized_model.train()
quantized_model = torch.quantization.prepare_qat(quantized_model, inplace=True)

# Fine-tuning the model after quantization preparation (if needed)
if os.path.exists('ckpt_qt.pth'):
    print('checkpoint found')
    quantized_model, optimizer = load_checkpoint(quantized_model, optimizer, device, 'ckpt_qt.pth')
else:
    print('start training')
    train(train_loader, val_loader, n_epochs, quantized_model, device, optimizer, criterion, scheduler, train_log,'ckpt_qt.pth')


# Convert the model to quantized form
torch.quantization.convert(quantized_model, inplace=True)

# 6. Calculate the number of parameters
params = sum(p.numel() for p in quantized_model.parameters())
print("PARAMS AFTER QUANT: ", str(params))

# 7. Validate the quantized model
device='cpu'
validate(val_loader, quantized_model, device, 100)