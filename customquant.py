from quantconv import DynamicQConv
import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from train_utils import Log, get_loss, get_lr_scheduler, load_checkpoint, train, validate
from torchvision import transforms
import torchvision.datasets as datasets

def replace_conv_with_dynamic_conv(model, quan_name_w='dorefa', quan_name_a='dorefa', list_nbit=[8, 8], has_offset=False):
    """
    Replace all Conv2d layers in the ResNet with DynamicQConv layers.
    
    Args:
        model (torch.nn.Module): Pre-trained ResNet model.
        quan_name_w (str): Quantization method for weights (e.g., 'dorefa', 'pact').
        quan_name_a (str): Quantization method for activations (e.g., 'dorefa', 'pact').
        list_nbit (list): List of bits for weight and activation quantization [nbit_w, nbit_a].
        has_offset (bool): Whether to include an offset parameter in the quantization.
        
    Returns:
        torch.nn.Module: Modified model with DynamicQConv layers.
    """
    
    # Iterate over the layers of the model
    for name, module in model.named_children():
        # If the module is a Conv2d layer, replace it with DynamicQConv
        if isinstance(module, nn.Conv2d):
            dynamic_conv = DynamicQConv(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                quan_name_w=quan_name_w,
                quan_name_a=quan_name_a,
                list_nbit=list_nbit,
                has_offset=has_offset,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None
            )
            
            # Transfer the pre-trained weights and bias
            dynamic_conv.weight.data = module.weight.data
            if module.bias is not None:
                dynamic_conv.bias.data = module.bias.data
            
            # Replace the original Conv2d layer with DynamicQConv
            setattr(model, name, dynamic_conv)
        
        # If it's a sequential or another nn.Module, apply the function recursively
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.Module):
            replace_conv_with_dynamic_conv(module, quan_name_w, quan_name_a, list_nbit, has_offset)
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ",device)
# 1. Load the pretrained ResNet model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

#Train FP model
if os.path.exists('ckpt_fp.pth'):
    print('checkpoint found')
    model, optimizer = load_checkpoint(model, optimizer, device, 'ckpt_fp.pth')
else:
    print('start training')
    train(train_loader,val_loader,n_epochs,model,device,optimizer,criterion,scheduler,train_log,'ckpt_fp.pth')
validate(val_loader, model,device,100)

# Replace all Conv2d layers with DynamicQConv and transfer weights/bias
quantized_model = replace_conv_with_dynamic_conv(model, quan_name_w='dorefa', quan_name_a='dorefa', list_nbit=[8, 5], has_offset=True)

print(quantized_model)

optimizer = torch.optim.SGD(quantized_model.parameters(), lr=0.01, momentum=0.9)
train_log = Log(log_each=10)
n_epochs=5
scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=n_epochs)

#Quantization Aware Training
if os.path.exists('ckpt_qt.pth'):
    print('checkpoint found')
    quantized_model, optimizer = load_checkpoint(quantized_model, optimizer, device, 'ckpt_qt.pth')
else:
    print('start training')
    train(train_loader,val_loader,n_epochs,quantized_model,device,optimizer,criterion,scheduler,train_log,'ckpt_qt.pth')

print(quantized_model)
validate(val_loader, quantized_model,device,100)

def check_quantized_range(model, nbit):
    for name, param in model.named_parameters():
        if 'weight' in name:
            min_val = torch.min(param.data)
            max_val = torch.max(param.data)
            expected_range = (-2**(nbit-1), 2**(nbit-1) - 1)
            print(f"Layer {name}: min={min_val}, max={max_val}, expected range={expected_range}")

def check_quantized_range_weights_activations(model):
    """
    Checks the quantized range for both weights and activations, considering different bit precision for each.
    
    Args:
        model (torch.nn.Module): The model to check for quantized weights and activations.
    """
    for name, module in model.named_modules():
        # Check if the module is a DynamicQConv layer
        if isinstance(module, DynamicQConv):
            # Get the nbit values for weights and activations
            nbit_w = module.nbit_w
            nbit_a = module.nbit_a
            
            # Expected range for weights based on nbit_w (signed integers)
            expected_range_w = (-2 ** (nbit_w - 1), 2 ** (nbit_w - 1) - 1)  # e.g., for 8-bit: (-127, 127)
            scale_w = (2 ** (nbit_w - 1) - 1)  # Scale factor for quantization

            # Quantize the weights and compute min/max as discrete values using self.quan_w
            w0 = module.quan_w(module.weight.data, nbit_w, module.alpha_w, module.offset)
            quantized_min_w, quantized_max_w = torch.min(w0), torch.max(w0)
            quantized_min_w, quantized_max_w = torch.round(quantized_min_w * scale_w).int(), torch.round(quantized_max_w * scale_w).int()

            print(f"Layer {name} (weights): min={quantized_min_w.item()}, max={quantized_max_w.item()}, expected range={expected_range_w}")

            # Check if the weights are within the expected range
            if quantized_min_w.item() < expected_range_w[0] or quantized_max_w.item() > expected_range_w[1]:
                print(f"Warning: Layer {name} weights are outside the expected quantized range!")

            # Expected range for activations based on nbit_a (unsigned integers)
            expected_range_a = (0, 2 ** nbit_a - 1)  # e.g., for 8-bit: (0, 255)
            scale_a = (2 ** nbit_a - 1)  # Scale factor for quantization

            # Simulate a forward pass to check activations, assuming a random input
            dummy_input = torch.randn(1, module.in_channels, 32, 32)  # Adjust the size as necessary
            with torch.no_grad():
                # Quantize activations in the forward pass using self.quan_a
                x0 = module.quan_a(dummy_input, nbit_a, module.alpha_a)
                quantized_min_a, quantized_max_a = torch.min(x0), torch.max(x0)
                quantized_min_a, quantized_max_a = torch.round(quantized_min_a * scale_a).int(), torch.round(quantized_max_a * scale_a).int()

                print(f"Layer {name} (activations): min={quantized_min_a.item()}, max={quantized_max_a.item()}, expected range={expected_range_a}")
            
            # Check if activations are within the expected range
            if quantized_min_a.item() < expected_range_a[0] or quantized_max_a.item() > expected_range_a[1]:
                print(f"Warning: Layer {name} activations are outside the expected quantized range!")

# Example usage
check_quantized_range_weights_activations(quantized_model)

