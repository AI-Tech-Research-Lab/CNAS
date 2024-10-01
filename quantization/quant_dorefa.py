import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
from scipy import io

def quantize_layers(module):
    """
    Recursively replaces Conv2d and Linear layers with QuanConv and Linear_Q layers in the given module.
    """

    # If the current module is Conv2d, replace with QuanConv
    if isinstance(module, nn.Conv2d):
        return QuanConv(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            nbit_w=4,
            nbit_a=8,
            q_alpha_w=0.5625,
        )
    
    # If the current module is Linear, replace with Linear_Q
    elif isinstance(module, nn.Linear):
        return Linear_Q(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            nbit_w=4,
            nbit_a=8,
            q_alpha_w=0.5625,
        )
    
    # Recursively go through all children (layers) of the module
    for name, child in module.named_children():
        if name !='downsample': # no quantization on skip connections
            # Replace the child with the recursively modified version
            setattr(module, name, quantize_layers(child))
    
    return module

class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


def get_drift(quantized_w, drift_values):

    idx = torch.abs(quantized_w).long() - 1 

    idx = idx.float()

    # Mask for zero values, which should remain unchanged
    zero_mask = quantized_w == 0

    new_weight = torch.where(zero_mask, torch.tensor(0.0, device=quantized_w.device), 
                             torch.where(quantized_w < 0, -drift_values[idx.long()], drift_values[idx.long()]))
    return new_weight.float().to(quantized_w.device)

class Quantizer(Function):

    @staticmethod
    def forward(ctx, input, nbit, alpha, drift):
        scale = (2 ** nbit) * alpha - 1
        if drift is not None:
            return get_drift(torch.round(input * scale),drift) / scale
        else:
            return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def quantize(input, nbit, alpha=1, drift=None):
    return Quantizer.apply(input, nbit, alpha, drift)


def dorefa_w(w, nbit_w, alpha=1, drift=None):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) # w in [-0.5, 5]
        w = 2 * quantize(w, nbit_w, alpha=alpha, drift=drift) #- 1

    return w


def dorefa_a(input, nbit_a, alpha=1, drift=None):
    return quantize(torch.clamp(0.1 * input, 0, 1), nbit_a, alpha=alpha, drift=drift)


class QuanConv(nn.Conv2d):
    """docstring for QuanConv"""
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=1,
                 nbit_a=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, q_alpha_w=1, drift_w=None, q_alpha_a=1, drift_a=None):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        self.q_alpha_w = q_alpha_w
        self.q_alpha_a = q_alpha_a
        self.drift_w=None
        self.drift_a=None
        if self.drift_w is not None:
            self.drift_w = torch.tensor(drift_w, requires_grad=False).to('cuda')
        if self.drift_a is not None:
            self.drift_a = torch.tensor(drift_a, requires_grad=False).to('cuda')

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.q_alpha_w, self.drift_w)
        else:
            w = self.weight

        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a, self.q_alpha_a, self.drift_a)
        else:
            x = input
        # print('x unique',np.unique(x.detach().numpy()).shape)
        # print('w unique',np.unique(w.detach().numpy()).shape)

        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output

class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=1, nbit_a=1, q_alpha_w=1, drift_w=None,
                 q_alpha_a=1, drift_a=None):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        self.q_alpha_w = q_alpha_w
        self.q_alpha_a = q_alpha_a
        self.drift_w=None
        self.drift_a=None
        if drift_w is not None:
            self.drift_w = torch.tensor(drift_w, requires_grad=False).to('cuda')
        if drift_a is not None:
            self.drift_a = torch.tensor(drift_a, requires_grad=False).to('cuda')

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.q_alpha_w, self.drift_w)
        else:
            w = self.weight

        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a, self.q_alpha_a, self.drift_a)
        else:
            x = input

        # print('x unique',np.unique(x.detach().numpy()))
        # print('w unique',np.unique(w.detach().numpy()))

        output = F.linear(x, w, self.bias)

        return output
    
# Test case: example with nbit=4 and alpha=0.5625


def matrix_weights_time():
    # Replace 'your_file.mat' with the path to your .mat file
    mat_contents = io.loadmat('4bit.mat')

    # mat_contents is now a dictionary containing the data from the .mat file
    # You can access the variables like this:
    for key in mat_contents:
        print(key, type(mat_contents[key]))

    data = mat_contents["ww_mdn"]


    # Determine the column indices to select (odd indices starting from 9)
    start_index = 10  # MATLAB index 911 corresponds to Python index 9
    odd_indices = np.arange(start_index, data.shape[0], 2)  # Generates 11, 13, ...

    # Create the new matrix with only the selected columns
    new_matrix = data[odd_indices,:]
    return new_matrix #returns matrix with temporal values from 10 to 40 microsiemens (shape: 4x8). Value 0 is always the same

'''
nbit = 4
alpha = 0.5625
w = torch.Tensor([[0.2793, 0.7009, 0.2287, 0.5550, 0.2293, 0.1877, 0.1774, 0.0294],
        [-0.6341, -0.3801, -0.3883, -0.1847, -0.2800, -0.6215, -0.3160, -0.4441],
        [0.2317, 0.4090, 0.3556, 0.4593, 1.0000, 0.6137, 0.1831, 0.4997],
        [-0.6911, -0.1305, -0.4234, -0.5113, -0.2601, -0.7405, -0.4078, 0.0000]]) #torch.randn(4, 8)  # Example input tensor (4x8 matrix)
# Get the archive data and move to the same device as quantized_w
w = w / (2 * torch.max(torch.abs(w))) 

archive = matrix_weights_time()
print(archive)
drift=0
# Select the appropriate column for this timestep
#drift_values = archive[:, drift] / 1e-5
drift_values = None
#print("Drift Values:\n", drift_values.detach().cpu().numpy())

# Perform quantization
print("Input Tensor:\n", w)
quantized_tensor = quantize(w, nbit, alpha, drift=drift_values)
print("Quantized Tensor without Drift:\n", quantized_tensor)

# Loop through drift values from 1 to 7
for drift in range(1, 8):
    drift_values = archive[:, drift] / 1e-5
    drift_values_tensor = torch.tensor(drift_values, requires_grad=False)
    
    # Print drift values
    print(f"Drift Values for drift={drift}:\n", drift_values_tensor)

    # Perform quantization
    quantized_tensor = quantize(w, nbit, alpha, drift=drift_values_tensor)
    print(f"Quantized Tensor with Drift={drift}:\n", quantized_tensor)
'''