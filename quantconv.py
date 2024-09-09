import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import quantizers as quantizers

class DynamicQConv(nn.Conv2d):
    # dynamic quantization for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w, quan_name_a, list_nbit, has_offset=False, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DynamicQConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbit_w = list_nbit[0]
        self.nbit_a = list_nbit[1]
        name_w_dict = {'dorefa': quantizers.dorefa_w, 'pact': quantizers.dorefa_w}
        name_a_dict = {'dorefa': quantizers.dorefa_a, 'pact': quantizers.pact_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        
        if quan_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_a', None)
        if quan_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_w', None)
        if has_offset:
            self.offset = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('offset', None)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_custome_parameters()
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.alpha_a is not None:
            nn.init.constant_(self.alpha_a, 10)
        if self.alpha_w is not None:
            nn.init.constant_(self.alpha_w, 10)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)
    
    def forward(self, input):
        '''
        count = 0
        x = 0
        for nbit in self.list_nbit:
            w0 = self.quan_w(self.weight, nbit, self.alpha_w, self.offset)
            x0 = self.quan_a(input, nbit, self.alpha_a)
            x0 = F.conv2d(x0, w0, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x += x0*mask[:,count].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            count += 1
        '''
        w0 = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        x0 = self.quan_a(input, self.nbit_a, self.alpha_a)
        x = F.conv2d(x0, w0, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x