import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class Signer(Function):
    '''
    take a real value x
    output sign(x)
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign(input):
    return Signer.apply(input)



# quantize for weights and activations
class Quantizer(Function):
    '''
    take a real value x in alpha*[0,1] or alpha*[-1,1]
    output a discrete-valued x in alpha*{0, 1/(2^k-1), ..., (2^k-1)/(2^k-1)} or likeness
    where k is nbit
    '''
    @staticmethod
    def forward(ctx, input, nbit, alpha=None, offset=None):
        ctx.alpha = alpha
        ctx.offset = offset
        scale = (2 ** nbit - 1) if alpha is None else (2 ** nbit - 1) / alpha
        ctx.scale = scale
        return torch.round(input * scale) / scale if offset is None \
                else (torch.round(input * scale) + torch.round(offset)) / scale

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.offset is None:
            return grad_output, None, None, None
        else:
            return grad_output, None, None, torch.sum(grad_output) / ctx.scale


def quantize(input, nbit, alpha=None, offset=None):
    return Quantizer.apply(input, nbit, alpha, offset)

# sign in dorefa-net for weights
class ScaleSigner(Function):
    '''
    take a real value x
    output sign(x) * E(|x|)
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    
def scale_sign(input):
    return ScaleSigner.apply(input)

def dorefa_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w

# dorefa quantize for activations
def dorefa_a(input, nbit_a, *args, **kwargs):
    return quantize(torch.clamp(input, 0, 1), nbit_a, *args, **kwargs)

# PACT quantize for activations
def pact_a(input, nbit_a, alpha, *args, **kwargs):
    x = 0.5*(torch.abs(input)-torch.abs(input-alpha)+alpha)
    return quantize(x, nbit_a, alpha, *args, **kwargs)