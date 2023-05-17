from functools import reduce
from operator import mul

import torch
from torch import nn


def conv_cost(image_shape, m):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
    bias_ops = 1 if m.bias is not None else 0
    curr_im_size = (image_shape[1] / m.stride[0], image_shape[2] / m.stride[1])

    # curr_im_size = (hparams['im_size'][0] / m.stride[0], hparams['im_size'][1] / m.stride[1])
    cost = m.kernel_size[0] * m.kernel_size[
        1] * m.in_channels * m.out_channels * curr_im_size[0] * curr_im_size[1]

    # nelement = reduce(mul, curr_im_size, 1)
    #
    # total_ops = nelement * (m.in_channels // m.groups * kernel_ops + bias_ops)

    # curr_im_size = (hparams['im_size'][0] / m.stride[0], hparams['im_size'][1] / m.stride[1])
    # cost = m.kernel_size[0]*\
    #        m.kernel_size[1]*\
    #        m.in_channels*\
    #        m.out_channels*\
    #        curr_im_size[0]*\
    #        curr_im_size[1]

    return cost


#
#
def maxpool_cost(image_shape, m):
    cost = image_shape[1] * image_shape[2] * image_shape[0]
    return cost


def sequential_cost(input_sample, m):
    cost = 0
    for name, m_int in m.named_children():
        image_shape = input_sample.shape[1:]
        c = module_cost(input_sample, m_int)
        input_sample = m_int(input_sample)
        cost += c
    return cost


# def maxpool_cost(hparams, m):
#     cost = hparams['im_size'][0]*hparams['im_size'][1]*hparams['n_channels']
#     hparams['im_size'] = (hparams['im_size'][0]/m.kernel_size, hparams['im_size'][1]/m.kernel_size)
#     return cost, hparams

avgpool_cost = maxpool_cost  # To check


def dense_cost(hparams, m):
    cost = m.in_features * m.out_features
    return cost, hparams


def module_cost(input_sample, m):
    image_shape = input_sample.shape[1:]

    if isinstance(m, nn.Conv2d):
        cost = conv_cost(image_shape, m)
    elif isinstance(m, nn.MaxPool2d):
        cost = maxpool_cost(image_shape, m)
    elif isinstance(m, nn.AvgPool2d):
        cost = avgpool_cost(image_shape, m)
    elif isinstance(m, nn.Linear):
        cost = dense_cost(image_shape, m)
    elif isinstance(m, nn.Sequential):  # == ['Sequential', 'BasicBlock']:
        cost = sequential_cost(input_sample, m)
    else:
        cost = 0

    return cost
