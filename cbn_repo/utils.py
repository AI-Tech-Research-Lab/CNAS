import csv
import os

import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset

from models.alexnet import AlexnetClassifier, AlexNet
from models.base import IntermediateBranch, BinaryIntermediateBranch
from torchvision.datasets.folder import default_loader

import torch
from torch import optim, nn
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomCrop, RandomRotation

from models.resnet import resnet20
from ofa.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
from models.mobilenet_v3 import MobileNetV3
#from ofa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3

import json
#from models.vgg import vgg11


class TinyImagenet(Dataset):
    """Tiny Imagenet Pytorch Dataset"""

    filename = ('tiny-imagenet-200.zip',
                'http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(
            self,
            root,
            *,
            train: bool = True,
            transform=None,
            target_transform=None,
            loader=default_loader,
            download=True):
        """
        Creates an instance of the Tiny Imagenet dataset.
        :param root: folder in which to download dataset. Defaults to None,
            which means that the default location for 'tinyimagenet' will be
            used.
        :param train: True for training set, False for test set.
        :param transform: Pytorch transformation function for x.
        :param target_transform: Pytorch transformation function for y.
        :param loader: the procedure to load the instance from the storage.
        :param bool download: If True, the dataset will be  downloaded if
            needed.
        """

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader

        # super(TinyImagenet, self).__init__(
        #     root, self.filename[1], self.md5, download=download, verbose=True)

        self.root = root

        # self._load_dataset()
        self._load_metadata()

    # def _load_dataset(self) -> None:
    #     """
    #     The standardized dataset download and load procedure.
    #     For more details on the coded procedure see the class documentation.
    #     This method shouldn't be overridden.
    #     This method will raise and error if the dataset couldn't be loaded
    #     or downloaded.
    #     :return: None
    #     """
    #     metadata_loaded = False
    #     metadata_load_error = None
    #
    #     try:
    #         metadata_loaded = self._load_metadata()
    #     except Exception as e:
    #         metadata_load_error = e
    #
    #     if metadata_loaded:
    #         if self.verbose:
    #             print('Files already downloaded and verified')
    #         return
    #
    #     if not self.download:
    #         msg = 'Error loading dataset metadata (dataset download was ' \
    #               'not attempted as "download" is set to False)'
    #         if metadata_load_error is None:
    #             raise RuntimeError(msg)
    #         else:
    #             print(msg)
    #             raise metadata_load_error

    def _load_metadata(self) -> bool:
        self.data_folder = self.root / 'tiny-imagenet-200'

        self.label2id, self.id2label = TinyImagenet.labels2dict(
            self.data_folder)
        self.data, self.targets = self.load_data()
        return True

    @staticmethod
    def labels2dict(data_folder):
        """
        Returns dictionaries to convert class names into progressive ids
        and viceversa.
        :param data_folder: The root path of tiny imagenet
        :returns: label2id, id2label: two Python dictionaries.
        """

        label2id = {}
        id2label = {}

        with open(str(data_folder / 'wnids.txt'), 'r') as f:

            reader = csv.reader(f)
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        return label2id, id2label

    def load_data(self):
        """
        Load all images paths and targets.
        :return: train_set, test_set: (train_X_paths, train_y).
        """

        data = [[], []]

        classes = list(range(200))
        for class_id in classes:
            class_name = self.id2label[class_id]

            if self.train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            data[0] += X
            data[1] += Y

        return data

    def get_train_images_paths(self, class_name):
        """
        Gets the training set image paths.
        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        train_img_folder = self.data_folder / 'train' / class_name / 'images'

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def get_test_images_paths(self, class_name):
        """
        Gets the test set image paths
        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """

        val_img_folder = self.data_folder / 'val' / 'images'
        annotations_file = self.data_folder / 'val' / 'val_annotations.txt'

        valid_names = []

        # filter validation images by class using appropriate file
        with open(str(annotations_file), 'r') as f:

            reader = csv.reader(f, dialect='excel-tab')
            for ll in reader:
                if ll[1] == class_name:
                    valid_names.append(ll[0])

        img_paths = [val_img_folder / f for f in valid_names]

        return img_paths

    def __len__(self):
        """ Returns the length of the set """
        return len(self.data)

    def __getitem__(self, index):
        """ Returns the index-th x, y pattern of the set """

        path, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_device(model: nn.Module):
    return next(model.parameters()).device


class EarlyStopping:
    def __init__(self, tolerance, min=True, **kwargs):
        self.initial_tolerance = tolerance
        self.tolerance = tolerance
        self.min = min

        if self.min:
            self.current_value = np.inf
            self.c = lambda a, b: a < b
        else:
            self.current_value = -np.inf
            self.c = lambda a, b: a > b

    def step(self, v):
        if self.c(v, self.current_value):
            self.tolerance = self.initial_tolerance
            self.current_value = v
            return 1
        else:
            self.tolerance -= 1
            if self.tolerance <= 0:
                return -1
        return 0

    def reset(self):
        self.tolerance = self.initial_tolerance
        self.current_value = 0
        if self.min:
            self.current_value = np.inf
            self.c = lambda a, b: a < b
        else:
            self.current_value = -np.inf
            self.c = lambda a, b: a > b


def get_intermediate_classifiers(model,
                                 image_size,
                                 num_classes,
                                 binary_branch=False,
                                 fix_last_layer=False):
    predictors = nn.ModuleList()
    x = torch.randn((1,) + image_size)
    model.eval() # this avoids batch norm error https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
    #outputs = get_blocks_mbv3(num_classes,x)
    outputs = model(x)

    for i, o in enumerate(outputs):
        chs = o.shape[1]

        if i == (len(outputs) - 1):
            od = torch.flatten(o, 1).shape[-1]

            if binary_branch:
                if fix_last_layer:
                    linear_layers = nn.Sequential(*[nn.ReLU(),
                                                    nn.Linear(od, num_classes)])

                    b = BinaryIntermediateBranch(preprocessing=nn.Flatten(),
                                                 classifier=linear_layers,
                                                 return_one=True)
                else:
                    linear_layers = nn.Sequential(*[nn.ReLU(),
                                                    nn.Linear(od,
                                                              num_classes + 1)])

                    b = BinaryIntermediateBranch(preprocessing=nn.Flatten(),
                                                 classifier=linear_layers,
                                                 )
            else:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, num_classes)])

                b = IntermediateBranch(preprocessing=nn.Flatten(),
                                       classifier=linear_layers)

            predictors.append(b)
        else:

            if o.shape[-1] >= 6:
                seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(chs, 128,
                              kernel_size=3, stride=1),
                    nn.MaxPool2d(3),
                    nn.ReLU())
            else:
                seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(chs, 128,
                              kernel_size=2, stride=1),
                    # nn.MaxPool2d(2),
                    nn.ReLU())

            seq.add_module('flatten', nn.Flatten())

            output = seq(o)
            output = torch.flatten(output, 1)
            od = output.shape[-1]

            if binary_branch:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, num_classes + 1)])

                predictors.append(
                    BinaryIntermediateBranch(preprocessing=seq,
                                             classifier=linear_layers,
                                             ))

            else:
                linear_layers = nn.Sequential(*[nn.ReLU(),
                                                nn.Linear(od, num_classes)])

                predictors.append(IntermediateBranch(preprocessing=seq,
                                                     classifier=linear_layers))
            predictors[-1](o)

    return predictors

## OFA ##

import torch
from torch.nn import Conv2d,ReLU,Linear,Sequential,Flatten,BatchNorm2d,AvgPool2d,MaxPool2d
import copy
import torch.backends.cudnn as cudnn
from torchprofile import profile_macs


def mobilenetv3(classes, path):
    # current path example ConfidenceBranchNetwork/outputs/cifar10/mobilenetv3/bernulli_logits
    #'./../../../../ofa/net.subnet'
    config = json.load(open(path))
    ofa = OFAMobileNetV3(n_classes=classes)
    ofa.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'])
    subnet = ofa.get_active_subnet(preserve_weight=True) # OFA MobileNetV3
    net = MobileNetV3(subnet.first_conv,subnet.blocks, subnet.final_expand_layer, subnet.feature_mix_layer) # Branch MobileNetV3
    return net


target_layers = {'Conv2D':Conv2d,
                 'Flatten':Flatten,
                 'Dense':Linear,
                 'BatchNormalization':BatchNorm2d,
                 'AveragePooling2D':AvgPool2d,
                 'MaxPooling2D':MaxPool2d
                 }

activations = {}

def hook_fn(m, i, o):
    #if (o.shape != NULL):
    activations[m] = [i,o]#.shape  #m is the layer

def get_all_layers(net):
  layers = {}
  names = {}
  index = 0
  for name, layer in net.named_modules():#net._modules.items():
    #print(name)
    layers[index] = layer
    names[index] = name
    index = index + 1
    
  #If it is a sequential or a block of modules, don't register a hook on it
  # but recursively register hook on all it's module children
  length = len(layers)
  for i in range(length):
    if (i==(length-1)):
      layers[i].register_forward_hook(hook_fn)
    else:
      if ((isinstance(layers[i], nn.Sequential)) or   #sequential
          (names[i+1].startswith(names[i] + "."))):  #container of layers
        continue
      else:
        layers[i].register_forward_hook(hook_fn)

def profile_activation_size(model,input):
    activations.clear()
    get_all_layers(model) #add hooks to model layers
    out = model(input) #computes activation while passing through layers
    
    total = 0
    
    for name, layer in model.named_modules():
      for label, target in target_layers.items():
        if(isinstance(layer,target)):
          #print(name)
          activation_shape = activations[layer][1].shape
          activation_size = 1
          for i in activation_shape:
            activation_size = activation_size * i
          total = total + activation_size
    
    return total


def get_net_info(net, input_shape=(32,32), measure_latency=None, print_info=True, clean=False, lut=None):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/
    35ddcb9ca30905829480770a6a282d49685aa282/ofa/imagenet_codebase/utils/pytorch_utils.py#L139
    """
    from ofa.imagenet_codebase.utils.pytorch_utils import count_parameters, measure_net_latency

    # artificial input data
    inputs = torch.randn(1, 3, input_shape[0], input_shape[1])

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        net = net.to(device)
        cudnn.benchmark = True
        inputs = inputs.to(device)

    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module
    
    net.eval() # this avoids batch norm error https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274

    # parameters
    net_info['params'] = count_parameters(net)

    net = copy.deepcopy(net)

    net_info['macs'] = int(profile_macs(net, inputs))
   
    # activation_size
    net_info['activations'] = int(profile_activation_size(net, inputs))

    if print_info:
        # print(net)
        print('Total training params: %.2fM' % (net_info['params'] / 1e6))
        print('Total MACs: %.2fM' % ( net_info['macs'] / 1e6))
        print('Total activations: %.2fM' % (net_info['activations'] / 1e6))

    return net_info

 ####

def get_model(name, image_size, classes, get_binaries=False,
              fix_last_layer=False, model_path = None):
    name = name.lower()
    if name == 'alexnet':
        model = AlexNet(image_size[0])
    elif 'vgg' in name:
        if name == 'vgg11':
            model = vgg11()
        else:
            assert False
    elif 'resnet' in name:
        if name == 'resnet20':
            model = resnet20()
        else:
            assert False
    elif 'mobilenet' in name:
        if name == 'mobilenetv3':
            model = mobilenetv3(classes, model_path)
        else:
            assert False
    else:
        assert False
    
    
    classifiers = get_intermediate_classifiers(model,
                                               image_size,
                                               classes,
                                               binary_branch=get_binaries,
                                               fix_last_layer=fix_last_layer)
    
    return model, classifiers


def get_dataset(name, model_name, augmentation=False):
    if name == 'mnist':
        t = [Resize((32, 32)),
             ToTensor(),
             Normalize((0.1307,), (0.3081,)),
             ]
        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = Compose(t)

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = (1, 32, 32)

    elif name == 'flat_mnist':
        t = Compose([ToTensor(),
                     Normalize(
                         (0.1307,), (0.3081,)),
                     torch.nn.Flatten(0)
                     ])

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = 28 * 28

    elif name == 'svhn':
        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([ToTensor(),
                   Normalize((0.4376821, 0.4437697, 0.47280442),
                             (0.19803012, 0.20101562, 0.19703614))])

        t = [
            ToTensor(),
            Normalize((0.4376821, 0.4437697, 0.47280442),
                      (0.19803012, 0.20101562, 0.19703614))]

        # if 'resnet' in model_name:
        #     tt = [transforms.Resize(256), transforms.CenterCrop(224)] + tt
        #     t = [transforms.Resize(256), transforms.CenterCrop(224)] + t

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.SVHN(
            root='~/loss_landscape_dataset/svhn', split='train', download=True,
            transform=train_transform)

        test_set = datasets.SVHN(
            root='~/loss_landscape_dataset/svhn', split='test', download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar10':

        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([ToTensor(),
                   Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

        t = [
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=False, download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar100':
        if augmentation:
            tt = [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
            ]
        else:
            tt = []

        tt.extend([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))])

        t = [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.CIFAR100(
            root='~/datasets/cifar100', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR100(
            root='~/datasets/cifar100', train=False, download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 100

    elif name == 'tinyimagenet':
        if augmentation:
            tt = [
                RandomRotation(20),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                Normalize((0.4802, 0.4481, 0.3975),
                          (0.2302, 0.2265, 0.2262)),
            ]
        else:
            tt = [
                Normalize((0.4802, 0.4481, 0.3975),
                          (0.2302, 0.2265, 0.2262)),
                ToTensor()]

        t = [
            ToTensor(),
            Normalize((0.4802, 0.4481, 0.3975),
                      (0.2302, 0.2265, 0.2262))
        ]

        transform = Compose(t)
        train_transform = Compose(tt)

        # train_set = TinyImageNet(
        #     root='./datasets/tiny-imagenet-200', split='train',
        #     transform=transform)
        train_set = TinyImagenet('~/datasets/tiny-imagenet-200/train',
                                 transform=train_transform)

        test_set = TinyImagenet('~/datasets/tiny-imagenet-200/eval',
                                transform=transform)

        # train_set = datasets.ImageFolder('~/datasets/tiny-imagenet-200/train',
        #                                  transform=train_transform)
        #
        # # for x, y in train_set:
        # #     if x.shape[0] == 1:
        # #         print(x.shape[0] == 1)
        #
        # # test_set = TinyImageNet(
        # #     root='./datasets/tiny-imagenet-200', split='val',
        # #     transform=train_transform)
        # test_set = datasets.ImageFolder('~/datasets/tiny-imagenet-200/test',
        #                                 transform=transform)

        # for x, y in test_set:
        #     if x.shape[0] == 1:
        #         print(x.shape[0] == 1)

        input_size, classes = (3, 64, 64), 200

    else:
        assert False

    return train_set, test_set, input_size, classes


def get_optimizer(parameters,
                  name: str,
                  lr: float,
                  momentum: float = 0.0,
                  weight_decay: float = 0):
    name = name.lower()
    if name == 'adam':
        return optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum,
                         weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be adam or sgd')
