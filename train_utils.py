import csv
import random
import time
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomCrop, RandomRotation
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

def save_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, device, filename='checkpoint.pth'):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def train(train_loader, val_loader, num_epochs, model, device, criterion, optimizer, print_freq=10, ckpt='checkpoint.pth'):

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, top1], prefix='Train: ')
    model = model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        end = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update training statistics
            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # Validation phase
        model.eval()
        top1_val = validate(val_loader, model, device, print_freq)  # Reuse the validate function

        # Print training and validation statistics
        print(f'Train Epoch: {epoch + 1}, Train Accuracy: {top1.avg:.2f}%, Val Accuracy: {top1_val:.2f}%')

    # Save the trained model weights
    save_checkpoint(model, optimizer, ckpt)


def validate(val_loader, model, device=None, print_freq=10):

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')
    model.to(device)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images, target = images.to(device), target.to(device)
            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

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


def get_dataset(name, model_name=None, augmentation=False, resolution=32):
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
            root='~/datasets/svhn', split='train', download=True,
            transform=train_transform)

        test_set = datasets.SVHN(
            root='~/datasets/svhn', split='test', download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar10':

        tt = [Resize((resolution, resolution))]

        if augmentation:
            tt.extend([RandomHorizontalFlip(),
                  RandomCrop(resolution, padding=resolution//8)])

        tt.extend([ToTensor(),
                   Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

        t = [
            Resize((resolution, resolution)),
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

        input_size, classes = (3, resolution, resolution), 10

    elif name == 'cifar100':

        tt = [Resize((resolution, resolution))]

        if augmentation:
            tt.extend([
                RandomCrop(resolution, padding=resolution//8),
                RandomHorizontalFlip(),
            ])

        tt.extend([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))])

        t = [
            Resize((resolution, resolution)),
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

        input_size, classes = (3, resolution, resolution), 100
    
    elif name == 'cinic10':

        tt = [Resize((resolution, resolution))]

        if augmentation:
            tt.extend([RandomHorizontalFlip(),
                  RandomCrop(resolution, padding=resolution//8)])

        tt.extend([ToTensor(),
                   Normalize([0.47889522, 0.47227842, 0.43047404],
                             [0.24205776, 0.23828046, 0.25874835])])

        t = [
            Resize((resolution, resolution)),
            ToTensor(),
            Normalize([0.47889522, 0.47227842, 0.43047404],
                             [0.24205776, 0.23828046, 0.25874835])]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.ImageFolder('~/datasets/cinic10/train',
                                         transform=train_transform)
        test_set = datasets.ImageFolder('~/datasets/cinic10/test',
                                         transform=transform)

        input_size, classes = (3, resolution, resolution), 10

    elif name == 'tinyimagenet':
        tt = [Resize((resolution, resolution))]

        if augmentation:
            tt.extend([
                RandomRotation(20),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                Normalize((0.4802, 0.4481, 0.3975),
                          (0.2302, 0.2265, 0.2262)),
            ])
        else:
            tt.extend([
                Normalize((0.4802, 0.4481, 0.3975),
                          (0.2302, 0.2265, 0.2262)),
                ToTensor()])

        t = [
            Resize((resolution, resolution)),
            ToTensor(),
            Normalize((0.4802, 0.4481, 0.3975),
                      (0.2302, 0.2265, 0.2262))
        ]

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.ImageFolder('../datasets/tiny-imagenet-200/train',
                                         transform=train_transform)
        test_set = datasets.ImageFolder('../datasets/tiny-imagenet-200/val',
                                         transform=transform)

        input_size, classes = (3, resolution, resolution), 200

        # train_set = TinyImageNet(
        #     root='./datasets/tiny-imagenet-200', split='train',
        #     transform=transform)
        #train_set = TinyImagenet('~/datasets/tiny-imagenet-200/train',
        #                         transform=train_transform)
        #
        #test_set = TinyImagenet('~/datasets/tiny-imagenet-200/eval',
        #                        transform=transform)
        #
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


def initialize_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)