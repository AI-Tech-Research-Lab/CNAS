import csv
import random
import time
import torch
import numpy as np

from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets.folder import default_loader
import torch.optim as optim

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomCrop, RandomRotation, RandomErasing, RandomResizedCrop, \
    TrivialAugmentWide, InterpolationMode
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2
from torch.nn.modules.batchnorm import _BatchNorm

class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"

class Log:
    def __init__(self, log_each: int, initial_epoch=-1):
        self.loading_bar = LoadingBar(length=27)
        self.best_accuracy = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch

        self.best_model = None
        self.best_loss = float('inf')


    def train(self, model, optim, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush(model, optim)

        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self._reset(len_dataset)

    def eval(self, model, optim, len_dataset: int) -> None:
        self.flush(model, optim)
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, model, loss, accuracy, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(model, loss, accuracy, learning_rate)
        else:
            self._eval_step(loss, accuracy)

    def flush(self, model, optim) -> None:
        if self.is_train:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{self.learning_rate:12.3e}  │{self._time():>12}  ┃",
                end="",
                flush=True,
            )

        else:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            print(f"{loss:12.4f}  │{100*accuracy:10.2f} %  ┃", flush=True)

            if loss<self.best_loss: #accuracy > self.best_accuracy:

                #print('LOSS: ', loss, 'BEST LOSS: ', self.best_loss)
                self.best_accuracy = accuracy
                self.best_loss = loss
                #save the state of the best model
                #self.best_model = {'weights_state': model.state_dict(), 'optim_state':optim.state_dict()}

    def _train_step(self, model, loss, accuracy, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += loss.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]

            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
            progress = self.step / self.len_dataset

            print(
                f"\r┃{self.epoch:12d}  ┃{loss:12.4f}  │{100*accuracy:10.2f} %  ┃{learning_rate:12.3e}  │{self._time():>12}  {self.loading_bar(progress)}",
                end="",
                flush=True,
            )

    def _eval_step(self, loss, accuracy) -> None:
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        print(f"┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃")
        print(f"┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃")
        print(f"┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")

def save_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def train(train_loader, val_loader, num_epochs, model, device, criterion, optimizer, print_freq=10, ckpt='ckpt'):

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


def train_mix(train_loader, val_loader, num_epochs, model, n_classes, device, criterion, optimizer, print_freq=10, ckpt='ckpt'):

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, top1], prefix='Train: ')
    model = model.to(device)
    cutmix = v2.CutMix(num_classes=n_classes)
    mixup = v2.MixUp(num_classes=n_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        end = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, ori_labels = images.to(device), labels.to(device)
            images, labels = cutmix_or_mixup(images, ori_labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update training statistics
            acc1 = accuracy(outputs, ori_labels, topk=(1,))
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

def validate(val_loader, model, device=None, print_info=True, print_freq=0):

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

            if print_info and print_freq and i % print_freq == 0:
                progress.display(i)
        
        if print_info:
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


def get_dataset(name, model_name=None, augmentation=False, resolution=32, val_fraction=0.0):

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

        tt = [RandomResizedCrop(resolution, scale=(0.08,1.0)),
                  RandomHorizontalFlip(),
                  ToTensor(),
                  Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                  ]

        '''
        print("New Augmentation")
        tt = [
            TrivialAugmentWide(interpolation = InterpolationMode.BILINEAR),
            ToTensor(), 
            RandomErasing(scale=(0.05,0.25), value='random'),
            RandomHorizontalFlip(),
            RandomResizedCrop(resolution, scale=(0.08,1.0)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        '''
        
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
        
        val_set = datasets.CIFAR10(
        root='~/datasets/cifar10', train=True, download=True,
        transform=transform)

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

    else:
        assert False

    # add splitting into train and validation
    # Split the dataset into training and validation sets
    if val_fraction:
        val_size = int(val_fraction * len(train_set))
        train_size = len(train_set) - val_size
        # Generate random indices for the entire dataset
        indices = list(range(len(train_set)))
        random.shuffle(indices)
        # Split the indices into training and validation sets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        # Create two new datasets using the random indices
        train_set = Subset(train_set, train_indices)
        val_set = Subset(val_set, val_indices)
        print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
    else:
        val_set = None

    return train_set, val_set, test_set, input_size, classes

def get_data_loaders(dataset, batch_size=32, threads=1, val_fraction=0, img_size=32, augmentation=False, eval_test=True):
    print(f"DATASET: {dataset}")
    
    # Retrieve train and test datasets
    print(f"Augmentation: {augmentation}")


    train_set, val_test, test_set, _, _ = get_dataset(dataset, augmentation=augmentation, resolution=img_size, val_fraction=val_fraction)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads, pin_memory=True)
    if val_fraction:
        val_loader = DataLoader(train_set, batch_size=batch_size*2, shuffle=False, num_workers=threads, pin_memory=True)
    else:
        val_loader = None
    
    # Create DataLoader for test set if args.eval_test is True
    if eval_test:
        test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False, num_workers=threads, pin_memory=True)
    else:
        test_loader=None
    
    return train_loader, val_loader, test_loader

# Example usage:
# train_loader, val_loader, test_loader = create_data_loaders(args)
# for batch in train_loader:
#     # Training loop
# for batch in test_loader:
#     # Testing loop (if args.eval_test is True)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def get_optimizer(parameters,
                  name: str,
                  lr: float,
                  momentum: float = 0.0,
                  weight_decay: float = 0,
                  rho: float=2.0,
                  adaptive: bool=True
                  ): #SAM

    name = name.lower()
    if name == 'adam':
        return optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum,
                         weight_decay=weight_decay)
    elif name == 'sam':
        base_optimizer = torch.optim.SGD
        return SAM(parameters, base_optimizer, rho=rho, adaptive=adaptive, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be adam, sgd, or SAM')

def get_lr_scheduler(optimizer, name, epochs, gamma=1.0):
    name=name.lower()
    if name=='step':
        return StepLR(optimizer, step_size=epochs, gamma=gamma)
    elif name=='cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    else:
        raise ValueError('Scheduler must be step or cosine')                     
    return scheduler

def initialize_seed(seed: int, use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

""" Mixup """


def mix_images(images, lam):
    flipped_images = torch.flip(images, dims=[0])  # flip along the batch dimension
    return lam * images + (1 - lam) * flipped_images


def mix_labels(target, lam, n_classes, label_smoothing=0.1):
    onehot_target = label_smooth(target, n_classes, label_smoothing)
    flipped_target = torch.flip(onehot_target, dims=[0])
    return lam * onehot_target + (1 - lam) * flipped_target


""" Label smooth """

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    soft_target = label_smooth(target, pred.size(1), label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target)

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

# Optimizers

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups