import math
import os
import shutil
import shutil
import torchvision
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import transforms
import numpy as np
from PIL import Image

def get_dataset(args):

    # Loading and normalizing the data.
    transformations_train, transformations_test = get_dataset_transformations(args)

    if args.dataset == "tiny-imagenet":
        train_dir = '~/datasets/tiny-imagenet-200/train'
        val_dir = '~/datasets/tiny-imagenet-200/val'
        train_set = ImageFolder(train_dir, transform=transformations_train)
        test_set = ImageFolder(val_dir, transform=transformations_test)
        input_size, nclasses = 64*64*3, 200
        return train_set, test_set, input_size, nclasses

    if args.dataset == "mnist":
        dataset = MNIST
        input_size, nclasses = 28*28, 10
    if args.dataset == "fashion":
        dataset = FashionMNIST
        input_size, nclasses = 28*28, 10
    elif args.dataset == "cifar10":
        dataset = CIFAR10
        input_size, nclasses = 32*32*3, 10
    elif args.dataset == "cifar100":
        dataset = CIFAR100
        input_size, nclasses = 32*32*3, 100

    # Create an instance for training.
    train_set = dataset(root="~/datasets", train=True, transform=transformations_train, download=True)
    # Create an instance for testing, note that train is set to False.
    test_set = dataset(root="~/datasets", train=False, transform=transformations_test, download=True)

    return train_set, test_set, input_size, nclasses

def get_dataset_transformations(args):

    if args.dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
    elif args.dataset == "mnist":
        mean, std = (0.2860,), (0.3530,)
    elif args.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif args.dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif args.dataset == "tiny-imagenet":
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

    # Define transformations for the training and test sets
    transformations_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transformations_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transformations_train, transformations_test

#CIFAR-10-C and CIFAR-10-P dataset from Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.
#These functions are used to load the dataset and save it in a format that is compatible with the ImageFolder class in PyTorch.
#The datasets available are CIFAR-10-C, CIFAR-100-C, CIFAR-10-P, and CIFAR-100-P.

def load_dataset_corrupted(path = '../datasets/CIFAR-10-P', dataset_name='cifar10p'):

    '''   
    In CIFAR-10(100)-C(P), the first 10,000 images in each .npy are the test set images corrupted at severity 1,
    and the last 10,000 images are the test set images corrupted at severity five. labels.npy is the label file for all other image files.
    '''
    #dataset_name = ['cifar10c', 'cifar100c', 'cifar10p', 'cifar100p']

    out_path = '../datasets/' + dataset_name

    if(os.path.exists(out_path)):
        #remove this folder
        print("Removing folder: " + out_path)
        shutil.rmtree(out_path)

    # Define corruption and severity for each subset

    if dataset_name=='cifar10c' or dataset_name=='cifar100c':
        distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                    'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                    'jpeg_compression']
    elif dataset_name=='cifar10p' or dataset_name=='cifar100p':
        distortions = ['gaussian_noise', 'shot_noise', 'tilt', 'snow', 'scale', 'brightness', 'translate', 'motion_blur', 'rotate', 'zoom_blur', 
         'gaussian_noise_2', 'gaussian_noise_3', 'shot_noise_2', 'shot_noise_3', 'shear', 'spatter', 'spleckle_noise', 'speckle_noise_2', 
         'speckle_noise_3', 'gaussian_blur'] 
    elif dataset_name=='tinyimagenet':
        print("TINYIMAGENET")
        #additional distortions are available in the repo

    freq=10000
    labels = np.load(path + '/labels.npy')

    for j, dist in enumerate(distortions):
         images = np.load(path + '/' + dist + '.npy')
         for idx, img in enumerate(images):
            
            print("Processing image: " )
            print(img.shape)

            if (idx==0):
                severity = 1
            else:
                severity= math.ceil(idx/freq)
            index = j*len(distortions) + idx 
            save_image(img, labels[idx], dist, severity, index, out_path)

def save_image(image_data, label, corruption, severity, index, output_dir):
    subfolder = os.path.join(output_dir, corruption, str(severity), str(label))
    print("SUBFOLDER: " +  subfolder)
    os.makedirs(subfolder, exist_ok=True)
    
    image = Image.fromarray(image_data)
    image.save(os.path.join(subfolder, f"{index}.png"))

def process_dataset(images, labels, corruptions, severities, output_dir):
    for i in range(len(images)):
        image_data = images[i]
        label = labels[i]
        
        for j, (corruption, severity) in enumerate(zip(corruptions, severities)):
            index = i * len(corruptions) + j
            save_image(image_data, label, corruption, severity, index, output_dir)