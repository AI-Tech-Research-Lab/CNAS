import math
import os
import shutil
import shutil
import numpy as np
from PIL import Image

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
