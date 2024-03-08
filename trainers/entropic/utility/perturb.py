import torch 
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import copy
from utils import get_net_info, get_correlation
from explainability import get_archive
from train_utils import validate


def set_random_seeds(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Function to save the model
def saveModel(model, path="./myFirstModel.pth"):
    torch.save(model.state_dict(), path)

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to test the model with a batch of images and show the labels predictions
def testBatch(model, test_loader, classes, batch_size):
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))
    

# Function to test what classes performed well
def testClasses(device, model, test_loader, batch_size, number_of_labels, classes):
    class_correct = list(0. for i in range(number_of_labels))
    class_total = list(0. for i in range(number_of_labels))
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(number_of_labels):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


# Function to test what classes performed well
def calc_accuracy(device, model, loader):
    correct = 0
    model.to(device)
    num_total = len(loader.dataset)
    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            correct += (torch.sum(c)).item()
    return 100 * correct / num_total


def create_perturb(model, noise="mult"):
    z = []
    for p in model.parameters():
        r = p.clone().detach().normal_()
        if noise == "mult":
            z.append(p.data * r)  # multiplicative noise
        else:
            z.append(r)           # additive noise 
    return z
    
def perturb(model, z, noise_ampl):
    for i,p in enumerate(model.parameters()):
        p.data += noise_ampl * z[i]

def calculate_robustness(device, net, loader, sigma):
    M = 20
    robustness = []

    acc0 = validate(loader, net, device, print_info=False)

    for _ in range(M):
        perturbed_net = copy.deepcopy(net)
        z = create_perturb(perturbed_net)
        perturb(perturbed_net, z, sigma)
        acc = validate(loader, perturbed_net, device, print_info=False)
        rob = abs(acc0 - acc) #/ acc0
        print(f"\nacc0: {acc0}; acc: {acc}; robustness: {rob}")
        robustness.append(rob)
    return sum(robustness) / len(robustness)

def calculate_robustness_list(device, net, loader, sigma_list):
    rob_list=[]
    for sigma in sigma_list:
        rob_list.append(calculate_robustness(device, net, loader, sigma))
    return rob_list

def compute_best_sigma(exp_path):

    #returns the idx of the best sigma on rho and the correlation metrics

    archive = get_archive(exp_path, 'top1', 'robustness')
    top1, rob = [v[1] for v in archive], [v[2] for v in archive]
    #print(np.argsort(top1)[:])#, rob[:10])
    #a = np.argsort(top1)
    #print([top1[i] for i in a][:])
    r_list = [] # list of robustness for each sigma
    n_sigmas=1#len(rob[0])
    for i in range(n_sigmas):
        r_list.append([v[i] for v in rob])
    #print([r_list[-1][i] for i in a][:])
    rmse_s = 0
    rho_s = float('-inf')
    tau_s = float('-inf')
    sigma_idx=0
    for idx, r in enumerate(r_list):
        rmse, rho, tau = get_correlation(np.array(top1),np.array(r))
        if (tau > tau_s):
            rmse_s, rho_s, tau_s = rmse, rho, tau
            sigma_idx = idx 
    return sigma_idx, rmse_s, rho_s, tau_s

def get_net_info_runtime(device, net, loader, sigma_list, input_shape=(3, 224, 224), print_info=False):

    # TODO: caricare i pesi della rete

    net_info = get_net_info(net, input_shape=input_shape, print_info=print_info)

    # robustness
    #sigma = 0.05
    net_info['robustness'] = calculate_robustness_list(device, net, loader, sigma_list)
    net_info['robustness'] = [np.round(x, 2) for x in net_info['robustness']]

    if print_info:
        # print(net)
        print('Robustness: ', (net_info['robustness']))

    return net_info
