
import os
import torch
import numpy as np

import sys
sys.path.append(os.getcwd())

from trainers.cbn.evaluators import binary_eval, get_confidences
from train_utils import get_dataset
from trainers.cbn.utils import get_model
from explainability import plot_histograms, plot_histogramsV2

import seaborn as sns
import matplotlib.pyplot as plt

batch_size = 64
get_binaries = True
fix_last_layer = True   
dataset_name = 'cifar10'  
model_name='mobilenetv3'  
#path = '../results/cifar10-cbn-mbv3-8sept/final/net-best-trade-off'
path = '../results/cifar10-cbn-mbv3-8sept-noconstraints/final/net-trade-off'
supernet='supernets/ofa_mbv3_d234_e346_k357_w1.0'
model_path = os.path.join(path,'net_1.subnet')
#model_path = os.path.join(path,'net_54.subnet')
pre_trained_model_path = os.path.join(path, 'bb.pt')
pre_trained_classifier_path = os.path.join(path, 'classifiers.pt')


train_set, test_set, input_size, n_classes = \
                    get_dataset(name=dataset_name,
                                model_name=None,
                                augmentation=True)

testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)

backbone, classifiers = get_model(model_name, image_size=input_size,
                                          n_classes=n_classes,
                                          get_binaries=get_binaries,
                                          fix_last_layer=fix_last_layer,
                                          model_path=model_path,
                                          pretrained=True,
                                          supernet=supernet
                                          )

backbone.load_state_dict(torch.load(pre_trained_model_path))
classifiers.load_state_dict(torch.load(pre_trained_classifier_path))

confidences = get_confidences(backbone, classifiers, testloader, cumulative_threshold=True, global_gate=None)

'''

# List of epsilon values
epsilon1_values = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
epsilon2_values = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]

# Initialize arrays to store accuracy values and counters
accuracy_values = np.zeros((len(epsilon1_values), len(epsilon2_values)))
counter_values = np.zeros((len(epsilon1_values), len(epsilon2_values)))

# Iterate through epsilon values and store accuracy and counters
for i, epsilon1 in enumerate(epsilon1_values):
    for j, epsilon2 in enumerate(epsilon2_values):
        a, b = binary_eval(model=backbone,
                           dataset_loader=testloader,
                           predictors=classifiers,
                           epsilon=[epsilon1] + [epsilon2] * (backbone.n_branches() - 1),
                           cumulative_threshold=True,
                           sample=False)

        a, b = dict(a), dict(b)
        accuracy_values[i, j] = a['global']
        #counter_values[i, j] = b['total_counter']

        print("Thresholds: ", epsilon1, epsilon2)
        print("Accuracy: ", a)
        print("Counters: ", b)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(accuracy_values, cmap="YlGnBu", annot=True, fmt=".2f",
            xticklabels=epsilon2_values, yticklabels=epsilon1_values)

# Set labels and title
plt.xlabel("Epsilon2")
plt.ylabel("Epsilon1")
plt.title("Accuracy of Early Exit Neural Network")

# Show the plot
plt.show()
'''

# Ensure confidences is a list of NumPy arrays
#confidences_np = [np.array(confidence).reshape(-1) for confidence in confidences]

#print(confidences_np[0].shape)
xlabels=([0.1,0.9,0.2,1],[0.1,0.9,0.2,1],[0.1,0.9,0.2,1])
plot_histogramsV2(confidences,36,path, xlabels)
