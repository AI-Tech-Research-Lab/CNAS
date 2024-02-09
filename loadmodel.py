import torch
from train_utils import get_dataset, validate
from utils import get_net_OFAMBV3

'''
model_path = '../results/cifar10-mbv3-r32-23nov/iter_0/net_0.subnet'
# Specify the path to your saved model file
output_path = '../results/model.pt'
subnet = get_net_OFAMBV3(model_path)
torch.save(subnet, output_path)
'''


output_path = '../results/model.pt'
#subnet = torch.load(output_path)
# Load the model state dictionary
device = torch.device("cuda:0")
model = torch.load(output_path, map_location=device)
dataset_name = 'cifar10'

train_set, test_set, input_size, n_classes = \
                    get_dataset(name=dataset_name,
                                model_name=None,
                                augmentation=True)

testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=64,
                                                 shuffle=False)

validate(testloader, model, device)

# Print the keys in the state dictionary
#print("State Dictionary Keys:")
#for key in model_state_dict.keys():
    #print(key)

