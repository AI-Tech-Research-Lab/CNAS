
import io
import numpy as np
import torch
from train_utils import validate

def matrix_weights_time():
    # Replace 'your_file.mat' with the path to your .mat file
    mat_contents = io.loadmat('quantization/4bit.mat')

    # mat_contents is now a dictionary containing the data from the .mat file
    # You can access the variables like this:
    #for key in mat_contents:
    #    print(key, type(mat_contents[key]))

    data = mat_contents["ww_mdn"]

    # Determine the column indices to select (odd indices starting from 9)
    start_index = 10  # MATLAB index 911 corresponds to Python index 9
    odd_indices = np.arange(start_index, data.shape[0], 2)  # Generates 11, 13, ...

    # Create the new matrix with only the selected columns
    new_matrix = data[odd_indices,:]
    return new_matrix #returns matrix with temporal values from 10 to 40 microsiemens (shape: 4x8). Value 0 is always the same

def update_drift_in_resnet(resnet, drift_values):
    # Loop over all modules in the model
    # Get the archive data and move to the same device as quantized_w
    #archive = matrix_weights_time() #torch.tensor(matrix_weights_time(), device='cuda').float()
    # Select the appropriate column for this timestep
    #drift_values = archive[:, drift] / 1e-5
    for name, module in resnet.named_modules():
        # Check if the module has a 'drift' attribute
        if hasattr(module, 'drift_w') and 'shortcut' not in name:
            # Update the 'drift' value
            module.drift_w = torch.tensor(drift_values,requires_grad=False).to('cuda')
            #print(f"Updated drift in {name} to {new_drift}")
    #return resnet

def validate_drift(test_loader, model, device):
    archive = matrix_weights_time() 
    accs = []
    for timestep in range(8):
        drift_values = archive[:, timestep] / 1e-5
        update_drift_in_resnet(model, drift_values)
        accs.append(validate(test_loader, model, device, print_freq=100))
    return sum(accs) / len(accs)