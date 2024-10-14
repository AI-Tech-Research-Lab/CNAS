import json
import os
from NasSearchSpace.ofa.evaluator import OFAEvaluator
import torch.nn as nn
import torchvision
import sys
import subprocess

'''
def count_layers(module):
    """
    Recursively count the number of layers in a given module.
    Skips downsample blocks and only counts the actual layers (e.g., Conv2d, Linear).
    """
    count = 0
    # Iterate over all named submodules
    for name, child in module.named_children():
        print("name",name)
        # Check if the current block is a downsample block, and skip it
        if 'downsample' in name:
            print(f"Skipping downsample block: {name}")
            continue
        # If the child is a container (e.g., Sequential or ModuleList), recurse into it
        if isinstance(child, (nn.Sequential, nn.ModuleList)) or 'Bottle' in child.__class__.__name__:
            print(f"Recursing into container: {name}")
            count += count_layers(child)
        # Otherwise, count the Conv2d and Linear layers
        elif isinstance(child, (nn.Conv2d, nn.Linear)):
            print(f"Counting layer: {name}")
            count += 1
            print(child)  # Optional: print to verify counted layers
    return count
'''

config = 'net.subnet'
n_classes = 10
supernet = 'NasSearchSpace/ofa/supernets/ofa_resnet50'
pretrained = False
net_config = json.load(open(config))
evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet, pretrained=pretrained)
subnet, _ = evaluator.sample(net_config)

# Specify the file name where you want to save the architecture description
architecture_file = 'model_architecture.txt'

# Save the current stdout
original_stdout = sys.stdout

try:
    # Open the file in write mode
    with open(architecture_file, 'w') as file:
        # Redirect stdout to the file
        sys.stdout = file

        # Print the model architecture description
        print(subnet)

except Exception as e:
    print(f"Error saving architecture: {e}")

finally:
    # Reset stdout to its original value
    sys.stdout = original_stdout

# Specify the Python script and its arguments
script_path = 'parser_CNAS.py'
input_file = architecture_file
y_value = str(net_config['r'])
x_value = str(net_config['r'])

# Construct the command to call the script with arguments
command = ['python', script_path, input_file, str(y_value), str(x_value)]

# Execute the command
try:
    subprocess.run(command, check=True)
    print("Script executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error executing script: {e}")

# Specify the file you want to delete
file_to_delete = 'model_architecture.txt'

# Check if the file exists before attempting to delete
if os.path.exists(file_to_delete):
    try:
        # Delete the file
        os.remove(file_to_delete)
        print(f"{file_to_delete} has been deleted.")
    except OSError as e:
        print(f"Error deleting file: {e}")
else:
    print(f"{file_to_delete} does not exist.")

# Specify the shell script path
#shell_script_path = 'run_gamma.sh'
shell_script_path = 'run_gamma.sh'

# Construct the command to call the shell script
shell_command = ['bash', shell_script_path]

# Execute the command
try:
    subprocess.run(shell_command, check=True)
    print("Shell script executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error executing shell script: {e}")
