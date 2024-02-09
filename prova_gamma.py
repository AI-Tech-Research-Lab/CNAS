import json
import os
from ofa_evaluator import OFAEvaluator
import sys
import subprocess

config = 'net.subnet'
n_classes = 10
supernet = 'supernets/ofa_mbv3_d234_e346_k357_w1.0'
pretrained = True
net_config = json.load(open(config))
evaluator = OFAEvaluator(n_classes=n_classes, model_path=supernet, pretrained=pretrained)
subnet, _ = evaluator.sample(net_config)
print(subnet)

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
shell_script_path = '../gamma/run_gamma.sh'

# Construct the command to call the shell script
shell_command = ['bash', shell_script_path]

# Execute the command
try:
    subprocess.run(shell_command, check=True)
    print("Shell script executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error executing shell script: {e}")