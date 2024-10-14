import os
import re
import sys
import math
import csv


def parse_conv_lines(file_path):
    conv_lines = []

    with open(file_path, 'r') as file:
        is_first_conv = True
        is_block = False
        block_number = 0

        for line in file:
            # Check if the line indicates the start of a block or a final expand layer
            block_match = re.match(r'\s+\((\d+)\)', line)
            final_layer_match = re.match(r'\s+\(final_expand_layer\)', line)

            if block_match:
                is_block = True
                block_number = int(block_match.group(1))
                continue
            elif final_layer_match:
                is_block = True
                block_number = "final"
                continue

            # Use regex to match lines starting with "Conv2d"
            if re.search(r'Conv2d', line):
                conv_lines.append((line.strip(), is_first_conv, is_block, block_number))
                is_first_conv = False

    return conv_lines


def extract_network_name(file_path):
    with open(file_path, 'r') as file:
        # Assuming the network name is on the first line and followed by a closing parenthesis
        first_line = file.readline()
        match = re.search(r'(\w+)\(', first_line)
        if match:
            return match.group(1)

    return None


def calculate_output_size(input_size, kernel_size, padding, stride):
    return math.floor((input_size + 2 * padding - kernel_size + 1) / stride)


def generate_cnn_model(lines, y, x, name):
    model_str = f"Network {name} {{\n"

    layer_count = 1
    prev_block = lines[0][3]
    first_line = True
    prev_output_dimensions = ()
    csv_data = [['K', 'C', 'Y', 'X', 'R', 'S', 'T', 'SX', 'SY', 'Precision']]  # Updated header

    for conv_line, is_first_conv, is_block, block_number in lines:
        if not is_block and not is_first_conv:
            continue  # Skip layers before blocks, excluding the first layer

        if block_number != prev_block:
            layer_count = 1  # Reset layer count for every new block
            prev_block = block_number

        layer_name = f"{'' if not is_block else f'block_{block_number}_'}{layer_count}"

        dimensions_match = re.search(
            r'Conv2d\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\)', conv_line)

        # Check for the presence of "groups" in the line to determine CONV or DSCONV
        is_dsconv = 'groups' in conv_line

        padding_match = re.search(r'padding=\((\d+), (\d+)\)', conv_line)
        padding_y, padding_x = map(int, padding_match.groups()) if padding_match else (0, 0)

        if dimensions_match:
            in_channels, out_channels, kernel_y, kernel_x, stride_y, stride_x = map(int, dimensions_match.groups())

            if first_line:
                x_output = x
                y_output = y
                first_line = False
            else:
                # Calculate X and Y using the provided formula
                x_output = calculate_output_size(prev_output_dimensions[0], prev_output_dimensions[3],
                                                 prev_output_dimensions[7], prev_output_dimensions[5])
                y_output = calculate_output_size(prev_output_dimensions[1], prev_output_dimensions[2],
                                                 prev_output_dimensions[6], prev_output_dimensions[4])

            model_str += f"Layer {layer_name} {{\n"
            model_str += f"Type: {'DSCONV' if is_dsconv else 'CONV'}\n"
            model_str += f"Stride {{ X: {stride_x}, Y: {stride_y} }}\n"
            model_str += (f"Dimensions {{ K: {out_channels}, C: {in_channels}, R: {kernel_y}, S: {kernel_x}," 
                           f" Y:{y_output}, X:{x_output} }}\n")
            model_str += "}\n\n"

            if is_block:
                layer_count += 1

            prev_output_dimensions = (x_output, y_output, kernel_y, kernel_x, stride_y, stride_x, padding_y,
                                      padding_x)
            csv_data.append([out_channels, in_channels, y_output, x_output, kernel_y, kernel_x,
                             2 if out_channels == 1 else 1, stride_x, stride_y, 'INT4'])  # Added SX, SY, and Precision (now set to 4)

    model_str += "}\n"

    return model_str, csv_data


def save_cnn_model(output_path, model_str):
    with open(output_path, 'w') as output_file:
        output_file.write(model_str)


def save_cnn_csv(csv_path, csv_data):
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py input_file y_value x_value")
        sys.exit(1)

    file_path = sys.argv[1]
    y_value = int(sys.argv[2])
    x_value = int(sys.argv[3])

    conv_lines = parse_conv_lines(file_path)
    network_name = extract_network_name(file_path)

    if network_name and conv_lines:
        output_path='../qgamma/data/model'
        output_path2 = '../qmaestro/data/model'
        # Create full paths for the output files
        model_output_path = os.path.join(output_path2, f'{network_name}_model.m')
        csv_output_path = os.path.join(output_path, f'{network_name}_model.csv')

        cnn_model_str, csv_data = generate_cnn_model(conv_lines, y_value, x_value, network_name)

        save_cnn_model(model_output_path, cnn_model_str)
        save_cnn_csv(csv_output_path, csv_data)

        print(f"{network_name}_model.m and {network_name}_model.csv files generated successfully.")
    elif not network_name:
        print("Unable to extract network name from the input file.")
    else:
        print("No lines starting with 'Conv2d' found in the input file.")
