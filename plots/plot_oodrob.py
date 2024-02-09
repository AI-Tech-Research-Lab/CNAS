import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
import seaborn as sns
# Set the style to 'seaborn-whitegrid'
plt.style.use('seaborn-whitegrid')

def extract_info(filename):
    # Split the filename on hyphens and other delimiters
    parts = filename.split('-')
    # Extract DATASET name
    dataset = next((part for part in parts if part in ['cifar10', 'cifar100']), None)
    # Extract 'SGD' or 'SAM'
    optimizer = next((part for part in parts if part in ['SGD', 'SAM']), None)
    # Extract the alpha value
    alpha_prefix = 'alpha'
    alpha = next((part for part in parts if part.startswith(alpha_prefix)), None)
    alpha_value = alpha[len(alpha_prefix):] if alpha else None  # Remove the 'alpha' prefix
    return dataset, optimizer, alpha_value

def plot_combined_noise_data_save_fig(file_paths, save_path, 
                                      plot_mean=True, plot_relative_error=False):
    """
    Reads multiple noise data files, processes each, and plots the data from all files on the same plot. 
    Each file's data is plotted using a unique colormap. The plot includes curves for each noise type 
    from each file as well as the overall mean with standard deviation for each file, each in a 
    different color. The combined plot is saved as a PNG file.

    :param file_paths: List of paths to the noise data files
    :param save_path: Path where the combined PNG file will be saved
    """
    plt.figure(figsize=(15, 8))
    mean_colors = cm.rainbow(np.linspace(0, 1, len(file_paths)))
    colormaps = [cm.Purples, cm.Blues, cm.Greens, cm.Oranges, cm.Reds]  # List of colormaps for different files

    for file_idx, file_path in enumerate(file_paths):
        #if not os.path.exists(file_path):
        #    continue
        dataset, opt, alpha = extract_info(file_path)
        # Read and parse the file
        with open(file_path, 'r') as file:
            parsed_json = json.loads(file.read())

        top1_value = parsed_json.get('top1', 'Not found')
        robustness_value = parsed_json.get('robustness', 'Not found')
        params_value = parsed_json.get('params', 'Not found')
        macs_value = parsed_json.get('macs', 'Not found')
        c_params_value = parsed_json.get('c_params', 'Not found')

        mce_data = parsed_json['mCE2']
        df_noise = pd.DataFrame.from_dict(mce_data, orient='index').drop('mean', axis=1).transpose()
        # Calculate the mean and standard deviation
        mean_per_intensity = df_noise.mean(axis=1)
        std_per_intensity = df_noise.std(axis=1)
        
        if plot_relative_error:
            mean_per_intensity -= top1_value

        # Choose colormap for this file's noise types
        #colormap = colormaps[file_idx % len(colormaps)]
        colormap = cm.tab20
        noise_colors = colormap(np.linspace(0.2, 1, len(df_noise.columns)))
        # Plot for each noise type using the selected colormap
        marker = "s" if opt=="SGD" else 'x'
        ls = "--" if opt=="SGD" else '-'
        if not plot_mean:
            for i, column in enumerate(df_noise.columns):
                plt.plot(df_noise.index, df_noise[column], 
                        marker=marker, ls=ls, color=noise_colors[i], 
                        label=f"{column} ({os.path.basename(file_path)})")

        print(mean_per_intensity)

        # Plot for the mean values across all noise types for each file with error bars
        if plot_mean:
            plt.plot(mean_per_intensity.index, mean_per_intensity, 
                        #yerr=std_per_intensity, ecolor=mean_colors[file_idx], elinewidth=3, capsize=5, 
                        marker='o', linestyle='-', color=mean_colors[file_idx], 
                        label=(f"{opt} alpha {alpha} " 
                        f"(top1 {top1_value}, robustness {robustness_value}, "
                        f"params {c_params_value}, macs {macs_value}, "
                        f"c_params {c_params_value})"))

    plt.title(f'Mean Values (Combined Noise Types) vs. Intensity {dataset}', fontsize=18)
    plt.xlabel('Noise Intensity', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    if plot_mean:
        plt.legend(loc='best', borderaxespad=0., fontsize=16) # , bbox_to_anchor=(1.05, 1)
    
    # Save the figure as a PNG file
    plt.savefig(save_path+f"{dataset}.png", format='png', bbox_inches='tight')

    # Close the figure to prevent it from displaying
    plt.close()

# Example usage:
home = "/home/fabriziop"
#res_folder_path = f"{home}/results/risultati-res32"; suffix = "10jan"; epochs=10
res_folder_path = f"{home}/results/multires"; suffix = "multires"; epochs=6
dataset = "cifar100"
plot_mean = True
plot_relative_error = False
plot_combined_noise_data_save_fig(
    [f'{res_folder_path}/entropic-mbv3-{dataset}-SGD-top1-c_params-max5.0-alpha0.5-sigma0.05-ep{epochs}-{suffix}/final/net-trade-off_0/net_0.stats',
    f'{res_folder_path}/entropic-mbv3-{dataset}-SGD-top1-c_params-max5.0-alpha0.5-sigma0.05-ep{epochs}-{suffix}/final/net-trade-off_0/sgd_with_sam/net_sam.stats',
     f'{res_folder_path}/entropic-mbv3-{dataset}-SAM-top1_robust-c_params-max5.0-alpha0.1-sigma0.05-ep{epochs}-{suffix}/final/net-trade-off_0/net_0.stats',
     f'{res_folder_path}/entropic-mbv3-{dataset}-SAM-top1_robust-c_params-max5.0-alpha0.5-sigma0.05-ep{epochs}-{suffix}/final/net-trade-off_0/net_0.stats',
     f'{res_folder_path}/entropic-mbv3-{dataset}-SAM-top1_robust-c_params-max5.0-alpha0.9-sigma0.05-ep{epochs}-{suffix}/final/net-trade-off_0/net_0.stats'
     ], 
     f'./figures/fig_{suffix}_', plot_mean=plot_mean, plot_relative_error=plot_relative_error)