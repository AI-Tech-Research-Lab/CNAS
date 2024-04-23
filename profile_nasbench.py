from itertools import product
import os
from nasbench201 import NASBench201
import matplotlib.pyplot as plt
import numpy as np

def neighbors_by_radius(N, possible_values, vector, radius):
    neighbors = []
    for config in product(possible_values, repeat=N):
        diff_count = sum([1 for x, y in zip(vector, config) if x != y])
        if diff_count == radius:
            neighbors.append(config)
    return neighbors

def avg_val_acc(bench, configs):
    avg_val_acc = 0
    avgs = []
    for c in configs:
        arch = bench.decode(c)
        val_acc = bench.get_info_from_arch(arch)['val-acc']
        avgs.append(val_acc)
        avg_val_acc += val_acc
    return avg_val_acc/len(configs), avgs

def rank_by_val_acc(bench):
    '''
    val_accs_configs = []
    for config in product(range(bench.num_operations), repeat=bench.nvar):
        arch = bench.decode(config)
        val_acc = bench.get_info_from_arch(arch)['val-acc']
        val_accs_configs.append((val_acc, config))
    sorted_val_accs_configs = sorted(val_accs_configs, key=lambda x: x[0])
    sorted_val_accs = [val_acc for val_acc, _ in sorted_val_accs_configs]
    sorted_configs = [config for _, config in sorted_val_accs_configs]
    return sorted_val_accs, sorted_configs
    '''
    val_accs = bench.archive['val-acc']['cifar10-valid']
    idxs = list(range(len(val_accs)))
    val_accs_idxs = list(zip(val_accs, idxs))
    sorted_val_accs_idxs = sorted(val_accs_idxs, key=lambda x: x[0])
    sorted_val_accs = [val_acc for val_acc, _ in sorted_val_accs_idxs]
    sorted_idxs = [idx for _, idx in sorted_val_accs_idxs]

    # Filter the indices for architectures with specific validation accuracies
    filtered_idxs = []
    filtered_idxs.append(sorted_idxs[-1])  # Add the best architecture
    found_80 = False
    found_70 = False
    for val_acc, idx in reversed(sorted_val_accs_idxs):
        if not found_80 and val_acc < 80:
            filtered_idxs.append(idx)
            found_80 = True
        
        if not found_70 and val_acc < 70:
            filtered_idxs.append(idx)
            found_70 = True
        
        if found_80 and found_70:
            break  # Stop if both conditions are met
    
    filtered_val_accs = [val_acc for val_acc, _ in sorted_val_accs_idxs if _ in filtered_idxs]

    return filtered_val_accs, filtered_idxs

def plot_avgacc_vs_radius():

    bench = NASBench201(dataset='cifar10')
    sorted_val_accs, sorted_idxs = rank_by_val_acc(bench)
    print("SORTED VAL ACCS: ", sorted_val_accs)

    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
    config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
    radius_range=range(1,4)
    # Placeholder lists to store accuracy values
    acc_config1 = [sorted_val_accs[2]]
    acc_config2 = [sorted_val_accs[1]]
    acc_config3 = [sorted_val_accs[0]]

    for radius in radius_range:
        # Calculate neighbors for each configuration
        neighbors_config1 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, radius)
        neighbors_config2 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config2, radius)
        neighbors_config3 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config3, radius)
        
        # Calculate average validation accuracy for each configuration
        avg_acc_config1 = avg_val_acc(bench, neighbors_config1)[0]
        avg_acc_config2 = avg_val_acc(bench, neighbors_config2)[0]
        avg_acc_config3 = avg_val_acc(bench, neighbors_config3)[0]
        
        # Append accuracy values to respective lists
        acc_config1.append(avg_acc_config1)
        acc_config2.append(avg_acc_config2)
        acc_config3.append(avg_acc_config3)

    # Plotting
    radius_range = range(4)
    plt.plot(radius_range, acc_config1, marker='s', label='Config 1')
    plt.plot(radius_range, acc_config2, marker='s', label='Config 2')
    plt.plot(radius_range, acc_config3, marker='s', label='Config 3')
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Accuracy vs. Radius for Different Configurations')
    plt.legend()
    plt.savefig('accuracy_vs_radius.png')
    plt.show()

def boxplot_acc_vs_radius():

    bench = NASBench201(dataset='cifar10')
    sorted_val_accs, sorted_idxs = rank_by_val_acc(bench)
    print("SORTED VAL ACCS: ", sorted_val_accs)

    config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
    config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
    config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
    
    radius_range=range(1,4)
    # Placeholder lists to store accuracy values
    accuracies_by_radius_config1 = []
    accuracies_by_radius_config2 = []
    accuracies_by_radius_config3 = []

    for radius in radius_range:
        # Calculate neighbors for each configuration
        neighbors_config1 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config1, radius)
        neighbors_config2 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config2, radius)
        neighbors_config3 = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config3, radius)
        
        # Calculate validation accuracy for each configuration
        acc_neighbors_config1 = avg_val_acc(bench, neighbors_config1)[1]
        acc_neighbors_config2 = avg_val_acc(bench, neighbors_config2)[1]
        acc_neighbors_config3 = avg_val_acc(bench, neighbors_config3)[1]
        
        # Append accuracy values to respective lists
        accuracies_by_radius_config1.append(acc_neighbors_config1)
        accuracies_by_radius_config2.append(acc_neighbors_config2)
        accuracies_by_radius_config3.append(acc_neighbors_config3)

    # Plotting the boxplots
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.boxplot(accuracies_by_radius_config1, patch_artist=True)
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Config 1')

    plt.subplot(1, 3, 2)
    plt.boxplot(accuracies_by_radius_config2, patch_artist=True)
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Config 2')

    plt.subplot(1, 3, 3)
    plt.boxplot(accuracies_by_radius_config3, patch_artist=True)
    plt.xlabel('Radius')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Config 3')

    # Adding grid
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Showing plot
    plt.savefig('accuracy_vs_radius_boxplot.png')
    plt.show()

def plot_histograms(data_array, bins=100, path=''):
    import matplotlib.colors as mcolors
    import seaborn as sns

    FONT_SIZE = 8
    FIGSIZE = (3.5, 3.0)
    COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]

    num_plots = len(data_array)

    # Set up subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=FIGSIZE, sharex=False)

    # Plot histograms and curves for each element in the array
    for i, data in enumerate(data_array):
        data = np.array(data)   
        data = data[data > 20]
        sns.histplot(data, bins=bins, color='darkblue', edgecolor='black', kde=True, line_kws={'linewidth': 1}, ax=axs[i])
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis values
        axs[i].set_xlim(60, 95)  # Set y-axis limits

    # Add common X-axis label
    axs[-1].set_xlabel('Value', fontsize=FONT_SIZE)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(path, bbox_inches='tight')

    # Show the plot
    plt.show()

bench = NASBench201(dataset='cifar10')
sorted_val_accs, sorted_idxs = rank_by_val_acc(bench)

config1 = bench.encode({'arch':bench.archive['str'][sorted_idxs[0]]})
config2 = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
config3 = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})

configs = [config1, config2, config3]

radius_range=range(1,4)
# Placeholder lists to store accuracy values
accuracies_by_radius_configs = []

for idx in range(len(configs)):

    acc_neighbors_configs = []
    for radius in radius_range:
        if not os.path.exists("accuracies_config_" + str(idx+1) + "_radius_" + str(radius) + ".npy"):
            print("Calculating array")
            config=configs[idx]
            # Calculate neighbors for each configuration
            neighbors_config = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), config, radius)
            
            # Calculate validation accuracy for each configuration
            acc_neighbors_config = avg_val_acc(bench, neighbors_config)[1]
            
            acc_neighbors_config = np.array(acc_neighbors_config)

            # Define the file path
            file_path = "accuracies_config_" + str(idx+1) 

            # Save the numpy arrays to a single file
            np.save(file_path + "_radius_" + str(radius), acc_neighbors_config) 

        else:
            print("Loading array")
            acc_neighbors_config = np.load("accuracies_config_" + str(idx+1) + "_radius_" + str(radius) + ".npy")
            acc_neighbors_configs.append(acc_neighbors_config)
            print("ACC NEIGHBORS CONFIG: ", acc_neighbors_config.shape)
    
    accuracies_by_radius_configs.append(acc_neighbors_configs)

#accuracies_by_radius_configs[0] = np.array(accuracies_by_radius_configs[0])
#print(accuracies_by_radius_configs[0])
for idx in range(len(configs)):
    plot_histograms(accuracies_by_radius_configs[idx], path='histograms_' + 'config_' + str(idx+1) + '.png') #plot 2nd config


