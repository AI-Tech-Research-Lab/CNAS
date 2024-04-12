from itertools import product
from nasbench201 import NASBench201
import matplotlib.pyplot as plt

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

'''
print("BEST CONFIG: ", best_config)
best_acc = bench.get_info_from_arch(bench.decode(best_config))['val-acc']
print("BEST ACC: ", best_acc)
neighbors = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), best_config, 1)
#print("NEIGHBORS: ", neighbors)
print("AVG VAL ACC BY RADIUS 1: ", avg_val_acc(bench, neighbors)[0])
#print("AVGS: ", avg_val_acc(bench, neighbors)[1])
neighbors = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), best_config, 2)
#print("NEIGHBORS: ", neighbors)
print("AVG VAL ACC BY RADIUS 2: ", avg_val_acc(bench, neighbors)[0])

best_config = bench.encode({'arch':bench.archive['str'][sorted_idxs[1]]})
print("BEST CONFIG: ", best_config)
best_acc = bench.get_info_from_arch(bench.decode(best_config))['val-acc']
print("BEST ACC: ", best_acc)
neighbors = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), best_config, 1)
#print("NEIGHBORS: ", neighbors)
print("AVG VAL ACC BY RADIUS 1: ", avg_val_acc(bench, neighbors)[0])
#print("AVGS: ", avg_val_acc(bench, neighbors)[1])
neighbors = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), best_config, 2)
#print("NEIGHBORS: ", neighbors)
print("AVG VAL ACC BY RADIUS 2: ", avg_val_acc(bench, neighbors)[0])

best_config = bench.encode({'arch':bench.archive['str'][sorted_idxs[2]]})
print("BEST CONFIG: ", best_config)
best_acc = bench.get_info_from_arch(bench.decode(best_config))['val-acc']
print("BEST ACC: ", best_acc)
neighbors = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), best_config, 1)
#print("NEIGHBORS: ", neighbors)
print("AVG VAL ACC BY RADIUS 1: ", avg_val_acc(bench, neighbors)[0])
#print("AVGS: ", avg_val_acc(bench, neighbors)[1])
neighbors = neighbors_by_radius(bench.nvar, list(range(bench.num_operations)), best_config, 2)
#print("NEIGHBORS: ", neighbors)
print("AVG VAL ACC BY RADIUS 2: ", avg_val_acc(bench, neighbors)[0])
'''