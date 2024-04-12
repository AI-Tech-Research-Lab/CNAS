import math
import random
from nasbench201 import NASBench201

save_path = '../results/simulated_annealing'
dataset = 'cifar10'

def random_action(vector):
    idx = random.randint(0, len(vector) - 1)
    interval = range(5)
    new_val = random.choice(interval)
    vector[idx] = new_val
    return vector

def P(delta_energy, temperature):
    if delta_energy < 0:
        return 1
    else:
        return math.exp(-delta_energy / temperature)

def temperature(i, iterations):
    if i==(iterations-1):
        T = 0
    else:
        T = 1 - ((i+1) / iterations)
    return T

def search(dataset, save_path):

    ss = NASBench201(dataset, save_path)
    
    arch_i = ss.sample(n_samples = 1)[0]
    vector_i = ss.encode(arch_i)
    top1_i = ss.get_info_from_arch(arch_i)['val-acc']
    iterations=100
    for i in range(iterations):
        T = temperature(i, iterations)
        print('Temperature:', T)
        vector_j = random_action(vector_i)
        arch_j = ss.decode(vector_j)
        top1_j = ss.get_info_from_arch(arch_j)['val-acc']
        print('Top1_i:', top1_i)
        print('Top1_j:', top1_j)
        if P(top1_i - top1_j, T) > random.random():
            arch_i = arch_j
            top1_i = top1_j
        print(f'Iteration {i+1}/{iterations} - Top1: {top1_i} - Temperature: {T}')

search('cifar10', '../results/simulated_annealing')

