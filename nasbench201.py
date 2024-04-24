import torch
import json
import os
from typing import List, Text, Union, Dict, Optional
import numpy as np 
    
class NASBench201(): #NASBench201 dataset

    def __init__(self, dataset='ImageNet16-120', output_path='./output', model_path='../datasets/nasbench201_info.pt', device='cpu'):
        self.archive = torch.load(model_path, map_location=device)
        self.num_nodes = 4
        self.num_operations = 5
        self.nvar = int(self.num_nodes*(self.num_nodes-1)/2) #nvar is the len of the encoding. 6 is the number of edges in a 4-node cell
        self.n_archs = 15625 # 5**6 
        self.dataset = dataset
        #self.sec_obj = sec_obj
        self.output_path = output_path

    def str2matrix(self, arch_str: Text,
                search_space: List[Text] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']) -> np.ndarray:
        """
        This func shows how to convert the string-based architecture encoding to the encoding strategy in NAS-Bench-101.

        :param
        arch_str: the input is a string indicates the architecture topology, such as
                        |nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|
        search_space: a list of operation string, the default list is the search space for NAS-Bench-201
            the default value should be be consistent with this line https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/models/cell_operations.py#L24
        :return
        the numpy matrix (2-D np.ndarray) representing the DAG of this architecture topology
        :usage
        matrix = api.str2matrix( '|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|' )
        This matrix is 4-by-4 matrix representing a cell with 4 nodes (only the lower left triangle is useful).
            [ [0, 0, 0, 0],  # the first line represents the input (0-th) node
            [2, 0, 0, 0],  # the second line represents the 1-st node, is calculated by 2-th-op( 0-th-node )
            [0, 0, 0, 0],  # the third line represents the 2-nd node, is calculated by 0-th-op( 0-th-node ) + 0-th-op( 1-th-node )
            [0, 0, 1, 0] ] # the fourth line represents the 3-rd node, is calculated by 0-th-op( 0-th-node ) + 0-th-op( 1-th-node ) + 1-th-op( 2-th-node )
        In NAS-Bench-201 search space, 0-th-op is 'none', 1-th-op is 'skip_connect',
            2-th-op is 'nor_conv_1x1', 3-th-op is 'nor_conv_3x3', 4-th-op is 'avg_pool_3x3'.
        :(NOTE)
        If a node has two input-edges from the same node, this function does not work. One edge will be overlapped.
        """
        node_strs = arch_str.split('+')
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, node_str in enumerate(node_strs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            for xi in inputs:
                op, idx = xi.split('~')
                if op not in search_space: raise ValueError('this op ({:}) is not in {:}'.format(op, search_space))
                op_idx, node_idx = search_space.index(op), int(idx)
                matrix[i+1, node_idx] = op_idx
        return matrix
    
    def matrix2str(self, matrix: np.ndarray,
                  search_space: List[Text] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']) -> Text:
        arch_str = ""
        for i in range(1, self.num_nodes):
            node_str = "|"
            for j, op_idx in enumerate(matrix[i]):
                if i>j:
                    op = search_space[int(op_idx)]
                    node_str += "{}~{}|".format(op, j)
            arch_str += node_str + "+"
        # Remove the trailing '+' character
        arch_str = arch_str[:-1]
        return arch_str

    def matrix2vector(self, matrix):
        # Flatten lower left triangle of the matrix into a vector (diagonal not included)
        #num_nodes = matrix.shape[0]
        vector=[]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i > j:
                    vector.append(int(matrix[i, j]))
        return vector

    def sample(self, n_samples):
        archs = []
        vectors= []
        for i in range(n_samples):
            # Sample a vector of length num_nodes*(num_nodes-1)/2 with values in [0, num_operations) and add if not present
            while True:
                vector = np.random.randint(0, self.num_operations, int(self.num_nodes*(self.num_nodes-1)/2))
                if not any((vector == arr).all() for arr in vectors):
                    vectors.append(vector)
                    break

        for v in vectors:
            assert sum([(v == arr).all() for arr in vectors]) == 1
            arch = {'arch': self.matrix2str(self.vector2matrix(v))}
            archs.append(arch)

        return archs

    def vector2matrix(self,vector):
        #l = len(vector)
        #num_nodes = int((1 + math.sqrt(1+8*l))/2)
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        idx = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i > j:
                    matrix[i, j] = vector[idx]
                    idx += 1
        return matrix
    
    def encode(self, config):
        return self.matrix2vector(self.str2matrix(config['arch']))
    
    def decode(self, vector):
        return {'arch':self.matrix2str(self.vector2matrix(vector))}
    
    def evaluate(self, archs, it=0):
        gen_dir = os.path.join(self.output_path, "iter_{}".format(it))
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir, exist_ok=True)
        #stats = []
        for n_subnet, arch in enumerate(archs):
            stat = self.get_info_from_arch(arch)
            net_path = os.path.join(gen_dir, "net_{}".format(n_subnet))
            if not os.path.exists(net_path):
                os.makedirs(net_path, exist_ok=True)
            save_path = os.path.join(net_path, 'net_{}.stats'.format(n_subnet)) 
            with open(save_path, 'w') as handle:
                json.dump(stat, handle)
            save_path = os.path.join(net_path, 'net_{}.subnet'.format(n_subnet))
            #config={'arch': arch}
            #print("CONFIG: ", arch)
            with open(save_path, 'w') as handle:
                json.dump(arch, handle)
            #f_obj = stat['top1']
            #s_obj = stat[self.sec_obj]
            #stats.append((f_obj, s_obj))
        #return stats

    def get_info_from_arch(self, config):
        str = config['arch']
        matrix = self.str2matrix(str)
        #matrix = self.vector2matrix(config['arch'])
        idx=0
        for i in range(self.n_archs):
            if np.array_equal(self.str2matrix(self.archive['str'][i]), matrix):
                break
            else:
                idx+=1
        info={}
        info['test-acc']=np.round(self.archive['test-acc'][self.dataset][idx],3)
        if self.dataset=='cifar10':
            val_dataset = self.dataset + '-valid'
        else:
            val_dataset = self.dataset
        info['val-acc']=np.round(self.archive['val-acc'][val_dataset][idx],3)
        info['flops']=np.round(self.archive['flops'][self.dataset][idx],3)
        info['params']=np.round(self.archive['params'][self.dataset][idx],3)
        str=self.archive['str'][idx]
        info['top1']=np.round(100 - self.archive['val-acc'][val_dataset][idx],3) # top1 error
        return info
    

