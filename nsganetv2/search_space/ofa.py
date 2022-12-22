import numpy as np


class OFASearchSpace:
    def __init__(self,supernet,lr,ur, blocks):
        self.num_blocks = 5  # number of blocks, default 5
        self.supernet = supernet

        '''
        if(supernet == 'mobilenetv3'):
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
        '''
        if(supernet == 'mobilenetv3'):
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
        elif(supernet == 'resnet50'):
            self.kernel_size = [3]  # depth-wise conv kernel size
            self.exp_ratio = [0.2,0.25,0.35]  # expansion rate
            self.depth = [0,1,2]  # number of Inverted Residual Bottleneck layers repetition          
        elif(supernet == 'resnet50_he'):
            self.num_blocks = 3
            self.kernel_size = [3]  # depth-wise conv kernel size
            self.exp_ratio = [1]  # expansion rate
            self.depth = [2,3,4,5,6,7]  # number of BasicBlock layers repetition          
           
        #STANDARD is lr = 192 and ur= 256
        min = lr
        max = ur + 1
        if (self.supernet != 'resnet50_he'):
          self.resolution = list(range(min, max, 4))
        else:
          self.resolution = list(range(min, max, 1))

    def sample(self, n_samples=1, nb=None, ks=None, e=None, d=None, r=None):
        """ randomly sample a architecture"""
        nb = self.num_blocks if nb is None else nb
        ks = self.kernel_size if ks is None else ks
        e = self.exp_ratio if e is None else e
        d = self.depth if d is None else d
        r = self.resolution if r is None else r

        data = []
        for n in range(n_samples):
            # first sample layers
            depth = np.random.choice(d, nb, replace=True).tolist()

            # then sample kernel size, expansion rate and resolution
            if(self.supernet == 'resnet50_he'):
              kernel_size = np.random.choice(ks, size=len(depth), replace=True).tolist()
              exp_ratio = np.random.choice(e, size=len(depth), replace=True).tolist()
            else:
              kernel_size = np.random.choice(ks, size=int(np.sum(depth)), replace=True).tolist()
              exp_ratio = np.random.choice(e, size=int(np.sum(depth)), replace=True).tolist()

            resolution = int(np.random.choice(r))
            
            data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r': resolution})

        return data

    def initialize(self, n_doe):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = [
            self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                        d=[min(self.depth)], r=[min(self.resolution)])[0],
            self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                        d=[max(self.depth)], r=[max(self.resolution)])[0]
        ]
        data.extend(self.sample(n_samples=n_doe - 2))
        return data

    def pad_zero(self, x, depth):
        # pad zeros to make bit-string of equal length
        new_x, counter = [], 0
        for d in depth:
            for _ in range(d):
                new_x.append(x[counter])
                counter += 1
            if d < max(self.depth):
                new_x += [0] * (max(self.depth) - d)
        return new_x

    def encode(self, config):
        # encode config ({'ks': , 'd': , etc}) to integer bit-string [1, 0, 2, 1, ...]
        x = []
        depth = [np.argwhere(_x == np.array(self.depth))[0, 0] for _x in config['d']]
        kernel_size = [np.argwhere(_x == np.array(self.kernel_size))[0, 0] for _x in config['ks']]
        exp_ratio = [np.argwhere(_x == np.array(self.exp_ratio))[0, 0] for _x in config['e']]

        if(self.supernet != 'resnet50_he'):
            kernel_size = self.pad_zero(kernel_size, config['d'])
            exp_ratio = self.pad_zero(exp_ratio, config['d'])
            for i in range(len(depth)):
              x = x + [depth[i]] + kernel_size[i * max(self.depth):i * max(self.depth) + max(self.depth)] \
                  + exp_ratio[i * max(self.depth):i * max(self.depth) + max(self.depth)]
        else:
            for i in range(len(depth)):
                x = x + [depth[i]] + [kernel_size[i]] + [exp_ratio[i]]
          
        x.append(np.argwhere(config['r'] == np.array(self.resolution))[0, 0])

        return x

    def decode(self, x):
        """
        remove un-expressed part of the chromosome
        assumes x = [block1, block2, ..., block5, resolution, width_mult];
        block_i = [depth, kernel_size, exp_rate]
        """

        depth, kernel_size, exp_rate = [], [], []
        if(self.supernet != 'resnet50_he'):
          for i in range(0, len(x) - 2, 9):
              depth.append(self.depth[x[i]])
              kernel_size.extend(np.array(self.kernel_size)[x[i + 1:i + 1 + self.depth[x[i]]]].tolist())
              exp_rate.extend(np.array(self.exp_ratio)[x[i + 5:i + 5 + self.depth[x[i]]]].tolist())
        else:
          for i in range(0,len(x)-1,self.num_blocks):
              depth.append(self.depth[x[i]])
              kernel_size.append(self.kernel_size[x[i+1]])
              exp_rate.append(self.exp_ratio[x[i+2]])
              
        return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'r': self.resolution[x[-1]]}


