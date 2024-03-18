import numpy as np

class OFASearchSpace:
    
    def __init__(self,supernet,lr,ur,rstep):

        self.num_blocks = 5  # number of blocks, default 5
        self.supernet = supernet

        if(supernet == 'mobilenetv3'):
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.nvar= 45 + int(lr!=ur) # length of the encoding 45 if fix_res else 46
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition 
        elif(supernet == 'eemobilenetv3'): # Early Exit Mbv3 (EDANAS)
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
            self.threshold = [0.1, 0.2, 1] #threshold value for selection scheme
            self.nvar=50
        elif(supernet == 'resnet50'):
            self.kernel_size = [3]  # depth-wise conv kernel size
            self.exp_ratio = [0.2,0.25,0.35]  # expansion rate
            self.depth = [0,1,2]  # number of Inverted Residual Bottleneck layers repetition    
            self.nvar=46      
        elif(supernet == 'resnet50_he'):
            self.num_blocks = 3
            self.kernel_size = [3]  # depth-wise conv kernel size
            self.exp_ratio = [1]  # expansion rate
            self.depth = [2,3,4,5,6,7]  # number of BasicBlock layers repetition 
            self.nvar=46
        elif(supernet == 'cbnmobilenetv3'): #NACHOS
            self.kernel_size = [3, 5, 7]  # depth-wise conv kernel size
            self.exp_ratio = [3, 4, 6]  # expansion rate
            self.depth = [2, 3, 4]  # number of Inverted Residual Bottleneck layers repetition
            self.num_branches=4
            self.branches = [0,1] #0=EEC, 1=No EEC
            self.nvar=49 + int(lr!=ur)
        else:
            raise NotImplementedError

        #STANDARD is lr = 192 and ur= 256
        min = lr
        max = ur + 1
        self.resolution = list(range(min, max, rstep))
        self.fix_res = lr==ur

    def sample(self, n_samples=1, nb=None, ks=None, e=None, d=None, t = None, r=None, b=None):

        """ randomly sample a architecture"""
        nb = self.num_blocks if nb is None else nb
        ks = self.kernel_size if ks is None else ks
        e = self.exp_ratio if e is None else e
        d = self.depth if d is None else d
        if (self.supernet == 'eemobilenetv3'):
          t = self.threshold if t is None else t
        r = self.resolution if r is None else r
        if (self.supernet == 'cbnmobilenetv3'):
          #ne = self.ne if ne is None else ne
          b = self.branches if b is None else b
    
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

            if (self.supernet == 'eemobilenetv3'):
                threshold = np.random.choice(t, size=(len(depth)-1), replace=True).tolist()
                data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 't': threshold, 'r': resolution})

            elif (self.supernet == 'mobilenetv3'):

                if self.fix_res:
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth})
                else:
                    resolution = int(np.random.choice(r))
                    data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'r':resolution})

            elif (self.supernet == 'cbnmobilenetv3'):

                #avoid branches [0,0,0,0] = no EECs
                while True:
                    branches = np.random.choice(b, self.num_branches, replace=True).tolist()
                    if any(branch != 0 for branch in branches):
                        break

                data.append({'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'b': branches}) 

        return data

    def initialize(self, n_doe):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        if (self.supernet == 'mobilenetv3'):

            if self.fix_res:
                data = [
                self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                            d=[min(self.depth)])[0],
                self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                            d=[max(self.depth)])[0]]
            else:
                data = [
                    self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                                d=[min(self.depth)], r=[min(self.resolution)])[0],
                    self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                                d=[max(self.depth)], r=[max(self.resolution)])[0]
                ]

        elif (self.supernet == 'eemobilenetv3'):
            data = [
                self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                            d=[min(self.depth)], t = [min(self.threshold)], r=[min(self.resolution)])[0],
                self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                            d=[max(self.depth)], t = [max(self.threshold)], r=[max(self.resolution)])[0]
            ]
        elif (self.supernet == 'cbnmobilenetv3'):
            data = [
                self.sample(1, ks=[min(self.kernel_size)], e=[min(self.exp_ratio)],
                            d=[min(self.depth)], b=[min(self.branches)])[0], # all EECs
                self.sample(1, ks=[max(self.kernel_size)], e=[max(self.exp_ratio)],
                            d=[max(self.depth)], b=[max(self.branches)])[0]  # no EECs
            ]
        else:
            print("Not yet implemented!")

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
        
        if (self.supernet == 'eemobilenetv3'):
            idxs = [np.argwhere(_x == np.array(self.threshold))[0, 0] for _x in config['t']]
            x = x + idxs

        if (self.supernet == 'cbnmobilenetv3'):
            branches = config['b']
            for i in range(self.num_branches):
                x = x + [np.argwhere(branches[i] == np.array(self.branches))[0, 0]]
        else:
            if not self.fix_res:
                x.append(np.argwhere(config['r'] == np.array(self.resolution))[0, 0])

        return x

    def decode(self, x):
        """
        remove un-expressed part of the chromosome
        assumes x = [block1, block2, ..., block5, resolution, width_mult];
        block_i = [depth, kernel_size, exp_rate]
        """
        
        x = x.astype(int)

        depth, kernel_size, exp_rate = [], [], []
        step = 1 + 2 * max(self.depth)
        if(self.supernet != 'resnet50_he'):
          for i in range(0, len(x) - 5, step):
              depth.append(self.depth[x[i]])
              kernel_size.extend(np.array(self.kernel_size)[x[i + 1:i + 1 + self.depth[x[i]]]].tolist())
              exp_rate.extend(np.array(self.exp_ratio)[x[i + 5:i + 5 + self.depth[x[i]]]].tolist())
        else:
          for i in range(0,len(x)- 1,self.num_blocks):
              depth.append(self.depth[x[i]])
              kernel_size.append(self.kernel_size[x[i+1]])
              exp_rate.append(self.exp_ratio[x[i+2]])
        
        if (self.supernet == 'mobilenetv3'):

            if self.fix_res:
                return {'ks': kernel_size, 'e': exp_rate, 'd': depth}
            else: 
                return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'r': self.resolution[x[-1]]}    

        elif (self.supernet == 'eemobilenetv3'): 
            t_config = x[-self.num_blocks:-1]
            t = []
            for c in t_config:
              t.append(self.threshold[c])   
            
            return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 
            't': t, 'r': self.resolution[x[-1]]}

        elif (self.supernet == 'cbnmobilenetv3'):
            branches = []
            for i in range(len(x) - self.num_branches, len(x)):
              branches.append(self.branches[x[i]])
            return {'ks': kernel_size, 'e': exp_rate, 'd': depth, 'b': branches}
        else: 
            print("Not yet implemented!")



