import os
import json
import shutil
import argparse
import glob
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from explainability import get_archive
from trainers.cbn.utils import get_subnet_folder
from ofa_evaluator import OFAEvaluator, get_net_info, get_adapt_net_info

from matplotlib import pyplot as plt


_DEBUG = False
                    
class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        #if self.normalize:
        #    F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            #np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):

    exp_path,_= os.path.splitext(args.expr)
    
    if args.get_archive:
       archive = get_archive(exp_path, args.first_obj, args.sec_obj)
    else:
       archive = json.load(open(args.expr))['archive']

    n_exits = args.n_exits
    if n_exits is not None:
        # filter according to n° of exits
        archive_temp = []
        for v in archive:
            subnet = v[0]
            b_config = subnet["b"]
            count_exits = len([element for element in b_config if element != 0])
            if(count_exits==args.n_exits):
                archive_temp.append(v)
        print("#EEcs:")
        print(args.n_exits)
        print("lunghezza archivio prima")        
        print(len(archive))
        archive = archive_temp
    
    print("NUM CANDIDATES")
    print(len(archive))

    if args.sec_obj is None:
        subnets, first_obj = [v[0] for v in archive], [v[1] for v in archive]
        prefer = args.first_obj
    else:
        subnets, first_obj, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
        prefer = 'trade-off'
        ps_sec_obj = np.array(sec_obj)

    if args.sec_obj is None:
        ps = np.array(subnets)
        ps_first_obj = np.array(first_obj)
        I = ps_first_obj.argsort()[:args.n]
    else:
        sort_idx = np.argsort(first_obj)
        F = np.column_stack((first_obj, sec_obj))[sort_idx, :]
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        ps = np.array(subnets)[sort_idx][front]
        pf = F[front, :]
        # choose the architectures with highest trade-off
        dm = HighTradeoffPoints(n_survive=args.n)
        I = dm.do(pf)

    # always add most accurate architectures
    #I = np.append(I, 0)

    # create the supernet
    #supernet = OFAEvaluator(n_classes = args.n_classes, model_path=args.supernet_path, pretrained = args.supernet_path)

    for rank, idx in enumerate(I):

        if(n_exits is not None):
          save = os.path.join(args.save, "net-"+ prefer +"_"+str(idx)+"_nExits:"+str(args.n_exits))
        else:
          save = os.path.join(args.save, "net-"+ prefer +"_"+str(rank))

        os.makedirs(save, exist_ok=True)
        config = ps[idx]
        print("CONFIG: {}".format(config))

        #subnet, _ = supernet.sample(config)
        subnet_folder = get_subnet_folder(exp_path,config)
        shutil.rmtree(save, ignore_errors=True)
        shutil.copytree(subnet_folder, save)
        #n_subnet = subnet_folder.rsplit("_", 1)[1]
        subnet_file = [filename for filename in os.listdir(save) if filename.endswith('.subnet')][0]
        stats_file = [filename for filename in os.listdir(save) if filename.endswith('.stats')][0]
        os.rename(os.path.join(save, subnet_file), os.path.join(save, "net.subnet"))
        os.rename(os.path.join(save, stats_file), os.path.join(save, "net.stats"))   

        print("SUBNET FOLDER: {}".format(subnet_folder))    

        stats_file = os.path.join(save, "net.stats")
        
        if os.path.exists(stats_file):
            
            stats = json.load(open(stats_file))
            print("INFO SUBNET RANK {}".format(rank))
            print(stats)

    if _DEBUG:
        # Plot

        pf = np.array(pf)
        x = pf[:,0]
        y = pf[:,1]
        plt.scatter(x, y, c='red')

        plt.title('Pareto front')
        plt.xlabel('1-top1')
        plt.ylabel('sec_obj')
        plt.legend()
        plt.show()
        plt.savefig(args.save + 'scatter_plot_pareto_front.png')

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--expr', type=str, default='',
                        help='location of search experiment dir')
    parser.add_argument('--first_obj', type=str, default='top1',
                        help='second objective to optimize')
    parser.add_argument('--sec_obj', type=str, default=None,
                        help='second objective to optimize')
    parser.add_argument('--n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--search_space', type=str, default='mobilenetv3',
                        help='type of search space')
    parser.add_argument('--get_archive', action='store_true', default=False,
                        help='create the archive scanning the iter folders')
    parser.add_argument('--n_classes', type=int, default=1000,
                        help='number of classes')                   
    parser.add_argument('--pmax', type = float, default=2.0,
                        help='max value of params for candidate architecture')
    parser.add_argument('--fmax', type = float, default=100,
                        help='max value of flops for candidate architecture')
    parser.add_argument('--amax', type = float, default=5.0,
                        help='max value of activations for candidate architecture')
    parser.add_argument('--wp', type = float, default=1.0,
                        help='weight for params')
    parser.add_argument('--wf', type = float, default=1/40,
                        help='weight for flops')
    parser.add_argument('--wa', type = float, default=1.0,
                        help='weight for activations')
    parser.add_argument('--penalty', type = float, default=10**10,
                        help='penalty factor')
    parser.add_argument('--n_exits', type=int, default=None,
                        help='number of EEcs desired')
    parser.add_argument('--lr', type = int , default=192,
                        help='minimum resolution')
    parser.add_argument('--ur', type = int, default=256,
                        help='maximum resolution')
    parser.add_argument('--rstep', type = int, default=4,
                        help='resolution step')
    cfgs = parser.parse_args()
    main(cfgs)

