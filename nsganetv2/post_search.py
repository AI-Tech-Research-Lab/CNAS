import os
import json
import argparse
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder

from matplotlib import pyplot as plt


_DEBUG = True


class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

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

            np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):

    ##compute the pareto front 
    archive = json.load(open(args.expr))['archive']

    n_exits = args.n_exits
    if n_exits is not None:
        # filter according to n° of exits
        archive_temp = []
        for v in archive:
            subnet = v[0]
            t = subnet["t"]
            count_exits = len(t)-t.count(1)
            if(count_exits==args.n_exits):
                archive_temp.append(v)
        print("#EEcs:")
        print(args.n_exits)
        print("lunghezza archivio prima")        
        print(len(archive))
        print("lunghezza archivio dopo")        
        print(len(archive_temp))
        archive = archive_temp
    
    for v in archive: #remove failed nets
        err_top1 = v[1]
        if(err_top1==100):
          archive.remove(v)

    print("ARCHIVE")
    print(v[0])
    print(v[1])
    print(v[2])
    #print(v[3])
    
    subnets, top1, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]

    sort_idx = np.argsort(top1)
    F = np.column_stack((top1, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pf = F[front, :]
    print(pf)
    ps = np.array(subnets)[sort_idx][front]
    ps_top1 = np.array(top1)[sort_idx][front]
    ps_sec_obj = np.array(sec_obj)[sort_idx][front]
    #ps_util = np.array(util)[sort_idx][front]

    if args.prefer != 'trade-off':
        # choose the best architecture for the sec_obj
        I = pf[:,1].argsort()
        I = I[:args.n]
    else:
        # choose the architectures with highest trade-off
        dm = HighTradeoffPoints(n_survive=args.n)
        I = dm.do(pf)

    # always add most accurate architectures
    I = np.append(I, 0)

    # create the supernet
    from evaluator import OFAEvaluator, get_net_info, get_adapt_net_info
    supernet = OFAEvaluator(n_classes = args.n_classes, model_path=args.supernet_path, pretrained = args.supernet_path)

    for idx in I:
        if(n_exits is not None):
          save = os.path.join(args.save, "net-"+args.prefer+"_"+str(idx)+"@{:.0f}".format(pf[idx, 1])+"_nExits:"+str(args.n_exits))
        else:
          save = os.path.join(args.save, "net-"+args.prefer+"_"+str(idx)+"@{:.0f}".format(pf[idx, 1]))
        #save = os.path.join(args.save, "net-"+args.prefer+"_"+str(args.n_exits)+"@{:.0f}".format(pf[idx, 1]))
        os.makedirs(save, exist_ok=True)
        subnet, _ = supernet.sample({'ks': ps[idx]['ks'], 'e': ps[idx]['e'], 'd': ps[idx]['d'], 't': ps[idx]['t']})
        with open(os.path.join(save, "net.subnet"), 'w') as handle:
            json.dump(ps[idx], handle)
        supernet.save_net_config(save, subnet, "net.config")
        supernet.save_net(save, subnet, "net.inherited")
        data_shape = (3,ps[idx]['r'],ps[idx]['r'])
        info = get_adapt_net_info(subnet,data_shape,pmax = args.pmax, fmax = args.fmax, amax = args.amax,
                  wp = args.wp, wf = args.wf, wa = args.wa, penalty = args.penalty)
        info['avg_macs'] = ps_sec_obj[idx] #update value with the avg_macs
        info['top1'] = 100 - ps_top1[idx]
        #info['util'] = list(ps_util[idx])
        with open(os.path.join(save, "net.stats"), "w") as handle:
                json.dump(info, handle)
   
    if args.save_stats_csv:
        
        import pandas as pd

        infos = [] ## array not dict
        idx = 0
        for s in subnets:
            subnet, _ = supernet.sample({'ks': subnets[idx]['ks'], 'e': subnets[idx]['e'], 'd': subnets[idx]['d'], 't': subnets[idx]['t']})
            data_shape = (3,subnets[idx]['r'],subnets[idx]['r'])
            info = get_adapt_net_info(subnet,data_shape,pmax = args.pmax, fmax = args.fmax, amax = args.amax,
                  wp = args.wp, wf = args.wf, wa = args.wa, penalty = args.penalty)
            info["top1"] = 100 - top1[idx]
            info["avg_macs"] = sec_obj[idx] #update value with the avg_macs
            info["subnet"] = subnets[idx]
            infos.append(info)
            idx = idx + 1

        df = pd.DataFrame(infos)
        df.to_csv(args.save + '/results.csv')

    if _DEBUG:
        # Plot
        x = pf[:,0]
        y = pf[:,1]
        plt.scatter(x, y, c='red')
        plt.title('Pareto front')
        plt.xlabel('1-top1')
        plt.ylabel('sec_obj')
        plt.show()
        plt.savefig(args.save + 'scatter_plot_pareto_front.png')
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--expr', type=str, default='',
                        help='location of search experiment dir')
    parser.add_argument('--sec_obj', type=str, default='params',
                        help='second objective to optimize')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--prefer', type=str, default='trade-off',
                        help='preferences in choosing architectures (top1#80+flops#150)')
    parser.add_argument('--save_stats_csv', action='store_true', default=False,
                        help='save csv with all stats')
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
    cfgs = parser.parse_args()
    main(cfgs)
