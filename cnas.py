import os
import json
import shutil
import argparse
import subprocess
from nasbench201 import NASBench201
import numpy as np
import math
import datetime
import torch

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from utils import get_correlation, get_net_info, tiny_ml
from ofa_evaluator import OFAEvaluator

from search_space import OFASearchSpace
from acc_predictor.factory import get_acc_predictor
from utils import prepare_eval_folder, MySampling, BinaryCrossover, MyMutation
from train_utils import initialize_seed
from explainability import get_archive

_DEBUG = False
if _DEBUG: from pymoo.visualization.scatter import Scatter


class CNAS:
    def __init__(self, kwargs):
        self.save_path = kwargs.pop('save', '.tmp')  # path to save results
        self.resume = kwargs.pop('resume', None)  # resume search from a checkpoint
        self.first_obj = kwargs.pop('first_obj', 'top1')  # 1st objective
        self.sec_obj = kwargs.pop('sec_obj', None)  # 2nd objective
        self.iterations = kwargs.pop('iterations', 30)  # number of iterations to run search
        self.n_doe = kwargs.pop('n_doe', 100)  # number of architectures to train before fit surrogate model
        self.n_iter = kwargs.pop('n_iter', 8)  # number of architectures to train in each iteration
        self.first_predictor = kwargs.pop('first_predictor', 'gp')  # surrogate 1st objective
        self.sec_predictor = kwargs.pop('sec_predictor', None)  # surrogate 2nd objective
        self.n_gpus = kwargs.pop('n_gpus', 1)  # number of available gpus
        self.gpu = kwargs.pop('gpu', 1)  # required number of gpus per evaluation job
        self.gpu_list = kwargs.pop('gpu_list', None)  # list of ids of available gpus
        print("GPU LIST:", str(self.gpu_list))
        self.data = kwargs.pop('data', '../data')  # location of the data files
        self.dataset = kwargs.pop('dataset', 'imagenet')  # which dataset to run search on
        self.model = kwargs.pop('model', 'mobilenetv3') 
        self.n_classes = kwargs.pop('n_classes', 1000)  # number of classes of the given dataset
        self.n_workers = kwargs.pop('n_workers', 6)  # number of threads for dataloader
        self.val_split = kwargs.pop('val_split', 0.0)  # 'percentage of train set for validation'
        self.trn_batch_size = kwargs.pop('trn_batch_size', 96)  # batch size for SGD training
        self.vld_batch_size = kwargs.pop('vld_batch_size', 250)  # batch size for validation
        self.n_epochs = kwargs.pop('n_epochs', 5)  # number of epochs to SGD training
        #self.test = kwargs.pop('test', True)  # evaluate performance on test set
        self.supernet_path = kwargs.pop(
            'supernet_path', './supernets/ofa_mbv3_d234_e346_k357_w1.0')  # supernet model path
        self.search_space = kwargs.pop(
            'search_space', 'mobilenetv3')  # supernet type
        self.pretrained = kwargs.pop('pretrained',True) #use pretrained weights
        #self.latency = self.sec_obj if "cpu" in self.sec_obj or "gpu" in self.sec_obj else None
        self.lr = kwargs.pop('lr',224) #minimum resolution
        self.ur = kwargs.pop('ur',224) #maximum resolution
        self.rstep = kwargs.pop('rstep',4) #resolution step
        self.seed = kwargs.pop('seed', 0)  # random seed
        self.optim = kwargs.pop('optim', "SGD") # training optimizer
        # Trainer type 
        self.trainer_type = kwargs.pop('trainer_type', 'single-exit')
        # Technological constraints params
        self.pmax = kwargs.pop('pmax',2) #max value of params of the candidate architecture
        self.mmax = kwargs.pop('mmax',100) #max value of flops of the candidate architecture
        self.amax = kwargs.pop('amax',5) #max value of activations of the candidate architecture
        self.wp = kwargs.pop('wp',1) #weight for params 
        self.wm = kwargs.pop('wm',1/40) #weight for macs
        self.wa = kwargs.pop('wa',1) #weight for activations
        self.penalty = kwargs.pop('penalty',10**10) #static penalty factor
        # Functional constraints params
        self.func_constr = kwargs.pop('func_constr',False) #use functional constraints
        # Robustness params
        self.sigma_min = kwargs.pop('sigma_min', 0.05) # min noise perturbation intensity
        self.sigma_max = kwargs.pop('sigma_max', 0.05) # max noise perturbation intensity
        self.sigma_step = kwargs.pop('sigma_step', 0) # noise perturbation intensity step
        self.alpha = kwargs.pop('alpha', 0.5) # alpha parameter for entropic figure
        sigma_step=self.sigma_step
        if self.sigma_max == self.sigma_min:
            sigma_step = 1
        n=round((self.sigma_max-self.sigma_min)/sigma_step)+1
        self.alpha_norm = 1.0 # alpha factor for entropic training
        # Early Exit params
        self.method = kwargs.pop('method', 'bernulli') # method for early exit training
        self.support_set = kwargs.pop('support_set', False) # use support set for early exit training
        self.tune_epsilon = kwargs.pop('tune_epsilon', False) # tune epsilon for early exit inference
        self.top1min = kwargs.pop('top1min', 0.1) #top1 constraint
        self.w_alpha = kwargs.pop('w_alpha', 1.0) # weight for alpha factor
        self.w_beta = kwargs.pop('w_beta', 1.0)
        self.w_gamma = kwargs.pop('w_gamma', 1.0)
        self.warmup_ee_epochs = kwargs.pop('warmup_ee_epochs', 5) # warmup epochs for early exit
        self.ee_epochs = kwargs.pop('ee_epochs', 0) # early exit epochs with support set

        if self.search_space != 'nasbench':
            self.search_space = OFASearchSpace(self.search_space, self.lr, self.ur, self.rstep)
        else:
            self.search_space = NASBench201(self.dataset, self.save_path)

    def search(self):

        use_cuda = torch.cuda.is_available() and self.gpu_list
        initialize_seed(self.seed, use_cuda)
        it_start = 1
        if self.resume:
            archive = self._resume_from_dir()
            split = self.resume.rsplit("_",1)
            it_start = int(split[1])
            it_start = it_start + 1

        else:

            # the following lines corresponding to Algo 1 line 1-7 in the paper
            archive = []  # initialize an empty archive to store all trained CNNs

            arch_doe = self.search_space.sample(n_samples = self.n_doe)
            
            stats = self._evaluate(arch_doe, it=0)

            for arch, info in zip(arch_doe,stats):
                archive.append((arch,*info))

        # reference point (nadir point) for calculating hypervolume
        if self.sec_obj is not None:
            ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])

        # main loop of the search
        for it in range(it_start, it_start + self.iterations):

            if self.first_obj == 'top1_robust':
              # Compute the new alpha_factor and update the archive with the new alpha_factor
              self.alpha_norm = self.compute_alpha_norm(os.path.join(self.save_path, "iter_"+str(it-1)))
              temp=[]
              for x in archive:
                  temp.append((x[0],x[3]*self.alpha + self.alpha_norm*(1-self.alpha)*x[4],x[2],x[3],x[4]))
              archive=temp

            # construct predictor surrogates model from archive
            if self.first_predictor is not None:
                first_predictor, a_first_err_pred = self._fit_first_predictor(archive)
            
            sec_predictor=None
            if self.sec_predictor is not None:
                sec_predictor, a_sec_err_pred = self._fit_sec_predictor(archive)
            
            # search for the next set of candidates for high-fidelity evaluation (lower level)
            if self.sec_obj is None:
                print("Optimizing for single objective")
                candidates, c_first_err_pred = self._nextSingleObj(archive, first_predictor, self.n_iter)
            else:
                print("Optimizing for multi-objective")
                candidates, c_first_err_pred, c_sec_err_pred = self._nextMultiObj(archive, first_predictor, sec_predictor, self.n_iter)

            # high-fidelity evaluation (lower level)
            # Algo 1 line 13-14 / Fig. 3(e) in the paper
            stats = self._evaluate(candidates, it=it)
            c_first_err = [t[0] for t in stats]
            complexity = [t[1] for t in stats]

            if self.first_predictor is not None:
            # check for accuracy predictor's performance
                rmse, rho, tau = get_correlation(
                    np.vstack((a_first_err_pred, c_first_err_pred)), np.array([x[1] for x in archive] + c_first_err))

            if self.sec_predictor is not None:
                # check for complexity predictor's performance
                rmse_c, rho_c, tau_c = get_correlation(
                    np.vstack((a_sec_err_pred, c_sec_err_pred)), np.array([x[2] for x in archive] + complexity))   

            n_candidates = self.n_iter       
            
            for arch, info in zip(candidates,stats):
                duplicate=False
                if isinstance(self.search_space,NASBench201):
                    for x in archive:
                        if x[0]['arch'] == arch['arch']:
                            duplicate=True
                if not duplicate:
                    archive.append((arch,*info))
                else:
                    n_candidates-=1
            
            if self.first_predictor is not None:
                print("fitting {}: RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall’s Tau = {:.4f}".format(
                    self.first_predictor, rmse, rho, tau))
                stats={'archive': archive, 'candidates': archive[-n_candidates:],
                            'first_surrogate': {
                                'model': self.first_predictor, 'name': first_predictor.name,
                                'winner': first_predictor.winner if self.first_predictor == 'as' else first_predictor.name,
                                'rmse': rmse, 'rho': rho, 'tau': tau}}
            
            if self.sec_predictor is not None:
                print("fitting {}: RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall’s Tau = {:.4f}".format(
                    self.sec_predictor, rmse_c, rho_c, tau_c))
                stats['sec_surrogate']={
                               'model': self.sec_predictor, 'name': sec_predictor.name,
                               'winner': sec_predictor.winner if self.sec_predictor == 'as' else sec_predictor.name,
                               'rmse': rmse_c, 'rho': rho_c, 'tau': tau_c, 'phi': self.phi}
            
            if self.sec_obj is not None:
                # calculate hypervolume
                hv = self._calc_hv(ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))
                # print iteration-wise statistics
                print("Iter {}: hv = {:.2f}".format(it, hv))
                stats['hv']=hv
            
            if self.first_obj =='top1_robust':
                stats['alpha_norm']=self.alpha_norm

            # dump the statistics
            with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                json.dump(stats, handle)
            
            #with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                #json.dump({'archive': archive, 'candidates': archive[-self.n_iter:]}, handle)

            if _DEBUG:
                # plot
                plot = Scatter(legend={'loc': 'lower right'})
                F = np.full((len(archive), 2), np.nan)
                F[:, 0] = np.array([x[2] for x in archive])  # second obj. (complexity)
                F[:, 1] = 100 - np.array([x[1] for x in archive])  # top-1 accuracy
                plot.add(F, s=15, facecolors='none', edgecolors='b', label='archive')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - np.array(c_first_err)
                plot.add(F, s=30, color='r', label='candidates evaluated')
                F = np.full((len(candidates), 2), np.nan)
                F[:, 0] = np.array(complexity)
                F[:, 1] = 100 - c_first_err_pred[:, 0]
                plot.add(F, s=20, facecolors='none', edgecolors='g', label='candidates predicted')
                plot.save(os.path.join(self.save_path, 'iter_{}.png'.format(it)))
        
        return
    
    def compute_alpha_norm(self,exp_path):
        archive = get_archive(exp_path,'top1','robustness')
        top1_err=[]
        robustness=[]
        for x in archive:
            top1_err.append(x[1])
            robustness.append(x[2])
        robustness = np.array(robustness)
        top1_err = np.array(top1_err)
        # avg robustness/top1 ratio
        mean_r = np.mean(robustness)
        mean_top1_err= np.mean(top1_err)
        robust_factor = mean_top1_err/mean_r
        return robust_factor
    
    def _resume_from_dir(self):
        """ resume search from a previous iteration """
        import glob
        archive = []
        split = self.resume.rsplit("_",1)
        maxiter = int(split[1])
        path = split[0]
        
        for file in glob.glob(os.path.join(path + '_*', "net_*/net_*.subnet")):
            arch = json.load(open(file))#['arch']
            pre,ext= os.path.splitext(file)
            split = pre.rsplit("_",3)  
            split2 = split[1].rsplit("/",1)
            niter = int(split2[0])
            split = pre.rsplit("_",2)  
            split2 = split[1].rsplit("/",1)
            nsubnet = int(split2[0])
            if (niter <= maxiter):

                path = pre + ".stats"

                #Remove duplicates
                for x in archive:
                    if x[0] == arch:
                        archive.remove(x) 
                        break
                
                if (os.path.exists(path)):
                    
                    stats = json.load(open(path))

                    first_obj = stats[self.first_obj]

                    sec_obj = stats.get(self.sec_obj, None)
                    
                    if self.sec_obj is not None:
                        v = (arch, first_obj, sec_obj)
                    else:
                        v = (arch, first_obj)
                    
                    if self.first_obj == 'top1_robust':
                        v = v + (stats['top1'], stats['robustness'],)

                    archive.append(v)

                else: #failed net
                    print("FAILED NET")
                    print(path)
                    print(nsubnet)
                    '''
                    if self.sec_obj is not None:
                        v = (arch, 100, 10**15)
                    else:
                        v = (arch, 100)
                    archive.append(v)    
                    '''
        
        print("LEN ARCHIVE")    
        print(len(archive))
        print(archive[:10])
    
        return archive

    def _evaluate(self, archs, it): #Train and evaluate subnets and save them in folders

        gen_dir = os.path.join(self.save_path, "iter_{}".format(it))

        if isinstance(self.search_space, OFASearchSpace):

            prepare_eval_folder(
                gen_dir, archs, self.gpu, self.n_gpus, 
                self.gpu_list, self.trainer_type, n_workers = self.n_workers,
                data=self.data, dataset=self.dataset, model=self.model, pmax = self.pmax, 
                mmax =self.mmax, amax = self.amax, wp=self.wp, wm=self.wm, wa=self.wa,
                top1min=self.top1min, penalty = self.penalty, func_constr=self.func_constr, supernet_path=self.supernet_path, pretrained=self.pretrained, 
                n_epochs = self.n_epochs, optim=self.optim, sigma_min=self.sigma_min,
                sigma_max=self.sigma_max, sigma_step=self.sigma_step, alpha=self.alpha, res=self.lr, alpha_norm=self.alpha_norm, val_split=self.val_split,
                method = self.method, support_set = self.support_set, tune_epsilon = self.tune_epsilon, 
                w_alpha = self.w_alpha, w_beta = self.w_beta, w_gamma = self.w_gamma, 
                warmup_ee_epochs = self.warmup_ee_epochs, ee_epochs = self.ee_epochs)

            subprocess.call("sh {}/run_bash.sh".format(gen_dir), shell=True)
        else:
            self.search_space.evaluate(archs, it)

        all_stats=[]
        for i in range(len(archs)):
            try:
                stats = json.load(open(os.path.join(gen_dir, "net_{}/net_{}.stats".format(i,i))))
            except FileNotFoundError:
                # just in case the subprocess evaluation failed
                stats = {self.first_obj: 0, self.sec_obj: 10**15}  # makes the solution artificially bad so it won't survive
                # store this architecture to a separate in case we want to revisit after the search
                os.makedirs(os.path.join(self.save_path, "failed"), exist_ok=True)
                shutil.copy(os.path.join(gen_dir, "net_{}/net_{}.subnet".format(i)),
                            os.path.join(self.save_path, "failed", "it_{}_net_{}".format(it, i)))
            
            f_obj=stats[self.first_obj]
            #s_obj=stats[self.sec_obj]
            s_obj=stats.get(self.sec_obj, stats['params'])
            stat=(f_obj,s_obj)

            if self.first_obj=='top1_robust':
                stat=stat+(stats['top1'], stats['robustness'],)

            all_stats.append(stat)
        
        return all_stats
        

    def _fit_first_predictor(self, archive):

        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])
        assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        acc_predictor = get_acc_predictor(self.first_predictor, inputs, targets)

        return acc_predictor, acc_predictor.predict(inputs)

    def _fit_sec_predictor(self, archive):
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[2] for x in archive])
        assert len(inputs) > len(inputs[0]), "# of training samples have to be > # of dimensions"

        acc_predictor = get_acc_predictor(self.sec_predictor, inputs, targets)

        return acc_predictor, acc_predictor.predict(inputs)

    def _nextSingleObj(self, archive, acc_predictor, K):

        # Sort the archive by error accuracy in ascending order
        archive.sort(key=lambda x: x[1])

        # Extract the top M subnets with the highest accuracy to init the population
        top_K_subnets = np.array([self.search_space.encode(x[0]) for x in archive[:K]])

        problem = AuxiliarySingleObjProblem(self.search_space, acc_predictor)  #optimize only accuracy (1st obj)

        method = GA(
                pop_size=40,
                sampling=top_K_subnets,  
                crossover=TwoPointCrossover(prob=0.9),#get_crossover("int_two_point", prob=0.9),
                mutation=PolynomialMutation(eta=1.0),#get_mutation("int_pm", eta=1.0),
                eliminate_duplicates=True) 

        # kick-off the search
        res = minimize(
            problem, method, termination=('n_gen', 60), save_history=True, verbose=True) #verbose=True displays some printouts

        #X is the set of optimal archs sorted in acc_error ascending order      
        X=res.pop.get("X")

        # check for duplicates in the archive
        not_duplicate = np.logical_not([x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in X]])

        X=X[not_duplicate]


        candidates = []
        
        # keep the top K archs
        for x in X[:K]:
            candidates.append(self.search_space.decode(x))
        
        return candidates, acc_predictor.predict(X[:K])


    def _nextMultiObj(self, archive, acc_predictor, compl_predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # the following lines corresponding to Algo 1 line 10 / Fig. 3(b) in the paper
        # get non-dominated architectures from archive
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))

        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(
            self.search_space, acc_predictor, compl_predictor, self.sec_obj, self.dataset,
            {'n_classes': self.n_classes, 'supernet_path': self.supernet_path, 'pretrained': self.pretrained},
            pmax = self.pmax, mmax = self.mmax, amax = self.amax, wp = self.wp, wm = self.wm, wa = self.wa, penalty = self.penalty)

        # initiate a multi-objective solver to optimize the problem
        method = NSGA2(pop_size=40, sampling=nd_X,  # initialize with current nd archs
            crossover=TwoPointCrossover(prob=0.9),
            mutation=PolynomialMutation(eta=1.0),
            eliminate_duplicates=True)
        
        # kick-off the search
        res = minimize(
            problem, method, termination=('n_gen', 20), save_history=True, verbose=True) #verbose=True displays some printouts #default 20 generations

        self.phi = problem.phi
        print("The ratio of feasible solutions (phi) is {:.2f}".format(problem.phi))

        # check for duplicates
        not_duplicate = np.logical_not([x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])

        # the following lines corresponding to Algo 1 line 11 / Fig. 3(c)-(d) in the paper
        # form a subset selection problem to short list K from pop_size
        indices = self._subset_selection(res.pop[not_duplicate], F[front, 1], K)
        pop = res.pop[not_duplicate][indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err

        pred_top1_err = acc_predictor.predict(pop.get("X"))
        pred_compl = None
        if(compl_predictor is not None):
            pred_compl = compl_predictor.predict(pop.get("X"))
        
        return candidates, pred_top1_err, pred_compl

    @staticmethod
    def _subset_selection(pop, nd_F, K):
        problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
        algorithm = GA(
            pop_size=100, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        # calculate hypervolume on the non-dominated set of F
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        ind = HV(ref_point=ref_point)
        hv = ind(nd_F) #get_performance_indicator("hv", ref_point=ref_point).calc(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv

class AuxiliarySingleObjProblem(Problem):

    def __init__(self, search_space, acc_predictor):

        super().__init__(n_var=search_space.nvar, n_obj=1, n_constr=0)

        self.ss=search_space
        self.acc_predictor = acc_predictor

        self.xl = np.zeros(self.n_var) #lower bounds
        if isinstance(self.ss,NASBench201):
            self.xu = self.ss.num_operations * np.ones(self.n_var) #upper bounds
        else:
            self.xu = 2 * np.ones(self.n_var) #upper bounds
    
    def _evaluate(self, x, out, *args, **kwargs):

        f = np.full((x.shape[0], self.n_obj), np.nan)

        top1_err = self.acc_predictor.predict(x)[:, 0]  # predicted top1 error

        for i,err in enumerate(top1_err):
            f[i,0]=abs(err)

        out["F"] = f


class AuxiliarySingleLevelProblem(Problem):
    """ The optimization problem for finding the next N candidate architectures """

    def __init__(self, search_space, acc_predictor, compl_predictor=None, sec_obj='flops', dataset='imagenet',supernet=None, pmax = 2, mmax = 100, amax = 5,
        wp = 1, wm = 1/40, wa = 1, penalty = 10**10):
        
        super().__init__(n_var=search_space.nvar, n_obj=2, n_constr=0) #type = np.int deprecated

        self.ss = search_space
        self.acc_predictor = acc_predictor
        self.compl_predictor = compl_predictor
        self.xl = np.zeros(self.n_var) #lower bounds
        if isinstance(self.ss,NASBench201):
            self.xu = self.ss.num_operations * np.ones(self.n_var) #upper bounds
        else:
            self.xu = 2 * np.ones(self.n_var) #upper bounds
        '''
        if self.ss=='cbnmobilenetv3':
          self.xu[-1] = 1 #EEC on/off
        else:
          self.xu[-1] = int(len(self.ss.resolution) - 1)
        '''
        self.sec_obj = sec_obj
        self.dataset = dataset
        self.lut = {'cpu': 'data/i7-8700K_lut.yaml'}
        self.pmax = pmax
        self.mmax = mmax
        self.amax = amax
        self.wp = wp
        self.wm = wm
        self.wa = wa
        self.penalty = penalty
        self.phi = 0

        self.engine = OFAEvaluator(
            n_classes=supernet['n_classes'], model_path=supernet['supernet_path'], pretrained = supernet['pretrained'] )
        
    def _evaluate(self, x, out, *args, **kwargs):

        f = np.full((x.shape[0], self.n_obj), np.nan)

        top1_err = self.acc_predictor.predict(x)[:, 0]  # predicted top1 error

        if self.compl_predictor is not None and self.ss.supernet == 'cbnmobilenetv3':

            compl = self.compl_predictor.predict(x)[:, 0]  # predicted compl error
            constraint = self.mmax
            #compute the ratio of feasible solutions in the population (phi)
            phi = len([el for el in compl if el <= constraint])/len(compl)
            self.phi=phi
            cmax = max(compl)

            for i, (_x, acc_err, ci) in enumerate(zip(x, top1_err, compl)):

                if not self._isvalid(_x):
                    f[i,0] = 10*15
                    f[i,1] = 10*15
                    continue

                ## Compute the normalized constraint violation (CV) (NACHOS)
                if(cmax!=constraint):
                    cv = max(0,(ci-constraint))/abs(cmax-constraint) 
                else:
                    cv = 0   
                sec_obj = phi*ci + (1-phi)*cv

                f[i, 0] = acc_err
                f[i, 1] = sec_obj 
        
        elif isinstance(self.ss,NASBench201):
                for i,(_x,acc_err) in enumerate(zip(x,top1_err)):
                    arch = self.ss.matrix2str(self.ss.vector2matrix(_x))
                    stats = self.ss.get_info_from_arch({'arch':arch})
                    f[i,0] = acc_err
                    f[i,1] = stats.get(self.sec_obj,None)
        else:

            for i, (_x, acc_err) in enumerate(zip(x, top1_err)):

                if(self.ss.supernet == 'resnet50_he'):
                    if not self._isvalid(_x):
                        f[i,0] = 10*15
                        f[i,1] = 10*15
                        continue

                if(self.ss.supernet == 'cbnmobilenetv3'):
                    if not self._isvalid(_x):
                        f[i,0] = 10*15
                        f[i,1] = 10*15
                        continue

                config = self.ss.decode(_x)

                if(self.ss.supernet == 'eemobilenetv3'):

                    subnet, _ = self.engine.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d'], 't': config['t']})
                else:
                    subnet, _ = self.engine.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d']})

                r = config.get("r",32) #default value: 32

                info = get_net_info(subnet, (3, r, r), print_info=False)
                info['tiny_ml'] = tiny_ml(params = info['params'],
                                 macs = info['macs'],
                                 activations = info['activations'],
                                 pmax = self.pmax,
                                 mmax = self.mmax,
                                 amax = self.amax,
                                 wp = self.wp,
                                 wm = self.wm,
                                 wa = self.wa,
                                 penalty = self.penalty)

                f[i, 0] = acc_err
                f[i, 1] = info.get(self.sec_obj,None) 

        out["F"] = f


    def _isvalid(self,x):
      is_valid = True
      branches = x[-self.ss.num_branches:]
      if any(el>1 for el in branches) or all(el==0 for el in branches):
          #1st check: elements stay in the range
          #2nd check: no zero EECs
          is_valid = False
      return is_valid

class SubsetProblem(Problem):
    """ select a subset to diversify the pareto front """
    def __init__(self, candidates, archive, K):
        super().__init__(n_var=len(candidates), n_obj=1,
                         n_constr=1, xl=0, xu=1)#, vtype=np.bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # append selected candidates to archive then sort
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])))
            f[i, 0] = np.std(np.diff(tmp))
            # we penalize if the number of selected candidates is not exactly K
            g[i, 0] = (self.n_max - np.sum(_x)) ** 2

        out["F"] = f
        out["G"] = g


def main(args):
    engine = CNAS(vars(args))
    engine.search()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp',
                        help='location of dir to save')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume search from a checkpoint')
    parser.add_argument('--first_obj', type=str, default='top1',
                        help='first objective to optimize simultaneously')
    parser.add_argument('--sec_obj', type=str, default=None,
                        help='second objective to optimize simultaneously')
    parser.add_argument('--iterations', type=int, default=30,
                        help='number of search iterations')
    parser.add_argument('--n_doe', type=int, default=100,
                        help='initial sample size for DOE')
    parser.add_argument('--n_iter', type=int, default=8,
                        help='number of architectures to high-fidelity eval (low level) in each iteration')
    parser.add_argument('--first_predictor', type=str, default='rbf',
                        help='which first obj predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--sec_predictor', type=str, default=None,
                        help='which sec obj predictor model to fit (rbf/gp/cart/mlp/as)')
    parser.add_argument('--n_gpus', type=int, default=8,
                        help='total number of available gpus')
    parser.add_argument('--gpu', type=int, default=1,
                        help='number of gpus per evaluation job')
    parser.add_argument('--gpu_list', metavar='N', type=int, nargs='+', default = None,
                        help='a list of integers representing the ids of the gpus to be used for evaluation')
    parser.add_argument('--data', type=str, default='/mnt/datastore/ILSVRC2012',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--model', type=str, default='mobilenetv3',
                        help='name of the model (mobilenetv3, ...)')
    parser.add_argument('--n_classes', type=int, default=1000,
                        help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--search_space', type=str, default='mobilenetv3',
                        help='type of search space')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained weights')                    
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers for dataloader per evaluation job')
    parser.add_argument('--val_split', type=float, default=0.0, help='percentage of train set for validation')
    parser.add_argument('--trn_batch_size', type=int, default=128,
                        help='train batch size for training')
    parser.add_argument('--vld_batch_size', type=int, default=200,
                        help='test batch size for inference')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='number of epochs for CNN training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluation performance on testing set')
    parser.add_argument('--lr', type = int , default=224,
                        help='minimum resolution')
    parser.add_argument('--ur', type = int, default=224,
                        help='maximum resolution')
    parser.add_argument('--rstep', type = int, default=4,
                        help='resolution step')
    parser.add_argument('--seed', type = int, default=0,
                        help='random seed')
    parser.add_argument('--trainer_type', type = str, default='single_exit',
                        help='trainer type (single_exit, multi_exits)')
    parser.add_argument('--func_constr', action='store_true', default=False,
                        help='use functional constraints')
    parser.add_argument('--pmax', type = float, default=2.0,
                        help='max value of params for candidate architecture')
    parser.add_argument('--mmax', type = float, default=100,
                        help='max value of macs for candidate architecture')
    parser.add_argument('--amax', type = float, default=5.0,
                        help='max value of activations for candidate architecture')
    parser.add_argument('--top1min', type = float, default=0.1, help='top1 constraint')
    parser.add_argument('--wp', type = float, default=1.0,
                        help='weight for params')
    parser.add_argument('--wm', type = float, default=1/40,
                        help='weight for flops')
    parser.add_argument('--wa', type = float, default=1.0,
                        help='weight for activations')
    parser.add_argument('--penalty', type = float, default=10**10,
                        help='penalty factor')
    parser.add_argument('--optim', type = str, default="SGD",
                        help='optimization algorithm')
    parser.add_argument('--sigma_min', type = float, default=0.05, help='min noise perturbation intensity')
    parser.add_argument('--sigma_max', type = float, default=0.05, help='max noise perturbation intensity')
    parser.add_argument('--sigma_step', type = float, default=0, help='noise perturbation intensity step')
    parser.add_argument('--alpha', type = float, default=0.5, help='alpha parameter for entropic figure')
    parser.add_argument('--res', type = int, default=32, help='fixed resolution for entropic training')
    parser.add_argument('--w_alpha', type = float, default=1.0, help='weight for alpha factor')
    parser.add_argument('--method', type = str, default='bernulli', help='method for early exit training')
    parser.add_argument('--support_set', action='store_true', default=False, help='use support set for early exit training')
    parser.add_argument('--tune_epsilon', action='store_true', default=False, help='tune epsilon for early exit inference')
    parser.add_argument('--w_beta', type = float, default=1.0, help='weight for beta factor')
    parser.add_argument('--w_gamma', type = float, default=1.0, help='weight for gamma factor')
    parser.add_argument('--warmup_ee_epochs', type = int, default=5, help='warmup epochs for early exit')
    parser.add_argument('--ee_epochs', type = int, default=0, help='early exit epochs with support set')
    cfgs = parser.parse_args()
    main(cfgs)

