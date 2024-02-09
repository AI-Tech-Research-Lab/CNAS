
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.factory import get_performance_indicator


## GENERAL

def get_archive(exp_path, first_obj, sec_obj=None):
        """ resume search from a previous iteration """
        import glob
        archive = []
        split = exp_path.rsplit("_",1)
        maxiter = int(split[1])
        path = split[0]

        for file in glob.glob(os.path.join(path + '_*', "net_*/net_*.subnet")):
            arch = json.load(open(file))
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

                    # dump the statistics
                    with open(path, "w") as handle:
                        json.dump(stats, handle)
                    
                    if sec_obj is not None:
                        v = (arch, stats[first_obj], stats.get(sec_obj, None)) # 100 - first obj for old compatibility
                    else:
                        v = (arch, stats[first_obj])
                    archive.append(v)

                else: #failed net
                    print("FAILED NET with path: ", path)   
        
        print("LEN ARCHIVE")    
        print(len(archive))
    
        return archive



def get_pareto_fronts(stat_path, obj, n_exits=None):

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    pre,_= os.path.splitext(stat_path)
    split = pre.rsplit("_",1)
    maxiter = int(split[1])
    PFs=[]
    for i in range(0,maxiter+1):
        file = os.path.join(split[0]+"_"+str(i)+".stats")
        split2,_ = os.path.splitext(file)
        niter = int(split2.rsplit("_",1)[1])  
        
        if(niter<=maxiter):
            
            if(os.path.exists(file)):
                ##compute the pareto front 
                archive = json.load(open(file))['archive']
            else:
                print("Missing file.. " + file)
                archive = get_archive(split2, 'top1', obj)

            if n_exits is not None:
                # filter according to n° of exits
                archive_temp = []
                for v in archive:
                    subnet = v[0]
                    #t = subnet["ne"]
                    b_config = subnet["b"]
                    count_exits = len([element for element in b_config if element != 0])
                    if(count_exits==n_exits):
                        archive_temp.append(v)
                print("#EEcs:")
                print(n_exits)
                print("lunghezza archivio prima")        
                print(len(archive))
                archive = archive_temp

            for v in archive: #remove failed nets
                err_top1 = v[1]
                if(err_top1==100):
                  archive.remove(v)
            
            subnets, top1, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]

            sort_idx = np.argsort(top1)
            F = np.column_stack((top1, sec_obj))[sort_idx, :]
            front = NonDominatedSorting().do(F, only_non_dominated_front=True)
            pf = F[front, :]

            # update the array of the pareto fronts
            PFs.append(pf)
        
    return PFs

def correlation_surrogate_plot(exp_path):

    taus = []
    prefix = exp_path.rsplit("/",1)[0]
    maxiter = int(exp_path.rsplit("_",1)[1])    

    for it in range(1,maxiter+1):
            stats_path = os.path.join(prefix, 'iter_'+ str(it) + ".stats")
            if (os.path.exists(stats_path)):
                archive = json.load(open(stats_path))
                info_surrogate = archive['first_surrogate'] #surrogate_acc for olf compatibility
                taus.append(info_surrogate['tau'])
                #print("TAU: "+str(info_surrogate['tau']))
            else:
                print("FAILED iter: ", it)
                taus.append(0)

    plt.close()
    # Plot the variance values over iterations
    plt.plot(range(maxiter), taus, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel("Kendall's Tau")
    plt.title("Kendall's Tau of surrogate over Iterations")
    plt.show()
    save=exp_path.rsplit('/',1)[0]
    label='tau_surrogate'
    plt.savefig(os.path.join(save,label))

def calc_hv(ref_pt, F, normalized=True):
        # calculate hypervolume on the non-dominated set of F
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = get_performance_indicator("hv", ref_point=ref_point).calc(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv

def calc_hv_it(exp_path, first_obj, sec_obj):
     
     #exp_path: exp_folder/iter_X
     prefix=exp_path.rsplit("_",1)[0]
     #archive at iteration 0
     archive = get_archive(prefix + '_0', first_obj, sec_obj)
     # reference point (nadir point) for calculating hypervolume
     ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])
     archive = get_archive(exp_path, first_obj, sec_obj)
     hv = calc_hv(ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))
     return hv

def hv_plot_by_stats(exp_path):

    hvs = []
    prefix = exp_path.rsplit("/",1)[0]
    maxiter = int(exp_path.rsplit("_",1)[1])    

    for it in range(1,maxiter+1):
            stats_path = os.path.join(prefix, 'iter_'+ str(it) + ".stats")
            if (os.path.exists(stats_path)):
                archive = json.load(open(stats_path))
                hvs.append(archive['hv']) 
            else:
                print("FAILED iter: ", it)

                #hvs.append(0)

    plt.close()
    # Plot the variance values over iterations
    plt.plot(range(maxiter), hvs, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume of the NAS over Iterations")
    plt.show()
    save=exp_path.rsplit('/',1)[0]
    label='hypervolume'
    plt.savefig(os.path.join(save,label))

def hv_plot_by_calchv(exp_path, first_obj, sec_obj):

    hvs = []
    prefix = exp_path.rsplit("/",1)[0]
    maxiter = int(exp_path.rsplit("_",1)[1])    

    for it in range(1,maxiter+1):
            it_path = os.path.join(prefix, 'iter_'+ str(it))
            hvs.append(calc_hv_it(it_path, first_obj, sec_obj))

    plt.close()
    # Plot the variance values over iterations
    plt.plot(range(maxiter), hvs, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume of the NAS over Iterations")
    plt.show()
    save=exp_path.rsplit('/',1)[0]
    label='hypervolume'
    plt.savefig(os.path.join(save,label))


def plot_histograms(data_array, bins=36, path=''):
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
        sns.histplot(data, bins=36, color='darkblue', edgecolor='black', kde=True, line_kws={'linewidth': 4}, ax=axs[i])
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis values

    # Add common X-axis label
    axs[-1].set_xlabel('Value', fontsize=FONT_SIZE)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(os.path.join(path, 'histograms.pdf'), dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_histogramsV2(data_array, bins=10, path='', xlabels=None):
    plt.close()
    sns.set(style="whitegrid", rc={"grid.linestyle": "--", "grid.linewidth": 0.5, "axes.grid.axis": "both"})
    FONT_SIZE = 8
    FIGSIZE = (4.0 * len(data_array), 2.5)  # Adjust the multiplier as needed
    COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]

    num_plots = len(data_array)

    # Set up subplots
    fig, axs = plt.subplots(1, num_plots, figsize=FIGSIZE, sharey=True)

    xmin = 0
    xmax = 1.0
    step = xmax/bins
    
    custom_bins = np.arange(xmin, xmax + step, step).tolist()

    # Plot histograms for each element in the array
    for i, data in enumerate(data_array):
        sns.histplot(data, bins=custom_bins, color=COLORS[0], edgecolor='black', kde=True, ax=axs[i])
        axs[i].set_title(f'Branch #{i + 1}', fontsize=FONT_SIZE)
        axs[i].set_ylabel('')  # Remove y-axis label
        axs[i].set_xlabel('Early-exit confidence', fontsize=FONT_SIZE)
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Set x-axis limits and ticks
        x_ticks = np.arange(0, xmax + xstep, xstep)
        axs[i].set_xticks(x_ticks)
        axs[i].set_xlim(0, min(xmax, max(x_ticks)))  # Limit x-axis

        # Set y-axis limits and ticks
        y_ticks = np.arange(0, ymax + ystep, ystep)
        axs[i].set_yticks(y_ticks)
        axs[i].set_ylim(0, min(ymax, max(y_ticks)))  # Limit y-axis

    # Add common Y-axis label
    axs[0].set_ylabel('# Samples', fontsize=FONT_SIZE)
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(os.path.join(path, 'histograms.pdf'), dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_heatmaps(data_array, xticklabels, yticklabels, path=''):
     # Create a figure and subplots
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create a heatmap for Counter b[1]
    sns.heatmap(data_array, cmap="YlGnBu", annot=True, fmt="",
                xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
    ax.set_xlabel("Threshold Branch #2")
    ax.set_ylabel("Threshold Branch #1")
    ax.set_title("Branch #2 Utilization")

    # Adjust layout
    plt.tight_layout()

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1)
    
    fig.savefig(os.path.join(path,'heatmaps.pdf'), dpi=300)

    # Show the plot
    plt.show()


def torch_summary(model, input_size, depth=3, col_width=16, col_names=None, verbose=True):

    #ref: https://pypi.org/project/torch-summary/

    from torchsummary import summary
    """
    Customized model summary function using torchsummary package

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_size (tuple): Input size of the model (e.g., (3, 32, 32)).
        depth (int): Number of nested layers to traverse (e.g. Sequentials).
        col_width (int): Column width for summary.
        col_names (list): List of column names for summary.
        verbose (bool): Verbose flag to whether print summary or not.
    """
    if col_names is None:
        col_names = ["output_size"]  #Alternatives: ["kernel_size", "output_size", "num_params", "mult_adds"]

    summary_file = str(summary(model = model, 
                          input_data = input_size, 
                          depth = depth,
                          col_width = col_width,
                          col_names = col_names,
                          verbose = verbose))
    
    return summary_file


def PF_plot(stats_path, label, title='plot', obj=None, targets=None, macs_constr=None, acc_constr=None, n_exits=None):
    plt.close() # close previous plots
    PFs=get_pareto_fronts(stats_path, obj, n_exits)
    save=stats_path.rsplit('/',1)[0]
    if targets is None:
        targets=tuple(range(len(PFs)))
    for idx,pf in enumerate(PFs):
        if idx in targets:
            x = pf[:,0]
            y = pf[:,1]
            plt.plot(x, y, marker='o', linestyle='-', label=f'ITER {idx}')

    # Add horizontal line
    if macs_constr is not None:
        plt.axhline(y=macs_constr, color='r', linestyle='--', label=f'$\\overline{{F}}_{{M}}={macs_constr}M$')
    
    if acc_constr is not None:
        plt.axvline(x=acc_constr, color='black', linestyle='--', label=f'$1-\\overline{{F}}_{{A}}={acc_constr}\%$')
    
    plt.title(title)

    plt.xlabel('1-top1 (%)')
    plt.ylabel('MACs (M)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5, fancybox=True, shadow=True, prop={'size': 8})
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(save,label),dpi=300)
    plt.show()

# CBNAS

def PF_subplot(ax, stats_path, obj=None, targets=None, macs_constr=None, acc_constr=None, n_exits=None):
    PFs=get_pareto_fronts(stats_path, obj, n_exits)
    if targets is None:
        targets=tuple(range(len(PFs)))
    print(targets)
    for idx,pf in enumerate(PFs):
        if idx in targets:
            x = pf[:,0]
            print(x)
            y = pf[:,1]
            ax.plot(x, y, marker='o', linestyle='-', label=f'ITER {idx}')

    # Add horizontal line
    if macs_constr is not None:
        ax.axhline(y=macs_constr, color='r', linestyle='--', label=f'$\\overline{{F}}_{{M}}={macs_constr}M$')
    
    if acc_constr is not None:
        ax.axvline(x=acc_constr, color='black', linestyle='--', label=f'$1-\\overline{{F}}_{{A}}={acc_constr}\%$')
    
    if 'noconstraints' in stats_path: 
        ax.set_title('Pareto fronts of CBNAS without constraints')
    else: 
        ax.set_title('Pareto fronts of CBNAS with constraints')

    ax.set_xlabel('1-top1 (%)')
    ax.set_ylabel('MACs (M)')


def PF_subplots(stats_path1, stats_path2, label, obj=None, targets=None, macs_constr=None, acc_constr=None, n_exits=None):
    # Create subplots
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    #create_PF_subplot(ax1, stats_path1, None, targets, macs_constr, acc_constr, n_exits) # old compatibility with CBNAS cifar exp
    PF_subplot(ax1, stats_path1, obj, targets, macs_constr, acc_constr, n_exits)
    PF_subplot(ax2, stats_path2, obj, targets, macs_constr, acc_constr, n_exits)
    save=stats_path1.rsplit('/',1)[0]
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5, fancybox=True, shadow=True, prop={'size': 9})
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(save,label),dpi=300)
    plt.show()


#import seaborn as sns
import matplotlib.colors as mcolors

# Constants and Styling
FONT_SIZE = 8
FIGSIZE = (3.5, 3.0)
COLORS = [mcolors.TABLEAU_COLORS[k] for k in mcolors.TABLEAU_COLORS.keys()]
#sns.set_style('whitegrid')

def PF_subplotV2(ax, stats_path, obj=None, targets=None, macs_constr=None, acc_constr=None, n_exits=None):
    PFs = get_pareto_fronts(stats_path, obj, n_exits)
    if targets is None:
        targets = tuple(range(len(PFs)))

    for idx, pf in enumerate(PFs):
        if idx in targets:
            x = pf[:, 0]
            y = pf[:, 1]
            ax.plot(x, y, marker='o', linestyle='-', label=f'ITER {idx}', markersize=3, linewidth=1)

    if macs_constr is not None:
        ax.axhline(y=macs_constr, color='r', linestyle='--', label=f'$\\overline{{F}}_{{M}}={macs_constr}M$', linewidth=1)

    if acc_constr is not None:
        ax.axvline(x=acc_constr, color='black', linestyle='--', label=f'$1-\\overline{{F}}_{{A}}={acc_constr}\%$', linewidth=1)

    title = 'Pareto fronts of CBNAS without constraints' if 'noconstraints' in stats_path else 'Pareto fronts of CBNAS with constraints'
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel('1-top1 (%)', fontsize=FONT_SIZE)
    ax.set_ylabel('MACs (M)', fontsize=FONT_SIZE)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5, fancybox=True, shadow=True, prop={'size': FONT_SIZE - 1})
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

def PF_subplotsV2(stats_path1, stats_path2, label, obj=None, targets=None, macs_constr=None, acc_constr=None, n_exits=None):
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE)
    PF_subplot(ax1, stats_path1, obj, targets, macs_constr, acc_constr, n_exits)
    PF_subplot(ax2, stats_path2, obj, targets, macs_constr, acc_constr, n_exits)
    
    save = stats_path1.rsplit('/', 1)[0]
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(save, label)+'.pdf') #, dpi=300, bbox_inches='tight')
    plt.show()

def compute_variance_ece(exp_path):
        
        split = exp_path.rsplit("_",1)
        maxiter = int(split[1])
        path = split[0]

        ## some stats per iteration
        var_ece_it = [0]*(maxiter+1)

        for it in range(0,maxiter+1):
            if(it==0):
                n_subnets=100
            else:
                n_subnets=8
            avg_ece=0
            var_ece=0
            for nsubnet in range(0,n_subnets):
                    file = os.path.join(path + '_' + str(it), "net_"+str(nsubnet)+"/net_"+str(nsubnet)+".stats")
                    if (os.path.exists(file)):
                            stats = json.load(open(file))
                            ece_scores = stats['ece_scores']
                            avg_ece += sum(ece_scores.values())*100 / len(ece_scores) 
                            #print("AVG_ECE: "+str(avg_ece))

                    else: #failed net
                            print("FAILED NET")
                            print(file) 
            avg_ece=avg_ece/n_subnets
            for nsubnet in range(0,n_subnets):
                    file = os.path.join(path + '_' + str(it), "net_"+str(nsubnet)+"/net_"+str(nsubnet)+".stats")
                    if (os.path.exists(file)):
                            stats = json.load(open(file))
                            ece_scores = stats['ece_scores']
                            ece = sum(ece_scores.values())*100 / len(ece_scores)
                            var_ece += (ece-avg_ece)**2

                    else: #failed net
                            print("FAILED NET")
                            print(path)   

            var_ece_it[it]=var_ece/n_subnets #variance of the ece of the subnets of iteration it
            print("Variance of the it {}: {:.2f}".format(it,var_ece_it[it]))
    
        return var_ece_it

def varianceECE_plot(exp_path):
    var_ece_it=compute_variance_ece(exp_path)
    maxiter=len(var_ece_it)
    print(maxiter)
    # Plot the variance values over iterations
    plt.plot(range(maxiter), var_ece_it, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Variance of ECE (%)')
    plt.title('Variance of ECE over Iterations')
    plt.show()
    save=exp_path.rsplit('/',1)[0]
    label='variance_ece'
    plt.savefig(os.path.join(save,label),dpi=300)
