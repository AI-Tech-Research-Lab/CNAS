#!/bin/bash
# SLURM script to be runned through `sbatch job.sh`
# In the following slurm options, customize (if needed) only the ones with comments

#SBATCH --job-name="EDANAS"            #the job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # number of threads
#SBATCH --time=72:00:00              # walltime limit
#SBATCH --gpus=1                     # num gpus. If set to 0 change the partition to defq or compute
#SBATCH --partition=gpu              # [gpu, defq, compute, debug, long]
#SBATCH --account=pittorino
#SBATCH --mail-type=NONE              #notify for NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=fabrizio.pittorino@unibocconi.it
#SBATCH --output=out/%x_%j.out       # where to write standard output. %j gives job id and %x gives job name
#SBATCH --error=err/%x_%j.err        # where to write standard error.
#### #SBATCH --mem-per-cpu=8000M     # memory per cpu core, default 8GB

# PARTITIONS
# If you have cpu job change partition to compute.
# defq, timelimit 3 days, Nodes=cnode0[1-4] (CPU)
# compute, timelimit 15 days, Nodes=cnode0[5-8] (CPU)
# gpu, timelimit 3 days, Nodes=gnode0[1-4] (GPU)
# debug, timelimit 30 minutes, Nodes=cnode01,gnode04 (short test on either CPU or GPU)
# QOS long: long jobs (7 days max) on gnode0[1-4] (GPU)

### export PATH="/home/Pittorino/miniconda3/bin:$PATH"
#export PATH="/home/Pittorino/miniconda3:$PATH"

module load cuda/12.3
#conda activate timefs

bash tests/search_edanas.sh
#bash scripts/train_neighbors.sh
