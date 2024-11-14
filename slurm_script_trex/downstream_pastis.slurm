#!/bin/bash
#SBATCH --job-name=pretrain_malice
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=16                     # number of tasks ( or --tasks=16)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --partition=gpu_a100        # partition
#SBATCH --qos=gpu_all               # QoS
#SBATCH --time=12:00:00             # Walltime 10mn
#SBATCH --mem-per-cpu=7G         # memory per core 
#SBATCH --account=cesbio       # MANDATORY : account (launch myaccounts to list your accounts) 
# #SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc
#SBATCH --signal=SIGHUP@90

module purge
module load conda
conda activate alise

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

export PYTHONWARNINGS="ignore"

cd /work/scratch/data/kalinie/src/MMDC/mmmv_sits_ssl
                    
python script/downstream_segmentation.py  >> output_$SLURM_JOBID.log
