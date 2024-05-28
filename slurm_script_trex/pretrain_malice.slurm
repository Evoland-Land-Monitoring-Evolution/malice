#!/bin/bash
#SBATCH --job-name=pretrain_malice     # job's name
# --output = name of the output file  --error= name of error file (%j = jobID )
#SBATCH --output=pretrain_malice-%j.out
#SBATCH --error=pretrain_malice-%j.err
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
conda activate /work/scratch/env/dumeuri/mt_ubarn_copy

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

export PYTHONWARNINGS="ignore"

cd /home/ad/dumeuri/code/mmmv_sits_ssl/script
python pretrain.py callbacks=mae_clb seed=3 train.w_inv=0 dataset=easy_mask_mm.yaml
