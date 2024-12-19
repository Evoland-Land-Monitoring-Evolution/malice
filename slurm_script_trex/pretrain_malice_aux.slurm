#!/bin/bash
#SBATCH --job-name=pretrain_malice_aux     # job's name
#SBATCH --output=pretrain_malice-%j.out
#SBATCH --error=pretrain_malice-%j.err
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=16        # number of tasks ( or --tasks=16)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --partition=gpu_a100        # partition
#SBATCH --qos=gpu_all               # QoS
#SBATCH --time=19:00:00             # Walltime 10mn
#SBATCH --mem-per-cpu=7G         # memory per core 
#SBATCH --account=cesbio       # MANDATORY : account (launch myaccounts to list your accounts) 
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc

export SCRATCH=/work/scratch/data/${USER}
export SRCDIR=${SCRATCH}/src/MMDC/mmmv_sits_ssl

export WORKING_DIR=${SCRATCH}/MMDC/jobs

cd ${WORKING_DIR}

module purge
module load conda
conda activate alise

export CUDA_LAUNCH_BLOCKING=1

export PYTHONWARNINGS="ignore"

HYDRA_FULL_ERROR=1 python -O ${SRCDIR}/script/pretrain.py --config-name=pretrain_proj_aux.yaml
