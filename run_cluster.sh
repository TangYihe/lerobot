#!/bin/bash
#SBATCH --job-name=lerobot_train
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --account viscam
#SBATCH --partition=viscam
#SBATCH --nodelist=viscam12
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yihetang@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Activate conda environment
source activate lerobot

# Set environment variables
export HUGGINGFACE_HUB_CACHE="/svl/u/yihetang/lerobot/.hf_cache/hub"
export HF_DATASETS_CACHE="/svl/u/yihetang/lerobot/.hf_cache/datasets"
export LEROBOT_HOME="/svl/u/yihetang/lerobot/.hf_cache/lerobot"

echo "Starting"
gpustat

# Run the training script
# python lerobot/scripts/train.py \
#     --policy.path=lerobot/pi0 \
#     --dataset.repo_id=Yiheyihe/galaxea-r1-shelf-debug-normalized

# python lerobot/scripts/train.py \
#     --policy.path=lerobot/pi0 \
#     --dataset.repo_id=Yiheyihe/galaxea-r1-shelf-10ep-normalized

python lerobot/scripts/train.py \
    --policy.path=lerobot/pi0 \
    --dataset.repo_id=Yiheyihe/galaxea-r1-shelf-full-normalized