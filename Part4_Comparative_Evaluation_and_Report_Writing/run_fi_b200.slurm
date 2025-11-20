#!/bin/bash
#SBATCH -A liguanpeng
#SBATCH --qos=liguanpeng
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abdullahnaveed@ufl.edu
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=6:00:00
#SBATCH -J fi_llm_turin_fast
#SBATCH --chdir=/blue/liguanpeng/abdullahnaveed/pattern_project/part4
#SBATCH --output=/blue/liguanpeng/abdullahnaveed/pattern_project/part4/logs/%x_%j.out
#SBATCH --error=/blue/liguanpeng/abdullahnaveed/pattern_project/part4/logs/%x_%j.err

set -euo pipefail
set -x

# --- Conda env ---
module purge
module load conda
conda activate llm_ft_new

# --- Thread env ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- HF cache setup ---
export HF_HOME=/blue/liguanpeng/abdullahnaveed/hf_cache_llm_ft_new
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

# --- CUDA behavior ---
export CUDA_MODULE_LOADING=LAZY

# --- PyTorch perf knobs ---
python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; \
torch.backends.cudnn.allow_tf32 = True; \
torch.backends.cudnn.benchmark = True; \
torch.set_float32_matmul_precision('high'); \
print('Perf knobs set.')"

# --- Sanity checks ---
nvidia-smi
python -V
which python

# --- Run job ---
python 4_caseStudy1_gatedGRU_resnet_CMA_SA_proj.py

echo "<======== DONE ========>"
