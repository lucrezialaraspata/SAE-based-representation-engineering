#!/bin/bash

#SBATCH -A IscrC_ARTURO
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=24:00:00
#SBATCH -N 1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=60000
#SBATCH --job-name=group_prompts
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

source .venv/bin/activate

MODEL_PATH="meta-llama/Meta-Llama-3-8B"
SAVE_DIR_NAME="grouped_prompts"  # save to PROJ_DIR / "cache_data" / model_name / SAVE_DIR_NAME

srun -u python -m spare.group_prompts \
  --model_path=${MODEL_PATH} \
  --save_dir_name=${SAVE_DIR_NAME} \
  --k_shot=3 \
  --seeds_to_encode 42 43 44 45 46 \
  --use_local