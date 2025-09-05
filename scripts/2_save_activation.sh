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
#SBATCH --job-name=save_activations
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

source .venv/bin/activate

srun -u python -m spare.save_grouped_activations \
  --data_name="nqswap" \
  --model_path="meta-llama/Meta-Llama-3-8B" \
  --load_data_name="grouped_prompts"\
  --shots_to_encode 3 \
  --seeds_to_encode 42 43 44 45 46 \
  --save_hiddens_name="grouped_activations" \
  --use_local
