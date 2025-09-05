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
#SBATCH --job-name=mi_expectation
#SBATCH --out=output.log
#SBATCH --err=error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.laraspata3@phd.uniba.it

source .venv/bin/activate

srun -u python -m spare.run_spare \
  --model_path="meta-llama/Meta-Llama-3-8B" \
  --data_name="nqswap" \
  --layer_ids 12 13 14 15 \
  --edit_degree=2.0 \
  --select_topk_proportion=0.07 \
  --seed=42 \
  --hiddens_name="grouped_activations" \
  --mutual_information_save_name="mutual_information" \
  --run_use_parameter \
  --run_use_context \
  --use_local