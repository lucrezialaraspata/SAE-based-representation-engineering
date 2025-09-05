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

LOAD_HIDDENS_NAME="grouped_activations"
MI_SAVE_NAME="mutual_information"
MODEL_PATH="meta-llama/Meta-Llama-3-8B"

srun -u python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=12 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME} \
  --use_local


srun -u python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=13 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME} \
  --use_local


srun -u python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=14 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME} \
  --use_local



srun -u python -m spare.mutual_information_and_expectation \
  --num_proc=64 \
  --data_name="nqswap" \
  --model_path=${MODEL_PATH} \
  --load_hiddens_name=${LOAD_HIDDENS_NAME} \
  --layer_idx=15 \
  --minmax_normalisation \
  --mutual_information_save_name=${MI_SAVE_NAME} \
  --use_local

