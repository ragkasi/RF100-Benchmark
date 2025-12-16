#!/bin/bash
#SBATCH --job-name=rex_rf100
#SBATCH --account=PAS3162
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark/logs/rex_rf100_%j.out
#SBATCH --error=/fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark/logs/rex_rf100_%j.err

source ~/.bashrc

conda activate rex_env

export TRANSFORMERS_USE_FLASH_ATTENTION_2=0
export USE_FLASH_ATTENTION_2=0

export ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY

cd /fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark

mkdir -p logs

python -u scripts/rex_benchmark.py
