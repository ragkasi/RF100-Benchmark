#!/bin/bash

#SBATCH --job-name=dino_stream

#SBATCH --account=PAS3162

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=05:00:00

#SBATCH --mem=32GB

#SBATCH --partition=gpu

#SBATCH --output=/fs/scratch/PAS3162/ragkasi/dino_stream_%j.out



# 1. Setup Environment

module load cuda/11.8.0

source ~/.bashrc

conda activate dino_bench



# 2. Key Environment Variables

export CUDA_HOME=$CUDA_HOME

export PYTHONPATH=$PYTHONPATH:/fs/scratch/PAS3162/ragkasi/GroundingDINO

# Ensure API Key is available (if not set in .bashrc, uncomment and add it here)

# export ROBOFLOW_API_KEY=rf_... 



# 3. Change to Project Root (Just to be safe)

cd /fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark



# 4. Run the Streaming Script using the ABSOLUTE PATH

echo "Starting Streaming Benchmark..."

python /fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark/scripts/groundingdino_streaming.py

echo "Done."
