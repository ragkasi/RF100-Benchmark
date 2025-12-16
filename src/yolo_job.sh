#!/bin/bash
#SBATCH --job-name=rf100_yolo_benchmark
#SBATCH --account=PAS3162
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out

export ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY
source ~/.bashrc
conda activate cv_env
cd /fs/ess/PAS3162/Siddiqui_Kasibhatla_Haikal_ess/rf100-benchmark
python scripts/yolov8_benchmark.py
