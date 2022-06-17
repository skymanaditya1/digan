#!/bin/bash 

#SBATCH --job-name=rdigan
#SBATCH --mem-per-cpu=2048
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=38
#SBATCH --nodes=1
#SBATCH --time 10-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode056

source /home2/aditya1/miniconda3/bin/activate torch
cd /ssd_scratch/cvit/aditya1/digan
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 python src/infra/launch.py hydra.run.dir=. +experiment_name=rainbow_jelly +dataset.name=rainbow_processed_128