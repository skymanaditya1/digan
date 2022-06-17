#!/bin/bash 

#SBATCH --job-name=h2sdigan
#SBATCH --mem-per-cpu=2048
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=38
#SBATCH --nodes=1
#SBATCH --time 10-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode045

source /home2/aditya1/miniconda3/bin/activate base

cd /ssd_scratch/cvit/aditya1/digan

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 \
python src/infra/launch.py \
hydra.run.dir=. +experiment_name=how2sign_resume \
+dataset.name=how2sign_faces_styleganv_resized \
+resume=/ssd_scratch/cvit/aditya1/digan/experiments/how2sign_faces-2709d68/00000-how2sign_faces_styleganv_resized-mirror-auto2-r1_gamma1-color-translation/network-snapshot-002200.pkl