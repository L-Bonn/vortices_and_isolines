#!/bin/bash
#SBATCH --job-name=defcou
#SBATCH --partition=astro3_short
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


srun /groups/astro/rsx187/anaconda3/bin/python defect_counts_simon.py --slurmd-debug=3
