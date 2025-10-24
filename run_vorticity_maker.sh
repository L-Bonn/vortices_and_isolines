#!/bin/bash
#SBATCH --job-name=vorticity
#SBATCH --partition=astro2_short
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


srun /groups/astro/rsx187/anaconda3/bin/python vorticity_maker.py >> binaryfieldout/slurm-$SLURM_JOB_ID.out
