#!/bin/bash
#SBATCH --job-name=ANAL
#SBATCH --partition=astro2_long
#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


srun /groups/astro/rsx187/anaconda3/bin/python runANALYSIS.py >> sleout/slurm-$SLURM_JOB_ID.out
