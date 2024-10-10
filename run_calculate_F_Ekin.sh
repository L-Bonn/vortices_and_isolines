#!/bin/bash
#SBATCH --job-name=FEkin
#SBATCH --partition=astro3_short
#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


/groups/astro/rsx187/anaconda3/bin/python calculate_F_Ekin.py
