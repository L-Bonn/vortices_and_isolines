#!/bin/bash
#SBATCH --job-name=an_vort
#SBATCH --partition=astro3_short
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


srun /groups/astro/rsx187/anaconda3/bin/python make_power_spectrum.py 