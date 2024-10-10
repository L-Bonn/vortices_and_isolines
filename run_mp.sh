#!/bin/bash
#SBATCH --job-name=an_vort
#SBATCH --partition=astro3_devel
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


/groups/astro/rsx187/anaconda3/bin/python mp.py 
