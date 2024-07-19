#!/bin/bash
#SBATCH --job-name=pressure
#SBATCH --partition=astro2_short
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


srun /groups/astro/rsx187/anaconda3/bin/python pressureiso_maker_simon.py >> binaryfieldout/slurm-$SLURM_JOB_ID.out
