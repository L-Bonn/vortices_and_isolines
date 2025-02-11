#!/bin/bash
#SBATCH --job-name=ANAL
#SBATCH --partition=astro3_short
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#module load astro python/anaconda3/2021.05
#module load ffmpeg/4.3.1

#srun python3 runANALYSIS.py >> sleout/slurm-$SLURM_JOB_ID.out

srun /groups/astro/rsx187/anaconda3/bin/python runANALYSIS.py >> sleout/slurm-$SLURM_JOB_ID.out
#srun /groups/astro/rsx187/miniconda3/bin/python runANALYSIS.py >> sleout/slurm-$SLURM_JOB_ID.out
