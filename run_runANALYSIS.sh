#!/bin/bash
#SBATCH --job-name=ANAL
#SBATCH --partition=astro2_short
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --account=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


echo "starting now"

srun /groups/astro/rsx187/anaconda3/bin/python -u runANALYSIS.py #> sleout/slurm-$SLURM_JOB_ID.out
#srun /groups/astro/rsx187/miniconda3/bin/python runANALYSIS.py > sleout/slurm-$SLURM_JOB_ID.out
