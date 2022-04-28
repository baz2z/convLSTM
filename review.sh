#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --array=0                              # Number of tasks (see below)
#SBATCH --nodes=1                                   # Ensure that all cores are on one machine
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-2080ti                     # Partition to submit to
#SBATCH --time=0-00:30           					# Runtime in D-HH:MM
#SBATCH --mem=3G                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=jthuemmel54_%j.out                  # File to which STDOUT will be written
#SBATCH --error=jthuemmel54_%j.err                   # File to which STDERR will be written
#SBATCH --mail-type FAIL           	                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=jannik.thuemmel@uni-tuebingen.de	# Email to which notifications will be sent
#SBATCH --gres=gpu:1

set -o errexit 

pwd

echo "JOB DATA"
scontrol show job=$SLURM_JOB_ID

echo "node:"
hostname

echo "RUN Script"

python run_weatherbench.py --epochs 1 --batch_size 32 --data_name 'Z500' --run_idx 42 --model_name 'teacher' --model_idx ${SLURM_ARRAY_TASK_ID} 


