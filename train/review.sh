#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --array=4-5                         # Number of tasks (see below)
#SBATCH --nodes=1                                   # Ensure that all cores are on one machine
#SBeTCH --cpus-per-task=1
#SBATCH --partition=gpu-2080ti                     # Partition to submit to
#SBATCH --time=0-72:00           					# Runtime in D-HH:MM
#SBATCH --mem=3G                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=svolz.out                  # File to which STDOUT will be written
#SBATCH --error=svolz.err                   # File to which STDERR will be written
#SBATCH --mail-type FAIL           	                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.volz@student.uni-tuebingen.de	# Email to which notifications will be sent
#SBATCH --gres=gpu:1

set -o errexit 

pwd

echo "JOB DATA"
scontrol show job=$SLURM_JOB_ID

echo "node:"
hostname

echo "RUN Script"

python ./trainAll.py --run_idx ${SLURM_ARRAY_TASK_ID} --model "baseline" --datasetTrain "wave-10-1-3-290" \
                   --datasetVal "wave-10-1-3-290" --mode "all-40-adapted" --context 20 --horizon 40 --learningRate 0.001 \
                   --epochs 400 --batchSize 32 --clip 1 --start 100
