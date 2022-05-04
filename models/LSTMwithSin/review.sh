#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --array=0                              # Number of tasks (see below)
#SBATCH --nodes=1                                   # Ensure that all cores are on one machine
#SBeTCH --cpus-per-task=1
#SBATCH --partition=gpu-2080ti                     # Partition to submit to
#SBATCH --time=0-00:30           					# Runtime in D-HH:MM
#SBATCH --mem=3G                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=svolz.out                  # File to which STDOUT will be written
#SBATCH --error=svolz.err                   # File to which STDERR will be written
#SBATCH --mail-type FAIL           	                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastien.volz@student.uni-tuebingen.de	# Email to which notifications will be sent
#SBATCH --gres=gpu:1

set -o errexit 

pwd

echo "JOB DATA"
scontrol show job=$SLURM_JOB_ID

echo "node:"
hostname

echo "RUN Script"

<<<<<<< HEAD
python ./LSTMTwoLayer.py --name_model "simple" --epochs 200 --hidden_size 51
=======
python ./LSTMTwoLayerWorking.py
>>>>>>> 506bb88be625e9601e433113580e04aaef808d69

