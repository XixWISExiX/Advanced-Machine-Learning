#!/bin/bash

# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments!)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu
#

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=1G

# The %j is translated into the job number
#SBATCH --output=results/hw0_%j_stdout.txt
#SBATCH --error=results/hw0_%j_stderr.txt

#SBATCH --time=00:30:00
#SBATCH --job-name=hw0_test
#SBATCH --mail-user=Joshua.T.Wiseman-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504318/xor
#
#Example
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Change this line to start an instance of your experiment
python hw0.py --exp 100 --epochs 10000 --hidden 1024 --lrate 0.001 --hidden_activation relu --output_activation sigmoid --project HW0 --label hw0_relu_1024 --render -vv
