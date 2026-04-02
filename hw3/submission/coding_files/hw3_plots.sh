#!/bin/bash
#
##SBATCH --partition=gpu_a100
#SBATCH --partition=normal
#SBATCH --mem=30G
##SBATCH --mem=45G
#SBATCH --ntasks=1
#SBATCH --output=results/hw3_shallow_%j_stdout.txt
#SBATCH --error=results/hw3_shallow_%j_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=hw3_plot_generator
##SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504318/hw3
##SBATCH --array=0-4    # the double ## means that this line is ignored

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh


conda activate dnn

## SHALLOW
python plot_generator.py
