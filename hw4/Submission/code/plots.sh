#!/bin/bash
#
##SBATCH --partition=gpu_a100
##SBATCH --partition=normal
##SBATCH --partition=gpu
#SBATCH --partition=sooner_gpu_test
#SBATCH --mem=30G
##SBATCH --mem=45G
#SBATCH --ntasks=1
#SBATCH --output=plot_results/hw4_plots_%j_stdout.txt
#SBATCH --error=plot_results/hw4_plots_%j_stderr.txt
#SBATCH --time=01:00:00
##SBATCH --time=00:30:00
#SBATCH --job-name=plot_generator
##SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs504318/hw4

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh

conda activate dnn

python plot_generator.py @exp.txt @oscer.txt
