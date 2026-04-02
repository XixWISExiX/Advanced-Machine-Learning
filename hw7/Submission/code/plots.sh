#!/bin/bash
#
##SBATCH --partition=gpu_a100
##SBATCH --partition=normal
##SBATCH --partition=gpu
#SBATCH --partition=sooner_gpu_test,gpu,gpu_a100
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
##SBATCH --mem=45G
#SBATCH --ntasks=1
#SBATCH --output=plot_results/hw7_plots_%j_stdout.txt
#SBATCH --error=plot_results/hw7_plots_%j_stderr.txt
##SBATCH --time=04:00:00
#SBATCH --time=01:00:00
##SBATCH --time=00:30:00
#SBATCH --job-name=plot_generator
##SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs504318/hw7

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh

conda activate dnn

python plot_generator.py
#python plot_generator.py @exp.txt @oscer.txt
