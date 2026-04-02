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
#SBATCH --output=results/hw2_%j_stdout.txt
#SBATCH --error=results/hw2_%j_stderr.txt

#SBATCH --time=00:40:00
#SBATCH --job-name=hw2_sbatch
#SBATCH --mail-user=Joshua.T.Wiseman-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504318/hw2
# Dont execute SBATCH --cpus-per-task=40
#
#Examples
#
#Naked submission
# python hw0.py --exp 100 --epochs 10000 --hidden 1024 --lrate 0.001 --hidden_activation relu --output_activation sigmoid --project HW0 --label hw0_relu_1024 --render -vv --generate_loss_metric_plots
# 
#Standalone Program Commands 
# --generate_loss_metric_plots
# --meta
# --check
# 
#Bonus command --cpus_per_task $SLURM_CPUS_PER_TASK
#
#Organized submission
# python hw2.py @./exp_instructions/problem_def.txt @./exp_instructions/network_details.txt @./exp_instructions/exp_details.txt @./exp_instructions/meta_exp_details.txt @./exp_instructions/results.txt @./exp_instructions/execution_control.txt --exp_index $SLURM_ARRAY_TASK_ID --meta
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate dnn

# Change this line to start an instance of your experiment

# Need a new architecture
#python hw2.py @./exp_instructions/problem_def.txt @./exp_instructions/network_details.txt @./exp_instructions/exp_details.txt @./exp_instructions/meta_exp_details.txt @./exp_instructions/results.txt @./exp_instructions/execution_control.txt --exp_index $SLURM_ARRAY_TASK_ID
python hw2_chart_builder.py
