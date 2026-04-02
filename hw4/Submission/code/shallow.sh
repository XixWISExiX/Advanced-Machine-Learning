#!/bin/bash
#
##SBATCH --partition=gpu_a100
##SBATCH --partition=gpu, gpu_a100, disc_dual_a100_students, gpu_kepler, el9_gpu_test, sooner_gpu_test, sooner_gpu_test_ada, sooner_gpu_test_dual_ada, sooner_gpu_test_quad_ada
##SBATCH --partition=debug_gpu
##SBATCH --partition=sooner_gpu_test_ada
#SBATCH --partition=sooner_gpu_test
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
##SBATCH --mem=45G
#SBATCH --output=results/hw4_shallow_%j_stdout.txt
#SBATCH --error=results/hw4_shallow_%j_stderr.txt
##SBATCH --time=12:00:00
##SBATCH --time=00:20:00
#SBATCH --time=00:01:00
#SBATCH --job-name=hw4_shallow
##SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs504318/hw4
##SBATCH --array=0-4    # the double ## means that this line is ignored

################################################# # Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh

conda activate dnn

## SHALLOW
python base.py -v @exp.txt @oscer.txt @net_shallow.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --epochs 100 -vvv  --cache "" --save_model --render --force
