#!/bin/bash
#
##SBATCH --partition=gpu_a100
##SBATCH --partition=gpu,gpu_a100,disc_dual_a100_students,gpu_kepler,el9_gpu_test,sooner_gpu_test,sooner_gpu_test_ada,sooner_gpu_test_dual_ada,sooner_gpu_test_quad_ada
##SBATCH --partition=debug_gpu
##SBATCH --partition=sooner_gpu_test_ada
##SBATCH --partition=gpu
#SBATCH --partition=gpu,gpu_a100,sooner_gpu_test
#SBATCH --gres=gpu:0 # use no GPUs
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
##SBATCH --mem=48G
#SBATCH --output=results/SimpleRNN_%j_stdout.txt
#SBATCH --error=results/SimpleRNN_%j_stderr.txt
#SBATCH --time=12:00:00
##SBATCH --time=00:30:00
##SBATCH --time=00:01:00
#SBATCH --job-name=SimpleRNN
##SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs504318/hw6
##SBATCH --array=0-4    # the double ## means that this line is ignored

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh

conda activate dnn

## SimpleRNN
## Does support L2_regularization=1e-2, 320 epochs
python base.py -v @exp.txt @oscer.txt @simpleRNN.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --epochs 500 -vvv  --cache "" --save_model --render --force
