#!/bin/bash
#
##SBATCH --partition=gpu_a100
##SBATCH --partition=gpu, gpu_a100, disc_dual_a100_students, gpu_kepler, el9_gpu_test, sooner_gpu_test, sooner_gpu_test_ada, sooner_gpu_test_dual_ada, sooner_gpu_test_quad_ada
##SBATCH --partition=debug_gpu
##SBATCH --partition=sooner_gpu_test_ada
#SBATCH --partition=gpu,gpu_a100,sooner_gpu_test
##SBATCH --partition=gpu,gpu_a100
##SBATCH --partition=sooner_gpu_test
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
##SBATCH --mem=64G
#SBATCH --output=results/hw7_%j_stdout.txt
#SBATCH --error=results/hw7_%j_stderr.txt
##SBATCH --time=12:00:00
#SBATCH --time=4:00:00
##SBATCH --time=1:00:00
##SBATCH --time=00:25:00
##SBATCH --time=00:10:00
#SBATCH --job-name=hw7
##SBATCH --mail-user=ADD YOUR OWN EMAIL
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/cs504318/hw7
##SBATCH --array=0-4    # the double ## means that this line is ignored

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh


conda activate dnn

## DEEP
python base.py -v @exp.txt @oscer.txt @diffusion.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --epochs 150 -vvv  --cache "" --save_model --render --force
