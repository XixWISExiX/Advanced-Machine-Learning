#!/bin/bash
#
##SBATCH --partition=gpu
##SBATCH --partition=normal
#SBATCH --partition=normal,sooner_test
##SBATCH --partition=sooner_test
#SBATCH --gres=gpu:0
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
##SBATCH --mem=45G
#SBATCH --output=results/hw5_pnn_%j_stdout.txt
#SBATCH --error=results/hw5_pnn_%j_stderr.txt
#SBATCH --time=10:00:00
##SBATCH --time=01:30:00
##SBATCH --time=00:10:00
#SBATCH --job-name=PNN-1
##SBATCH --mail-user=ADD YOUR OWN EMAIL
##SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504318/hw5
##SBATCH --array=0-5    # the double ## means that this line is ignored

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh


conda activate dnn

## PNN sbatch (max distance was 540
## in 5.5 hours, 1200 epochs)
python base.py -v @exp.txt @oscer.txt @pnn.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --epochs 700 -vvv  --cache "" --save_model --render --force
