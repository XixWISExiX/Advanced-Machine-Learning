# How to execute a single experiment

You must type
`sbatch --array=0 batch.sh`

and **DON'T** have arguments
`--check`
or
`--generate_loss_metric_plots`
or
`--meta`

## Things to note

batch.sh arguments are in the exp_instructions folder and any changes you want to make must be in those files (they are categories baised on argument type).

## Code for bringing individual results files together

This code is provided in the hw2.py code as a function called aggregate_stats().

And this is turned on via the
`--meta`
argument.
