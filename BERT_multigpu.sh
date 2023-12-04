#!/bin/bash

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=BERT_finetune    # Job name
#SBATCH --output=BERT_finetune.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=BERT_finetune.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=32     # Number of tasks(processes)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --mem=300G   # memory per NODE
#SBATCH --mail-user=bax001@ucsd.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

source ~/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
conda activate LLM

python /new-stg/home/banghua/Amazon-Rating-Prediction/BERT_multigpu.py