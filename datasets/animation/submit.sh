#!/bin/bash
 
# Initialize and Load Modules
source /etc/profile
module load anaconda/2023a-pytorch
 
echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

export CUDA_VISIBLE_DEVICES=($LLSUB_RANK % 2)
 
python taichi_init.py $LLSUB_RANK $LLSUB_SIZE

# to submit this 
# LLsub ./submit.sh [4,2,1] -g volta:2
