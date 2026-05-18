#!/bin/bash


#  1) compilers/cuda/12.1   2) cudnn/8.9.4.25_cuda12.x  
module load compilers/cuda/12.1
module load cudnn/8.9.4.25_cuda12.x

# Activate the conda environment
# Check if the conda environment is activated
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RoboFactoryNew
#To check if the conda environment is activated
pip list

# Run the workspace.py script
python workspace.py --config-name=local_dp2.yaml task=2a_two_robots_stack_cube_active_local.yaml

