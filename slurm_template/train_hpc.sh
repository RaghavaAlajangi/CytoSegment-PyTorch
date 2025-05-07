#!/bin/bash -l
#SBATCH -o {{PATH_OUT}}/{{EXP_NAME}}/job_files/out.%j.log
#SBATCH -e {{PATH_OUT}}/{{EXP_NAME}}/job_files/err.%j.log
#SBATCH -D ./
#SBATCH -J {{JOB_NAME}}
# We need a GPU node
#SBATCH --partition=gpu
#SBATCH --constraint="gpu"
# request one GPU for model training
#SBATCH --gres=gpu:1
# We only need one node
#SBATCH --nodes=1
# Request one CPU per task
#SBATCH --cpus-per-task=16
# This is probably redundant, but we only have one task!
#SBATCH --ntasks-per-node=1
# Initial tests showed about ~2:30h with bloody_bunny
#SBATCH --time={{MAX_TIME}}
# memory required per node. Default units are megabytes.
#SBATCH --mem={{MAX_MEM}}
#SBATCH --mail-type=all
#SBATCH --mail-user={{MAIL_ID}}

# Import modules
module purge
module load anaconda/3/2021.11
module load cuda/11.6
module load cudnn/8.8.1
module load onnx/1.8.1
#Pytorch
module load pytorch/gpu-cuda-11.6/2.0.0
pip install virtualenv
# Create and activate a new venv
virtualenv --system-site-packages {{PATH_OUT}}/{{EXP_NAME}}/venv --python=python3.9.7
. {{PATH_OUT}}/{{EXP_NAME}}/venv/bin/activate

pip install tensorboard
srun python -m cytosegment --params_path {{PARAMS_PATH}}

# Remove existing venv
if [ -d "{{PATH_OUT}}/{{EXP_NAME}}/venv" ]; then
    rm -rf {{PATH_OUT}}/{{EXP_NAME}}/venv
fi
