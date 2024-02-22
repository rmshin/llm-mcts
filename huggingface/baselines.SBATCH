#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=llm

singularity exec --nv --overlay /scratch/akp7833/overlays/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash -c "source /ext3/env.sh; cd /scratch/akp7833/llm-mcts/huggingface; python hf.py"
