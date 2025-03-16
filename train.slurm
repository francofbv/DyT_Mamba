#!/bin/bash
#SBATCH --job-name=vit_train        # Job name
#SBATCH --time=48:00:00             # Time limit (HH:MM:SS)
#SBATCH --cpus-per-task=10          # Number of CPU cores per task
#SBATCH --mem-per-cpu=10G           # Memory per CPU core
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --constraint=gpu80          # Constraint to specific GPU type
#SBATCH --gres=gpu:8                # Request 8 GPUs
#SBATCH --partition=highgpu         # Partition to submit to
#SBATCH --output=vit_train_%j.out   # Output file
#SBATCH --error=vit_train_%j.err    # Error file
#SBATCH --mail-type=END             # Send email when job ends
#SBATCH --mail-user=francobicudovidal@gmail.com # Replace with your email


# Load necessary modules (if required)
module load cuda/11.8  # Adjust based on your environment

# Activate your Conda environment
source ~/.bashrc
conda activate DyT # Use your correct environment

# Run the training script
srun torchrun --nproc_per_node=8 main.py \
    --model vit_base_patch16_224 \
    --drop_path 0.1 \
    --batch_size 64 \
    --lr 4e-3 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_path /datasets/ImageNet2012nonpub/ \
    --output_dir /home/fvidal/DyT_output \
    --dynamic_tanh true \
    --epochs 100
