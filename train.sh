#!/bin/bash
#SBATCH --partition=11001207 
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 60G
#SBATCH -c 30
#SBATCH -J train-thunder
#SBATCH --output=./logs/train_%j.txt

module purge

module load shared
module load cuda11.8/toolkit/11.8.0

# python train.py +experiment=ncsnpp 
python train.py +experiment=gagnet trainer.optimizer.lr=0.0005 trainer.ema=null
# python train.py model.loss.name="sdsnr" trainer.optimizer.lr=0.001 trainer.ema=null
# python train.py trainer.optimizer.lr=0.001 trainer.ema=null trainer.callbacks.2.inference.transform="raw"