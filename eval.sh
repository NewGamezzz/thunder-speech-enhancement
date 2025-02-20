# #!/bin/bash
# #SBATCH --partition=11001207 
# #SBATCH --gpus-per-task=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem 60G
# #SBATCH -c 30
# #SBATCH -J eval-thunder
# #SBATCH --output=./logs/eval_%j.txt 

# module purge

# module load shared
# module load cuda11.8/toolkit/11.8.0

python eval.py --inference_config "./config/inference/two_stage.yaml" --model_path "./outputs/2024-11-28/22-10-52/weights/epoch=75-pesq=2.78.ckpt"