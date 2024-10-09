# Thunder : Unified Regression-Diffusion Speech Enhancement with a Single Reverse Step using Brownian Bridge
This repository contains the official PyTorch implementations for [Trachu, T., Piansaddhayanon, C., Chuangsuwanich, E. (2024) Thunder : Unified Regression-Diffusion Speech Enhancement with a Single Reverse Step using Brownian Bridge. Proc. Interspeech 2024, 1180-1184](https://www.isca-archive.org/interspeech_2024/trachu24_interspeech.html#)

# Installation
- Create a new virtual environment with Python 3.11 (we have not tested other Python versions, but they may work).
- Install the package dependencies via pip install -r requirements.txt.
- Set up W&B logging:
  - Set up a wandb.ai account
  - Insert a wandb token in ./wandb_config/default.yaml first before training.

# Training
Training is done by executing train.py. A minimal running example with default settings can be run with
```
python train.py --config ./config/default.yaml --wandb_config ./wandb_config/default.yaml 
```

# Evaluation
To evaluate on a test set, run
```
python eval.py --config ./config/default.yaml --inference_config "./config/inference/two_stage.yaml" --model_path <path_to_model_checkpoint> --N <number_of_reverse_step>
```
We provide a pretrained checkpoints for a model that have been trained on the VoiceBank-Demand dataset where you can download [here](https://drive.google.com/file/d/1_dtVsJVfG1H01-35BvU48LTtyrzAMvTV/view?usp=sharing).

To generate the enhanced .wav files
```
python inference.py --config ./config/default.yaml --inference_config "./config/inference/two_stage.yaml" --model_path <path_to_model_checkpoint> --N <number_of_reverse_step> --inference_folder <your_test_dir> --output_folder <your_output_dir>
```
The ```inference_folder``` must contains only .wav files and the enhanced .wav files will be saved at ```output_folder```

# Acknowledgement
This repository is mainly based on [SGMSE](https://github.com/sp-uhh/sgmse/blob/main/sgmse/backbones/ncsnpp.py) repository.

# Citations / References
Please cite our paper when using our code:
```bib
@inproceedings{trachu24_interspeech,
  title     = {Thunder : Unified Regression-Diffusion Speech Enhancement with a Single Reverse Step using Brownian Bridge},
  author    = {Thanapat Trachu and Chawan Piansaddhayanon and Ekapol Chuangsuwanich},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {1180--1184},
  doi       = {10.21437/Interspeech.2024-841},
  issn      = {2958-1796},
}
```
