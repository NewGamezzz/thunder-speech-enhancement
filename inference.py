import os
import glob
import torch
import argparse
import yaml
import pandas as pd
from soundfile import write
from torchaudio import load
from tqdm import tqdm
from src import (
    Diffusion,
    InferenceRegistry,
    BackboneRegistry,
    SDERegistry,
    SpecsDataModule,
    count_parameters,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--inference_config",
        type=str,
        required=True,
        help="Path YAML config file for inference class",
    )
    parser.add_argument(
        "--inference_folder", type=str, required=True, help="A path to be inference."
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Save outputs to this folder."
    )
    parser.add_argument("--N", type=int, default=None, help="Number of reverse diffusion step.")
    parser.add_argument(
        "--interpolate_weight",
        type=float,
        default=None,
        help="Interpolation weight for two-stage system",
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Load model from the given path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference on CPU or GPU",
    )
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)
    inference_config = load_config(args.inference_config)

    # Initialize SDE
    print(f"SDE: {config['sde']['name']}")
    sde_class = SDERegistry.get_by_name(config["sde"]["name"])
    sde = sde_class(**config["sde"])

    # Initialize Backbone
    backbone_name = config["backbone"].pop("name")
    print(f"Backbone: {backbone_name}")
    backbone_cls = BackboneRegistry.get_by_name(backbone_name)
    backbone = backbone_cls(**config["backbone"])

    # Initialize Diffusion Model
    diffusion = Diffusion(backbone, sde, pred_type=config["pred_type"], t_eps=config["t_eps"])
    print(f"Number of Parameters: {count_parameters(diffusion)}")

    # Initialize Dataset
    data_module = SpecsDataModule(**config["dataset"])

    # Load diffusion weight
    state_dict = torch.load(args.model_path)
    diffusion.load_state_dict(state_dict["model"])
    diffusion = diffusion.to(args.device)
    diffusion.eval()

    # Initialize Inference from inference_config
    inference_name = inference_config.pop("name")
    print(f"Inference with {inference_name}")
    inference_cls = InferenceRegistry.get_by_name(inference_name)
    inference = inference_cls(diffusion, data_module, **inference_config)
    ## Overwrite reverse step
    if args.N is not None and getattr(inference, "N", None) is not None:
        inference.N = args.N
    if (
        args.interpolate_weight is not None
        and getattr(inference, "interpolate_weight", None) is not None
    ):
        inference.interpolate_weight = args.interpolate_weight

    # NOTE: Evaluate
    noisy_files = glob.glob(os.path.join(args.inference_folder, "*"))

    total_num_files = len(noisy_files)

    with torch.inference_mode():
        for i in tqdm(range(total_num_files)):
            filename = noisy_files[i].split("/")[-1]
            y, _ = load(noisy_files[i])
            y = y.to("cuda")
            x_hat = inference.inference(y)

            x_hat = x_hat.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            write(os.path.join(args.output_folder, filename), x_hat, 16000)
