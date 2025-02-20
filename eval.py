import warnings

warnings.simplefilter("ignore", UserWarning)

import os
import torch
import argparse
import yaml
import pandas as pd
from torchaudio import load
from tqdm import tqdm

from pesq import pesq
from pystoi import stoi
from src.utils.other import si_sdr

from src import factory


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_config",
        type=str,
        required=True,
        help="Path YAML config file for inference class",
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Load model from the given path"
    )
    parser.add_argument("--csv_path", type=str, default=None, help="Save metric score to csv_path")
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


def evaluate(inference, data_module):
    clean_files = data_module.test_set.clean_files
    noisy_files = data_module.test_set.noisy_files

    total_num_files = len(clean_files)

    metric_data = {"filename": [], "si_sdr": [], "pesq": [], "estoi": []}
    with torch.inference_mode():
        for i in tqdm(range(total_num_files)):
            x, _ = load(clean_files[i])
            y, _ = load(noisy_files[i])
            x, y = x.to("cuda"), y.to("cuda")
            x_hat = inference.inference(y)

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()

            _si_sdr = si_sdr(x, x_hat)
            _pesq = pesq(16000, x, x_hat, "wb")
            _estoi = stoi(x, x_hat, 16000, extended=True)
            metric_data["filename"].append(clean_files[i].split("/")[-1])
            metric_data["si_sdr"].append(_si_sdr)
            metric_data["pesq"].append(_pesq)
            metric_data["estoi"].append(_estoi)

    metric_df = pd.DataFrame(metric_data)
    return metric_df


if __name__ == "__main__":
    args = get_args()
    model_path = args.model_path
    config_path = os.path.join("/".join(model_path.split("/")[:-2]), ".hydra/config.yaml")
    config = load_config(config_path)
    inference_config = load_config(args.inference_config)

    data_module = factory.create_dataset(config["dataset"])
    data_module.setup("test")

    model_state_dict = torch.load(model_path)
    diffusion = factory.create_diffusion(config["model"])
    diffusion.load_state_dict(model_state_dict["model"])
    diffusion = diffusion.to("cuda")
    diffusion.eval()

    inference = factory.create_inference(inference_config, diffusion, data_module)

    metric_df = evaluate(inference, data_module)

    if args.csv_path is not None:
        if os.path.exists(args.csv_path):
            print("Error: Cannot overwrite the existing csv. Please change csv_path.")
        metric_df.to_csv(args.csv_path, index=False)
    print(
        f"PESQ: {metric_df['pesq'].mean()}, SI-SDR: {metric_df['si_sdr'].mean()}, ESTOI: {metric_df['estoi'].mean()}"
    )
