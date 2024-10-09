import argparse
import yaml
import torch
import wandb
from torch_ema import ExponentialMovingAverage
from src import (
    Trainer,
    Diffusion,
    SpecsDataModule,
    SDERegistry,
    BackboneRegistry,
    LossRegistry,
    InferenceRegistry,
    TQDMProgressBar,
    WanDBLogger,
    ValidationInference,
    count_parameters,
    SchedulerRegister,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--wandb_config", type=str, required=True, help="Path to wandb config file")
    parser.add_argument("--save_path", type=str, default=None, help="Save Directory")
    parser.add_argument(
        "--resume_path", type=str, default=None, help="Load trainer state dict from the given path"
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A comment to be added to config name"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Overwrite epochs in the config")
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)
    config["name"] += args.comment
    # Overwrite config
    if args.epochs is not None:
        config["epochs"] = args.epochs
    # Wandb Config
    wandb_config = load_config(args.wandb_config)
    wandb_token = wandb_config["WANDB_TOKEN"]
    wandb_config = wandb_config["WANDB_INIT"]
    print(f"Initialize Diffusion Model")

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
    data_module.setup(stage="fit")

    # Initialize Trainer
    print(f"Loss: {config['loss']}")
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=config["lr"])
    ema = ExponentialMovingAverage(diffusion.parameters(), decay=config["ema_decay"])
    loss = LossRegistry.get_by_name(config["loss"])()

    ## Initialize Callback (TQDMProgressBar, WanDB, and ValidationInference)
    wandb.login(key=wandb_token)  # login wandb
    wandb_config["config"] = config
    wandb_config["name"] = config["name"]
    inference = InferenceRegistry.get_by_name(config["inference"]["name"])(
        diffusion, data_module, **config["inference"]
    )
    if args.save_path is None:
        args.save_path = f"./weights/{config['name']}"
    callbacks = [
        TQDMProgressBar(),
        WanDBLogger(wandb_config),
        ValidationInference(
            config["validation"]["val_interval"],
            config["validation"]["num_eval_files"],
            data_module.valid_set,
            inference,
            save_path=args.save_path,
        ),
    ]
    ## Initialize Scheduler
    scheduler = None
    if config.get("scheduler", None) is not None:
        scheduler_name = config["scheduler"].pop("name")
        print(f"Scheduler: {scheduler_name}")
        scheduler_cls = SchedulerRegister.get_by_name(scheduler_name)
        scheduler = scheduler_cls(optimizer, **config["scheduler"])

    trainer = Trainer(
        diffusion, data_module, optimizer, ema, loss, callbacks=callbacks, scheduler=scheduler
    )
    if args.resume_path is not None:
        trainer.load(args.resume_path)
        print(f"Load trainer from: {args.resume_path}")
    trainer.fit(epochs=config["epochs"])
