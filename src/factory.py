import os
import torch
import wandb
import hydra
import copy
from torch_ema import ExponentialMovingAverage
from src import (
    BackboneRegistry,
    SDERegistry,
    InferenceRegistry,
    SchedulerRegister,
    LossRegistry,
    InferenceRegistry,
    Regression,
    Diffusion,
    Trainer,
    SpecsDataModule,
    TQDMProgressBar,
    WanDBLogger,
    ValidationInference,
)


def create_model(config, data_module=None):
    name_to_func = {"diffusion": create_diffusion, "regression": create_regression}

    name = "diffusion" if config.get("name") is None else config.pop("name")
    model = name_to_func[name](config, data_module)
    return model


def create_regression(config, data_module=None):
    backbone_config = config.pop("backbone")
    backbone_name = backbone_config.pop("name")
    backbone_class = BackboneRegistry.get_by_name(backbone_name)
    backbone = backbone_class(**backbone_config)

    transform = config.pop("transform", None)
    transform_name_to_func = {"spec_to_raw": create_spec_to_raw_transform(data_module)}
    transform = None if transform is None else transform_name_to_func[transform]

    loss_config = config.pop("loss")
    loss_name = loss_config.pop("name")
    loss_class = LossRegistry.get_by_name(loss_name)
    if loss_config.get("transform"):
        transform_name = loss_config.pop("transform")
        transform_name_to_func = {"spec_to_raw": create_spec_to_raw_transform(data_module)}
        loss_config["transform"] = transform_name_to_func[transform_name]
    loss = loss_class(**loss_config)

    regression = Regression(backbone, loss, transform, **config)
    return regression


def create_diffusion(config, data_module=None):
    sde_config = config.pop("sde")
    sde_name = sde_config.pop("name")
    sde_class = SDERegistry.get_by_name(sde_name)
    sde = sde_class(**sde_config)

    backbone_config = config.pop("backbone")
    backbone_name = backbone_config.pop("name")
    backbone_class = BackboneRegistry.get_by_name(backbone_name)
    backbone = backbone_class(**backbone_config)

    loss_config = config.pop("loss")
    loss_name = loss_config.pop("name")
    loss_class = LossRegistry.get_by_name(loss_name)
    if loss_config.get("transform"):
        transform_name = loss_config.pop("transform")
        transform_name_to_func = {"spec_to_raw": create_spec_to_raw_transform(data_module)}
        loss_config["transform"] = transform_name_to_func[transform_name]
    loss = loss_class(**loss_config)

    diffusion = Diffusion(backbone, sde, loss, **config)
    return diffusion


def create_spec_to_raw_transform(data_module):
    def spec_to_mel(X):
        x = data_module.istft(data_module.spec_back(X.squeeze()))
        return x

    return spec_to_mel


def create_dataset(config):
    dataset = SpecsDataModule(**config)
    dataset.setup(stage="fit")
    return dataset


def create_scheduler(optimizer, config):
    if config.get("scheduler") is None:
        return None
    scheduler_config = config.pop("scheduler")
    if scheduler_config.get("name") is None:
        return None

    scheduler_name = scheduler_config.pop("name")
    scheduler_class = SchedulerRegister.get_by_name(scheduler_name)
    scheduler = scheduler_class(optimizer, **scheduler_config)
    return scheduler


def create_inference(config, diffusion, data_module):
    inference_name = config.pop("name")
    inference_cls = InferenceRegistry.get_by_name(inference_name)
    inference = inference_cls(diffusion, data_module, **config)
    return inference


def create_callback(config, **kwargs):
    callbacks_names_to_func = {
        "tqdm": create_tqdm_callback,
        "wandb": create_wandb_callback,
        "validation_inference": create_validation_inference_callback,
    }
    name = config.pop("name")
    return callbacks_names_to_func[name](config=config, **kwargs)


def create_tqdm_callback(*args, **kwargs):
    return TQDMProgressBar()


def create_wandb_callback(config, general_config, *args, **kwargs):
    wandb_token = config.pop("WANDB_TOKEN")
    wandb_config = config.pop("WANDB_INIT")
    wandb.login(key=wandb_token)
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    wandb_name = "/".join(output_path.split("/")[-2:])
    general_config["output_path"] = output_path

    # Check wandb callback and pop
    wandb_idx = -1
    for idx, callback in enumerate(
        general_config["trainer"]["callbacks"]
    ):  # There must be callback in the config to call this function
        if callback["name"] == "wandb":
            wandb_idx = idx
    if wandb_idx != -1:
        general_config["trainer"]["callbacks"].pop(wandb_idx)

    wandb_config["config"] = general_config
    wandb_config["name"] = wandb_name
    return WanDBLogger(wandb_config)


def create_validation_inference_callback(config, diffusion, data_module, *args, **kwargs):
    inference_config = config.pop("inference")
    inference_name = inference_config.pop("name")
    inference_class = InferenceRegistry.get_by_name(inference_name)

    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    folder_name = config["save_path"].replace("./", "")
    config["save_path"] = os.path.join(output_path, folder_name)
    inference = inference_class(diffusion, data_module, **inference_config)

    return ValidationInference(val_dataset=data_module.valid_set, inference=inference, **config)


def create_trainer(diffusion, data_module, config):
    general_config = copy.deepcopy(config)
    trainer_config = config["trainer"]

    optimizer_config = trainer_config.pop("optimizer")
    optimizer = torch.optim.Adam(diffusion.parameters(), **optimizer_config)

    ema_config = trainer_config.pop("ema", None)
    if ema_config is not None:
        ema = ExponentialMovingAverage(diffusion.parameters(), **ema_config)
    else:
        ema = None

    scheduler = create_scheduler(optimizer, config)
    kwargs = {"diffusion": diffusion, "data_module": data_module, "general_config": general_config}
    callbacks = []
    callbacks_config = trainer_config.get("callbacks", [])
    for callback_config in callbacks_config:
        callback = create_callback(callback_config, **kwargs)
        callbacks.append(callback)

    trainer = Trainer(
        diffusion, data_module, optimizer, ema, callbacks=callbacks, scheduler=scheduler
    )

    return trainer
