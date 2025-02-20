import hydra
from omegaconf import DictConfig, OmegaConf
from src import factory


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(config: DictConfig) -> None:
    data_config = OmegaConf.to_container(config.dataset)
    data_module = factory.create_dataset(data_config)

    model_config = OmegaConf.to_container(config.model)
    model = factory.create_model(model_config, data_module=data_module)

    all_config = OmegaConf.to_container(config)
    trainer = factory.create_trainer(model, data_module, all_config)
    trainer.fit(config.trainer.epochs)


if __name__ == "__main__":
    train()
