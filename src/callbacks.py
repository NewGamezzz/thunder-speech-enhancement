import os
import torch
import wandb
from torchaudio import load
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from .utils.other import si_sdr, get_lr


class Callback:
    def on_start(self, trainer):
        """Called at the beginning of training"""
        pass

    def on_train_epoch_start(self, trainer, epoch):
        """Called at the beginning of each training epoch"""
        pass

    def on_train_batch_start(self, trainer, batch, batch_idx):
        """Called at the beginning of each training batch"""
        pass

    def on_train_batch_end(self, trainer, batch, batch_idx, loss):
        """Called at the end of each training batch"""
        pass

    def on_train_epoch_end(self, trainer, epoch, logs=None):
        """Called at the end of each training epoch"""
        pass

    def on_validation_epoch_start(self, trainer, epoch):
        """Called at the beginning of validation epoch"""
        pass

    def on_validation_batch_start(self, trainer, batch, batch_idx):
        """Called at the beginning of validation batch"""
        pass

    def on_validation_batch_end(self, trainer, batch, batch_idx, loss):
        """Called at the end of validation batch"""
        pass

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        """Called at the end of each training epoch"""
        pass

    def on_end(self, trainer):
        """Called at the end of validation"""
        pass


class TQDMProgressBar(Callback):
    def __init__(self):
        self.pbar = None
        self.training_loss = None
        self.validation_loss = None

    def on_train_batch_start(self, trainer, batch, batch_idx):
        batch_length = len(trainer.data_module.train_dataloader())
        if self.pbar is None:
            self.pbar = tqdm(total=batch_length, desc=f"Training")

    def on_train_batch_end(self, trainer, batch, batch_idx, loss):
        self.pbar.update(1)
        self.pbar.set_postfix({"loss": loss.item()})

    def on_train_epoch_end(self, trainer, epoch, logs=None):
        self.pbar.close()
        self.pbar = None
        self.training_loss = logs.get("loss")

    def on_validation_batch_start(self, trainer, batch, batch_idx):
        batch_length = len(trainer.data_module.val_dataloader())
        if self.pbar is None:
            self.pbar = tqdm(total=batch_length, desc=f"Validation")

    def on_validation_batch_end(self, trainer, batch, batch_idx, loss):
        self.pbar.update(1)

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        self.pbar.close()
        self.pbar = None
        self.validation_loss = logs.get("loss")
        print(
            f"Epoch {epoch} ended with loss: Training loss -> {self.training_loss:.4f} Validation loss -> {self.validation_loss:.4f}"
        )


class WanDBLogger(Callback):
    def __init__(self, wandb_config):
        self.wandb_config = wandb_config

    def on_start(self, trainer):
        wandb.init(**self.wandb_config)


class ValidationInference(Callback):
    def __init__(self, val_interval, num_eval_files, val_dataset, inference, save_path):
        """Validate model and save best model

        Args:
            val_interval: Validate model for every `val_interval` epoch.
            num_eval_files: Use only `num_eval_files` utterances for validation.
            val_dataset: Validate model on `val_dataset`.
            inference: Generate sample using given `inference`.
            save_path: Save model at the given directory.

        """
        self.val_interval = val_interval
        self.num_eval_files = num_eval_files
        self.val_dataset = val_dataset
        self.inference = inference
        self.save_path = save_path
        self.best_pesq = None
        self.best_si_sdr = None
        os.makedirs(self.save_path)

    def on_train_epoch_start(self, trainer, epoch):
        wandb.log({"epoch": epoch, "lr": get_lr(trainer.optimizer)})

    def on_train_epoch_end(self, trainer, epoch, logs=None):
        wandb.log({"epoch": epoch, "train_loss": logs.get("loss")})

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        wandb.log({"epoch": epoch, "val_loss": logs.get("loss")})
        if epoch % self.val_interval:
            return

        sr = 16000
        trainer.eval()
        clean_files = self.val_dataset.clean_files
        noisy_files = self.val_dataset.noisy_files

        total_num_files = len(clean_files)
        indices = torch.linspace(0, total_num_files - 1, self.num_eval_files, dtype=torch.int)
        clean_files = list(clean_files[i] for i in indices)
        noisy_files = list(noisy_files[i] for i in indices)

        _pesq, _si_sdr, _estoi = 0.0, 0.0, 0.0

        for i in tqdm(range(self.num_eval_files)):
            x, _ = load(clean_files[i])
            y, _ = load(noisy_files[i])
            x, y = x.to(trainer.device), y.to(trainer.device)
            x_hat = self.inference.inference(y)

            x_hat = x_hat.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()

            _si_sdr += si_sdr(x, x_hat)
            _pesq += pesq(sr, x, x_hat, "wb")
            _estoi += stoi(x, x_hat, sr, extended=True)

        _pesq /= self.num_eval_files
        _si_sdr /= self.num_eval_files
        _estoi /= self.num_eval_files

        wandb.log({"epoch": epoch, "pesq": _pesq, "si_sdr": _si_sdr, "estoi": _estoi})

        if self.best_pesq is None or self.best_pesq < _pesq:
            self.best_pesq = _pesq
            save_path = f"epoch={epoch}-pesq={_pesq:.2f}.ckpt"
            save_path = os.path.join(self.save_path, save_path)
            trainer.save(save_path)
        if self.best_si_sdr is None or self.best_si_sdr < _si_sdr:
            self.best_si_sdr = _si_sdr
            save_path = f"epoch={epoch}-si_sdr={_si_sdr:.2f}.ckpt"
            save_path = os.path.join(self.save_path, save_path)
            trainer.save(save_path)
        save_path = f"epoch={epoch}-last.ckpt"
        save_path = os.path.join(self.save_path, save_path)
        trainer.save(save_path)
