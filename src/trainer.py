from dataclasses import dataclass
from typing import List
from .callbacks import Callback
import torch


@dataclass
class OutputData:
    pred_x0: torch.Tensor
    pred_score: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    z: torch.Tensor  # gaussian noise for calculating denoising score matching loss
    x0: torch.Tensor


class Trainer:
    def __init__(
        self,
        model,
        data_module,
        optimizer,
        ema,
        loss,
        scheduler=None,
        callbacks: List[Callback] = None,
        device="cuda",
    ):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.ema = ema
        self.loss = loss
        self.scheduler = scheduler
        self.callbacks = callbacks if callbacks is not None else []
        self.device = device
        self.state = None  # Keep track whether model is training or eval.
        self.model.to(device)
        self.ema.to(device)

    def _call_callbacks(self, hook_name, **kwargs):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if callable(hook):
                hook(self, **kwargs)

    def fit(self, epochs):
        self._call_callbacks("on_start")
        for epoch in range(1, epochs + 1):
            self.training_loop(epoch)
            self.validation_loop(epoch)
        self._call_callbacks("on_end")

    def training_loop(self, epoch):
        self._call_callbacks("on_train_epoch_start", epoch=epoch)
        self.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(self.data_module.train_dataloader(), 1):
            self._call_callbacks("on_train_batch_start", batch=batch, batch_idx=batch_idx)
            loss = self.training_step(batch)
            epoch_loss += loss.item()
            self._call_callbacks("on_train_batch_end", batch=batch, batch_idx=batch_idx, loss=loss)
            if self.scheduler:
                self.scheduler.step()
        average_loss = epoch_loss / len(self.data_module.train_dataloader())
        self._call_callbacks("on_train_epoch_end", epoch=epoch, logs={"loss": average_loss})

    def training_step(self, batch):
        self.optimizer.zero_grad()
        out = self._step(batch)
        loss = self.loss(out)
        loss.backward()
        self.optimizer_step()
        return loss

    def validation_loop(self, epoch):
        self._call_callbacks("on_validation_epoch_start", epoch=epoch)
        self.eval()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(self.data_module.val_dataloader(), 1):
            self._call_callbacks("on_validation_batch_start", batch=batch, batch_idx=batch_idx)
            loss = self.validation_step(batch)
            epoch_loss += loss.item()
            self._call_callbacks(
                "on_validation_batch_end", batch=batch, batch_idx=batch_idx, loss=loss
            )
        average_loss = epoch_loss / len(self.data_module.val_dataloader())
        self._call_callbacks("on_validation_epoch_end", epoch=epoch, logs={"loss": average_loss})

    def validation_step(self, batch):
        with torch.no_grad():
            out = self._step(batch)
            loss = self.loss(out)
        return loss

    def _step(self, batch):
        x0, y = batch
        x0, y = x0.to(self.device), y.to(self.device)

        t = (
            torch.rand(x0.shape[0], device=x0.device) * (self.model.sde.T - self.model.t_eps)
            + self.model.t_eps
        )
        mean, std = self.model.sde.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,) * (y.ndim - std.ndim)))
        xt = mean + std * z
        out = self.model(xt, t, y)

        if self.model.pred_type == "score":
            out = OutputData(
                pred_x0=None,  # TODO Derive pred_x0 from pred_score
                pred_score=out,
                mean=mean,
                std=std,
                z=z,
                x0=x0,
            )
        elif self.model.pred_type == "x0":
            out = OutputData(
                pred_x0=out,
                pred_score=None,  # TODO Derive pred_score from pred_x0
                mean=mean,
                std=std,
                z=z,
                x0=x0,
            )
        return out

    def optimizer_step(self, *args, **kwargs):
        self.optimizer.step()
        self.ema.update(self.model.parameters())

    def train(self):
        if self.state == "train":
            return
        self.state = "train"
        self.model.train()
        if self.ema and self.ema.collected_params is not None:
            self.ema.restore(self.model.parameters())  # restore the EMA weights

    def eval(self):
        if self.state == "eval":
            return
        self.state = "eval"
        self.model.eval()
        if self.ema:
            self.ema.store(self.model.parameters())  # store current params in EMA
            self.ema.copy_to(
                self.model.parameters()
            )  # copy EMA parameters over current params from evaluation

    def save(self, save_path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ema": self.ema.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "state": self.state,
            },
            save_path,
        )

    def load(self, load_path):
        state_dict = torch.load(load_path)
        self.state = state_dict.get("state", "eval")
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.ema.load_state_dict(state_dict["ema"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict.get("scheduler", None))
