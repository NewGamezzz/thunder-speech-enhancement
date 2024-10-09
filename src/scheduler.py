import torch
import numpy as np
from .utils.registry import Registry

SchedulerRegister = Registry("Scheduler")


@SchedulerRegister.register("warmup")
class CMScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr,
        max_lr,
        warmup_duration,
        iter_per_ep,
        lr_decay,
    ):
        self.cur_iter = 0
        if warmup_duration != 0:
            self.its = [0, warmup_duration * iter_per_ep, iter_per_ep * 1000]
        else:
            self.its = [0, iter_per_ep * 1000]
        self.schedulers = []
        if warmup_duration != 0:
            self.schedulers.append(
                torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=base_lr,
                    max_lr=max_lr,
                    step_size_up=warmup_duration * iter_per_ep,
                    cycle_momentum=False,
                )
            )
        self.schedulers.append(
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=iter_per_ep, gamma=lr_decay)
        )

    def step(self):
        self.cur_iter += 1
        # Changes the learning rate here
        ind = np.searchsorted(self.its, self.cur_iter, side="right") - 1
        self.schedulers[ind].step()

    def state_dict(self):
        state_dict = {}
        for key, value in self.__dict__.items():
            if key == "schedulers":
                state_dict["schedulers"] = []
                for scheduler in value:
                    state_dict["schedulers"].append(scheduler.state_dict())
            else:
                state_dict[key] = value
        return state_dict

    def load_state_dict(self, state_dict):
        scheduler_state_dict = state_dict.pop("schedulers")
        self.__dict__.update(state_dict)

        assert len(scheduler_state_dict) == len(
            self.schedulers
        ), "Length of scheduler state dict does not match with scheduler list."
        for idx in len(self.schedulers):
            self.schedulers[idx].load_state_dict(scheduler_state_dict[idx])
