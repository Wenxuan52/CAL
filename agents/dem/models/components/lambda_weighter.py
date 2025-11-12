from abc import ABC, abstractmethod

import torch

from .noise_schedules import BaseNoiseSchedule


class BaseLambdaWeighter(ABC):
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BasicLambdaWeighter(BaseLambdaWeighter):
    def __init__(self, noise_schedule: BaseNoiseSchedule, epsilon: float = 1e-3):
        self.noise_schedule = noise_schedule
        self.epsilon = epsilon

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        value = self.noise_schedule.h(t)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=t.device, dtype=t.dtype)
        else:
            value = value.to(device=t.device, dtype=t.dtype)
        return value + self.epsilon


class NoLambdaWeighter(BasicLambdaWeighter):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1
