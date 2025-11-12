"""Utility helpers for the DEM agent without Lightning dependencies."""

from __future__ import annotations

from typing import Any, Dict

from agents.dem.utils.pylogger import RankedLogger, rank_prefixed_message, rank_zero_only

__all__ = [
    "RankedLogger",
    "rank_prefixed_message",
    "rank_zero_only",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
    "enforce_tags",
    "extras",
    "task_wrapper",
    "get_metric_value",
]


def instantiate_callbacks(cfg: Any):
    from agents.dem.utils.instantiators import instantiate_callbacks as _instantiate_callbacks

    return _instantiate_callbacks(cfg)


def instantiate_loggers(cfg: Any):
    from agents.dem.utils.instantiators import instantiate_loggers as _instantiate_loggers

    return _instantiate_loggers(cfg)


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    from agents.dem.utils.logging_utils import log_hyperparameters as _log_hparams

    _log_hparams(object_dict)


def print_config_tree(*args, **kwargs):
    from agents.dem.utils.rich_utils import print_config_tree as _print_config_tree

    return _print_config_tree(*args, **kwargs)


def enforce_tags(*args, **kwargs):
    from agents.dem.utils.rich_utils import enforce_tags as _enforce_tags

    return _enforce_tags(*args, **kwargs)


def extras(*args, **kwargs):
    from agents.dem.utils.utils import extras as _extras

    return _extras(*args, **kwargs)


def task_wrapper(*args, **kwargs):
    from agents.dem.utils.utils import task_wrapper as _task_wrapper

    return _task_wrapper(*args, **kwargs)


def get_metric_value(*args, **kwargs):
    from agents.dem.utils.utils import get_metric_value as _get_metric_value

    return _get_metric_value(*args, **kwargs)
