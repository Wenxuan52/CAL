import logging
import os
from functools import wraps
from typing import Callable, Mapping, Optional


def _detect_rank() -> int:
    """Best-effort detection of the current process rank."""

    for env_var in ("LOCAL_RANK", "RANK"):
        value = os.environ.get(env_var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def rank_prefixed_message(message: str, rank: int) -> str:
    """Prefix log messages with rank information when running distributed jobs."""

    if rank == 0:
        return message
    return f"[rank {rank}] {message}"


def rank_zero_only(fn: Callable) -> Callable:
    """Decorator that only executes the wrapped function on rank zero."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _detect_rank() == 0:
            return fn(*args, **kwargs)

    return wrapper


class RankedLogger(logging.LoggerAdapter):
    """A lightweight logger that prefixes messages with rank information."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        if not self.isEnabledFor(level):
            return

        msg, kwargs = self.process(msg, kwargs)
        current_rank = _detect_rank()
        msg = rank_prefixed_message(msg, current_rank)

        if self.rank_zero_only and current_rank != 0:
            return

        if rank is not None and current_rank != rank:
            return

        self.logger.log(level, msg, *args, **kwargs)
