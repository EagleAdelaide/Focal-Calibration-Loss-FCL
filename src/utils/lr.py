import math
from typing import List, Sequence, Tuple, Union


def cosine_lr(base_lr: float, epoch: int, max_epoch: int) -> float:
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epoch))


def piecewise_lr(lr_schedule: Sequence[Sequence[Union[int, float]]], epoch: int) -> float:
    """Piecewise-constant schedule.

    lr_schedule: list of [start_epoch, lr], sorted by start_epoch.
    Example: [[0, 0.1], [150, 0.01], [250, 0.001]]
    """
    if lr_schedule is None or len(lr_schedule) == 0:
        raise ValueError("lr_schedule must be a non-empty list of [start_epoch, lr].")
    cur = float(lr_schedule[0][1])
    for start, lr in lr_schedule:
        if epoch >= int(start):
            cur = float(lr)
        else:
            break
    return cur
