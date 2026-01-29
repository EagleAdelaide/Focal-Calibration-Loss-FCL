from __future__ import annotations
from typing import Dict, Callable, Any
import torch.nn as nn

_REG: Dict[str, Callable[[dict, int], nn.Module]] = {}

def register(name: str):
    def deco(fn):
        _REG[name] = fn
        return fn
    return deco

def build_loss(cfg: dict, num_classes: int) -> nn.Module:
    lname = cfg["loss"]["name"].lower()
    if lname not in _REG:
        raise ValueError(f"Unknown loss: {lname}. Available: {sorted(_REG.keys())}")
    return _REG[lname](cfg, num_classes)
