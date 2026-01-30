from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import torch

def save_ckpt(path: str, state: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)

def load_ckpt(path: str, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
