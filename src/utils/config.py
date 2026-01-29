import os
import copy
from typing import Any, Dict, List, Union

import yaml


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dicts (src overwrites dst)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _resolve_path(base_file: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(os.path.dirname(base_file), rel_or_abs))


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with optional inheritance via `_base_`.

    `_base_` can be a string or list of strings. Bases are loaded first (in order),
    then the current file overwrites via deep merge.
    """
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    bases = cfg.pop("_base_", None)
    if bases is None:
        return cfg

    if isinstance(bases, (str,)):
        bases = [bases]
    if not isinstance(bases, list):
        raise TypeError(f"_base_ must be a string or list[str], got: {type(bases)}")

    merged: Dict[str, Any] = {}
    for b in bases:
        bpath = _resolve_path(path, b)
        bcfg = load_config(bpath)
        _deep_update(merged, bcfg)

    _deep_update(merged, cfg)
    return merged


def merge_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Merge CLI overrides like a.b.c=1 into cfg."""
    if not overrides:
        return cfg

    def _parse_value(x: str):
        # yaml parser gives us numbers/bools/lists conveniently
        try:
            return yaml.safe_load(x)
        except Exception:
            return x

    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Bad override '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        v = _parse_value(v)
        keys = k.split(".")
        cur = cfg
        for kk in keys[:-1]:
            if kk not in cur or not isinstance(cur[kk], dict):
                cur[kk] = {}
            cur = cur[kk]
        cur[keys[-1]] = v
    return cfg
