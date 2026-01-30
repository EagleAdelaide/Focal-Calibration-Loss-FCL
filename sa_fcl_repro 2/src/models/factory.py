from typing import Any, Dict, Union

from .resnet_tvm_mod import resnet18, resnet50
from .cifar_resnet import resnet110
from .wideresnet import wideresnet_26_10
from .densenet_mod import densenet121

def build_model(model: Union[Dict[str, Any], str], num_classes: int):
    """Build a model.

    This scaffold originally accepted a dict config (e.g. cfg["model"]).
    Several entrypoints (train/eval scripts) pass a plain string like
    "resnet18". To make the repo runnable out-of-the-box, we accept both.
    """

    if isinstance(model, str):
        cfg: Dict[str, Any] = {"name": model}
    else:
        cfg = model

    name = str(cfg.get("name", "")).lower()
    cifar_stem = bool(cfg.get("cifar_stem", False))
    weights = cfg.get("weights", None)  # e.g., "DEFAULT" for ImageNet pretrain

    # torchvision weights enums are objects; allow string "DEFAULT" to mean default weights.
    # We import lazily to avoid hard dependency if user doesn't use pretrained.
    tv_weights = None
    if isinstance(weights, str) and weights.upper() == "DEFAULT":
        try:
            import torchvision.models as tvm
            if name in ["resnet18", "resnet50"]:
                tv_weights = getattr(getattr(tvm, f"{name.capitalize()}_Weights"), "DEFAULT", None)
            elif name in ["densenet121"]:
                tv_weights = getattr(getattr(tvm, "DenseNet121_Weights"), "DEFAULT", None)
        except Exception:
            tv_weights = None

    if name == "resnet18":
        return resnet18(num_classes=num_classes, cifar_stem=cifar_stem, weights=tv_weights)
    if name == "resnet50":
        return resnet50(num_classes=num_classes, cifar_stem=cifar_stem, weights=tv_weights)
    if name == "resnet110":
        return resnet110(num_classes=num_classes)
    if name in ["wideresnet", "wideresnet_26_10", "wrn26_10"]:
        dropout = float(cfg.get("dropout", 0.0))
        return wideresnet_26_10(num_classes=num_classes, dropout_rate=dropout)
    if name in ["densenet121", "densenet"]:
        return densenet121(num_classes=num_classes, cifar_stem=cifar_stem, weights=tv_weights)

    raise ValueError(f"Unknown model: {name}")
