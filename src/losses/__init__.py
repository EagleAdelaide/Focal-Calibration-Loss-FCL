from .registry import build_loss, register

from .ce import CrossEntropyLoss
from .brier import BrierScoreLoss
from .mmce import MMCE
from .focal import FocalLoss
from .flsd53 import FLSD53Loss
from .dual_focal import DualFocalLoss
from .safcl import FCLUngated, SAFCL, SAFCLNoDetach
from .composite import CEBrier, CEMMCE

@register("ce_wd")
def _build_ce(cfg, num_classes: int):
    return CrossEntropyLoss(label_smoothing=0.0)

@register("label_smoothing")
def _build_ls(cfg, num_classes: int):
    smoothing = float(cfg["loss"].get("smoothing", 0.1))
    return CrossEntropyLoss(label_smoothing=smoothing)

@register("brier")
def _build_brier(cfg, num_classes: int):
    return BrierScoreLoss(num_classes=num_classes)

@register("mmce")
def _build_mmce(cfg, num_classes: int):
    sigma = float(cfg["loss"].get("sigma", 0.4))
    return MMCE(sigma=sigma)

@register("focal")
def _build_focal(cfg, num_classes: int):
    gamma = float(cfg["loss"].get("gamma", 2.0))
    return FocalLoss(gamma=gamma)

@register("flsd53")
def _build_flsd(cfg, num_classes: int):
    pth = float(cfg["loss"].get("p_threshold", 0.5))
    gh = float(cfg["loss"].get("gamma_hard", 5.0))
    ge = float(cfg["loss"].get("gamma_easy", 3.0))
    return FLSD53Loss(p_threshold=pth, gamma_hard=gh, gamma_easy=ge)

@register("dual_focal")
def _build_dual(cfg, num_classes: int):
    gp = float(cfg["loss"].get("gamma_pos", cfg["loss"].get("gamma", 2.0)))
    gn = float(cfg["loss"].get("gamma_neg", cfg["loss"].get("gamma", 2.0)))
    beta = float(cfg["loss"].get("beta", 1.0))
    return DualFocalLoss(gamma_pos=gp, gamma_neg=gn, beta=beta)

@register("fcl_ungated")
def _build_fcl(cfg, num_classes: int):
    gf = float(cfg["loss"].get("gamma_focal", cfg["loss"].get("gamma", 2.0)))
    lam = float(cfg["loss"].get("lam", 1.0))
    return FCLUngated(num_classes=num_classes, gamma_focal=gf, lam=lam)

@register("safcl")
def _build_safcl(cfg, num_classes: int):
    gf = float(cfg["loss"].get("gamma_focal", 2.0))
    gc = float(cfg["loss"].get("gamma_cal", 2.0))
    lam = float(cfg["loss"].get("lam", 1.0))
    return SAFCL(num_classes=num_classes, gamma_focal=gf, gamma_cal=gc, lam=lam)

@register("safcl_nodetach")
def _build_safcl_nd(cfg, num_classes: int):
    gf = float(cfg["loss"].get("gamma_focal", 2.0))
    gc = float(cfg["loss"].get("gamma_cal", 2.0))
    lam = float(cfg["loss"].get("lam", 1.0))
    return SAFCLNoDetach(num_classes=num_classes, gamma_focal=gf, gamma_cal=gc, lam=lam)


@register("ce_brier")
def _build_ce_brier(cfg, num_classes: int):
    lam = float(cfg["loss"].get("lam", 1.0))
    smoothing = float(cfg["loss"].get("smoothing", 0.0))
    return CEBrier(num_classes=num_classes, lam=lam, label_smoothing=smoothing)

@register("ce_mmce")
def _build_ce_mmce(cfg, num_classes: int):
    lam = float(cfg["loss"].get("lam", 1.0))
    sigma = float(cfg["loss"].get("sigma", 0.4))
    smoothing = float(cfg["loss"].get("smoothing", 0.0))
    return CEMMCE(lam=lam, sigma=sigma, label_smoothing=smoothing)
