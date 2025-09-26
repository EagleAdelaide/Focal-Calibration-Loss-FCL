import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_softmax(logits, dim=1):
    z = logits - logits.max(dim=dim, keepdim=True).values
    return F.softmax(z, dim=dim)

class BrierScore(nn.Module):
    def forward(self, input, target):
        target = target.view(-1,1)
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target, 1)
        pt = safe_softmax(input, dim=1)
        return ((one_hot - pt).pow(2).sum(dim=1)).mean()

class BrierCombo(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    def forward(self, input, target):
        ce = F.cross_entropy(input, target)
        target = target.view(-1,1)
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target, 1)
        pt = safe_softmax(input, dim=1)
        brier = ((one_hot - pt).pow(2).sum(dim=1)).mean()
        return ce + self.gamma * brier

class FocalLoss(nn.Module):
    def __init__(self, gamma=5.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.reduction = reduction
    def forward(self, input, target):
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1).gather(1, target).view(-1)
        pt = logpt.exp().clamp_min(1e-6)
        loss = - (1-pt).pow(self.gamma) * logpt
        return loss.mean() if self.reduction=="mean" else loss.sum()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, device='cuda'):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.device = device
    def forward(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device).long()
        target_prob_dist = torch.full(size=(targets.size(0), self.cls),
                                      fill_value=self.smoothing / (self.cls - 1), device=self.device)
        target_prob_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        return F.kl_div(inputs.log_softmax(dim=-1), target_prob_dist, reduction='batchmean')
