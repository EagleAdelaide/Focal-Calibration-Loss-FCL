import torch, math, os, json
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def safe_softmax(logits, dim=1):
    z = logits - logits.max(dim=dim, keepdim=True).values
    return F.softmax(z, dim=dim)

class ModelWithTemperature(nn.Module):
    def __init__(self, model, device='cuda'):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.device = device
    def set_temperature(self, val_loader, max_iter=50, lr=0.01):
        self.to(self.device)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter) if hasattr(optim, 'LBFGS') else optim.Adam([self.temperature], lr=lr)
        self.temperature.data = torch.ones_like(self.temperature)
        logits_list = []; labels_list = []
        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                logits_list.append(logits); labels_list.append(y)
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        if isinstance(optimizer, optim.LBFGS):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = nll_criterion(logits / self.temperature, labels)
                loss.backward(); return loss
            optimizer.step(closure)
        else:
            for _ in range(max_iter):
                optimizer.zero_grad(set_to_none=True)
                loss = nll_criterion(logits / self.temperature, labels)
                loss.backward(); optimizer.step()
        return float(self.temperature.detach().cpu())

def eval_logits(model, loader, device):
    model.eval()
    logits_list = []; labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_list.append(logits); labels_list.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)

def ece_histogram(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.zeros(1, device=probs.device)
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i+1])
            if m.any():
                bin_conf = conf[m].mean()
                bin_acc  = correct[m].mean()
                ece += (m.float().mean()) * torch.abs(bin_conf - bin_acc)
        return float(ece.item())

def adaece_equalfreq(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()
        N = conf.numel()
        if N == 0: return 0.0
        conf_sorted, idx = torch.sort(conf)
        correct_sorted = correct[idx]
        bin_sizes = [N // n_bins + (1 if x < (N % n_bins) else 0) for x in range(n_bins)]
        start = 0; ece = 0.0
        for bs in bin_sizes:
            if bs == 0: continue
            end = start + bs
            cbin = conf_sorted[start:end]
            abin = correct_sorted[start:end]
            bin_conf = cbin.mean(); bin_acc  = abin.mean()
            weight = float(bs) / float(N)
            ece += weight * float(torch.abs(bin_conf - bin_acc))
            start = end
        return float(ece)

def classwise_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15, num_classes: int = 10) -> float:
    with torch.no_grad():
        conf, pred = probs.max(dim=1)
        N = labels.numel(); total = 0.0
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        for c in range(num_classes):
            mask = (pred == c)
            Nc = int(mask.sum().item())
            if Nc == 0: continue
            conf_c = conf[mask]
            correct_c = (labels[mask] == c).float()
            ece_c = torch.zeros(1, device=probs.device)
            for i in range(n_bins):
                m = (conf_c > bins[i]) & (conf_c <= bins[i+1])
                if m.any():
                    bin_conf = conf_c[m].mean(); bin_acc  = correct_c[m].mean()
                    ece_c += (m.float().mean()) * torch.abs(bin_conf - bin_acc)
            total += (Nc / N) * float(ece_c.item())
        return float(total)

def metrics_from_logits(logits, labels, n_bins=15, num_classes=10, temperature=None):
    if temperature is not None:
        logits = logits / temperature
    probs = safe_softmax(logits, dim=1)
    acc = (probs.argmax(dim=1) == labels).float().mean().item()
    ece = ece_histogram(probs, labels, n_bins=n_bins)
    adae = adaece_equalfreq(probs, labels, n_bins=n_bins)
    cwe = classwise_ece(probs, labels, n_bins=n_bins, num_classes=num_classes)
    nll = nn.functional.cross_entropy(logits, labels, reduction='mean').item()
    return acc, ece, adae, cwe, nll, probs
