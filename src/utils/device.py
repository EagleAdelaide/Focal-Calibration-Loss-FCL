import torch

def resolve_device(device: str) -> str:
    device = device.lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
