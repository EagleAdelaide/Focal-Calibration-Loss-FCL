import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep fast kernels; determinism can be enabled if needed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
