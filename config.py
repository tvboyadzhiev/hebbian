from pathlib import Path

import torch

bin_root = Path(*Path(__file__).absolute().parts[:-1]) / "bin"
data_root = bin_root / "data"
exp_root = bin_root / "exp"

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
