import torch.nn as nn

approx_gelu = lambda: nn.GELU(approximate="tanh")
