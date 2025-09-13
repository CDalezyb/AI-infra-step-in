import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_size):
        super(SwiGLU, self).__init__()
        self.linear = nn.Linear(d_model, 2 * hidden_size)

    def forward(self, x: Tensor):
        x = self.linear(x)

        a, b = torch.spilt(x, x.size(-1) / 2, dim=-1)

        return a * F.silu(b)
