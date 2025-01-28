import torch.nn as nn


class ScaledTanh(nn.Module):
    def __init__(self, scale):
        super(ScaledTanh, self).__init__()
        self.scale = scale
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.scale * self.tanh(x)