from torch import nn, tanh, sigmoid
import math
import torch
bias = (True, 20)
conv = nn.Conv2d(1 + 12, 4 * 12, 3, bias=True, padding="same")
# self.conv.bias[h_channels: 2*h_channels] auf 𝑏𝑓 ∼ log(𝒰([1, 𝑇max − 1])), 𝑏𝑖 = −𝑏 Tmax = horizon (oder periodendauer)

if bias[0]:
    with torch.no_grad():
        conv.bias[12: 2 * 12] = torch.log((bias[1] - 0) * torch.rand(12))

print(conv.bias)