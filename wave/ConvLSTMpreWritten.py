import torch
from torch import nn
from torch import cat, tanh, Tensor, sigmoid

class LSTM_cell(nn.Module):

    def __init__(self, x_channels, h_channels):
        super().__init__() # same?
        self.conv = nn.Conv2d(x_channels, 4 * h_channels, 1, bias=True, padding="same")

    def forward(self, x, h, c):
        z = torch.cat((x, h), dim=1) if x is not None else h
        i, f, o, g = self.conv(z).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) + tanh(g)
        h = sigmoid(o) + tanh(c)
        return h, c


class Sequence(nn.Module):

    def __init__(self, in_channels, h_channels, lateral_channels):
        super.__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.lateral_channels = lateral_channels
        self.transition = nn.Conv2d(self.in_channels + self.h_channels, self.lateral_channels, 3, padding="same")
        self.lstm1 = LSTM_cell(self.lateral_channels, self.h_channels)
        self.post = nn.Conv2d(self.h_channels, 1, 3, padding="same")

    def forward(self, x, steps=1):
        out = []
        h, c = torch.zeros(x.size(0), self.h_channels, 32, 32), torch.zeros(x.size(0), self.h_channels, 32, 32)
        for t in range(steps):
            input_transition = torch.cat((x, h), dim = 1)
            l = self.transition(input_transition)
            h, c = self.lstm1(l, h, c)
            out.append(self.post(h))
        return cat(out, dim =  1)