import torch
from torch import nn
from torch import cat, tanh, Tensor, sigmoid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM_cell(nn.Module):

    def __init__(self, x_channels, h_channels):
        super(LSTM_cell, self).__init__()
        self.conv = nn.Conv2d(x_channels + h_channels, 4 * h_channels, 3, bias=True, padding="same")

    def forward(self, x, h, c):
        z = torch.cat((x, h), dim=1)
        i, f, o, g = self.conv(z).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) + tanh(g)
        h = sigmoid(o) + tanh(c)
        return h, c


class Sequence(nn.Module):

    def __init__(self, in_channels, h_channels):
        super(Sequence, self).__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.lstm1 = LSTM_cell(self.in_channels, self.h_channels)
        self.post = nn.Conv2d(self.h_channels, 1, 3, padding="same")

    def forward(self, x, future=0):
        out = []
        h, c = torch.zeros(x.size(0), self.h_channels, 64, 64, device=device), torch.zeros(x.size(0), self.h_channels, 64, 64, device=device)
        for i, x in enumerate(x.chunk(x.size(1), dim=1)):
            h, c = self.lstm1(x, h, c)
            x = self.post(h)
            out.append(x)
        for i in range(future):
            h, c = self.lstm1(x, h, c)
            x = self.post(h)
            out.append(x)

        return cat(out, dim =  1)

