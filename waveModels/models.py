import torch
from torch import nn, tanh, sigmoid

class baseline(nn.Module):

    def __init__(self, x_channels, h_channels, k = 1):
        super(baseline, self).__init__()
        self.conv = nn.Conv2d(x_channels + h_channels, 4 * h_channels, k, bias=True, padding="same")

    def forward(self, x, h, c):
        z = torch.cat((x, h), dim=1) if x is not None else h
        i, f, o, g = self.conv(z).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) + tanh(g)
        h = sigmoid(o) + tanh(c)
        return h, c



class lateral(nn.Module):

    def __init__(self, x_channels, h_channels, lateral_channels):
        super(lateral, self).__init__()
        self.transition = nn.Conv2d(x_channels + h_channels, lateral_channels, 3 ,bias=True, padding="same")
        self.conv = nn.Conv2d(lateral_channels, 4 * h_channels, 1, bias=True, padding="same")

    def forward(self, x, h, c):
        z = torch.cat((x, h), dim=1)  if x is not None else h
        l = self.transition(z)
        i, f, o, g = self.conv(l).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) + tanh(g)
        h = sigmoid(o) + tanh(c)
        return h, c



class twoLayer(nn.Module):

    def __init__(self, x_channels, h_channels, lateral_channels):
        super(twoLayer, self).__init__()
        self.transition = nn.Conv2d(x_channels + h_channels, lateral_channels, 3 ,bias=True, padding="same")
        self.transition_deep = nn.Conv2d(lateral_channels, lateral_channels, 3, bias = True, padding="same")
        self.conv = nn.Conv2d(lateral_channels, 4 * h_channels, 1, bias=True, padding="same")

    def forward(self, x, h, c):
        z = torch.cat((x, h), dim=1) if x is not None else h
        l = self.transition(z)
        l_deep = self.transition_deep(l)
        i, f, o, g = self.conv(l_deep).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) + tanh(g)
        h = sigmoid(o) + tanh(c)
        return h, c



class skipConnection(nn.Module):

    def __init__(self, x_channels, h_channels, lateral_channels):
        super(skipConnection, self).__init__()
        self.transition = nn.Conv2d(x_channels + h_channels, lateral_channels, 3 ,bias=True, padding="same")
        self.conv = nn.Conv2d(lateral_channels + h_channels, 4 * h_channels, 1, bias=True, padding="same")

    def forward(self, x, h, c):
        z = torch.cat((x, h), dim=1) if x is not None else h
        l = self.transition(z)
        lWithSkip = torch.cat((l, h), dim = 1)
        i, f, o, g = self.conv(lWithSkip).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) + tanh(g)
        h = sigmoid(o) + tanh(c)
        return h, c


    
class depthWise(nn.Module):

    def __init__(self, x_channels, h_channels, lateral_channels_multipl):
        super(depthWise, self).__init__()
        self.transition = nn.Conv2d(x_channels + h_channels, lateral_channels_multipl * (x_channels + h_channels), 3, bias=True, padding="same", groups=(x_channels + h_channels))
        self.conv = nn.Conv2d(lateral_channels_multipl * (x_channels + h_channels), 4 * h_channels, 1, bias=True, padding="same")

    def forward(self, x, h, c):
        z = torch.cat((x, h), dim=1)  if x is not None else h
        l = self.transition(z)
        i, f, o, g = self.conv(l).chunk(chunks = 4, axis = 1)
        c = sigmoid(f) * c + sigmoid(i) + tanh(g)
        h = sigmoid(o) + tanh(c)
        return h, c