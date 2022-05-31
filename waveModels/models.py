import torch
from torch import nn, tanh, sigmoid

class baseline(nn.Module):

    def __init__(self, x_channels, h_channels, k = 1):
        super(baseline, self).__init__()
        self.conv = nn.Conv2d(x_channels + h_channels, 4 * h_channels, k, bias=True, padding="same")
        #self.conv.bias[h_channels: 2*h_channels] auf 𝑏𝑓 ∼ log(𝒰([1, 𝑇max − 1])), 𝑏𝑖 = −𝑏 Tmax = horizon (oder periodendauer)

        self.conv.bias[h_channels: 2 * h_channels]

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



class Forecaster(nn.Module):
    '''
    Encoder-Forecaster network architecture.
    An input sequence of arbitrary length is processed by the encoder.
    Then the state of the encoder is used to initialise the decoder states.
    Then the decoder state is projected into the future for a desired number of time steps.
    '''

    def __init__(self, h_channels: int, lstm_block: callable, num_blocks: int = 1, lstm_kwargs={}):
        '''
        :param h_channels: Number of hidden channels per layer (e.g. 12)
        :param lstm_block: A nn.Module that computes a single step of h, c = LSTM(x, h, c)
        :param num_blocks: Number of layers in the encoder/decoder network (e.g. 2)
        :param kwargs: Additional arguments to provide to the LSTM block. (e.g. lateral_channels)
        '''
        super().__init__()
        self.h_channels = h_channels
        self.init = nn.Conv2d(1, h_channels, 1)

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(num_blocks):
            x_channels = 0 if i == 0 else h_channels
            #x_channels = 1 if i == 0 else h_channels
            self.encoder_layers.add_module(f'block_{i}', lstm_block(h_channels, h_channels, **lstm_kwargs))
            self.decoder_layers.add_module(f'block_{i}', lstm_block(x_channels, h_channels, **lstm_kwargs))

        self.read = nn.Conv2d(h_channels, 1, 1)

    def forward(self, x, horizon: int = 1):
        '''
        Processes a batch of videos and generates a prediction.
        :param x: A batch of videos. Expected shape is (batch, time, height, width).
        :param horizon: The number of time-steps to predict into the future.
        :output:
        '''
        batch_size, context, height, width = x.shape
        assert horizon >= 1, 'Predictions will only be generated for horizon >= 1'
        h = [torch.zeros((batch_size, self.h_channels, height, width), device=x.device)
             for i in range(len(self.encoder_layers))]
        c = [torch.zeros((batch_size, self.h_channels, height, width), device=x.device)
             for i in range(len(self.encoder_layers))]

        for t in range(context):
            for i, layer in enumerate(self.encoder_layers):
                z = self.init(x[:, t].unsqueeze(1)) if i == 0 else h[i - 1]
                h[i], c[i] = layer(z, h[i], c[i])

        latent = None  # could be h[t].copy() alternatively,
        # this would feed the original context vector into every closed loop step!
        output = torch.zeros((batch_size, horizon, height, width), device=x.device)

        for t in range(horizon):
            for i, layer in enumerate(self.decoder_layers):
                z = latent if i == 0 else h[i - 1]
                h[i], c[i] = layer(z, h[i], c[i])
            output[:, t] = self.read(h[-1]).squeeze()
            #latent = output[:, t]
        return output