import torch as th
from torch import nn
from depthWise import depthWise

def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class LSTMCell2d(nn.Module):
    '''LSTMCell for 2d inputs'''

    def __init__(self,
                 x_channels: int,
                 h_channels: int,
                 k: int = 1,
                 rnn_act: callable = None):
        '''
        :param x_channels: Input channels
        :param h_channels: Latent state channels
        :param k: Size of the convolution kernel. Default to pointwise.
        :param rnn_act: (Optional) Activation function to use inside the LSTM. Default to tanh.
        '''
        super().__init__()

        self.phi = rnn_act() if rnn_act is not None else nn.Tanh()
        self.conv = nn.Conv2d(
            in_channels=x_channels + h_channels,
            out_channels=4 * h_channels,
            kernel_size=k,
            padding='same')

    def forward(self, x, h, c) -> th.Tensor:
        '''
        LSTM forward pass
        :param x: Input
        :param h: Hidden state
        :param c: Cell state
        '''
        z = th.cat((x, h), dim=1) if x is not None else h
        i, f, o, g = self.conv(z).chunk(chunks=4, axis=1)
        c = th.sigmoid(f) * c + th.sigmoid(i) * self.phi(g)
        h = th.sigmoid(o) * self.phi(c)
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
        h = [th.zeros((batch_size, self.h_channels, height, width), device=x.device)
             for i in range(len(self.encoder_layers))]
        c = [th.zeros((batch_size, self.h_channels, height, width), device=x.device)
             for i in range(len(self.encoder_layers))]

        for t in range(context):
            for i, layer in enumerate(self.encoder_layers):
                z = self.init(x[:, t].unsqueeze(1)) if i == 0 else h[i - 1]
                h[i], c[i] = layer(z, h[i], c[i])

        latent = None  # could be h[t].copy() alternatively,
        # this would feed the original context vector into every closed loop step!
        output = th.zeros((batch_size, horizon, height, width), device=x.device)

        for t in range(horizon):
            for i, layer in enumerate(self.decoder_layers):
                z = latent if i == 0 else h[i - 1]
                h[i], c[i] = layer(z, h[i], c[i])
            output[:, t] = self.read(h[-1]).squeeze()
        return output


if __name__ == '__main__':
    net = Forecaster(6, depthWise, num_blocks=2, lstm_kwargs={"lateral_channels_multipl": 1})
    print(net)
    print(f'Total number of trainable parameters: {count_params(net)}')

    x = th.randn((32, 20, 64, 64))  # dummy input for testing
    y = net(x, horizon=10)
    #loss = (x - y).pow(2).mean()
    #loss.backward()
