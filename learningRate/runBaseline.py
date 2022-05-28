import argparse
import torch
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader, default_collate
import torch.optim as optim
from models import baseline, lateral, twoLayer, depthWise, skipConnection
import h5py
import matplotlib.pyplot as plt
import math
import os
from torch.optim.lr_scheduler import MultiStepLR


def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        f = h5py.File("../../data/wave/" + file, 'r')
        self.isTrain = isTrain
        self.data = f['data']['train'] if self.isTrain else f['data']['test']

    def __getitem__(self, item):
        return self.data[f'{item}'.zfill(3)][:, :, :]

    def __len__(self):
        return len(self.data)


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
            # x_channels = 1 if i == 0 else h_channels
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
            # latent = output[:, t]
        return output


def map(run):
    match run:
        case 1 | 2 | 3 | 4 | 5:
            return 0.002
        case 6 | 7 | 8 | 9 | 10:
            return 0.0018
        case 11 | 12 | 13 | 14 | 15:
            return 0.0016
        case 16 | 17 | 18 | 19 | 20:
            return 0.0014
        case 21 | 22 | 23 | 24 | 25:
            return 0.0012
        case 26 | 27 | 28 | 29 | 30:
            return 0.001
        case 31 | 32 | 33 | 34 | 35:
            return 0.0009
        case 36 | 37 | 38 | 39 | 40:
            return 0.0008
        case 41 | 42 | 43 | 44 | 45:
            return 0.0007
        case 46 | 47 | 48 | 49 | 50:
            return 0.0006


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_idx', type=int, default=4)
    args = parser.parse_args()
    run = args.run_idx
    hiddenSize = 8
    seq, modelName = Forecaster(hiddenSize, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device), "baseline"
    params = count_params(seq)
    batch_size = 32
    epochs = 10
    learningRate = map(run)
    dataloader = DataLoader(dataset=Wave("wave-5000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=lambda x: default_collate(x).to(device, torch.float))

    validation = DataLoader(dataset=Wave("wave-5000-90", isTrain=False), batch_size=batch_size, shuffle=True,
                            drop_last=True,
                            collate_fn=lambda x: default_collate(x).to(device, torch.float))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(seq.parameters(), lr=learningRate)
    scheduler = MultiStepLR(optimizer, milestones=[150, 200, 250, 300, 350, 400, 450, 500], gamma=0.8)
    # begin to train
    loss_plot_train, loss_plot_val = [], []

    for j in range(epochs):
        for i, images in enumerate(dataloader):
            input_images = images[:, :20, :, :]
            labels = images[:, 20:, :, :]
            output = seq(input_images, 70)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq.parameters(), 20)
            optimizer.step()
        loss_plot_train.append(loss.item())

        with torch.no_grad():
            for i, images in enumerate(validation):
                input_images = images[:, :20, :, :]
                labels = images[:, 20:, :, :]
                output = seq(input_images, 70)
                loss = criterion(output, labels)
            loss_plot_val.append(loss)

    # save model and test and train loss and parameters in txt file and python file with class
    if not os.path.exists("../trainedModels/wave/lr/baseline/run" + str(run)):
        os.makedirs("../trainedModels/wave/lr/baseline/run" + str(run))
    os.chdir("../trainedModels/wave/lr/baseline/run" + str(run))
    torch.save(seq.state_dict(), "model.pt")
    torch.save(loss_plot_train, "trainingLoss")
    torch.save(loss_plot_val, "validationLoss")

    # save config
    configuration = {"model": modelName,
                     "epochs": epochs,
                     "batchSize": batch_size,
                     "learningRate": learningRate,
                     "parameters": params,
                     "hiddenSize": hiddenSize,
                     "Loss": criterion,
                     "dataset": "mnist-5000-60"
                     }
    with open('configuration.txt', 'w') as f:
        print(configuration, file=f)
