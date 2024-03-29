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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy

def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class mMnist(Dataset):
    def __init__(self, data):
        self.data = numpy.load("../../data/movingMNIST/" + data + ".npz")["arr_0"].reshape(-1, 60, 64, 64)

    def __getitem__(self, item):
        return self.data[item,:,:,:]

    def __len__(self):
        return self.data.shape[0]

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
        self.read2 = nn.Sigmoid() # for BCE loss

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
            output[:, t] = self.read2(self.read(h[-1]).squeeze())
            #latent = output[:, t]
        return output


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_idx', type=int, default=1)
    args = parser.parse_args()
    run = args.run_idx
    hiddenSize = 14
    seq, modelName = Forecaster(hiddenSize, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device), "baseline"
    params = count_params(seq)
    batch_size = 36
    epochs = 1
    learningRate = 0.001

    dataloader = DataLoader(dataset=mMnist("mnist-5000-60-const"), batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=lambda x: default_collate(x).to(device, torch.float))

    validation = DataLoader(dataset=mMnist("mnist-100-60-const"), batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=lambda x: default_collate(x).to(device, torch.float))
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(seq.parameters(), lr=learningRate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    # begin to train
    loss_plot_train, loss_plot_val = [], []

    for j in range(epochs):
        for i, images in enumerate(dataloader):
            input_images = images[:, :10, :, :]
            labels = images[:, 10:11, :, :]
            output = seq(input_images, 1)
            b, t, w, h = output.shape
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq.parameters(), 20)
            optimizer.step()
        scheduler.step()
        loss_plot_train.append(loss.item())

        with torch.no_grad():
            for i, images in enumerate(validation):
                input_images = images[:, :10, :, :]
                labels = images[:, 10:11, :, :]
                output = seq(input_images, 1)
                b, t, w, h = output.shape
                loss = criterion(output, labels)
            loss_plot_val.append(loss)


    # save model and test and train loss and parameters in txt file and python file with class
    path = "../trainedModels/mnist/baseline/run" + str(run)
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
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
                     "dataset": "mnist-5000-60-const"
                     }
    with open('configuration.txt', 'w') as f:
        print(configuration, file=f)


