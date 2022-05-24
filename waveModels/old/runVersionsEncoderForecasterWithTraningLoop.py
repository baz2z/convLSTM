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
        return self.data[f'{item}'.zfill(3)][:,:,:]

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
            #latent = output[:, t]
        return output



def visualize_wave(imgs, modelName):
    t, w, h = imgs.shape
    for i in range(t):
        plt.subplot(math.ceil(t ** 0.5), math.ceil(t ** 0.5), i + 1)
        plt.title(i)
        image = imgs[i,:,:]
        plt.imshow(image, cmap="gray")
    plt.subplots_adjust(hspace=1.5)
    plt.savefig("prediction_" + modelName)

def map_run(n):
    if n == 0:
        seq = Forecaster(12, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device)
        modelName = "baseline"
    elif n == 1:
        seq = Forecaster(12, lateral, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        modelName = "lateral"
    elif n == 2:
        seq = Forecaster(12, twoLayer, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        modelName = "twoLayer"
    elif n == 3:
        seq = Forecaster(12, skipConnection, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        modelName = "skipConnection"
    elif n == 4:
        seq = Forecaster(12, depthWise, num_blocks=2, lstm_kwargs={'lateral_channels_multipl': 8}).to(device)
        modelName = "depthWise"
    print(seq)
    print(f'Total number of trainable parameters: {count_params(seq)}')
    return seq, modelName

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_idx', type=int, default=0)
    args = parser.parse_args()
    run = args.run_idx
    seq, modelName = map_run(run)

    batch_size = 32
    epochs = 1
    dataloader = DataLoader(dataset=Wave("testWave"), batch_size=batch_size, shuffle=True, drop_last=False,
                            collate_fn=lambda x: default_collate(x).to(device, torch.float))

    validation = DataLoader(dataset=Wave("testWave", isTrain=False), batch_size=batch_size, shuffle=True, drop_last=False,
                            collate_fn=lambda x: default_collate(x).to(device, torch.float))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=0.0001)
    # begin to train
    loss_plot_train, loss_plot_val = [], []

    for j in range(epochs):
        for i, images in enumerate(dataloader):
            input_images = images[:, :20, :, :]
            labels = images[:, 20:, :, :]
            output = seq(input_images, 21)
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
                output = seq(input_images, 21)
                loss = criterion(output, labels)
            loss_plot_val.append(loss)

    plt.yscale("log")
    plt.plot(loss_plot_train, label = "trainLoss")
    plt.plot(loss_plot_val, label = "valLoss")
    plt.legend()
    plt.savefig("lossPlot_" + modelName)


    with torch.no_grad():
        visData = iter(dataloader).__next__()
        pred = seq(visData[:, :20, :, :], horizon = 21).detach().cpu().numpy()
        w, h = pred.shape[2], pred.shape[3]
        groundTruth = visData[0, 20:, int(w/2), int(h/2)]
        prediction =  pred[0, :, int(w/2), int(h/2)]
        plt.plot(groundTruth, label = "groundTruth")
        plt.plot(prediction, label = "prediction")
        plt.legend()
        plt.show()
        visualize_wave(pred[0, :, :, :], modelName)


    # save model and test and train loss and parameters in txt file and python file with class

    torch.save(seq.state_dict(), "../trainedModels/wave/horizon-20-40/baseline/baseline.pt")


