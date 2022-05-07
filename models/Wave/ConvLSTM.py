import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
import h5py
import math
import argparse

class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        f = h5py.File(file, 'r')
        self.isTrain = isTrain
        self.data = f['data']['train'] if self.isTrain else f['data']['test']

    def __getitem__(self, item):
        return self.data[f'{item}'.zfill(3)][:,:,:]

    def __len__(self):
        return len(self.data)





class LSTM_cell(torch.nn.Module):

    def __init__(self, in_channels=10, out_channels=20, kernel_size = 3):
        super(LSTM_cell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kernel_size
        # Combined Gates
        self.W = nn.Conv2d(self.in_channels, 4 * self.out_channels, self.kern_size, bias=True, padding="same")
        self.U = nn.Conv2d(self.out_channels, 4 * self.out_channels, self.kern_size, padding="same")


    def forward(self, x, tuple_in):
        # x = [batch, firstImageSequenc(1), in_channels, 32, 32]
        (h, c_prev) = tuple_in
        # gates
        gates = self.W(x) + self.U(h) #n, 1, 4 * out_channels, width, height
        i_t, f_t, g_t, o_t = gates.chunk(chunks = 4, dim=1)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t) #n, 1, out_channels, width, height

        return h_t, c_t




class Sequence(nn.Module):
    def __init__(self, hiddenSize):
        super(Sequence, self).__init__()
        self.hiddenSize = hiddenSize
        self.rnn1 = LSTM_cell(1, self.hiddenSize)
        self.final = nn.Conv2d(self.hiddenSize, 1, 3, padding="same")

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hiddenSize, 32, 32 , device=device)
        c_t = torch.zeros(input.size(0), self.hiddenSize, 32, 32, device=device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            # input_t = [batch, sequence, width, height]
            # use dim1 as in_channels
            h_t, c_t = self.rnn1(input_t, (h_t, c_t))
            output = self.final(h_t)
            outputs += [output]


        # if we should predict the future
        for i in range(future):
            h_t, c_t = self.rnn1(output, (h_t, c_t))
            output = self.final(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def visualize_wave(data, nbrImages = 10, fromStart = True):
    # data = [sequence, width, height]
    for i in range(nbrImages):
        plt.subplot(math.ceil(nbrImages**0.5), math.ceil(nbrImages**0.5), i + 1)
        image = data[i,:,:] if fromStart else data[len(data)-nbrImages + i,:,:]
        plt.imshow(image, cmap='gray')
    plt.show()



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 6
    hidden_size = 5
    epochs = 4
    dataloader = DataLoader(dataset=Wave("wave10-40"), batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn = lambda x: default_collate(x).to(device,torch.float))
    seq = Sequence(hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=0.001)
    # begin to train
    loss_plot = []
    for j in range(epochs):
        for i, images in enumerate(dataloader):
            input_images = images[:,:-21,:,:]
            labels = images[:,1:,:,:]
            output = seq(input_images, future = 20)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_plot.append(loss.item())
        print(loss.item())

    #plt.plot(loss_plot)
    #plt.show()


"""
    with torch.no_grad():
        pred = seq(data[:,:30,:,:], future=10)
        visualize_wave(pred[0,:,:,:], nbrImages=20, fromStart=False)
        visualize_wave(pred[1,:,:,:], nbrImages=20, fromStart=False)
        visualize_wave(pred[2,:,:,:], nbrImages=20, fromStart=False)
"""


