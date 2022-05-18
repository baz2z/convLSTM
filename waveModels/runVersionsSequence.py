import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
import h5py
#from baseline import *
#from lateral import *
#from skipConnection import *
#from twoLayer import *
from models import *
import math
import argparse

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

class Sequence(nn.Module):

    def __init__(self, in_channels, h_channels):
        super(Sequence, self).__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.lstm1 = baseline(self.in_channels, self.h_channels)
        self.post = nn.Conv2d(self.h_channels, 1, 3, padding="same")

    def forward(self, x, future=0):
        out = []
        h, c = torch.zeros(x.size(0), self.h_channels, 32, 32, device=device), torch.zeros(x.size(0), self.h_channels, 32, 32, device=device)
        for i, x in enumerate(x.chunk(x.size(1), dim=1)):
            h, c = self.lstm1(x, h, c)
            x = self.post(h)
            out.append(x)
        for i in range(future):
            h, c = self.lstm1(x, h, c)
            x = self.post(h)
            out.append(x)

        return cat(out, dim =  1)

def visualize_wave(data, row, nbrImages = 10, fromStart = True, ):
    # data = [sequence, width, height]
    for i in range(nbrImages):
        plt.subplot(math.ceil(nbrImages**0.5), math.ceil(nbrImages**0.5), i + 1)
        image = data[i,:,:] if fromStart else data[len(data)-nbrImages + i,:,:]
        plt.imshow(image, cmap='gray')

    plt.savefig("prediction" + str(row))


def map_run(n):
    model = "baseline"
    if n == 0:
        model = "baseline"
    elif n == 1:
        model = "lateral"
    elif n == 2:
        model = "twoLayer"
    elif n == 3:
        model = "depthWise"
    elif n == 4:
        model = "skipConnections"
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_idx', type=int)
    args = parser.parse_args()
    run = args.run_idx
    model = map_run(run)

    batch_size = 32
    hidden_size = 6
    epochs = 250
    dataloader = DataLoader(dataset=Wave("wave1000-40"), batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn = lambda x: default_collate(x).to(device,torch.float))
    seq = Sequence(1, hidden_size, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=0.0008)
    # begin to train
    loss_plot = []
    for j in range(epochs):
        for i, images in enumerate(dataloader):
            input_images = images[:,:-1,:,:]
            labels = images[:,1:,:,:]
            output = seq(input_images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            # here maybe clipping with 2 or more
            torch.nn.utils.clip_grad_norm_(seq.parameters(), 20)
            optimizer.step()
        loss_plot.append(loss.item())
        print(loss.item())
    plt.yscale("log")
    plt.plot(loss_plot)
    plt.savefig("lossPlot")


    with torch.no_grad():
        visData = iter(dataloader).__next__()
        pred = seq(visData[:,:30,:,:], future=10).detach().cpu().numpy()
        visualize_wave(pred[0,:,:,:], 1, nbrImages=20, fromStart=False)
        visualize_wave(pred[1,:,:,:], 2, nbrImages=20, fromStart=False)
        visualize_wave(pred[2,:,:,:], 3, nbrImages=20, fromStart=False)
