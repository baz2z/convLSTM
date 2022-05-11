import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
import h5py
import math
from mMnistModelBaseline import *
import numpy



class mMnist(Dataset):
    def __init__(self):
        self.data = numpy.load("../../data/movingMNIST/movingmnistdata.npz")["arr_0"].reshape(-1, 20, 64, 64)

    def __getitem__(self, item):
        return self.data[item,:,:,:]

    def __len__(self):
        return self.data.shape[0]

def visualize_wave(data, row, nbrImages = 10, fromStart = True, ):
    # data = [sequence, width, height]
    for i in range(nbrImages):
        plt.subplot(math.ceil(nbrImages**0.5), math.ceil(nbrImages**0.5), i + 1)
        image = data[i,:,:] if fromStart else data[len(data)-nbrImages + i,:,:]
        plt.imshow(image, cmap='gray')

    plt.savefig("prediction" + str(row))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    hidden_size = 5
    epochs = 50
    dataloader = DataLoader(dataset=mMnist(), batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn = lambda x: default_collate(x).to(device,torch.float))
    seq = Sequence(1, hidden_size).to(device)
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
            optimizer.step()
        loss_plot.append(loss.item())
        print(loss.item())

    plt.plot(loss_plot)
    plt.savefig("lossPlotOneFuture")


    with torch.no_grad():
        visData = iter(dataloader).__next__()
        pred = seq(visData[:,:10,:,:], future=10).detach().cpu().numpy()
        visualize_wave(pred[0,:,:,:], 1, nbrImages=20, fromStart=False)
        visualize_wave(pred[1,:,:,:], 2, nbrImages=20, fromStart=False)
        visualize_wave(pred[2,:,:,:], 3, nbrImages=20, fromStart=False)




