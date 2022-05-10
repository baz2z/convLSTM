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
from depthWise import *
import math

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
    epochs = 20
    dataloader = DataLoader(dataset=Wave("wave1000-40"), batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn = lambda x: default_collate(x).to(device,torch.float))
    seq = Sequence(1, hidden_size, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=0.0008)
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
            # here maybe clipping with 2 or more
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
