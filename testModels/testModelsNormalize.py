import argparse

from models import baseline, lateral, skipConnection, depthWise, twoLayer, Forecaster
import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
from torch.utils.data import Dataset, DataLoader, default_collate
from torch import nn



class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        f = h5py.File("../../data/wave/" + file, 'r')
        self.isTrain = isTrain
        self.data = f['data']['train'] if self.isTrain else f['data']['test']
        means, stds = [], []
        for i in range(len(self.data)):
            data = self.data[f'{i}'.zfill(3)][:, :, :]
            means.append(numpy.mean(data))
            stds.append(numpy.std(data))
        self.mu = numpy.mean(means)
        self.std = numpy.mean(stds)


    def __getitem__(self, item):
        data = self.data[f'{item}'.zfill(3)][:, :, :]
        data = (data - self.mu) / self.std
        return data

    def __len__(self):
        return len(self.data)


class mMnist(Dataset):
    def __init__(self, data):
        self.data = numpy.load("../../data/movingMNIST/" + data + ".npz")["arr_0"].reshape(-1, 60, 64, 64)

    def __getitem__(self, item):
        return self.data[item, :, :, :]

    def __len__(self):
        return self.data.shape[0]


def mapModel(model):
    match model:
        case "baseline":
            return Forecaster(8, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device)
        case "lateral":
            return Forecaster(12, lateral, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        case "twoLayer":
            return Forecaster(12, twoLayer, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        case "skip":
            return Forecaster(12, skipConnection, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        case "depthWise":
            return Forecaster(8, depthWise, num_blocks=2, lstm_kwargs={'lateral_channels_multipl': 6}).to(device)


def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def visualize_wave(imgs):
    t, w, h = imgs.shape
    for i in range(t):
        plt.subplot(math.ceil(t ** 0.5), math.ceil(t ** 0.5), i + 1)
        plt.title(i, fontsize=9)
        plt.axis("off")
        image = imgs[i, :, :]
        plt.imshow(image, cmap="gray")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="baseline",
                    choices=["baseline", "lateral", "twoLayer", "skip", "depthWise"])
parser.add_argument('--mode', type=str, default="horizon-20-40")

args = parser.parse_args()
modelName = args.model
mode = args.mode
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = "wave"
model = mapModel(modelName)
context = 20
horizon = 40

datasetLoader = Wave("wave-5000-90")
dataloader = DataLoader(dataset=datasetLoader, batch_size=25, shuffle=False, drop_last=True,
                        collate_fn=lambda x: default_collate(x).to(device, torch.float))

#os.chdir("../trainedModels/" + dataset + "/" + mode + "/" + modelName )
os.chdir("C:/Users/Sebastian/Desktop/remote/convLSTM/trainedModels/wave/norm/baseline/withNormalize")

# calculated train loss on new dataset and average the loss


modelsLoss = []
for runNbr in range(5):
    runNbr = runNbr + 1
    os.chdir(f'./run{runNbr}')
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()
    runningLoss = []
    with torch.no_grad():
        for i, images in enumerate(dataloader):
            input_images = images[:, :context, :, :]
            labels = images[:, context:context + horizon, :, :]
            output = model(input_images, horizon)
            output_not_normalized = (output * datasetLoader.std) + datasetLoader.mu
            labels_not_normalized = (labels * datasetLoader.std) + datasetLoader.mu
            loss = criterion(output_not_normalized, labels_not_normalized)
            runningLoss.append(loss.cpu())
        modelsLoss.append(numpy.mean(runningLoss))
        print(numpy.mean(runningLoss))
    os.chdir("../")

finalLoss = numpy.mean(modelsLoss)

print(os.getcwd())
configuration = {f'{modelName}loss': finalLoss}
print(configuration)
with open('configuration.txt', 'w') as f:
    print(configuration, file=f)

