from models import baseline, lateral, skipConnection, depthWise, twoLayer, Forecaster
import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
from torch.utils.data import Dataset, DataLoader, default_collate

def visualize_wave(imgs):
    t, w, h = imgs.shape
    for i in range(t):
        plt.subplot(math.ceil(t ** 0.5), math.ceil(t ** 0.5), i + 1)
        plt.title(i, fontsize=9)
        plt.axis("off")
        image = imgs[i,:,:]
        plt.imshow(image, cmap="gray")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


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

class mMnist(Dataset):
    def __init__(self, data):
        self.data = numpy.load("../../data/movingMNIST/" + data + ".npz")["arr_0"].reshape(-1, 60, 64, 64)

    def __getitem__(self, item):
        return self.data[item,:,:,:]

    def __len__(self):
        return self.data.shape[0]

def mapModel(model, hiddenSize, lateralSize):
    match model:
        case "baseline":
            return Forecaster(hiddenSize, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device)
        case "lateral":
            return Forecaster(hiddenSize, lateral, num_blocks=2, lstm_kwargs={'lateral_channels': lateralSize}).to(device)
        case "twoLayer":
            return Forecaster(hiddenSize, twoLayer, num_blocks=2, lstm_kwargs={'lateral_channels': lateralSize}).to(device)
        case "skip":
            return Forecaster(hiddenSize, skipConnection, num_blocks=2, lstm_kwargs={'lateral_channels': lateralSize}).to(device)
        case "depthWise":
            return Forecaster(hiddenSize, depthWise, num_blocks=2, lstm_kwargs={'lateral_channels_multipl': lateralSize}).to(device)


def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def mse(values):
    return ((values - values.mean(axis = 0))**2).mean(axis = 0)


def mostSignificantPixel(imgs):
    # images of shape: frames, width, height
    f, w, h = imgs.shape
    msp = [(0, 0), -1000]
    for i in range(w):
        for j in range(h):
            values = numpy.array([])
            for k in range(f):
                value = imgs[k, i, j]
                values = numpy.append(values, value)
            var = mse(values)
            if var > msp[1]:
                msp = [(i, j), var]
    return msp[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = "AdamW"
modelName = "baseline"
model = mapModel(modelName, 6, 12)
run = "5"
horizon = 40

dataloader = DataLoader(dataset=Wave("wave-5000-90"), batch_size=10, shuffle=False, drop_last=False,
                        collate_fn=lambda x: default_collate(x).to(device, torch.float))

os.chdir("../trainedModels/wave/" + mode + "/" + modelName + "/" + "run" + run)

# model

model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
print(count_params(model))

# loss
trainLoss = torch.load("trainingLoss", map_location=device)
valLoss = torch.load("validationLoss", map_location=device)
print(trainLoss[-1])

plt.yscale("log")
plt.plot(trainLoss, label="trainLoss")
plt.plot(valLoss, label="valLoss")
plt.legend()
plt.show()

# example wave
visData = iter(dataloader).__next__()
pred = model(visData[:, :20, :, :], horizon=70).detach().cpu().numpy()


sequence = 2
# for one pixel

w, h = mostSignificantPixel(pred[sequence, :, :, :])
groundTruth = visData[sequence, 20:, int(w / 2), int(h / 2)]
prediction = pred[sequence, :, int(w / 2), int(h / 2)]
plt.plot(groundTruth, label="groundTruth")
plt.plot(prediction, label="prediction")
plt.legend()
plt.title(f'{(w, h)}')
plt.show()

# for entire sequence
visualize_wave(pred[sequence, :, :, :])
visualize_wave(visData[sequence, :20, :, :])
f = open("configuration.txt", "r")
print(f.read())