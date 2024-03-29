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
        image = imgs[i, :, :]
        plt.imshow(image, cmap="gray")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        f = h5py.File("../../../../../../work2/butz1/svolz67/data/wave/"+ file, 'r')
        self.isTrain = isTrain
        self.data = f['data']['train'] if self.isTrain else f['data']['test']


    def __getitem__(self, item):
        data = self.data[f'{item}'.zfill(3)][:, :, :]
        #data = (data - self.mu) / self.std
        return data

    def __len__(self):
        return len(self.data)



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



def mapParas(modelName, multiplier, paramsIndex):
    modelParams = (0, 0)

    if modelName == "baseline":
        if multiplier == 1:
            match paramsIndex:
                case 1:
                    modelParams = (4, 1)
                case 2:
                    modelParams = (10, 1)
                case 3:
                    modelParams = (14, 1)
    elif modelName == "lateral":
        if multiplier == 0.5:
            match paramsIndex:
                case 1:
                    modelParams = (10, 5)
                case 2:
                    modelParams = (24, 12)
                case 3:
                    modelParams = (36, 18)
        if multiplier == 1:
            match paramsIndex:
                case 1:
                    modelParams = (8, 8)
                case 2:
                    modelParams = (18, 18)
                case 3:
                    modelParams = (25, 25)
        if multiplier == 2:
            match paramsIndex:
                case 1:
                    modelParams = (6, 12)
                case 2:
                    modelParams = (13, 26)
                case 3:
                    modelParams = (18, 36)
    elif modelName == "twoLayer":
        if multiplier == 0.5:
            match paramsIndex:
                case 1:
                    modelParams = (10, 5)
                case 2:
                    modelParams = (22, 11)
                case 3:
                    modelParams = (32, 16)
        if multiplier == 1:
            match paramsIndex:
                case 1:
                    modelParams = (6, 6)
                case 2:
                    modelParams = (15, 15)
                case 3:
                    modelParams = (21, 21)
        if multiplier == 2:
            match paramsIndex:
                case 1:
                    modelParams = (4, 8)
                case 2:
                    modelParams = (9, 18)
                case 3:
                    modelParams = (13, 26)
    elif modelName == "skip":
        if multiplier == 0.5:
            match paramsIndex:
                case 1:
                    modelParams = (10, 5)
                case 2:
                    modelParams = (22, 11)
                case 3:
                    modelParams = (30, 15)
        if multiplier == 1:
            match paramsIndex:
                case 1:
                    modelParams = (7, 7)
                case 2:
                    modelParams = (16, 16)
                case 3:
                    modelParams = (23, 23)
        if multiplier == 2:
            match paramsIndex:
                case 1:
                    modelParams = (5, 10)
                case 2:
                    modelParams = (12, 24)
                case 3:
                    modelParams = (17, 34)
    elif modelName == "depthWise":
        if multiplier == 1:
            match paramsIndex:
                case 1:
                    modelParams = (12, 1)
                case 2:
                    modelParams = (28, 1)
                case 3:
                    modelParams = (41, 1)
        if multiplier == 2:
            match paramsIndex:
                case 1:
                    modelParams = (8, 2)
                case 2:
                    modelParams = (20, 2)
                case 3:
                    modelParams = (29, 2)
        if multiplier == 4:
            match paramsIndex:
                case 1:
                    modelParams = (5, 4)
                case 2:
                    modelParams = (14, 4)
                case 3:
                    modelParams = (20, 4)

    return modelParams

def smoothess(arr):
    window_size = 49
    padding = window_size // 2
    i = 0
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]
        # Calculate the average of current window
        window_average = sum(window) / window_size
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        i += 1


    diff = [a-b for (a, b) in zip(arr[padding:-padding], moving_averages)]
    mse = (numpy.square(diff)).mean(axis=0)
    moving_averages_final = numpy.pad(moving_averages, (padding, padding), "constant", constant_values=(0, 0))
    return moving_averages_final, mse

def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def mse(values):
    return ((values - values.mean(axis=0)) ** 2).mean(axis=0)


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

def mapDataloader(speed):
    name = "wave-0-0-1-290-" + speed
    datasetLoader = Wave(name, isTrain=False)
    dataloader = DataLoader(dataset=datasetLoader, batch_size=32, shuffle=False, drop_last=False,
                             collate_fn=lambda x: default_collate(x).to(device, torch.float))
    return dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = "wave"
mode = "speed/gap"
horizon = 170
modelName = "baseline"
multiplier = 1.0
paramLevel = 2
hiddenSize, lateralSize = mapParas(modelName, multiplier, paramLevel)
model = mapModel(modelName, hiddenSize, lateralSize)
params = count_params(model)
run = 2
learningRate = 0.001
start = 0
speedTrain = "44"
speedTest = "44"

dataloader = mapDataloader(speedTest)

#path = f'../trainedModels/{mode}/{modelName}/run{run}'

path = f'../trainedModels/all-40-adapted/baseline/1/2/run{run}'
os.chdir(path)

# model
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

visData = iter(dataloader).__next__()

pred = model(visData[:, 100:120, :, :], horizon=horizon).detach().cpu().numpy()

sequence = 0
# for one pixel
#
w, h = mostSignificantPixel(pred[sequence, :, :, :])
w, h = 15, 15
groundTruth = visData[sequence, 120:290, int(w / 2), int(h / 2)].detach().cpu().numpy()
prediction = pred[sequence, :, int(w / 2), int(h / 2)]
plt.plot(groundTruth, label="groundTruth")
plt.plot(prediction, label="prediction")
plt.legend()
plt.title(f'{(w, h)} - center pixel')
plt.xlabel("time step")
plt.ylabel("amplitude of wave")
#plt.savefig("perPixel")
plt.savefig("baseline-single-44")
plt.show()


