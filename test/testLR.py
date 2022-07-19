import argparse

import pandas as pd

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
        #data = (data - self.mu) / self.std
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



def calcLoss(model, context, horizon, dataloader):
    criterion = nn.MSELoss()
    modelsLoss = []
    for runNbr in range(3):
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
                loss = criterion(output, labels)
                runningLoss.append(loss.cpu())
            modelsLoss.append(numpy.mean(runningLoss))
        os.chdir("../")
    finalLoss = numpy.mean(modelsLoss)
    return finalLoss
# calculated train loss on new dataset and average the loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="baseline",
                    choices=["baseline", "lateral", "twoLayer", "skip", "depthWise"])

args = parser.parse_args()
modelName = args.model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = "learningRate"
dataset = "wave"
context = 20
horizon = 40
multiplier = 1.0
paramLevel = 2

#params = count_params(model)
criterion = nn.MSELoss()


datasetLoader = Wave("wave-10-1-3-290", isTrain=False)
dataloader = DataLoader(dataset=datasetLoader, batch_size=32, shuffle=False, drop_last=True,
                        collate_fn=lambda x: default_collate(x).to(device, torch.float))





df = pd.DataFrame(columns=["modelName", "learningRate", "loss"])
counter = 0

for modelName in ["baseline", "lateral", "twoLayer", "skip", "depthWise"]:
    hiddenSize, lateralSize = mapParas(modelName, multiplier, paramLevel)
    model = mapModel(modelName, hiddenSize, lateralSize)
    path = f'../trainedModels/{mode}/{modelName}/{multiplier}/{paramLevel}'
    os.chdir(path)
    for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 1e-05, 5e-05]:
        os.chdir(f'./{lr}')
        loss = calcLoss(model, context, horizon, dataloader)
        df.loc[counter] = [modelName, lr, loss]
        counter += 1
        os.chdir("..")
    os.chdir("../../../../../test")

df.to_csv("learningRateLoss")

