from models import baseline, lateral, skipConnection, depthWise, twoLayer, Forecaster
import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
from torch.utils.data import Dataset, DataLoader, default_collate
from torch import nn
import itertools
from matplotlib import pyplot
import matplotlib.lines as mlines


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def f(x):
    return {
        0.5: 1,
        1: 2,
        2: 4
    }[x]

def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)



dataset = "wave"
mode = "horizon-20-40"
datasetLoader = Wave("wave-10000-90")
dataloader = DataLoader(dataset=datasetLoader, batch_size=25, shuffle=False, drop_last=True,
                        collate_fn=lambda x: default_collate(x).to(device, torch.float))
context = 20
horizon = 40


def calcLoss(model):
    criterion = nn.MSELoss()
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
                loss = criterion(output, labels)
                runningLoss.append(loss.cpu())
            modelsLoss.append(numpy.mean(runningLoss))
        os.chdir("../")
        finalLoss = numpy.mean(modelsLoss)
        return finalLoss

def matchColor(modelName):
    return{
        "baseline": "blue",
        "lateral": "red",
        "twoLayer":"green",
        "skip": "purple",
        "depthWise": "chocolate"
    }[modelName]
    pass
def matchMarker(multiplier):
    return{
        0.5:"^",
        1:"s",
        2:"+",
        4:"o"
    }[multiplier]




fig, ax = plt.subplots()
for mult in [0.5, 1, 2]:
    for modelName in ["baseline", "lateral", "twoLayer", "skip", "depthWise"]:
        for param in [1, 2, 3]:
            mult_tmp = f(mult)
            if modelName == "baseline":
                mult = 1
                hs, ls = mapParas(modelName, mult, param)
                model = mapModel(modelName, hs, ls)
            elif modelName == "depthWise":
                hs, ls = mapParas(modelName, mult_tmp, param)
                model = mapModel(modelName, hs, ls)
            else:
                modelParas = mapParas(modelName, mult, param)
                hs, ls = mapParas(modelName, mult, param)
                model = mapModel(modelName, hs, ls)

            if modelName == "depthWise":
                path = f'../trainedModels/{dataset}/{mode}/{modelName}/{mult_tmp}/{param}'
            else:
                path = f'../trainedModels/{dataset}/{mode}/{modelName}/{mult}/{param}'


            os.chdir(path)
            # argument: horizon
            parameters = count_params(model)
            loss = calcLoss(model)
            if modelName == "depthWise":
                marker = matchMarker(mult_tmp)
            else:
                marker = matchMarker(mult)
            col = matchColor(modelName)
            ax.scatter(parameters, loss, marker=marker, color=col)
            pathBack = f'../../../../../../plots'
            os.chdir(pathBack)


# add legend
# modelName
blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=12, label='baseline', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=12, label='lateral', linestyle="none")
green_line = mlines.Line2D([], [], color='green', marker='o',
                          markersize=12, label='twoLayer', linestyle="none")
purple_line = mlines.Line2D([], [], color='purple', marker='o',
                          markersize=12, label='skip', linestyle="none")
chocolate_line = mlines.Line2D([], [], color='chocolate', marker='o',
                          markersize=12, label='depthWise', linestyle="none")

# multiplier
mult1 = mlines.Line2D([], [], color='gray', marker='^',
                          markersize=12, label='0.5:1', linestyle="none")
mult2 = mlines.Line2D([], [], color='gray', marker='s',
                          markersize=12, label='1:1', linestyle="none")
mult3 = mlines.Line2D([], [], color='gray', marker='+',
                          markersize=12, label='1:2', linestyle="none")
mult4 = mlines.Line2D([], [], color='gray', marker='o',
                          markersize=12, label='1:4 (multiplication)', linestyle="none")

plt.legend(handles=[blue_line, red_line, green_line, purple_line
                    , chocolate_line, mult1, mult2, mult3, mult4], bbox_to_anchor=(1.05, 1), loc = 2)

name = f'lossToParas-{mode}-{horizon}'
fig.savefig(name, bbox_inches="tight")








