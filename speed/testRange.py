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
import itertools
from matplotlib import pyplot
import matplotlib.lines as mlines

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
                    modelParams = (40, 1)
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

    return modelParams #(hs, ls)
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
        # means, stds = [], []
        # for i in range(len(self.data)):
        #     data = self.data[f'{i}'.zfill(3)][:, :, :]
        #     means.append(numpy.mean(data))
        #     stds.append(numpy.std(data))
        # self.mu = numpy.mean(means)
        # self.std = numpy.mean(stds)


    def __getitem__(self, item):
        data = self.data[f'{item}'.zfill(3)][:, :, :]
        #data = (data - self.mu) / self.std
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


def totaSmoothness():
    modelsSmoothness = []
    for runNbr in range(4):
        runNbr = runNbr + 1
        os.chdir(f'./run{runNbr}')
        trainLoss = torch.load("trainingLoss", map_location=device)
        #valLoss = torch.load("validationLoss", map_location=device)
        movingAvg, smoothness = smoothess(trainLoss)
        modelsSmoothness.append(smoothness)
        os.chdir("../")

    return numpy.mean(modelsSmoothness)





def calcLoss(model, start, context, horizon, dataloader, og = False):
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
                input_images = images[:, start:start+context, :, :]
                labels = images[:,start+context:start+context + horizon, :, :]
                output = model(input_images, horizon)
                loss = criterion(output, labels)
                runningLoss.append(loss.cpu())
            modelsLoss.append(numpy.mean(runningLoss))
        os.chdir("../")
    finalLoss = numpy.mean(modelsLoss)
    return finalLoss


def mapDataloader(speed):
    name = "wave-0-0-1-290-" + speed
    datasetLoader = Wave(name, isTrain=False)
    dataloader = DataLoader(dataset=datasetLoader, batch_size=32, shuffle=False, drop_last=True,
                             collate_fn=lambda x: default_collate(x).to(device, torch.float))
    return dataloader



if __name__ == '__main__':
    mode = "speed/range"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    datasetNameLower = "wave-5000-190-[16-28]"
    datasetNameUpper = "wave-5000-190-[32-44]"
    datasetValLower = Wave(datasetNameLower, isTrain=False)
    datasetValUpper = Wave(datasetNameUpper, isTrain=False)
    dataloaderTestLower = DataLoader(dataset=datasetValLower, batch_size=batch_size, shuffle=True, drop_last=True,
                                    collate_fn=lambda x: default_collate(x).to(device, torch.float))
    dataloaderTestUpper = DataLoader(dataset=datasetValUpper, batch_size=batch_size, shuffle=True, drop_last=True,
                                    collate_fn=lambda x: default_collate(x).to(device, torch.float))

    df = pd.DataFrame(columns=["name", "speed", "loss100_70"])# , "loss40_og", "loss70_og", "loss170_og"])
    param = 2
    mult = 1
    counter = 0

    for modelName in ["baseline", "lateral", "twoLayer", "skip", "depthWise"]:
        for rangeTained in ["lower","upper"]:
            if rangeTained == "lower":
                dataLoader = dataloaderTestLower
                hs, ls = mapParas(modelName, mult, param)
                model = mapModel(modelName, hs, ls)
                path = f'../trainedModels/{mode}/{modelName}/{rangeTained}'
                os.chdir(path)
                loss70 = calcLoss(model, 100, 20, 70, dataLoader)
                df.loc[counter] = [modelName, rangeTained, loss170]  # , loss40_og, loss70_og, loss170_og]
                counter += 1
                pathBack = f'../../../../../speed'
                os.chdir(pathBack)
            elif rangeTained == "upper":
                dataLoader = dataloaderTestUpper
                hs, ls = mapParas(modelName, mult, param)
                model = mapModel(modelName, hs, ls)
                path = f'../trainedModels/{mode}/{modelName}/{rangeTained}'
                os.chdir(path)
                loss70 = calcLoss(model, 100, 20, 70, dataLoader)
                df.loc[counter] = [modelName, rangeTained, loss70]# , loss40_og, loss70_og, loss170_og]
                counter += 1
                pathBack = f'../../../../../speed'
                os.chdir(pathBack)


    df.to_csv("./df/speed-range-adapted-singleTest")


"""
if __name__ == '__main__':
    mode = "speed/range"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    df = pd.DataFrame(columns=["name", "speedTrained", "speedTest", "loss100_170"])# , "loss40_og", "loss70_og", "loss170_og"])
    param = 2
    mult = 1
    counter = 0

    for modelName in ["baseline", "lateral", "twoLayer", "skip", "depthWise"]:
        for rangeTrained in ["lower","upper"]:
            if rangeTrained == "lower":
                for speedTest in ["12", "14", "16", "18", "20"]:
                    dataLoader = mapDataloader(speedTest)
                    hs, ls = mapParas(modelName, mult, param)
                    model = mapModel(modelName, hs, ls)
                    path = f'../trainedModels/{mode}/{modelName}/{rangeTrained}'
                    os.chdir(path)
                    loss170 = calcLoss(model, 100, 20, 170, dataLoader)
                    df.loc[counter] = [modelName, rangeTrained, speedTest, loss170]  # , loss40_og, loss70_og, loss170_og]
                    counter += 1
                    pathBack = f'../../../../../speed'
                    os.chdir(pathBack)
            elif rangeTrained == "upper":
                for speedTest in ["40", "42", "44", "46", "48"]:
                    dataLoader = mapDataloader(speedTest)
                    hs, ls = mapParas(modelName, mult, param)
                    model = mapModel(modelName, hs, ls)
                    path = f'../trainedModels/{mode}/{modelName}/{rangeTrained}'
                    os.chdir(path)
                    loss170 = calcLoss(model, 100, 20, 170, dataLoader)
                    df.loc[counter] = [modelName, rangeTrained, speedTest, loss170]# , loss40_og, loss70_og, loss170_og]
                    counter += 1
                    pathBack = f'../../../../../speed'
                    os.chdir(pathBack)


    df.to_csv("./df/speed-range-adapted-multipleTest")


"""