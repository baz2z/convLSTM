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
        f = h5py.File("../../../../../../work2/butz1/svolz67/data/wave/"+ file, 'r')
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
    bestRun = [-10, 1, 1]
    worstRun = [10, 1, 1]
    for runNbr in [3]:
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

                if loss > bestRun[0]:
                    bestRun[0] = loss
                    bestRun[1] = output
                    bestRun[2] = labels
                if loss < worstRun[0]:
                    worstRun[0] = loss
                    worstRun[1] = output
                    worstRun[2] = labels
                runningLoss.append(loss.cpu())
            modelsLoss.append(numpy.mean(runningLoss))
        os.chdir("../")
    finalLoss = numpy.mean(modelsLoss)
    return finalLoss, worstRun, bestRun


def mapDataloader(speed):
    name = "wave-0-0-1-290-" + speed
    datasetLoader = Wave(name, isTrain=False)
    dataloader = DataLoader(dataset=datasetLoader, batch_size=1, shuffle=False, drop_last=True,
                             collate_fn=lambda x: default_collate(x).to(device, torch.float))
    return dataloader


if __name__ == '__main__':
    mode = "speed/gap"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    df = pd.DataFrame(columns=["name", "speedTest", "loss100_170"])# , "loss40_og", "loss70_og", "loss170_og"])
    param = 2
    counter = 0

    for modelName in ["baseline"]:#, "lateral", "twoLayer", "skip05", "skip2"]:
        mps = 1
        for speedTest in ["16"]:
            dataLoader = mapDataloader(speedTest)
            hs, ls = mapParas(modelName, mps, param)
            model = mapModel(modelName, hs, ls)
            path = f'../trainedModels/{mode}/{modelName}'
            os.chdir(path)
            loss170, bestSlow, worseSlow = calcLoss(model, 100, 20, 170, dataLoader)
            df.loc[counter] = [modelName, speedTest, loss170]  # , loss40_og, loss70_og, loss170_og]
            counter += 1
            pathBack = f'../../../../speed'
            os.chdir(pathBack)
        for speedTest in ["30"]:
            dataLoader = mapDataloader(speedTest)
            hs, ls = mapParas(modelName, mps, param)
            model = mapModel(modelName, hs, ls)
            path = f'../trainedModels/{mode}/{modelName}'
            os.chdir(path)
            loss170, bestMedium, worseMedium = calcLoss(model, 100, 20, 170, dataLoader)
            df.loc[counter] = [modelName, speedTest, loss170]  # , loss40_og, loss70_og, loss170_og]
            counter += 1
            pathBack = f'../../../../speed'
            os.chdir(pathBack)
        for speedTest in ["44"]:
            dataLoader = mapDataloader(speedTest)
            hs, ls = mapParas(modelName, mps, param)
            model = mapModel(modelName, hs, ls)
            path = f'../trainedModels/{mode}/{modelName}'
            os.chdir(path)
            loss170, bestFast, worseFast = calcLoss(model, 100, 20, 170, dataLoader)
            df.loc[counter] = [modelName, speedTest, loss170]  # , loss40_og, loss70_og, loss170_og]
            counter += 1
            pathBack = f'../../../../speed'
            os.chdir(pathBack)#

    #df.to_csv(f"./df/speed-gap-{modelName}-plot")

    fig, axs = plt.subplots(2, 3)

    w, h = 15, 15

    groundTruth = worseSlow[2][0, :, w, h].detach().cpu().numpy()
    prediction = worseSlow[1][0, :, w, h].detach().cpu().numpy()
    axs[0, 0].plot(groundTruth, label="groundTruth")
    axs[0, 0].plot(prediction, label="prediction")
    loss = worseSlow[0]
    axs[0, 0].set_title(f'1.6 - {"%.4f" % loss}')
    axs[0, 0].set_xlabel('time step')
    axs[0, 0].set_ylabel('amplitude')


    groundTruth = bestSlow[2][0, :, w, h].detach().cpu().numpy()
    prediction = bestSlow[1][0, :, w, h].detach().cpu().numpy()
    axs[1, 0].plot(groundTruth, label="groundTruth")
    axs[1, 0].plot(prediction, label="prediction")
    loss = bestSlow[0]
    axs[1, 0].legend()
    axs[1, 0].set_title(f'1.6 - {"%.4f" % loss}')





    groundTruth = worseMedium[2][0, :, w, h].detach().cpu().numpy()
    prediction = worseMedium[1][0, :, w, h].detach().cpu().numpy()
    loss = worseMedium[0]
    axs[0, 1].plot(groundTruth)
    axs[0, 1].plot(prediction)
    axs[0, 1].set_title(f'3.0 - {"%.4f" % loss}')

    groundTruth = bestMedium[2][0, :, w, h].detach().cpu().numpy()
    prediction = bestMedium[1][0, :, w, h].detach().cpu().numpy()
    loss = bestMedium[0]
    axs[1, 1].plot(groundTruth)
    axs[1, 1].plot(prediction)
    axs[1, 1].set_title(f'3.0 - {"%.4f" % loss}')







    groundTruth = worseFast[2][0, :, w, h].detach().cpu().numpy()
    prediction = worseFast[1][0, :, w, h].detach().cpu().numpy()
    loss = worseFast[0]
    axs[0, 2].plot(groundTruth)
    axs[0, 2].plot(prediction)
    axs[0, 2].set_title(f'4.4 - {"%.4f" % loss}')
    axs[0, 2].set(xlabel='time step', ylabel='amplitude of wave')

    groundTruth = bestFast[2][0, :, w, h].detach().cpu().numpy()
    prediction = bestFast[1][0, :, w, h].detach().cpu().numpy()
    loss = bestFast[0]
    axs[1, 2].plot(groundTruth)
    axs[1, 2].plot(prediction)
    axs[1, 2].set_title(f'4.4 - {"%.4f" % loss}')


    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for i, ax in enumerate(axs.flat):
        if i != 0:
            ax.label_outer()

    fig.tight_layout()
    fig.savefig("gap-worse-best")