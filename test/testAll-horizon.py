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
        f = h5py.File("../../../../../../work2/butz1/svolz67/data/wave/"+ file, 'r')
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
                labels = images[:, start+context:start+context + horizon, :, :]
                output = model(input_images, horizon)
                if og:
                    output_not_normalized = (output * dataloader.dataset.std) + dataloader.dataset.mu
                    labels_not_normalized = (labels * dataloader.dataset.std) + dataloader.dataset.mu
                    loss = criterion(output_not_normalized, labels_not_normalized)
                else:
                    loss = criterion(output, labels)
                runningLoss.append(loss.cpu())
            modelsLoss.append(numpy.mean(runningLoss))
        os.chdir("../")
    finalLoss = numpy.mean(modelsLoss)
    return finalLoss

def calcHorizonLoss(model, start, context, horizon, dataloader, og = False):
    criterion = nn.MSELoss()
    images = iter(dataloader).__next__()
    lossHorizonAllRuns = numpy.zeros(horizon)
    for runNbr in range(3):
        runNbr = runNbr + 1
        os.chdir(f'./run{runNbr}')
        model.load_state_dict(torch.load("model.pt", map_location=device))
        model.eval()
        lossHorizon = []
        with torch.no_grad():
            input_images = images[:, start:start+context, :, :]
            for future in range(horizon):
                future += 1
                labels = images[:, start+context:start+context + future, :, :]
                output = model(input_images, future)
                # loss = numpy.sum((output - labels).detach().numpy())
                loss = criterion(output, labels).cpu().numpy()
                lossHorizon.append(loss)
        lossHorizonAllRuns = numpy.add(lossHorizonAllRuns, lossHorizon)
        os.chdir("../")
    return lossHorizonAllRuns/5
# calculated train loss on new dataset and average the loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="baseline",
                    choices=["baseline", "lateral", "twoLayer", "skip", "depthWise"])

args = parser.parse_args()
modelName = args.model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = "all-40-adapted"
context = 20


#params = count_params(model)
criterion = nn.MSELoss()


datasetLoader = Wave("wave-0-0-3-390", isTrain=False)
dataloader = DataLoader(dataset=datasetLoader, batch_size=32, shuffle=False, drop_last=True,
                        collate_fn=lambda x: default_collate(x).to(device, torch.float))





df = pd.DataFrame(columns=["modelName", "mp", "paramLevel", "parasExact", "lossHorizon"])
counter = 0

for modelName in ["baseline", "lateral", "twoLayer", "skip", "depthWise"]:
    if modelName == "baseline":
        mps = [1]
    elif modelName == "depthWise":
        mps = [1, 2, 4]
    else:
        mps = [0.5, 1, 2]
    for mp in mps:
        for paramLevel in [1, 2, 3]:
            hiddenSize, lateralSize = mapParas(modelName, mp, paramLevel)
            model = mapModel(modelName, hiddenSize, lateralSize)
            parasExact = count_params(model)
            path = f'../trainedModels/{mode}/{modelName}/{mp}/{paramLevel}'
            os.chdir(path)
            print(modelName, mp, paramLevel)
            lossHorizon = calcHorizonLoss(model, 100, 20, 270, dataloader)
            df.loc[counter] = [modelName, mp, paramLevel, parasExact, lossHorizon]
            counter += 1
            os.chdir("../../../../../test")

df.to_csv("allLoss-horizon")

"""
run for each model
"""








