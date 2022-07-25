import argparse
import torch
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader, default_collate, ConcatDataset
import torch.optim as optim
from models import baseline, lateral, twoLayer, depthWise, skipConnection, Forecaster
import h5py
import matplotlib.pyplot as plt
import math
import os
import numpy
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


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


def mapDataset(datasetTrain, datasetVal, batch_size):
    train = None
    val = None

    match datasetTrain:
        case "wave-5000-90-1.0":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))
        case "wave-5000-90-1.2":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))
        case "wave-5000-90-1.4":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))
        case "wave-5000-90-1.6":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))
        case "wave-5000-90-4.4":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))
        case "wave-5000-90-4.6":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))
        case "wave-5000-90-4.8":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))
        case "wave-5000-90-4.0":
            train = DataLoader(dataset=Wave("wave-10000-90"), batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))



    match datasetVal:
        case "wave-5000-90":
            val = DataLoader(dataset=Wave("wave-10000-90", isTrain=False), batch_size=batch_size, shuffle=True,
                             drop_last=True,
                             collate_fn=lambda x: default_collate(x).to(device, torch.float))

    return train, val

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

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="baseline")
    parser.add_argument('--dataset', type=str, default="wave")
    parser.add_argument('--mode', type=str, default="speed-basic-adapted")
    parser.add_argument('--context', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--learningRate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--run_idx', type=int, default=1)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--multiplier', type=float, default=1)
    parser.add_argument('--paramLevel', type=int, default=1)
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    mode = args.mode
    context = args.context
    horizon = args.horizon
    learningRate = args.learningRate
    epochs = args.epochs
    run = args.run_idx
    batch_size = args.batchSize
    clip = args.clip
    mp = args.multiplier
    paramLevel = args.paramLevel
    # begin to train
    loss_plot_train, loss_plot_val = [], []
    speedRange = "RANGE"
    hiddenSize, lateralSize = mapParas(model, mp, paramLevel)
    seq = mapModel(model, hiddenSize, lateralSize)
    params = count_params(seq)
    datasetNameLower = "wave-10000-190-lower"
    datasetNameUpper = "wave-10000-190-upper"
    datasetTrainLower = Wave(datasetNameLower)
    datasetTrainUpper = Wave(datasetNameUpper)

    datasetTrainConcat = ConcatDataset([datasetTrainLower, datasetTrainUpper])

    datasetTrainLowerVal = Wave(datasetNameLower, isTrain=False)
    datasetTrainUpperVal = Wave(datasetNameUpper, isTrain=False)
    datasetTrainConcatVal = ConcatDataset([datasetTrainLower, datasetTrainUpper])
    dataloaderTrain = DataLoader(dataset=datasetTrainConcat, batch_size=batch_size, shuffle=True, drop_last=True,
                                 collate_fn=lambda x: default_collate(x).to(device, torch.float))
    dataloaderVal = DataLoader(dataset=datasetTrainConcatVal, batch_size=batch_size, shuffle=True, drop_last=True,
                               collate_fn=lambda x: default_collate(x).to(device, torch.float))

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(seq.parameters(), lr=learningRate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    for j in range(epochs):
        lossPerBatch = []
        for i, images in enumerate(dataloaderTrain):
            input_images = images[:, 100:100+context, :, :]
            labels = images[:, 100+context:100+context + horizon, :, :]
            output = seq(input_images, horizon)
            loss = criterion(output, labels)
            lossPerBatch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq.parameters(), clip)
            optimizer.step()
        scheduler.step()

        loss_plot_train.append(numpy.mean(lossPerBatch))

        with torch.no_grad():
            lossPerBatch = []
            for i, images in enumerate(dataloaderVal):
                input_images = images[:, 100:100 + context, :, :]
                labels = images[:, 100 + context:100 + context + horizon, :, :]
                output = seq(input_images, horizon)
                loss = criterion(output, labels)
                lossPerBatch.append(loss.item())
            loss_plot_val.append(numpy.mean(lossPerBatch))

    # # save model and test and train loss and parameters in txt file and python file with class
    path = f'../trainedModels/{mode}/{model}/{speedRange}/run{run}'
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    torch.save(seq.state_dict(), "model.pt")
    torch.save(loss_plot_train, "trainingLoss")
    torch.save(loss_plot_val, "validationLoss")

    # save config

    configuration = {"model": model,
                     "epochs": epochs,
                     "batchSize": batch_size,
                     "learningRate": learningRate,
                     "parameters": params,
                     "context": context,
                     "horizon": horizon,
                     "Loss": criterion,
                     "dataset": datasetTrainConcat,
                     "clip": clip,
                     "scheduler": scheduler,
                     "hiddenSize": hiddenSize,
                     "lateralSize": lateralSize,
                     "lowerStd": lowerStd,
                     "lowerMean": lowerMean,
                     "upperStd": upperStd,
                     "upperMean": upperMean
                     }
    with open('configuration.txt', 'w') as f:
        print(configuration, file=f)
    os.chdir("../../../../../../speed-basic")
