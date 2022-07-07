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
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def matchMarker(modelName):
    return{
        "baseline": "^",
        "lateral": "s",
        "twoLayer":"+",
        "skip": "o",
        "depthWise": "d"
    }[modelName]
    pass
def matchColor(speed):
    return{
        16:"blue",
        44:"red",
    }[speed]



dataset = "wave"
mode = "horizon-20-70"
context = 20
horizon = 40


fig, ax = plt.subplots()

df = pd.read_csv("speed")
df.reset_index()
for index, row in df.iterrows():
    modelName = row["name"]
    basic = row["basic"]
    speed = row["speed-basic"]
    loss = row["loss40"]
    col = matchColor(speed)
    marker = matchMarker(modelName)
    ax.scatter(basic, loss, marker=marker, color=col, s=16, alpha=0.7)

ax.set_yscale('log')
blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=12, label='16', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=12, label='44', linestyle="none")



# multiplier
mult1 = mlines.Line2D([], [], color='gray', marker='^',
                          markersize=12, label='baseline', linestyle="none")
mult2 = mlines.Line2D([], [], color='gray', marker='s',
                          markersize=12, label='lateral', linestyle="none")
mult3 = mlines.Line2D([], [], color='gray', marker='+',
                          markersize=12, label='twoLayer', linestyle="none")
mult4 = mlines.Line2D([], [], color='gray', marker='o',
                          markersize=12, label='skip', linestyle="none")
mult5 = mlines.Line2D([], [], color='gray', marker='d',
                          markersize=12, label='depthWise', linestyle="none")

plt.legend(handles=[blue_line, red_line, mult1, mult2, mult3, mult4, mult5], bbox_to_anchor=(1.05, 1), loc = 2)
#plt.ylim([0.0001, 0.001])

name = f'./createdPlots/speedLoss'
fig.savefig(name, bbox_inches="tight")






