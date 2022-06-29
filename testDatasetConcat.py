import argparse
import torch
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader, default_collate, TensorDataset, ConcatDataset
import torch.optim as optim
import h5py
import matplotlib.pyplot as plt
import math
import os
import numpy
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


datasets = []
for i in range(3):
    datasets.append(TensorDataset(torch.arange(i*10, (i+1)*10)))

dataset = ConcatDataset(datasets)
loader = DataLoader(
    dataset,
    shuffle=True,
    num_workers=0,
    batch_size=3
)

# for data in loader:
#     print(data)



class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        self.f = h5py.File("../data/wave/" + file, 'r')
        self.isTrain = isTrain
        self.data = self.f['data']['train'] if self.isTrain else self.f['data']['val']
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


datasetNameLower = "wave-10000-190-10"
datasetNameUpper = "wave-10000-190-12"
datasetLower = Wave(datasetNameLower)
datasetLower.mu = 0
datasetLower.std = 1
datasetUpper = Wave(datasetNameUpper)
datasetUpper.mu = -100
datasetUpper.std = 1
datasetAll = ConcatDataset([datasetLower, datasetUpper])
dataloader = DataLoader(dataset=datasetAll, batch_size=2, shuffle=False, collate_fn=lambda x: default_collate(x).to("cpu", torch.float), drop_last=False)

for data in dataloader:
    print(numpy.mean(data.numpy()))
