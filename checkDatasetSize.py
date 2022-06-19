import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
from torch.utils.data import Dataset, DataLoader, default_collate




class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        f = h5py.File("../data/wave/" + file, 'r')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Wave("wave-10000-90", isTrain=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=lambda x: default_collate(x).to("cpu", torch.float), drop_last=True)
print(dataset.__getitem__(9)[0, 0,0])
print(dataset.__getitem__(99)[0, 0,0])
print(dataset.__getitem__(999)[0, 0,0])
print(dataset.__getitem__(9999)[0, 0,0])
print(len(dataloader.dataset))
