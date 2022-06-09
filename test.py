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



dataloader = DataLoader(dataset=Wave("wave-10000-90"), batch_size=10, shuffle=False, drop_last=False,
                        collate_fn=lambda x: default_collate(x).to("cpu", torch.float))

data = enumerate(dataloader).__next__()
i, data = data


for i, data in enumerate(dataloader):
    print(torch.mean(data))
    print(torch.std(data))