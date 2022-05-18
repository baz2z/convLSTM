import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
from torch.utils.data import Dataset, DataLoader, default_collate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        f = h5py.File("../data/wave/" + file, 'r')
        self.isTrain = isTrain
        self.data = f['data']['train'] if self.isTrain else f['data']['test']

    def __getitem__(self, item):
        return self.data[f'{item}'.zfill(3)][:,:,:]

    def __len__(self):
        return len(self.data)

dataloader = DataLoader(dataset=Wave("wave-5000-60"), batch_size=10, shuffle=False, drop_last=False,
                        collate_fn=lambda x: default_collate(x).to(device, torch.float))

visData = iter(dataloader).__next__()