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
        self.f = h5py.File("../data/wave/" + file, 'r')
        self.isTrain = isTrain
        self.data = self.f['data']['train'] if self.isTrain else self.f['data']['val']


    def __getitem__(self, item):
        data = self.data[f'{item}'.zfill(3)][:, :, :]
        return data

    def __len__(self):
        return len(self.data)


def load_attributes(f, name='params'):
    '''
    Function to load attributes of a param group in an hdf5 file.
    File structure is assumed to be:
    params/param_types
    The function will return a list of dictionaries containing the attributes of param_types
    '''
    param_list = []
    for group in f[name].keys():
        att = f[f'{name}/{group}'].attrs
        params = {}
        for key in att.keys():
            params[key] = att[key]
        param_list.append(params)
    return param_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for j in range(21):
#     if j != 13:
#         datasetName = "wave-3000-60_" + str(j - 10)
#         dataset = Wave(datasetName, isTrain=True)
#         dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=lambda x: default_collate(x).to("cpu", torch.float), drop_last=True)
#         # print(dataloader.dataset.mu, dataloader.dataset.std)
#         print(datasetName, load_attributes(dataset.f))
#         # print(numpy.mean(iter(dataloader).__next__().numpy()))
#         # print(numpy.std(iter(dataloader).__next__().numpy()))
#

speed = 16
datasetName = "wave-5000-190-[16-28]"
datasetTrain = Wave(datasetName)
datasetVal = Wave(datasetName, isTrain=False)
dataloaderTrain = DataLoader(dataset=datasetTrain, batch_size=32, shuffle=True, drop_last=True,
                             collate_fn=lambda x: default_collate(x).to(device, torch.float))
dataloaderVal = DataLoader(dataset=datasetVal, batch_size=32, shuffle=True, drop_last=True,
                           collate_fn=lambda x: default_collate(x).to(device, torch.float))
print(datasetName, load_attributes(datasetTrain.f))


