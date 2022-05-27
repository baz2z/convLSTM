from models import baseline, lateral, skipConnection, depthWise, twoLayer, Forecaster
import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
from torch.utils.data import Dataset, DataLoader, default_collate

def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)



def mapModel(model, hiddenSize, lateralSize):
    match model:
        case "baseline":
            return Forecaster(8, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device)
        case "lateral":
            return Forecaster(12, lateral, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        case "twoLayer":
            return Forecaster(12, twoLayer, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        case "skip":
            return Forecaster(12, skipConnection, num_blocks=2, lstm_kwargs={'lateral_channels': 12}).to(device)
        case "depthWise":
            return Forecaster(8, depthWise, num_blocks=2, lstm_kwargs={'lateral_channels_multipl': 6}).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = "wave"
mode = "horizon-20-40"
modelName = "skip"
run = "3"




# loss
lossTotal = []
for i in range(5):
    i = i+1
    os.chdir(f"C:/Users/Sebastian/Desktop/remote/convLSTM/trainedModels/wave/horizon-20-70/{modelName}/run{i}")
    trainLoss = torch.load("trainingLoss", map_location=device)
    valLoss = torch.load("validationLoss", map_location=device)
    averageLastLoss = (sum(valLoss[-5:]) / 5).item()
    lossTotal.append(averageLastLoss)

print((sum(lossTotal[:]) / 5))



# model
model = mapModel(modelName, 8, 6)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
print(count_params(model))
