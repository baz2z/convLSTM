from runVersionsEncoderForecasterWithTraningLoop import Forecaster
from models import baseline, lateral, skipConnection, depthWise, twoLayer
import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
from torch.utils.data import Dataset, DataLoader, default_collate

def visualize_wave(imgs):
    t, w, h = imgs.shape
    for i in range(t):
        plt.subplot(math.ceil(t ** 0.5), math.ceil(t ** 0.5), i + 1)
        plt.title(i, fontsize=9)
        plt.axis("off")
        image = imgs[i,:,:]
        plt.imshow(image, cmap="gray")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


class Wave(Dataset):
    def __init__(self, file, isTrain=True):
        # data loading
        f = h5py.File("../../data/wave/" + file, 'r')
        self.isTrain = isTrain
        self.data = f['data']['train'] if self.isTrain else f['data']['test']

    def __getitem__(self, item):
        return self.data[f'{item}'.zfill(3)][:,:,:]

    def __len__(self):
        return len(self.data)

def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = "horizon-20-70"
model, modelName = Forecaster(12, depthWise, num_blocks=2, lstm_kwargs={'lateral_channels_multipl': 8}).to(device), "depthWise"
run = "3"
horizon = 40

dataloader = DataLoader(dataset=Wave("wave-5000-60"), batch_size=10, shuffle=False, drop_last=False,
                        collate_fn=lambda x: default_collate(x).to(device, torch.float))
f = h5py.File("../../data/wave/testWave", 'r')

os.chdir("../trainedModels/wave/" + mode + "/" + modelName + "/" + "run" + run)

# model

model.load_state_dict(torch.load("baseline.pt", map_location=device))
model.eval()
print(count_params(model))

# loss
trainLoss = torch.load("trainingLoss", map_location=device)
valLoss = torch.load("validationLoss", map_location=device)
print(trainLoss[-1])

plt.yscale("log")
plt.plot(trainLoss, label="trainLoss")
plt.plot(valLoss, label="valLoss")
plt.legend()
plt.show()

# example wave
visData = iter(dataloader).__next__()
pred = model(visData[:, :20, :, :], horizon=40).detach().cpu().numpy()


sequence = 7
# for one pixel
w, h = pred.shape[2], pred.shape[3]
groundTruth = visData[sequence, 20:, int(w / 2), int(h / 2)]
prediction = pred[sequence, :, int(w / 2), int(h / 2)]
plt.plot(groundTruth, label="groundTruth")
plt.plot(prediction, label="prediction")
plt.legend()
plt.show()

# for entire sequence
visualize_wave(pred[sequence, :, :, :])
#visualize_wave(visData[sequence, :20, :, :], modelName)