from models import baseline, lateral, skipConnection, depthWise, twoLayer, Forecaster
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


class mMnist(Dataset):
    def __init__(self):
        self.data = numpy.load("../../data/movingMNIST/movingmnistdata.npz")["arr_0"].reshape(-1, 20, 64, 64)

    def __getitem__(self, item):
        return self.data[item,:,:,:]

    def __len__(self):
        return self.data.shape[0]

def count_params(net):
    '''
    A utility function that counts the total number of trainable parameters in a network.
    '''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = "horizon-20-40"
model, modelName = Forecaster(12, baseline, num_blocks=2, lstm_kwargs={'k': 3}).to(device), "baseline"
run = "1"
horizon = 40

dataloader = DataLoader(dataset=mMnist(), batch_size=2, shuffle=True, drop_last=True,
                            collate_fn=lambda x: default_collate(x).to(device, torch.float))

os.chdir("../trainedModels/mMnist/" + mode + "/" + modelName + "/" + "run" + run)

# model

model.load_state_dict(torch.load("model.pt", map_location=device))
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
pred = model(visData[:, :10, :, :], horizon=10).detach().cpu().numpy()

sequence = 1
# for entire sequence
visualize_wave(pred[sequence, :, :, :])
visualize_wave(visData[sequence, 10:, :, :])
f = open("configuration.txt", "r")
print(f.read())