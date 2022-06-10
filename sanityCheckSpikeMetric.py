import torch
import os
import h5py
import matplotlib.pyplot as plt
import math
import numpy
import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate
import matplotlib.axes as ax

def smoothess(arr):
    window_size = 24
    padding = window_size // 2
    i = 0
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]
        # Calculate the average of current window
        window_average = sum(window) / window_size
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        i += 1


    diff = [a-b for (a, b) in zip(arr[padding:-padding], moving_averages)]
    mse = (numpy.square(diff)).mean(axis=0)
    moving_averages_final = numpy.pad(moving_averages, (padding, padding), "constant", constant_values=(0, 0))
    return moving_averages_final, mse


a = np.arange(1, 10, 0.05).repeat(1)
b = numpy.random.random([180])
b2 = numpy.random.random([180])
d = np.exp(b)/200
d2 = np.exp(b2)/100
c = np.exp(-a) + 0.1
#print(c + d)
def createDataset(x, scatter):
    for i in range(len(x)):
        a = -scatter[i] if np.random.random(1) > 0.5 else scatter[i]
        x[i] += a
    return x

#print(createDataset(c, d))
plt.plot(createDataset(c, d), label=f'scatter:{round(smoothess(createDataset(c, d))[1], 5)}')
plt.plot(createDataset(c, d2), label=f'scatter:{round(smoothess(createDataset(c, d2))[1], 5)}')
plt.yscale("log")
plt.legend()
plt.show()

k = createDataset(c, d2)
plt.plot(k, label=f'scatter:{round(smoothess(k)[1], 5)}')
plt.plot(smoothess(k)[0], label="movingAverage")
plt.legend()
plt.show()