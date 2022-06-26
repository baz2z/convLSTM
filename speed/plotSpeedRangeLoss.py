import matplotlib.pyplot as plt
import numpy as np

waveSpeed = np.arange(1, 5.2, step = 0.2)
lossBaseline = [1.6718009, 1.6349819, 1.4854617, 1.3920763, 1.2698158, 1.1290376, 0.95480716, 0.7032467, 0.40395522, 0.11950036, 0.0103748925, 0.17785898, 0.5127502, 0.8674181, 1.1620183, 1.3986183, 1.6101824, 1.7723801, 1.9330692, 2.072893, 2.1872299]
# lossLateral1:1 = 1
# lossTwoLayer2:1 = 1
# lossSkip2:1 = 1
# lossDW1:2 = 1
plt.scatter(waveSpeed, lossBaseline, label="Baseline", c="blue")
#plt.scatter(waveSpeed, lossBaseline, label="Baseline", c="red", s=2)
plt.xlabel("waveSpeed")
plt.ylim("loss")
plt.legend()
plt.yscale("log")
plt.show()