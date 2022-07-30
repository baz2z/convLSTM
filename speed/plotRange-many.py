import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f(x):
    return {
        0.5: 1,
        1: 2,
        2: 4
    }[x]

def matchColor(modelName):
    return{
        "baseline": "blue",
        "lateral": "red",
        "twoLayer":"green",
        "skip": "purple",
        "depthWise": "chocolate"
    }[modelName]
    pass
def matchMarker(multiplier):
    return{
        0.5:"^",
        1:"s",
        2:"+",
        4:"o"
    }[multiplier]

fig, (ax1, ax2) = plt.subplots(2, 1)

df = pd.read_csv("./df/speed-range-adapted-multipleTest-170")
df.reset_index()
for index, row in df.iterrows():
    modelName = row["name"]
    range = row["speedTrained"]
    testSpeed = row["speedTest"]
    testSpeed = float(str(testSpeed)[:1] + "." + str(testSpeed)[1:])
    loss70 = row["loss100_170"]
    col = matchColor(modelName)
    if range == "lower":
        ax1.scatter(testSpeed, loss70, color=col, s=16, alpha=0.7)
        ax1.plot(testSpeed, loss70, color=col, alpha=0.7)

    else:
        ax2.scatter(testSpeed, loss70, color=col, s=16, alpha=0.7)
        ax2.plot(testSpeed, loss70, color=col, alpha=0.7)


ax1.set_xticks(np.arange(1.2, 3.4, 0.2))
ax1.set_yscale('log')
ax2.set_xticks(np.arange(2.8, 5.0, 0.2))
ax2.set_yscale('log')

blue_line = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=12, label='baseline', linestyle="none")
red_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=12, label='lateral', linestyle="none")
green_line = mlines.Line2D([], [], color='green', marker='o',
                          markersize=12, label='twoLayer', linestyle="none")
purple_line = mlines.Line2D([], [], color='purple', marker='o',
                          markersize=12, label='skip', linestyle="none")
chocolate_line = mlines.Line2D([], [], color='chocolate', marker='o',
                          markersize=12, label='depthWise', linestyle="none")


ax1.legend(handles=[blue_line, red_line, green_line, purple_line, chocolate_line], bbox_to_anchor=(1.05, 1), loc = 2)
#plt.ylim([0.0001, 0.001])
# ax1.title.set_text("model trained on speed 1.6")
# ax2.title.set_text("model trained on speed 4.4")
ax2.set_xlabel("wave speed tested on")
ax2.set_ylabel("loss")
name = f'./createdPlots/speed-range-adapted-170'
fig.savefig(name, bbox_inches="tight")






