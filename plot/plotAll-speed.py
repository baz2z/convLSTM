import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

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

fig, ax1 = plt.subplots(1, 1)

df = pd.read_csv("../test/allLoss-speed")
df.reset_index()
for index, row in df.iterrows():
    modelName = row["name"]
    testSpeed = row["speedTest"]
    testSpeed = float(str(testSpeed)[:1] + "." + str(testSpeed)[1:])
    loss170 = row["loss100_170"]
    mp = row["mp"]
    col = matchColor(modelName)
    if mp == 1:
        ax1.scatter(testSpeed, loss170, color=col, s=16, alpha=0.7)




ax1.set_yscale('log')


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


plt.legend(handles=[blue_line, red_line, green_line, purple_line, chocolate_line], bbox_to_anchor=(1.05, 1), loc = 2)
#plt.ylim([0.0001, 0.001])
ax1.set_xlabel("speed tested on")
ax1.set_ylabel("loss")
name = f'./createdPlots/all-speed'
fig.savefig(name, bbox_inches="tight")






