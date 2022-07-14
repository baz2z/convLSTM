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


context = 20
horizon = 40

fig, ax = plt.subplots()

df = pd.read_csv("../test/learningRateLoss")
df.reset_index()
for index, row in df.iterrows():
    modelName = row["modelName"]
    lr = row["learningRate"]
    loss = row["loss"]
    col = matchColor(modelName)
    ax.scatter(lr, loss, color=col, s=16, alpha=0.7)

ax.set_xscale('log')
ax.set_yscale('log')
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
plt.xlabel("learning Rate")
plt.ylabel("loss")
#plt.ylim([0.0001, 0.001])
name = f'./createdPlots/lrLoss'
fig.savefig(name, bbox_inches="tight")






