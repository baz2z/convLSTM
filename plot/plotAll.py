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


fig, ax = plt.subplots()

df = pd.read_csv("../test/allLoss")
df.reset_index()
for index, row in df.iterrows():
    modelName = row["modelName"]
    mp = row["mp"]
    paras = row["parasExact"]
    loss40 = row["loss40"]
    loss70 = row["loss70"]
    loss170 = row["loss170"]
    col = matchColor(modelName)
    marker = matchMarker(mp)
    ax.scatter(paras, loss40, color=col, s=16, alpha=0.7, marker=marker)
    # ax.scatter(paras, loss70, color=col, s=16, alpha=0.7, marker=matchMarker("70"))
    # ax.scatter(paras, loss170, color=col, s=16, alpha=0.7, marker=matchMarker("170"))


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

# multiplier
marker1 = mlines.Line2D([], [], color='gray', marker='^',
                          markersize=12, label='2:1 (hs:ls)', linestyle="none")
marker2 = mlines.Line2D([], [], color='gray', marker='s',
                          markersize=12, label='1:1', linestyle="none")
marker3 = mlines.Line2D([], [], color='gray', marker='+',
                          markersize=12, label='1:2', linestyle="none")
marker4 = mlines.Line2D([], [], color='gray', marker='o',
                          markersize=12, label='1:4 (multiplication)', linestyle="none")


plt.legend(handles=[blue_line, red_line, green_line, purple_line, chocolate_line, marker1, marker2, marker3, marker4], bbox_to_anchor=(1.05, 1), loc = 2)
#plt.ylim([0.0001, 0.001])
print()
name = f'./createdPlots/loss40'
plt.xlabel("parameters of models")
plt.ylabel("loss")
fig.savefig(name, bbox_inches="tight")






